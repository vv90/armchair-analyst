use std::collections::{HashMap, HashSet};

use burn::{
    module::Param,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::{
        Distribution, Slice, activation::softmax, backend::AutodiffBackend, cast::ToElement,
    },
};
use thiserror::Error;

use crate::OptimizationError;
use crate::pool_reserves::{PoolReserves, VirtualReserveValues};
use petgraph::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ColumnIndex(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RowIndex(usize);

pub struct ModelLayout<T: Clone, I: Clone + Copy + PartialEq + Eq + std::hash::Hash> {
    // represents the 2D data matrix being built
    // rows are outputs, columns are inputs
    rows: Vec<Vec<Option<T>>>,
    // maps tokens to the column index
    input_indexes: HashMap<I, ColumnIndex>,
    // maps tokens to a set of row indexes (multiple outputs can return the same token)
    output_indexes: HashMap<I, HashSet<RowIndex>>,
}

// for safe indexing into state rows
// to safeguard against typos
macro_rules! access_reserve {
    ($state:expr, $row_index:expr, $column_index:expr) => {
        $state.rows[{
            let RowIndex(idx) = $row_index;
            idx
        }][{
            let ColumnIndex(idx) = $column_index;
            idx
        }]
    };
}

impl<T: Clone, I: Clone + Copy + PartialEq + Eq + std::hash::Hash> ModelLayout<T, I> {
    fn new() -> Self {
        ModelLayout {
            rows: Vec::new(),
            input_indexes: HashMap::new(),
            output_indexes: HashMap::new(),
        }
    }

    fn with_indexed_input(self, input_token: I) -> (Self, ColumnIndex) {
        match self.input_indexes.get(&input_token).copied() {
            Some(index) => (self, index),
            None => {
                let new_index = ColumnIndex(self.input_indexes.len());
                let mut new_state = self;
                new_state.input_indexes.insert(input_token, new_index);

                new_state.rows.iter_mut().for_each(|row| {
                    if row.len() <= new_index.0 {
                        row.resize(new_index.0 + 1, None);
                    }
                });

                (new_state, new_index)
            }
        }
    }

    fn with_indexed_output(self, column_index: ColumnIndex, output_token: I, reserve: T) -> Self {
        let mut new_state = self;
        let row_indexes = new_state
            .output_indexes
            .entry(output_token)
            .or_insert(HashSet::new());

        let available_row_index = row_indexes
            .iter()
            .find(|&&row_index| access_reserve!(new_state, row_index, column_index).is_none())
            .copied();

        match available_row_index {
            Some(row_index) => {
                access_reserve!(new_state, row_index, column_index) = Some(reserve);
                new_state
            }
            None => {
                let new_row_index = RowIndex(new_state.rows.len());
                new_state
                    .rows
                    .push(vec![None; new_state.input_indexes.len()]);
                row_indexes.insert(new_row_index);
                access_reserve!(new_state, new_row_index, column_index) = Some(reserve);
                new_state
            }
        }
    }

    fn with_reserve_values(self, input_token: I, output_token: I, reserve: T) -> Self {
        // to make sure the each output token is also present in the input indexes
        let (new_state, _) = self.with_indexed_input(output_token);
        let (new_state, column_index) = new_state.with_indexed_input(input_token);

        let new_state = new_state.with_indexed_output(column_index, output_token, reserve);

        new_state
    }

    // fn into_data<U: Default>(self, fn_value_into_data: impl Fn(T) -> U) -> Vec<U> {
    //     let fn_ref = &fn_value_into_data;

    //     self.rows
    //         .into_iter()
    //         .flat_map(|row| {
    //             row.into_iter()
    //                 .map(|opt_value| opt_value.map_or(U::default(), fn_ref))
    //         })
    //         .collect()
    // }

    fn get_indexes(&self, input_token: &I, output_token: &I) -> Vec<(RowIndex, ColumnIndex)> {
        self.input_indexes
            .get(input_token)
            .and_then(|column_index| {
                self.output_indexes.get(output_token).map(|set| {
                    set.into_iter()
                        .map(|row_index| (*row_index, *column_index))
                        .collect()
                })
            })
            .unwrap_or_default()
    }

    fn bypass_indexes(&self, bridges: &HashSet<(I, I)>) -> HashSet<(RowIndex, ColumnIndex)> {
        self.input_indexes
            .iter()
            .filter_map(move |(input_token_address, column_index)| {
                self.output_indexes
                    .get(input_token_address)
                    .map(|set| (column_index, set))
                // .and_then(|set| set.iter().next())
                // .map(|row_index| (*row_index, *column_index))
            })
            .flat_map(|(column_index, set)| set.iter().map(|row_index| (*row_index, *column_index)))
            .chain(
                bridges
                    .into_iter()
                    .flat_map(|(token0, token1)| self.get_indexes(token0, token1)),
            )
            .collect()
    }

    fn output_token_indexes(&self) -> Vec<usize> {
        let mut indexes = vec![0; self.rows.len()];

        self.output_indexes
            .iter()
            .flat_map(|(token_address, idxs)| idxs.iter().map(move |idx| (token_address, idx)))
            .for_each(|(token_address, RowIndex(idx))| {
                indexes[*idx] = self.input_indexes.get(token_address).unwrap().0;
            });

        indexes
    }

    fn inputs(&self) -> Vec<I> {
        let mut tokens = vec![None; self.input_indexes.len()];
        self.input_indexes
            .iter()
            .for_each(|(token_address, ColumnIndex(idx))| {
                tokens[*idx] = Some(*token_address);
            });

        tokens.into_iter().collect::<Option<Vec<_>>>().unwrap()
    }

    fn outputs(&self) -> Vec<I> {
        let mut tokens = vec![None; self.rows.len()];
        for (token_address, RowIndex(idx)) in
            self.output_indexes
                .iter()
                .flat_map(|(token_address, indexes_set)| {
                    indexes_set.iter().map(move |idx| (token_address, idx))
                })
        {
            tokens[*idx] = Some(*token_address);
        }
        tokens.into_iter().collect::<Option<Vec<_>>>().unwrap()
    }

    pub fn shape(&self) -> [usize; 2] {
        let rows_len = self.rows.len();
        if rows_len == 0 {
            [0, 0]
        } else {
            [rows_len, self.rows[0].len()]
        }
    }
}

impl<T: Copy + PartialEq, I: Clone + Copy + PartialEq + Eq + std::hash::Hash> ModelLayout<T, I> {
    fn find_reserve_position(
        &self,
        input_token: &I,
        output_token: &I,
        reserve: &T,
    ) -> Option<(RowIndex, ColumnIndex)> {
        let col = *self.input_indexes.get(input_token)?;
        let row_set = self.output_indexes.get(output_token)?;

        row_set.iter().find_map(|&row| {
            access_reserve!(self, row, col)
                .filter(|id| *id == *reserve)
                .map(|_| (row, col))
        })
    }
}

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    weights: Param<Tensor<B, 2>>,
    output_size: usize,
}

impl<B: Backend> Layer<B> {
    pub fn init(dims: [usize; 2], output_size: usize, device: &B::Device) -> Self {
        let weights = Tensor::random(Shape::new(dims), Distribution::Uniform(0.0, 1.0), device);

        Self {
            weights: Param::from_tensor(weights),
            output_size,
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 1>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        gamma: &Tensor<B, 2>,
        bypass_mask: &Tensor<B, 2, Bool>,
        max_swap: &Tensor<B, 2>,
        token_indexes: &Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let dims = self.weights.shape();
        let out_dim = dims.dims[0];
        let w_normalized = softmax(self.weights.val(), 0);

        let input_amounts: Tensor<B, 2> = input.clone().expand(dims).mul(w_normalized);
        let input_amounts_after_fee: Tensor<B, 2> = input_amounts.clone().mul(gamma.clone());

        let capped_input_amounts_after_fee =
            input_amounts_after_fee.clone().min_pair(max_swap.clone());

        // U = input * G * softmax(W)
        // Z = Y * U / (X + U + epsilon)
        let denominator = x
            .clone()
            .add(capped_input_amounts_after_fee.clone())
            .add_scalar(f32::EPSILON);

        let z = y
            .clone()
            .mul(capped_input_amounts_after_fee.clone().div(denominator))
            .clamp_min(0.0)
            .mask_where(bypass_mask.clone(), input_amounts);

        let out = z.sum_dim(1).reshape([out_dim]);

        Tensor::zeros([self.output_size], &input.device()).select_assign(
            0,
            token_indexes.clone(),
            out,
        )
    }
}

#[derive(Module, Debug)]
struct LayerBlock<B: Backend, const LAYERS: usize> {
    reserves_in: Tensor<B, 2>,
    reserves_out: Tensor<B, 2>,
    fee_multiplier: Tensor<B, 2>,
    bypass_mask: Tensor<B, 2, Bool>,
    max_swap: Tensor<B, 2>,
    init_asset_index: i64,
    output_asset_indexes: Tensor<B, 1, Int>,
    layer_out_output_asset_indexes: Tensor<B, 1, Int>,
    layer_out_bypass_mask: Tensor<B, 2, Bool>,
    layer_out_pool_indexes: Tensor<B, 1, Int>,
    layer_in: Layer<B>,
    layers: [Layer<B>; LAYERS],
    layer_out: Layer<B>,
}

impl<B: Backend, const LAYERS: usize> LayerBlock<B, LAYERS> {
    fn layer_in_params(
        &self,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2, Bool>,
        Tensor<B, 2>,
    ) {
        let input_asset_range = [
            Slice::from(..),
            Slice::from(self.init_asset_index..self.init_asset_index + 1),
        ];
        (
            self.reserves_in.clone().slice(input_asset_range),
            self.reserves_out.clone().slice(input_asset_range),
            self.fee_multiplier.clone().slice(input_asset_range),
            self.bypass_mask.clone().slice(input_asset_range),
            self.max_swap.clone().slice(input_asset_range),
        )
    }

    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let (x_in, y_in, gamma_in, bypass_mask_in, max_swap_in) = self.layer_in_params();

        let layer_in_output = self.layer_in.forward(
            input.clone(),
            &x_in,
            &y_in,
            &gamma_in,
            &bypass_mask_in,
            &max_swap_in,
            &self.output_asset_indexes,
        );

        let mut layer_output = layer_in_output;

        for layer in 0..LAYERS {
            layer_output = self.layers[layer].forward(
                layer_output,
                &self.reserves_in,
                &self.reserves_out,
                &self.fee_multiplier,
                &self.bypass_mask,
                &self.max_swap,
                &self.output_asset_indexes,
            );
        }

        let x_out = self
            .reserves_in
            .clone()
            .select(0, self.layer_out_pool_indexes.clone());
        let y_out = self
            .reserves_out
            .clone()
            .select(0, self.layer_out_pool_indexes.clone());
        let gamma_out = self
            .fee_multiplier
            .clone()
            .select(0, self.layer_out_pool_indexes.clone());
        let max_swap_out = self
            .max_swap
            .clone()
            .select(0, self.layer_out_pool_indexes.clone());

        self.layer_out.forward(
            layer_output,
            &x_out,
            &y_out,
            &gamma_out,
            &self.layer_out_bypass_mask,
            &max_swap_out,
            &self.layer_out_output_asset_indexes,
        )
    }
}

pub struct Model<
    B: Backend,
    U: Copy,
    I: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    const LAYERS: usize,
> {
    layout: ModelLayout<U, I>,
    reserves_in_data: Vec<B::FloatElem>,
    reserves_out_data: Vec<B::FloatElem>,
    max_swap_data: Vec<B::FloatElem>,
    block: LayerBlock<B, LAYERS>,
}

pub struct ModelOptimizer<B: AutodiffBackend, const LAYERS: usize> {
    optimizer: OptimizerAdaptor<Adam, LayerBlock<B, LAYERS>, B>,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ModelUpdateError<TPool, TToken> {
    #[error("reserve not found in model layout")]
    ReserveNotFound {
        pool_id: TPool,
        token0: TToken,
        token1: TToken,
    },
}

impl<
    B: AutodiffBackend,
    U: Copy,
    I: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    const LAYERS: usize,
> Model<B, U, I, LAYERS>
{
    pub fn init_optimizer() -> ModelOptimizer<B, LAYERS> {
        let optimizer_config = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8);

        ModelOptimizer {
            optimizer: optimizer_config.init(),
        }
    }

    pub fn optimize(self, input_elem: B::FloatElem, num_iterations: usize) -> Self {
        let optimizer = Self::init_optimizer();
        let (model, _optimizer) = self.optimize_with(optimizer, input_elem, num_iterations);
        model
    }

    pub fn optimize_with(
        self,
        mut optimizer: ModelOptimizer<B, LAYERS>,
        input_elem: B::FloatElem,
        num_iterations: usize,
    ) -> (Self, ModelOptimizer<B, LAYERS>) {
        let input: Tensor<B, 1> = Tensor::from([input_elem]);
        let mut model = self;

        for _ in 0..num_iterations {
            let output = model.block.forward(input.clone());
            let loss = input.clone().sub(output);
            let raw_grads = loss.backward();
            let grads = GradientsParams::from_grads(raw_grads, &model.block);

            model.block = optimizer.optimizer.step(0.1, model.block, grads);
        }

        (model, optimizer)
    }
}

impl<B: Backend, U: Copy, I: Clone + Copy + PartialEq + Eq + std::hash::Hash, const LAYERS: usize>
    Model<B, U, I, LAYERS>
{
    pub fn init(
        init_asset: I,
        pool_reserves: Vec<PoolReserves<U, I>>,
        bridges: &HashSet<(I, I)>,
    ) -> Result<Self, OptimizationError> {
        let device = &B::Device::default();
        let model_layout = pool_reserves
            .into_iter()
            .fold(ModelLayout::new(), |state, reserves| {
                state.with_reserve_values(
                    reserves.token0,
                    reserves.token1,
                    (reserves.pool_id, reserves.value),
                )
            });

        let init_asset_pool_indexes = model_layout
            .output_indexes
            .get(&init_asset)
            .map(|idxs| idxs.iter().map(|RowIndex(idx)| *idx).collect::<Vec<_>>())
            .unwrap();
        let init_asset_pools_count = init_asset_pool_indexes.len();

        let bypass_indexes = model_layout.bypass_indexes(bridges);

        let init_asset_index = model_layout
            .input_indexes
            .get(&init_asset)
            .map(|ColumnIndex(col_index)| *col_index)
            .ok_or(OptimizationError::InitAssetNotFound)?;

        let size_tokens: usize = model_layout.input_indexes.len();
        let size_pools = model_layout.rows.len();
        let dims = [size_pools, size_tokens];
        let buf_size = size_tokens * size_pools;

        let asset_indexes_data = model_layout.output_token_indexes();

        let (
            reserves_in_data,
            reserves_out_data,
            fee_multiplier_data,
            max_swap_data,
            bypass_mask_data,
        ) = model_layout
            .rows
            .iter()
            .enumerate()
            .flat_map(|(row_index, row)| {
                row.iter().enumerate().map(move |(col_index, value)| {
                    (RowIndex(row_index), ColumnIndex(col_index), value)
                })
            })
            .fold(
                (
                    Vec::with_capacity(buf_size),
                    Vec::with_capacity(buf_size),
                    Vec::with_capacity(buf_size),
                    Vec::with_capacity(buf_size),
                    Vec::with_capacity(buf_size),
                ),
                |(
                    mut reserves_in_data,
                    mut reserves_out_data,
                    mut fee_multiplier_data,
                    mut max_swap_data,
                    mut bypass_mask_data,
                ): (
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<bool>,
                ),
                 (row_index, column_index, reserve): (
                    RowIndex,
                    ColumnIndex,
                    &Option<(U, VirtualReserveValues)>,
                )|
                 -> (
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<bool>,
                ) {
                    let reserve =
                        reserve
                            .as_ref()
                            .map(|(_, r)| r)
                            .unwrap_or(&VirtualReserveValues {
                                token_0: 0.0,
                                token_1: 0.0,
                                fee_multiplier: 0.0,
                                max_swap_0: 0.0,
                                max_swap_1: 0.0,
                            });
                    reserves_in_data.push(B::FloatElem::from_elem(reserve.token_0));
                    reserves_out_data.push(B::FloatElem::from_elem(reserve.token_1));
                    fee_multiplier_data.push(B::FloatElem::from_elem(reserve.fee_multiplier));
                    max_swap_data.push(B::FloatElem::from_elem(reserve.max_swap_0));

                    bypass_mask_data.push(if bypass_indexes.contains(&(row_index, column_index)) {
                        true
                    } else {
                        false
                    });

                    (
                        reserves_in_data,
                        reserves_out_data,
                        fee_multiplier_data,
                        max_swap_data,
                        bypass_mask_data,
                    )
                },
            );
        let bypass_mask =
            Tensor::<B, 1, Bool>::from_data(bypass_mask_data.as_slice(), device).reshape(dims);

        let layer_out_pool_indexes =
            Tensor::<B, 1, Int>::from_data(init_asset_pool_indexes.as_slice(), device);

        let layer_out_bypass_mask = bypass_mask
            .clone()
            .int()
            .select(0, layer_out_pool_indexes.clone())
            .bool();
        // let layer_out_bypass_mask_data = vec![true; size_tokens * init_asset_pools_count];

        Ok(Model {
            block: LayerBlock {
                reserves_in: Tensor::<B, 1>::from_data(reserves_in_data.as_slice(), device)
                    .reshape(dims),
                reserves_out: Tensor::<B, 1>::from_data(reserves_out_data.as_slice(), device)
                    .reshape(dims),
                fee_multiplier: Tensor::<B, 1>::from_data(fee_multiplier_data.as_slice(), device)
                    .reshape(dims),
                max_swap: Tensor::<B, 1>::from_data(max_swap_data.as_slice(), device).reshape(dims),
                // input_asset_index: Tensor::<B, 1, Int>::from_data([0 as i32], device),
                init_asset_index: init_asset_index as i64,
                output_asset_indexes: Tensor::<B, 1, Int>::from_data(
                    asset_indexes_data.as_slice(),
                    device,
                ),
                // layer_out output is of shape [1] (only single token) which means all ouput_asset_indexes should be 0
                layer_out_output_asset_indexes: Tensor::<B, 1, Int>::full(
                    [init_asset_pools_count],
                    0,
                    device,
                ),
                bypass_mask,
                layer_out_bypass_mask,
                layer_out_pool_indexes,
                layer_in: Layer::init([size_pools, 1], size_tokens, device),
                layers: [(); LAYERS]
                    .map(|_| Layer::init([size_pools, size_tokens], size_tokens, device)),
                layer_out: Layer::init([init_asset_pools_count, size_tokens], 1, device),
            },
            reserves_in_data,
            reserves_out_data,
            max_swap_data,
            layout: ModelLayout {
                input_indexes: model_layout.input_indexes,
                output_indexes: model_layout.output_indexes,
                rows: model_layout
                    .rows
                    .into_iter()
                    .map(|row| row.into_iter().map(|cell| cell.map(|(id, _)| id)).collect())
                    .collect(),
            },
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        self.layout.shape()
    }

    pub fn evaluate(&self, input: B::FloatElem) -> B::FloatElem {
        let input_tensor: Tensor<B, 1> = Tensor::from([input]);
        self.block.forward(input_tensor).into_scalar()
    }

    pub fn swaps_graph<L1: Default, F: Fn(&U) -> L1>(
        &self,
        layout_value_map: F,
        min_weight: f32,
    ) -> Result<Graph<I, (f32, L1), Directed>, OptimizationError> {
        let graph: Graph<I, (f32, L1), Directed> = Graph::new();

        let inputs = self.layout.inputs();
        let outputs = self.layout.outputs();

        let init_asset = inputs
            .get(self.block.init_asset_index as usize)
            .ok_or(OptimizationError::InvalidInitAssetIndex)?;

        let (graph, init_asset_node_index) = {
            let mut graph = graph;
            let init_asset_node_index = graph.add_node(*init_asset);
            (graph, init_asset_node_index)
        };

        let layer_in_layout = self
            .layout
            .rows
            .iter()
            .map(|row| {
                row.get(self.block.init_asset_index as usize)
                    .ok_or(OptimizationError::InvalidLayoutIndex)
            })
            .collect::<Result<Vec<_>, OptimizationError>>()?;

        let check_and_map_layout_value =
            |token_in: I, token_out: I, layout_value: &Option<U>| -> Option<(I, I, L1)> {
                layout_value
                    .as_ref()
                    .map(&layout_value_map)
                    .or_else(|| {
                        if token_in == token_out {
                            Some(L1::default())
                        } else {
                            None
                        }
                    })
                    .map(|layout_value| (token_in, token_out, layout_value))
            };

        let (graph, layer_in_outputs) = softmax(self.block.layer_in.weights.val(), 0)
            .to_data()
            .iter::<f32>()
            .zip(outputs.iter())
            .zip(layer_in_layout.into_iter())
            .filter(|((weight, _), _)| *weight >= min_weight)
            .filter_map(|((weight, token_out), layout_value)| {
                check_and_map_layout_value(*init_asset, *token_out, layout_value)
                    .map(|(_, token_out, val)| (token_out, (weight, val)))
            })
            .fold(
                (graph, HashMap::<I, NodeIndex>::new()),
                |(mut graph, mut layer_outputs), (token_out, value)| {
                    let node_index = layer_outputs
                        .entry(token_out)
                        .or_insert_with(|| graph.add_node(token_out));

                    graph.add_edge(init_asset_node_index, *node_index, value);
                    (graph, layer_outputs)
                },
            );

        let (graph, layers_output) = self.block.layers.iter().fold(
            (graph, layer_in_outputs),
            |(graph, layer_inputs), layer| {
                let layer_normalized_weights = softmax(layer.weights.val(), 0).to_data();

                let (graph, layer_outputs) = layer_normalized_weights
                    .iter::<f32>()
                    .zip(outputs.iter().flat_map(|token_out| {
                        inputs.iter().map(|token_in| (*token_in, *token_out))
                    }))
                    .zip(self.layout.rows.iter().flat_map(|row| row.iter()))
                    .filter(|((weight, _), _)| *weight >= min_weight)
                    .filter_map(|((weight, (token_in, token_out)), layout_value)| {
                        layer_inputs.get(&token_in).map(|node_index_in| {
                            (
                                *node_index_in,
                                (token_in, token_out),
                                (weight, layout_value),
                            )
                        })
                    })
                    .filter_map(
                        |(node_index_in, (token_in, token_out), (weight, layout_value))| {
                            check_and_map_layout_value(token_in, token_out, layout_value).map(
                                |(_, token_out, val)| (node_index_in, token_out, (weight, val)),
                            )
                        },
                    )
                    .fold(
                        (graph, HashMap::<I, NodeIndex>::new()),
                        |(mut graph, mut layer_outputs), (node_index_in, token_out, value)| {
                            let node_index_out = layer_outputs
                                .entry(token_out)
                                .or_insert_with(|| graph.add_node(token_out));

                            graph.add_edge(node_index_in, *node_index_out, value);
                            (graph, layer_outputs)
                        },
                    );

                (graph, layer_outputs)
            },
        );

        let output_pool_indexes = self.block.layer_out_pool_indexes.to_data();
        let layer_out_layout = output_pool_indexes
            .iter::<i32>()
            .map(|output_index| {
                let r = self
                    .layout
                    .rows
                    .get(output_index as usize)
                    .ok_or(OptimizationError::InvalidOutputIndex);
                r
            })
            .collect::<Result<Vec<_>, OptimizationError>>()?;

        let (graph, output_asset_node_index) = {
            let mut graph = graph;
            let output_asset_node_index = graph.add_node(*init_asset);
            (graph, output_asset_node_index)
        };

        let graph = softmax(self.block.layer_out.weights.val(), 0)
            .to_data()
            .iter::<f32>()
            .zip(
                output_pool_indexes
                    .iter::<i32>()
                    .flat_map(|_| inputs.iter().copied()),
            )
            .zip(layer_out_layout.into_iter().flatten())
            .filter(|((weight, _), _)| *weight >= min_weight)
            .filter_map(|((weight, token_in), layout_value)| {
                layers_output
                    .get(&token_in)
                    .map(|node_index_in| (*node_index_in, token_in, (weight, layout_value)))
            })
            .filter_map(|(node_index_in, token_in, (weight, layout_value))| {
                check_and_map_layout_value(token_in, *init_asset, layout_value)
                    .map(|(_, _, val)| (node_index_in, (weight, val)))
            })
            .fold(graph, |mut graph, (node_index_in, value)| {
                graph.add_edge(node_index_in, output_asset_node_index, value);
                graph
            });

        Ok(graph)
    }
}

impl<
    B: Backend,
    U: Copy + PartialEq,
    I: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    const LAYERS: usize,
> Model<B, U, I, LAYERS>
{
    pub fn update(
        mut self,
        reserves: Vec<PoolReserves<U, I>>,
    ) -> Result<Self, ModelUpdateError<U, I>> {
        let [_, m] = self.layout.shape();
        // let mut max_swap_data = vec![0.0; n * m];
        // let mut mask_data = vec![false; n * m];

        for reserve in reserves {
            let Some((RowIndex(row), ColumnIndex(col))) = self.layout.find_reserve_position(
                &reserve.token0,
                &reserve.token1,
                &reserve.pool_id,
            ) else {
                return Err(ModelUpdateError::ReserveNotFound {
                    pool_id: reserve.pool_id,
                    token0: reserve.token0,
                    token1: reserve.token1,
                });
            };

            let index = row * m + col;
            self.reserves_in_data[index] = B::FloatElem::from_elem(reserve.value.token_0);
            self.reserves_out_data[index] = B::FloatElem::from_elem(reserve.value.token_1);
            self.max_swap_data[index] = B::FloatElem::from_elem(reserve.value.max_swap_0);
        }

        self.block.reserves_in.inplace(|r_in| {
            Tensor::<B, 1>::from_data(self.reserves_in_data.as_slice(), &r_in.device())
                .reshape(r_in.shape())
        });
        self.block.reserves_out.inplace(|r_out| {
            Tensor::<B, 1>::from_data(self.reserves_out_data.as_slice(), &r_out.device())
                .reshape(r_out.shape())
        });
        self.block.max_swap.inplace(|max_swap| {
            Tensor::<B, 1>::from_data(self.max_swap_data.as_slice(), &max_swap.device())
                .reshape(max_swap.shape())
        });

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::pool_reserves::test::plant_arbitrage;
    use crate::tokens::test::{self as tokens, TokenAddress};
    use crate::utils::Invertible;

    use super::*;
    type WgpuBackend = burn::backend::Autodiff<burn::backend::Wgpu<f32>>;
    type CpuBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    use burn::tensor::{Int, Tensor};
    use proptest::prelude::*;

    /// Probes whether a usable WGPU adapter exists, swallowing the adapter-selection panic
    /// that the backend raises in headless environments (e.g. CI / devcontainers without a GPU).
    ///
    /// Memoized in a `OnceLock` so the probe runs exactly once per process: it mutates the global
    /// panic hook, which is not safe to do concurrently from the parallel test harness.
    fn wgpu_adapter_available() -> bool {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            let previous_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let available = std::panic::catch_unwind(|| {
                let device = burn::backend::wgpu::WgpuDevice::default();
                // Force actual execution + host readback: cubecl acquires the adapter lazily, so
                // merely allocating a tensor does not surface a missing GPU. Reading data back
                // does, raising (and here catching) the adapter-selection panic in headless envs.
                let _data = Tensor::<WgpuBackend, 1>::zeros([1], &device)
                    .add_scalar(1.0)
                    .into_data();
            })
            .is_ok();
            std::panic::set_hook(previous_hook);
            available
        })
    }

    /// Runs `wgpu` when a WGPU adapter is available, otherwise falls back to `cpu`, so
    /// backend-generic tests still exercise their logic (on ndarray) without a GPU.
    fn run_on_available_backend(wgpu: impl FnOnce(), cpu: impl FnOnce()) {
        if wgpu_adapter_available() {
            wgpu();
        } else {
            cpu();
        }
    }

    fn scatter_example_on<B: Backend>() {
        let device = B::Device::default();

        let pool_outputs: Tensor<B, 1> =
            Tensor::from_data([5.0, 5.0, 3.0, 3.0, 4.0, 10.0], &device);

        let pool_token_indices: Tensor<B, 1, Int> =
            Tensor::from_data([0, 0, 1, 1, 1, 2], &device);

        let output: Tensor<B, 1> =
            Tensor::zeros([3], &device).scatter(0, pool_token_indices, pool_outputs);

        println!("{}", output);
    }

    #[test]
    fn scatter_example() {
        run_on_available_backend(
            || scatter_example_on::<WgpuBackend>(),
            || scatter_example_on::<CpuBackend>(),
        );
    }

    #[test]
    fn optimize_with_accepts_and_returns_optimizer() {
        type CpuBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

        let reserve = PoolReserves {
            token0: tokens::USDC.address,
            token1: tokens::WETH.address,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 1_000.0,
                fee_multiplier: 1.0,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        };
        let model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            vec![reserve, reserve.inverse()],
            &HashSet::new(),
        )
        .unwrap();

        let optimizer = Model::<CpuBackend, i32, TokenAddress, 1>::init_optimizer();
        let (model, optimizer) = model.optimize_with(optimizer, 100.0, 0);
        let (_model, _optimizer) = model.optimize_with(optimizer, 100.0, 0);
    }

    #[test]
    fn model_update_returns_error_for_unknown_directional_reserve() {
        type CpuBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

        let known_reserve = PoolReserves {
            token0: tokens::USDC.address,
            token1: tokens::WETH.address,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 1_000.0,
                fee_multiplier: 1.0,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        };
        let unknown_reserve = PoolReserves {
            pool_id: 2,
            ..known_reserve
        };
        let model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            vec![known_reserve, known_reserve.inverse()],
            &HashSet::new(),
        )
        .unwrap();

        let error = match model.update(vec![unknown_reserve]) {
            Ok(_) => panic!("unknown reserve update unexpectedly succeeded"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            ModelUpdateError::ReserveNotFound {
                pool_id: unknown_reserve.pool_id,
                token0: unknown_reserve.token0,
                token1: unknown_reserve.token1,
            }
        );
    }

    #[test]
    fn test_pools_fold_state_layout_example() {
        // let device = burn::backend::wgpu::WgpuDevice::default();
        //  pool_values in prop::collection::hash_map(any::<i64>(),prop::collection::hash_set(any::<TokenAddress>(), 2), 1..1000)
        let pool_values = [
            // USDC/WBTC
            (tokens::USDC.address, tokens::WBTC.address, "USDC/WBTC-1"),
            (tokens::USDC.address, tokens::WBTC.address, "USDC/WBTC-2"),
            // USDC/WETH
            (tokens::USDC.address, tokens::WETH.address, "USDC/WETH-1"),
            (tokens::USDC.address, tokens::WETH.address, "USDC/WETH-2"),
            (tokens::USDC.address, tokens::WETH.address, "USDC/WETH-3"),
            // WBTC/WETH
            (tokens::WBTC.address, tokens::WETH.address, "WBTC/WETH-1"),
            (tokens::WBTC.address, tokens::WETH.address, "WBTC/WETH-2"),
            (tokens::WBTC.address, tokens::WETH.address, "WBTC/WETH-3"),
            // reverse
            // WBTC/USDC
            (tokens::WBTC.address, tokens::USDC.address, "WBTC/USDC-1"),
            (tokens::WBTC.address, tokens::USDC.address, "WBTC/USDC-2"),
            // WETH/USDC
            (tokens::WETH.address, tokens::USDC.address, "WETH/USDC-1"),
            (tokens::WETH.address, tokens::USDC.address, "WETH/USDC-2"),
            (tokens::WETH.address, tokens::USDC.address, "WETH/USDC-3"),
            // WETH/WBTC
            (tokens::WETH.address, tokens::WBTC.address, "WETH/WBTC-1"),
            (tokens::WETH.address, tokens::WBTC.address, "WETH/WBTC-2"),
            (tokens::WETH.address, tokens::WBTC.address, "WETH/WBTC-3"),
        ];

        let state = pool_values
            .into_iter()
            .fold(ModelLayout::new(), |state, (from, to, value)| {
                state.with_reserve_values(from, to, value)
            });

        for row in state.rows.iter() {
            println!(
                "{:?}",
                row.iter()
                    .map(|v| v.unwrap_or("----/------"))
                    .collect::<Vec<_>>()
            );
        }

        // output:
        // ["----/------", "USDC/WBTC-1", "WETH/WBTC-1"]
        // ["----/------", "USDC/WBTC-2", "WETH/WBTC-2"]
        // ["WBTC/WETH-1", "USDC/WETH-1", "----/------"]
        // ["WBTC/WETH-3", "USDC/WETH-2", "----/------"]
        // ["WBTC/WETH-2", "USDC/WETH-3", "----/------"]
        // ["WBTC/USDC-1", "----/------", "WETH/USDC-2"]
        // ["WBTC/USDC-2", "----/------", "WETH/USDC-1"]
        // ["----/------", "----/------", "WETH/USDC-3"]
        // ["----/------", "----/------", "WETH/WBTC-3"]

        println!("------------");
        println!(
            "{:?}",
            state
                .rows
                .iter()
                .flat_map(|r| r.iter().map(|v| v.unwrap_or("----/------")))
                .collect::<Vec<_>>()
        );
        println!("------------");

        let pool_values = [
            // USDC/WBTC
            (tokens::USDC.address, tokens::WBTC.address, 9.0),
            (tokens::USDC.address, tokens::WBTC.address, 9.1),
            // USDC/WETH
            (tokens::USDC.address, tokens::WETH.address, 4.0),
            (tokens::USDC.address, tokens::WETH.address, 4.1),
            (tokens::USDC.address, tokens::WETH.address, 4.2),
            // WBTC/WETH
            (tokens::WBTC.address, tokens::WETH.address, 7.0),
            (tokens::WBTC.address, tokens::WETH.address, 7.1),
            (tokens::WBTC.address, tokens::WETH.address, 7.2),
            // reverse
            // WBTC/USDC
            (tokens::WBTC.address, tokens::USDC.address, -9.0),
            (tokens::WBTC.address, tokens::USDC.address, -9.1),
            // WETH/USDC
            (tokens::WETH.address, tokens::USDC.address, -4.0),
            (tokens::WETH.address, tokens::USDC.address, -4.1),
            (tokens::WETH.address, tokens::USDC.address, -4.2),
            // WETH/WBTC
            (tokens::WETH.address, tokens::WBTC.address, -7.0),
            (tokens::WETH.address, tokens::WBTC.address, -7.1),
            (tokens::WETH.address, tokens::WBTC.address, -7.2),
        ];

        let state = pool_values
            .into_iter()
            .fold(ModelLayout::new(), |state, (from, to, value)| {
                state.with_reserve_values(from, to, value)
            });

        for row in state.rows.iter() {
            println!(
                "{:?}",
                row.iter().map(|v| v.unwrap_or(0.0)).collect::<Vec<_>>()
            );
        }

        println!("------------");

        // let tensor: Tensor<WgpuBackend, 2> =
        //     Tensor::<WgpuBackend, 1>::from_data(state.into_data(|v| v).as_slice(), &device)
        //         .reshape([9, 3]);

        // println!("{}", tensor);
    }

    #[test]
    fn model_layout_treats_reserves_as_directional_edges() {
        let forward = "USDC/WETH";
        let reverse = "WETH/USDC";

        let state = ModelLayout::new().with_reserve_values(
            tokens::USDC.address,
            tokens::WETH.address,
            forward,
        );

        assert_eq!(state.shape(), [1, 2]);
        assert_eq!(
            state
                .get_indexes(&tokens::USDC.address, &tokens::WETH.address)
                .len(),
            1
        );
        assert!(
            state
                .get_indexes(&tokens::WETH.address, &tokens::USDC.address)
                .is_empty(),
            "a USDC -> WETH reserve must not create a WETH -> USDC cell"
        );

        let state = state.with_reserve_values(tokens::WETH.address, tokens::USDC.address, reverse);

        assert_eq!(state.shape(), [2, 2]);

        let forward_cells = state
            .get_indexes(&tokens::USDC.address, &tokens::WETH.address)
            .into_iter()
            .filter_map(|(row, column)| access_reserve!(state, row, column))
            .collect::<Vec<_>>();
        let reverse_cells = state
            .get_indexes(&tokens::WETH.address, &tokens::USDC.address)
            .into_iter()
            .filter_map(|(row, column)| access_reserve!(state, row, column))
            .collect::<Vec<_>>();

        assert_eq!(forward_cells, vec![forward]);
        assert_eq!(reverse_cells, vec![reverse]);
    }

    #[test]
    fn test_model_layout_bridges() {
        let pool_values = [
            // USDC/WBTC
            (tokens::USDC.address, tokens::WBTC.address, "USDC/WBTC-1"),
            (tokens::USDC.address, tokens::WBTC.address, "USDC/WBTC-2"),
            // USDC/WETH
            (tokens::USDC.address, tokens::WETH.address, "USDC/WETH-1"),
            (tokens::USDC.address, tokens::WETH.address, "USDC/WETH-2"),
            (tokens::USDC.address, tokens::WETH.address, "USDC/WETH-3"),
            // WBTC/WETH
            (tokens::WBTC.address, tokens::WETH.address, "WBTC/WETH-1"),
            (tokens::WBTC.address, tokens::WETH.address, "WBTC/WETH-2"),
            (tokens::WBTC.address, tokens::WETH.address, "WBTC/WETH-3"),
            // reverse
            // WBTC/USDC
            (tokens::WBTC.address, tokens::USDC.address, "WBTC/USDC-1"),
            (tokens::WBTC.address, tokens::USDC.address, "WBTC/USDC-2"),
            // WETH/USDC
            (tokens::WETH.address, tokens::USDC.address, "WETH/USDC-1"),
            (tokens::WETH.address, tokens::USDC.address, "WETH/USDC-2"),
            (tokens::WETH.address, tokens::USDC.address, "WETH/USDC-3"),
            // WETH/WBTC
            (tokens::WETH.address, tokens::WBTC.address, "WETH/WBTC-1"),
            (tokens::WETH.address, tokens::WBTC.address, "WETH/WBTC-2"),
            (tokens::WETH.address, tokens::WBTC.address, "WETH/WBTC-3"),
            // ETH/USDC
            (tokens::ETH.address, tokens::USDC.address, " ETH/USDC-1"),
            // WBTC/ETH
            (tokens::WBTC.address, tokens::ETH.address, "WBTC/ETH -1"),
        ];

        let state = pool_values
            .into_iter()
            .fold(ModelLayout::new(), |state, (from, to, value)| {
                state.with_reserve_values(from, to, value)
            });

        for row in state.rows.iter() {
            println!(
                "{:?}",
                row.iter()
                    .map(|v| v.unwrap_or("----/------"))
                    .collect::<Vec<_>>()
            );
        }

        let bridges = HashSet::from([
            (tokens::ETH.address, tokens::WETH.address),
            (tokens::WETH.address, tokens::ETH.address),
        ]);
        let inputs = state.inputs();
        let outputs = state.outputs();

        let bypasses = state
            .bypass_indexes(&bridges)
            .into_iter()
            .map(|(RowIndex(row), ColumnIndex(col))| (inputs[col], outputs[row]))
            .collect::<HashSet<_>>();

        assert!(
            bypasses.contains(&(tokens::USDC.address, tokens::USDC.address)),
            "USDC bypass not found"
        );

        assert!(
            bypasses.contains(&(tokens::WETH.address, tokens::WETH.address)),
            "WETH bypass not found"
        );

        assert!(
            bypasses.contains(&(tokens::WBTC.address, tokens::WBTC.address)),
            "WBTC bypass not found"
        );
        assert!(
            bypasses.contains(&(tokens::ETH.address, tokens::ETH.address)),
            "ETH bypass not found"
        );
        assert!(
            bypasses.contains(&(tokens::ETH.address, tokens::WETH.address)),
            "ETH/WETH bypass not found"
        );
        assert!(
            bypasses.contains(&(tokens::WETH.address, tokens::ETH.address)),
            "WETH/ETH bypass not found"
        )
    }

    fn test_model_v4_arbitrage_on<B: AutodiffBackend + Backend<FloatElem = f32>>() {
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 262218805704089.0, token_1: 6.871874430049188e22, fee_multiplier: 0.9995, max_swap_0: 10192072198.0, max_swap_1: 3.1694012912707076e19 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 43855854099.0, token_1: 1.3556597708069536e22, fee_multiplier: 0.997, max_swap_0: 94586211.0, max_swap_1: 1.1465868119198167e19 }
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 80376539818526.0, token_1: 2.102347156773711e22, fee_multiplier: 0.997, max_swap_0: 126770855354.0, max_swap_1: 2.995623004901809e19 }
        // "WBTC"-"USDC" | PoolVirtualReserves { token_0: 462151969675.0, token_1: 545420903077309.0, fee_multiplier: 0.997, max_swap_0: 1335853624.0, max_swap_1: 61915457710.0 }
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 45720116003508.0, token_1: 1.1980926258959092e22, fee_multiplier: 0.9999, max_swap_0: 286962261.0, max_swap_1: 5.2382977845721344e17 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 12632666761.0, token_1: 3.913756590229623e21, fee_multiplier: 0.9995, max_swap_0: 3510134.0, max_swap_1: 8.695450339924197e17 }
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 745343052233.0, token_1: 1.962743914341458e20, fee_multiplier: 0.99, max_swap_0: 3700026621.0, max_swap_1: 9.932168687766484e17 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 131646243.0, token_1: 4.101824832163563e19, fee_multiplier: 0.99, max_swap_0: 411558.0, max_swap_1: 2.8310161695547008e17 }
        // "WBTC"-"USDC" | PoolVirtualReserves { token_0: 39322226618.0, token_1: 46496788110179.0, fee_multiplier: 0.9995, max_swap_0: 13810835.0, max_swap_1: 6919913682.0 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 1621482.0, token_1: 5.0402880355547795e17, fee_multiplier: 0.9999, max_swap_0: 66.0, max_swap_1: 4658623672271.0 }
        // "WBTC"-"USDC" | PoolVirtualReserves { token_0: 322758584.0, token_1: 377899950726.0, fee_multiplier: 0.99, max_swap_0: 942717.0, max_swap_1: 2686146478.0 }

        let pool_reserves_map = vec![
            (
                (tokens::USDC.address, tokens::WETH.address, 500),
                VirtualReserveValues {
                    token_0: 262218805704089.0,
                    token_1: 6.871874430049188e22,
                    fee_multiplier: 0.9995,
                    max_swap_0: 10192072198.0,
                    max_swap_1: 3.1694012912707076e19,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 3000),
                VirtualReserveValues {
                    token_0: 43855854099.0,
                    token_1: 1.3556597708069536e22,
                    fee_multiplier: 0.997,
                    max_swap_0: 94586211.0,
                    max_swap_1: 1.1465868119198167e19,
                },
            ),
            (
                (tokens::USDC.address, tokens::WETH.address, 3000),
                VirtualReserveValues {
                    token_0: 80376539818526.0,
                    token_1: 2.102347156773711e22,
                    fee_multiplier: 0.997,
                    max_swap_0: 126770855354.0,
                    max_swap_1: 2.995623004901809e19,
                },
            ),
            (
                (tokens::WBTC.address, tokens::USDC.address, 3000),
                VirtualReserveValues {
                    token_0: 462151969675.0,
                    token_1: 545420903077309.0,
                    fee_multiplier: 0.997,
                    max_swap_0: 1335853624.0,
                    max_swap_1: 61915457710.0,
                },
            ),
            (
                (tokens::USDC.address, tokens::WETH.address, 100),
                VirtualReserveValues {
                    token_0: 45720116003508.0,
                    token_1: 1.1980926258959092e22,
                    fee_multiplier: 0.9999,
                    max_swap_0: 286962261.0,
                    max_swap_1: 5.2382977845721344e17,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 500),
                VirtualReserveValues {
                    token_0: 12632666761.0,
                    token_1: 3.913756590229623e21,
                    fee_multiplier: 0.9995,
                    max_swap_0: 3510134.0,
                    max_swap_1: 8.695450339924197e17,
                },
            ),
            (
                (tokens::USDC.address, tokens::WETH.address, 10000),
                VirtualReserveValues {
                    token_0: 745343052233.0,
                    token_1: 1.962743914341458e20,
                    fee_multiplier: 0.99,
                    max_swap_0: 3700026621.0,
                    max_swap_1: 9.932168687766484e17,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 10000),
                VirtualReserveValues {
                    token_0: 131646243.0,
                    token_1: 4.101824832163563e19,
                    fee_multiplier: 0.99,
                    max_swap_0: 411558.0,
                    max_swap_1: 2.8310161695547008e17,
                },
            ),
            (
                (tokens::WBTC.address, tokens::USDC.address, 500),
                VirtualReserveValues {
                    token_0: 39322226618.0,
                    token_1: 46496788110179.0,
                    fee_multiplier: 0.9995,
                    max_swap_0: 13810835.0,
                    max_swap_1: 6919913682.0,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 100),
                VirtualReserveValues {
                    token_0: 1621482.0,
                    token_1: 5.0402880355547795e17,
                    fee_multiplier: 0.9999,
                    max_swap_0: 66.0,
                    max_swap_1: 4658623672271.0,
                },
            ),
            (
                (tokens::WBTC.address, tokens::USDC.address, 10000),
                VirtualReserveValues {
                    token_0: 322758584.0,
                    token_1: 377899950726.0,
                    fee_multiplier: 0.99,
                    max_swap_0: 942717.0,
                    max_swap_1: 2686146478.0,
                },
            ),
        ]
        .into_iter()
        .flat_map(|((t0, t1, fee), reserve)| {
            [
                ((t0, t1, fee), reserve.clone()),
                ((t1, t0, fee), reserve.inverse()),
            ]
        })
        .collect::<HashMap<_, _>>();

        let (pool_reserves_map, arbitrage_amount) = plant_arbitrage(pool_reserves_map);
        let reserves = pool_reserves_map
            .into_iter()
            .filter(|((_, _, fee), _)| *fee == 3000)
            .map(|((from, to, _), reserve)| {
                // (
                //     from,
                //     to,
                //     (dex::pool_id::PoolId::UniswapV3(Address::default()), reserve),
                // )
                PoolReserves {
                    token0: from,
                    token1: to,
                    pool_id: [0u8; 20],
                    value: reserve,
                }
            })
            .collect::<Vec<_>>();

        let model = Model::<B, [u8; 20], TokenAddress, 1>::init(
            tokens::USDC.address,
            reserves,
            &HashSet::new(),
        )
        .unwrap();

        let input_amount = 1000.0;

        let output_before = model.evaluate(input_amount);
        println!("Output before: {}", output_before);

        let model = model.optimize(input_amount, 100);

        let output_amount = model.evaluate(input_amount);

        assert!(
            (output_amount - input_amount - arbitrage_amount as f32).abs()
                < arbitrage_amount as f32 * 0.01,
            "Planted arbitrage not found. Expected to find {}. Found {}",
            arbitrage_amount,
            output_amount - input_amount
        );
    }

    // Ignored: the assertion is miscalibrated, not the optimizer. `plant_arbitrage` perturbs
    // ~1e22-scale reserves by +100.0 (negligible), so the "planted" USDC->WETH->WBTC->USDC cycle
    // is actually a ~1% net loss (just fees). The optimizer correctly declines and returns
    // ~break-even, which the assertion then rejects. This test never ran before (it was
    // WGPU-gated and CI has no GPU); it now falls back to CPU and is ready to re-enable once
    // `plant_arbitrage` is reworked to create a genuinely profitable cycle.
    #[ignore = "plant_arbitrage produces a net loss, not an arbitrage; assertion needs rework"]
    #[test]
    fn test_model_v4_arbitrage() {
        run_on_available_backend(
            || test_model_v4_arbitrage_on::<WgpuBackend>(),
            || test_model_v4_arbitrage_on::<CpuBackend>(),
        );
    }

    fn test_model_v4_update_on<B: AutodiffBackend + Backend<FloatElem = f32>>() {
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 262218805704089.0, token_1: 6.871874430049188e22, fee_multiplier: 0.9995, max_swap_0: 10192072198.0, max_swap_1: 3.1694012912707076e19 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 43855854099.0, token_1: 1.3556597708069536e22, fee_multiplier: 0.997, max_swap_0: 94586211.0, max_swap_1: 1.1465868119198167e19 }
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 80376539818526.0, token_1: 2.102347156773711e22, fee_multiplier: 0.997, max_swap_0: 126770855354.0, max_swap_1: 2.995623004901809e19 }
        // "WBTC"-"USDC" | PoolVirtualReserves { token_0: 462151969675.0, token_1: 545420903077309.0, fee_multiplier: 0.997, max_swap_0: 1335853624.0, max_swap_1: 61915457710.0 }
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 45720116003508.0, token_1: 1.1980926258959092e22, fee_multiplier: 0.9999, max_swap_0: 286962261.0, max_swap_1: 5.2382977845721344e17 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 12632666761.0, token_1: 3.913756590229623e21, fee_multiplier: 0.9995, max_swap_0: 3510134.0, max_swap_1: 8.695450339924197e17 }
        // "USDC"-"WETH" | PoolVirtualReserves { token_0: 745343052233.0, token_1: 1.962743914341458e20, fee_multiplier: 0.99, max_swap_0: 3700026621.0, max_swap_1: 9.932168687766484e17 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 131646243.0, token_1: 4.101824832163563e19, fee_multiplier: 0.99, max_swap_0: 411558.0, max_swap_1: 2.8310161695547008e17 }
        // "WBTC"-"USDC" | PoolVirtualReserves { token_0: 39322226618.0, token_1: 46496788110179.0, fee_multiplier: 0.9995, max_swap_0: 13810835.0, max_swap_1: 6919913682.0 }
        // "WBTC"-"WETH" | PoolVirtualReserves { token_0: 1621482.0, token_1: 5.0402880355547795e17, fee_multiplier: 0.9999, max_swap_0: 66.0, max_swap_1: 4658623672271.0 }
        // "WBTC"-"USDC" | PoolVirtualReserves { token_0: 322758584.0, token_1: 377899950726.0, fee_multiplier: 0.99, max_swap_0: 942717.0, max_swap_1: 2686146478.0 }

        let pool_reserves_map = vec![
            (
                (tokens::USDC.address, tokens::WETH.address, 500),
                VirtualReserveValues {
                    token_0: 262218805704089.0,
                    token_1: 6.871874430049188e22,
                    fee_multiplier: 0.9995,
                    max_swap_0: 10192072198.0,
                    max_swap_1: 3.1694012912707076e19,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 3000),
                VirtualReserveValues {
                    token_0: 43855854099.0,
                    token_1: 1.3556597708069536e22,
                    fee_multiplier: 0.997,
                    max_swap_0: 94586211.0,
                    max_swap_1: 1.1465868119198167e19,
                },
            ),
            (
                (tokens::USDC.address, tokens::WETH.address, 3000),
                VirtualReserveValues {
                    token_0: 80376539818526.0,
                    token_1: 2.102347156773711e22,
                    fee_multiplier: 0.997,
                    max_swap_0: 126770855354.0,
                    max_swap_1: 2.995623004901809e19,
                },
            ),
            (
                (tokens::WBTC.address, tokens::USDC.address, 3000),
                VirtualReserveValues {
                    token_0: 462151969675.0,
                    token_1: 545420903077309.0,
                    fee_multiplier: 0.997,
                    max_swap_0: 1335853624.0,
                    max_swap_1: 61915457710.0,
                },
            ),
            (
                (tokens::USDC.address, tokens::WETH.address, 100),
                VirtualReserveValues {
                    token_0: 45720116003508.0,
                    token_1: 1.1980926258959092e22,
                    fee_multiplier: 0.9999,
                    max_swap_0: 286962261.0,
                    max_swap_1: 5.2382977845721344e17,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 500),
                VirtualReserveValues {
                    token_0: 12632666761.0,
                    token_1: 3.913756590229623e21,
                    fee_multiplier: 0.9995,
                    max_swap_0: 3510134.0,
                    max_swap_1: 8.695450339924197e17,
                },
            ),
            (
                (tokens::USDC.address, tokens::WETH.address, 10000),
                VirtualReserveValues {
                    token_0: 745343052233.0,
                    token_1: 1.962743914341458e20,
                    fee_multiplier: 0.99,
                    max_swap_0: 3700026621.0,
                    max_swap_1: 9.932168687766484e17,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 10000),
                VirtualReserveValues {
                    token_0: 131646243.0,
                    token_1: 4.101824832163563e19,
                    fee_multiplier: 0.99,
                    max_swap_0: 411558.0,
                    max_swap_1: 2.8310161695547008e17,
                },
            ),
            (
                (tokens::WBTC.address, tokens::USDC.address, 500),
                VirtualReserveValues {
                    token_0: 39322226618.0,
                    token_1: 46496788110179.0,
                    fee_multiplier: 0.9995,
                    max_swap_0: 13810835.0,
                    max_swap_1: 6919913682.0,
                },
            ),
            (
                (tokens::WBTC.address, tokens::WETH.address, 100),
                VirtualReserveValues {
                    token_0: 1621482.0,
                    token_1: 5.0402880355547795e17,
                    fee_multiplier: 0.9999,
                    max_swap_0: 66.0,
                    max_swap_1: 4658623672271.0,
                },
            ),
            (
                (tokens::WBTC.address, tokens::USDC.address, 10000),
                VirtualReserveValues {
                    token_0: 322758584.0,
                    token_1: 377899950726.0,
                    fee_multiplier: 0.99,
                    max_swap_0: 942717.0,
                    max_swap_1: 2686146478.0,
                },
            ),
        ]
        .into_iter()
        .flat_map(|((t0, t1, fee), reserve)| {
            [
                ((t0, t1, fee), reserve.clone()),
                ((t1, t0, fee), reserve.inverse()),
            ]
        })
        .collect::<HashMap<_, _>>();

        let initial_reserves = pool_reserves_map
            .iter()
            // .map(|((from, to, fee), reserve)| (*from, *to, (*fee, reserve.clone())))
            .map(|((from, to, fee), reserve)| PoolReserves {
                token0: *from,
                token1: *to,
                pool_id: *fee,
                value: reserve.clone(),
            })
            // .filter(|(from, to, _)| from < to)
            .collect::<Vec<_>>();

        let updated_reserves = initial_reserves
            .iter()
            .take(4)
            .map(|res| PoolReserves {
                token0: res.token0,
                token1: res.token1,
                pool_id: res.pool_id,
                value: VirtualReserveValues {
                    token_0: res.value.token_0 * 1.01,
                    token_1: res.value.token_1 * 0.99,
                    fee_multiplier: res.value.fee_multiplier,
                    max_swap_0: res.value.max_swap_0 * 1.01,
                    max_swap_1: res.value.max_swap_1 * 0.99,
                },
            })
            .collect::<Vec<_>>();

        let final_reserves = vec![
            updated_reserves.clone(),
            initial_reserves.iter().skip(4).cloned().collect::<Vec<_>>(),
        ]
        .concat();

        assert_eq!(
            initial_reserves.len(),
            final_reserves.len(),
            "Reserve lengths mismatch"
        );

        for i in 0..4 {
            assert_eq!(
                final_reserves[i], updated_reserves[i],
                "Updated reserves mismatch at index {}",
                i
            );
        }

        for i in 4..initial_reserves.len() {
            assert_eq!(
                final_reserves[i], initial_reserves[i],
                "Initial reserves mismatch at index {}",
                i
            );
        }

        let mut model_expected = Model::<B, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            final_reserves,
            &HashSet::new(),
        )
        .expect("expected model init failed");

        model_expected.block.layer_in.weights =
            Param::from_tensor(Tensor::ones_like(&model_expected.block.layer_in.weights));

        for i in 0..model_expected.block.layers.len() {
            model_expected.block.layers[i].weights =
                Param::from_tensor(Tensor::ones_like(&model_expected.block.layers[i].weights));
        }

        model_expected.block.layer_out.weights =
            Param::from_tensor(Tensor::ones_like(&model_expected.block.layer_out.weights));

        let mut model_updated = Model::<B, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            initial_reserves,
            &HashSet::new(),
        )
        .expect("base model init failed")
        .update(updated_reserves)
        .expect("model update failed");

        model_updated.block.layer_in.weights =
            Param::from_tensor(Tensor::ones_like(&model_updated.block.layer_in.weights));

        for i in 0..model_updated.block.layers.len() {
            model_updated.block.layers[i].weights =
                Param::from_tensor(Tensor::ones_like(&model_updated.block.layers[i].weights));
        }

        model_updated.block.layer_out.weights =
            Param::from_tensor(Tensor::ones_like(&model_updated.block.layer_out.weights));

        // Relative, not absolute: the fresh-`init` and `init().update()` paths build their
        // tensors in different orders, so f32 reductions land a few ULP apart. At this output
        // magnitude (~1e3) a single ULP is already ~1e-4 absolute, so an absolute 1e-6 bound is
        // unsatisfiable for any f32 result here. A 1e-4 relative bound absorbs the rounding noise
        // while still catching a genuinely divergent update.
        let relative_tolerance = 1e-4;

        let input_amount = 1000.0;

        let output_amount_expected = model_expected.evaluate(input_amount);
        let output_amount_updated = model_updated.evaluate(input_amount);
        let diff = output_amount_expected - output_amount_updated;

        println!("input amount: {}", input_amount);
        println!("output expected: {}", output_amount_expected);
        println!("output updated: {}", output_amount_updated);

        assert!(
            diff.abs() <= output_amount_expected.abs() * relative_tolerance,
            "Output amount difference is too large. Expected: {}, Actual: {}",
            output_amount_expected,
            output_amount_updated
        );
        // for (token_in, col_upd) in model_updated.layout.input_indexes {
        //     let col_exp = *model_expected
        //         .layout
        //         .input_indexes
        //         .get(&token_in)
        //         .expect("Didn't find token_in in expected model");

        //     for (token_out, out_set_upd) in model_updated.layout.output_indexes.iter() {
        //         let out_set_exp = model_expected
        //             .layout
        //             .output_indexes
        //             .get(&token_out)
        //             .expect("Didn't find token_out in expected model");

        //         for val in out_set_upd
        //             .iter()
        //             .map(|&row_upd| access_reserve!(model_updated.layout, row_upd, col_upd))
        //         {

        //         }
        //     }
        // }

        // let reserves_in_diff = (model_expected.block.reserves_in - model_updated.block.reserves_in)
        //     .abs()
        //     .into_data()
        //     .into_vec::<f32>()
        //     .expect("failed to convert reserves_in_diff into data");

        // for (i, diff) in reserves_in_diff.iter().enumerate() {
        //     assert!(*diff < tolerance, "reserves_in mismatch at index {}", i);
        // }

        // assert_eq!(
        //     model_expected.block.reserves_out.to_data(),
        //     model_updated.block.reserves_out.to_data(),
        //     "reserves_out mismatch"
        // );

        // assert_eq!(
        //     model_expected.block.max_swap.to_data(),
        //     model_updated.block.max_swap.to_data(),
        //     "max_swap mismatch"
        // );

        // assert_eq!(
        //     model_expected.block.bypass_mask.to_data(),
        //     model_updated.block.bypass_mask.to_data(),
        //     "bypass_mask mismatch"
        // );

        // assert_eq!(
        //     model_expected.block.fee_multiplier.to_data(),
        //     model_updated.block.fee_multiplier.to_data(),
        //     "fee_multiplier mismatch"
        // );

        // assert_eq!(
        //     model_expected.block.output_asset_indexes.to_data(),
        //     model_updated.block.output_asset_indexes.to_data(),
        //     "output_asset_indexes mismatch"
        // );

        // assert_eq!(
        //     model_expected.block.layer_out_bypass_mask.to_data(),
        //     model_updated.block.layer_out_bypass_mask.to_data(),
        //     "layer_out_bypass_mask mismatch"
        // );

        // assert_eq!(
        //     model_expected
        //         .block
        //         .layer_out_output_asset_indexes
        //         .to_data(),
        //     model_updated.block.layer_out_output_asset_indexes.to_data(),
        //     "layer_out_output_asset_indexes mismatch"
        // );

        // assert_eq!(
        //     model_expected.block.layer_out_pool_indexes.to_data(),
        //     model_updated.block.layer_out_pool_indexes.to_data(),
        //     "layer_out_pool_indexes mismatch"
        // );
    }

    #[test]
    fn test_model_v4_update() {
        run_on_available_backend(
            || test_model_v4_update_on::<WgpuBackend>(),
            || test_model_v4_update_on::<CpuBackend>(),
        );
    }

    proptest! {

        #[test]
        fn test_pools_fold_state_all_inputs_assigned(
            pool_values in prop::collection::hash_map(any::<i64>(),prop::collection::hash_set(any::<TokenAddress>(), 2), 1..1000)
        ) {
            let inputs = pool_values.iter()
                .map(|(value, tokens)| {
                    let tokens = tokens.into_iter().collect::<Vec<_>>();
                    (*tokens[0], *tokens[1], *value)
                })
                .collect::<Vec<_>>();

            let state = inputs.iter().cloned()
                .fold(ModelLayout::new(), |state, (from, to, value)| {
                    state.with_reserve_values(from, to, value)
                });

            let recorded_pool_values = state.rows.iter()
                .flat_map(|row| row.iter())
                .filter_map(|&v| v)
                .collect::<Vec<_>>();

            prop_assert_eq!(
                recorded_pool_values.len(),
                inputs.len(),
                "Recorded pool values length does not match input pool values length"
            );

            prop_assert!(
                recorded_pool_values.iter().all(|v| pool_values.contains_key(v)),
                "Recorded pool values do not match input pool values"
            )
        }

        #[test]
        fn test_pools_fold_state_into_data_returns_correct_size(
            pool_values in prop::collection::hash_map(any::<i64>(),prop::collection::hash_set(any::<TokenAddress>(), 2), 1..1000)
        ) {
            let inputs = pool_values.iter()
                .map(|(value, tokens)| {
                    let tokens = tokens.into_iter().collect::<Vec<_>>();
                    (*tokens[0], *tokens[1], *value)
                })
                .collect::<Vec<_>>();

            let state = inputs.iter().cloned()
                .fold(ModelLayout::new(), |state, (from, to, value)| {
                    state.with_reserve_values(from, to, value)
                });

            let inputs_size = state.input_indexes.len();
            let outputs_size = state.output_indexes.iter().map(|(_, row_index_set)| row_index_set.len()).sum::<usize>();
            // let data = state.into_data::<i64>(|v| v);
            let data = state.rows.iter().flat_map(|row| row.iter()).collect::<Vec<_>>();

            prop_assert_eq!(
                data.len(),
                inputs_size * outputs_size,
                "Data length does not match expected size: {} * {} = {}",
                inputs_size,
                outputs_size,
                data.len()
            );
        }

        #[test]
        fn test_pools_fold_state_inputs(
            pool_values in prop::collection::hash_map(any::<i64>(),prop::collection::hash_set(any::<TokenAddress>(), 2), 1..1000)
        ) {
            let inputs = pool_values.iter()
                .map(|(value, tokens)| {
                    let tokens = tokens.into_iter().collect::<Vec<_>>();
                    (*tokens[0], *tokens[1], (*tokens[0], *tokens[1], *value))
                })
                .collect::<Vec<_>>();

            let state = inputs.iter().cloned()
                .fold(ModelLayout::new(), |state, (from, to, value)| {
                    state.with_reserve_values(from, to, value)
                });

            let tokens = state.inputs();

            prop_assert_eq!(tokens.len(), state.input_indexes.len());

            for (index,token) in tokens.iter().enumerate() {
                prop_assert_eq!(index, state.input_indexes.get(&token).unwrap().0);
            }

            for row in state.rows {
                for (token, value) in tokens.iter().zip(row.iter()) {
                    if let Some((input_token, _, _)) = value {
                        prop_assert_eq!(input_token, token, "Input token does not match");
                    }
                }
            }
        }

        #[test]
        fn test_pools_fold_state_outputs(   pool_values in prop::collection::hash_map(any::<i64>(),prop::collection::hash_set(any::<TokenAddress>(), 2), 1..1000)
        ) {
            let inputs = pool_values.iter()
                .map(|(value, tokens)| {
                    let tokens = tokens.into_iter().collect::<Vec<_>>();
                    (*tokens[0], *tokens[1], (*tokens[0], *tokens[1], *value))
                })
                .collect::<Vec<_>>();

            let state = inputs.iter().cloned()
                .fold(ModelLayout::new(), |state, (from, to, value)| {
                    state.with_reserve_values(from, to, value)
                });

            let tokens = state.outputs();

            for (index, token) in tokens.iter().enumerate() {
                match state.output_indexes.get(&token) {
                    Some(indexes_set) => {
                        prop_assert!(indexes_set.contains(&RowIndex(index)), "Token index not found in output indexes");
                    }
                    None => {
                        prop_assert!(false, "Token not found in output indexes");
                    }
                }

            }
        }

        #[test]
        fn test_pools_fold_state_table_invariants(
            pool_values in prop::collection::hash_map(any::<i64>(),prop::collection::hash_set(any::<TokenAddress>(), 2), 1..1000)
        ) {
            let state = pool_values.iter()
                .map(|(value, tokens)| {
                    let tokens = tokens.into_iter().collect::<Vec<_>>();
                    (*tokens[0], *tokens[1], (*tokens[0], *tokens[1], *value))
                })
                .fold(ModelLayout::new(), |state, (from, to, value)| {
                    state.with_reserve_values(from, to, value)
                });

            let outputs = state.outputs();
            prop_assert_eq!(state.rows.len(), outputs.len(), "Number of rows does not match number of outputs");

            // all rows have the same output token
            for (row, expected_output_token) in state.rows.iter().zip(outputs.iter()) {
                for value in row.iter() {
                    if let Some((_, output_token, _)) = value {
                        prop_assert_eq!(*expected_output_token, *output_token, "Output token does not match");
                    }
                }
            }
        }

        #[test]
        fn test_pools_fold_state_bypassed_tokens_match(
            pool_values in prop::collection::hash_map(any::<i64>(),prop::collection::hash_set(any::<TokenAddress>(), 2), 1..1000)
        ) {
            let state = pool_values.iter()
                .map(|(value, tokens)| {
                    let tokens = tokens.into_iter().collect::<Vec<_>>();
                    (*tokens[0], *tokens[1], (*tokens[0], *tokens[1], *value))
                })
                .fold(ModelLayout::new(), |state, (from, to, value)| {
                    state.with_reserve_values(from, to, value)
                });

            let inputs = state.inputs();
            let outputs = state.outputs();

            for (RowIndex(row), ColumnIndex(col)) in state.bypass_indexes(&HashSet::new()).into_iter() {
                prop_assert_eq!(inputs[col], outputs[row], "Bypassed input token does not match output token");
            }
        }


    }
}
