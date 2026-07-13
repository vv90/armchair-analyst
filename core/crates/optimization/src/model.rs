use std::collections::{HashMap, HashSet};

use burn::{
    module::Param,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::{Distribution, Slice, activation::softmax, backend::AutodiffBackend},
};
use crate::OptimizationError;
use crate::pool_reserves::{PoolReserves, VirtualReserveValues};
use petgraph::prelude::*;

/// Initial pre-softmax weight for a newly grown-in pool's cells. Moderately negative so the pool's
/// initial routing share is negligible (existing routing is preserved the instant it is added) yet
/// the softmax gradient stays large enough for the optimizer to discover the pool over subsequent
/// chunks — unlike the `-1e9` disable fill, whose gradient vanishes by design.
const COLD_WEIGHT: f32 = -8.0;

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
            })
            .flat_map(|(column_index, set)| set.iter().map(|row_index| (*row_index, *column_index)))
            .chain(
                bridges
                    .into_iter()
                    .flat_map(|(token0, token1)| self.get_indexes(token0, token1)),
            )
            .collect()
    }

    fn output_token_indexes(&self) -> Result<Vec<usize>, OptimizationError> {
        let mut indexes = vec![0; self.rows.len()];

        for (token_address, RowIndex(idx)) in self
            .output_indexes
            .iter()
            .flat_map(|(token_address, idxs)| idxs.iter().map(move |idx| (token_address, idx)))
        {
            // Invariant-backed: `with_reserve_values` indexes every output token as an input
            // column too, so a miss here means a corrupted layout.
            let ColumnIndex(column) = self
                .input_indexes
                .get(token_address)
                .ok_or(OptimizationError::InvalidLayoutIndex)?;
            indexes[*idx] = *column;
        }

        Ok(indexes)
    }

    fn inputs(&self) -> Result<Vec<I>, OptimizationError> {
        let mut tokens = vec![None; self.input_indexes.len()];
        self.input_indexes
            .iter()
            .for_each(|(token_address, ColumnIndex(idx))| {
                tokens[*idx] = Some(*token_address);
            });

        tokens
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(OptimizationError::InvalidLayoutIndex)
    }

    fn outputs(&self) -> Result<Vec<I>, OptimizationError> {
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
        tokens
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(OptimizationError::InvalidLayoutIndex)
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

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input: Tensor<B, 1>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        gamma: &Tensor<B, 2>,
        bypass_mask: &Tensor<B, 2, Bool>,
        disabled_mask: &Tensor<B, 2, Bool>,
        max_swap: &Tensor<B, 2>,
        token_indexes: &Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let dims = self.weights.shape();
        let out_dim = dims.dims[0];
        // Disabled cells are pushed to a large negative weight *before* softmax: their
        // post-softmax share collapses to ~0 (flow renormalizes onto healthy pools) and, because
        // `mask_fill` yields a zero gradient at filled positions, their trained weights are frozen
        // rather than trained away. A finite fill (not -inf) keeps softmax numerically defined even
        // when every cell in a column is disabled (uniform split, no NaN).
        let masked_weights = self.weights.val().mask_fill(disabled_mask.clone(), -1e9);
        let w_normalized = softmax(masked_weights, 0);

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
    disabled_mask: Tensor<B, 2, Bool>,
    max_swap: Tensor<B, 2>,
    init_asset_index: i64,
    output_asset_indexes: Tensor<B, 1, Int>,
    layer_out_output_asset_indexes: Tensor<B, 1, Int>,
    layer_out_bypass_mask: Tensor<B, 2, Bool>,
    layer_out_disabled_mask: Tensor<B, 2, Bool>,
    layer_out_pool_indexes: Tensor<B, 1, Int>,
    layer_in: Layer<B>,
    layers: [Layer<B>; LAYERS],
    layer_out: Layer<B>,
}

impl<B: Backend, const LAYERS: usize> LayerBlock<B, LAYERS> {
    #[allow(clippy::type_complexity)]
    fn layer_in_params(
        &self,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2, Bool>,
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
            self.disabled_mask.clone().slice(input_asset_range),
            self.max_swap.clone().slice(input_asset_range),
        )
    }

    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let (x_in, y_in, gamma_in, bypass_mask_in, disabled_mask_in, max_swap_in) =
            self.layer_in_params();

        let layer_in_output = self.layer_in.forward(
            input.clone(),
            &x_in,
            &y_in,
            &gamma_in,
            &bypass_mask_in,
            &disabled_mask_in,
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
                &self.disabled_mask,
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
            &self.layer_out_disabled_mask,
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
    fee_multiplier_data: Vec<B::FloatElem>,
    max_swap_data: Vec<B::FloatElem>,
    block: LayerBlock<B, LAYERS>,
}

pub struct ModelOptimizer<B: AutodiffBackend, const LAYERS: usize> {
    optimizer: OptimizerAdaptor<Adam, LayerBlock<B, LAYERS>, B>,
}

/// Whether a [`Model::reconcile`] changed tensor shapes. `Grew` means new pool rows/columns were
/// appended (existing weights preserved, new cells cold-started) and every weight tensor has a fresh
/// `ParamId`, so the caller must reset the optimizer. `Refreshed` means values and the mask were
/// updated in place, so the optimizer state is still valid. Making this explicit keeps a growth from
/// being silently treated as an in-place refresh (which would desync the optimizer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconcileOutcome {
    Grew,
    Refreshed,
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

impl<
    B: Backend,
    U: Copy + Eq + std::hash::Hash,
    I: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    const LAYERS: usize,
> Model<B, U, I, LAYERS>
{
    /// Builds the model tensors from a reserve snapshot. `disabled` holds pool ids that should be
    /// masked out of the softmax routing (see [`Model::update`]); pass an empty set for the
    /// all-pools-active case. Ids in `disabled` that no reserve carries are simply ignored.
    pub fn init(
        init_asset: I,
        pool_reserves: Vec<PoolReserves<U, I>>,
        bridges: &HashSet<(I, I)>,
        disabled: &HashSet<U>,
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
            .ok_or(OptimizationError::InitAssetOutputNotFound)?;
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

        let asset_indexes_data = model_layout.output_token_indexes()?;

        let (
            reserves_in_data,
            reserves_out_data,
            fee_multiplier_data,
            max_swap_data,
            bypass_mask_data,
            disabled_mask_data,
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
                    Vec::with_capacity(buf_size),
                ),
                |(
                    mut reserves_in_data,
                    mut reserves_out_data,
                    mut fee_multiplier_data,
                    mut max_swap_data,
                    mut bypass_mask_data,
                    mut disabled_mask_data,
                ): (
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<B::FloatElem>,
                    Vec<bool>,
                    Vec<bool>,
                ),
                 (row_index, column_index, cell): (
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
                    Vec<bool>,
                ) {
                    disabled_mask_data.push(
                        cell.as_ref()
                            .map(|(pool_id, _)| disabled.contains(pool_id))
                            .unwrap_or(false),
                    );

                    let reserve =
                        cell.as_ref().map(|(_, r)| r).unwrap_or(&VirtualReserveValues {
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

                    bypass_mask_data.push(bypass_indexes.contains(&(row_index, column_index)));

                    (
                        reserves_in_data,
                        reserves_out_data,
                        fee_multiplier_data,
                        max_swap_data,
                        bypass_mask_data,
                        disabled_mask_data,
                    )
                },
            );
        let bypass_mask =
            Tensor::<B, 1, Bool>::from_data(bypass_mask_data.as_slice(), device).reshape(dims);
        let disabled_mask =
            Tensor::<B, 1, Bool>::from_data(disabled_mask_data.as_slice(), device).reshape(dims);

        let layer_out_pool_indexes =
            Tensor::<B, 1, Int>::from_data(init_asset_pool_indexes.as_slice(), device);

        let layer_out_bypass_mask = bypass_mask
            .clone()
            .int()
            .select(0, layer_out_pool_indexes.clone())
            .bool();

        let layer_out_disabled_mask = disabled_mask
            .clone()
            .int()
            .select(0, layer_out_pool_indexes.clone())
            .bool();

        Ok(Model {
            block: LayerBlock {
                reserves_in: Tensor::<B, 1>::from_data(reserves_in_data.as_slice(), device)
                    .reshape(dims),
                reserves_out: Tensor::<B, 1>::from_data(reserves_out_data.as_slice(), device)
                    .reshape(dims),
                fee_multiplier: Tensor::<B, 1>::from_data(fee_multiplier_data.as_slice(), device)
                    .reshape(dims),
                max_swap: Tensor::<B, 1>::from_data(max_swap_data.as_slice(), device).reshape(dims),
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
                disabled_mask,
                layer_out_bypass_mask,
                layer_out_disabled_mask,
                layer_out_pool_indexes,
                layer_in: Layer::init([size_pools, 1], size_tokens, device),
                layers: [(); LAYERS]
                    .map(|_| Layer::init([size_pools, size_tokens], size_tokens, device)),
                layer_out: Layer::init([init_asset_pools_count, size_tokens], 1, device),
            },
            reserves_in_data,
            reserves_out_data,
            fee_multiplier_data,
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

    pub fn evaluate(&self, input: B::FloatElem) -> B::FloatElem {
        let input_tensor: Tensor<B, 1> = Tensor::from([input]);
        self.block.forward(input_tensor).into_scalar()
    }

    /// Number of pool output-slots (layout rows) the model currently holds. Under grow-only this is
    /// monotonic non-decreasing and includes masked (absent/disabled) slots — a run diagnostic.
    pub fn pool_slots(&self) -> usize {
        let [rows, _] = self.layout.shape();
        rows
    }

    // Not wired into the runner yet: this is the route-extraction surface for turning trained
    // weights into an executable swap path. Exercised by tests until the execution layer
    // exposes it.
    #[allow(dead_code)]
    pub fn swaps_graph<L1: Default, F: Fn(&U) -> L1>(
        &self,
        layout_value_map: F,
        min_weight: f32,
    ) -> Result<Graph<I, (f32, L1), Directed>, OptimizationError> {
        let graph: Graph<I, (f32, L1), Directed> = Graph::new();

        let inputs = self.layout.inputs()?;
        let outputs = self.layout.outputs()?;

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

/// Grows a trained weight matrix to `[new_rows, new_cols]`, preserving the old `[old_rows, old_cols]`
/// block in the top-left and cold-filling the appended rows/columns with [`COLD_WEIGHT`]. Correct
/// only because the layout fold is append-only, so old cells never move off the top-left block.
fn cold_grow_2d<B: Backend>(
    old: Tensor<B, 2>,
    new_rows: usize,
    new_cols: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [old_rows, old_cols] = old.dims();
    let widened = if new_cols > old_cols {
        let right = Tensor::<B, 2>::full([old_rows, new_cols - old_cols], COLD_WEIGHT, device);
        Tensor::cat(vec![old, right], 1)
    } else {
        old
    };
    if new_rows > old_rows {
        let bottom = Tensor::<B, 2>::full([new_rows - old_rows, new_cols], COLD_WEIGHT, device);
        Tensor::cat(vec![widened, bottom], 0)
    } else {
        widened
    }
}

impl<
    B: Backend,
    U: Copy + PartialEq + Eq + std::hash::Hash,
    I: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    const LAYERS: usize,
> Model<B, U, I, LAYERS>
{
    /// Reconciles the model to `reserves` without ever reinitializing (grow-only):
    ///
    /// - pools already in the layout are value-refreshed in place;
    /// - pools not yet in the layout are appended (rows/columns grow), their trained-neighbour
    ///   weights preserved and their own cells cold-started so current routing is essentially
    ///   unchanged the instant they appear;
    /// - pools in the layout but absent from `reserves` are masked out of routing (their slot and
    ///   trained weights retained for a cheap later re-add), together with the externally `disabled`
    ///   pools. Ids in `disabled` this model does not carry are ignored.
    ///
    /// Returns whether the layout grew ([`ReconcileOutcome::Grew`]) so the caller can reset the
    /// optimizer only when the weight tensors were rebuilt. `bridges` is needed only on growth, to
    /// recompute the bypass mask over any new cells.
    pub fn reconcile(
        self,
        reserves: Vec<PoolReserves<U, I>>,
        bridges: &HashSet<(I, I)>,
        disabled: &HashSet<U>,
    ) -> Result<(Self, ReconcileOutcome), OptimizationError> {
        let present: HashSet<U> = reserves.iter().map(|reserve| reserve.pool_id).collect();
        let new_reserves: Vec<PoolReserves<U, I>> = reserves
            .iter()
            .filter(|reserve| {
                self.layout
                    .find_reserve_position(&reserve.token0, &reserve.token1, &reserve.pool_id)
                    .is_none()
            })
            .copied()
            .collect();

        if new_reserves.is_empty() {
            let model = self.refresh(&reserves, &present, disabled)?;
            Ok((model, ReconcileOutcome::Refreshed))
        } else {
            let model = self.grow(reserves, new_reserves, bridges, &present, disabled)?;
            Ok((model, ReconcileOutcome::Grew))
        }
    }

    /// In-place path: no new pools, so tensor shapes are unchanged. Refreshes present pools' values
    /// via `inplace` (preserving `ParamId`) and rebuilds the softmax mask as `disabled ∪ absent`.
    fn refresh(
        mut self,
        reserves: &[PoolReserves<U, I>],
        present: &HashSet<U>,
        disabled: &HashSet<U>,
    ) -> Result<Self, OptimizationError> {
        let [_, m] = self.layout.shape();

        for reserve in reserves {
            // Present by construction (refresh only runs when no reserve is new), but stay
            // panic-free: a miss means a corrupted layout, not a normal not-found.
            let Some((RowIndex(row), ColumnIndex(col))) = self.layout.find_reserve_position(
                &reserve.token0,
                &reserve.token1,
                &reserve.pool_id,
            ) else {
                return Err(OptimizationError::InvalidLayoutIndex);
            };

            let index = row * m + col;
            self.reserves_in_data[index] = B::FloatElem::from_elem(reserve.value.token_0);
            self.reserves_out_data[index] = B::FloatElem::from_elem(reserve.value.token_1);
            self.fee_multiplier_data[index] = B::FloatElem::from_elem(reserve.value.fee_multiplier);
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
        self.block.fee_multiplier.inplace(|fee| {
            Tensor::<B, 1>::from_data(self.fee_multiplier_data.as_slice(), &fee.device())
                .reshape(fee.shape())
        });
        self.block.max_swap.inplace(|max_swap| {
            Tensor::<B, 1>::from_data(self.max_swap_data.as_slice(), &max_swap.device())
                .reshape(max_swap.shape())
        });

        let disabled_mask_data = self
            .layout
            .rows
            .iter()
            .flat_map(|row| row.iter())
            .map(|cell| is_masked(cell.as_ref(), present, disabled))
            .collect::<Vec<_>>();
        let device = self.block.disabled_mask.device();
        let disabled_mask = Tensor::<B, 1, Bool>::from_data(disabled_mask_data.as_slice(), &device)
            .reshape(self.block.disabled_mask.shape());
        self.block.layer_out_disabled_mask = disabled_mask
            .clone()
            .int()
            .select(0, self.block.layer_out_pool_indexes.clone())
            .bool();
        self.block.disabled_mask = disabled_mask;

        Ok(self)
    }

    /// Growth path: append `new_reserves` to the layout and rebuild the block at the larger shape,
    /// preserving trained weights (top-left block) and cold-starting the appended cells.
    fn grow(
        self,
        all_reserves: Vec<PoolReserves<U, I>>,
        new_reserves: Vec<PoolReserves<U, I>>,
        bridges: &HashSet<(I, I)>,
        present: &HashSet<U>,
        disabled: &HashSet<U>,
    ) -> Result<Self, OptimizationError> {
        let device = &B::Device::default();

        // Snapshot the trained weights and the init-asset output-row order before the layout moves.
        let old_layer_in_w = self.block.layer_in.weights.val();
        let old_layers_w: Vec<Tensor<B, 2>> = self
            .block
            .layers
            .iter()
            .map(|layer| layer.weights.val())
            .collect();
        let old_layer_out_w = self.block.layer_out.weights.val();
        let init_asset_index = self.block.init_asset_index;
        let old_layer_out_pool_indexes: Vec<usize> = self
            .block
            .layer_out_pool_indexes
            .to_data()
            .iter::<i64>()
            .map(|idx| idx as usize)
            .collect();

        // Grow the layout: existing (row, col) slots never move (append-only fold).
        let layout = new_reserves.into_iter().fold(self.layout, |layout, reserve| {
            layout.with_reserve_values(reserve.token0, reserve.token1, reserve.pool_id)
        });
        let [rows, cols] = layout.shape();
        let dims = [rows, cols];

        // Fresh value for every current reserve, keyed by its slot in the grown layout.
        let mut position_value: HashMap<(usize, usize), VirtualReserveValues> = HashMap::new();
        for reserve in &all_reserves {
            if let Some((RowIndex(row), ColumnIndex(col))) =
                layout.find_reserve_position(&reserve.token0, &reserve.token1, &reserve.pool_id)
            {
                position_value.insert((row, col), reserve.value);
            }
        }

        let bypass_indexes = layout.bypass_indexes(bridges);
        let outputs = layout.outputs()?;

        // Build the per-cell value + mask vectors row-major, mirroring `init`'s packing.
        let buf_size = rows * cols;
        let mut reserves_in_data = Vec::with_capacity(buf_size);
        let mut reserves_out_data = Vec::with_capacity(buf_size);
        let mut fee_multiplier_data = Vec::with_capacity(buf_size);
        let mut max_swap_data = Vec::with_capacity(buf_size);
        let mut bypass_mask_data = Vec::with_capacity(buf_size);
        let mut disabled_mask_data = Vec::with_capacity(buf_size);

        for (row_index, row) in layout.rows.iter().enumerate() {
            for (col_index, cell) in row.iter().enumerate() {
                disabled_mask_data.push(is_masked(cell.as_ref(), present, disabled));
                bypass_mask_data
                    .push(bypass_indexes.contains(&(RowIndex(row_index), ColumnIndex(col_index))));

                let value = position_value.get(&(row_index, col_index)).copied().unwrap_or(
                    VirtualReserveValues {
                        token_0: 0.0,
                        token_1: 0.0,
                        fee_multiplier: 0.0,
                        max_swap_0: 0.0,
                        max_swap_1: 0.0,
                    },
                );
                reserves_in_data.push(B::FloatElem::from_elem(value.token_0));
                reserves_out_data.push(B::FloatElem::from_elem(value.token_1));
                fee_multiplier_data.push(B::FloatElem::from_elem(value.fee_multiplier));
                max_swap_data.push(B::FloatElem::from_elem(value.max_swap_0));
            }
        }

        let reserves_in =
            Tensor::<B, 1>::from_data(reserves_in_data.as_slice(), device).reshape(dims);
        let reserves_out =
            Tensor::<B, 1>::from_data(reserves_out_data.as_slice(), device).reshape(dims);
        let fee_multiplier =
            Tensor::<B, 1>::from_data(fee_multiplier_data.as_slice(), device).reshape(dims);
        let max_swap = Tensor::<B, 1>::from_data(max_swap_data.as_slice(), device).reshape(dims);
        let bypass_mask =
            Tensor::<B, 1, Bool>::from_data(bypass_mask_data.as_slice(), device).reshape(dims);
        let disabled_mask =
            Tensor::<B, 1, Bool>::from_data(disabled_mask_data.as_slice(), device).reshape(dims);

        // layer_out rows: preserved trained order, then new init-asset-output rows appended, so the
        // grown layer_out weight rows stay aligned with their pools.
        let init_asset = *layout
            .inputs()?
            .get(init_asset_index as usize)
            .ok_or(OptimizationError::InvalidInitAssetIndex)?;
        let old_index_set: HashSet<usize> = old_layer_out_pool_indexes.iter().copied().collect();
        let mut layer_out_pool_indexes_vec = old_layer_out_pool_indexes;
        for (row_index, output_token) in outputs.iter().enumerate() {
            if *output_token == init_asset && !old_index_set.contains(&row_index) {
                layer_out_pool_indexes_vec.push(row_index);
            }
        }
        let init_asset_pools_count = layer_out_pool_indexes_vec.len();
        let layer_out_pool_indexes = Tensor::<B, 1, Int>::from_data(
            layer_out_pool_indexes_vec
                .iter()
                .map(|idx| *idx as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        let asset_indexes_data = layout.output_token_indexes()?;
        let output_asset_indexes =
            Tensor::<B, 1, Int>::from_data(asset_indexes_data.as_slice(), device);

        let layer_out_bypass_mask = bypass_mask
            .clone()
            .int()
            .select(0, layer_out_pool_indexes.clone())
            .bool();
        let layer_out_disabled_mask = disabled_mask
            .clone()
            .int()
            .select(0, layer_out_pool_indexes.clone())
            .bool();

        // Preserve trained weights; cold-fill the appended rows/columns. `cat` produces a non-leaf
        // autodiff tensor, so `detach` back into a leaf before wrapping as a `Param` — the optimizer
        // requires parameter tensors to be graph leaves.
        let layer_in_w = cold_grow_2d(old_layer_in_w, rows, 1, device).detach();
        let layers: [Layer<B>; LAYERS] = std::array::from_fn(|index| {
            let grown = old_layers_w
                .get(index)
                .cloned()
                .map(|weights| cold_grow_2d(weights, rows, cols, device))
                .unwrap_or_else(|| Tensor::<B, 2>::full(dims, COLD_WEIGHT, device));
            Layer {
                weights: Param::from_tensor(grown.detach()),
                output_size: cols,
            }
        });
        let layer_out_w =
            cold_grow_2d(old_layer_out_w, init_asset_pools_count, cols, device).detach();

        Ok(Model {
            block: LayerBlock {
                reserves_in,
                reserves_out,
                fee_multiplier,
                max_swap,
                init_asset_index,
                output_asset_indexes,
                layer_out_output_asset_indexes: Tensor::<B, 1, Int>::full(
                    [init_asset_pools_count],
                    0,
                    device,
                ),
                bypass_mask,
                disabled_mask,
                layer_out_bypass_mask,
                layer_out_disabled_mask,
                layer_out_pool_indexes,
                layer_in: Layer {
                    weights: Param::from_tensor(layer_in_w),
                    output_size: cols,
                },
                layers,
                layer_out: Layer {
                    weights: Param::from_tensor(layer_out_w),
                    output_size: 1,
                },
            },
            reserves_in_data,
            reserves_out_data,
            fee_multiplier_data,
            max_swap_data,
            layout,
        })
    }
}

/// A layout cell is masked out of routing when its pool is externally `disabled` or absent from the
/// current snapshot (`present`). Empty cells are never masked.
fn is_masked<U: Copy + Eq + std::hash::Hash>(
    cell: Option<&U>,
    present: &HashSet<U>,
    disabled: &HashSet<U>,
) -> bool {
    cell.map(|pool_id| disabled.contains(pool_id) || !present.contains(pool_id))
        .unwrap_or(false)
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

    /// Replaces every layer's random weights with ones, making forward passes deterministic
    /// (softmax of a constant column is a uniform split).
    fn force_ones_weights<B, U, I, const LAYERS: usize>(model: &mut Model<B, U, I, LAYERS>)
    where
        B: Backend,
        U: Copy,
        I: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    {
        model.block.layer_in.weights =
            Param::from_tensor(Tensor::ones_like(&model.block.layer_in.weights));
        for layer in model.block.layers.iter_mut() {
            layer.weights = Param::from_tensor(Tensor::ones_like(&layer.weights));
        }
        model.block.layer_out.weights =
            Param::from_tensor(Tensor::ones_like(&model.block.layer_out.weights));
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
            &HashSet::new(),
        )
        .unwrap();

        let optimizer = Model::<CpuBackend, i32, TokenAddress, 1>::init_optimizer();
        let (model, optimizer) = model.optimize_with(optimizer, 100.0, 0);
        let (_model, _optimizer) = model.optimize_with(optimizer, 100.0, 0);
    }

    #[test]
    fn model_init_without_init_asset_output_returns_error() {
        // A single directional reserve: USDC is an input column but never an output row, so a
        // USDC-rooted model has no way back to the init asset. This must surface as a typed
        // error, not a panic.
        let reserve = PoolReserves {
            token0: tokens::USDC.address,
            token1: tokens::WETH.address,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 1_000.0,
                fee_multiplier: 0.997,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        };

        let result = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            vec![reserve],
            &HashSet::new(),
            &HashSet::new(),
        );

        match result {
            Ok(_) => panic!("init without init-asset output unexpectedly succeeded"),
            Err(error) => assert_eq!(error, OptimizationError::InitAssetOutputNotFound),
        }
    }

    #[test]
    fn model_update_refreshes_fee_multiplier() {
        let reserve = PoolReserves {
            token0: tokens::USDC.address,
            token1: tokens::WETH.address,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 1_000.0,
                fee_multiplier: 0.997,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        };
        let model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            vec![reserve, reserve.inverse()],
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("model init failed");

        let updated = PoolReserves {
            value: VirtualReserveValues {
                fee_multiplier: 0.5,
                ..reserve.value
            },
            ..reserve
        };
        let (model, outcome) = model
            .reconcile(vec![updated, updated.inverse()], &HashSet::new(), &HashSet::new())
            .expect("model reconcile failed");
        assert_eq!(
            outcome,
            ReconcileOutcome::Refreshed,
            "refreshing existing pools must not grow the layout"
        );

        let [_, columns] = model.layout.shape();
        let fee_cells = float_cells(&model.block.fee_multiplier);
        for directional in [updated, updated.inverse()] {
            let (RowIndex(row), ColumnIndex(col)) = model
                .layout
                .find_reserve_position(
                    &directional.token0,
                    &directional.token1,
                    &directional.pool_id,
                )
                .expect("reserve missing from layout");
            assert_eq!(
                fee_cells[row * columns + col],
                0.5,
                "fee_multiplier update must reach the tensor cell"
            );
        }
    }

    #[test]
    fn reconcile_with_a_new_pool_grows_the_layout() {
        // Under grow-only a reserve the model has never seen is not an error — it extends the
        // layout. Reconciling a snapshot that adds a second pool must report `Grew` and append rows.
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
        let new_reserve = PoolReserves {
            pool_id: 2,
            value: VirtualReserveValues {
                token_0: 2_000.0,
                token_1: 1_500.0,
                ..known_reserve.value
            },
            ..known_reserve
        };
        let model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            vec![known_reserve, known_reserve.inverse()],
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();
        let [rows_before, _] = model.layout.shape();

        let (model, outcome) = model
            .reconcile(
                vec![
                    known_reserve,
                    known_reserve.inverse(),
                    new_reserve,
                    new_reserve.inverse(),
                ],
                &HashSet::new(),
                &HashSet::new(),
            )
            .expect("model reconcile failed");

        assert_eq!(outcome, ReconcileOutcome::Grew);
        let [rows_after, _] = model.layout.shape();
        assert!(
            rows_after > rows_before,
            "adding a pool must append rows ({} -> {})",
            rows_before,
            rows_after
        );
        assert!(model.evaluate(100.0).is_finite());
    }

    /// Two parallel USDC<->WETH pools (both directions), so an init-asset input can cycle
    /// USDC -> WETH -> USDC through either.
    fn two_parallel_reserves() -> Vec<PoolReserves<i32, TokenAddress>> {
        let pool_1 = PoolReserves {
            token0: tokens::USDC.address,
            token1: tokens::WETH.address,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 1_000.0,
                fee_multiplier: 0.997,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        };
        let pool_2 = PoolReserves {
            pool_id: 2,
            value: VirtualReserveValues {
                token_0: 2_000.0,
                token_1: 1_500.0,
                ..pool_1.value
            },
            ..pool_1
        };
        vec![pool_1, pool_1.inverse(), pool_2, pool_2.inverse()]
    }

    /// [`two_parallel_reserves`] built into a model with weights forced to ones (deterministic
    /// 50/50 softmax split) and the given pools disabled.
    fn two_parallel_pool_model(
        disabled: &HashSet<i32>,
    ) -> Model<CpuBackend, i32, TokenAddress, 1> {
        let mut model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            two_parallel_reserves(),
            &HashSet::new(),
            disabled,
        )
        .expect("model init failed");
        force_ones_weights(&mut model);
        model
    }

    #[test]
    fn empty_disabled_set_leaves_routing_unchanged() {
        // Regression pin for workstream 3: an empty disable set must produce an all-false mask,
        // so `mask_fill` is a no-op and behavior is identical to a model with no disable support.
        let model = two_parallel_pool_model(&HashSet::new());
        let mask: Vec<i64> = model
            .block
            .disabled_mask
            .clone()
            .int()
            .into_data()
            .iter::<i64>()
            .collect();
        assert!(
            mask.iter().all(|&flag| flag == 0),
            "empty disabled set must yield an all-false mask"
        );
        assert!(model.evaluate(100.0).is_finite());
    }

    #[test]
    fn disabling_a_pool_changes_routing_and_reenabling_restores_it() {
        let baseline = two_parallel_pool_model(&HashSet::new()).evaluate(100.0);

        // Disable pool 2 through `update` with the key set unchanged; trained (ones) weights and
        // the pool's reserves stay in place, only its routing weight is masked.
        let (disabled_model, _) = two_parallel_pool_model(&HashSet::new())
            .reconcile(two_parallel_reserves(), &HashSet::new(), &HashSet::from([2]))
            .expect("reconcile failed");
        let disabled_output = disabled_model.evaluate(100.0);

        assert!(
            (disabled_output - baseline).abs() > 1e-3,
            "disabling a pool must change the routed output (baseline {}, disabled {})",
            baseline,
            disabled_output
        );

        // Re-enabling (same reserves, empty disable set) restores the exact baseline: the weights
        // were frozen, never trained away.
        let (reenabled_model, _) = disabled_model
            .reconcile(two_parallel_reserves(), &HashSet::new(), &HashSet::new())
            .expect("reconcile failed");
        let reenabled_output = reenabled_model.evaluate(100.0);

        assert!(
            (reenabled_output - baseline).abs() <= baseline.abs() * 1e-5,
            "re-enabling must restore the baseline output (baseline {}, re-enabled {})",
            baseline,
            reenabled_output
        );
    }

    /// A third USDC<->WETH pool, distinct from [`two_parallel_reserves`]'s pools 1 and 2 — the pool
    /// grown in by the growth tests.
    fn third_pool() -> PoolReserves<i32, TokenAddress> {
        PoolReserves {
            token0: tokens::USDC.address,
            token1: tokens::WETH.address,
            pool_id: 3,
            value: VirtualReserveValues {
                token_0: 3_000.0,
                token_1: 2_400.0,
                fee_multiplier: 0.997,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        }
    }

    /// [`two_parallel_reserves`] plus [`third_pool`] (both directions).
    fn three_parallel_reserves() -> Vec<PoolReserves<i32, TokenAddress>> {
        let mut reserves = two_parallel_reserves();
        reserves.push(third_pool());
        reserves.push(third_pool().inverse());
        reserves
    }

    #[test]
    fn growth_preserves_trained_weights_and_cold_starts_new_cells() {
        // Ones-weight 2-pool model, then grow in a third pool. By the append-only layout invariant
        // the old pools keep their (row, col) slots, so their trained weights (all 1.0) must survive
        // untouched in the top-left block and every appended cell must equal COLD_WEIGHT.
        let model = two_parallel_pool_model(&HashSet::new());
        let [old_rows, old_cols] = model.layout.shape();

        let (model, outcome) = model
            .reconcile(three_parallel_reserves(), &HashSet::new(), &HashSet::new())
            .expect("reconcile failed");
        assert_eq!(outcome, ReconcileOutcome::Grew);

        let [_, cols] = model.layout.shape();
        let layer = model.block.layers.first().expect("model has one middle layer");
        let weights = float_cells(&layer.weights.val());

        for (index, &weight) in weights.iter().enumerate() {
            let row = index / cols;
            let col = index % cols;
            if row < old_rows && col < old_cols {
                assert_eq!(weight, 1.0, "trained weight at ({}, {}) must be preserved", row, col);
            } else {
                assert_eq!(
                    weight, COLD_WEIGHT,
                    "appended cell at ({}, {}) must be cold-started",
                    row, col
                );
            }
        }
    }

    #[test]
    fn cold_start_growth_preserves_current_routing() {
        // The instant a pool is grown in, its cold weight gives it a negligible share, so the
        // routed output is essentially unchanged from before the growth.
        let baseline = two_parallel_pool_model(&HashSet::new()).evaluate(100.0);

        let (grown, outcome) = two_parallel_pool_model(&HashSet::new())
            .reconcile(three_parallel_reserves(), &HashSet::new(), &HashSet::new())
            .expect("reconcile failed");
        assert_eq!(outcome, ReconcileOutcome::Grew);
        let grown_output = grown.evaluate(100.0);

        assert!(
            (grown_output - baseline).abs() <= baseline.abs() * 1e-2,
            "cold-start growth must preserve routing (baseline {}, grown {})",
            baseline,
            grown_output
        );
    }

    #[test]
    fn a_cold_started_pool_is_trainable() {
        // Rebuts the vanishing-gradient trap: COLD_WEIGHT must be moderate enough that the optimizer
        // can still move a newly grown-in pool's weight. (A -1e9 fill would freeze it.)
        let (grown, _) = two_parallel_pool_model(&HashSet::new())
            .reconcile(three_parallel_reserves(), &HashSet::new(), &HashSet::new())
            .expect("reconcile failed");

        let (RowIndex(new_row), _) = grown
            .layout
            .find_reserve_position(&tokens::USDC.address, &tokens::WETH.address, &3)
            .expect("new pool missing from layout");
        // layer_in weights are [pools, 1]: the flat index is just the row.
        let weight_before = float_cells(&grown.block.layer_in.weights.val())
            .into_iter()
            .nth(new_row)
            .expect("new row missing from layer_in weights");

        let optimizer = Model::<CpuBackend, i32, TokenAddress, 1>::init_optimizer();
        let (trained, _) = grown.optimize_with(optimizer, 100.0, 200);
        let weight_after = float_cells(&trained.block.layer_in.weights.val())
            .into_iter()
            .nth(new_row)
            .expect("new row missing from layer_in weights");

        assert!(
            (weight_after - weight_before).abs() > 0.1,
            "a cold-started pool's weight must move under training ({} -> {})",
            weight_before,
            weight_after
        );
    }

    #[test]
    fn removing_a_pool_masks_it_and_re_adding_restores_output() {
        // Grow-only removal: dropping a pool from the snapshot masks it (routing changes, slot kept),
        // and re-adding it restores the exact prior output because its weights were frozen, not
        // retrained — the value analog of disable/re-enable but driven by snapshot membership.
        let baseline = two_parallel_pool_model(&HashSet::new()).evaluate(100.0);
        let slots = two_parallel_pool_model(&HashSet::new()).pool_slots();

        let pool_one_only: Vec<_> = two_parallel_reserves()
            .into_iter()
            .filter(|reserve| reserve.pool_id == 1)
            .collect();
        let (dropped, outcome) = two_parallel_pool_model(&HashSet::new())
            .reconcile(pool_one_only, &HashSet::new(), &HashSet::new())
            .expect("reconcile failed");
        assert_eq!(outcome, ReconcileOutcome::Refreshed);
        assert_eq!(dropped.pool_slots(), slots, "removal must not shrink the layout");

        let dropped_output = dropped.evaluate(100.0);
        assert!(
            (dropped_output - baseline).abs() > 1e-3,
            "masking a pool must change the routed output (baseline {}, dropped {})",
            baseline,
            dropped_output
        );

        let (readded, outcome) = dropped
            .reconcile(two_parallel_reserves(), &HashSet::new(), &HashSet::new())
            .expect("reconcile failed");
        assert_eq!(outcome, ReconcileOutcome::Refreshed);
        let readded_output = readded.evaluate(100.0);
        assert!(
            (readded_output - baseline).abs() <= baseline.abs() * 1e-5,
            "re-adding a pool must restore the baseline (baseline {}, re-added {})",
            baseline,
            readded_output
        );
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
        let inputs = state.inputs().unwrap();
        let outputs = state.outputs().unwrap();

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
            &HashSet::new(),
        )
        .unwrap();

        let input_amount = 1000.0;

        let optimizer = Model::<B, [u8; 20], TokenAddress, 1>::init_optimizer();
        let (model, _optimizer) = model.optimize_with(optimizer, input_amount, 100);

        let profit = model.evaluate(input_amount) - input_amount;

        // Not an exact-recovery oracle: the optimizer legitimately splits flow across pools,
        // hits `max_swap` caps, and burns some softmax mass in empty cells, so it captures a
        // fraction of the planted cycle, not all of it. Finding a real, substantial profit is
        // the behavior under test.
        assert!(
            arbitrage_amount > 0.0,
            "fixture must plant a profitable cycle, planted {}",
            arbitrage_amount
        );
        assert!(
            profit > 0.0 && profit >= arbitrage_amount * 0.5,
            "Planted arbitrage not found. Planted {}. Found {}",
            arbitrage_amount,
            profit
        );
    }

    #[test]
    fn test_model_v4_arbitrage() {
        run_on_available_backend(
            || test_model_v4_arbitrage_on::<WgpuBackend>(),
            || test_model_v4_arbitrage_on::<CpuBackend>(),
        );
    }

    #[test]
    fn swaps_graph_respects_min_weight_and_reaches_every_node() {
        let usdc_weth = PoolReserves {
            token0: tokens::USDC.address,
            token1: tokens::WETH.address,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 1_000.0,
                fee_multiplier: 0.997,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        };
        let weth_wbtc = PoolReserves {
            token0: tokens::WETH.address,
            token1: tokens::WBTC.address,
            pool_id: 2,
            ..usdc_weth
        };
        let mut model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            vec![
                usdc_weth,
                usdc_weth.inverse(),
                weth_wbtc,
                weth_wbtc.inverse(),
            ],
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("model init failed");
        force_ones_weights(&mut model);

        // With ones weights every softmax column is a uniform split, so every surviving edge
        // weight is a known 1/rows fraction — pick a threshold safely below it.
        let min_weight = 0.05;
        let graph = model
            .swaps_graph(|pool_id| *pool_id, min_weight)
            .expect("swaps_graph failed");

        assert!(graph.edge_count() > 0, "expected at least one swap edge");
        for (weight, _) in graph.edge_weights() {
            assert!(
                *weight >= min_weight,
                "edge weight {} below min_weight {}",
                weight,
                min_weight
            );
        }

        let mut bfs = Bfs::new(&graph, NodeIndex::new(0));
        let mut visited = 0;
        while bfs.next(&graph).is_some() {
            visited += 1;
        }
        assert_eq!(
            visited,
            graph.node_count(),
            "every node must be reachable from the init asset"
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
            .map(|((from, to, fee), reserve)| PoolReserves {
                token0: *from,
                token1: *to,
                pool_id: *fee,
                value: reserve.clone(),
            })
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
            final_reserves.clone(),
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("expected model init failed");
        force_ones_weights(&mut model_expected);

        // Grow-only reconcile masks any pool absent from the snapshot, so it receives the *full*
        // updated snapshot (`final_reserves` = the 4 changed pools plus the unchanged remainder),
        // not just the changed delta. Same key set as init → `Refreshed`, nothing masked.
        let (mut model_updated, update_outcome) = Model::<B, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            initial_reserves,
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("base model init failed")
        .reconcile(final_reserves, &HashSet::new(), &HashSet::new())
        .expect("model reconcile failed");
        assert_eq!(
            update_outcome,
            ReconcileOutcome::Refreshed,
            "same-key reconcile must refresh in place, not grow"
        );
        force_ones_weights(&mut model_updated);

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
    }

    #[test]
    fn test_model_v4_update() {
        run_on_available_backend(
            || test_model_v4_update_on::<WgpuBackend>(),
            || test_model_v4_update_on::<CpuBackend>(),
        );
    }

    /// Strategy: a set of pools with distinct ids, each between two distinct tokens, with
    /// positive finite reserve values — the raw material for `Model::init` proptests.
    fn arbitrary_pool_reserves()
    -> impl Strategy<Value = Vec<PoolReserves<i64, TokenAddress>>> {
        prop::collection::hash_map(
            any::<i64>(),
            (
                prop::collection::hash_set(any::<TokenAddress>(), 2),
                0.0f32..1e12,
                0.0f32..1e12,
                0.0f32..1.0,
                0.0f32..1e12,
                0.0f32..1e12,
            ),
            1..20,
        )
        .prop_map(|pools| {
            pools
                .into_iter()
                .map(
                    |(pool_id, (tokens, token_0, token_1, fee_multiplier, max_swap_0, max_swap_1))| {
                        let tokens = tokens.into_iter().collect::<Vec<_>>();
                        PoolReserves {
                            token0: tokens[0],
                            token1: tokens[1],
                            pool_id,
                            value: VirtualReserveValues {
                                token_0,
                                token_1,
                                fee_multiplier,
                                max_swap_0,
                                max_swap_1,
                            },
                        }
                    },
                )
                .collect()
        })
    }

    /// A deterministic, always-valid init asset for a generated reserve set: some token that at
    /// least one reserve outputs (`Model::init` requires the init asset to have output rows).
    fn init_asset_for(reserves: &[PoolReserves<i64, TokenAddress>]) -> TokenAddress {
        reserves
            .iter()
            .map(|reserve| reserve.token1)
            .min()
            .expect("reserve set is never empty")
    }

    fn float_cells<B: Backend>(tensor: &Tensor<B, 2>) -> Vec<f32> {
        tensor.clone().into_data().iter::<f32>().collect()
    }

    /// Straight-Rust mirror of the tensor math in [`Layer::forward`], cell by cell:
    /// `U = input[col] * softmax(W)[row][col]`, after-fee `U * gamma` capped by `max_swap`,
    /// `Z = Y * U / (X + U + ε)` clamped at zero, bypass cells substitute the raw (pre-fee,
    /// uncapped) routed amount, rows summed and scatter-accumulated into `token_indexes`.
    #[allow(clippy::too_many_arguments)]
    fn forward_reference(
        weights: &[f32],
        input: &[f32],
        x: &[f32],
        y: &[f32],
        gamma: &[f32],
        bypass: &[bool],
        disabled: &[bool],
        max_swap: &[f32],
        token_indexes: &[usize],
        rows: usize,
        columns: usize,
        output_size: usize,
    ) -> Vec<f32> {
        // Disabled cells are pushed to a large negative weight before the column softmax, matching
        // the `mask_fill` in `Layer::forward`.
        let filled = |index: usize| {
            if disabled[index] {
                -1e9
            } else {
                weights[index]
            }
        };

        let mut w_normalized = vec![0.0f32; rows * columns];
        for col in 0..columns {
            let max_weight = (0..rows)
                .map(|row| filled(row * columns + col))
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = (0..rows)
                .map(|row| (filled(row * columns + col) - max_weight).exp())
                .sum();
            for row in 0..rows {
                let index = row * columns + col;
                w_normalized[index] = (filled(index) - max_weight).exp() / denominator;
            }
        }

        let mut output = vec![0.0f32; output_size];
        for row in 0..rows {
            let mut row_sum = 0.0f32;
            for col in 0..columns {
                let index = row * columns + col;
                let routed = input[col] * w_normalized[index];
                let capped_after_fee = (routed * gamma[index]).min(max_swap[index]);
                let z = (y[index] * capped_after_fee
                    / (x[index] + capped_after_fee + f32::EPSILON))
                    .max(0.0);
                row_sum += if bypass[index] { routed } else { z };
            }
            output[token_indexes[row]] += row_sum;
        }

        output
    }

    fn bool_cells<B: Backend>(tensor: &Tensor<B, 2, Bool>) -> Vec<i64> {
        tensor.clone().int().into_data().iter::<i64>().collect()
    }

    /// Strategy: dimensions plus every per-cell tensor `Layer::forward` consumes, as flat
    /// row-major vectors.
    #[allow(clippy::type_complexity)]
    fn arbitrary_layer_forward_inputs() -> impl Strategy<
        Value = (
            usize,
            usize,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<bool>,
            Vec<bool>,
            Vec<f32>,
            Vec<usize>,
        ),
    > {
        (1usize..6, 1usize..5).prop_flat_map(|(rows, columns)| {
            let cells = rows * columns;
            (
                Just(rows),
                Just(columns),
                prop::collection::vec(-5.0f32..5.0, cells),
                prop::collection::vec(0.0f32..1e9, columns),
                prop::collection::vec(0.0f32..1e9, cells),
                prop::collection::vec(0.0f32..1e9, cells),
                prop::collection::vec(0.0f32..1.0, cells),
                prop::collection::vec(any::<bool>(), cells),
                prop::collection::vec(any::<bool>(), cells),
                prop::collection::vec(0.0f32..1e9, cells),
                prop::collection::vec(0..columns, rows),
            )
        })
    }

    proptest! {

        /// `Layer::forward` agrees with the scalar reference implementation — pins the swap
        /// math (softmax split, fee, `max_swap` cap, ε-guarded quote, bypass substitution,
        /// scatter-accumulate by output token).
        #[test]
        fn layer_forward_matches_scalar_reference(
            (rows, columns, weights, input, x, y, gamma, bypass, disabled, max_swap, token_indexes)
                in arbitrary_layer_forward_inputs()
        ) {
            type B = CpuBackend;
            let device = <B as Backend>::Device::default();
            let dims = [rows, columns];

            let to_matrix = |data: &[f32]| {
                Tensor::<B, 1>::from_data(data, &device).reshape(dims)
            };
            let to_bool_matrix = |data: &[bool]| {
                Tensor::<B, 1, Bool>::from_data(data, &device).reshape(dims)
            };
            let layer = Layer::<B> {
                weights: Param::from_tensor(to_matrix(&weights)),
                output_size: columns,
            };
            let token_index_data = token_indexes
                .iter()
                .map(|&index| index as i64)
                .collect::<Vec<_>>();
            let token_index_tensor =
                Tensor::<B, 1, Int>::from_data(token_index_data.as_slice(), &device);

            let output = layer
                .forward(
                    Tensor::<B, 1>::from_data(input.as_slice(), &device),
                    &to_matrix(&x),
                    &to_matrix(&y),
                    &to_matrix(&gamma),
                    &to_bool_matrix(&bypass),
                    &to_bool_matrix(&disabled),
                    &to_matrix(&max_swap),
                    &token_index_tensor,
                )
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>();

            let expected = forward_reference(
                &weights, &input, &x, &y, &gamma, &bypass, &disabled, &max_swap, &token_indexes,
                rows, columns, columns,
            );

            prop_assert_eq!(output.len(), expected.len());
            for (index, (&actual, &reference)) in output.iter().zip(expected.iter()).enumerate() {
                let tolerance = 1e-3 * actual.abs().max(reference.abs()) + 1e-3;
                prop_assert!(
                    (actual - reference).abs() <= tolerance,
                    "output[{}] diverged: tensor {} vs reference {}",
                    index,
                    actual,
                    reference
                );
            }
        }

        /// `evaluate` never produces a NaN/inf, for any reserve set and any subset of disabled
        /// pools — including disabling every pool and disabling ids absent from the snapshot. The
        /// large-negative softmax fill (not `-inf`) is what keeps fully-masked columns finite.
        #[test]
        fn model_evaluate_is_finite_under_arbitrary_disable(
            reserves in arbitrary_pool_reserves(),
            extra_disabled in prop::collection::hash_set(any::<i64>(), 0..8),
            disable_all in any::<bool>(),
        ) {
            let init_asset = init_asset_for(&reserves);
            let disabled = if disable_all {
                reserves.iter().map(|reserve| reserve.pool_id).collect::<HashSet<_>>()
            } else {
                extra_disabled
            };

            let model = Model::<CpuBackend, i64, TokenAddress, 1>::init(
                init_asset,
                reserves,
                &HashSet::new(),
                &disabled,
            )
            .expect("model init failed");

            prop_assert!(model.evaluate(100.0).is_finite());
        }

        /// A grow-only reconcile sequence — arbitrary subsets of a pool universe (adding pools grows
        /// the layout, dropping them masks the slot) plus arbitrary disable sets — always leaves the
        /// model finite and its slot count monotonic non-decreasing (never shrinks).
        #[test]
        fn reconcile_sequence_stays_finite_and_slots_never_shrink(
            reserves in arbitrary_pool_reserves(),
            steps in prop::collection::vec(
                (
                    prop::collection::vec(any::<bool>(), 1..24),
                    prop::collection::hash_set(any::<i64>(), 0..6),
                ),
                1..6,
            ),
        ) {
            let init_asset = init_asset_for(&reserves);
            // Seed from a single reserve that outputs the init asset, so `init` is always valid; the
            // rest of the universe is grown in across the sequence.
            let seed = reserves
                .iter()
                .find(|reserve| reserve.token1 == init_asset)
                .copied()
                .expect("init_asset_for guarantees an outputting reserve");

            let mut model = Model::<CpuBackend, i64, TokenAddress, 1>::init(
                init_asset,
                vec![seed],
                &HashSet::new(),
                &HashSet::new(),
            )
            .expect("model init failed");
            let mut previous_slots = model.pool_slots();

            for (keep_flags, disabled) in steps {
                let mut subset = vec![seed];
                subset.extend(
                    reserves
                        .iter()
                        .zip(keep_flags.iter())
                        .filter(|(_, keep)| **keep)
                        .map(|(reserve, _)| *reserve),
                );

                let (next, _) = model
                    .reconcile(subset, &HashSet::new(), &disabled)
                    .expect("reconcile failed");
                prop_assert!(next.evaluate(100.0).is_finite());
                prop_assert!(
                    next.pool_slots() >= previous_slots,
                    "slots shrank: {} -> {}",
                    previous_slots,
                    next.pool_slots()
                );
                previous_slots = next.pool_slots();
                model = next;
            }
        }

        /// `Model::init` packs the layout into the value tensors row-major: every reserve's
        /// values land exactly at its layout cell, every unclaimed cell is all-zero, and the
        /// bypass mask is true precisely on `bypass_indexes`.
        #[test]
        fn model_init_packs_reserves_into_tensor_cells(
            reserves in arbitrary_pool_reserves()
        ) {
            let init_asset = init_asset_for(&reserves);
            let bridges = HashSet::new();

            let model = Model::<CpuBackend, i64, TokenAddress, 1>::init(
                init_asset,
                reserves.clone(),
                &bridges,
                &HashSet::new(),
            )
            .expect("model init failed");

            let [rows, columns] = model.layout.shape();
            let reserves_in = float_cells(&model.block.reserves_in);
            let reserves_out = float_cells(&model.block.reserves_out);
            let fee_multiplier = float_cells(&model.block.fee_multiplier);
            let max_swap = float_cells(&model.block.max_swap);
            let bypass_mask = bool_cells(&model.block.bypass_mask);

            let mut claimed = vec![false; rows * columns];
            for reserve in &reserves {
                let (RowIndex(row), ColumnIndex(col)) = model
                    .layout
                    .find_reserve_position(&reserve.token0, &reserve.token1, &reserve.pool_id)
                    .expect("reserve missing from layout");
                let index = row * columns + col;

                prop_assert_eq!(reserves_in[index], reserve.value.token_0);
                prop_assert_eq!(reserves_out[index], reserve.value.token_1);
                prop_assert_eq!(fee_multiplier[index], reserve.value.fee_multiplier);
                prop_assert_eq!(max_swap[index], reserve.value.max_swap_0);
                claimed[index] = true;
            }

            let bypass_indexes = model.layout.bypass_indexes(&bridges);
            for row in 0..rows {
                for col in 0..columns {
                    let index = row * columns + col;
                    if !claimed[index] {
                        prop_assert_eq!(reserves_in[index], 0.0);
                        prop_assert_eq!(reserves_out[index], 0.0);
                        prop_assert_eq!(fee_multiplier[index], 0.0);
                        prop_assert_eq!(max_swap[index], 0.0);
                    }
                    prop_assert_eq!(
                        bypass_mask[index] != 0,
                        bypass_indexes.contains(&(RowIndex(row), ColumnIndex(col)))
                    );
                }
            }
        }

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

            let tokens = state.inputs().unwrap();

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

            let tokens = state.outputs().unwrap();

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

            let outputs = state.outputs().unwrap();
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

            let inputs = state.inputs().unwrap();
            let outputs = state.outputs().unwrap();

            for (RowIndex(row), ColumnIndex(col)) in state.bypass_indexes(&HashSet::new()).into_iter() {
                prop_assert_eq!(inputs[col], outputs[row], "Bypassed input token does not match output token");
            }
        }


    }
}
