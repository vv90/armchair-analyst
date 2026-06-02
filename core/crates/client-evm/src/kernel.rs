use std::collections::HashSet;

use alloy::primitives::{Address, U256};

struct State {
    block_hash: U256,
    observed_addresses: HashSet<Address>,
}

enum Event {}
