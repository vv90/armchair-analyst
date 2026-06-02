use std::collections::HashSet;

use alloy::primitives::{Address, B256};

struct State {
    block_hash: B256,
    observed_addresses: HashSet<Address>,
}

enum Event {}
