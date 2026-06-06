#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tick(u64);

const REQUEST_TTL: u64 = 10;

impl Tick {
    pub fn initial() -> Self {
        Tick(0)
    }
    pub fn next(self) -> Self {
        Tick(self.0.wrapping_add(1))
    }

    pub fn elapsed_since(self, earlier: Tick) -> u64 {
        self.0.wrapping_sub(earlier.0)
    }

    pub fn is_expired_since(self, dispatched_at: Tick) -> bool {
        self.elapsed_since(dispatched_at) >= REQUEST_TTL
    }
}

#[cfg(test)]
pub(crate) const REQUEST_TTL_FOR_TEST: u64 = REQUEST_TTL;

#[cfg(test)]
impl Tick {
    pub(crate) fn from_raw_for_test(value: u64) -> Self {
        Tick(value)
    }

    pub(crate) fn raw_for_test(self) -> u64 {
        self.0
    }
}
