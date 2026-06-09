pub trait Invertible {
    fn inverse(self) -> Self;
}

impl<T> Invertible for (T, T) {
    fn inverse(self) -> Self {
        (self.1, self.0)
    }
}
