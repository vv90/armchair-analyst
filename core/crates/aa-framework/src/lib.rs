//! Domain-agnostic framework definitions for pure state machines.

pub mod framework;

pub use crate::framework::{Application, ApplicationError, Runtime, Transition};

/// Common framework imports for application implementations and runtimes.
pub mod prelude {
    pub use crate::{Application, ApplicationError, Runtime, Transition};
}
