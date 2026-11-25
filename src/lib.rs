//! rustling: an inference engine for Thinking Machines' Inkling 2 Mini.

mod accel;
pub mod config;
pub mod engine;
pub mod generate;
#[cfg(target_os = "macos")]
pub mod gpu;
pub mod model;
pub mod prompt;
pub mod timing;
pub mod trace;
pub mod weights;
