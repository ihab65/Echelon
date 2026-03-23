//! # solvers
//!
//! Sparse direct solvers for symmetric positive definite systems `Ku = f`.
//!
//!
//! ## Dependency direction
//!
//! `solvers` depends on `sparse`.  `sparse` has no knowledge of `solvers`.
//! This is enforced by keeping them as separate crates in the workspace.

pub mod error;
pub mod ordering;
pub mod cholesky;

pub use error::{SolverError, Result};