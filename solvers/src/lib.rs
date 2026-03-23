//! # solvers
//!
//! Sparse direct solvers for symmetric positive definite systems `Ku = f`.
//!
//! ## Planned contents
//!
//! ```text
//! src/
//!   lib.rs            — this file
//!   error.rs          — SolverError (wraps SparseError)
//!   ordering/
//!     mod.rs
//!     rcm.rs          — Reverse Cuthill-McKee fill reduction
//!     amd.rs          — Approximate Minimum Degree (later)
//!   cholesky/
//!     mod.rs          — public SparseSolver interface
//!     symbolic.rs     — elimination tree, L sparsity pattern
//!     numeric.rs      — numeric Cholesky factorization
//!     solve.rs        — forward/backward substitution
//! ```
//!
//! ## Dependency direction
//!
//! `solvers` depends on `sparse`.  `sparse` has no knowledge of `solvers`.
//! This is enforced by keeping them as separate crates in the workspace.

pub mod error;
pub mod ordering;
pub mod cholesky;

pub use error::{SolverError, Result};