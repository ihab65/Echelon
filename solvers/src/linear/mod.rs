//! Linear direct solvers for symmetric positive definite systems `Ku = f`.
//!
//! ## Trait hierarchy
//!
//! ```text
//! LinearSolver<T>                 ← this module
//!     └── CholeskySolver<T>       ← linear::cholesky
//! ```
//!
//! ## Three-phase design
//!
//! All linear solvers in Echelon follow the same three-phase protocol so that
//! the `analysis` crate can drive any concrete solver through the trait:
//!
//! ```text
//! solver.analyze(&K)       — symbolic phase, once per topology change
//! solver.factorize(&K)     — numeric phase, once per Newton iteration
//! solver.solve(&f, &mut u) — triangular solve, once per RHS
//! ```
//!
//! `analyze_and_factorize` is a provided convenience method that calls both
//! in sequence and is the correct entry point for a fresh solve.

pub mod cholesky;
pub mod ldlt;

pub use cholesky::CholeskySolver;
pub use ldlt::LdltSolver;

use sparse::{SparseScalar, SymCsrMatrix};
use crate::error::Result;

// -----------------------------------------------------------------
// LinearSolver trait
// -----------------------------------------------------------------

/// Interface for sparse direct solvers of symmetric positive definite systems.
///
/// Implementors follow the three-phase protocol:
///
/// 1. **`analyze`** — symbolic phase. Computes fill-reduction ordering and
///    determines the sparsity pattern of the factor `L`. Runs **once per
///    topology change** (when the non-zero structure of `K` changes). Cheap
///    to call when the pattern is unchanged; the result is cached internally.
///
/// 2. **`factorize`** — numeric phase. Computes the numerical values of `L`
///    from `K` and the symbolic pattern. Runs **once per Newton iteration**.
///    Requires `analyze` to have been called first.
///
/// 3. **`solve`** — triangular solve. Computes `u = K⁻¹ f` via forward/
///    backward substitution. Runs **once per right-hand side**. Requires
///    `factorize` to have been called first.
///
/// # Type parameter `T`
///
/// `T` is the scalar type. In practice `T = f64` for all structural analyses.
/// The bound is kept generic so the trait can accommodate `f32` or dual-number
/// types in future.
pub trait LinearSolver<T: SparseScalar> {
    /// Symbolic phase: compute the fill-reduction ordering and the non-zero
    /// pattern of the factor.
    ///
    /// Call once after the mesh topology is fixed. Reuse across all Newton
    /// iterations and load steps while the connectivity is unchanged.
    ///
    /// # Errors
    /// Propagates any error from the ordering or symbolic factorization step.
    fn analyze(&mut self, k: &SymCsrMatrix<T>) -> Result<()>;

    /// Numeric phase: compute the numerical values of the factor from `K`.
    ///
    /// Re-uses the symbolic pattern from `analyze`. Call once per Newton
    /// iteration whenever the stiffness values change.
    ///
    /// # Errors
    /// - [`crate::error::SolverError::NotAnalyzed`] if `analyze` has not been called.
    /// - [`crate::error::SolverError::NotPositiveDefinite`] if `K` is not SPD.
    fn factorize(&mut self, k: &SymCsrMatrix<T>) -> Result<()>;

    /// Triangular solve: compute `u = K⁻¹ f`.
    ///
    /// Both `f` and `u` are in the original (unpermuted) DOF order.
    ///
    /// # Errors
    /// - [`crate::error::SolverError::NotFactorized`] if `factorize` has not been called.
    /// - [`crate::error::SolverError::RhsSizeMismatch`] if vector lengths are inconsistent.
    fn solve(&mut self, f: &[T], u: &mut [T]) -> Result<()>;

    /// Convenience: `analyze` then `factorize` in one call.
    ///
    /// This is the correct entry point for the first solve on a new model.
    /// Subsequent Newton iterations should call only `factorize` (reusing
    /// the symbolic phase) and then `solve`.
    ///
    /// # Errors
    /// Propagates errors from either `analyze` or `factorize`.
    fn analyze_and_factorize(&mut self, k: &SymCsrMatrix<T>) -> Result<()> {
        self.analyze(k)?;
        self.factorize(k)
    }
}