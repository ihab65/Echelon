pub mod symbolic;
pub mod numeric;
pub mod solve;

use crate::error::{SolverError, Result};
use sparse::SymCsrMatrix;

/// Sparse Cholesky solver for symmetric positive definite systems `Ku = f`.
///
/// # Three-phase design
///
/// 1. **`analyze(&K)`** — symbolic phase: compute the fill pattern of `L`
///    using only the sparsity structure of `K`.  Run **once per topology**;
///    re-use across Newton iterations and load steps.
///
/// 2. **`factorize(&K)`** — numeric phase: compute the values of `L`.
///    Run **once per Newton iteration**.  Requires `analyze` to have been
///    called first.
///
/// 3. **`solve(&f, &mut u)`** — triangular solve: forward/backward
///    substitution.  Run **once per RHS**.  Requires `factorize` to have
///    been called first.
///
/// # Example (not yet runnable — numeric not implemented)
/// ```ignore
/// let mut solver = SparseSolver::new();
/// solver.analyze(&K)?;
/// solver.factorize(&K)?;
/// solver.solve(&f, &mut u)?;
/// ```
pub struct SparseSolver {
    symbolic: Option<symbolic::SymbolicCholesky>,
    numeric:  Option<numeric::NumericCholesky>,
}

impl SparseSolver {
    /// Create a new solver.  No allocations occur until `analyze` is called.
    pub fn new() -> Self {
        Self { symbolic: None, numeric: None }
    }

    /// Symbolic phase: analyse the sparsity pattern of `K` and pre-compute
    /// the pattern of `L`.
    ///
    /// Must be called before `factorize`.  Safe to call again if the
    /// topology (pattern) of `K` changes.
    pub fn analyze(&mut self, k: &SymCsrMatrix) -> Result<()> {
        self.symbolic = Some(symbolic::analyze(k)?);
        self.numeric  = None; // invalidate any previous factorization
        Ok(())
    }

    /// Numeric phase: factorize `K = LLᵀ`.
    ///
    /// # Errors
    /// - [`SolverError::NotAnalyzed`] if `analyze` has not been called
    /// - [`SolverError::NotPositiveDefinite`] if `K` is not SPD
    pub fn factorize(&mut self, k: &SymCsrMatrix) -> Result<()> {
        let sym = self.symbolic.as_ref().ok_or(SolverError::NotAnalyzed)?;
        self.numeric = Some(numeric::factorize(k, sym)?);
        Ok(())
    }

    /// Triangular solve: compute `u = K⁻¹ f`.
    ///
    /// # Errors
    /// - [`SolverError::NotFactorized`] if `factorize` has not been called
    /// - [`SolverError::RhsSizeMismatch`] if `f.len() != K.n`
    pub fn solve(&self, f: &[f64], u: &mut [f64]) -> Result<()> {
        let num = self.numeric.as_ref().ok_or(SolverError::NotFactorized)?;
        solve::solve(num, f, u)
    }

    /// Convenience: analyze + factorize in one call.
    pub fn analyze_and_factorize(&mut self, k: &SymCsrMatrix) -> Result<()> {
        self.analyze(k)?;
        self.factorize(k)
    }
}

impl Default for SparseSolver {
    fn default() -> Self { Self::new() }
}