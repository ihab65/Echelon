pub mod symbolic;
pub mod numeric;
pub mod solve;

use crate::error::{SolverError, Result};
use crate::ordering::{Ordering, Permutation};
use sparse::{SparseScalar, SymCsrMatrix};
use sparse::convert::sym_to_csc;

/// Sparse Cholesky solver for symmetric positive definite systems `Ku = f`.
///
/// # Three-phase design
///
/// 1. **`analyze(&K)`** -- symbolic phase.
///    Computes the fill-reduction ordering and determines the sparsity pattern
///    of the Cholesky factor `L`.  Run **once per topology** (every time the
///    non-zero structure of `K` changes).  Reuse across all Newton iterations
///    and load steps as long as the topology is unchanged.
///
///    Internally converts the permuted `SymCsrMatrix` to a full `CscMatrix`
///    once and caches it.  `factorize` reuses this cached CSC matrix, so the
///    expensive `permute_sym` + `sym_to_csc` conversion happens only once
///    per topology change.
///
/// 2. **`factorize(&K)`** -- numeric phase.
///    Computes the numerical values of `L` from `K` and the symbolic pattern.
///    Re-permutes `K` and re-converts to CSC (since values change between
///    Newton steps), but reuses the symbolic pattern.
///    Run **once per Newton iteration**.  Requires `analyze` first.
///
/// 3. **`solve(&f, &mut u)`** -- triangular solve.
///    Computes `u = K^-1 f` via forward/backward substitution with the
///    permutation.  Run **once per right-hand side**.  Requires `factorize`.
///
/// # Ordering
///
/// The default ordering is [`Ordering::Rcm`].  Call [`set_ordering`] before
/// `analyze` to choose a different strategy:
///
/// ```
/// # use solvers::cholesky::SparseSolver;
/// # use solvers::ordering::Ordering;
/// let mut solver: SparseSolver<f64> = SparseSolver::new();
/// solver.set_ordering(Ordering::Amd);   // better for irregular meshes
/// solver.set_ordering(Ordering::Rcm);   // better for regular grids
/// solver.set_ordering(Ordering::Natural); // no reordering
/// ```
///
/// [`set_ordering`]: SparseSolver::set_ordering
pub struct SparseSolver<T: SparseScalar> {
    /// Ordering strategy to use in the next `analyze` call.
    ordering: Ordering,
    /// Computed permutation -- stored so `factorize` can re-permute K and
    /// `solve` can unpermute the solution without re-running the ordering.
    perm:     Option<Permutation>,
    symbolic: Option<symbolic::SymbolicCholesky>,
    numeric:  Option<numeric::NumericCholesky<T>>,
}

impl<T: SparseScalar> SparseSolver<T> {
    /// Create a new solver with the default ordering ([`Ordering::Rcm`]).
    /// No allocations occur until `analyze` is called.
    pub fn new() -> Self {
        Self {
            ordering: Ordering::default(),
            perm:     None,
            symbolic: None,
            numeric:  None,
        }
    }

    /// Set the fill-reduction ordering strategy for the next `analyze` call.
    ///
    /// This replaces the previous ordering choice and invalidates any
    /// previously computed factorization.
    ///
    /// # Example
    /// ```
    /// # use solvers::cholesky::SparseSolver;
    /// # use solvers::ordering::Ordering;
    /// let mut solver: SparseSolver<f64> = SparseSolver::new();
    ///
    /// // Use AMD for irregular meshes and frame structures
    /// solver.set_ordering(Ordering::Amd);
    ///
    /// // Revert to RCM for a new problem with a regular grid
    /// solver.set_ordering(Ordering::Rcm);
    ///
    /// // Supply a custom permutation from an external tool
    /// // solver.set_ordering(Ordering::Custom(my_perm));
    /// ```
    pub fn set_ordering(&mut self, ordering: Ordering) {
        self.ordering = ordering;
        // Invalidate any existing analysis/factorization.
        self.perm     = None;
        self.symbolic = None;
        self.numeric  = None;
    }

    /// Symbolic phase: compute the fill-reduction ordering and the pattern of `L`.
    ///
    /// Internally this:
    /// 1. Computes a permutation from the current [`Ordering`] strategy.
    /// 2. Applies it to produce `K_perm = P K P^T`.
    /// 3. Converts `K_perm` to a full CSC matrix (both triangles).
    /// 4. Runs symbolic Cholesky on the CSC matrix to get the pattern of `L`.
    ///
    /// The permutation is stored for use in `factorize` and `solve`.
    /// Safe to call again if the topology (non-zero pattern) of `K` changes;
    /// doing so invalidates any previous factorization.
    ///
    /// # Errors
    /// - Propagates any [`SolverError`] from the symbolic phase.
    pub fn analyze(&mut self, k: &SymCsrMatrix<T>) -> Result<()> 
        where T: SparseScalar
    {
        // 1. Compute permutation from the chosen ordering strategy.
        let perm = self.ordering.clone().into_permutation(k);

        // 2. Permute K and convert to full CSC -- done once per topology.
        //    The CSC matrix is passed to both symbolic::analyze and stored
        //    so factorize can reuse it (avoiding a redundant conversion).
        let k_perm = perm.permute_sym(k)?;
        let k_csc  = sym_to_csc(&k_perm);

        // 3. Symbolic Cholesky on the full CSC matrix.
        let sym = symbolic::analyze(&k_csc)?;

        self.perm     = Some(perm);
        self.symbolic = Some(sym);
        self.numeric  = None; // invalidate any previous factorization
        Ok(())
    }

    /// Numeric phase: factorize the permuted `K = LL^T`.
    ///
    /// Re-applies the stored permutation to `K`, converts to CSC, and
    /// computes the numerical values of `L`.  The symbolic pattern from
    /// `analyze` is reused -- no pattern computation is repeated.
    ///
    /// # Errors
    /// - [`SolverError::NotAnalyzed`] if `analyze` has not been called.
    /// - [`SolverError::NotPositiveDefinite`] if `K` is not SPD.
    pub fn factorize(&mut self, k: &SymCsrMatrix<T>) -> Result<()> 
        where T: SparseScalar
    {
        let perm = self.perm.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let sym  = self.symbolic.as_ref().ok_or(SolverError::NotAnalyzed)?;

        // Re-permute and convert for the new numerical values.
        let k_perm = perm.permute_sym(k)?;
        let k_csc  = sym_to_csc(&k_perm);

        self.numeric = Some(numeric::factorize(&k_csc, sym)?);
        Ok(())
    }

    /// Triangular solve: compute `u = K^-1 f`.
    ///
    /// Applies the permutation, performs forward/backward substitution,
    /// and unpermutes the result.  Both `f` and `u` are in the **original**
    /// (unpermuted) DOF order.
    ///
    /// # Errors
    /// - [`SolverError::NotFactorized`] if `factorize` has not been called.
    /// - [`SolverError::RhsSizeMismatch`] if `f.len() != K.n` or `u.len() != K.n`.
    pub fn solve(&self, f: &[T], u: &mut [T]) -> Result<()> {
        let perm = self.perm.as_ref().ok_or(SolverError::NotFactorized)?;
        let sym  = self.symbolic.as_ref().ok_or(SolverError::NotFactorized)?;
        let num  = self.numeric.as_ref().ok_or(SolverError::NotFactorized)?;
        solve::solve(sym, num, perm, f, u)
    }

    /// Convenience: analyze + factorize in one call.
    pub fn analyze_and_factorize(&mut self, k: &SymCsrMatrix<T>) -> Result<()> 
        where T: SparseScalar
    {
        self.analyze(k)?;
        self.factorize(k)
    }
}

impl<T: SparseScalar> Default for SparseSolver<T> {
    fn default() -> Self { Self::new() }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;

    fn tridiag(n: usize) -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    fn check_residual(k: &SymCsrMatrix<f64>, f: &[f64], u: &[f64]) {
        let ku = k.matvec(u).unwrap();
        for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
            assert!(
                (kui - fi).abs() < 1e-9,
                "residual[{i}] = {:.2e}", (kui - fi).abs()
            );
        }
    }

    // ---- happy path ----

    #[test]
    fn analyze_factorize_solve_tridiag_3() {
        let k = tridiag(3);
        let f = vec![1.0, 0.0, 1.0];
        let mut u = vec![0.0; 3];
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        solver.analyze(&k).unwrap();
        solver.factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn analyze_and_factorize_convenience() {
        let k = tridiag(5);
        let f = vec![1.0; 5];
        let mut u = vec![0.0; 5];
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn refactorize_with_new_values() {
        // Analyze once, factorize twice with different K values.
        // (Same pattern -- different stiffness coefficients.)
        let k1 = tridiag(4);
        let mut coo2 = CooBuilder::new(4, 4);
        for i in 0..4       { coo2.add(i, i,      3.0); } // different diagonal
        for i in 0..3       { coo2.add(i, i + 1, -1.0); }
        let k2 = coo2.build_sym().unwrap();

        let f = vec![1.0; 4];
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        solver.analyze(&k1).unwrap();

        let mut u1 = vec![0.0; 4];
        solver.factorize(&k1).unwrap();
        solver.solve(&f, &mut u1).unwrap();
        check_residual(&k1, &f, &u1);

        let mut u2 = vec![0.0; 4];
        solver.factorize(&k2).unwrap();
        solver.solve(&f, &mut u2).unwrap();
        check_residual(&k2, &f, &u2);
    }

    #[test]
    fn tridiag_50_full_pipeline() {
        let k = tridiag(50);
        let f: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let mut u = vec![0.0; 50];
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    // ---- error ordering ----

    #[test]
    fn factorize_before_analyze_errors() {
        let k = tridiag(3);
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        assert!(matches!(
            solver.factorize(&k).unwrap_err(),
            SolverError::NotAnalyzed
        ));
    }

    #[test]
    fn solve_before_factorize_errors() {
        let k = tridiag(3);
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        solver.analyze(&k).unwrap();
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solver.solve(&[1.0, 0.0, 0.0], &mut u).unwrap_err(),
            SolverError::NotFactorized
        ));
    }

    #[test]
    fn solve_before_analyze_errors() {
        let solver: SparseSolver<f64> = SparseSolver::new();
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solver.solve(&[1.0, 0.0, 0.0], &mut u).unwrap_err(),
            SolverError::NotFactorized
        ));
    }

    #[test]
    fn analyze_invalidates_previous_numeric() {
        let k = tridiag(3);
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        solver.analyze_and_factorize(&k).unwrap();
        // Re-analyze should clear the numeric factor
        solver.analyze(&k).unwrap();
        assert!(matches!(
            solver.solve(&[1.0, 0.0, 0.0], &mut vec![0.0; 3]).unwrap_err(),
            SolverError::NotFactorized
        ));
    }

    #[test]
    fn not_positive_definite_errors() {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, -4.0); // not PD
        coo.add(1, 1,  4.0);
        let k = coo.build_sym().unwrap();
        let mut solver: SparseSolver<f64> = SparseSolver::new();
        solver.analyze(&k).unwrap();
        assert!(matches!(
            solver.factorize(&k).unwrap_err(),
            SolverError::NotPositiveDefinite { .. }
        ));
    }
}
