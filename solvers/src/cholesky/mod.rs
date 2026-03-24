pub mod symbolic;
pub mod numeric;
pub mod solve;

use crate::error::{SolverError, Result};
use crate::ordering::{Graph, Permutation, rcm};
use sparse::SymCsrMatrix;
use sparse::convert::sym_to_csc;

/// Sparse Cholesky solver for symmetric positive definite systems `Ku = f`.
///
/// # Three-phase design
///
/// 1. **`analyze(&K)`** — symbolic phase.
///    Computes the fill-reduction ordering (either RCM or a user‑supplied
///    permutation), applies it to `K`, then determines the sparsity pattern
///    of the Cholesky factor `L`.  Run **once per topology** (every time the
///    non-zero structure of `K` changes, e.g. when elements are added or
///    removed).  Reuse across all Newton iterations and load steps as long
///    as the topology is unchanged.
///
/// 2. **`factorize(&K)`** — numeric phase.
///    Computes the numerical values of `L` from `K` and the symbolic pattern.
///    Run **once per Newton iteration** (or any time the values of `K` change
///    while its pattern stays the same).  Requires `analyze` to have been
///    called first.
///
/// 3. **`solve(&f, &mut u)`** — triangular solve.
///    Computes `u = K⁻¹ f` via forward/backward substitution with the
///    permutation.  Run **once per right-hand side**.  Requires `factorize`.
///
/// # Custom ordering
/// By default, the solver uses the Reverse Cuthill–McKee (RCM) ordering.
/// To supply a custom permutation, call [`set_ordering`] before `analyze`.
pub struct SparseSolver {
    /// User‑provided permutation (if any).  If `None`, RCM will be computed.
    user_perm: Option<Permutation>,
    /// RCM permutation — stored so `factorize` can re-permute K and `solve`
    /// can unpermute the solution without re-running the ordering.
    perm:     Option<Permutation>,
    symbolic: Option<symbolic::SymbolicCholesky>,
    numeric:  Option<numeric::NumericCholesky>,
}

impl SparseSolver {
    /// Create a new solver.  No allocations occur until `analyze` is called.
    pub fn new() -> Self {
        Self {
            user_perm: None,
            perm: None,
            symbolic: None,
            numeric: None,
        }
    }

    /// Set a custom permutation to be used in the next call to `analyze`.
    ///
    /// The permutation must be a bijection from the reordered index space
    /// to the original index space (i.e., `perm[new] = old`).  It must be
    /// compatible with the matrix that will be passed to `analyze`.
    ///
    /// Calling this method invalidates any previously computed factorization.
    pub fn set_ordering(&mut self, perm: Permutation) {
        self.user_perm = Some(perm);
        // Invalidate any existing analysis/factorization.
        self.perm = None;
        self.symbolic = None;
        self.numeric = None;
    }

    /// Symbolic phase: compute the fill‑reduction ordering and the pattern of `L`.
    ///
    /// Internally this:
    /// 1. Builds the adjacency graph of `K`.
    /// 2. Uses either the user‑supplied permutation (from `set_ordering`) or
    ///    computes the RCM ordering.
    /// 3. Applies `P` to produce `K_perm = P K Pᵀ`.
    /// 4. Runs symbolic Cholesky on `K_perm` to get the pattern of `L`.
    ///
    /// The permutation is stored in `self` for use in `factorize` and `solve`.
    ///
    /// Safe to call again if the topology (non-zero pattern) of `K` changes.
    /// Calling `analyze` again invalidates any previous factorization.
    ///
    /// # Errors
    /// - Propagates any [`SolverError`] from the symbolic phase.
    pub fn analyze(&mut self, k: &SymCsrMatrix) -> Result<()> {
        // 1. Determine the ordering: user‑supplied or RCM
        let perm = if let Some(p) = self.user_perm.take() {
            p
        } else {
            let g = Graph::from_sym(k);
            rcm(&g)
        };

        // 2. Permute K and convert to CSC (pattern-only; values unused here)
        let k_perm = perm.permute_sym(k)?;

        // 3. Symbolic Cholesky on the permuted matrix
        let sym = symbolic::analyze(&k_perm)?;

        self.perm = Some(perm);
        self.symbolic = Some(sym);
        self.numeric = None; // invalidate any previous factorization
        Ok(())
    }

    /// Numeric phase: factorize the permuted `K = LLᵀ`.
    ///
    /// Re-applies the stored RCM permutation to `K` and computes the
    /// numerical values of `L`.  The permuted matrix is converted to CSC
    /// format for column-oriented factorization.
    ///
    /// # Errors
    /// - [`SolverError::NotAnalyzed`] if `analyze` has not been called.
    /// - [`SolverError::NotPositiveDefinite`] if `K` is not SPD.
    pub fn factorize(&mut self, k: &SymCsrMatrix) -> Result<()> {
        let perm = self.perm.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let sym = self.symbolic.as_ref().ok_or(SolverError::NotAnalyzed)?;

        let k_perm = perm.permute_sym(k)?;
        let k_csc = sym_to_csc(&k_perm);

        self.numeric = Some(numeric::factorize(&k_csc, sym)?);
        Ok(())
    }

    /// Triangular solve: compute `u = K⁻¹ f`.
    ///
    /// Applies the permutation, performs forward/backward substitution,
    /// and unpermutes the result.  Both `f` and `u` are in the **original**
    /// (unpermuted) DOF order.
    ///
    /// # Errors
    /// - [`SolverError::NotFactorized`] if `factorize` has not been called.
    /// - [`SolverError::RhsSizeMismatch`] if `f.len() != K.n` or `u.len() != K.n`.
    pub fn solve(&self, f: &[f64], u: &mut [f64]) -> Result<()> {
        let perm = self.perm.as_ref().ok_or(SolverError::NotFactorized)?;
        let sym = self.symbolic.as_ref().ok_or(SolverError::NotFactorized)?;
        let num = self.numeric.as_ref().ok_or(SolverError::NotFactorized)?;
        solve::solve(sym, num, perm, f, u)
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

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;

    fn tridiag(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    fn check_residual(k: &SymCsrMatrix, f: &[f64], u: &[f64]) {
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
        let mut solver = SparseSolver::new();
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
        let mut solver = SparseSolver::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn refactorize_with_new_values() {
        // Analyze once, factorize twice with different K values.
        // (Same pattern — different stiffness coefficients.)
        let k1 = tridiag(4);
        let mut coo2 = CooBuilder::new(4, 4);
        for i in 0..4       { coo2.add(i, i,      3.0); } // different diagonal
        for i in 0..3       { coo2.add(i, i + 1, -1.0); }
        let k2 = coo2.build_sym().unwrap();

        let f = vec![1.0; 4];
        let mut solver = SparseSolver::new();
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
        let mut solver = SparseSolver::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    // ---- error ordering ----

    #[test]
    fn factorize_before_analyze_errors() {
        let k = tridiag(3);
        let mut solver = SparseSolver::new();
        assert!(matches!(
            solver.factorize(&k).unwrap_err(),
            SolverError::NotAnalyzed
        ));
    }

    #[test]
    fn solve_before_factorize_errors() {
        let k = tridiag(3);
        let mut solver = SparseSolver::new();
        solver.analyze(&k).unwrap();
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solver.solve(&[1.0, 0.0, 0.0], &mut u).unwrap_err(),
            SolverError::NotFactorized
        ));
    }

    #[test]
    fn solve_before_analyze_errors() {
        let solver = SparseSolver::new();
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solver.solve(&[1.0, 0.0, 0.0], &mut u).unwrap_err(),
            SolverError::NotFactorized
        ));
    }

    #[test]
    fn analyze_invalidates_previous_numeric() {
        let k = tridiag(3);
        let mut solver = SparseSolver::new();
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
        let mut solver = SparseSolver::new();
        solver.analyze(&k).unwrap();
        assert!(matches!(
            solver.factorize(&k).unwrap_err(),
            SolverError::NotPositiveDefinite { .. }
        ));
    }
}