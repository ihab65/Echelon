//! Sparse Cholesky direct solver.
//!
//! [`CholeskySolver`] implements [`LinearSolver`] using left-looking sparse
//! Cholesky factorization with a fill-reduction ordering. It is the primary
//! solver for symmetric positive definite systems in Echelon.
//!
//! ## Three-phase usage
//!
//! ```rust,ignore
//! use solvers::linear::{CholeskySolver, LinearSolver};
//! use solvers::ordering::Ordering;
//!
//! let mut solver = CholeskySolver::new();           // default: Amd ordering
//! solver.set_ordering(Ordering::Rcm);               // optional override
//!
//! solver.analyze(&k)?;                              // once per topology
//! loop {
//!     // ... assemble K, apply BCs ...
//!     solver.factorize(&k)?;                        // once per Newton step
//!     solver.solve(&f, &mut u)?;                    // once per RHS
//! }
//! ```
//!
//! ## Ordering selection
//!
//! The ordering is a property of the concrete `CholeskySolver`, not the
//! `LinearSolver` trait. Set it before the first `analyze` call:
//!
//! | Ordering | Best for |
//! |----------|----------|
//! | `Ordering::Amd` (default) | Irregular meshes, frame joints, mixed topologies |
//! | `Ordering::Rcm` | Regular 1-D / 2-D grids with natural numbering |
//! | `Ordering::Natural` | Already-optimal orderings; debugging |
//! | `Ordering::Custom(p)` | Pre-computed permutation from an external tool |

use sparse::{ConvertWorkspace, SparseScalar, SymCsrMatrix};
use sparse::convert::{sym_to_csc, sym_to_csc_into};

use crate::cholesky::{symbolic, numeric};
use crate::cholesky::numeric::CholeskyWorkspace;
use crate::cholesky::solve as chol_solve;
use crate::error::{Result, SolverError};
use crate::linear::LinearSolver;
use crate::ordering::{Ordering, Permutation};

// -----------------------------------------------------------------
// Pre-allocated workspace
// -----------------------------------------------------------------

/// Cached buffers for the numeric factorization and solve phases.
///
/// Allocated once in `analyze()` and reused across all `factorize()`
/// and `solve()` calls, eliminating per-iteration heap allocations.
struct CholWorkspace<T: SparseScalar> {
    /// Workspace for the left-looking column Cholesky (w, active, touched, stack).
    chol_ws: CholeskyWorkspace<T>,
    /// Scratch buffer for the unpermute step in `solve()`.
    x_perm: Vec<T>,
    /// Constant index mapping from original `K` values to `k_perm` values.
    k_perm_map: Vec<usize>,
    /// Persistent permuted symmetric matrix `K`.
    k_perm: SymCsrMatrix<T>,
    /// Persistent CSC column pointers for the permuted matrix.
    csc_col_ptr: Vec<usize>,
    /// Persistent CSC row indices for the permuted matrix.
    csc_row_idx: Vec<usize>,
    /// Persistent CSC values for the permuted matrix.
    csc_values: Vec<T>,
    /// Workspace for `sym_to_csc_into` (col_count, cursor, pairs).
    convert_ws: ConvertWorkspace<T>,
}

impl<T: SparseScalar> CholWorkspace<T> {
    fn new(n: usize, k_perm: SymCsrMatrix<T>, k_perm_map: Vec<usize>) -> Self {
        Self {
            chol_ws: CholeskyWorkspace::new(n),
            x_perm: vec![T::zero(); n],
            k_perm_map,
            k_perm,
            csc_col_ptr: vec![0; n + 1],
            csc_row_idx: Vec::new(),
            csc_values: Vec::new(),
            convert_ws: ConvertWorkspace::new(n),
        }
    }
}

// -----------------------------------------------------------------
// CholeskySolver
// -----------------------------------------------------------------

/// Sparse Cholesky solver for symmetric positive definite systems `Ku = f`.
///
/// Implements [`LinearSolver<T>`] following the standard three-phase protocol.
/// The fill-reduction ordering strategy is configured on the struct before the
/// first `analyze` call and does not change the trait interface.
///
/// # Zero-allocation factorize/solve
///
/// After the initial `analyze()` call, all subsequent `factorize()` and
/// `solve()` calls reuse pre-allocated workspace buffers mapping values 
/// consistently. There are no allocations within the `factorize` and
/// `solve` hot-loops.
///
/// # Default ordering
///
/// `CholeskySolver::new()` uses [`Ordering::Amd`] by default. AMD consistently
/// outperforms RCM for the irregular frame/truss topologies common in structural
/// engineering. Override with [`set_ordering`](CholeskySolver::set_ordering).
pub struct CholeskySolver<T: SparseScalar> {
    /// Fill-reduction ordering strategy used in the next `analyze` call.
    ordering: Ordering,
    /// Computed permutation — reused by `factorize` and `solve`.
    perm:     Option<Permutation>,
    /// Cached symbolic phase (elimination tree + fill pattern + children).
    pub symbolic: Option<symbolic::SymbolicCholesky>,
    /// Pre-allocated numeric factor (values overwritten each `factorize`).
    numeric:  Option<numeric::NumericCholesky<T>>,
    /// Pre-allocated workspace buffers for factorize and solve.
    workspace: Option<CholWorkspace<T>>,
    /// Whether `factorize` has been called since the last `analyze`.
    ///
    /// The `numeric` buffer is pre-allocated in `analyze()` but only
    /// contains valid factor values after `factorize()` succeeds.
    factorized: bool,
}

impl<T: SparseScalar> CholeskySolver<T> {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Create a new solver with [`Ordering::Amd`] (the recommended default).
    pub fn new() -> Self {
        Self {
            ordering: Ordering::Amd,
            perm:     None,
            symbolic: None,
            numeric:  None,
            workspace: None,
            factorized: false,
        }
    }

    /// Create a new solver with [`Ordering::Rcm`].
    ///
    /// Convenience constructor for cases where the caller knows the matrix
    /// has a regular banded structure that benefits from RCM.
    pub fn with_rcm() -> Self {
        Self {
            ordering: Ordering::Rcm,
            ..Self::new()
        }
    }

    // -----------------------------------------------------------------
    // Ordering selection (concrete type only — not on the trait)
    // -----------------------------------------------------------------

    /// Override the fill-reduction ordering strategy.
    ///
    /// Must be called **before** `analyze`. Calling it after `analyze` clears
    /// the symbolic factorization, requiring a new `analyze` call.
    ///
    /// ```rust,ignore
    /// use solvers::linear::CholeskySolver;
    /// use solvers::ordering::Ordering;
    ///
    /// let mut solver = CholeskySolver::<f64>::new();
    /// solver.set_ordering(Ordering::Rcm);      // regular grid
    /// solver.set_ordering(Ordering::Amd);      // irregular mesh
    /// solver.set_ordering(Ordering::Natural);  // debug / pre-ordered
    /// ```
    pub fn set_ordering(&mut self, ordering: Ordering) {
        self.ordering = ordering;
        // Invalidate any existing analysis — the ordering change makes it stale.
        self.perm      = None;
        self.symbolic  = None;
        self.numeric   = None;
        self.workspace = None;
        self.factorized = false;
    }

    /// Return a reference to the current ordering strategy.
    #[inline]
    pub fn ordering(&self) -> &Ordering {
        &self.ordering
    }

    /// Return `true` if `analyze` has been called and its result is still valid.
    #[inline]
    pub fn is_analyzed(&self) -> bool {
        self.symbolic.is_some()
    }

    /// Return `true` if `factorize` has been called since the last `analyze`.
    #[inline]
    pub fn is_factorized(&self) -> bool {
        self.factorized
    }
}

// -----------------------------------------------------------------
// LinearSolver trait implementation
// -----------------------------------------------------------------

impl<T: SparseScalar> LinearSolver<T> for CholeskySolver<T> {
    /// Symbolic phase: compute the fill-reduction ordering and the pattern of `L`.
    ///
    /// 1. Computes a permutation from the configured [`Ordering`] strategy.
    /// 2. Permutes `K` and converts to full CSC format.
    /// 3. Runs symbolic Cholesky to obtain the pattern of `L`.
    /// 4. Pre-allocates the numeric factor and all workspace buffers.
    ///
    /// The permutation is cached and reused by `factorize` and `solve`.
    /// Calling `analyze` again invalidates any previous factorization.
    ///
    /// # Errors
    /// - [`SolverError::Sparse`] (via generic error mapping) if matrix permutation or format conversion fails.
    fn analyze(&mut self, k: &SymCsrMatrix<T>) -> Result<()> {
        let perm = self.ordering.clone().into_permutation(k);
        let (k_perm, k_perm_map) = perm.permute_sym_with_map(k)?;
        let k_csc = sym_to_csc(&k_perm);
        let sym   = symbolic::analyze(&k_csc)?;

        // Pre-allocate numeric factors and workspaces based on the pattern
        let n   = sym.n;
        let nnz = sym.nnz_l();
        self.numeric    = Some(numeric::NumericCholesky::new(n, nnz));
        self.workspace  = Some(CholWorkspace::new(n, k_perm, k_perm_map));
        self.factorized = false;

        self.perm     = Some(perm);
        self.symbolic = Some(sym);
        Ok(())
    }

    /// Numeric phase: compute the values of `L` from `K` and the cached pattern.
    ///
    /// Re-permutes `K` using the stored permutation and refactorizes. The
    /// symbolic pattern from `analyze` is reused — no pattern work is repeated.
    /// CSC conversion and numeric factorization use pre-allocated workspace
    /// buffers, avoiding per-iteration heap allocations.
    ///
    /// # Errors
    /// - [`SolverError::NotAnalyzed`] if `analyze` has not been called.
    /// - [`SolverError::NotPositiveDefinite`] if the matrix is found to not be strictly positive definite.
    fn factorize(&mut self, k: &SymCsrMatrix<T>) -> Result<()> {
        let perm = self.perm.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let sym  = self.symbolic.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let num  = self.numeric.as_mut().ok_or(SolverError::NotAnalyzed)?;
        let ws   = self.workspace.as_mut().ok_or(SolverError::NotAnalyzed)?;

        // Zero-allocation permutation reusing the fixed sparsity pattern
        perm.permute_sym_into(k, &mut ws.k_perm, &ws.k_perm_map);

        // Zero-allocation CSC conversion using cached buffers
        sym_to_csc_into(
            &ws.k_perm,
            &mut ws.csc_col_ptr,
            &mut ws.csc_row_idx,
            &mut ws.csc_values,
            &mut ws.convert_ws,
        );

        // Zero-allocation numeric factorization using cached workspace
        numeric::factorize_into(
            sym, num, &mut ws.chol_ws,
            &ws.csc_col_ptr, &ws.csc_row_idx, &ws.csc_values,
        )?;
        self.factorized = true;
        Ok(())
    }

    /// Triangular solve: compute `u = K⁻¹ f`.
    ///
    /// Applies the permutation, performs forward/backward substitution, and
    /// unpermutes. Both `f` and `u` are in the original (unpermuted) DOF order.
    /// Uses a pre-allocated scratch buffer for the unpermute step.
    ///
    /// # Errors
    /// - [`SolverError::NotFactorized`] if `factorize` has not been called.
    /// - [`SolverError::RhsSizeMismatch`] if vector lengths are inconsistent with matrix dimensions.
    fn solve(&mut self, f: &[T], u: &mut [T]) -> Result<()> {
        if !self.factorized {
            return Err(SolverError::NotFactorized);
        }
        let perm = self.perm.as_ref().ok_or(SolverError::NotFactorized)?;
        let sym  = self.symbolic.as_ref().ok_or(SolverError::NotFactorized)?;
        let num  = self.numeric.as_ref().ok_or(SolverError::NotFactorized)?;
        let ws   = self.workspace.as_mut().ok_or(SolverError::NotFactorized)?;

        chol_solve::solve_with_buffer(sym, num, perm, f, u, &mut ws.x_perm)
    }
}

// -----------------------------------------------------------------
// Default
// -----------------------------------------------------------------

impl<T: SparseScalar> Default for CholeskySolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;
    use crate::ordering::Ordering;

    fn tridiag(n: usize) -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    fn check_residual(k: &SymCsrMatrix<f64>, f: &[f64], u: &[f64]) {
        let ku = k.matvec(u).unwrap();
        for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
            assert!((kui - fi).abs() < 1e-9, "residual[{i}] = {:.2e}", (kui - fi).abs());
        }
    }

    // ---- basic three-phase pipeline ----

    #[test]
    fn analyze_factorize_solve_tridiag_3() {
        let k = tridiag(3);
        let f = vec![1.0, 0.0, 1.0];
        let mut u = vec![0.0; 3];
        let mut solver = CholeskySolver::new();
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
        let mut solver = CholeskySolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    // ---- ordering variants ----

    #[test]
    fn rcm_ordering_same_solution() {
        let k = tridiag(20);
        let f: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut u = vec![0.0; 20];
        let mut solver = CholeskySolver::with_rcm();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn natural_ordering_same_solution() {
        let k = tridiag(10);
        let f = vec![1.0; 10];
        let mut u = vec![0.0; 10];
        let mut solver = CholeskySolver::<f64>::new();
        solver.set_ordering(Ordering::Natural);
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn set_ordering_invalidates_previous_analysis() {
        let k = tridiag(4);
        let mut solver = CholeskySolver::<f64>::new();
        solver.analyze(&k).unwrap();
        assert!(solver.is_analyzed());

        solver.set_ordering(Ordering::Rcm);
        assert!(!solver.is_analyzed());
        assert!(!solver.is_factorized());
    }

    // ---- reanalysis ----

    #[test]
    fn refactorize_reuses_symbolic() {
        let k1 = tridiag(4);
        let mut coo2 = CooBuilder::new(4, 4);
        for i in 0..4       { coo2.add(i, i, 3.0); }
        for i in 0..3       { coo2.add(i, i + 1, -1.0); }
        let k2 = coo2.build_sym().unwrap();

        let f = vec![1.0; 4];
        let mut solver = CholeskySolver::new();
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

    // ---- error ordering ----

    #[test]
    fn factorize_before_analyze_errors() {
        let k = tridiag(3);
        let mut solver = CholeskySolver::<f64>::new();
        assert!(matches!(solver.factorize(&k).unwrap_err(), SolverError::NotAnalyzed));
    }

    #[test]
    fn solve_before_factorize_errors() {
        let k = tridiag(3);
        let mut solver = CholeskySolver::<f64>::new();
        solver.analyze(&k).unwrap();
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solver.solve(&[1.0, 0.0, 0.0], &mut u).unwrap_err(),
            SolverError::NotFactorized
        ));
    }

    #[test]
    fn not_positive_definite_errors() {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, -4.0);
        coo.add(1, 1,  4.0);
        let k = coo.build_sym().unwrap();
        let mut solver = CholeskySolver::<f64>::new();
        solver.analyze(&k).unwrap();
        assert!(matches!(
            solver.factorize(&k).unwrap_err(),
            SolverError::NotPositiveDefinite { .. }
        ));
    }

    // ---- state helpers ----

    #[test]
    fn is_analyzed_and_is_factorized() {
        let k = tridiag(4);
        let mut solver = CholeskySolver::<f64>::new();
        assert!(!solver.is_analyzed());
        assert!(!solver.is_factorized());

        solver.analyze(&k).unwrap();
        assert!(solver.is_analyzed());
        assert!(!solver.is_factorized());

        solver.factorize(&k).unwrap();
        assert!(solver.is_analyzed());
        assert!(solver.is_factorized());
    }
}