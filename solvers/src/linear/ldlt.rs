//! Sparse LDLᵀ direct solver for symmetric indefinite systems `Ku = f`.
//!
//! [`LdltSolver`] implements [`LinearSolver`] using a left-looking sparse
//! LDLᵀ factorization. It **reuses the symbolic phase from [`CholeskySolver`]**
//! (same elimination tree + fill pattern) and therefore carries identical
//! sparsity-analysis cost. The numeric phase differs: instead of computing
//! `L` with `L[j,j] = sqrt(D[j,j])`, it extracts diagonal pivots `D[j,j]`
//! and stores `L` with unit diagonal, which avoids the square-root and
//! handles **symmetric indefinite** matrices (negative or zero pivots) that
//! would cause Cholesky to fail.
//!
//! ## When to use this solver
//!
//! | Condition | Solver |
//! |-----------|--------|
//! | `K` is symmetric positive definite (SPD) | [`CholeskySolver`] |
//! | `K` is symmetric but possibly indefinite (negative eigenvalues) | [`LdltSolver`] |
//!
//! Common indefinite cases in structural engineering:
//! - **Geometric stiffness** at or beyond buckling load: `K_T = K_e + λK_g`
//!   can have negative eigenvalues when `λ ≥ λ_cr`.
//! - **Saddle-point systems**: mixed displacement-pressure elements.
//! - **Dynamic systems** (`K - ω²M`): the shifted stiffness has the sign
//!   of eigenvalue differences and can be indefinite.
//! - **Pre-buckling continuation paths**: tangent stiffness passes through
//!   a limit point.
//!
//! ## Factorization — `K = L D Lᵀ`
//!
//! The factor `L` is **unit lower triangular** (`L[j,j] = 1` always) and
//! `D` is diagonal. The sparsity pattern of `L` is identical to that of
//! the Cholesky factor — so `SymbolicCholesky` is reused verbatim.
//!
//! ### Algorithm (left-looking, column by column)
//!
//! For column `j`:
//!
//! 1. **Scatter** column `j` of `K` (rows `≥ j`) into dense workspace `w`.
//! 2. **Left-looking update**: for every descendant `c < j` with `L[j,c] ≠ 0`:
//!    ```text
//!    w[i] -= L[j,c] * D[c] * L[i,c]   for all i ≥ j in pattern(L[:,c])
//!    ```
//! 3. **Pivot**: `D[j] = w[j]`.  Zero pivot → singular matrix, but negative
//!    pivot is **allowed** (indefinite).
//! 4. **Sub-diagonal entries**: `L[i,j] = w[i] / D[j]`  for `i > j`.
//! 5. Clear workspace.
//!
//! ## Solve — `K u = f`
//!
//! Given `K = P L D Lᵀ Pᵀ` (with permutation `P`):
//!
//! ```text
//! b = P f          1. permute RHS
//! L y = b          2. forward substitution (unit L)
//! D z = y          3. diagonal solve: z[j] = y[j] / D[j]
//! Lᵀ x = z         4. backward substitution (unit Lᵀ)
//! u = Pᵀ x         5. unpermute solution
//! ```
//!
//! ## Pivot tolerance and singular detection
//!
//! A zero (or near-zero) pivot `|D[j]| < 1e-14` is treated as structural
//! singularity and returns [`SolverError::NotPositiveDefinite`] (repurposed
//! as a generic singularity indicator). Negative pivots are allowed and
//! produce correct results.
//!
//! ## References
//! - Duff, I.S., Reid, J.K. (1983). "The multifrontal solution of indefinite
//!   sparse symmetric linear systems." *ACM TOMS 9*(3).
//! - Davis, T.A. (2006). *Direct Methods for Sparse Linear Systems*. §4.5.
//! - Ashcraft, C., Grimes, R., Lewis, J. (1998). "Accurate symmetric indefinite
//!   linear equation solvers." *SIAM J. Matrix Anal. Appl. 20*(2).

use sparse::{ConvertWorkspace, SparseScalar, SymCsrMatrix};
use sparse::convert::{sym_to_csc, sym_to_csc_into};

use crate::cholesky::symbolic;
use crate::cholesky::symbolic::SymbolicCholesky;
use crate::error::{Result, SolverError};
use crate::linear::LinearSolver;
use crate::ordering::{Ordering, Permutation};

/// Pre-allocated workspaces for the numeric factorization and solve phases.
struct LdltWorkspace<T: SparseScalar> {
    w: Vec<T>,
    active: Vec<bool>,
    touched: Vec<usize>,
    stack: Vec<usize>,
    x_perm: Vec<T>, // For the unpermute step in solve()

    // Arrays to hold the converted CSC matrix
    csc_col_ptr: Vec<usize>,
    csc_row_idx: Vec<usize>,
    csc_values: Vec<T>,
    
    // The workspace for the conversion math
    convert_ws: ConvertWorkspace<T>,
}

impl<T: SparseScalar> LdltWorkspace<T> {
    fn new(n: usize) -> Self {
        Self {
            w: vec![T::zero(); n],
            active: vec![false; n],
            touched: Vec::with_capacity(n),
            stack: Vec::with_capacity(n),
            x_perm: vec![T::zero(); n],
            csc_col_ptr: vec![0; n + 1],
            csc_row_idx: vec![0; 0], // Will be resized in analyze
            csc_values: vec![T::zero(); 0], // Will be resized in analyze
            convert_ws: ConvertWorkspace::new(n),
        }
    }
}

// -----------------------------------------------------------------
// Numeric LDLT factor
// -----------------------------------------------------------------

/// Values of the LDLᵀ factor, stored in CSC format (matching `SymbolicCholesky`).
///
/// `L` is unit lower triangular — the diagonal entries of `L` are always 1
/// and are **not stored**. Instead, the position `col_ptr[j]` in `l_values`
/// that would hold `L[j,j]` is unused (set to zero) so that index arithmetic
/// stays identical to the Cholesky case. `d_values[j]` holds `D[j,j]`.
///
/// # Memory layout
///
/// ```text
/// col_ptr[j]       → diagonal slot (skipped — L[j,j] = 1 always)
/// col_ptr[j]+1 ..  → L[i,j] for i > j in pattern(L[:,j])
/// ```
///
/// This mirrors `NumericCholesky` exactly, making the solve loops
/// structurally identical (just replacing `lv[col_ptr[j]]` with `d[j]`).
struct NumericLdlt<T: SparseScalar> {
    /// Sub-diagonal values of `L` (unit diagonal not stored).
    /// Indexed identically to `SymbolicCholesky::row_idx`.
    /// `l_values[col_ptr[j]]` is a dummy slot (value ignored).
    l_values: Vec<T>,
    /// Diagonal entries `D[j,j]`.  `d_values.len() == n`.
    d_values: Vec<T>,
    /// Dimension of the factored system.
    n: usize,
}

// Add a constructor to NumericLdlt so we can pre-allocate it
impl<T: SparseScalar> NumericLdlt<T> {
    fn new(n: usize, nnz: usize) -> Self {
        Self {
            l_values: vec![T::zero(); nnz],
            d_values: vec![T::zero(); n],
            n,
        }
    }
}

// -----------------------------------------------------------------
// LdltSolver — public struct
// -----------------------------------------------------------------

/// Sparse LDLᵀ solver for symmetric indefinite systems `Ku = f`.
///
/// Implements [`LinearSolver<T>`] following the standard three-phase protocol.
/// The symbolic phase is **identical** to [`CholeskySolver`] — both use the
/// same [`SymbolicCholesky`] elimination tree and fill-pattern analysis.
/// Only the numeric phase differs: `LdltSolver` avoids the square-root and
/// accepts matrices with negative eigenvalues.
///
/// # When to prefer over `CholeskySolver`
///
/// - Geometric non-linearity near or past buckling load.
/// - Dynamic residual stiffness `K - ω²M` for frequency-response analysis.
/// - Any symmetric system whose positive-definiteness is not guaranteed.
///
/// # Default ordering
///
/// Uses [`Ordering::Amd`] (same as `CholeskySolver`). Override with
/// [`set_ordering`](LdltSolver::set_ordering) before the first `analyze` call.
///
/// # Three-phase usage
///
/// ```rust,ignore
/// use solvers::linear::{LdltSolver, LinearSolver};
///
/// let mut solver = LdltSolver::<f64>::new();
///
/// solver.analyze(&k)?;        // once per topology change
/// loop {
///     solver.factorize(&k)?;  // once per Newton iteration
///     solver.solve(&f, &mut u)?;
/// }
/// ```
pub struct LdltSolver<T: SparseScalar> {
    /// Fill-reduction ordering strategy.
    ordering: Ordering,
    /// Cached fill-reduction permutation from the last `analyze` call.
    perm:     Option<Permutation>,
    /// Cached symbolic phase (elimination tree + fill pattern).
    symbolic: Option<SymbolicCholesky>,
    /// Cached numeric LDLᵀ factors from the last `factorize` call.
    numeric:  Option<NumericLdlt<T>>,
    /// Pre-allocated workspaces for numeric factorization and solve phases.
    workspace: Option<LdltWorkspace<T>>,
    /// Whether `factorize` has been called since the last `analyze`.
    ///
    /// The `numeric` buffer is pre-allocated in `analyze()` but only
    /// contains valid factor values after `factorize()` succeeds.
    factorized: bool,
}

impl<T: SparseScalar> LdltSolver<T> {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Create a new `LdltSolver` with the default [`Ordering::Amd`] strategy.
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
    /// Suitable when the matrix has a regular banded structure.
    pub fn with_rcm() -> Self {
        Self {
            ordering: Ordering::Rcm,
            ..Self::new()
        }
    }

    // -----------------------------------------------------------------
    // Ordering selection
    // -----------------------------------------------------------------

    /// Override the fill-reduction ordering.
    ///
    /// Must be called **before** `analyze`. Calling it after invalidates
    /// any cached symbolic analysis.
    pub fn set_ordering(&mut self, ordering: Ordering) {
        self.ordering = ordering;
        self.perm     = None;
        self.symbolic = None;
        self.numeric  = None;
        self.workspace = None;
        self.factorized = false;
    }

    /// Return the current ordering strategy.
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

    /// Number of negative diagonal pivots in the last factorization.
    ///
    /// By Sylvester's law of inertia, this equals the number of negative
    /// eigenvalues of the permuted matrix (and therefore of `K`). Returns
    /// `None` if `factorize` has not been called.
    ///
    /// # Application
    ///
    /// For structural stability analysis:
    /// - `negative_pivots() == 0` → `K` is positive definite (stable).
    /// - `negative_pivots() > 0`  → `K` is indefinite (structure has passed
    ///   one or more limit / bifurcation points).
    pub fn negative_pivots(&self) -> Option<usize> {
        if !self.factorized { return None; }
        self.numeric.as_ref().map(|num| {
            num.d_values
                .iter()
                .filter(|d| d.real_part() < 0.0)
                .count()
        })
    }

    /// The diagonal pivot vector `D` from the last factorization, in the
    /// **permuted** DOF order.
    ///
    /// Returns `None` if `factorize` has not been called.
    pub fn pivots(&self) -> Option<&[T]> {
        if !self.factorized { return None; }
        self.numeric.as_ref().map(|num| num.d_values.as_slice())
    }
}

// -----------------------------------------------------------------
// LinearSolver trait implementation
// -----------------------------------------------------------------

impl<T: SparseScalar> LinearSolver<T> for LdltSolver<T> {
    /// Symbolic phase: identical to `CholeskySolver::analyze`.
    ///
    /// Computes a fill-reduction permutation, converts to CSC, and runs
    /// symbolic Cholesky to obtain the elimination tree and fill pattern.
    /// The result is reused across all subsequent `factorize` calls as long
    /// as the non-zero pattern of `K` is unchanged.
    ///
    /// # Errors
    /// - [`SolverError::Sparse`] if permutation fails.
    fn analyze(&mut self, k: &SymCsrMatrix<T>) -> Result<()> {
        let perm   = self.ordering.clone().into_permutation(k);
        let k_perm = perm.permute_sym(k)?;
        let k_csc  = sym_to_csc(&k_perm);
        let sym    = symbolic::analyze(&k_csc)?;

        // Pre-allocate numeric factors and workspaces based on the pattern
        let n = sym.n;
        let nnz = sym.nnz_l();
        self.numeric   = Some(NumericLdlt::new(n, nnz));
        self.workspace = Some(LdltWorkspace::new(n));
        self.factorized = false;

        self.perm     = Some(perm);
        self.symbolic = Some(sym);
        Ok(())
    }

    /// Numeric phase: compute `L` and `D` from `K` and the cached pattern.
    ///
    /// Performs a left-looking LDLᵀ factorization. Negative pivots are
    /// allowed and stored in `D`. A pivot with `|D[j]| < 1e-14` indicates
    /// structural singularity and returns an error.
    ///
    /// # Errors
    /// - [`SolverError::NotAnalyzed`] if `analyze` has not been called.
    /// - [`SolverError::NotPositiveDefinite`] if a near-zero pivot is
    ///   encountered (matrix is structurally singular).
    fn factorize(&mut self, k: &SymCsrMatrix<T>) -> Result<()> {
        let perm = self.perm.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let sym  = self.symbolic.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let num  = self.numeric.as_mut().ok_or(SolverError::NotAnalyzed)?;
        let ws   = self.workspace.as_mut().ok_or(SolverError::NotAnalyzed)?;

        // TODO: Next refactor is perm.permute_sym_into(k, &mut ws.k_perm)
        let k_perm = perm.permute_sym(k)?; 
        
        sym_to_csc_into(
            &k_perm, 
            &mut ws.csc_col_ptr, 
            &mut ws.csc_row_idx, 
            &mut ws.csc_values, 
            &mut ws.convert_ws
        );

        ldlt_factorize(sym, num, ws)?;
        self.factorized = true;
        Ok(())
    }

    /// Triangular solve: compute `u = K⁻¹ f` using the LDLᵀ factors.
    ///
    /// Applies the permutation, performs forward / diagonal / backward
    /// substitution with the unit-diagonal `L` and `D`, then unpermutes.
    ///
    /// # Errors
    /// - [`SolverError::NotFactorized`] if `factorize` has not been called.
    /// - [`SolverError::RhsSizeMismatch`] if vector lengths are inconsistent.
    fn solve(&mut self, f: &[T], u: &mut [T]) -> Result<()> {
        if !self.factorized {
            return Err(SolverError::NotFactorized);
        }
        let perm = self.perm.as_ref().ok_or(SolverError::NotFactorized)?;
        let sym  = self.symbolic.as_ref().ok_or(SolverError::NotFactorized)?;
        let num  = self.numeric.as_ref().ok_or(SolverError::NotFactorized)?;
        
        ldlt_solve(sym, num, perm, f, u, &mut self.workspace.as_mut().unwrap().x_perm)
    }
}

// -----------------------------------------------------------------
// Default
// -----------------------------------------------------------------

impl<T: SparseScalar> Default for LdltSolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------
// Numeric factorization (private)
// -----------------------------------------------------------------

/// Compute the numeric LDLᵀ factorization of `k_csc`.
///
/// `k_csc` is the full (both-triangle) CSC matrix produced by
/// `sym_to_csc(perm.permute_sym(k))`.
///
/// The algorithm is the left-looking column Cholesky adapted for LDLᵀ:
/// the square-root step on the diagonal is replaced by a direct extraction
/// of `D[j] = w[j]`, and sub-diagonal entries are `L[i,j] = w[i] / D[j]`
/// (no division by `sqrt(D[j])`).
fn ldlt_factorize<T>(
    sym:   &SymbolicCholesky,
    num:   &mut NumericLdlt<T>,
    ws:    &mut LdltWorkspace<T>,
) -> Result<()>
where
    T: SparseScalar,
{
    let n = sym.n;
    
    // Clear numeric arrays just in case
    num.l_values.fill(T::zero());
    num.d_values.fill(T::zero());

    for j in 0..n {
        ws.touched.clear();

        // Step 1 — scatter directly from the workspace CSC arrays!
        let k_start = ws.csc_col_ptr[j];
        let k_end   = ws.csc_col_ptr[j + 1];

        for idx in k_start..k_end {
            let row = ws.csc_row_idx[idx];
            if row >= j {
                ws.w[row] = ws.csc_values[idx];
                if !ws.active[row] {
                    ws.active[row] = true;
                    ws.touched.push(row);
                }
            }
        }

        // Step 2 — left-looking update using CACHED children!
        ws.stack.clear();
        for &c in &sym.children[j] {
            ws.stack.push(c);
        }

        while let Some(c) = ws.stack.pop() {
            let col_start = sym.col_ptr[c];
            let col_end   = sym.col_ptr[c + 1];
            let col_rows  = &sym.row_idx[col_start..col_end];

            let local_j = match col_rows.binary_search(&j) {
                Ok(pos)  => pos,
                Err(_)   => continue,
            };

            let ljc = num.l_values[col_start + local_j];
            let dc  = num.d_values[c];

            for pos in local_j..col_rows.len() {
                let row = col_rows[pos];
                ws.w[row] -= ljc * dc * num.l_values[col_start + pos];
                if !ws.active[row] {
                    ws.active[row] = true;
                    ws.touched.push(row);
                }
            }

            for &gc in &sym.children[c] {
                ws.stack.push(gc);
            }
        }

        // Step 3 — pivot
        let dj = ws.w[j];
        if dj.real_part().abs() < 1e-14 {
            for &r in &ws.touched {
                ws.w[r]      = T::zero();
                ws.active[r] = false;
            }
            return Err(SolverError::NotPositiveDefinite { index: j, value: dj.real_part() });
        }
        num.d_values[j] = dj;

        // Step 4 — sub-diagonal
        let l_col_start = sym.col_ptr[j];
        let l_col_end   = sym.col_ptr[j + 1];

        for pos in (l_col_start + 1)..l_col_end {
            let row = sym.row_idx[pos];
            num.l_values[pos] = ws.w[row] / dj;
        }

        // Step 5 — clear workspace
        for &r in &ws.touched {
            ws.w[r]      = T::zero();
            ws.active[r] = false;
        }
    }

    Ok(())
}

// -----------------------------------------------------------------
// Triangular solve (private)
// -----------------------------------------------------------------

/// Solve `K u = f` using the LDLᵀ factors and permutation `P`.
///
/// The five-step solve:
/// ```text
/// b = P f       (permute RHS)
/// L y = b       (forward substitution with unit L)
/// D z = y       (diagonal solve)
/// Lᵀ x = z      (backward substitution with unit Lᵀ)
/// u = Pᵀ x      (unpermute)
/// ```
fn ldlt_solve<T>(
    sym:  &SymbolicCholesky,
    num:  &NumericLdlt<T>,
    perm: &Permutation,
    f:    &[T],
    u:    &mut [T],
    x_perm: &mut [T]
) -> Result<()>
where
    T: SparseScalar,
{
    let n = num.n;
    if f.len() != n {
        return Err(SolverError::RhsSizeMismatch { expected: n, got: f.len() });
    }
    if u.len() != n {
        return Err(SolverError::RhsSizeMismatch { expected: n, got: u.len() });
    }
    debug_assert_eq!(sym.n, n);
    debug_assert_eq!(perm.len(), n);

    // ------------------------------------------------------------------
    // Step 1 — permute RHS: b[i] = f[perm[i]]
    // ------------------------------------------------------------------
    for i in 0..n {
        u[i] = f[perm.old_index(i)];
    }

    // ------------------------------------------------------------------
    // Step 2 — forward substitution: L y = b  (unit diagonal, in-place)
    //
    // For unit lower triangular L:
    //   y[j] = b[j]   (no division by diagonal)
    //   b[i] -= L[i,j] * y[j]  for i > j in column j
    // ------------------------------------------------------------------
    for j in 0..n {
        let l_start = sym.col_ptr[j];
        let l_end   = sym.col_ptr[j + 1];
        let yj      = u[j]; // y[j] = b[j] (unit diagonal)

        for pos in (l_start + 1)..l_end {
            let i = sym.row_idx[pos];
            u[i] -= num.l_values[pos] * yj;
        }
    }

    // ------------------------------------------------------------------
    // Step 3 — diagonal solve: z[j] = y[j] / D[j]  (in-place)
    // ------------------------------------------------------------------
    for j in 0..n {
        u[j] /= num.d_values[j];
    }

    // ------------------------------------------------------------------
    // Step 4 — backward substitution: Lᵀ x = z  (unit diagonal, in-place)
    //
    // For unit upper triangular Lᵀ (reading column j of L right-to-left):
    //   x[j] = z[j] - Σ_{i>j} L[i,j] * x[i]   (no diagonal division)
    // ------------------------------------------------------------------
    for j in (0..n).rev() {
        let l_start = sym.col_ptr[j];
        let l_end   = sym.col_ptr[j + 1];

        for pos in (l_start + 1)..l_end {
            let i = sym.row_idx[pos];
            u[j] -= num.l_values[pos] * u[i];
        }
        // Unit diagonal: no division by L[j,j].
    }

    // ------------------------------------------------------------------
    // Step 5 — unpermute: u_final[perm[i]] = x_perm[i]
    // ------------------------------------------------------------------
    x_perm.copy_from_slice(u); // Replaces `let x_perm = u.to_vec();`
    
    for i in 0..n {
        u[perm.old_index(i)] = x_perm[i];
    }

    Ok(())
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::{CooBuilder, SymCsrMatrix};
    use crate::ordering::Ordering;

    // ---- helpers ----

    fn tridiag(n: usize) -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    /// Symmetric indefinite tridiagonal: diagonal = +2 except d[k] = -2.
    fn indefinite_tridiag(n: usize, k: usize) -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n {
            let d = if i == k { -2.0 } else { 2.0 };
            coo.add(i, i, d);
        }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    /// Residual check: assert ‖Ku - f‖_∞ < tol.
    fn check_residual(k: &SymCsrMatrix<f64>, f: &[f64], u: &[f64]) {
        let ku = k.matvec(u).unwrap();
        for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
            let err = (kui - fi).abs();
            assert!(
                err < 1e-9,
                "residual[{i}] = {err:.2e}  (Ku={kui:.8}, f={fi:.8})"
            );
        }
    }

    // ---- SPD matrices (LDLᵀ must agree with Cholesky) ----

    #[test]
    fn spd_tridiag_3_residual() {
        let k = tridiag(3);
        let f = vec![1.0, 0.0, 1.0];
        let mut u = vec![0.0; 3];
        let mut solver = LdltSolver::new();
        solver.analyze(&k).unwrap();
        solver.factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn spd_tridiag_10_residual() {
        let k = tridiag(10);
        let f: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let mut u = vec![0.0; 10];
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn spd_diagonal_residual() {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 1.0); coo.add(1, 1, 4.0); coo.add(2, 2, 9.0);
        let k = coo.build_sym().unwrap();
        let f = vec![1.0, 4.0, 9.0];  // u = [1,1,1]
        let mut u = vec![0.0; 3];
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        for (i, &ui) in u.iter().enumerate() {
            assert!((ui - 1.0).abs() < 1e-12, "u[{i}] = {ui}");
        }
    }

    #[test]
    fn spd_dense_3_residual() {
        // K = [[4,1,1],[1,4,1],[1,1,4]] — SPD
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 4.0); coo.add(0, 1, 1.0); coo.add(0, 2, 1.0);
        coo.add(1, 1, 4.0); coo.add(1, 2, 1.0);
        coo.add(2, 2, 4.0);
        let k = coo.build_sym().unwrap();
        let f = vec![6.0, 6.0, 6.0];
        let mut u = vec![0.0; 3];
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    // ---- Indefinite matrices (the key advantage over Cholesky) ----

    /// 2×2 indefinite: K = [[-4,0],[0,4]].
    /// Cholesky fails; LDLᵀ must succeed and produce correct u.
    #[test]
    fn indefinite_2x2_residual() {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, -4.0);
        coo.add(1, 1,  4.0);
        let k = coo.build_sym().unwrap();
        // K u = f  →  u = [-0.25, 0.25] for f = [1, 1]
        let f = vec![1.0, 1.0];
        let mut u = vec![0.0; 2];
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
        assert!((u[0] - (-0.25)).abs() < 1e-12, "u[0] = {}", u[0]);
        assert!((u[1] - 0.25).abs() < 1e-12, "u[1] = {}", u[1]);
    }

    #[test]
    fn indefinite_tridiag_n5_residual() {
        let k = indefinite_tridiag(5, 2); // d[2] = -2
        let f = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let mut u = vec![0.0; 5];
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn indefinite_tridiag_n10_residual() {
        let k = indefinite_tridiag(10, 5);
        let f: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let mut u = vec![0.0; 10];
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    /// All-negative diagonal — fully negative definite.
    #[test]
    fn fully_negative_definite_residual() {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, -4.0); coo.add(1, 1, -4.0); coo.add(2, 2, -4.0);
        coo.add(0, 1, 1.0); coo.add(1, 2, 1.0);
        let k = coo.build_sym().unwrap();
        let f = vec![1.0, 0.0, 1.0];
        let mut u = vec![0.0; 3];
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    // ---- negative_pivots() stability indicator ----

    #[test]
    fn negative_pivots_spd_is_zero() {
        let k = tridiag(5);
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        assert_eq!(solver.negative_pivots(), Some(0));
    }

    #[test]
    fn negative_pivots_before_factorize_is_none() {
        let k = tridiag(5);
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze(&k).unwrap();
        assert_eq!(solver.negative_pivots(), None);
    }

    #[test]
    fn negative_pivots_indefinite_nonzero() {
        let k = indefinite_tridiag(5, 2);
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        // The indefinite matrix must have at least one negative pivot.
        assert!(solver.negative_pivots().unwrap() > 0);
    }

    // ---- ordering variants ----

    #[test]
    fn rcm_ordering_correct() {
        let k = tridiag(20);
        let f: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut u = vec![0.0; 20];
        let mut solver = LdltSolver::with_rcm();
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn natural_ordering_correct() {
        let k = tridiag(10);
        let f = vec![1.0; 10];
        let mut u = vec![0.0; 10];
        let mut solver = LdltSolver::<f64>::new();
        solver.set_ordering(Ordering::Natural);
        solver.analyze_and_factorize(&k).unwrap();
        solver.solve(&f, &mut u).unwrap();
        check_residual(&k, &f, &u);
    }

    #[test]
    fn set_ordering_invalidates_analysis() {
        let k = tridiag(4);
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze(&k).unwrap();
        assert!(solver.is_analyzed());
        solver.set_ordering(Ordering::Rcm);
        assert!(!solver.is_analyzed());
        assert!(!solver.is_factorized());
    }

    // ---- reuse of symbolic phase ----

    #[test]
    fn refactorize_reuses_symbolic() {
        // Two matrices with the same sparsity pattern but different values.
        let k1 = tridiag(6);
        let mut coo2 = CooBuilder::new(6, 6);
        for i in 0..6       { coo2.add(i, i, 3.0); }
        for i in 0..5       { coo2.add(i, i + 1, -1.0); }
        let k2 = coo2.build_sym().unwrap();

        let f = vec![1.0; 6];
        let mut solver = LdltSolver::new();
        solver.analyze(&k1).unwrap();

        let mut u1 = vec![0.0; 6];
        solver.factorize(&k1).unwrap();
        solver.solve(&f, &mut u1).unwrap();
        check_residual(&k1, &f, &u1);

        let mut u2 = vec![0.0; 6];
        solver.factorize(&k2).unwrap(); // reuses symbolic
        solver.solve(&f, &mut u2).unwrap();
        check_residual(&k2, &f, &u2);
    }

    // ---- error ordering ----

    #[test]
    fn factorize_before_analyze_errors() {
        let k = tridiag(3);
        let mut solver = LdltSolver::<f64>::new();
        assert!(matches!(
            solver.factorize(&k).unwrap_err(),
            SolverError::NotAnalyzed
        ));
    }

    #[test]
    fn solve_before_factorize_errors() {
        let k = tridiag(3);
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze(&k).unwrap();
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solver.solve(&[1.0, 0.0, 0.0], &mut u).unwrap_err(),
            SolverError::NotFactorized
        ));
    }

    #[test]
    fn singular_matrix_errors() {
        // K = [[1,1],[1,1]] — singular (D[1] = 0)
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, 1.0); coo.add(0, 1, 1.0);
        coo.add(1, 1, 1.0);
        let k = coo.build_sym().unwrap();
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze(&k).unwrap();
        assert!(matches!(
            solver.factorize(&k).unwrap_err(),
            SolverError::NotPositiveDefinite { .. }
        ));
    }

    #[test]
    fn rhs_size_mismatch_f() {
        let k = tridiag(3);
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solver.solve(&[1.0, 2.0], &mut u).unwrap_err(),
            SolverError::RhsSizeMismatch { expected: 3, got: 2 }
        ));
    }

    #[test]
    fn rhs_size_mismatch_u() {
        let k = tridiag(3);
        let mut solver = LdltSolver::<f64>::new();
        solver.analyze_and_factorize(&k).unwrap();
        let mut u = vec![0.0; 5];
        assert!(matches!(
            solver.solve(&[1.0, 2.0, 3.0], &mut u).unwrap_err(),
            SolverError::RhsSizeMismatch { expected: 3, got: 5 }
        ));
    }

    // ---- state helpers ----

    #[test]
    fn is_analyzed_and_is_factorized() {
        let k = tridiag(4);
        let mut solver = LdltSolver::<f64>::new();
        assert!(!solver.is_analyzed());
        assert!(!solver.is_factorized());

        solver.analyze(&k).unwrap();
        assert!(solver.is_analyzed());
        assert!(!solver.is_factorized());

        solver.factorize(&k).unwrap();
        assert!(solver.is_analyzed());
        assert!(solver.is_factorized());
    }

    #[test]
    fn pivots_accessor() {
        let k = tridiag(4);
        let mut solver = LdltSolver::<f64>::new();
        assert!(solver.pivots().is_none());
        solver.analyze_and_factorize(&k).unwrap();
        let pivots = solver.pivots().unwrap();
        assert_eq!(pivots.len(), 4);
        // All pivots should be positive for an SPD matrix.
        for &d in pivots {
            assert!(d > 0.0, "pivot = {d}");
        }
    }
}