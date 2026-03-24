//! Triangular solve: `K u = f` via forward/backward substitution.
//!
//! The solver stored `K_perm = P K Pᵀ = L Lᵀ` where `P` is the RCM
//! permutation.  Given a right-hand side `f` in the original DOF order,
//! the solve proceeds in four steps:
//!
//! ```text
//! b[i]  = f[perm[i]]          1. permute RHS into reordered space
//! L y   = b                   2. forward substitution
//! Lᵀ x  = y                   3. backward substitution
//! u[perm[i]] = x[i]           4. unpermute solution to original order
//! ```
//!
//! ## Forward substitution — `L y = b`
//!
//! L is lower-triangular.  Column-oriented right-looking form:
//! ```text
//! for j = 0..n:
//!     y[j] = b[j] / L[j,j]
//!     b[i] -= L[i,j] * y[j]   for i in pattern(L[:,j]), i > j
//! ```
//! Each column of L is a contiguous slice in CSC storage — cache-friendly.
//!
//! ## Backward substitution — `Lᵀ x = y`
//!
//! Lᵀ is upper-triangular.  Row-oriented form using column j of L:
//! ```text
//! for j = n-1 downto 0:
//!     x[j] = (y[j] - Σ_{i>j} L[i,j] * x[i]) / L[j,j]
//! ```
//! When we reach column j (processing right-to-left), `x[i]` for all
//! `i > j` are already computed in the working buffer.  We read them
//! together with `L[i,j]` from column j of L.

use crate::error::{SolverError, Result};
use crate::ordering::Permutation;
use super::numeric::NumericCholesky;
use super::symbolic::SymbolicCholesky;

/// Solve `K u = f` using the factored `K_perm = L Lᵀ` and permutation `P`.
///
/// # Parameters
/// - `sym`  — symbolic factor (pattern of L)
/// - `num`  — numeric factor (values of L)
/// - `perm` — RCM permutation: `perm[new] = old`
/// - `f`    — right-hand side in **original** DOF order
/// - `u`    — output solution in **original** DOF order
///
/// # Errors
/// - [`SolverError::RhsSizeMismatch`] if `f.len() != n` or `u.len() != n`
pub fn solve(
    sym:  &SymbolicCholesky,
    num:  &NumericCholesky,
    perm: &Permutation,
    f:    &[f64],
    u:    &mut [f64],
) -> Result<()> {
    let n = num.n;
    if f.len() != n {
        return Err(SolverError::RhsSizeMismatch { expected: n, got: f.len() });
    }
    if u.len() != n {
        return Err(SolverError::RhsSizeMismatch { expected: n, got: u.len() });
    }
    debug_assert_eq!(sym.n, n);
    debug_assert_eq!(perm.len(), n);

    // Working buffer in the permuted (reordered) DOF space.
    // We use `u` itself as the buffer to avoid an extra allocation:
    //   pass 1: u ← b_perm   (permuted RHS)
    //   pass 2: u ← y        (result of forward solve, in-place)
    //   pass 3: u ← x_perm   (result of backward solve, in-place)
    // Then we scatter x_perm back to the original order.

    // ------------------------------------------------------------------
    // Step 1 — permute RHS: b[i] = f[perm[i]]
    // ------------------------------------------------------------------
    for i in 0..n {
        u[i] = f[perm.old_index(i)];
    }

    // ------------------------------------------------------------------
    // Step 2 — forward substitution: L y = b  (in-place in u)
    // ------------------------------------------------------------------
    for j in 0..n {
        let l_start = sym.col_ptr[j];
        let l_end   = sym.col_ptr[j + 1];

        // Diagonal is always stored first in each L column.
        let ljj = num.values[l_start];
        debug_assert!(ljj > 0.0, "L[{j},{j}] must be positive");

        u[j] /= ljj;
        let yj = u[j];

        // Update b[i] -= L[i,j] * y[j]  for i > j in column j of L.
        for pos in (l_start + 1)..l_end {
            let i = sym.row_idx[pos];
            u[i] -= num.values[pos] * yj;
        }
    }

    // ------------------------------------------------------------------
    // Step 3 — backward substitution: Lᵀ x = y  (in-place in u)
    //
    // Row-oriented, right-to-left.  When we process column j (from n-1
    // down to 0), u[i] for i > j already holds the final x[i].
    // We accumulate the dot product Σ_{i>j} L[i,j]*x[i] by reading
    // column j of L, then divide by L[j,j].
    // ------------------------------------------------------------------
    for j in (0..n).rev() {
        let l_start = sym.col_ptr[j];
        let l_end   = sym.col_ptr[j + 1];

        // Subtract the already-computed x[i] contributions:
        //   u[j] -= L[i,j] * x[i]  for i > j in column j of L.
        // x[i] is stored in u[i] (computed in a previous iteration of j).
        for pos in (l_start + 1)..l_end {
            let i = sym.row_idx[pos];
            u[j] -= num.values[pos] * u[i];
        }

        // Diagonal division: x[j] = (y[j] - accumulated) / L[j,j]
        let ljj = num.values[l_start];
        u[j] /= ljj;
    }

    // ------------------------------------------------------------------
    // Step 4 — unpermute: u_final[perm[i]] = x_perm[i]
    //
    // x_perm is in u[0..n].  We need to scatter it back to original order.
    // We need a temporary copy to avoid aliasing.
    // ------------------------------------------------------------------
    let x_perm: Vec<f64> = u.to_vec();
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
    use crate::cholesky::{symbolic::analyze, numeric::factorize};
    use crate::ordering::{Graph, rcm, Permutation};
    use sparse::{CooBuilder, SymCsrMatrix};
    use sparse::convert::sym_to_csc;

    // ---- solve helpers ----

    fn solve_direct(k: &SymCsrMatrix, f: &[f64]) -> Vec<f64> {
        let p   = Permutation::identity(k.n);
        let csc = sym_to_csc(k);
        let sym = analyze(k).unwrap();
        let num = factorize(&csc, &sym).unwrap();
        let mut u = vec![0.0; k.n];
        solve(&sym, &num, &p, f, &mut u).unwrap();
        u
    }

    fn solve_rcm(k: &SymCsrMatrix, f: &[f64]) -> Vec<f64> {
        let g    = Graph::from_sym(k);
        let perm = rcm(&g);
        let kp   = perm.permute_sym(k).unwrap();
        let csc  = sym_to_csc(&kp);
        let sym  = analyze(&kp).unwrap();
        let num  = factorize(&csc, &sym).unwrap();
        let mut u = vec![0.0; k.n];
        solve(&sym, &num, &perm, f, &mut u).unwrap();
        u
    }

    /// Check that K * u ≈ f (residual check).
    fn check_residual(k: &SymCsrMatrix, f: &[f64], u: &[f64]) {
        let ku = k.matvec(u).unwrap();
        for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
            let err = (kui - fi).abs();
            assert!(
                err < 1e-9,
                "residual[{i}] = {err:.2e}  (Ku[{i}]={kui:.6}, f[{i}]={fi:.6})"
            );
        }
    }

    // ---- 1×1 ----

    #[test]
    fn one_by_one() {
        let mut coo = CooBuilder::new(1, 1);
        coo.add(0, 0, 9.0);
        let k = coo.build_sym().unwrap();
        let u = solve_direct(&k, &[3.0]);
        assert!((u[0] - 1.0 / 3.0).abs() < 1e-13);
    }

    // ---- diagonal ----

    #[test]
    fn diagonal_3() {
        // K = diag(1, 4, 9),  f = [1, 4, 9]  →  u = [1, 1, 1]
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 1.0); coo.add(1, 1, 4.0); coo.add(2, 2, 9.0);
        let k = coo.build_sym().unwrap();
        let u = solve_direct(&k, &[1.0, 4.0, 9.0]);
        for (i, &ui) in u.iter().enumerate() {
            assert!((ui - 1.0).abs() < 1e-13, "u[{i}]={ui}");
        }
    }

    // ---- 2×2 with known solution ----

    #[test]
    fn two_by_two_known() {
        // K = [[4,-1],[-1,4]],  f = [3,3]  →  u = [1,1]
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, 4.0); coo.add(0, 1, -1.0);
        coo.add(1, 1, 4.0);
        let k = coo.build_sym().unwrap();
        let u = solve_direct(&k, &[3.0, 3.0]);
        assert!((u[0] - 1.0).abs() < 1e-12, "u[0]={}", u[0]);
        assert!((u[1] - 1.0).abs() < 1e-12, "u[1]={}", u[1]);
    }

    // ---- tridiagonal, identity permutation ----

    fn tridiag(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    #[test]
    fn tridiag_3_residual() {
        let k = tridiag(3);
        let f = vec![1.0, 0.0, 1.0];
        check_residual(&k, &f, &solve_direct(&k, &f));
    }

    #[test]
    fn tridiag_10_residual() {
        let k = tridiag(10);
        let f: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        check_residual(&k, &f, &solve_direct(&k, &f));
    }

    #[test]
    fn tridiag_50_residual() {
        let k = tridiag(50);
        let f = vec![1.0; 50];
        check_residual(&k, &f, &solve_direct(&k, &f));
    }

    // ---- tridiagonal, RCM permutation ----

    #[test]
    fn tridiag_3_rcm_residual() {
        let k = tridiag(3);
        let f = vec![1.0, 0.0, 1.0];
        check_residual(&k, &f, &solve_rcm(&k, &f));
    }

    #[test]
    fn tridiag_10_rcm_residual() {
        let k = tridiag(10);
        let f: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        check_residual(&k, &f, &solve_rcm(&k, &f));
    }

    #[test]
    fn tridiag_50_rcm_residual() {
        let k = tridiag(50);
        let f = vec![1.0; 50];
        check_residual(&k, &f, &solve_rcm(&k, &f));
    }

    #[test]
    fn tridiag_100_rcm_residual() {
        let k = tridiag(100);
        let f: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        check_residual(&k, &f, &solve_rcm(&k, &f));
    }

    // ---- dense 3×3 SPD ----

    #[test]
    fn dense_3_rcm_residual() {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 4.0); coo.add(0, 1, 1.0); coo.add(0, 2, 1.0);
        coo.add(1, 1, 4.0); coo.add(1, 2, 1.0);
        coo.add(2, 2, 4.0);
        let k = coo.build_sym().unwrap();
        let f = vec![6.0, 6.0, 6.0];
        check_residual(&k, &f, &solve_rcm(&k, &f));
    }

    // ---- results match between direct and RCM ----

    #[test]
    fn rcm_and_direct_agree() {
        let k = tridiag(15);
        let f: Vec<f64> = (1..=15).map(|i| i as f64).collect();
        let u_direct = solve_direct(&k, &f);
        let u_rcm    = solve_rcm(&k, &f);
        for (i, (a, b)) in u_direct.iter().zip(u_rcm.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "u_direct[{i}]={a:.8} u_rcm[{i}]={b:.8}"
            );
        }
    }

    // ---- error paths ----

    #[test]
    fn rhs_size_mismatch_f() {
        let k   = tridiag(3);
        let csc = sym_to_csc(&k);
        let sym = analyze(&k).unwrap();
        let num = factorize(&csc, &sym).unwrap();
        let p   = Permutation::identity(3);
        let mut u = vec![0.0; 3];
        assert!(matches!(
            solve(&sym, &num, &p, &[1.0, 2.0], &mut u).unwrap_err(),
            SolverError::RhsSizeMismatch { expected: 3, got: 2 }
        ));
    }

    #[test]
    fn rhs_size_mismatch_u() {
        let k   = tridiag(3);
        let csc = sym_to_csc(&k);
        let sym = analyze(&k).unwrap();
        let num = factorize(&csc, &sym).unwrap();
        let p   = Permutation::identity(3);
        let mut u = vec![0.0; 5];
        assert!(matches!(
            solve(&sym, &num, &p, &[1.0, 2.0, 3.0], &mut u).unwrap_err(),
            SolverError::RhsSizeMismatch { expected: 3, got: 5 }
        ));
    }
}