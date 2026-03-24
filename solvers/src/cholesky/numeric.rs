//! Numeric Cholesky factorization: compute the values of `L` from `K`
//! and the pre-computed symbolic pattern.
//!
//! ## Algorithm — left-looking column Cholesky
//!
//! For each column `j` (0-indexed):
//!
//! 1. **Scatter** the entries of column `j` of `K` (rows `≥ j`) into
//!    a dense work vector `w`.
//!
//! 2. **Left-looking update**: for every descendant `c` of `j` in the
//!    elimination tree (i.e., every `c < j` with `L[j,c] ≠ 0`):
//!    ```text
//!    w[i] -= L[j,c] * L[i,c]   for all i in pattern(L[:,c]) with i >= j
//!    ```
//!    The descendants are visited by a DFS on the children list.
//!
//! 3. **Diagonal**: `L[j,j] = sqrt(w[j])`.  Non‑positive ⇒ not SPD.
//!
//! 4. **Sub-diagonal**: `L[i,j] = w[i] / L[j,j]` for `i` in the column
//!    pattern of `L[:,j]` with `i > j`.
//!
//! 5. **Clear** all touched workspace entries.
//!
//! ## Dense workspace
//!
//! `w[0..n]` is the accumulator.  An `active[0..n]` boolean array tracks
//! which entries have been written this column; entries are cleared in
//! O(touched) per column rather than O(n).
//!
//! ## References
//! - Davis, T.A. (2006). *Direct Methods for Sparse Linear Systems*. §4.6.

use crate::error::{SolverError, Result};
use super::symbolic::SymbolicCholesky;
use sparse::CscMatrix;

// -----------------------------------------------------------------
// Public type
// -----------------------------------------------------------------

/// Values of the Cholesky factor `L`, stored in CSC format.
///
/// The sparsity pattern (`col_ptr` / `row_idx`) lives in the paired
/// [`SymbolicCholesky`]; `NumericCholesky` stores only the `f64` values.
/// Both are required together to perform the triangular solve.
#[derive(Debug)]
pub struct NumericCholesky {
    /// Non-zero values of `L`, indexed identically to `SymbolicCholesky::row_idx`.
    /// `values[col_ptr[j]..col_ptr[j+1]]` are the entries of column `j`.
    pub values: Vec<f64>,
    /// Dimension of the factored system.
    pub n: usize,
}

// -----------------------------------------------------------------
// Factorization
// -----------------------------------------------------------------

/// Compute the numeric Cholesky factorization `K = LLᵀ`.
///
/// # Arguments
/// * `k_csc` — the permuted, full matrix in CSC format produced by
///   `sym_to_csc(permute_sym(k))`.
/// * `sym`   — the symbolic factor computed from the same permuted matrix.
///
/// # Errors
/// - [`SolverError::NotPositiveDefinite`] if the matrix is not SPD.
pub fn factorize(k_csc: &CscMatrix, sym: &SymbolicCholesky) -> Result<NumericCholesky> {
    let n = sym.n;
    debug_assert_eq!(k_csc.nrows, n);
    debug_assert_eq!(k_csc.ncols, n);

    let nnz_l = sym.nnz_l();
    let mut lv = vec![0.0_f64; nnz_l];

    // Build children lists from the elimination tree.
    let mut children = vec![Vec::new(); n];
    for (c, &p) in sym.parent.iter().enumerate() {
        if p < n {
            children[p].push(c);
        }
    }
    // Sort children for deterministic order (not required for correctness).
    for ch in &mut children {
        ch.sort_unstable();
    }

    // Dense workspace.  `active[i]` is true iff w[i] has been written this
    // column.  We track touched indices so we can clear both in O(touched).
    let mut w = vec![0.0_f64; n];
    let mut active = vec![false; n];
    let mut touched: Vec<usize> = Vec::with_capacity(64);

    for j in 0..n {
        touched.clear();

        // ------------------------------------------------------------------
        // Step 1 — scatter column j of K (lower triangle: row >= j) into w.
        // ------------------------------------------------------------------
        {
            let k_start = k_csc.col_ptr()[j];
            let k_end = k_csc.col_ptr()[j + 1];
            let k_rows = k_csc.row_idx();
            let k_vals = k_csc.values();

            for idx in k_start..k_end {
                let row = k_rows[idx];
                if row >= j {
                    w[row] = k_vals[idx];
                    if !active[row] {
                        active[row] = true;
                        touched.push(row);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 2 — left-looking update.
        //
        // All columns c that are descendants of j (c < j and L[j,c] ≠ 0)
        // are visited by a DFS starting from the direct children of j.
        // For each such c, we subtract L[j,c] * L[:,c] from w.
        // ------------------------------------------------------------------
        {
            let mut stack = Vec::new();
            // Push direct children onto the stack (they will be processed
            // together with their descendants).
            for &c in &children[j] {
                stack.push(c);
            }
            while let Some(c) = stack.pop() {
                // Find the position of row j in column c.
                let col_start = sym.col_ptr[c];
                let col_end = sym.col_ptr[c + 1];
                let col_rows = &sym.row_idx[col_start..col_end];
                let local_j = match col_rows.binary_search(&j) {
                    Ok(pos) => pos,
                    Err(_) => continue, // should never happen for a descendant
                };
                let ljc = lv[col_start + local_j];

                // Update w[i] for all i >= j in column c.
                for pos in local_j..col_rows.len() {
                    let row = col_rows[pos];
                    w[row] -= lv[col_start + pos] * ljc;
                    if !active[row] {
                        active[row] = true;
                        touched.push(row);
                    }
                }

                // Push children of c (they are also descendants of j).
                for &gc in &children[c] {
                    stack.push(gc);
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 3 — diagonal: L[j,j] = sqrt(w[j]).
        // ------------------------------------------------------------------
        let wj = w[j];
        // Use a small tolerance to detect numerical zero (e.g., for singular matrices).
        if wj <= 1e-12 {
            // Clean workspace before returning the error.
            for &r in &touched {
                w[r] = 0.0;
                active[r] = false;
            }
            return Err(SolverError::NotPositiveDefinite { index: j, value: wj });
        }
        let ljj = wj.sqrt();

        // The diagonal is always the first entry in each L column.
        let l_col_start = sym.col_ptr[j];
        let l_col_end = sym.col_ptr[j + 1];
        debug_assert_eq!(
            sym.row_idx[l_col_start], j,
            "diagonal must be the first entry in L column {j}"
        );
        lv[l_col_start] = ljj;

        // ------------------------------------------------------------------
        // Step 4 — sub-diagonal: L[i,j] = w[i] / L[j,j].
        // ------------------------------------------------------------------
        for pos in (l_col_start + 1)..l_col_end {
            let row = sym.row_idx[pos];
            lv[pos] = w[row] / ljj;
        }

        // ------------------------------------------------------------------
        // Step 5 — clear workspace.
        // ------------------------------------------------------------------
        for &r in &touched {
            w[r] = 0.0;
            active[r] = false;
        }
    }

    Ok(NumericCholesky { values: lv, n })
}

// -----------------------------------------------------------------
// Tests (unchanged, included for completeness)
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cholesky::symbolic::analyze;
    use sparse::{CooBuilder, SymCsrMatrix};
    use sparse::convert::sym_to_csc;

    // ---- helpers ----

    /// Factorize a `SymCsrMatrix` directly (no RCM — unit test of numeric
    /// routine in isolation).
    fn factorize_direct(k: &SymCsrMatrix) -> (SymbolicCholesky, NumericCholesky) {
        let csc = sym_to_csc(k);
        let sym = analyze(k).unwrap();
        let num = factorize(&csc, &sym).unwrap();
        (sym, num)
    }

    fn tridiag(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    fn diagonal_mat(vals: &[f64]) -> SymCsrMatrix {
        let n = vals.len();
        let mut coo = CooBuilder::new(n, n);
        for (i, &v) in vals.iter().enumerate() { coo.add(i, i, v); }
        coo.build_sym().unwrap()
    }

    fn dense_spd_3() -> SymCsrMatrix {
        // K = [[4,1,1],[1,4,1],[1,1,4]] — strictly diagonally dominant → SPD
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 4.0); coo.add(0, 1, 1.0); coo.add(0, 2, 1.0);
        coo.add(1, 1, 4.0); coo.add(1, 2, 1.0);
        coo.add(2, 2, 4.0);
        coo.build_sym().unwrap()
    }

    // ---- LLᵀ reconstruction ----
    //
    // The gold-standard check: reconstruct K' = LLᵀ in dense form and
    // compare with K entry-by-entry.

    fn check_llt(k: &SymCsrMatrix) {
        let (sym, num) = factorize_direct(k);
        let n = sym.n;

        let mut llt = vec![vec![0.0f64; n]; n];

        // For each column j of L, accumulate its outer product contribution.
        for j in 0..n {
            let l_start = sym.col_ptr[j];
            let l_end   = sym.col_ptr[j + 1];

            for pos_i in l_start..l_end {
                let i   = sym.row_idx[pos_i];
                let lij = num.values[pos_i];
                for pos_r in l_start..=pos_i {
                    let r   = sym.row_idx[pos_r];
                    let lrj = num.values[pos_r];
                    llt[i][r] += lij * lrj;
                    if i != r {
                        llt[r][i] += lij * lrj;
                    }
                }
            }
        }

        for row in 0..n {
            for col in 0..n {
                let kval = k.get(row, col).unwrap();
                let diff = (llt[row][col] - kval).abs();
                assert!(
                    diff < 1e-10,
                    "LLᵀ[{row},{col}]={:.8} K[{row},{col}]={:.8}  diff={diff:.2e}",
                    llt[row][col], kval
                );
            }
        }
    }

    // ---- diagonal matrices ----

    #[test]
    fn diagonal_1x1() {
        let (sym, num) = factorize_direct(&diagonal_mat(&[4.0]));
        let diag = num.values[sym.col_ptr[0]];
        assert!((diag - 2.0).abs() < 1e-14, "L[0,0]={diag} expected 2");
    }

    #[test]
    fn diagonal_llt() {
        check_llt(&diagonal_mat(&[1.0, 4.0, 9.0]));
    }

    #[test]
    fn diagonal_values_are_sqrt() {
        let vals = [1.0_f64, 4.0, 9.0, 16.0];
        let (sym, num) = factorize_direct(&diagonal_mat(&vals));
        for (i, &v) in vals.iter().enumerate() {
            let diag     = num.values[sym.col_ptr[i]];
            let expected = v.sqrt();
            assert!(
                (diag - expected).abs() < 1e-13,
                "L[{i},{i}]={diag} expected {expected}"
            );
        }
    }

    // ---- tridiagonal ----

    #[test]
    fn tridiag_3_llt()  { check_llt(&tridiag(3));  }

    #[test]
    fn tridiag_5_llt()  { check_llt(&tridiag(5));  }

    #[test]
    fn tridiag_10_llt() { check_llt(&tridiag(10)); }

    #[test]
    fn tridiag_20_llt() { check_llt(&tridiag(20)); }

    // ---- dense SPD ----

    #[test]
    fn dense_3_llt() { check_llt(&dense_spd_3()); }

    // ---- not positive definite ----

    #[test]
    fn indefinite_returns_not_positive_definite() {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, -1.0);
        coo.add(1, 1,  1.0);
        let k   = coo.build_sym().unwrap();
        let csc = sym_to_csc(&k);
        let sym = analyze(&k).unwrap();
        assert!(matches!(
            factorize(&csc, &sym).unwrap_err(),
            SolverError::NotPositiveDefinite { index: 0, .. }
        ));
    }

    #[test]
    fn singular_returns_not_positive_definite() {
        // K = [[1,1],[1,1]] — singular
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, 1.0); coo.add(0, 1, 1.0);
        coo.add(1, 1, 1.0);
        let k   = coo.build_sym().unwrap();
        let csc = sym_to_csc(&k);
        let sym = analyze(&k).unwrap();
        assert!(matches!(
            factorize(&csc, &sym).unwrap_err(),
            SolverError::NotPositiveDefinite { .. }
        ));
    }

    // ---- metadata ----

    #[test]
    fn n_field_correct() {
        let (_, num) = factorize_direct(&tridiag(7));
        assert_eq!(num.n, 7);
    }

    #[test]
    fn values_len_matches_nnz_l() {
        let k = tridiag(8);
        let (sym, num) = factorize_direct(&k);
        assert_eq!(num.values.len(), sym.nnz_l());
    }
}