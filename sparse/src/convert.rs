//! Conversions between the three sparse matrix formats.
//!
//! | From             | To           | Function                        |
//! |------------------|--------------|---------------------------------|
//! | [`CsrMatrix`]    | [`CscMatrix`]| [`csr_to_csc`]                  |
//! | [`CscMatrix`]    | [`CsrMatrix`]| [`csc_to_csr`]                  |
//! | [`SymCsrMatrix`] | [`CscMatrix`]| [`sym_to_csc`]                  |
//! | [`CsrMatrix`]    | [`SymCsrMatrix`] | [`csr_to_sym`]              |
//!
//! All conversions are O(nnz) using the standard two-pass algorithm:
//! 1. Count entries per destination row/column → build `ptr` array
//! 2. Fill entries using the `ptr` as insertion cursors

use crate::{SparseMatrix, CsrMatrix, SymCsrMatrix, CscMatrix};
use crate::error::{SparseError, Result};

// -----------------------------------------------------------------
// CsrMatrix → CscMatrix
// -----------------------------------------------------------------

/// Convert a general CSR matrix to CSC format.
///
/// This is the standard sparse matrix transpose algorithm: O(nnz).
pub fn csr_to_csc(csr: &CsrMatrix) -> CscMatrix {
    let nrows = csr.nrows;
    let ncols = csr.ncols;
    let nnz   = csr.nnz();

    // Pass 1: count entries per column
    let mut col_count = vec![0usize; ncols];
    for &c in &csr.col_idx {
        col_count[c] += 1;
    }

    // Build col_ptr from counts
    let mut col_ptr = Vec::with_capacity(ncols + 1);
    col_ptr.push(0usize);
    for &cnt in &col_count {
        col_ptr.push(col_ptr.last().unwrap() + cnt);
    }

    // Pass 2: fill row_idx and values
    let mut row_idx = vec![0usize; nnz];
    let mut values  = vec![0.0_f64; nnz];
    let mut cursor  = col_ptr[..ncols].to_vec(); // insertion cursors

    for row in 0..nrows {
        let start = csr.row_ptr[row];
        let end   = csr.row_ptr[row + 1];
        for idx in start..end {
            let col = csr.col_idx[idx];
            let pos = cursor[col];
            row_idx[pos] = row;
            values[pos]  = csr.values[idx];
            cursor[col] += 1;
        }
    }

    // row_idx within each column is already sorted because we iterated rows
    // in ascending order
    CscMatrix::from_raw(nrows, ncols, col_ptr, row_idx, values)
}

// -----------------------------------------------------------------
// CscMatrix → CsrMatrix
// -----------------------------------------------------------------

/// Convert a CSC matrix to CSR format.
pub fn csc_to_csr(csc: &CscMatrix) -> CsrMatrix {
    let nrows = csc.nrows;
    let ncols = csc.ncols;
    let nnz   = csc.nnz();

    // Pass 1: count entries per row
    let mut row_count = vec![0usize; nrows];
    for &r in &csc.row_idx {
        row_count[r] += 1;
    }

    // Build row_ptr
    let mut row_ptr = Vec::with_capacity(nrows + 1);
    row_ptr.push(0usize);
    for &cnt in &row_count {
        row_ptr.push(row_ptr.last().unwrap() + cnt);
    }

    // Pass 2: fill col_idx and values
    let mut col_idx = vec![0usize; nnz];
    let mut values  = vec![0.0_f64; nnz];
    let mut cursor  = row_ptr[..nrows].to_vec();

    for col in 0..ncols {
        let start = csc.col_ptr[col];
        let end   = csc.col_ptr[col + 1];
        for idx in start..end {
            let row = csc.row_idx[idx];
            let pos = cursor[row];
            col_idx[pos] = col;
            values[pos]  = csc.values[idx];
            cursor[row] += 1;
        }
    }

    // col_idx within each row may not be sorted because we iterated columns,
    // but that's what from_raw is for — it's an internal constructor that
    // trusts the caller; here we can guarantee sorted order because columns
    // are iterated in ascending order per row.
    CsrMatrix::from_raw(nrows, ncols, row_ptr, col_idx, values)
}

// -----------------------------------------------------------------
// SymCsrMatrix → CscMatrix  (expand upper triangle to full)
// -----------------------------------------------------------------

/// Convert a symmetric upper-triangle CSR matrix to a full CSC matrix.
///
/// The resulting CSC matrix stores both triangles explicitly.
/// This is what you pass to the solver after applying BCs.
pub fn sym_to_csc(sym: &SymCsrMatrix) -> CscMatrix {
    let n   = sym.n;
    let nnz_upper = sym.nnz();

    // Count entries in the full matrix: each off-diagonal entry appears twice,
    // each diagonal entry once.
    let mut col_count = vec![0usize; n];
    for (row, col, _) in sym.iter_upper() {
        col_count[col] += 1;
        if col != row {
            col_count[row] += 1; // mirror in lower triangle
        }
    }

    let nnz_full: usize = col_count.iter().sum();

    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0usize);
    for &cnt in &col_count {
        col_ptr.push(col_ptr.last().unwrap() + cnt);
    }

    let mut row_idx = vec![0usize; nnz_full];
    let mut values  = vec![0.0_f64; nnz_full];
    let mut cursor  = col_ptr[..n].to_vec();

    // Insert upper triangle and its mirror
    for (row, col, val) in sym.iter_upper() {
        // upper entry (row, col): goes into CSC column `col`
        let pos = cursor[col];
        row_idx[pos] = row;
        values[pos]  = val;
        cursor[col] += 1;

        if col != row {
            // mirror entry (col, row): goes into CSC column `row`
            let pos2 = cursor[row];
            row_idx[pos2] = col;
            values[pos2]  = val;
            cursor[row]  += 1;
        }
    }

    // Row indices within each column are not yet sorted — sort them
    for col in 0..n {
        let start = col_ptr[col];
        let end   = col_ptr[col + 1];
        // sort (row_idx, values) together by row_idx
        let slice_len = end - start;
        let mut pairs: Vec<(usize, f64)> = row_idx[start..end]
            .iter()
            .zip(&values[start..end])
            .map(|(&r, &v)| (r, v))
            .collect();
        pairs.sort_unstable_by_key(|&(r, _)| r);
        for (k, (r, v)) in pairs.into_iter().enumerate() {
            row_idx[start + k] = r;
            values[start + k]  = v;
        }
        let _ = slice_len;
    }

    let _ = nnz_upper;
    CscMatrix::from_raw(n, n, col_ptr, row_idx, values)
}

// -----------------------------------------------------------------
// CsrMatrix → SymCsrMatrix  (extract upper triangle)
// -----------------------------------------------------------------

/// Extract the upper triangle of a CSR matrix and return a `SymCsrMatrix`.
///
/// Only entries with `col >= row` are kept.  If the input matrix is not
/// actually symmetric the lower-triangle entries are silently discarded —
/// the result represents only the upper-triangle values.
///
/// # Errors
/// - [`SparseError::NotSquare`] if the matrix is not square
pub fn csr_to_sym(csr: &CsrMatrix) -> Result<SymCsrMatrix> {
    if csr.nrows != csr.ncols {
        return Err(SparseError::NotSquare { nrows: csr.nrows, ncols: csr.ncols });
    }
    let n = csr.nrows;

    // Build upper-triangle pattern
    let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (row, col, _) in csr.iter_nonzeros() {
        if col >= row {
            pattern[row].push(col);
        }
    }
    // pattern rows are already sorted (CSR row_iter is sorted) and unique

    let mut sym = SymCsrMatrix::from_pattern(n, &pattern)?;

    // Fill values
    for (row, col, val) in csr.iter_nonzeros() {
        if col >= row {
            sym.add_value(row, col, val)?;
        }
    }

    Ok(sym)
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseMatrix;

    fn sym_tridiag() -> SymCsrMatrix {
        // [ 4 -1  0]
        // [-1  4 -1]
        // [ 0 -1  4]
        let pattern = vec![vec![0usize, 1], vec![1, 2], vec![2]];
        let mut m = SymCsrMatrix::from_pattern(3, &pattern).unwrap();
        m.set_value(0, 0,  4.0).unwrap();
        m.set_value(0, 1, -1.0).unwrap();
        m.set_value(1, 1,  4.0).unwrap();
        m.set_value(1, 2, -1.0).unwrap();
        m.set_value(2, 2,  4.0).unwrap();
        m
    }

    fn full_csr() -> CsrMatrix {
        // [1 0 2]
        // [0 3 4]
        // [0 0 5]
        let pattern = vec![vec![0usize, 2], vec![1, 2], vec![2]];
        let mut m = CsrMatrix::from_pattern(3, 3, &pattern).unwrap();
        m.add_value(0, 0, 1.0).unwrap();
        m.add_value(0, 2, 2.0).unwrap();
        m.add_value(1, 1, 3.0).unwrap();
        m.add_value(1, 2, 4.0).unwrap();
        m.add_value(2, 2, 5.0).unwrap();
        m
    }

    // --- csr_to_csc ---

    #[test]
    fn csr_to_csc_preserves_values() {
        let csr = full_csr();
        let csc = csr_to_csc(&csr);
        csc.validate().unwrap();
        assert_eq!(csc.nrows, 3);
        assert_eq!(csc.ncols, 3);
        assert_eq!(csc.nnz(), csr.nnz());

        // spot-check values through get()
        assert_eq!(csc.get(0, 0).unwrap(), 1.0);
        assert_eq!(csc.get(0, 2).unwrap(), 2.0);
        assert_eq!(csc.get(1, 1).unwrap(), 3.0);
        assert_eq!(csc.get(2, 2).unwrap(), 5.0);
        assert_eq!(csc.get(0, 1).unwrap(), 0.0);
    }

    #[test]
    fn csr_to_csc_matvec_agrees() {
        let csr = full_csr();
        let csc = csr_to_csc(&csr);
        let x = vec![1.0_f64, 2.0, 3.0];
        assert_eq!(csr.matvec(&x).unwrap(), csc.matvec(&x).unwrap());
    }

    // --- csc_to_csr ---

    #[test]
    fn csc_to_csr_roundtrip() {
        let csr_orig = full_csr();
        let csc      = csr_to_csc(&csr_orig);
        let csr_back = csc_to_csr(&csc);
        csr_back.validate().unwrap();
        // matvec must agree
        let x = vec![1.0_f64, 2.0, 3.0];
        assert_eq!(csr_orig.matvec(&x).unwrap(), csr_back.matvec(&x).unwrap());
    }

    // --- sym_to_csc ---

    #[test]
    fn sym_to_csc_expands_both_triangles() {
        let sym = sym_tridiag();
        let csc = sym_to_csc(&sym);
        csc.validate().unwrap();

        // Full matrix has 7 entries: 3 diagonal + 4 off-diagonal
        assert_eq!(csc.nnz(), 7);

        // symmetry: (0,1) == (1,0) == -1
        assert_eq!(csc.get(0, 1).unwrap(), -1.0);
        assert_eq!(csc.get(1, 0).unwrap(), -1.0);
    }

    #[test]
    fn sym_to_csc_matvec_agrees_with_sym_matvec() {
        let sym = sym_tridiag();
        let csc = sym_to_csc(&sym);
        let x   = vec![1.0_f64, 2.0, 3.0];

        let y_sym: Vec<f64> = sym.matvec(&x).unwrap();
        let y_csc: Vec<f64> = csc.matvec(&x).unwrap();

        for (a, b) in y_sym.iter().zip(y_csc.iter()) {
            assert!((a - b).abs() < 1e-14, "sym={a} csc={b}");
        }
    }

    // --- csr_to_sym ---

    #[test]
    fn csr_to_sym_extracts_upper() {
        // Build a full symmetric matrix in CSR then extract upper triangle
        let pattern = vec![vec![0usize,1], vec![0,1,2], vec![1,2]];
        let mut csr = CsrMatrix::from_pattern(3, 3, &pattern).unwrap();
        csr.set_value(0, 0,  4.0).unwrap(); csr.set_value(0, 1, -1.0).unwrap();
        csr.set_value(1, 0, -1.0).unwrap(); csr.set_value(1, 1,  4.0).unwrap();
        csr.set_value(1, 2, -1.0).unwrap(); csr.set_value(2, 1, -1.0).unwrap();
        csr.set_value(2, 2,  4.0).unwrap();

        let sym = csr_to_sym(&csr).unwrap();
        sym.validate().unwrap();

        assert_eq!(sym.get(0, 0).unwrap(),  4.0);
        assert_eq!(sym.get(0, 1).unwrap(), -1.0);
        assert_eq!(sym.get(1, 1).unwrap(),  4.0);
        assert_eq!(sym.get(1, 2).unwrap(), -1.0);
        assert_eq!(sym.get(2, 2).unwrap(),  4.0);
    }

    #[test]
    fn csr_to_sym_err_not_square() {
        let m = CsrMatrix::from_pattern(2, 3, &[vec![0usize], vec![0usize]]).unwrap();
        assert!(matches!(csr_to_sym(&m).unwrap_err(), SparseError::NotSquare { .. }));
    }

    // --- full roundtrip: sym → csc → values agree ---

    #[test]
    fn sym_csc_csr_roundtrip_matvec() {
        let sym = sym_tridiag();
        let csc = sym_to_csc(&sym);
        let csr = csc_to_csr(&csc);
        csr.validate().unwrap();

        let x = vec![1.0_f64, 2.0, 3.0];
        let y_sym = sym.matvec(&x).unwrap();
        let y_csr = csr.matvec(&x).unwrap();
        for (a, b) in y_sym.iter().zip(y_csr.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }
}