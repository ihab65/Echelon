use crate::error::{SparseError, Result};
use crate::sym::SymCsrMatrix;

/// Iterator over the **logically** non-zero entries of one row of a
/// [`SymCsrMatrix`], including the mirrored lower-triangle entries.
///
/// For a row `r`, this yields:
/// 1. All entries `(col, val)` with `col >= r` from stored upper-triangle.
/// 2. All entries `(col, val)` with `col < r` by scanning other rows for
///    entries `(col, r)` — i.e. the mirrored lower triangle.
///
/// If you only need the stored (upper-triangle) entries, use
/// [`SymCsrMatrix::upper_row_iter`] instead.
pub struct SymRowIter<'a> {
    mat:     &'a SymCsrMatrix,
    row:     usize,
    // Phase 1: upper stored entries (col >= row)
    upper_cols: &'a [usize],
    upper_vals: &'a [f64],
    // Phase 2: lower mirrored entries (col < row), counted down
    lower_col:  usize, // current lower col to check (starts at 0)
}

impl<'a> Iterator for SymRowIter<'a> {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        // --- Phase 1: drain upper stored entries ---
        if let Some((&col, rest_cols)) = self.upper_cols.split_first() {
            let (&val, rest_vals) = self.upper_vals.split_first().unwrap();
            self.upper_cols = rest_cols;
            self.upper_vals = rest_vals;
            return Some((col, val));
        }

        // --- Phase 2: scan lower-triangle cols (col < self.row) ---
        while self.lower_col < self.row {
            let col = self.lower_col;
            self.lower_col += 1;
            // Look for entry (col, self.row) in the upper triangle of row `col`
            if let Some(idx) = self.mat.find_idx(col, self.row) {
                return Some((col, self.mat.values[idx]));
            }
        }

        None
    }
}

/// Iterator over only the **stored** (upper-triangle) entries of one row.
/// Yields `(col, value)` pairs with `col >= row`, in ascending column order.
pub struct UpperRowIter<'a> {
    cols: &'a [usize],
    vals: &'a [f64],
}

impl<'a> Iterator for UpperRowIter<'a> {
    type Item = (usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&col, rest_cols) = self.cols.split_first()?;
        let (&val, rest_vals) = self.vals.split_first().unwrap();
        self.cols = rest_cols;
        self.vals = rest_vals;
        Some((col, val))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.cols.len(), Some(self.cols.len()))
    }
}

impl ExactSizeIterator for UpperRowIter<'_> {}

impl SymCsrMatrix {
    /// Iterate over all logically non-zero entries in `row` (both triangles).
    ///
    /// Yields `(col, value)` pairs, **not** guaranteed to be in column order
    /// because upper and lower entries are emitted in two separate phases.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] if `row >= n`
    pub fn row_iter(&self, row: usize) -> Result<SymRowIter<'_>> {
        if row >= self.n {
            return Err(SparseError::RowOutOfRange { row, nrows: self.n });
        }
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        Ok(SymRowIter {
            mat: self,
            row,
            upper_cols: &self.col_idx[start..end],
            upper_vals: &self.values[start..end],
            lower_col: 0,
        })
    }

    /// Iterate over only the stored upper-triangle entries in `row`.
    ///
    /// Yields `(col, value)` with `col >= row`, in ascending column order.
    /// This is faster than [`row_iter`] and is what the solver uses.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] if `row >= n`
    pub fn upper_row_iter(&self, row: usize) -> Result<UpperRowIter<'_>> {
        if row >= self.n {
            return Err(SparseError::RowOutOfRange { row, nrows: self.n });
        }
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        Ok(UpperRowIter {
            cols: &self.col_idx[start..end],
            vals: &self.values[start..end],
        })
    }

    /// Iterate over every stored (upper-triangle + diagonal) entry.
    /// Yields `(row, col, value)` triples in row-major order.
    pub fn iter_upper(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.n).flat_map(move |row| {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            self.col_idx[start..end]
                .iter()
                .zip(&self.values[start..end])
                .map(move |(&col, &val)| (row, col, val))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tridiag() -> SymCsrMatrix {
        let pattern = vec![vec![0usize, 1], vec![1, 2], vec![2]];
        let mut m = SymCsrMatrix::from_pattern(3, &pattern).unwrap();
        m.set_value(0, 0,  4.0).unwrap();
        m.set_value(0, 1, -1.0).unwrap();
        m.set_value(1, 1,  4.0).unwrap();
        m.set_value(1, 2, -1.0).unwrap();
        m.set_value(2, 2,  4.0).unwrap();
        m
    }

    #[test]
    fn upper_row_iter_row0() {
        let m = tridiag();
        let v: Vec<_> = m.upper_row_iter(0).unwrap().collect();
        assert_eq!(v, vec![(0, 4.0), (1, -1.0)]);
    }

    #[test]
    fn upper_row_iter_row2() {
        let m = tridiag();
        let v: Vec<_> = m.upper_row_iter(2).unwrap().collect();
        assert_eq!(v, vec![(2, 4.0)]);
    }

    #[test]
    fn upper_row_iter_exact_size() {
        assert_eq!(tridiag().upper_row_iter(0).unwrap().len(), 2);
    }

    #[test]
    fn row_iter_row1_both_triangles() {
        // row 1 logically: (-1, 4, -1) at cols (0, 1, 2)
        let m = tridiag();
        let mut v: Vec<_> = m.row_iter(1).unwrap().collect();
        v.sort_by_key(|&(col, _)| col);
        assert_eq!(v, vec![(0, -1.0), (1, 4.0), (2, -1.0)]);
    }

    #[test]
    fn row_iter_err_out_of_range() {
        assert!(matches!(
            tridiag().row_iter(99),
            Err(SparseError::RowOutOfRange { .. })
        ));
    }

    #[test]
    fn iter_upper_count() {
        // 5 stored entries: (0,0),(0,1),(1,1),(1,2),(2,2)
        assert_eq!(tridiag().iter_upper().count(), 5);
    }
}