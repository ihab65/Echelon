use crate::error::{SparseError, Result};
use crate::csr::CsrMatrix;

/// Iterator over the non-zero `(col, value)` pairs of a single row,
/// in ascending column order.
pub struct RowIter<'a> {
    cols: &'a [usize],
    vals: &'a [f64],
}

impl<'a> Iterator for RowIter<'a> {
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

impl ExactSizeIterator for RowIter<'_> {}

impl CsrMatrix {
    /// Iterate over `(col, value)` pairs in `row`, in column order.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] if `row >= nrows`
    pub fn row_iter(&self, row: usize) -> Result<RowIter<'_>> {
        if row >= self.nrows {
            return Err(SparseError::RowOutOfRange { row, nrows: self.nrows });
        }
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        Ok(RowIter {
            cols: &self.col_idx[start..end],
            vals: &self.values[start..end],
        })
    }

    /// Iterate over every structurally non-zero entry as `(row, col, value)`,
    /// in row-major order.
    pub fn iter_nonzeros(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.nrows).flat_map(move |row| {
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

    fn sample() -> CsrMatrix {
        let pattern = vec![vec![0usize, 2], vec![1, 2], vec![2]];
        let mut m = CsrMatrix::from_pattern(3, 3, &pattern).unwrap();
        m.add_value(0, 0, 1.0).unwrap();
        m.add_value(0, 2, 2.0).unwrap();
        m.add_value(1, 1, 3.0).unwrap();
        m.add_value(1, 2, 4.0).unwrap();
        m.add_value(2, 2, 5.0).unwrap();
        m
    }

    #[test]
    fn row_iter_row0() {
        let v: Vec<_> = sample().row_iter(0).unwrap().collect();
        assert_eq!(v, vec![(0, 1.0), (2, 2.0)]);
    }

    #[test]
    fn row_iter_exact_size() {
        assert_eq!(sample().row_iter(0).unwrap().len(), 2);
    }

    #[test]
    fn row_iter_err_out_of_range() {
        assert!(matches!(
            sample().row_iter(99),
            Err(SparseError::RowOutOfRange { row: 99, .. })
        ));
    }

    #[test]
    fn iter_nonzeros_order_and_count() {
        let entries: Vec<_> = sample().iter_nonzeros().collect();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0], (0, 0, 1.0));
        assert_eq!(entries[1], (0, 2, 2.0));
        assert_eq!(entries[4], (2, 2, 5.0));
    }
}