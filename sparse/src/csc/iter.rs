use crate::error::{SparseError, Result};
use crate::csc::CscMatrix;

/// Iterator over the non-zero `(row, value)` pairs of a single column,
/// in ascending row order.
pub struct ColIter<'a> {
    rows: &'a [usize],
    vals: &'a [f64],
}

impl<'a> Iterator for ColIter<'a> {
    type Item = (usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&row, rest_rows) = self.rows.split_first()?;
        let (&val, rest_vals) = self.vals.split_first().unwrap();
        self.rows = rest_rows;
        self.vals = rest_vals;
        Some((row, val))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rows.len(), Some(self.rows.len()))
    }
}

impl ExactSizeIterator for ColIter<'_> {}

impl CscMatrix {
    /// Iterate over `(row, value)` pairs in `col`, in row order.
    ///
    /// This is the natural access pattern for Cholesky factorization.
    ///
    /// # Errors
    /// - [`SparseError::ColOutOfRange`] if `col >= ncols`
    pub fn col_iter(&self, col: usize) -> Result<ColIter<'_>> {
        if col >= self.ncols {
            return Err(SparseError::ColOutOfRange { col, ncols: self.ncols });
        }
        let start = self.col_ptr[col];
        let end   = self.col_ptr[col + 1];
        Ok(ColIter {
            rows: &self.row_idx[start..end],
            vals: &self.values[start..end],
        })
    }

    /// Iterate over every non-zero entry as `(row, col, value)`,
    /// in column-major order.
    pub fn iter_nonzeros(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.ncols).flat_map(move |col| {
            let start = self.col_ptr[col];
            let end   = self.col_ptr[col + 1];
            self.row_idx[start..end]
                .iter()
                .zip(&self.values[start..end])
                .map(move |(&row, &val)| (row, col, val))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CscMatrix {
        let pattern = vec![vec![0usize], vec![1], vec![0, 1, 2]];
        let mut m = CscMatrix::from_pattern(3, 3, &pattern).unwrap();
        m.set_value(0, 0, 1.0).unwrap();
        m.set_value(1, 1, 3.0).unwrap();
        m.set_value(0, 2, 2.0).unwrap();
        m.set_value(1, 2, 4.0).unwrap();
        m.set_value(2, 2, 5.0).unwrap();
        m
    }

    #[test]
    fn col_iter_col2() {
        let v: Vec<_> = sample().col_iter(2).unwrap().collect();
        assert_eq!(v, vec![(0, 2.0), (1, 4.0), (2, 5.0)]);
    }

    #[test]
    fn col_iter_exact_size() {
        assert_eq!(sample().col_iter(2).unwrap().len(), 3);
    }

    #[test]
    fn col_iter_err_out_of_range() {
        assert!(matches!(
            sample().col_iter(99),
            Err(SparseError::ColOutOfRange { .. })
        ));
    }

    #[test]
    fn iter_nonzeros_col_major_order() {
        let entries: Vec<_> = sample().iter_nonzeros().collect();
        assert_eq!(entries.len(), 5);
        // first entry: col 0, row 0
        assert_eq!(entries[0], (0, 0, 1.0));
        // last entry: col 2, row 2
        assert_eq!(entries[4], (2, 2, 5.0));
    }
}