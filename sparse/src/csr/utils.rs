use crate::{SparseScalar, csr::CsrMatrix};

impl<T: SparseScalar> CsrMatrix<T> {
    /// Summary statistics for the matrix.  Cheap to call at any time.
    pub fn stats(&self) -> MatrixStats {
        let nnz = self.nnz();
        let total = self.nrows * self.ncols;
        let max_row_nnz = (0..self.nrows)
            .map(|r| self.row_ptr[r + 1] - self.row_ptr[r])
            .max()
            .unwrap_or(0);
        MatrixStats {
            nrows: self.nrows,
            ncols: self.ncols,
            nnz,
            density:     if total > 0 { nnz as f64 / total as f64 } else { 0.0 },
            max_row_nnz,
            avg_row_nnz: if self.nrows > 0 { nnz as f64 / self.nrows as f64 } else { 0.0 },
        }
    }

    /// Dense `Vec<Vec<f64>>` representation.
    ///
    /// **Debug builds only** — allocates `nrows × ncols` memory.
    /// Use for small matrices in tests; never call in production.
    #[cfg(debug_assertions)]
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let mut dense = vec![vec![T::zero(); self.ncols]; self.nrows];
        for (row, col, val) in self.iter_nonzeros() {
            dense[row][col] = val;
        }
        dense
    }
}

/// Summary statistics returned by [`CsrMatrix::stats`].
#[derive(Debug, Clone)]
pub struct MatrixStats {
    pub nrows:       usize,
    pub ncols:       usize,
    pub nnz:         usize,
    pub density:     f64,
    pub max_row_nnz: usize,
    pub avg_row_nnz: f64,
}

impl std::fmt::Display for MatrixStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}×{} sparse | nnz={} | density={:.4}% | \
             max_row_nnz={} | avg_row_nnz={:.1}",
            self.nrows, self.ncols, self.nnz,
            self.density * 100.0, self.max_row_nnz, self.avg_row_nnz,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CsrMatrix<f64> {
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
    fn stats_values() {
        let s = sample().stats();
        assert_eq!(s.nrows, 3);
        assert_eq!(s.nnz, 5);
        assert_eq!(s.max_row_nnz, 2);
        assert!((s.density - 5.0/9.0).abs() < 1e-12);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn to_dense_correct() {
        let d = sample().to_dense();
        assert_eq!(d[0], vec![1.0, 0.0, 2.0]);
        assert_eq!(d[1], vec![0.0, 3.0, 4.0]);
        assert_eq!(d[2], vec![0.0, 0.0, 5.0]);
    }
}