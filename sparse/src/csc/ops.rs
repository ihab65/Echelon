use crate::SparseScalar;
use crate::error::{SparseError, Result};
use crate::csc::CscMatrix;
use crate::csr::ops::MatvecWorkspace;
use std::iter::Sum;
use std::ops::AddAssign;

impl<T: SparseScalar + Sum + AddAssign> CscMatrix<T> {
    /// Compute `y = A * x` — **no heap allocation**.
    ///
    /// In CSC, `Ax` is less natural than `Aᵀx`: we compute it as a sum
    /// of scaled columns — `y = Σⱼ x[j] * col_j(A)`.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`]
    pub fn matvec_into(&self, x: &[T], ws: &mut MatvecWorkspace<T>) -> Result<()> {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols, got: x.len(),
            });
        }
        if ws.buffer.len() != self.nrows {
            return Err(SparseError::DimensionMismatch {
                expected: self.nrows, got: ws.buffer.len(),
            });
        }
        ws.buffer.fill(T::zero());
        for col in 0..self.ncols {
            let start = self.col_ptr[col];
            let end   = self.col_ptr[col + 1];
            let xj = x[col];
            for idx in start..end {
                ws.buffer[self.row_idx[idx]] += self.values[idx] * xj;
            }
        }
        Ok(())
    }

    /// Compute `y = A * x`, returning an allocated `Vec<T>`.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `x.len() != self.ncols`.
    pub fn matvec(&self, x: &[T]) -> Result<Vec<T>> {
        let mut ws = MatvecWorkspace::new(self.nrows);
        self.matvec_into(x, &mut ws)?;
        Ok(ws.buffer)
    }

    /// Compute `y = Aᵀ * x` — the **natural** CSC operation, no allocation.
    ///
    /// Each output element `y[j] = col_j(A) · x`.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`]
    pub fn matvec_transpose_into(&self, x: &[T], ws: &mut MatvecWorkspace<T>) -> Result<()> {
        if x.len() != self.nrows {
            return Err(SparseError::DimensionMismatch {
                expected: self.nrows, got: x.len(),
            });
        }
        if ws.buffer.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols, got: ws.buffer.len(),
            });
        }
        ws.buffer.fill(T::zero());
        for col in 0..self.ncols {
            let start = self.col_ptr[col];
            let end   = self.col_ptr[col + 1];
            let dot: T = self.row_idx[start..end]
                .iter()
                .zip(&self.values[start..end])
                .map(|(&row, &val)| val * x[row])
                .sum();
            ws.buffer[col] = dot;
        }
        Ok(())
    }

    /// Compute `y = Aᵀ * x`, returning an allocated `Vec<T>`.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `x.len() != self.nrows`.
    pub fn matvec_transpose(&self, x: &[T]) -> Result<Vec<T>> {
        let mut ws = MatvecWorkspace::new(self.ncols);
        self.matvec_transpose_into(x, &mut ws)?;
        Ok(ws.buffer)
    }

    /// Scale every stored value by `alpha` in place.
    pub fn scale(&mut self, alpha: T) {
        self.values.iter_mut().for_each(|v| *v *= alpha);
    }

    /// Frobenius norm: `sqrt(Σ aᵢⱼ²)`.
    pub fn frobenius_norm(&self) -> T {
        self.values.iter().map(|&v| v * v).sum::<T>().scalar_sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CscMatrix<f64> {
        // [1 0 2]
        // [0 3 4]
        // [0 0 5]
        let pattern = vec![vec![0usize], vec![1], vec![0, 1, 2]];
        let mut m = CscMatrix::<f64>::from_pattern(3, 3, &pattern).unwrap();
        m.set_value(0, 0, 1.0).unwrap();
        m.set_value(1, 1, 3.0).unwrap();
        m.set_value(0, 2, 2.0).unwrap();
        m.set_value(1, 2, 4.0).unwrap();
        m.set_value(2, 2, 5.0).unwrap();
        m
    }

    #[test]
    fn matvec_correct() {
        // same as CsrMatrix: row0=7, row1=18, row2=15
        assert_eq!(sample().matvec(&[1.0, 2.0, 3.0]).unwrap(), vec![7.0, 18.0, 15.0]);
    }

    #[test]
    fn matvec_transpose_correct() {
        // Aᵀ: col0=1, col1=6, col2=25
        assert_eq!(sample().matvec_transpose(&[1.0, 2.0, 3.0]).unwrap(), vec![1.0, 6.0, 25.0]);
    }

    #[test]
    fn matvec_matches_csr() {
        use crate::csr::CsrMatrix;
        let pattern_csr = vec![vec![0usize, 2], vec![1, 2], vec![2]];
        let mut csr = CsrMatrix::from_pattern(3, 3, &pattern_csr).unwrap();
        csr.add_value(0, 0, 1.0).unwrap();
        csr.add_value(0, 2, 2.0).unwrap();
        csr.add_value(1, 1, 3.0).unwrap();
        csr.add_value(1, 2, 4.0).unwrap();
        csr.add_value(2, 2, 5.0).unwrap();

        let x = vec![1.0_f64, 2.0, 3.0];
        let y_csc = sample().matvec(&x).unwrap();
        let y_csr = csr.matvec(&x).unwrap();
        assert_eq!(y_csc, y_csr);
    }

    #[test]
    fn scale_doubles() {
        let mut m = sample();
        m.scale(2.0);
        assert_eq!(m.get(0, 0).unwrap(), 2.0);
    }

    #[test]
    fn frobenius_norm_correct() {
        // stored: 1,3,2,4,5 → sum sq = 1+9+4+16+25 = 55
        let expected = 55.0_f64.sqrt();
        assert!((sample().frobenius_norm() - expected).abs() < 1e-12);
    }
}