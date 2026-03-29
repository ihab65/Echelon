use crate::error::{SparseError, Result};
use crate::csr::CsrMatrix;
use crate::SparseScalar;
use std::iter::Sum;
use std::ops::AddAssign;

// -----------------------------------------------------------------
// Pre-allocated workspace
// -----------------------------------------------------------------

/// Reusable output buffer — eliminates per-call heap allocation inside
/// Newton-Raphson loops.
///
/// ```
/// # use sparse::{CsrMatrix, csr::ops::MatvecWorkspace};
/// # let k = CsrMatrix::from_pattern(2, 2, &[vec![0usize,1],vec![0,1]]).unwrap();
/// let mut ws = MatvecWorkspace::new(k.nrows);
/// k.matvec_into(&[1.0, 0.0], &mut ws).unwrap();
/// ```
pub struct MatvecWorkspace<T> {
    pub(crate) buffer: Vec<T>,
}

impl<T: SparseScalar> MatvecWorkspace<T> {
    /// Allocate a workspace for a matrix with `n` rows.
    pub fn new(n: usize) -> Self {
        Self { buffer: vec![T::zero(); n] }
    }

    /// Resize in-place (avoids reallocation when model size is unchanged).
    pub fn resize(&mut self, n: usize) {
        self.buffer.resize(n, T::zero());
    }

    /// Read the result after a `matvec_into` call.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.buffer
    }
}

// -----------------------------------------------------------------
// Matrix-vector products
// -----------------------------------------------------------------

impl<T: SparseScalar + Sum + AddAssign> CsrMatrix<T> {
    /// Compute `y = A * x`, writing into `ws` with **no heap allocation**.
    ///
    /// Preferred over [`matvec`] in any hot path.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `x.len() != ncols` or
    ///   `ws.buffer.len() != nrows`
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
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            // accumulate into a register — one write to ws.buffer per row
            let acc: T = self.col_idx[start..end]
                .iter()
                .zip(&self.values[start..end])
                .map(|(&col, &val)| val * x[col])
                .sum();
            ws.buffer[row] = acc;
        }
        Ok(())
    }

    /// Compute `y = A * x`, returning an allocated `Vec<f64>`.
    ///
    /// Convenience wrapper — use [`matvec_into`] in performance-critical paths.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `x.len() != ncols`
    pub fn matvec(&self, x: &[T]) -> Result<Vec<T>> {
        let mut ws = MatvecWorkspace::new(self.nrows);
        self.matvec_into(x, &mut ws)?;
        Ok(ws.buffer)
    }

    /// Compute `y = Aᵀ * x`, writing into `ws` with no heap allocation.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if sizes don't match
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
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            let xr = x[row];
            for idx in start..end {
                ws.buffer[self.col_idx[idx]] += self.values[idx] * xr;
            }
        }
        Ok(())
    }

    /// Compute `y = Aᵀ * x`, returning an allocated `Vec<f64>`.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `x.len() != nrows`
    pub fn matvec_transpose(&self, x: &[T]) -> Result<Vec<T>> {
        let mut ws = MatvecWorkspace::new(self.ncols);
        self.matvec_transpose_into(x, &mut ws)?;
        Ok(ws.buffer)
    }

    // -----------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------

    /// Scale every stored value by `alpha` in place.
    pub fn scale(&mut self, alpha: T) {
        self.values.iter_mut().for_each(|v| *v *= alpha);
    }

    /// Frobenius norm: `sqrt(Σ aᵢⱼ²)` over structurally non-zero entries.
    pub fn frobenius_norm(&self) -> T {
        self.values.iter().map(|&v| v * v).sum::<T>().sqrt()
    }

    /// Return `true` if `|K[i,j] - K[j,i]| ≤ tol` for every stored entry.
    ///
    /// # Errors
    /// - [`SparseError::NotSquare`]
    pub fn is_symmetric(&self, tol: T) -> Result<bool> {
        if self.nrows != self.ncols {
            return Err(SparseError::NotSquare { nrows: self.nrows, ncols: self.ncols });
        }
        for (row, col, val) in self.iter_nonzeros() {
            let mirror = self.find_idx(col, row).map_or(T::zero(), |i| self.values[i]);
            if (val - mirror).abs() > tol {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upper() -> CsrMatrix<f64> {
        let pattern = vec![vec![0usize, 2], vec![1, 2], vec![2]];
        let mut m = CsrMatrix::from_pattern(3, 3, &pattern).unwrap();
        m.add_value(0, 0, 1.0).unwrap();
        m.add_value(0, 2, 2.0).unwrap();
        m.add_value(1, 1, 3.0).unwrap();
        m.add_value(1, 2, 4.0).unwrap();
        m.add_value(2, 2, 5.0).unwrap();
        m
    }

    fn sym() -> CsrMatrix<f64> {
        let pattern = vec![vec![0usize,1], vec![0,1,2], vec![1,2]];
        let mut m = CsrMatrix::from_pattern(3, 3, &pattern).unwrap();
        m.set_value(0, 0,  4.0).unwrap(); m.set_value(0, 1, -1.0).unwrap();
        m.set_value(1, 0, -1.0).unwrap(); m.set_value(1, 1,  4.0).unwrap();
        m.set_value(1, 2, -1.0).unwrap(); m.set_value(2, 1, -1.0).unwrap();
        m.set_value(2, 2,  4.0).unwrap();
        m
    }

    #[test]
    fn matvec_correct() {
        // row0: 1*1 + 2*3 = 7   row1: 3*2 + 4*3 = 18   row2: 5*3 = 15
        assert_eq!(upper().matvec(&[1.0, 2.0, 3.0]).unwrap(), vec![7.0, 18.0, 15.0]);
    }

    #[test]
    fn matvec_dimension_err() {
        assert!(matches!(
            upper().matvec(&[1.0, 2.0]).unwrap_err(),
            SparseError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn matvec_into_same_result() {
        let m = upper();
        let mut ws = MatvecWorkspace::new(3);
        m.matvec_into(&[1.0, 2.0, 3.0], &mut ws).unwrap();
        assert_eq!(ws.as_slice(), &[7.0, 18.0, 15.0]);
    }

    #[test]
    fn matvec_into_reuse_gives_same_result() {
        let m = upper();
        let mut ws = MatvecWorkspace::new(3);
        for _ in 0..100 {
            m.matvec_into(&[1.0, 2.0, 3.0], &mut ws).unwrap();
        }
        assert_eq!(ws.as_slice(), &[7.0, 18.0, 15.0]);
    }

    #[test]
    fn matvec_transpose_correct() {
        // Aᵀ: col0=1  col1=6  col2=25
        assert_eq!(upper().matvec_transpose(&[1.0, 2.0, 3.0]).unwrap(), vec![1.0, 6.0, 25.0]);
    }

    #[test]
    fn is_symmetric_true() {
        assert!(sym().is_symmetric(1e-14).unwrap());
    }

    #[test]
    fn is_symmetric_false() {
        assert!(!upper().is_symmetric(1e-14).unwrap());
    }

    #[test]
    fn scale_doubles() {
        let mut m = upper();
        m.scale(2.0);
        assert_eq!(m.get(0, 0).unwrap(), 2.0);
        assert_eq!(m.get(2, 2).unwrap(), 10.0);
    }

    #[test]
    fn frobenius_norm_sym() {
        // entries: 4,-1,-1,4,-1,-1,4  → sum sq = 3*16+4 = 52
        let expected = 52.0_f64.sqrt();
        assert!((sym().frobenius_norm() - expected).abs() < 1e-12);
    }
}