use crate::error::{SparseError, Result};
use crate::sym::SymCsrMatrix;
use crate::csr::ops::MatvecWorkspace;

impl SymCsrMatrix {
    /// Compute `y = A * x` exploiting symmetry — **no heap allocation**.
    ///
    /// Because only the upper triangle is stored, each off-diagonal entry
    /// `A[i,j]` must contribute to **both** `y[i]` and `y[j]`.
    /// This is the correct symmetric matvec: O(nnz) but slightly more complex
    /// than the general CSR version.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `x.len() != n` or
    ///   `ws.buffer.len() != n`
    pub fn matvec_into(&self, x: &[f64], ws: &mut MatvecWorkspace) -> Result<()> {
        if x.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n, got: x.len(),
            });
        }
        if ws.buffer.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n, got: ws.buffer.len(),
            });
        }
        ws.buffer.fill(0.0);

        for row in 0..self.n {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            for idx in start..end {
                let col = self.col_idx[idx];
                let val = self.values[idx];
                if col == row {
                    // diagonal: contributes once
                    ws.buffer[row] += val * x[col];
                } else {
                    // off-diagonal: contributes to both y[row] and y[col]
                    ws.buffer[row] += val * x[col];
                    ws.buffer[col] += val * x[row];
                }
            }
        }
        Ok(())
    }

    /// Compute `y = A * x`, returning an allocated `Vec<f64>`.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `x.len() != n`
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        let mut ws = MatvecWorkspace::new(self.n);
        self.matvec_into(x, &mut ws)?;
        Ok(ws.buffer)
    }

    /// Scale every stored value by `alpha` in place.
    ///
    /// Because each off-diagonal value is stored once but represents two
    /// entries in the full matrix, scaling by `alpha` correctly scales the
    /// full matrix — no double-counting needed.
    pub fn scale(&mut self, alpha: f64) {
        self.values.iter_mut().for_each(|v| *v *= alpha);
    }

    /// Frobenius norm of the **full** symmetric matrix: `sqrt(Σ aᵢⱼ²)`.
    ///
    /// Off-diagonal stored entries are counted twice (upper and lower).
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum_sq = 0.0_f64;
        for row in 0..self.n {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            for idx in start..end {
                let col = self.col_idx[idx];
                let v = self.values[idx];
                if col == row {
                    sum_sq += v * v;         // diagonal: once
                } else {
                    sum_sq += 2.0 * v * v;  // off-diagonal: upper + lower
                }
            }
        }
        sum_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tridiag() -> SymCsrMatrix {
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

    #[test]
    fn matvec_correct() {
        // A * [1, 1, 1]ᵀ = [3, 2, 3]ᵀ
        // row0: 4*1 + (-1)*1 = 3
        // row1: (-1)*1 + 4*1 + (-1)*1 = 2
        // row2: (-1)*1 + 4*1 = 3
        let y = tridiag().matvec(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(y, vec![3.0, 2.0, 3.0]);
    }

    #[test]
    fn matvec_matches_csr_expansion() {
        // Build the same matrix as a full CsrMatrix and compare matvec results
        use crate::csr::CsrMatrix;
        let pattern_full = vec![vec![0usize,1], vec![0,1,2], vec![1,2]];
        let mut full = CsrMatrix::from_pattern(3, 3, &pattern_full).unwrap();
        full.set_value(0, 0,  4.0).unwrap(); full.set_value(0, 1, -1.0).unwrap();
        full.set_value(1, 0, -1.0).unwrap(); full.set_value(1, 1,  4.0).unwrap();
        full.set_value(1, 2, -1.0).unwrap(); full.set_value(2, 1, -1.0).unwrap();
        full.set_value(2, 2,  4.0).unwrap();

        let x = vec![1.0_f64, 2.0, 3.0];
        let y_sym  = tridiag().matvec(&x).unwrap();
        let y_full = full.matvec(&x).unwrap();

        for (a, b) in y_sym.iter().zip(y_full.iter()) {
            assert!((a - b).abs() < 1e-14, "sym={a} full={b}");
        }
    }

    #[test]
    fn matvec_into_workspace_reuse() {
        let m = tridiag();
        let mut ws = MatvecWorkspace::new(3);
        for _ in 0..100 {
            m.matvec_into(&[1.0, 1.0, 1.0], &mut ws).unwrap();
        }
        assert_eq!(ws.as_slice(), &[3.0, 2.0, 3.0]);
    }

    #[test]
    fn matvec_dimension_err() {
        assert!(matches!(
            tridiag().matvec(&[1.0, 2.0]).unwrap_err(),
            SparseError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn scale_halves_values() {
        let mut m = tridiag();
        m.scale(0.5);
        assert_eq!(m.get(0, 0).unwrap(), 2.0);
        assert_eq!(m.get(0, 1).unwrap(), -0.5);
    }

    #[test]
    fn frobenius_norm_correct() {
        // full matrix entries: 4,-1,0,-1,4,-1,0,-1,4
        // sum of squares = 3*16 + 4*1 = 52
        let expected = 52.0_f64.sqrt();
        assert!((tridiag().frobenius_norm() - expected).abs() < 1e-12);
    }
}