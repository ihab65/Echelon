use crate::{errors::{Result, CsrError}};
use crate::matrix::CsrMatrix;

impl CsrMatrix {
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> { // A*b -> c
        if x.len() != self.ncols {
            return Err(CsrError::DimensionMismatch);
        }
        let mut y = vec![0.0; self.nrows];
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for idx in start..end {
                y[row] += self.values[idx] * x[self.col_idx[idx]];
            }
        }
        Ok(y)
    }

    pub fn matvec_transpose(&self, x: &[f64]) -> Result<Vec<f64>> { // A^T*b -> c
        if x.len() != self.nrows {
            return Err(CsrError::DimensionMismatch);
        }
        let mut y = vec![0.0; self.ncols];
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for idx in start..end {
                y[self.col_idx[idx]] += self.values[idx] * x[row];
            }
        }
        Ok(y)
    }
    
    pub fn zero(&mut self) {
        self.values.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::CsrMatrix;
    use crate::errors::Result; // if needed for Result type

    #[test]
    fn test_matvec_operations() -> Result<()> {
        // 3x3 matrix with pattern:
        // [1 0 2]
        // [0 3 4]
        // [0 0 5]
        let pattern = vec![
            vec![0, 2],
            vec![1, 2],
            vec![2],
        ];

        let mut csr = CsrMatrix::from_pattern(3, 3, &pattern);

        // Add values
        csr.add_value(0, 0, 1.0);
        csr.add_value(0, 2, 2.0);
        csr.add_value(1, 1, 3.0);
        csr.add_value(1, 2, 4.0);
        csr.add_value(2, 2, 5.0);

        // Vector for multiplication
        let x = vec![1.0, 2.0, 3.0];

        // matvec: y = A*x
        let y = csr.matvec(&x)?;
        assert_eq!(y, vec![1.0*1.0 + 2.0*3.0, 3.0*2.0 + 4.0*3.0, 5.0*3.0]);

        // matvec_transpose: y = A^T*x
        let y_t = csr.matvec_transpose(&x)?;
        // A^T = 
        // [1 0 0]
        // [0 3 0]
        // [2 4 5]
        assert_eq!(y_t, vec![
            1.0*1.0 + 0.0*2.0 + 0.0*3.0,       // col 0
            0.0*1.0 + 3.0*2.0 + 0.0*3.0,       // col 1
            2.0*1.0 + 4.0*2.0 + 5.0*3.0        // col 2
        ]);

        // zero the matrix
        csr.zero();
        assert!(csr.values.iter().all(|&v| v == 0.0));

        Ok(())
    }
}