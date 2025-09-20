use crate::matrix::CsrMatrix;

impl CsrMatrix {
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for row in 0..self.nrows {
            for (col, val) in self.row_iter(row) {
                dense[row][col] = val;
            }
        }
        dense
    }

    pub fn print_pretty(&self) {
        for row in self.to_dense() {
            println!("{:?}", row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CsrMatrix;

    #[test]
    fn test_to_dense_and_print() {
        // 3x3 matrix with pattern
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

        // Convert to dense
        let dense = csr.to_dense();
        let expected = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 4.0],
            vec![0.0, 0.0, 5.0],
        ];
        assert_eq!(dense, expected);

        // Print (manual check)
        csr.print_pretty();
        // Expected console output:
        // [1.0, 0.0, 2.0]
        // [0.0, 3.0, 4.0]
        // [0.0, 0.0, 5.0]
    }
}

