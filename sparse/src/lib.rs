pub mod matrix;
pub mod ops;
pub mod iter;
pub mod utils;
pub mod errors;

pub use matrix::CsrMatrix;
pub use errors::{CsrError, Result};

#[cfg(test)]
mod tests {
    use super::CsrMatrix;

    #[test]
    fn test_csr_from_pattern() {
        // Define a 3x3 sparsity pattern
        // Row 0: columns 0,2
        // Row 1: columns 0,1,2
        // Row 2: columns 1,2
        let pattern = vec![
            vec![0, 2],
            vec![0, 1, 2],
            vec![1, 2],
        ];

        let mut csr = CsrMatrix::from_pattern(3, 3, &pattern);

        // Check row_ptr
        assert_eq!(csr.row_ptr, vec![0, 2, 5, 7]);

        // Check col_idx
        assert_eq!(csr.col_idx, vec![0, 2, 0, 1, 2, 1, 2]);

        // Initially, all values are zeros
        assert_eq!(csr.values, vec![0.0; 7]);

        // Add values
        csr.add_value(0, 0, 1.5);
        csr.add_value(1, 2, 2.5);
        csr.add_value(2, 2, 3.0);

        // Check values
        assert_eq!(csr.values, vec![1.5, 0.0, 0.0, 0.0, 2.5, 0.0, 3.0]);

        // Number of non-zero entries
        assert_eq!(csr.nnz(), 7);
    }
}