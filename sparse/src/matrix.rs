#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub values: Vec<f64>,
    pub col_idx: Vec<usize>,
    pub row_ptr: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

impl CsrMatrix {
    /// Build an empty CSR structure from a sparsity pattern
    pub fn from_pattern(nrows: usize, ncols: usize, pattern: &[Vec<usize>]) -> Self {
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();

        row_ptr.push(0);
        for row in 0..nrows {
            let mut cols = pattern[row].clone();
            cols.sort();
            cols.dedup();

            col_idx.extend(cols);
            row_ptr.push(col_idx.len());
        }

        let nnz = col_idx.len();
        let values = vec![0.0; nnz];

        CsrMatrix { values, col_idx, row_ptr, nrows, ncols }
    }

    /// Add value at (row, col)
    pub fn add_value(&mut self, row: usize, col: usize, val: f64) {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        for idx in start..end {
            if self.col_idx[idx] == col {
                self.values[idx] += val;
                return;
            }
        }

        panic!("Invalid (row, col) = ({}, {}) not in sparsity pattern", row, col);
    }

    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}