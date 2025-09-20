use crate::matrix::CsrMatrix;

pub struct RowIter<'a> {
    cols: &'a [usize],
    vals: &'a [f64],
}

impl<'a> Iterator for RowIter<'a> {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cols.is_empty() { None }
        else {
            let col = self.cols[0];
            let val = self.vals[0];
            self.cols = &self.cols[1..];
            self.vals = &self.vals[1..];
            Some((col, val))
        }
    }
}

impl CsrMatrix {
    pub fn row_iter(&self, row: usize) -> RowIter<'_> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        RowIter {
            cols: &self.col_idx[start..end],
            vals: &self.values[start..end],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CsrMatrix;

    #[test]
    fn test_row_iter() {
        // 3x3 matrix with pattern:
        // Row 0: columns 0,2
        // Row 1: columns 1,2
        // Row 2: column 2
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

        // Test row 0 iterator
        let row0: Vec<(usize, f64)> = csr.row_iter(0).collect();
        assert_eq!(row0, vec![(0, 1.0), (2, 2.0)]);

        // Test row 1 iterator
        let row1: Vec<(usize, f64)> = csr.row_iter(1).collect();
        assert_eq!(row1, vec![(1, 3.0), (2, 4.0)]);

        // Test row 2 iterator
        let row2: Vec<(usize, f64)> = csr.row_iter(2).collect();
        assert_eq!(row2, vec![(2, 5.0)]);
    }
}

