use std::fmt;
use crate::error::{SparseError, Result};
use crate::SparseMatrix;

/// Compressed Sparse Column matrix — column-oriented storage.
///
/// The dual of [`CsrMatrix`]: `col_ptr` plays the role of `row_ptr`,
/// `row_idx` plays the role of `col_idx`.
///
/// Produced by converting from [`CsrMatrix`] or [`SymCsrMatrix`].
/// The solver crate consumes `CscMatrix` because Cholesky factorization
/// accesses columns sequentially — the natural direction for CSC.
///
/// # Invariants
/// - `col_ptr.len() == ncols + 1`, non-decreasing, `col_ptr[0] == 0`
/// - every `row_idx[k] < nrows`
/// - within each column row indices are **sorted ascending and unique**
/// - `values.len() == row_idx.len() == col_ptr[ncols]`
#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix {
    pub(crate) values:  Vec<f64>,
    pub(crate) row_idx: Vec<usize>,
    pub(crate) col_ptr: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

// -----------------------------------------------------------------
// SparseMatrix trait
// -----------------------------------------------------------------

impl SparseMatrix for CscMatrix {
    #[inline] fn nrows(&self)  -> usize { self.nrows }
    #[inline] fn ncols(&self)  -> usize { self.ncols }
    #[inline] fn nnz(&self)    -> usize { self.values.len() }
    fn validate(&self)         -> Result<()> { self.validate() }
}

// -----------------------------------------------------------------
// Construction
// -----------------------------------------------------------------

impl CscMatrix {
    /// Build a zero-valued CSC matrix from a column-oriented sparsity pattern.
    ///
    /// `pattern[j]` lists the row indices that are structurally non-zero in
    /// column `j`.  Duplicates and unsorted entries are cleaned up.
    ///
    /// # Errors
    /// - [`SparseError::PatternLengthMismatch`] if `pattern.len() != ncols`
    /// - [`SparseError::RowOutOfRange`] if any row index `>= nrows`
    pub fn from_pattern(
        nrows: usize,
        ncols: usize,
        pattern: &[Vec<usize>],
    ) -> Result<Self> {
        if pattern.len() != ncols {
            return Err(SparseError::PatternLengthMismatch {
                pattern_len: pattern.len(),
                nrows: ncols, // reusing field: "expected ncols patterns"
            });
        }

        let total_hint: usize = pattern.iter().map(|c| c.len()).sum();
        let mut col_ptr = Vec::with_capacity(ncols + 1);
        let mut row_idx: Vec<usize> = Vec::with_capacity(total_hint);

        col_ptr.push(0usize);

        for rows in pattern.iter() {
            for &r in rows {
                if r >= nrows {
                    return Err(SparseError::RowOutOfRange { row: r, nrows });
                }
            }

            let start = row_idx.len();
            row_idx.extend_from_slice(rows);
            let end = row_idx.len();

            row_idx[start..end].sort_unstable();
            if end > start {
                let mut write = start + 1;
                for read in (start + 1)..end {
                    if row_idx[read] != row_idx[write - 1] {
                        row_idx[write] = row_idx[read];
                        write += 1;
                    }
                }
                row_idx.truncate(write);
            }

            col_ptr.push(row_idx.len());
        }

        let nnz = row_idx.len();
        Ok(Self { values: vec![0.0; nnz], row_idx, col_ptr, nrows, ncols })
    }

    /// Build from raw CSC arrays (used by the conversion utilities).
    ///
    /// # Safety
    /// The caller guarantees the arrays satisfy all `CscMatrix` invariants.
    pub(crate) fn from_raw(
        nrows: usize,
        ncols: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        Self { values, row_idx, col_ptr, nrows, ncols }
    }
}

// -----------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------

impl CscMatrix {
    /// Number of structurally non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Value at `(row, col)`.
    ///
    /// Returns `0.0` for structural zeros absent from the pattern.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] / [`SparseError::ColOutOfRange`]
    pub fn get(&self, row: usize, col: usize) -> Result<f64> {
        self.check_bounds(row, col)?;
        Ok(self.find_idx(row, col).map_or(0.0, |i| self.values[i]))
    }
}

// -----------------------------------------------------------------
// Mutation
// -----------------------------------------------------------------

impl CscMatrix {
    /// Accumulate `val` into `(row, col)`.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] / [`SparseError::ColOutOfRange`]
    /// - [`SparseError::IndexOutOfBounds`] if absent from pattern
    pub fn add_value(&mut self, row: usize, col: usize, val: f64) -> Result<()> {
        self.check_bounds(row, col)?;
        let idx = self.find_idx(row, col)
            .ok_or(SparseError::IndexOutOfBounds { row, col })?;
        self.values[idx] += val;
        Ok(())
    }

    /// Overwrite `(row, col)` with `val`.
    ///
    /// # Errors
    /// Same as [`add_value`].
    pub fn set_value(&mut self, row: usize, col: usize, val: f64) -> Result<()> {
        self.check_bounds(row, col)?;
        let idx = self.find_idx(row, col)
            .ok_or(SparseError::IndexOutOfBounds { row, col })?;
        self.values[idx] = val;
        Ok(())
    }

    /// Set all stored values to `0.0` while keeping the pattern.
    #[inline]
    pub fn zero(&mut self) {
        self.values.fill(0.0);
    }

    /// Extract the diagonal into a `Vec<f64>`.
    ///
    /// # Errors
    /// - [`SparseError::NotSquare`]
    pub fn extract_diagonal(&self) -> Result<Vec<f64>> {
        if self.nrows != self.ncols {
            return Err(SparseError::NotSquare { nrows: self.nrows, ncols: self.ncols });
        }
        Ok((0..self.ncols)
            .map(|j| self.find_idx(j, j).map_or(0.0, |i| self.values[i]))
            .collect())
    }
}

// -----------------------------------------------------------------
// Validation
// -----------------------------------------------------------------

impl CscMatrix {
    /// Verify all internal invariants.
    pub fn validate(&self) -> Result<()> {
        if self.col_ptr.len() != self.ncols + 1 {
            return Err(SparseError::PatternLengthMismatch {
                pattern_len: self.col_ptr.len().saturating_sub(1),
                nrows: self.ncols,
            });
        }
        if self.values.len() != self.row_idx.len() {
            return Err(SparseError::DimensionMismatch {
                expected: self.row_idx.len(),
                got: self.values.len(),
            });
        }
        for col in 0..self.ncols {
            let start = self.col_ptr[col];
            let end   = self.col_ptr[col + 1];
            for &r in &self.row_idx[start..end] {
                if r >= self.nrows {
                    return Err(SparseError::RowOutOfRange { row: r, nrows: self.nrows });
                }
            }
            for w in self.row_idx[start..end].windows(2) {
                if w[0] >= w[1] {
                    return Err(SparseError::IndexOutOfBounds { row: w[0], col });
                }
            }
        }
        Ok(())
    }
}

// -----------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------

impl CscMatrix {
    /// Binary search for the storage index of `(row, col)`.
    /// Row indices within each column are sorted.
    #[inline]
    pub(crate) fn find_idx(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.col_ptr[col];
        let end   = self.col_ptr[col + 1];
        self.row_idx[start..end]
            .binary_search(&row)
            .ok()
            .map(|local| start + local)
    }

    #[inline]
    fn check_bounds(&self, row: usize, col: usize) -> Result<()> {
        if row >= self.nrows {
            return Err(SparseError::RowOutOfRange { row, nrows: self.nrows });
        }
        if col >= self.ncols {
            return Err(SparseError::ColOutOfRange { col, ncols: self.ncols });
        }
        Ok(())
    }
}

// -----------------------------------------------------------------
// Display
// -----------------------------------------------------------------

impl fmt::Display for CscMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.nrows {
            write!(f, "[")?;
            for col in 0..self.ncols {
                let val = self.find_idx(row, col).map_or(0.0, |i| self.values[i]);
                if col > 0 { write!(f, ", ")?; }
                write!(f, "{val:8.4}")?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// -----------------------------------------------------------------
// Unit tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CscMatrix {
        // Same matrix as CsrMatrix sample:
        // [1 0 2]
        // [0 3 4]
        // [0 0 5]
        // Column patterns: col0=[0], col1=[1], col2=[0,1,2]
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
    fn from_pattern_structure() {
        let m = sample();
        assert_eq!(m.col_ptr, vec![0, 1, 2, 5]);
        assert_eq!(m.row_idx, vec![0, 1, 0, 1, 2]);
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn from_pattern_deduplicates_and_sorts() {
        let m = CscMatrix::from_pattern(3, 1, &[vec![2usize, 0, 0, 1]]).unwrap();
        assert_eq!(m.row_idx, vec![0, 1, 2]);
    }

    #[test]
    fn from_pattern_err_length() {
        assert!(matches!(
            CscMatrix::from_pattern(3, 2, &[vec![0usize]]).unwrap_err(),
            SparseError::PatternLengthMismatch { .. }
        ));
    }

    #[test]
    fn from_pattern_err_row_range() {
        assert!(matches!(
            CscMatrix::from_pattern(3, 1, &[vec![99usize]]).unwrap_err(),
            SparseError::RowOutOfRange { row: 99, .. }
        ));
    }

    #[test]
    fn get_values() {
        let m = sample();
        assert_eq!(m.get(0, 0).unwrap(), 1.0);
        assert_eq!(m.get(1, 1).unwrap(), 3.0);
        assert_eq!(m.get(2, 2).unwrap(), 5.0);
        assert_eq!(m.get(0, 1).unwrap(), 0.0); // structural zero
    }

    #[test]
    fn add_value_accumulates() {
        let mut m = sample();
        m.add_value(0, 0, 9.0).unwrap();
        assert_eq!(m.get(0, 0).unwrap(), 10.0);
    }

    #[test]
    fn zero_clears_keeps_pattern() {
        let mut m = sample();
        m.zero();
        assert!(m.values.iter().all(|&v| v == 0.0));
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn extract_diagonal() {
        assert_eq!(sample().extract_diagonal().unwrap(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn validate_passes() {
        sample().validate().unwrap();
    }

    #[test]
    fn display_non_empty() {
        assert!(!format!("{}", sample()).is_empty());
    }
}