use std::fmt;
use crate::error::{SparseError, Result};
use crate::SparseMatrix;

/// Compressed Sparse Row matrix — general (unsymmetric) storage.
///
/// Stores both upper and lower triangles.  Used for assembly because
/// `scatter_add` is row-oriented by nature.
///
/// # Invariants
/// - `row_ptr.len() == nrows + 1`, `row_ptr[0] == 0`, non-decreasing
/// - every `col_idx[k] < ncols`
/// - within each row column indices are **sorted ascending and unique**
/// - `values.len() == col_idx.len() == row_ptr[nrows]`
///
/// Fields are `pub(crate)` — sibling modules can read internals directly
/// without exposing them outside the crate.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub(crate) values:  Vec<f64>,
    pub(crate) col_idx: Vec<usize>,
    pub(crate) row_ptr: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

// -----------------------------------------------------------------
// SparseMatrix trait impl
// -----------------------------------------------------------------

impl SparseMatrix for CsrMatrix {
    #[inline] fn nrows(&self)    -> usize { self.nrows }
    #[inline] fn ncols(&self)    -> usize { self.ncols }
    #[inline] fn nnz(&self)      -> usize { self.nnz() }
    fn validate(&self)           -> Result<()> { self.validate() }
}

// -----------------------------------------------------------------
// Construction
// -----------------------------------------------------------------

impl CsrMatrix {
    /// Build a zero-valued CSR matrix from a sparsity pattern.
    ///
    /// `pattern[i]` is the list of column indices that are structurally
    /// non-zero in row `i`.  Duplicates and unsorted entries are accepted
    /// and cleaned up in-place (no per-row allocation).
    ///
    /// # Errors
    /// - [`SparseError::PatternLengthMismatch`] if `pattern.len() != nrows`
    /// - [`SparseError::ColOutOfRange`] if any column index `>= ncols`
    pub fn from_pattern(
        nrows: usize,
        ncols: usize,
        pattern: &[Vec<usize>],
    ) -> Result<Self> {
        if pattern.len() != nrows {
            return Err(SparseError::PatternLengthMismatch {
                pattern_len: pattern.len(),
                nrows,
            });
        }

        let total_hint: usize = pattern.iter().map(|r| r.len()).sum();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx: Vec<usize> = Vec::with_capacity(total_hint);

        row_ptr.push(0usize);

        for cols in pattern.iter() {
            for &c in cols {
                if c >= ncols {
                    return Err(SparseError::ColOutOfRange { col: c, ncols });
                }
            }

            let start = col_idx.len();
            col_idx.extend_from_slice(cols);
            let end = col_idx.len();

            col_idx[start..end].sort_unstable();
            if end > start {
                let mut write = start + 1;
                for read in (start + 1)..end {
                    if col_idx[read] != col_idx[write - 1] {
                        col_idx[write] = col_idx[read];
                        write += 1;
                    }
                }
                col_idx.truncate(write);
            }

            row_ptr.push(col_idx.len());
        }

        let nnz = col_idx.len();
        Ok(Self { values: vec![0.0; nnz], col_idx, row_ptr, nrows, ncols })
    }

    /// Build the global stiffness pattern from element DOF connectivity.
    ///
    /// Every `(i, j)` pair within an element's DOF list becomes a structural
    /// non-zero (both triangles stored).  Call this once per model topology.
    ///
    /// # Errors
    /// - [`SparseError::ColOutOfRange`] if any DOF index `>= n_dof`
    pub fn from_dof_connectivity(
        n_dof: usize,
        element_dofs: &[Vec<usize>],
    ) -> Result<Self> {
        use std::collections::BTreeSet;
        let mut pattern: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_dof];

        for dofs in element_dofs {
            for &i in dofs {
                if i >= n_dof {
                    return Err(SparseError::ColOutOfRange { col: i, ncols: n_dof });
                }
                for &j in dofs {
                    if j >= n_dof {
                        return Err(SparseError::ColOutOfRange { col: j, ncols: n_dof });
                    }
                    pattern[i].insert(j);
                }
            }
        }

        let vec_pattern: Vec<Vec<usize>> = pattern
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();

        Self::from_pattern(n_dof, n_dof, &vec_pattern)
    }

    /// Build from raw CSR arrays (used by conversion utilities).
    ///
    /// The caller guarantees all invariants are satisfied.
    pub(crate) fn from_raw(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        Self { values, col_idx, row_ptr, nrows, ncols }
    }
}

// -----------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------

impl CsrMatrix {
    /// Number of structurally non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Raw row pointer array: `row_ptr[i]..row_ptr[i+1]` is the storage
    /// range for row `i`.
    #[inline]
    pub fn row_ptr(&self) -> &[usize] { &self.row_ptr }

    /// Raw column index array.
    #[inline]
    pub fn col_idx(&self) -> &[usize] { &self.col_idx }

    /// Raw values array.
    #[inline]
    pub fn values(&self) -> &[f64] { &self.values }
    
    /// Value at `(row, col)`.
    ///
    /// Returns `0.0` for structural zeros (entries absent from the pattern)
    /// rather than an error — consistent with how sparse arithmetic works.
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

impl CsrMatrix {
    /// Accumulate `val` into `(row, col)`.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] / [`SparseError::ColOutOfRange`]
    /// - [`SparseError::IndexOutOfBounds`] if `(row, col)` is absent
    pub fn add_value(&mut self, row: usize, col: usize, val: f64) -> Result<()> {
        self.check_bounds(row, col)?;
        let idx = self.find_idx(row, col)
            .ok_or(SparseError::IndexOutOfBounds { row, col })?;
        self.values[idx] += val;
        Ok(())
    }

    /// Overwrite `(row, col)` with `val`.
    ///
    /// Use this when applying boundary conditions (set diagonal to `1.0`).
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

    /// Set all stored values to `0.0` while keeping the sparsity pattern.
    ///
    /// Call at the start of every assembly pass.
    #[inline]
    pub fn zero(&mut self) {
        self.values.fill(0.0);
    }

    /// Scatter dense element stiffness `ke` (row-major `n×n`) into the global
    /// matrix using `dof_map` as global DOF indices (length `n`).
    ///
    /// This is the innermost FEM assembly loop — called once per element per
    /// Newton iteration.
    ///
    /// # Errors
    /// - [`SparseError::ScatterSizeMismatch`] if `ke.len() != dof_map.len()²`
    /// - [`SparseError::IndexOutOfBounds`] if a mapped position is absent
    pub fn scatter_add(&mut self, ke: &[f64], dof_map: &[usize]) -> Result<()> {
        let n = dof_map.len();
        let expected = n * n;
        if ke.len() != expected {
            return Err(SparseError::ScatterSizeMismatch {
                ke_len: ke.len(),
                n,
                expected,
            });
        }
        for i in 0..n {
            for j in 0..n {
                self.add_value(dof_map[i], dof_map[j], ke[i * n + j])?;
            }
        }
        Ok(())
    }

    /// Extract the main diagonal into a `Vec<f64>`.
    ///
    /// Entries absent from the pattern are returned as `0.0`.
    ///
    /// # Errors
    /// - [`SparseError::NotSquare`]
    pub fn extract_diagonal(&self) -> Result<Vec<f64>> {
        if self.nrows != self.ncols {
            return Err(SparseError::NotSquare { nrows: self.nrows, ncols: self.ncols });
        }
        Ok((0..self.nrows)
            .map(|i| self.find_idx(i, i).map_or(0.0, |idx| self.values[idx]))
            .collect())
    }

    /// Apply a Dirichlet BC to DOF `dof`:
    /// zero row `dof`, zero column `dof`, set `K[dof, dof] = 1.0`.
    ///
    /// After this call set `F[dof] = prescribed_value`.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] if `dof >= nrows`
    /// - [`SparseError::IndexOutOfBounds`] if `(dof, dof)` is absent
    pub fn zero_row_col(&mut self, dof: usize) -> Result<()> {
        if dof >= self.nrows {
            return Err(SparseError::RowOutOfRange { row: dof, nrows: self.nrows });
        }
        let start = self.row_ptr[dof];
        let end   = self.row_ptr[dof + 1];
        for idx in start..end {
            self.values[idx] = 0.0;
        }
        for row in 0..self.nrows {
            if row == dof { continue; }
            if let Some(idx) = self.find_idx(row, dof) {
                self.values[idx] = 0.0;
            }
        }
        self.set_value(dof, dof, 1.0)
    }
}

// -----------------------------------------------------------------
// Validation
// -----------------------------------------------------------------

impl CsrMatrix {
    /// Verify all internal invariants.
    pub fn validate(&self) -> Result<()> {
        if self.row_ptr.len() != self.nrows + 1 {
            return Err(SparseError::PatternLengthMismatch {
                pattern_len: self.row_ptr.len().saturating_sub(1),
                nrows: self.nrows,
            });
        }
        if self.values.len() != self.col_idx.len() {
            return Err(SparseError::DimensionMismatch {
                expected: self.col_idx.len(),
                got: self.values.len(),
            });
        }
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            for &c in &self.col_idx[start..end] {
                if c >= self.ncols {
                    return Err(SparseError::ColOutOfRange { col: c, ncols: self.ncols });
                }
            }
            for w in self.col_idx[start..end].windows(2) {
                if w[0] >= w[1] {
                    return Err(SparseError::IndexOutOfBounds { row, col: w[0] });
                }
            }
        }
        Ok(())
    }
}

// -----------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------

impl CsrMatrix {
    /// Binary search for the storage index of `(row, col)`.
    /// Columns within each row are sorted — O(log nnz/row).
    #[inline]
    pub(crate) fn find_idx(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        self.col_idx[start..end]
            .binary_search(&col)
            .ok()
            .map(|local| start + local)
    }

    #[inline]
    pub(crate) fn check_bounds(&self, row: usize, col: usize) -> Result<()> {
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

impl fmt::Display for CsrMatrix {
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
// Unit tests  (internal / white-box)
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn sample() -> CsrMatrix {
        // [1 0 2]
        // [0 3 4]
        // [0 0 5]
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
    fn from_pattern_structure() {
        let pattern = vec![vec![0usize, 2], vec![0, 1, 2], vec![1, 2]];
        let csr = CsrMatrix::from_pattern(3, 3, &pattern).unwrap();
        assert_eq!(csr.row_ptr, vec![0, 2, 5, 7]);
        assert_eq!(csr.col_idx, vec![0, 2, 0, 1, 2, 1, 2]);
        assert!(csr.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn from_pattern_deduplicates_and_sorts() {
        let m = CsrMatrix::from_pattern(1, 3, &[vec![2usize, 0, 0, 1]]).unwrap();
        assert_eq!(m.col_idx, vec![0, 1, 2]);
    }

    #[test]
    fn from_pattern_err_length() {
        assert!(matches!(
            CsrMatrix::from_pattern(2, 3, &[vec![0usize]]).unwrap_err(),
            SparseError::PatternLengthMismatch { .. }
        ));
    }

    #[test]
    fn from_pattern_err_col_range() {
        assert!(matches!(
            CsrMatrix::from_pattern(1, 3, &[vec![0usize, 99]]).unwrap_err(),
            SparseError::ColOutOfRange { col: 99, .. }
        ));
    }

    #[test]
    fn add_value_accumulates() {
        let mut m = sample();
        m.add_value(0, 0, 10.0).unwrap();
        assert_eq!(m.get(0, 0).unwrap(), 11.0);
    }

    #[test]
    fn add_value_err_not_in_pattern() {
        let mut m = sample();
        assert!(matches!(
            m.add_value(0, 1, 1.0).unwrap_err(),
            SparseError::IndexOutOfBounds { row: 0, col: 1 }
        ));
    }

    #[test]
    fn get_structural_zero_returns_0() {
        assert_eq!(sample().get(0, 1).unwrap(), 0.0);
    }

    #[test]
    fn zero_clears_keeps_pattern() {
        let mut m = sample();
        m.zero();
        assert!(m.values.iter().all(|&v| v == 0.0));
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn scatter_add_correct() {
        let pattern = vec![vec![0usize, 1], vec![0, 1]];
        let mut k = CsrMatrix::from_pattern(2, 2, &pattern).unwrap();
        k.scatter_add(&[2.0_f64, -2.0, -2.0, 2.0], &[0, 1]).unwrap();
        assert_eq!(k.get(0, 0).unwrap(),  2.0);
        assert_eq!(k.get(0, 1).unwrap(), -2.0);
        assert_eq!(k.get(1, 0).unwrap(), -2.0);
        assert_eq!(k.get(1, 1).unwrap(),  2.0);
    }

    #[test]
    fn scatter_add_err_size() {
        let pattern = vec![vec![0usize, 1], vec![0, 1]];
        let mut k = CsrMatrix::from_pattern(2, 2, &pattern).unwrap();
        assert!(matches!(
            k.scatter_add(&[1.0, 0.0, 0.0], &[0, 1]).unwrap_err(),
            SparseError::ScatterSizeMismatch { .. }
        ));
    }

    #[test]
    fn extract_diagonal() {
        assert_eq!(sample().extract_diagonal().unwrap(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn zero_row_col_bc() {
        let pattern = vec![vec![0usize,1,2], vec![0,1,2], vec![0,1,2]];
        let mut k = CsrMatrix::from_pattern(3, 3, &pattern).unwrap();
        for i in 0..3 { for j in 0..3 { k.set_value(i, j, 2.0).unwrap(); } }
        k.zero_row_col(1).unwrap();
        assert_eq!(k.get(1, 1).unwrap(), 1.0);
        assert_eq!(k.get(1, 0).unwrap(), 0.0);
        assert_eq!(k.get(0, 1).unwrap(), 0.0);
        assert_eq!(k.get(0, 0).unwrap(), 2.0);
    }

    #[test]
    fn validate_good_matrix() {
        sample().validate().unwrap();
    }

    #[test]
    fn display_non_empty() {
        assert!(!format!("{}", sample()).is_empty());
    }
}