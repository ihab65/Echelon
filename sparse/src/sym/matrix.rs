use std::fmt;
use crate::error::{SparseError, Result};
use crate::{SparseMatrix, SparseScalar};

/// Symmetric Compressed Sparse Row matrix — upper triangle storage only.
///
/// For every entry `(i, j)` stored here, `j >= i` (on or above the diagonal).
/// The full matrix is implicitly `A[i,j] = A[j,i]` for `j != i`.
///
/// This halves the memory of [`CsrMatrix`] for symmetric matrices and is
/// the required input format for Cholesky factorization.
///
/// # Invariants
/// - `row_ptr.len() == nrows + 1`, non-decreasing, `row_ptr[0] == 0`
/// - `nrows == ncols` (always square)
/// - every stored entry satisfies `col_idx[k] >= row` for its row
/// - within each row column indices are **sorted ascending and unique**
/// - `values.len() == col_idx.len() == row_ptr[nrows]`
/// - the diagonal entry for every row is **always present** in the pattern
///   (required by Cholesky; enforced in `from_pattern`)
#[derive(Debug, Clone, PartialEq)]
pub struct SymCsrMatrix<T: SparseScalar> {
    pub(crate) values:  Vec<T>,
    pub(crate) col_idx: Vec<usize>,
    pub(crate) row_ptr: Vec<usize>,
    /// Dimension of the square matrix.
    pub n: usize,
}

// -----------------------------------------------------------------
// SparseMatrix trait
// -----------------------------------------------------------------

impl<T: SparseScalar> SparseMatrix for SymCsrMatrix<T> {
    #[inline] fn nrows(&self)  -> usize { self.n }
    #[inline] fn ncols(&self)  -> usize { self.n }
    #[inline] fn nnz(&self)    -> usize { self.nnz() }
    fn validate(&self)         -> Result<()> { self.validate() }
}

// -----------------------------------------------------------------
// Construction
// -----------------------------------------------------------------

impl<T: SparseScalar> SymCsrMatrix<T> {
    /// Build a zero-valued symmetric CSR matrix from an upper-triangle pattern.
    ///
    /// `pattern[i]` must only contain column indices `j >= i` (on or above
    /// the diagonal).  The diagonal entry `i` **must** appear in `pattern[i]`
    /// for every row — this is required for Cholesky and for `zero_row_col`.
    ///
    /// Duplicate and unsorted column indices are accepted and cleaned up.
    ///
    /// # Errors
    /// - [`SparseError::PatternLengthMismatch`] if `pattern.len() != n`
    /// - [`SparseError::ColOutOfRange`] if any column index `>= n`
    /// - [`SparseError::LowerTriangleEntry`] if any column index `< row`
    pub fn from_pattern(n: usize, pattern: &[Vec<usize>]) -> Result<Self> {
        if pattern.len() != n {
            return Err(SparseError::PatternLengthMismatch {
                pattern_len: pattern.len(),
                nrows: n,
            });
        }

        let total_hint: usize = pattern.iter().map(|r| r.len()).sum();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx: Vec<usize> = Vec::with_capacity(total_hint);

        row_ptr.push(0usize);

        for (row, cols) in pattern.iter().enumerate() {
            for &c in cols {
                if c >= n {
                    return Err(SparseError::ColOutOfRange { col: c, ncols: n });
                }
                if c < row {
                    return Err(SparseError::LowerTriangleEntry { row, col: c });
                }
            }

            let start = col_idx.len();
            col_idx.extend_from_slice(cols);
            let end = col_idx.len();

            // sort + dedup in-place within the appended slice
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

            // enforce: diagonal must be present
            let row_start = col_idx.len() - (col_idx.len() - start);
            let has_diag = col_idx[start..col_idx.len()]
                .binary_search(&row)
                .is_ok();
            if !has_diag {
                // insert diagonal at the right position to keep sorted order
                let insert_at = col_idx[start..col_idx.len()]
                    .partition_point(|&c| c < row)
                    + start;
                col_idx.insert(insert_at, row);
            }
            let _ = row_start; // silence warning

            row_ptr.push(col_idx.len());
        }

        let nnz = col_idx.len();
        Ok(Self { values: vec![T::zero(); nnz], col_idx, row_ptr, n })
    }

    /// Build the upper-triangle pattern from element DOF connectivity.
    ///
    /// Only `(i, j)` with `j >= i` are stored.  Use this instead of
    /// [`CsrMatrix::from_dof_connectivity`] when you know K is symmetric.
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
                // always include diagonal
                pattern[i].insert(i);
                for &j in dofs {
                    if j >= n_dof {
                        return Err(SparseError::ColOutOfRange { col: j, ncols: n_dof });
                    }
                    // upper triangle only: store (min, max) pair
                    if j >= i {
                        pattern[i].insert(j);
                    }
                }
            }
        }

        let vec_pattern: Vec<Vec<usize>> = pattern
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();

        Self::from_pattern(n_dof, &vec_pattern)
    }
}

// -----------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------

impl<T: SparseScalar> SymCsrMatrix<T> {
    /// Number of stored entries (upper triangle + diagonal).
    ///
    /// The full (logically symmetric) matrix has
    /// `2 * nnz() - n` non-zero entries.
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
    pub fn values(&self) -> &[T] { &self.values }

    /// Mutable raw values array.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] { &mut self.values }

    /// Value at `(row, col)` in the full symmetric matrix.
    ///
    /// Automatically redirects `(col, row)` when `col < row` so you
    /// can query either triangle transparently.
    ///
    /// Returns `0.0` for structural zeros (absent from the pattern).
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] / [`SparseError::ColOutOfRange`]
    pub fn get(&self, row: usize, col: usize) -> Result<T> {
        if row >= self.n {
            return Err(SparseError::RowOutOfRange { row, nrows: self.n });
        }
        if col >= self.n {
            return Err(SparseError::ColOutOfRange { col, ncols: self.n });
        }
        // redirect to upper triangle
        let (r, c) = if col >= row { (row, col) } else { (col, row) };
        Ok(self.find_idx(r, c).map_or(T::zero(), |i| self.values[i]))
    }
}

// -----------------------------------------------------------------
// Mutation
// -----------------------------------------------------------------

impl<T: SparseScalar> SymCsrMatrix<T> {
    /// Accumulate `val` into the upper-triangle entry `(row, col)`.
    ///
    /// # Errors
    /// - [`SparseError::LowerTriangleEntry`] if `col < row`
    /// - [`SparseError::RowOutOfRange`] / [`SparseError::ColOutOfRange`]
    /// - [`SparseError::IndexOutOfBounds`] if `(row, col)` is absent
    pub fn add_value(&mut self, row: usize, col: usize, val: T) -> Result<()> {
        self.add_value_and_return_index(row, col, val).map(|_| ())
    }

    /// Accumulate `val` into the upper-triangle entry `(row, col)` and return its flat index.
    ///
    /// Used internally for building efficient index mappings across matrices
    /// with identical topologies.
    ///
    /// # Errors
    /// Same as [`add_value`].
    pub fn add_value_and_return_index(&mut self, row: usize, col: usize, val: T) -> Result<usize> {
        self.check_upper(row, col)?;
        let idx = self.find_idx(row, col)
            .ok_or(SparseError::IndexOutOfBounds { row, col })?;
        self.values[idx] += val;
        Ok(idx)
    }

    /// Overwrite `(row, col)` with `val`.
    ///
    /// # Errors
    /// Same as [`add_value`].
    pub fn set_value(&mut self, row: usize, col: usize, val: T) -> Result<()> {
        self.check_upper(row, col)?;
        let idx = self.find_idx(row, col)
            .ok_or(SparseError::IndexOutOfBounds { row, col })?;
        self.values[idx] = val;
        Ok(())
    }

    /// Set all stored values to `0.0` while keeping the pattern.
    #[inline]
    pub fn zero(&mut self) {
        self.values.fill(T::zero());
    }

    /// Scatter the upper triangle of symmetric element stiffness `ke`
    /// (row-major `n×n`) into the global matrix via `dof_map`.
    ///
    /// Only the entries `(dof_map[i], dof_map[j])` with
    /// `dof_map[j] >= dof_map[i]` are scattered (upper triangle).
    ///
    /// # Errors
    /// - [`SparseError::ScatterSizeMismatch`] if `ke.len() != n²`
    /// - [`SparseError::IndexOutOfBounds`] if a mapped position is absent
    pub fn scatter_add(&mut self, ke: &[T], dof_map: &[usize]) -> Result<()> {
        let n = dof_map.len();
        let expected = n * n;
        if ke.len() != expected {
            return Err(SparseError::ScatterSizeMismatch {
                ke_len: ke.len(), n, expected,
            });
        }
        for i in 0..n {
            for j in i..n {
                // only upper triangle: global_row <= global_col
                let (gr, gc, val) = if dof_map[i] <= dof_map[j] {
                    (dof_map[i], dof_map[j], ke[i * n + j])
                } else {
                    (dof_map[j], dof_map[i], ke[j * n + i])
                };
                self.add_value(gr, gc, val)?;
            }
        }
        Ok(())
    }

    /// Extract the diagonal into a `Vec<f64>`.
    ///
    /// Every row is guaranteed to have a diagonal entry (enforced at
    /// construction), so this never returns zeros for a valid matrix.
    pub fn extract_diagonal(&self) -> Vec<T> {
        (0..self.n)
            .map(|i| {
                // diagonal is always the first entry in each row (col >= row,
                // and diagonal col == row is the smallest possible col)
                let idx = self.find_idx(i, i)
                    .expect("diagonal always present — invariant violated");
                self.values[idx]
            })
            .collect()
    }

    /// Apply a Dirichlet BC: zero the row and column for `dof`,
    /// then set `K[dof, dof] = 1.0`.
    ///
    /// In upper-triangle storage "zero the column for `dof`" means:
    /// scan rows `0..dof` and zero entry `(row, dof)` in each.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] if `dof >= n`
    pub fn zero_row_col(&mut self, dof: usize) -> Result<()> {
        if dof >= self.n {
            return Err(SparseError::RowOutOfRange { row: dof, nrows: self.n });
        }
        // zero the stored row (all entries with row index == dof)
        let start = self.row_ptr[dof];
        let end   = self.row_ptr[dof + 1];
        for idx in start..end {
            self.values[idx] = T::zero();
        }
        // zero entries (row, dof) for rows above dof — these are in the
        // upper triangle because row < dof
        for row in 0..dof {
            if let Some(idx) = self.find_idx(row, dof) {
                self.values[idx] = T::zero();
            }
        }
        // set diagonal to 1 (diagonal is always stored)
        let diag_idx = self.find_idx(dof, dof)
            .expect("diagonal invariant violated");
        self.values[diag_idx] = T::one();
        Ok(())
    }
}

// -----------------------------------------------------------------
// Validation
// -----------------------------------------------------------------

impl<T: SparseScalar> SymCsrMatrix<T> {
    /// Verify all internal invariants.
    ///
    /// # Errors
    /// - [`SparseError::PatternLengthMismatch`] if the row pointer array length is invalid.
    /// - [`SparseError::DimensionMismatch`] if the values array does not match the column indices.
    /// - [`SparseError::ColOutOfRange`] if any column index exceeds the matrix dimensions.
    /// - [`SparseError::LowerTriangleEntry`] if lower triangle entries are found.
    /// - [`SparseError::IndexOutOfBounds`] if the diagonal is missing or column indices are not strictly increasing.
    pub fn validate(&self) -> Result<()> {
        if self.row_ptr.len() != self.n + 1 {
            return Err(SparseError::PatternLengthMismatch {
                pattern_len: self.row_ptr.len().saturating_sub(1),
                nrows: self.n,
            });
        }
        if self.values.len() != self.col_idx.len() {
            return Err(SparseError::DimensionMismatch {
                expected: self.col_idx.len(),
                got: self.values.len(),
            });
        }
        for row in 0..self.n {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            // diagonal must be present
            if self.col_idx[start..end].binary_search(&row).is_err() {
                return Err(SparseError::IndexOutOfBounds { row, col: row });
            }
            for &c in &self.col_idx[start..end] {
                if c >= self.n {
                    return Err(SparseError::ColOutOfRange { col: c, ncols: self.n });
                }
                if c < row {
                    return Err(SparseError::LowerTriangleEntry { row, col: c });
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

impl<T: SparseScalar> SymCsrMatrix<T> {
    #[inline]
    pub(crate) fn find_idx(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.row_ptr[row];
        let end   = self.row_ptr[row + 1];
        self.col_idx[start..end]
            .binary_search(&col)
            .ok()
            .map(|local| start + local)
    }

    /// Verify a given `(row, col)` coordinate is within dimensions and in the upper triangle.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] if `row >= nrows`.
    /// - [`SparseError::ColOutOfRange`] if `col >= ncols`.
    /// - [`SparseError::LowerTriangleEntry`] if `col < row`.
    #[inline]
    fn check_upper(&self, row: usize, col: usize) -> Result<()> {
        if row >= self.n {
            return Err(SparseError::RowOutOfRange { row, nrows: self.n });
        }
        if col >= self.n {
            return Err(SparseError::ColOutOfRange { col, ncols: self.n });
        }
        if col < row {
            return Err(SparseError::LowerTriangleEntry { row, col });
        }
        Ok(())
    }
}

// -----------------------------------------------------------------
// Display
// -----------------------------------------------------------------

impl<T: SparseScalar> fmt::Display for SymCsrMatrix<T> {
    /// Prints the full symmetric matrix (both triangles expanded).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.n {
            write!(f, "[")?;
            for col in 0..self.n {
                let (r, c) = if col >= row { (row, col) } else { (col, row) };
                let val = self.find_idx(r, c).map_or(T::zero(), |i| self.values[i]);
                if col > 0 { write!(f, ", ")?; }
                write!(f, "{val:8.4}")?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// -----------------------------------------------------------------
// IO (Matrix Market)
// -----------------------------------------------------------------

#[cfg(feature = "io")]
impl<T: SparseScalar> SymCsrMatrix<T> {
    /// Exports the symmetric matrix to a Matrix Market (.mtx) file.
    /// Only the stored upper triangle is written, with the 'symmetric' header.
    ///
    /// # Errors
    /// Returns a generic IO error dynamically wrapped in an `IoError` if writing to the file fails.
    pub fn to_mtx<P: AsRef<std::path::Path>>(&self, path: P) -> crate::error::Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header: Note the 'symmetric' tag
        writeln!(writer, "%%MatrixMarket matrix coordinate real symmetric")?;
        
        // Shape: n n nnz (where nnz is just the stored triangle)
        writeln!(writer, "{} {} {}", self.n, self.n, self.nnz())?;

        // Data: 1-based indexing
        for i in 0..self.n {
            for k in self.row_ptr[i]..self.row_ptr[i+1] {
                let j = self.col_idx[k];
                let val = self.values[k];
                // In SymCsrMatrix, j >= i is guaranteed by invariants
                writeln!(writer, "{} {} {:.16}", i + 1, j + 1, val)?;
            }
        }
        writer.flush()?;
        Ok(())
    }
}

// -----------------------------------------------------------------
// Unit tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 3×3 symmetric:
    /// [ 4 -1  0]
    /// [-1  4 -1]
    /// [ 0 -1  4]
    fn tridiag() -> SymCsrMatrix<f64> {
        // upper triangle: row0=[0,1], row1=[1,2], row2=[2]
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
    fn from_pattern_structure() {
        let m = tridiag();
        assert_eq!(m.row_ptr, vec![0, 2, 4, 5]);
        assert_eq!(m.col_idx, vec![0, 1, 1, 2, 2]);
    }

    #[test]
    fn from_pattern_inserts_diagonal_automatically() {
        // pattern missing diagonal for row 0 — should be inserted
        let pattern = vec![vec![1usize], vec![1]]; // row0 has only col1, row1 has only col1
        let m = SymCsrMatrix::<f64>::from_pattern(2, &pattern).unwrap();
        // diagonal must now be present in both rows
        assert!(m.find_idx(0, 0).is_some());
        assert!(m.find_idx(1, 1).is_some());
    }

    #[test]
    fn from_pattern_err_lower_triangle() {
        // col 0 < row 1 — lower triangle
        let pattern = vec![vec![0usize], vec![0usize, 1]];
        assert!(matches!(
            SymCsrMatrix::<f64>::from_pattern(2, &pattern).unwrap_err(),
            SparseError::LowerTriangleEntry { row: 1, col: 0 }
        ));
    }

    #[test]
    fn from_pattern_err_col_out_of_range() {
        let pattern = vec![vec![0usize, 99]];
        assert!(matches!(
            SymCsrMatrix::<f64>::from_pattern(1, &pattern).unwrap_err(),
            SparseError::ColOutOfRange { col: 99, .. }
        ));
    }

    #[test]
    fn get_both_triangles() {
        let m = tridiag();
        // upper stored
        assert_eq!(m.get(0, 1).unwrap(), -1.0);
        // lower redirected
        assert_eq!(m.get(1, 0).unwrap(), -1.0);
        // structural zero
        assert_eq!(m.get(0, 2).unwrap(),  0.0);
    }

    #[test]
    fn add_value_accumulates() {
        let mut m = tridiag();
        m.add_value(0, 0, 1.0).unwrap();
        assert_eq!(m.get(0, 0).unwrap(), 5.0);
    }

    #[test]
    fn add_value_err_lower_triangle() {
        let mut m = tridiag();
        assert!(matches!(
            m.add_value(1, 0, 1.0).unwrap_err(),
            SparseError::LowerTriangleEntry { .. }
        ));
    }

    #[test]
    fn zero_clears_keeps_pattern() {
        let mut m = tridiag();
        m.zero();
        assert!(m.values.iter().all(|&v| v == 0.0));
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn extract_diagonal() {
        assert_eq!(tridiag().extract_diagonal(), vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn zero_row_col_bc() {
        let mut m = tridiag();
        m.zero_row_col(1).unwrap();
        assert_eq!(m.get(1, 1).unwrap(), 1.0);  // diagonal → 1
        assert_eq!(m.get(1, 2).unwrap(), 0.0);  // row zeroed
        assert_eq!(m.get(0, 1).unwrap(), 0.0);  // column zeroed (upper entry)
        assert_eq!(m.get(0, 0).unwrap(), 4.0);  // untouched
    }

    #[test]
    fn validate_passes() {
        tridiag().validate().unwrap();
    }

    #[test]
    fn display_non_empty() {
        assert!(!format!("{}", tridiag()).is_empty());
    }
}