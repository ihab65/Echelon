//! Coordinate (COO) format builder.
//!
//! `CooBuilder` is the most ergonomic way to construct a sparse matrix
//! when you don't know the sparsity pattern in advance.  Collect triplets
//! `(row, col, value)` freely, then call `.build_csr()` or `.build_sym()`.
//!
//! ```
//! use sparse::CooBuilder;
//!
//! let mut coo = CooBuilder::new(3, 3);
//! coo.add(0, 0,  4.0);
//! coo.add(0, 1, -1.0);
//! coo.add(1, 1,  4.0);
//! coo.add(1, 2, -1.0);
//! coo.add(2, 2,  4.0);
//!
//! let csr = coo.build_csr().unwrap();
//! ```

use crate::SparseScalar;
use crate::error::{SparseError, Result};
use crate::csr::CsrMatrix;
use crate::sym::SymCsrMatrix;

/// Accumulates `(row, col, value)` triplets and builds a sparse matrix.
///
/// Duplicate entries `(i, j)` are **summed** into a single stored value,
/// matching FEM assembly semantics.
#[derive(Debug, Clone)]
pub struct CooBuilder<T> {
    nrows:    usize,
    ncols:    usize,
    triplets: Vec<(usize, usize, T)>,
}

impl<T: SparseScalar> CooBuilder<T> {
    /// Create a new builder for an `nrows × ncols` matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, triplets: Vec::new() }
    }

    /// Create a new builder with a pre-allocated capacity hint.
    pub fn with_capacity(nrows: usize, ncols: usize, cap: usize) -> Self {
        Self { nrows, ncols, triplets: Vec::with_capacity(cap) }
    }

    /// Add a triplet `(row, col, val)`.
    ///
    /// Out-of-bounds triplets are silently accepted here and will cause an
    /// error only when `.build_csr()` / `.build_sym()` is called.
    #[inline]
    pub fn add(&mut self, row: usize, col: usize, val: T) {
        self.triplets.push((row, col, val));
    }

    /// Number of triplets accumulated so far (before deduplication).
    pub fn len(&self) -> usize {
        self.triplets.len()
    }

    /// Returns `true` if no triplets have been added.
    pub fn is_empty(&self) -> bool {
        self.triplets.is_empty()
    }

    /// Build a [`CsrMatrix`], summing duplicate entries.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] / [`SparseError::ColOutOfRange`]
    ///   if any triplet index is out of bounds
    pub fn build_csr(self) -> Result<CsrMatrix<T>> {
        // Validate bounds
        for &(row, col, _) in &self.triplets {
            if row >= self.nrows {
                return Err(SparseError::RowOutOfRange { row, nrows: self.nrows });
            }
            if col >= self.ncols {
                return Err(SparseError::ColOutOfRange { col, ncols: self.ncols });
            }
        }

        // Collect unique (row, col) pairs — use BTreeMap to sort and sum
        use std::collections::BTreeMap;
        let mut map: BTreeMap<(usize, usize), T> = BTreeMap::new();
        for (row, col, val) in self.triplets {
            *map.entry((row, col)).or_insert(T::zero()) += val;
        }

        // Build pattern and fill values in one pass
        let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); self.nrows];
        for &(row, col) in map.keys() {
            pattern[row].push(col);
        }
        // patterns are already sorted (BTreeMap iterates in key order)

        let mut csr = CsrMatrix::from_pattern(self.nrows, self.ncols, &pattern)?;
        for ((row, col), val) in map {
            csr.add_value(row, col, val)?;
        }
        Ok(csr)
    }

    /// Build a [`SymCsrMatrix`] from the upper-triangle entries only.
    ///
    /// Triplets with `col < row` (lower triangle) are **mirrored** to
    /// `(col, row)` before assembly, so you can pass either triangle.
    /// Diagonal entries are passed through unchanged.
    ///
    /// # Errors
    /// - [`SparseError::RowOutOfRange`] / [`SparseError::ColOutOfRange`]
    ///   if any triplet index is out of bounds
    /// - [`SparseError::NotSquare`] if `nrows != ncols`
    pub fn build_sym(self) -> Result<SymCsrMatrix<T>> {
        if self.nrows != self.ncols {
            return Err(SparseError::NotSquare {
                nrows: self.nrows,
                ncols: self.ncols,
            });
        }
        let n = self.nrows;

        for &(row, col, _) in &self.triplets {
            if row >= n {
                return Err(SparseError::RowOutOfRange { row, nrows: n });
            }
            if col >= n {
                return Err(SparseError::ColOutOfRange { col, ncols: n });
            }
        }

        use std::collections::BTreeMap;
        let mut map: BTreeMap<(usize, usize), T> = BTreeMap::new();
        for (row, col, val) in self.triplets {
            // mirror to upper triangle
            let (r, c) = if col >= row { (row, col) } else { (col, row) };
            *map.entry((r, c)).or_insert(T::zero()) += val;
        }

        let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(row, col) in map.keys() {
            pattern[row].push(col);
        }

        let mut sym = SymCsrMatrix::from_pattern(n, &pattern)?;
        for ((row, col), val) in map {
            sym.add_value(row, col, val)?;
        }
        Ok(sym)
    }
}

// Optional MTX loading support, gated behind the "io" feature. This is a convenient
// way to get real-world sparse matrices into the library for testing and benchmarking.
#[cfg(feature = "io")]
impl<T: SparseScalar> CooBuilder<T> {
    pub fn from_mtx<P: AsRef<std::path::Path>>(path: P) -> Result<Self>
    where
        T: From<f64>,
    {
        // matrix_market_rs returns MtxData<T, NDIM>. We use f64 and default 2 dims.
        let mtx = matrix_market_rs::MtxData::<f64>::from_file(path)
            .map_err(|e| SparseError::IoError(format!("MTX parse error: {:?}", e)))?;

        match mtx {
            matrix_market_rs::MtxData::Sparse(shape, indices, values, sym_info) => {
                let mut builder = Self::new(shape[0], shape[1]);

                for (idx, &val) in indices.iter().zip(values.iter()) {
                    // Note: matrix-market-rs source shows it already does:
                    // dims[i] = num - 1; 
                    // So these are already 0-indexed.
                    let mut r = idx[0];
                    let mut c = idx[1];

                    match sym_info {
                        matrix_market_rs::SymInfo::General => {
                            builder.add(r, c, <T as From<f64>>::from(val));
                        }
                        matrix_market_rs::SymInfo::Symmetric => {
                            // Ensure upper triangle (r <= c) for build_sym compatibility
                            if r > c {
                                std::mem::swap(&mut r, &mut c);
                            }
                            builder.add(r, c, <T as From<f64>>::from(val));
                        }
                    }
                }
                Ok(builder)
            }
            matrix_market_rs::MtxData::Dense(_, _, _) => {
                Err(SparseError::IoError("Dense MTX not supported".into()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_csr_basic() {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 1.0);
        coo.add(0, 2, 2.0);
        coo.add(1, 1, 3.0);
        coo.add(1, 2, 4.0);
        coo.add(2, 2, 5.0);
        let csr = coo.build_csr().unwrap();
        csr.validate().unwrap();
        assert_eq!(csr.get(0, 0).unwrap(), 1.0);
        assert_eq!(csr.get(1, 2).unwrap(), 4.0);
        assert_eq!(csr.get(2, 2).unwrap(), 5.0);
    }

    #[test]
    fn build_csr_sums_duplicates() {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, 1.0);
        coo.add(0, 0, 2.0); // duplicate — should sum to 3.0
        let csr = coo.build_csr().unwrap();
        assert_eq!(csr.get(0, 0).unwrap(), 3.0);
    }

    #[test]
    fn build_csr_err_row_out_of_range() {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(99, 0, 1.0);
        assert!(matches!(
            coo.build_csr().unwrap_err(),
            SparseError::RowOutOfRange { .. }
        ));
    }

    #[test]
    fn build_sym_basic() {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0,  4.0);
        coo.add(0, 1, -1.0);
        coo.add(1, 1,  4.0);
        coo.add(1, 2, -1.0);
        coo.add(2, 2,  4.0);
        let sym = coo.build_sym().unwrap();
        sym.validate().unwrap();
        assert_eq!(sym.get(0, 0).unwrap(),  4.0);
        assert_eq!(sym.get(0, 1).unwrap(), -1.0);
        assert_eq!(sym.get(1, 0).unwrap(), -1.0); // mirrored read
    }

    #[test]
    fn build_sym_mirrors_lower_triangle() {
        // Feed lower-triangle entries — should become upper entries
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, 2.0);
        coo.add(1, 0, -1.0); // lower → mirrored to (0,1)
        coo.add(1, 1,  2.0);
        let sym = coo.build_sym().unwrap();
        assert_eq!(sym.get(0, 1).unwrap(), -1.0);
        assert_eq!(sym.get(1, 0).unwrap(), -1.0);
    }

    #[test]
    fn build_sym_err_not_square() {
        let mut coo = CooBuilder::new(2, 3);
        coo.add(0, 0, 1.0);
        assert!(matches!(
            coo.build_sym().unwrap_err(),
            SparseError::NotSquare { .. }
        ));
    }

    #[test]
    fn len_and_is_empty() {
        let mut coo = CooBuilder::new(2, 2);
        assert!(coo.is_empty());
        coo.add(0, 0, 1.0);
        assert_eq!(coo.len(), 1);
    }
}