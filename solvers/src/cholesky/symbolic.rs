//! Symbolic Cholesky factorization.
//!
//! Computes the sparsity pattern of the Cholesky factor `L` from the
//! sparsity pattern of `K` alone — no floating point work involved.
//!
//! ## Algorithm
//!
//! Uses the **elimination tree** (etree):
//! - The parent of node `j` in the etree is the smallest `i > j` such
//!   that `L[i,j] != 0`.
//! - The non-zero pattern of column `j` of `L` is: `{j}` ∪ the patterns
//!   of all children of `j` in the etree, intersected with rows `> j`.
//!
//! This is the Liu (1986) / Davis (2006) algorithm.

use crate::error::Result;
use sparse::SymCsrMatrix;

/// Result of the symbolic Cholesky phase.
pub struct SymbolicCholesky {
    /// Elimination tree: `parent[j]` is the parent of column `j`.
    /// `parent[j] == n` means `j` is a root.
    pub parent: Vec<usize>,
    /// Column pointers for `L` (CSC format): `col_ptr[j]..col_ptr[j+1]`
    /// are the row indices of non-zeros in column `j` of `L`.
    pub col_ptr: Vec<usize>,
    /// Row indices for the non-zeros in `L` (sorted within each column).
    pub row_idx: Vec<usize>,
    /// Dimension of the matrix.
    pub n: usize,
}

/// Compute the symbolic factorization of `K`.
pub fn analyze(_k: &SymCsrMatrix) -> Result<SymbolicCholesky> {
    // TODO: implement
    // 1. Compute elimination tree (etree)
    // 2. Post-order the etree (improves supernodal efficiency)
    // 3. Compute column counts for L
    // 4. Build col_ptr from column counts
    // 5. Fill row_idx using reach() in the etree
    let n = _k.n;
    Ok(SymbolicCholesky {
        parent:  vec![n; n], // identity: every node is its own root
        col_ptr: vec![0; n + 1],
        row_idx: Vec::new(),
        n,
    })
}