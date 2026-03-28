//! Symbolic Cholesky factorization.
//!
//! Computes the sparsity pattern of the Cholesky factor `L` from the
//! sparsity pattern of `K` alone — no floating-point work involved.
//!
//! ## Algorithm
//!
//! ### Step 1 — Elimination tree
//!
//! The **elimination tree** (etree) encodes the dependency structure of
//! the factorization.  For column `j`:
//!
//! ```text
//! parent[j] = min { i > j : L[i,j] ≠ 0 }
//! ```
//!
//! i.e. the row index of the first sub-diagonal non-zero in column `j` of `L`.
//! A root column has `parent[j] = n` (sentinel for "no parent").
//!
//! We compute this in O(n α(n)) using **path compression** (Liu 1986):
//! for each row `j`, walk up the etree from every off-diagonal entry
//! `(j, i)` in `K` (with `i > j`), compressing paths as we go.
//!
//! ### Step 2 — Sparsity pattern of L
//!
//! The non-zero rows in column `j` of `L` are:
//! - the diagonal `j`,
//! - for every off‑diagonal entry `(j, i)` with `i > j` (stored in row `j` of `K`),
//!   all nodes in the **subtree** of `i` (descendants) that are `≥ j`.
//!
//! This is the classic reach algorithm (Liu 1986) using the elimination tree.
//! We compute it by pre‑building children lists and traversing each subtree
//! with a stack, marking visited nodes per column.
//!
//! ## References
//! - Liu, J. W. H. (1986). "A compact row storage scheme for Cholesky
//!   factors using elimination trees." ACM TOMS 12(2).
//! - Davis, T. A. (2006). "Direct Methods for Sparse Linear Systems."
//!   SIAM. Ch. 4–5.

use crate::error::Result;
use sparse::SymCsrMatrix;

// -----------------------------------------------------------------
// Public types
// -----------------------------------------------------------------

/// Result of the symbolic Cholesky phase.
///
/// Stores the elimination tree and the CSC sparsity pattern of `L`.
/// Both are computed from the pattern of `K` alone and are reused
/// across all Newton iterations as long as the topology is unchanged.
pub struct SymbolicCholesky {
    /// Elimination tree: `parent[j]` is the parent of column `j`.
    /// `parent[j] == n` means column `j` is a root.
    pub parent: Vec<usize>,

    /// Column pointers for `L` in CSC format.
    /// `col_ptr[j]..col_ptr[j+1]` indexes into `row_idx` for column `j`.
    /// Includes the diagonal entry.
    pub col_ptr: Vec<usize>,

    /// Row indices of the non-zeros in `L`, sorted ascending within
    /// each column.  `row_idx.len() == col_ptr[n] == nnz(L)`.
    pub row_idx: Vec<usize>,

    /// Dimension of the matrix (`K` is `n × n`).
    pub n: usize,
}

impl SymbolicCholesky {
    /// Number of non-zeros in `L` (including the diagonal).
    #[inline]
    pub fn nnz_l(&self) -> usize {
        self.row_idx.len()
    }
}

// -----------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------

/// Compute the symbolic Cholesky factorization of `K`.
///
/// `K` must be symmetric positive definite, stored in upper-triangle
/// CSR format (`SymCsrMatrix`).  No floating-point arithmetic is
/// performed — only the sparsity pattern is analysed.
///
/// The returned [`SymbolicCholesky`] should be cached and reused for
/// every subsequent call to [`super::numeric::factorize`] as long as
/// the topology (non-zero pattern) of `K` does not change.
pub fn analyze(k: &SymCsrMatrix) -> Result<SymbolicCholesky> {
    let n = k.n;
    let parent = elimination_tree(k);
    let children = build_children(&parent, n);
    let counts = column_counts(k, &children);
    let col_ptr = counts_to_ptr(&counts, n);
    let row_idx = fill_row_idx(k, &children, &col_ptr);
    Ok(SymbolicCholesky { parent, col_ptr, row_idx, n })
}

// -----------------------------------------------------------------
// Step 1 — elimination tree with path compression
// -----------------------------------------------------------------

/// Compute the elimination tree of the upper-triangle matrix `K`.
///
/// Returns `parent[0..n]` where `parent[j] = n` denotes a root.
///
/// ### Path-compression algorithm (Liu 1986)
///
/// For each row `j` we look at the off-diagonal upper-triangle entries
/// `(j, i)` with `i > j`.  In column-oriented terms, `j` is a row
/// index that appears below the diagonal in column `i` of `K`.
///
/// We walk up the etree from `j`, compressing the path, until we
/// reach a node whose ancestor is already `>= i`.  At that point we
/// set `parent[node] = i` (unless already set).
fn elimination_tree(k: &SymCsrMatrix) -> Vec<usize> {
    let n = k.n;
    // Use 1‑indexed arrays to keep sentinel 0
    let mut parent1 = vec![0; n + 1];   // parent1[0] unused
    let mut ancestor = vec![0; n + 1];  // ancestor[0] unused

    for j in 0..n {
        let j1 = j + 1;
        let start = k.row_ptr()[j];
        let end = k.row_ptr()[j + 1];

        for &i in &k.col_idx()[start..end] {
            if i == j {
                continue;
            }
            let i1 = i + 1;
            let mut node = j1;
            while ancestor[node] < i1 {
                let next = ancestor[node];
                ancestor[node] = i1;
                if parent1[node] == 0 {
                    parent1[node] = i1;
                }
                if next == 0 {
                    break;
                }
                node = next;
            }
        }
    }

    // Convert back to 0‑indexed with n as root sentinel
    let mut parent = vec![n; n];
    for j in 0..n {
        let p = parent1[j + 1];
        if p != 0 {
            parent[j] = p - 1;
        }
    }
    parent
}

// -----------------------------------------------------------------
// Step 2 helpers – children lists and subtree traversal
// -----------------------------------------------------------------

/// Build children lists from the elimination tree.
fn build_children(parent: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); n];
    for j in 0..n {
        let p = parent[j];
        if p < n {
            children[p].push(j);
        }
    }
    // Sort for deterministic order (not strictly necessary, but good practice)
    for ch in &mut children {
        ch.sort_unstable();
    }
    children
}

/// Compute nnz per column of `L` (including diagonal).
///
/// For each column `j`:
/// - start with 1 for the diagonal.
/// - for each off‑diagonal `(j, i)` with `i > j`, traverse the subtree of `i`
///   (descendants) and count all nodes `≥ j` that are not already counted.
fn column_counts(k: &SymCsrMatrix, children: &[Vec<usize>]) -> Vec<usize> {
    let n = k.n;
    let mut count = vec![1usize; n];
    let mut mark = vec![n; n]; // mark[node] == col → already counted

    for col in 0..n {
        mark[col] = col; // diagonal is always counted

        let start = k.row_ptr()[col];
        let end = k.row_ptr()[col + 1];
        for &i in &k.col_idx()[start..end] {
            if i > col {
                let mut stack = vec![i];
                while let Some(node) = stack.pop() {
                    if mark[node] != col {
                        mark[node] = col;
                        count[col] += 1;
                        // push children that are ≥ col (descendants have smaller indices)
                        for &child in &children[node] {
                            if child >= col {
                                stack.push(child);
                            }
                        }
                    }
                }
            }
        }
    }

    count
}

/// Build `col_ptr` (length `n+1`) from per-column counts.
fn counts_to_ptr(counts: &[usize], n: usize) -> Vec<usize> {
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0usize);
    for &c in counts {
        col_ptr.push(col_ptr.last().unwrap() + c);
    }
    col_ptr
}

/// Fill the `row_idx` array by performing the reach for each column.
///
/// For each column `j`:
/// - insert diagonal `j`
/// - for each off‑diagonal `(j, i)` with `i > j`, traverse the subtree of `i`
///   (descendants) and add all nodes `≥ j` that are not already inserted.
///
/// Row indices are sorted ascending within each column.
fn fill_row_idx(
    k: &SymCsrMatrix,
    children: &[Vec<usize>],
    col_ptr: &[usize],
) -> Vec<usize> {
    let n = k.n;
    let nnz = *col_ptr.last().unwrap();
    let mut row_idx = vec![0usize; nnz];
    let mut cursor: Vec<usize> = col_ptr[..n].to_vec(); // insertion cursors
    let mut mark = vec![n; n]; // mark[node] == col → already inserted

    for col in 0..n {
        // Insert diagonal
        row_idx[cursor[col]] = col;
        cursor[col] += 1;
        mark[col] = col;

        let start = k.row_ptr()[col];
        let end = k.row_ptr()[col + 1];
        for &i in &k.col_idx()[start..end] {
            if i > col {
                let mut stack = vec![i];
                while let Some(node) = stack.pop() {
                    if mark[node] != col {
                        mark[node] = col;
                        row_idx[cursor[col]] = node;
                        cursor[col] += 1;
                        for &child in &children[node] {
                            if child >= col {
                                stack.push(child);
                            }
                        }
                    }
                }
            }
        }

        // Sort the row indices for this column
        let s = col_ptr[col];
        let e = col_ptr[col + 1];
        row_idx[s..e].sort_unstable();
    }

    row_idx
}
// -----------------------------------------------------------------
// Tests (unchanged)
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::{SymCsrMatrix, CooBuilder};

    // ---- matrix helpers ----

    fn tridiag(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
        }
        for i in 0..(n - 1) {
            coo.add(i, i + 1, -1.0);
        }
        coo.build_sym().unwrap()
    }

    fn diagonal(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n {
            coo.add(i, i, (i + 1) as f64);
        }
        coo.build_sym().unwrap()
    }

    fn dense_3() -> SymCsrMatrix {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 4.0);
        coo.add(0, 1, 1.0);
        coo.add(0, 2, 1.0);
        coo.add(1, 1, 4.0);
        coo.add(1, 2, 1.0);
        coo.add(2, 2, 4.0);
        coo.build_sym().unwrap()
    }

    // ---- elimination tree ----

    #[test]
    fn etree_diagonal_all_roots() {
        let parent = elimination_tree(&diagonal(4));
        assert_eq!(parent, vec![4, 4, 4, 4]);
    }

    #[test]
    fn etree_tridiag_3_chain() {
        let parent = elimination_tree(&tridiag(3));
        assert_eq!(parent[0], 1);
        assert_eq!(parent[1], 2);
        assert_eq!(parent[2], 3);
    }

    #[test]
    fn etree_tridiag_n_is_chain() {
        let n = 8;
        let parent = elimination_tree(&tridiag(n));
        for j in 0..(n - 1) {
            assert_eq!(parent[j], j + 1, "parent[{j}] should be {}", j + 1);
        }
        assert_eq!(parent[n - 1], n);
    }

    #[test]
    fn etree_dense_3() {
        let parent = elimination_tree(&dense_3());
        assert_eq!(parent[0], 1);
        assert_eq!(parent[1], 2);
        assert_eq!(parent[2], 3);
    }

    // ---- sparsity pattern ----

    #[test]
    fn pattern_diagonal_no_fill() {
        let sym = analyze(&diagonal(3)).unwrap();
        assert_eq!(sym.nnz_l(), 3);
        assert_eq!(sym.col_ptr, vec![0, 1, 2, 3]);
        assert_eq!(sym.row_idx, vec![0, 1, 2]);
    }

    #[test]
    fn pattern_tridiag_3_bidiagonal() {
        let sym = analyze(&tridiag(3)).unwrap();
        assert_eq!(sym.nnz_l(), 5);
        assert_eq!(&sym.row_idx[sym.col_ptr[0]..sym.col_ptr[1]], &[0, 1]);
        assert_eq!(&sym.row_idx[sym.col_ptr[1]..sym.col_ptr[2]], &[1, 2]);
        assert_eq!(&sym.row_idx[sym.col_ptr[2]..sym.col_ptr[3]], &[2]);
    }

    #[test]
    fn pattern_tridiag_4_nnz_7() {
        let sym = analyze(&tridiag(4)).unwrap();
        assert_eq!(sym.nnz_l(), 7);
    }

    #[test]
    fn pattern_dense_3_full_lower() {
        let sym = analyze(&dense_3()).unwrap();
        assert_eq!(sym.nnz_l(), 6);
        assert_eq!(&sym.row_idx[sym.col_ptr[0]..sym.col_ptr[1]], &[0, 1, 2]);
        assert_eq!(&sym.row_idx[sym.col_ptr[1]..sym.col_ptr[2]], &[1, 2]);
        assert_eq!(&sym.row_idx[sym.col_ptr[2]..sym.col_ptr[3]], &[2]);
    }

    // ---- structural invariants ----

    fn check_invariants(sym: &SymbolicCholesky) {
        let n = sym.n;
        assert_eq!(sym.col_ptr.len(), n + 1);
        assert_eq!(sym.col_ptr[0], 0);
        assert_eq!(*sym.col_ptr.last().unwrap(), sym.row_idx.len());

        for col in 0..n {
            let s = sym.col_ptr[col];
            let e = sym.col_ptr[col + 1];
            let rows = &sym.row_idx[s..e];
            assert!(rows.contains(&col), "diagonal missing in col {col}");
            for &row in rows {
                assert!(row >= col, "upper-triangle entry ({row},{col}) in L");
            }
            for w in rows.windows(2) {
                assert!(w[0] < w[1], "rows not sorted in col {col}");
            }
        }
    }

    #[test]
    fn invariants_diagonal() {
        check_invariants(&analyze(&diagonal(5)).unwrap());
    }

    #[test]
    fn invariants_tridiag_small() {
        check_invariants(&analyze(&tridiag(3)).unwrap());
    }

    #[test]
    fn invariants_tridiag_large() {
        check_invariants(&analyze(&tridiag(20)).unwrap());
    }

    #[test]
    fn invariants_dense_3() {
        check_invariants(&analyze(&dense_3()).unwrap());
    }

    #[test]
    fn n_field_matches_input() {
        let sym = analyze(&tridiag(7)).unwrap();
        assert_eq!(sym.n, 7);
    }
}