//! Symbolic Cholesky factorization.
//!
//! Computes the sparsity pattern of the Cholesky factor `L` from the
//! sparsity pattern of `K` alone -- no floating-point work involved.
//!
//! ## Algorithm
//!
//! ### Step 1 -- Elimination tree (Davis 2006, Algorithm 4.2)
//!
//! The **elimination tree** encodes the dependency structure of the
//! factorization.  For column `j`:
//!
//! ```text
//! parent[j] = min { i > j : L[i,j] != 0 }
//! ```
//!
//! A root column has `parent[j] = n` (sentinel for "no parent").
//!
//! Computed from the UPPER triangle of K in CSC format using ancestor
//! path compression (Liu 1986): for each column `j`, walk up from every
//! row `i < j` (an upper-triangle entry), compressing ancestor pointers
//! toward `j` and recording the first parent seen.
//!
//! ### Step 2 -- Sparsity pattern of L (children-propagation)
//!
//! The correct characterisation (Parter 1961, Rose 1972):
//! `L[i,j] != 0` (for `i > j`) iff there exists a path from `i` to `j`
//! in the graph of `K` using only nodes `< j` as intermediaries.
//!
//! Efficient computation via the elimination tree:
//!
//! ```text
//! for j = 0..n:
//!     reach[j] = { i > j : K[i,j] != 0 }    // direct lower-triangle entries
//!     for each child c of j in the etree:     // parent[c] == j
//!         reach[j] |= { r in reach[c] : r > j }
//!     pattern(L[:,j]) = {j} | reach[j]
//! ```
//!
//! The child-propagation step is always applied (no K[j,c] guard), because
//! `parent[c] = j` guarantees `L[j,c] != 0` by the etree definition -- even
//! when `K[j,c] = 0` (i.e., `L[j,c]` is a fill entry).  Skipping this for
//! structural zeros in K produces an underestimated fill pattern.
//!
//! ## References
//! - Liu, J. W. H. (1986). "A compact row storage scheme for Cholesky
//!   factors using elimination trees." ACM TOMS 12(2).
//! - Davis, T. A. (2006). "Direct Methods for Sparse Linear Systems."
//!   SIAM. Ch. 4.

use crate::error::Result;
use sparse::CscMatrix;

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

    /// Dimension of the matrix (`K` is `n x n`).
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
/// `k_csc` must be the full (both-triangle) CSC representation of the
/// permuted SPD matrix, produced by `sym_to_csc(perm.permute_sym(k))`.
/// No floating-point arithmetic is performed -- only the sparsity pattern
/// is analysed.
///
/// The returned [`SymbolicCholesky`] should be cached and reused for
/// every subsequent call to [`super::numeric::factorize`] as long as
/// the topology (non-zero pattern) of `K` does not change.
pub fn analyze(k_csc: &CscMatrix) -> Result<SymbolicCholesky> {
    let n = k_csc.ncols;
    let parent = elimination_tree(k_csc);
    let children = build_children(&parent, n);
    let (col_ptr, row_idx) = fill_pattern(k_csc, &parent, &children);
    Ok(SymbolicCholesky { parent, col_ptr, row_idx, n })
}

// -----------------------------------------------------------------
// Step 1 -- Elimination tree (Davis 2006, Algorithm 4.2)
// -----------------------------------------------------------------

/// Compute the elimination tree from the upper triangle of the CSC matrix.
///
/// For each column `j`, we look at rows `i < j` (upper-triangle entries)
/// and walk the path-compressed ancestor chain from `i` toward `j`,
/// recording `parent[node] = j` for the first unset ancestor and
/// compressing the path.
///
/// `ancestor[node]` is initialised to `node` itself (each node starts as
/// its own root).  The loop terminates when `ancestor[r] == j` (we have
/// reached j's component).
fn elimination_tree(k_csc: &CscMatrix) -> Vec<usize> {
    let n = k_csc.ncols;
    let mut parent   = vec![n; n];
    let mut ancestor = (0..n).collect::<Vec<_>>(); // ancestor[j] = j initially

    let col_ptr = k_csc.col_ptr();
    let row_idx = k_csc.row_idx();

    for j in 0..n {
        let start = col_ptr[j];
        let end   = col_ptr[j + 1];
        for &i in &row_idx[start..end] {
            if i >= j {
                continue; // only upper-triangle entries: i < j
            }
            // Walk up the ancestor chain from i toward j, compressing as we go.
            let mut r = i;
            while ancestor[r] != j {
                let next = ancestor[r];
                ancestor[r] = j;
                if parent[r] == n {
                    parent[r] = j;
                }
                r = next;
            }
        }
    }

    parent
}

// -----------------------------------------------------------------
// Step 2 -- children list + fill pattern
// -----------------------------------------------------------------

/// Build children lists from the elimination tree.
///
/// `children[j]` = list of nodes `c` where `parent[c] == j`.
/// All children have `c < j` (since `parent[c] > c` always).
fn build_children(parent: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); n];
    for c in 0..n {
        let p = parent[c];
        if p < n {
            children[p].push(c);
        }
    }
    children
}

/// Compute the CSC sparsity pattern of `L` using children-propagation.
///
/// For each column `j`:
/// 1. Seed `reach[j]` with the direct lower-triangle entries of `K[:,j]`
///    (rows `i > j` where `K[i,j] != 0`).
/// 2. For every child `c` of `j` in the etree (`parent[c] == j`),
///    propagate `reach[c]` into `reach[j]` (keeping only rows `> j`).
///    This step is applied unconditionally -- the etree definition
///    guarantees `L[j,c] != 0` so `c`'s fill always affects column `j`.
/// 3. `pattern(L[:,j]) = {j} | reach[j]`.
///
/// Returns `(col_ptr, row_idx)` with row indices sorted ascending per column.
fn fill_pattern(
    k_csc:    &CscMatrix,
    _parent:   &[usize],
    children: &[Vec<usize>],
) -> (Vec<usize>, Vec<usize>) {
    let n = k_csc.ncols;
    let col_ptr_k = k_csc.col_ptr();
    let row_idx_k = k_csc.row_idx();

    // reach[j] stores the off-diagonal rows > j of L[:,j].
    // We use a Vec<Vec<usize>> here; for large n a bitset would be faster,
    // but correctness is the priority and FEM matrices are sparse.
    let mut reach: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Process columns left to right so reach[c] is complete before we use it.
    for j in 0..n {
        // Step 1: seed from direct lower-triangle K entries (rows i > j in col j).
        let start = col_ptr_k[j];
        let end   = col_ptr_k[j + 1];
        for &i in &row_idx_k[start..end] {
            if i > j {
                reach[j].push(i);
            }
        }

        // Step 2: propagate from each child c (no K[j,c] guard).
        // parent[c] == j means L[j,c] != 0 (fill or original), so c's reach
        // feeds into j's reach.
        for ci in 0..children[j].len() {
            let c = children[j][ci];
            // Collect nodes from reach[c] that are > j.
            // We need to swap-out reach[c] temporarily to avoid borrow conflict.
            let child_reach = std::mem::take(&mut reach[c]);
            for &r in &child_reach {
                if r > j {
                    reach[j].push(r);
                }
            }
            reach[c] = child_reach;
        }

        // Deduplicate and sort reach[j].
        reach[j].sort_unstable();
        reach[j].dedup();
    }

    // Build col_ptr and row_idx from reach.
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0usize);
    for j in 0..n {
        let count = 1 + reach[j].len(); // diagonal + off-diagonal
        col_ptr.push(col_ptr.last().unwrap() + count);
    }

    let nnz = *col_ptr.last().unwrap();
    let mut row_idx = vec![0usize; nnz];

    for j in 0..n {
        let base = col_ptr[j];
        // Diagonal first.
        row_idx[base] = j;
        // Then sorted off-diagonal rows (already sorted by sort+dedup above).
        for (k, &r) in reach[j].iter().enumerate() {
            row_idx[base + 1 + k] = r;
        }
        // The column is already sorted: j < reach[j][0] < reach[j][1] < ...
    }

    (col_ptr, row_idx)
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::{CooBuilder, SymCsrMatrix};
    use sparse::convert::sym_to_csc;

    fn to_csc(sym: &SymCsrMatrix) -> CscMatrix {
        sym_to_csc(sym)
    }

    fn tridiag(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

    fn diagonal(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n { coo.add(i, i, (i + 1) as f64); }
        coo.build_sym().unwrap()
    }

    fn dense_3() -> SymCsrMatrix {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 4.0); coo.add(0, 1, 1.0); coo.add(0, 2, 1.0);
        coo.add(1, 1, 4.0); coo.add(1, 2, 1.0);
        coo.add(2, 2, 4.0);
        coo.build_sym().unwrap()
    }

    /// Star with hub as the LAST node (hub = n-1). Previously failed with
    /// the children-DFS approach; must also pass with the new algorithm.
    fn star_hub_last(n: usize) -> SymCsrMatrix {
        let hub = n - 1;
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n   { coo.add(i, i, (n as f64) + 1.0); }
        for i in 0..hub { coo.add(i, hub, -1.0); }
        coo.build_sym().unwrap()
    }

    /// k x k grid Laplacian, row-major numbering.
    fn grid_laplacian(k: usize) -> SymCsrMatrix {
        let n = k * k;
        let mut coo = CooBuilder::new(n, n);
        for r in 0..k {
            for c in 0..k {
                let i = r * k + c;
                coo.add(i, i, 4.0);
                if c + 1 < k { coo.add(i, r * k + c + 1, -1.0); }
                if r + 1 < k { coo.add(i, (r + 1) * k + c, -1.0); }
            }
        }
        coo.build_sym().unwrap()
    }

    // ---- elimination tree ----

    #[test]
    fn etree_diagonal_all_roots() {
        let parent = elimination_tree(&to_csc(&diagonal(4)));
        assert_eq!(parent, vec![4, 4, 4, 4]);
    }

    #[test]
    fn etree_tridiag_3_chain() {
        let parent = elimination_tree(&to_csc(&tridiag(3)));
        assert_eq!(parent[0], 1);
        assert_eq!(parent[1], 2);
        assert_eq!(parent[2], 3);
    }

    #[test]
    fn etree_tridiag_n_is_chain() {
        let n = 8;
        let parent = elimination_tree(&to_csc(&tridiag(n)));
        for j in 0..(n - 1) {
            assert_eq!(parent[j], j + 1, "parent[{j}] should be {}", j + 1);
        }
        assert_eq!(parent[n - 1], n);
    }

    #[test]
    fn etree_dense_3() {
        let parent = elimination_tree(&to_csc(&dense_3()));
        assert_eq!(parent[0], 1);
        assert_eq!(parent[1], 2);
        assert_eq!(parent[2], 3);
    }

    // ---- sparsity pattern ----

    #[test]
    fn pattern_diagonal_no_fill() {
        let sym = analyze(&to_csc(&diagonal(3))).unwrap();
        assert_eq!(sym.nnz_l(), 3);
        assert_eq!(sym.col_ptr, vec![0, 1, 2, 3]);
        assert_eq!(sym.row_idx, vec![0, 1, 2]);
    }

    #[test]
    fn pattern_tridiag_3_bidiagonal() {
        let sym = analyze(&to_csc(&tridiag(3))).unwrap();
        assert_eq!(sym.nnz_l(), 5);
        assert_eq!(&sym.row_idx[sym.col_ptr[0]..sym.col_ptr[1]], &[0, 1]);
        assert_eq!(&sym.row_idx[sym.col_ptr[1]..sym.col_ptr[2]], &[1, 2]);
        assert_eq!(&sym.row_idx[sym.col_ptr[2]..sym.col_ptr[3]], &[2]);
    }

    #[test]
    fn pattern_tridiag_4_nnz_7() {
        let sym = analyze(&to_csc(&tridiag(4))).unwrap();
        assert_eq!(sym.nnz_l(), 7);
    }

    #[test]
    fn pattern_dense_3_full_lower() {
        let sym = analyze(&to_csc(&dense_3())).unwrap();
        assert_eq!(sym.nnz_l(), 6);
        assert_eq!(&sym.row_idx[sym.col_ptr[0]..sym.col_ptr[1]], &[0, 1, 2]);
        assert_eq!(&sym.row_idx[sym.col_ptr[1]..sym.col_ptr[2]], &[1, 2]);
        assert_eq!(&sym.row_idx[sym.col_ptr[2]..sym.col_ptr[3]], &[2]);
    }

    /// Regression: star with hub last. Each spoke column should have exactly
    /// {col, hub} in its pattern -- no cross-spoke fill.
    #[test]
    fn star_hub_last_fill() {
        let n = 5;
        let hub = n - 1;
        let sym = analyze(&to_csc(&star_hub_last(n))).unwrap();
        for col in 0..hub {
            let s = sym.col_ptr[col];
            let e = sym.col_ptr[col + 1];
            let rows = &sym.row_idx[s..e];
            assert_eq!(rows, &[col, hub],
                "col {col}: expected [{col}, {hub}], got {rows:?}");
        }
        let s = sym.col_ptr[hub];
        let e = sym.col_ptr[hub + 1];
        assert_eq!(&sym.row_idx[s..e], &[hub]);
    }

    /// Grid Laplacian: fill patterns must include fill entries induced by
    /// the chain elimination order (e.g., L[5,3] != 0 for the 3x3 grid).
    #[test]
    fn grid_3x3_fill_not_underestimated() {
        let sym = analyze(&to_csc(&grid_laplacian(3))).unwrap();
        let n = 9;
        // col 3 should contain row 5 (fill from path 5-2-1-0-3)
        let col3_rows: Vec<usize> = sym.row_idx[sym.col_ptr[3]..sym.col_ptr[4]].to_vec();
        assert!(col3_rows.contains(&5),
            "col 3 of L must contain row 5 (fill entry): col3={col3_rows:?}");
        // col 6 should contain row 8 (fill from path 8-5-4-3-6... similar)
        let col6_rows: Vec<usize> = sym.row_idx[sym.col_ptr[6]..sym.col_ptr[7]].to_vec();
        assert!(col6_rows.contains(&8),
            "col 6 of L must contain row 8 (fill entry): col6={col6_rows:?}");
        let _ = n;
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
    fn invariants_diagonal()        { check_invariants(&analyze(&to_csc(&diagonal(5))).unwrap()); }
    #[test]
    fn invariants_tridiag_small()   { check_invariants(&analyze(&to_csc(&tridiag(3))).unwrap()); }
    #[test]
    fn invariants_tridiag_large()   { check_invariants(&analyze(&to_csc(&tridiag(20))).unwrap()); }
    #[test]
    fn invariants_dense_3()         { check_invariants(&analyze(&to_csc(&dense_3())).unwrap()); }
    #[test]
    fn invariants_star_hub_last()   { check_invariants(&analyze(&to_csc(&star_hub_last(6))).unwrap()); }
    #[test]
    fn invariants_grid_3x3()        { check_invariants(&analyze(&to_csc(&grid_laplacian(3))).unwrap()); }
    #[test]
    fn invariants_grid_5x5()        { check_invariants(&analyze(&to_csc(&grid_laplacian(5))).unwrap()); }

    #[test]
    fn n_field_matches_input() {
        let sym = analyze(&to_csc(&tridiag(7))).unwrap();
        assert_eq!(sym.n, 7);
    }
}
