//! Sparse graph representation for ordering algorithms.
//!
//! [`Graph`] is built from a [`SymCsrMatrix`] and exposes the adjacency
//! structure (off-diagonal neighbours only — the diagonal is not a graph
//! edge) needed by RCM and, later, AMD.
//!
//! Because `SymCsrMatrix` stores only the upper triangle, neighbour
//! lists must reconstruct both directions.  We do this once at
//! construction and store the full (symmetric) adjacency so that
//! neighbour queries are O(degree) with no extra work per call.

use sparse::{SparseScalar, SymCsrMatrix};

/// Undirected sparse graph derived from the sparsity pattern of a
/// symmetric matrix.
///
/// Nodes are DOF indices `0..n`.  An edge `(i, j)` exists iff
/// `K[i,j] != 0` and `i != j` (the diagonal is not an edge).
///
/// Stored as a CSR-style adjacency list: `neighbours[i]` is a sorted
/// slice of the neighbours of node `i`, in ascending index order.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Flat neighbour array (all adjacency lists concatenated).
    neighbours: Vec<usize>,
    /// `adj_ptr[i]..adj_ptr[i+1]` is the neighbour slice for node `i`.
    adj_ptr:    Vec<usize>,
    /// Number of nodes.
    pub n: usize,
}

impl Graph {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build from the sparsity pattern of a symmetric matrix.
    ///
    /// Only off-diagonal entries are treated as graph edges.
    /// Both directions `(i→j)` and `(j→i)` are recorded even though
    /// `SymCsrMatrix` stores only `i < j`.
    pub fn from_sym<T>(k: &SymCsrMatrix<T>) -> Self where T: SparseScalar {
        let n = k.n;

        // Count neighbours per node — two passes to avoid reallocation.
        // Upper-triangle entry (i, j) with j != i contributes one
        // neighbour to node i and one to node j.
        let mut degree = vec![0usize; n];
        for row in 0..n {
            let start = k.row_ptr()[row];
            let end   = k.row_ptr()[row + 1];
            for &col in &k.col_idx()[start..end] {
                if col != row {
                    degree[row] += 1; // edge i→j
                    degree[col] += 1; // edge j→i (mirrored)
                }
            }
        }

        // Build adj_ptr from degree counts
        let mut adj_ptr = Vec::with_capacity(n + 1);
        adj_ptr.push(0usize);
        for &d in &degree {
            adj_ptr.push(adj_ptr.last().unwrap() + d);
        }

        let total_edges = *adj_ptr.last().unwrap();
        let mut neighbours = vec![usize::MAX; total_edges];
        // Use cursor array to track insertion position per node
        let mut cursor = adj_ptr[..n].to_vec();

        // Fill neighbours
        for row in 0..n {
            let start = k.row_ptr()[row];
            let end   = k.row_ptr()[row + 1];
            for &col in &k.col_idx()[start..end] {
                if col != row {
                    neighbours[cursor[row]] = col;
                    cursor[row] += 1;
                    neighbours[cursor[col]] = row;
                    cursor[col] += 1;
                }
            }
        }

        // Sort each neighbour list — required for deterministic BFS
        // and for degree-ordered enqueuing in RCM
        for i in 0..n {
            let start = adj_ptr[i];
            let end   = adj_ptr[i + 1];
            neighbours[start..end].sort_unstable();
        }

        Graph { neighbours, adj_ptr, n }
    }

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    /// Degree of node `i` (number of off-diagonal neighbours).
    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        self.adj_ptr[i + 1] - self.adj_ptr[i]
    }

    /// Neighbours of node `i`, in ascending index order.
    #[inline]
    pub fn neighbours(&self, i: usize) -> &[usize] {
        let start = self.adj_ptr[i];
        let end   = self.adj_ptr[i + 1];
        &self.neighbours[start..end]
    }

    /// Iterator over `(node, degree)` pairs for all nodes, sorted by
    /// ascending degree.  Used to find the minimum-degree starting node.
    pub fn nodes_by_degree(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut pairs: Vec<(usize, usize)> = (0..self.n)
            .map(|i| (i, self.degree(i)))
            .collect();
        pairs.sort_unstable_by_key(|&(_, d)| d);
        pairs.into_iter()
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::SymCsrMatrix;

    /// Path graph: 0 — 1 — 2 — 3
    fn path4() -> Graph {
        let pattern = vec![
            vec![0usize, 1],       // row 0: diagonal + edge to 1
            vec![1usize, 2],       // row 1: diagonal + edge to 2
            vec![2usize, 3],       // row 2: diagonal + edge to 3
            vec![3usize],          // row 3: diagonal only
        ];
        let mut m = SymCsrMatrix::from_pattern(4, &pattern).unwrap();
        // values don't matter for graph structure — set diagonal to 1
        for i in 0..4 { m.set_value(i, i, 1.0).unwrap(); }
        m.set_value(0, 1, 1.0).unwrap();
        m.set_value(1, 2, 1.0).unwrap();
        m.set_value(2, 3, 1.0).unwrap();
        Graph::from_sym(&m)
    }

    #[test]
    fn degree_path_graph() {
        let g = path4();
        assert_eq!(g.degree(0), 1); // endpoint
        assert_eq!(g.degree(1), 2); // interior
        assert_eq!(g.degree(2), 2); // interior
        assert_eq!(g.degree(3), 1); // endpoint
    }

    #[test]
    fn neighbours_path_graph() {
        let g = path4();
        assert_eq!(g.neighbours(0), &[1]);
        assert_eq!(g.neighbours(1), &[0, 2]);
        assert_eq!(g.neighbours(2), &[1, 3]);
        assert_eq!(g.neighbours(3), &[2]);
    }

    #[test]
    fn nodes_by_degree_ascending() {
        let g = path4();
        let degrees: Vec<usize> = g.nodes_by_degree().map(|(_, d)| d).collect();
        // should be non-decreasing
        for w in degrees.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn isolated_node_has_degree_zero() {
        // 2×2 diagonal matrix — no off-diagonal entries
        let pattern = vec![vec![0usize], vec![1usize]];
        let mut m = SymCsrMatrix::from_pattern(2, &pattern).unwrap();
        m.set_value(0, 0, 1.0).unwrap();
        m.set_value(1, 1, 1.0).unwrap();
        let g = Graph::from_sym(&m);
        assert_eq!(g.degree(0), 0);
        assert_eq!(g.degree(1), 0);
    }

    #[test]
    fn complete_graph_k4() {
        // K4: every pair connected
        let pattern = vec![
            vec![0usize, 1, 2, 3],
            vec![1usize, 2, 3],
            vec![2usize, 3],
            vec![3usize],
        ];
        let mut m = SymCsrMatrix::from_pattern(4, &pattern).unwrap();
        for i in 0..4 { m.set_value(i, i, 1.0).unwrap(); }
        for i in 0..4 { for j in (i+1)..4 { m.set_value(i, j, 1.0).unwrap(); } }
        let g = Graph::from_sym(&m);
        // every node has degree 3
        for i in 0..4 { assert_eq!(g.degree(i), 3); }
    }
}