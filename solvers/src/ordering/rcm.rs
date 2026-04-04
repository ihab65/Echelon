//! Reverse Cuthill–McKee (RCM) ordering for fill reduction.
//!
//! ## Background
//!
//! Cholesky factorization of `K` produces a lower triangular factor `L`
//! such that `K = L Lᵀ`.  For sparse `K`, `L` can have far more non-zeros
//! than `K` due to **fill-in**: positions that are zero in `K` but non-zero
//! in `L`.  In the worst case `L` is dense.
//!
//! Reordering the DOFs before factorization — permuting rows and columns
//! simultaneously — changes the fill pattern without changing the solution.
//! RCM is a cheap, graph-based heuristic that reliably reduces bandwidth
//! and fill for FEM stiffness matrices **when the natural ordering is poor**.
//!
//! ## Algorithm
//!
//! ### Step 1 — pseudo-peripheral node
//! A **peripheral node** is one at maximum eccentricity (graph diameter).
//! Exact computation is O(n·m); we use the standard two-BFS approximation:
//!
//! 1. Pick the minimum-degree unvisited node `s` as a seed.
//! 2. BFS from `s`; the last node dequeued, `t`, approximates a peripheral node.
//!
//! ### Step 2 — BFS with degree ordering (Cuthill–McKee)
//! BFS from `t`, enqueueing unvisited neighbours sorted by **ascending degree**
//! at every step.  Low-degree nodes are placed early; high-degree hub nodes
//! are placed late.
//!
//! ### Step 3 — reverse
//! Reversing the CM ordering gives RCM.  Empirically and theoretically
//! (for several graph classes) RCM produces smaller bandwidth than CM.
//!
//! ## Permutation convention
//!
//! Returns a [`Permutation`] `p` where `p[new_index] = old_index`.
//! Apply it to `K` via [`Permutation::permute_sym`] before factorization.
//! This produces `K_perm[i,j] = K[p[i], p[j]]`.
//!
//! ## When RCM helps and when it does not
//!
//! RCM is most effective when the natural ordering is poor: random meshes,
//! star-connected DOF sets, or matrices with large off-diagonal bandwidth.
//!
//! RCM provides **little benefit** when the input is already well-ordered.
//! A 2-D grid in row-major order already has compact band width `nx`; the
//! RCM diagonal-wavefront ordering gives band ≈ `2·nx − 1`, which can
//! *increase* fill.  For such problems, AMD is strongly preferred.
//!
//! ## Correctness guarantee
//!
//! RCM always produces a **valid permutation** (bijection of `0..n`).
//! The permutation and its inverse are applied symmetrically, so the
//! mathematical solution is identical to the unpermuted problem.
//!
//! ## References
//! - Cuthill & McKee (1969), "Reducing the bandwidth of sparse symmetric matrices"
//! - Liu & Sherman (1976), "Comparative analysis of the CM and RCM algorithms"
//! - George & Liu (1981), "Computer Solution of Large Sparse Positive Definite Systems"
//! - Davis (2006), "Direct Methods for Sparse Linear Systems", §7

use std::collections::VecDeque;
use super::graph::Graph;
use super::permutation::Permutation;

// -----------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------

/// Compute the RCM permutation for the graph of a symmetric matrix.
///
/// Takes a [`Graph`] (built from a [`SymCsrMatrix`] via
/// [`Graph::from_sym`]) and returns a [`Permutation`] that can be
/// applied to the matrix before Cholesky factorization.
///
/// For a disconnected graph (rare in FEM but possible) each connected
/// component is ordered independently and the components are
/// concatenated in the final permutation.
///
/// # Panics
/// Panics if `graph.n == 0`.
pub fn rcm(graph: &Graph) -> Permutation {
    assert!(graph.n > 0, "graph must have at least one node");

    let n = graph.n;
    let mut visited = vec![false; n];
    // Final CM order — we collect here then reverse at the end
    let mut cm_order: Vec<usize> = Vec::with_capacity(n);

    // Handle disconnected graphs: repeat until all nodes are visited
    while cm_order.len() < n {
        // Find the seed for this component: minimum-degree unvisited node
        let seed = unvisited_min_degree_node(graph, &visited);

        // BFS from the pseudo-peripheral node of this component
        let peripheral = find_pseudo_peripheral(graph, seed, &visited);
        bfs_degree_ordered(graph, peripheral, &mut visited, &mut cm_order);
    }

    // RCM = reverse of CM
    cm_order.reverse();

    // BFS visits every node exactly once so cm_order is a valid permutation.
    Permutation::new(cm_order)
        .expect("BFS produced an invalid permutation — this is a bug")
}

// -----------------------------------------------------------------
// Step 1 — pseudo-peripheral node
// -----------------------------------------------------------------

/// Find an approximation of a peripheral node reachable from `start`,
/// considering only unvisited nodes.
///
/// Algorithm: BFS from `start`; take the last node enqueued (which is
/// in the last level set of the BFS tree) as the pseudo-peripheral node.
fn find_pseudo_peripheral(
    graph:   &Graph,
    start:   usize,
    visited: &[bool],
) -> usize {
    let n = graph.n;
    let mut local_seen = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut last_node = start;

    queue.push_back(start);
    local_seen[start] = true;

    while let Some(node) = queue.pop_front() {
        last_node = node;
        for &nb in graph.neighbours(node) {
            if !visited[nb] && !local_seen[nb] {
                local_seen[nb] = true;
                queue.push_back(nb);
            }
        }
    }

    last_node
}

// -----------------------------------------------------------------
// Step 2 — degree-ordered BFS
// -----------------------------------------------------------------

/// BFS from `start`, enqueuing unvisited neighbours in ascending-degree
/// order at each step.  Appends visited nodes to `order` and marks them
/// in `visited`.
fn bfs_degree_ordered(
    graph:   &Graph,
    start:   usize,
    visited: &mut Vec<bool>,
    order:   &mut Vec<usize>,
) {
    let mut queue: VecDeque<usize> = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        order.push(node);

        // Collect unvisited neighbours, sort by (degree, index) for
        // determinism when degrees are equal, then enqueue
        let mut unvisited_nbs: Vec<usize> = graph
            .neighbours(node)
            .iter()
            .copied()
            .filter(|&nb| !visited[nb])
            .collect();

        unvisited_nbs.sort_unstable_by_key(|&nb| (graph.degree(nb), nb));

        for nb in unvisited_nbs {
            // Guard: another neighbour in the same iteration may have
            // already enqueued this node via a different path
            if !visited[nb] {
                visited[nb] = true;
                queue.push_back(nb);
            }
        }
    }
}

// -----------------------------------------------------------------
// Helper
// -----------------------------------------------------------------

/// Return the unvisited node with the minimum degree, breaking ties
/// by smallest index for determinism.
fn unvisited_min_degree_node(graph: &Graph, visited: &[bool]) -> usize {
    (0..graph.n)
        .filter(|&i| !visited[i])
        .min_by_key(|&i| (graph.degree(i), i))
        .expect("called with all nodes visited — caller bug")
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::graph::Graph;
    use super::super::permutation::Permutation;
    use sparse::{CooBuilder, SparseScalar, SymCsrMatrix};

    // ---- graph / matrix helpers ----

    fn path_graph(n: usize) -> (Graph, SymCsrMatrix<f64>) {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i, 2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        let m = coo.build_sym().unwrap();
        let g = Graph::from_sym(&m);
        (g, m)
    }

    fn star_graph_matrix(n: usize) -> (Graph, SymCsrMatrix<f64>) {
        // Hub = node 0, spokes = nodes 1..n
        let mut coo = CooBuilder::new(n, n);
        coo.add(0, 0, n as f64);
        for i in 1..n {
            coo.add(i, i, 2.0);
            coo.add(0, i, -1.0);
        }
        let m = coo.build_sym().unwrap();
        let g = Graph::from_sym(&m);
        (g, m)
    }

    /// k×k 2-D grid Laplacian, row-major node numbering.
    fn grid_laplacian(k: usize) -> (Graph, SymCsrMatrix<f64>) {
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
        let m = coo.build_sym().unwrap();
        let g = Graph::from_sym(&m);
        (g, m)
    }

    // ---- validity assertion ----

    fn assert_valid_permutation(p: &Permutation, n: usize) {
        assert_eq!(p.len(), n, "permutation length mismatch");
        let mut seen = vec![false; n];
        for new_i in 0..n {
            let old = p.old_index(new_i);
            assert!(old < n, "out-of-range old-index {old}");
            assert!(!seen[old], "duplicate old-index {old}");
            seen[old] = true;
        }
        assert!(seen.iter().all(|&s| s), "missing old indices");
    }

    // ---- bandwidth helpers ----

    /// Maximum `|new_i − new_j|` over all edges `(i,j)`.
    fn bandwidth_after_permutation(g: &Graph, p: &Permutation) -> usize {
        let inv = p.inverse();
        let inv = &inv; // avoid move into closure

        (0..g.n)
            .flat_map(|old_i| {
                let new_i = inv.old_index(old_i);
                g.neighbours(old_i).iter().map(move |&old_j| {
                    new_i.abs_diff(inv.old_index(old_j))
                })
            })
            .max()
            .unwrap_or(0)
    }

    fn natural_bandwidth(g: &Graph) -> usize {
        (0..g.n)
            .flat_map(|i| g.neighbours(i).iter().map(move |&j| i.abs_diff(j)))
            .max()
            .unwrap_or(0)
    }

    // ---- fill helper ----

    fn nnz_l<T>(m: &SymCsrMatrix<T>) -> usize where T: SparseScalar {
        use crate::LinearSolver;
        let mut solver = crate::cholesky::SparseSolver::<T>::new();
        solver.analyze(&m).unwrap();
        solver.symbolic.as_ref().unwrap().nnz_l()
    }

    // ================================================================
    // Validity tests — these must pass for ALL inputs
    // ================================================================

    #[test]
    fn single_node_valid() {
        let mut coo = CooBuilder::new(1, 1);
        coo.add(0, 0, 1.0);
        let m = coo.build_sym().unwrap();
        let g = Graph::from_sym(&m);
        let p = rcm(&g);
        assert_valid_permutation(&p, 1);
        assert!(p.is_identity());
    }

    #[test]
    fn two_nodes_valid() {
        let (g, _) = path_graph(2);
        let p = rcm(&g);
        assert_valid_permutation(&p, 2);
    }

    #[test]
    fn path4_valid() {
        let (g, _) = path_graph(4);
        let p = rcm(&g);
        assert_valid_permutation(&p, 4);
    }

    #[test]
    fn path_large_valid() {
        let (g, _) = path_graph(200);
        let p = rcm(&g);
        assert_valid_permutation(&p, 200);
    }

    #[test]
    fn star_valid() {
        let (g, _) = star_graph_matrix(12);
        let p = rcm(&g);
        assert_valid_permutation(&p, 12);
    }

    #[test]
    fn grid_valid_all_sizes() {
        for k in [3, 5, 8, 10] {
            let (g, _) = grid_laplacian(k);
            let p = rcm(&g);
            assert_valid_permutation(&p, k * k);
        }
    }

    #[test]
    fn disconnected_graph_all_nodes_covered() {
        // Two disconnected edges: 0-1 and 2-3
        let mut coo = CooBuilder::new(4, 4);
        for i in 0..4 { coo.add(i, i, 2.0); }
        coo.add(0, 1, -1.0);
        coo.add(2, 3, -1.0);
        let m = coo.build_sym().unwrap();
        let g = Graph::from_sym(&m);
        let p = rcm(&g);
        assert_valid_permutation(&p, 4);
    }

    #[test]
    fn isolated_nodes_covered() {
        // 5-node graph: 0-1 connected, 2,3,4 isolated
        let mut coo = CooBuilder::new(5, 5);
        for i in 0..5 { coo.add(i, i, 1.0); }
        coo.add(0, 1, -1.0);
        let m = coo.build_sym().unwrap();
        let g = Graph::from_sym(&m);
        let p = rcm(&g);
        assert_valid_permutation(&p, 5);
    }

    // ================================================================
    // Structural property tests — first node, hub placement
    // ================================================================

    #[test]
    fn path_first_node_is_endpoint() {
        // RCM must start from a peripheral (minimum-degree) node.
        // For a path graph that means degree-1 endpoints.
        let (g, _) = path_graph(10);
        let p = rcm(&g);
        assert_eq!(
            g.degree(p.old_index(0)), 1,
            "first RCM node must be an endpoint (degree 1)"
        );
    }

    #[test]
    fn star_hub_in_second_half() {
        // Hub (node 0, max degree) should not appear in the first half
        // of the RCM ordering — high-degree nodes should be placed late.
        let n = 10;
        let (g, _) = star_graph_matrix(n);
        let p = rcm(&g);
        let inv = p.inverse();
        let hub_new_idx = inv.old_index(0);
        assert!(
            hub_new_idx >= n / 2,
            "star hub should be in the second half: hub_new_idx={hub_new_idx} n={n}"
        );
    }

    // ================================================================
    // Bandwidth tests — where RCM is guaranteed to help
    // ================================================================

    #[test]
    fn path_bandwidth_does_not_increase() {
        // A path graph: natural bandwidth = 1, RCM must not worsen it.
        let (g, _) = path_graph(20);
        let p      = rcm(&g);
        let bw_rcm = bandwidth_after_permutation(&g, &p);
        assert!(bw_rcm <= 1, "path bandwidth must stay 1, got {bw_rcm}");
    }

    #[test]
    fn path_large_bandwidth_does_not_increase() {
        let (g, _) = path_graph(500);
        let p      = rcm(&g);
        let bw_rcm = bandwidth_after_permutation(&g, &p);
        assert!(bw_rcm <= 1, "path bandwidth must stay 1, got {bw_rcm}");
    }

    /// For a RANDOMLY-NUMBERED path (nodes shuffled), RCM must reduce
    /// bandwidth back toward 1.
    ///
    /// We shuffle with a stride permutation: new_i → old_i = (i * 3) % n.
    /// This turns a compact band into a scattered one (bw ≈ n/2), and RCM
    /// should compress it back to near-1.
    #[test]
    fn path_bandwidth_reduced_from_bad_ordering() {
        let n = 101; // n prime helps ensure stride permutation is a bijection
        let (_, m_natural) = path_graph(n);

        // Stride permutation (3 is coprime to 101)
        let bad_perm_vec: Vec<usize> = (0..n).map(|i| (i * 3) % n).collect();
        let bad_perm = Permutation::new(bad_perm_vec).unwrap();
        let m_bad    = bad_perm.permute_sym(&m_natural).unwrap();
        let g_bad    = Graph::from_sym(&m_bad);

        let bw_bad = natural_bandwidth(&g_bad);
        assert!(bw_bad > n / 4, "stride should scatter: bw_bad={bw_bad}");

        let p      = rcm(&g_bad);
        let bw_rcm = bandwidth_after_permutation(&g_bad, &p);
        assert!(
            bw_rcm <= 2,
            "RCM must compress path bandwidth toward 1: bw_bad={bw_bad}, bw_rcm={bw_rcm}"
        );
    }

    // ================================================================
    // Fill tests — correct expectations per problem class
    // ================================================================

    /// For a path graph (tridiagonal), L is bidiagonal → nnz(L) = 2n−1.
    /// RCM must not increase this (path is already optimally ordered).
    #[test]
    fn path_fill_not_increased() {
        for n in [20, 50, 100] {
            let (_, m) = path_graph(n);
            let nnz_natural = nnz_l(&m);

            let g      = Graph::from_sym(&m);
            let p      = rcm(&g);
            let m_rcm  = p.permute_sym(&m).unwrap();
            let nnz_rcm = nnz_l(&m_rcm);

            assert_eq!(
                nnz_natural, 2 * n - 1,
                "path L should be bidiagonal: nnz={nnz_natural}"
            );
            assert_eq!(
                nnz_rcm, 2 * n - 1,
                "RCM must not add fill to an already-optimal path: nnz_rcm={nnz_rcm}"
            );
        }
    }

    /// For a BADLY-ORDERED path, RCM must reduce fill significantly.
    #[test]
    fn rcm_reduces_fill_on_badly_ordered_path() {
        // A path graph has treewidth 1 — fill is O(n) regardless of ordering,
        // so no permutation can produce 5× fill. Use a path but assert only
        // what is actually achievable: RCM must reduce fill from a bad ordering.
        let n = 101;
        let (_, m_natural) = path_graph(n);
        let nnz_natural = nnz_l(&m_natural); // 2n−1 (optimal)

        // Stride-50 on n=101 (gcd=1) gives maximum displacement per edge.
        let bad_perm_vec: Vec<usize> = (0..n).map(|i| (i * 50) % n).collect();
        let bad_perm = Permutation::new(bad_perm_vec).unwrap();
        let m_bad    = bad_perm.permute_sym(&m_natural).unwrap();
        let nnz_bad  = nnz_l(&m_bad);

        // Verify the bad ordering actually creates more fill than optimal.
        assert!(
            nnz_bad > nnz_natural,
            "stride-50 ordering should produce more fill than optimal: \
            nnz_bad={nnz_bad} nnz_natural={nnz_natural}"
        );

        // RCM should bring fill back near the natural value.
        let g       = Graph::from_sym(&m_bad);
        let p       = rcm(&g);
        let m_rcm   = p.permute_sym(&m_bad).unwrap();
        let nnz_rcm = nnz_l(&m_rcm);

        assert!(
            nnz_rcm <= nnz_natural + 10,
            "RCM should recover near-optimal fill: \
            nnz_natural={nnz_natural}, nnz_rcm={nnz_rcm}"
        );
        assert!(
            nnz_rcm < nnz_bad,
            "RCM must reduce fill from bad ordering: bad={nnz_bad}, rcm={nnz_rcm}"
        );
    }

    /// For a row-major 2-D grid, RCM may INCREASE fill relative to the natural
    /// ordering because the natural ordering is already near-optimal for bandwidth.
    /// We assert only that the permutation is valid and the fill increase is bounded.
    #[test]
    fn grid_row_major_rcm_fill_bounded() {
        for k in [4_usize, 6, 8] {
            let (g, m) = grid_laplacian(k);
            let p      = rcm(&g);
            assert_valid_permutation(&p, k * k);

            let m_rcm    = p.permute_sym(&m).unwrap();
            let nnz_nat  = nnz_l(&m);
            let nnz_rcm  = nnz_l(&m_rcm);

            // Bound: RCM must not increase fill by more than 4× for small grids.
            // This catches a broken permutation without asserting the impossible
            // guarantee that RCM always improves fill on well-ordered matrices.
            assert!(
                nnz_rcm <= nnz_nat * 4,
                "k={k}: RCM fill {nnz_rcm} should be ≤4× natural fill {nnz_nat}"
            );
        }
    }

    /// For a BADLY-ORDERED grid (stride permutation), RCM must reduce both
    /// bandwidth and fill isn't guranteed to be reduced.
    #[test]
    fn grid_badly_ordered_rcm_reduces_bandwidth() {
        let k = 7;   // 49 nodes
        let (_, m_natural) = grid_laplacian(k);
        let n = k * k;

        // Create a bad ordering (stride permutation)
        let bad_perm_vec: Vec<usize> = (0..n).map(|i| (i * 11) % n).collect();
        let bad_perm = Permutation::new(bad_perm_vec).unwrap();
        let m_bad = bad_perm.permute_sym(&m_natural).unwrap();

        let g = Graph::from_sym(&m_bad);
        let p = rcm(&g);
        let m_rcm = p.permute_sym(&m_bad).unwrap();

        // Compute bandwidth before and after
        let bw_bad = natural_bandwidth(&g);
        let g_rcm = Graph::from_sym(&m_rcm);
        let bw_rcm = natural_bandwidth(&g_rcm);

        assert!(
            bw_rcm < bw_bad,
            "RCM should reduce bandwidth: bw_bad={bw_bad} -> bw_rcm={bw_rcm}"
        );

        // Optionally: check that fill is not astronomical
        let nnz_rcm = nnz_l(&m_rcm);
        let nnz_nat = nnz_l(&m_natural);
        assert!(
            nnz_rcm <= 4 * nnz_nat,
            "Fill after RCM should not be excessive: nnz_nat={nnz_nat}, nnz_rcm={nnz_rcm}"
        );
    }

    // ================================================================
    // Mathematical correctness: permute → solve must give same result
    // ================================================================

    #[test]
    fn permute_and_solve_invariant_path() {
        use crate::{LinearSolver, CholeskySolver};

        let n = 30;
        let (_, m) = path_graph(n);
        let f: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();

        // Solve without permutation
        let mut solver = CholeskySolver::new();
        solver.analyze_and_factorize(&m).unwrap();
        let mut u1 = vec![0.0_f64; n];
        solver.solve(&f, &mut u1).unwrap();

        // Solve with RCM permutation applied internally
        let g    = Graph::from_sym(&m);
        let perm = rcm(&g);
        let mut solver2 = CholeskySolver::new();
        solver2.set_ordering(crate::ordering::Ordering::Custom(perm));
        solver2.analyze_and_factorize(&m).unwrap();
        let mut u2 = vec![0.0_f64; n];
        solver2.solve(&f, &mut u2).unwrap();

        for (i, (&a, &b)) in u1.iter().zip(u2.iter()).enumerate() {
            let rel = (a - b).abs() / a.abs().max(1e-14);
            assert!(rel < 1e-10, "path solve mismatch at dof {i}: {a:.8e} vs {b:.8e}");
        }
    }

    #[test]
    fn permute_and_solve_invariant_grid() {
        use crate::{LinearSolver, CholeskySolver};

        let k = 6;
        let (g, m) = grid_laplacian(k);
        let n = k * k;
        let f: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();

        let mut solver1 = CholeskySolver::new();
        solver1.analyze_and_factorize(&m).unwrap();
        let mut u1 = vec![0.0_f64; n];
        solver1.solve(&f, &mut u1).unwrap();

        let perm = rcm(&g);
        let mut solver2 = CholeskySolver::new();
        solver2.set_ordering(crate::ordering::Ordering::Custom(perm));
        solver2.analyze_and_factorize(&m).unwrap();
        let mut u2 = vec![0.0_f64; n];
        solver2.solve(&f, &mut u2).unwrap();

        for (i, (&a, &b)) in u1.iter().zip(u2.iter()).enumerate() {
            let rel = (a - b).abs() / a.abs().max(1e-14);
            assert!(rel < 1e-10, "grid solve mismatch at dof {i}");
        }
    }

    /// Verify the K_perm * x = P * (K * P^{-1}x) invariant directly.
    #[test]
    fn permute_sym_matvec_invariant() {
        let n = 10;
        let (g, m) = path_graph(n);
        let p      = rcm(&g);
        let km     = p.permute_sym(&m).unwrap();
        km.validate().unwrap();

        let pinv = p.inverse();
        let x: Vec<f64> = (1..=(n as u64)).map(|i| i as f64).collect();
        let x_tilde   = pinv.apply_to_slice(&x);
        let kx_tilde  = m.matvec(&x_tilde).unwrap();
        let y_perm    = km.matvec(&x).unwrap();
        let recovered = p.apply_to_slice(&kx_tilde);

        for (a, b) in y_perm.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-11, "matvec invariant: y_perm={a} recovered={b}");
        }
    }
}