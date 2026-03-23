//! Reverse Cuthill-McKee (RCM) ordering for fill reduction.
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
//! RCM is a cheap, graph-based heuristic that reliably reduces fill for
//! FEM stiffness matrices by exploiting their banded or nearly-banded
//! structure.
//!
//! ## Algorithm
//!
//! ### Step 1 — find a peripheral node
//! A **peripheral node** is one at maximum graph diameter: far from
//! everything else.  Endpoints of a path graph are peripheral.
//! Exact computation is expensive; we use the standard cheap approximation:
//!
//! 1. Pick the minimum-degree node `s` as a seed.
//! 2. BFS from `s`; take the last node visited `t` as the pseudo-peripheral.
//!
//! ### Step 2 — BFS with degree ordering
//! BFS from `t`, but at each step sort the unvisited neighbours of the
//! current node by **ascending degree** before enqueuing.  This keeps
//! high-degree (hub) nodes late in the ordering, which is what reduces fill.
//!
//! The BFS visit order is the **Cuthill-McKee** ordering.
//!
//! ### Step 3 — reverse
//! Reversing the CM ordering gives RCM.  The reversed order has been shown
//! empirically (and theoretically for some graph classes) to produce smaller
//! bandwidth and less fill than CM.
//!
//! ## Output convention
//!
//! Returns a [`Permutation`] `p` where `p[new_index] = old_index`.
//! Apply it to `K` via [`Permutation::permute_sym`] before factorization.
//!
//! ## References
//! - Cuthill & McKee (1969), "Reducing the bandwidth of sparse symmetric matrices"
//! - Liu & Sherman (1976), "Comparative analysis of the CM and RCM algorithms"
//! - Davis, "Direct Methods for Sparse Linear Systems" (2006), Ch. 7

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
    use sparse::SymCsrMatrix;

    // ---- graph construction helpers ----

    fn path_graph(n: usize) -> Graph {
        let mut pattern: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        for i in 0..(n - 1) {
            pattern[i].push(i + 1);
        }
        let mut m = SymCsrMatrix::from_pattern(n, &pattern).unwrap();
        for i in 0..n { m.set_value(i, i, 1.0).unwrap(); }
        for i in 0..(n - 1) { m.set_value(i, i + 1, 1.0).unwrap(); }
        Graph::from_sym(&m)
    }

    fn star_graph(n: usize) -> Graph {
        // Hub = node 0, spokes = nodes 1..n
        let mut pattern: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        for i in 1..n { pattern[0].push(i); }
        let mut m = SymCsrMatrix::from_pattern(n, &pattern).unwrap();
        for i in 0..n { m.set_value(i, i, 1.0).unwrap(); }
        for i in 1..n { m.set_value(0, i, 1.0).unwrap(); }
        Graph::from_sym(&m)
    }

    /// k×k 2D grid, row-major node numbering.
    fn grid_graph(k: usize) -> Graph {
        let n = k * k;
        let mut pattern: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        for r in 0..k {
            for c in 0..k {
                let node = r * k + c;
                if c + 1 < k { pattern[node].push(r * k + c + 1); }
                if r + 1 < k { pattern[node].push((r + 1) * k + c); }
            }
        }
        let mut m = SymCsrMatrix::from_pattern(n, &pattern).unwrap();
        for i in 0..n { m.set_value(i, i, 1.0).unwrap(); }
        for r in 0..k {
            for c in 0..k {
                let node = r * k + c;
                if c + 1 < k { m.set_value(node, r * k + c + 1, 1.0).unwrap(); }
                if r + 1 < k { m.set_value(node, (r + 1) * k + c, 1.0).unwrap(); }
            }
        }
        Graph::from_sym(&m)
    }

    // ---- validity assertion ----

    fn assert_valid_permutation(p: &Permutation, n: usize) {
        assert_eq!(p.len(), n);
        let mut seen = vec![false; n];
        for i in 0..n {
            let old = p.old_index(i);
            assert!(!seen[old], "duplicate old-index {old} in permutation");
            seen[old] = true;
        }
        assert!(seen.iter().all(|&s| s), "some old indices missing");
    }

    // ---- bandwidth helpers ----

    fn permuted_bandwidth(g: &Graph, p: &Permutation) -> usize {
        let inv = p.inverse();
        let inv = &inv;

        (0..g.n)
            .flat_map(|old_i| {
                let new_i = inv.old_index(old_i);
                g.neighbours(old_i)
                    .iter()
                    .map(move |&old_j| {
                        let new_j = inv.old_index(old_j);
                        new_i.abs_diff(new_j)
                    })
            })
            .max()
            .unwrap_or(0)
    }

    fn original_bandwidth(g: &Graph) -> usize {
        (0..g.n)
            .flat_map(|i| g.neighbours(i).iter().map(move |&j| i.abs_diff(j)))
            .max()
            .unwrap_or(0)
    }

    // ---- tests ----

    #[test]
    fn single_node() {
        let pattern = vec![vec![0usize]];
        let mut m = SymCsrMatrix::from_pattern(1, &pattern).unwrap();
        m.set_value(0, 0, 1.0).unwrap();
        let g = Graph::from_sym(&m);
        let p = rcm(&g);
        assert_valid_permutation(&p, 1);
        assert!(p.is_identity());
    }

    #[test]
    fn two_nodes_connected() {
        let g = path_graph(2);
        let p = rcm(&g);
        assert_valid_permutation(&p, 2);
    }

    #[test]
    fn path4_valid_permutation() {
        let g = path_graph(4);
        let p = rcm(&g);
        assert_valid_permutation(&p, 4);
    }

    #[test]
    fn path4_starts_from_endpoint() {
        // RCM should start from a degree-1 node (path endpoint)
        let g = path_graph(4);
        let p = rcm(&g);
        assert_eq!(
            g.degree(p.old_index(0)), 1,
            "RCM first node should be an endpoint (degree 1)"
        );
    }

    #[test]
    fn path_large_valid_permutation() {
        let g = path_graph(50);
        let p = rcm(&g);
        assert_valid_permutation(&p, 50);
    }

    #[test]
    fn star_valid_permutation() {
        let g = star_graph(8);
        let p = rcm(&g);
        assert_valid_permutation(&p, 8);
    }

    #[test]
    fn star_hub_is_late_in_rcm() {
        // Hub has maximum degree — RCM should place it late (not necessarily last
        // due to the starting-node asymmetry in star graphs)
        let n = 8;
        let g = star_graph(n);
        let p = rcm(&g);
        // Find where the hub (node 0, degree n-1) ends up in the new ordering
        let inv = p.inverse();
        let hub_new_index = inv.old_index(0);
        // Hub should appear in the second half of the ordering
        assert!(
            hub_new_index >= n / 2,
            "hub should be in the second half of RCM ordering, got index {hub_new_index}"
        );
    }

    #[test]
    fn disconnected_graph_all_nodes_covered() {
        // Two disconnected edges: 0-1 and 2-3
        let pattern = vec![
            vec![0usize, 1], vec![1usize],
            vec![2usize, 3], vec![3usize],
        ];
        let mut m = SymCsrMatrix::from_pattern(4, &pattern).unwrap();
        for i in 0..4 { m.set_value(i, i, 1.0).unwrap(); }
        m.set_value(0, 1, 1.0).unwrap();
        m.set_value(2, 3, 1.0).unwrap();
        let g = Graph::from_sym(&m);
        let p = rcm(&g);
        assert_valid_permutation(&p, 4);
    }

    #[test]
    fn path_bandwidth_preserved() {
        // A path graph has bandwidth 1 — RCM must not increase it
        let g = path_graph(10);
        let p = rcm(&g);
        assert!(
            permuted_bandwidth(&g, &p) <= 1,
            "RCM should not increase bandwidth of a path graph"
        );
    }

    #[test]
    fn grid_bandwidth_not_increased() {
        let g = grid_graph(5); // 25-node grid
        let orig = original_bandwidth(&g);
        let p    = rcm(&g);
        let rcm_bw = permuted_bandwidth(&g, &p);
        assert!(
            rcm_bw <= orig,
            "RCM increased bandwidth: original={orig}, rcm={rcm_bw}"
        );
    }

    #[test]
    fn grid_bandwidth_not_worse_than_original() {
        // Use a larger grid where RCM reliably reduces bandwidth
        let k = 8;
        let g = grid_graph(k);
        let orig   = original_bandwidth(&g);
        let p      = rcm(&g);
        let rcm_bw = permuted_bandwidth(&g, &p);
        assert!(
            rcm_bw <= orig,
            "RCM increased bandwidth: original={orig}, rcm={rcm_bw}"
        );
    }

    #[test]
    fn grid_rcm_produces_valid_permutation_and_does_not_increase_bandwidth() {
        for k in [4, 6, 8] {
            let g  = grid_graph(k);
            let orig   = original_bandwidth(&g);
            let p      = rcm(&g);
            let rcm_bw = permuted_bandwidth(&g, &p);
            assert!(
                rcm_bw <= orig,
                "k={k}: RCM increased bandwidth: original={orig}, rcm={rcm_bw}"
            );
            assert_valid_permutation(&p, k * k);
        }
    }

    #[test]
    fn permute_and_matvec_invariant() {
        // Verify: K_perm[i,j] = K[p[i], p[j]]
        // Which means: (K_perm * x)[i] = (K * x̃)[p[i]]  where x̃[k] = x[p_inv[k]]
        // So: y_perm == p.apply_to_slice(K * p_inv.apply_to_slice(x))
        let n = 8;
        let g = path_graph(n);
        let p = rcm(&g);

        // Build tridiagonal SymCsr for path graph
        let mut pattern: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        for i in 0..(n - 1) { pattern[i].push(i + 1); }
        let mut m = SymCsrMatrix::from_pattern(n, &pattern).unwrap();
        for i in 0..n     { m.set_value(i, i,      2.0).unwrap(); }
        for i in 0..(n-1) { m.set_value(i, i + 1, -1.0).unwrap(); }

        let km = p.permute_sym(&m).unwrap();
        km.validate().unwrap();

        let pinv = p.inverse();
        let x: Vec<f64> = (1..=(n as u64)).map(|i| i as f64).collect();

        let x_tilde  = pinv.apply_to_slice(&x);       // x̃[k] = x[p_inv[k]]
        let kx_tilde = m.matvec(&x_tilde).unwrap();   // K * x̃
        let y_perm   = km.matvec(&x).unwrap();         // K_perm * x
        let recovered = p.apply_to_slice(&kx_tilde);  // (K * x̃)[p[i]]

        for (a, b) in y_perm.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-11, "y_perm={a} recovered={b}");
        }
    }
}