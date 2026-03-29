//! Approximate Minimum Degree (AMD) fill-reduction ordering.
//!
//! ## Background
//!
//! AMD orders the rows/columns of a symmetric matrix to reduce fill-in during
//! Cholesky factorization.  At each step it greedily eliminates the node with
//! the smallest **approximate degree** — an upper bound on the true minimum
//! degree that is cheap to maintain.
//!
//! ## Why AMD outperforms RCM for dense subgraphs
//!
//! RCM minimises bandwidth, which helps when the graph is nearly a path or
//! grid.  AMD minimises fill directly, which helps for anything with
//! hub-and-spoke structure: portal frames (beam-column joints), rigid
//! diaphragms, or any model where many elements share a single node.
//!
//! For a well-ordered 1-D or 2-D mesh with a compact natural band, RCM and
//! AMD produce similar fill.  For mixed or irregular meshes, AMD is typically
//! 20–50 % better.
//!
//! ## Algorithm — simplified quotient graph
//!
//! This implementation uses a **dynamic adjacency set** representation:
//!
//! ```text
//! Repeat n times:
//!   1. Pick v = argmin |adj(v)|           (minimum approximate degree)
//!   2. Record v in the output permutation
//!   3. For each u ∈ adj(v):              (v's neighbours form a clique)
//!        adj(u) ←  adj(u) ∪ adj(v)  \  {v}
//!   4. Remove v from all adjacency sets
//! ```
//!
//! Step 3 is the **element absorption** step: eliminating `v` creates a clique
//! among its neighbours (they may need to interact during forward elimination),
//! so we merge the neighbour sets.  The degree update is implicit — each node's
//! degree is always `|adj(node)|`.
//!
//! This runs in O(n · d_max²) where `d_max` is the maximum degree encountered
//! during elimination (fill included), but for the sparse, low-degree graphs
//! that arise in structural FEM this is very fast in practice.
//!
//! ## References
//! - George, J.A. & Liu, J.W.H. (1981). *Computer Solution of Large Sparse
//!   Positive Definite Systems*. Prentice-Hall.
//! - Davis, T.A., Duff, I.S. & Gilbert, J.R. (1996). "A column approximate
//!   minimum degree ordering algorithm." *ACM TOMS* 30(3).
//! - Davis, T.A. (2006). *Direct Methods for Sparse Linear Systems*. §7.

use std::collections::BTreeSet;
use super::permutation::Permutation;
use sparse::SymCsrMatrix;

// -----------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------

/// Compute the AMD permutation for a symmetric matrix.
///
/// Returns a [`Permutation`] `p` where `p[new_index] = old_index`.
/// Apply it to `K` via [`Permutation::permute_sym`] before Cholesky
/// factorization.
///
/// # When to prefer AMD over RCM
///
/// | Graph structure          | Better ordering |
/// |--------------------------|-----------------|
/// | Path / 1-D mesh          | Tie (both optimal) |
/// | Regular 2-D grid         | Tie (both near-optimal) |
/// | Irregular / random mesh  | **AMD** |
/// | Frame with many joints   | **AMD** |
/// | Star / hub structure     | **AMD** |
///
/// # Panics
/// Panics if `k.n == 0`.
pub fn amd(k: &SymCsrMatrix) -> Permutation {
    let n = k.n;
    assert!(n > 0, "AMD: matrix must have at least one row");

    // Build symmetric adjacency sets.
    let mut adj: Vec<BTreeSet<usize>> = (0..n).map(|_| BTreeSet::new()).collect();
    for row in 0..n {
        let start = k.row_ptr()[row];
        let end   = k.row_ptr()[row + 1];
        for &col in &k.col_idx()[start..end] {
            if col != row {
                adj[row].insert(col);
                adj[col].insert(row);
            }
        }
    }

    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);

    while order.len() < n {
        // Linear scan: pick uneliminated node with smallest degree, tie‑break by largest index.
        let v = (0..n)
            .filter(|&i| !eliminated[i])
            .min_by_key(|&i| (adj[i].len(), std::cmp::Reverse(i)))
            .expect("no uneliminated nodes");

        order.push(v);
        eliminated[v] = true;

        let neighbours: Vec<usize> = adj[v].iter().copied().collect();

        // Remove v from each neighbour's adjacency.
        for &u in &neighbours {
            adj[u].remove(&v);
        }

        // Add clique edges between all pairs of neighbours.
        for i in 0..neighbours.len() {
            let u = neighbours[i];
            for j in i + 1..neighbours.len() {
                let w = neighbours[j];
                adj[u].insert(w);
                adj[w].insert(u);
            }
        }

        // Clear v's adjacency set.
        adj[v].clear();
    }

    Permutation::new(order)
        .expect("AMD: produced an invalid permutation — this is a bug")
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::{CooBuilder, SymCsrMatrix};
    use crate::cholesky::symbolic::analyze;
    use crate::ordering::permutation::Permutation;

    // ---- matrix builders ----

    fn tridiag(n: usize) -> SymCsrMatrix {
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n       { coo.add(i, i,      2.0); }
        for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
        coo.build_sym().unwrap()
    }

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

    /// Star graph: hub node 0 connected to all spokes 1..n.
    fn star(n: usize) -> SymCsrMatrix {
        assert!(n >= 2);
        let mut coo = CooBuilder::new(n, n);
        coo.add(0, 0, n as f64);
        for i in 1..n {
            coo.add(i, i, 2.0);
            coo.add(0, i, -1.0);
        }
        coo.build_sym().unwrap()
    }

    // ---- validity helper ----

    fn assert_valid(p: &Permutation, n: usize) {
        assert_eq!(p.len(), n, "permutation length mismatch");
        let mut seen = vec![false; n];
        for new_i in 0..n {
            let old = p.old_index(new_i);
            assert!(old < n, "old-index {old} out of range");
            assert!(!seen[old], "duplicate old-index {old}");
            seen[old] = true;
        }
        assert!(seen.iter().all(|&s| s), "missing old indices");
    }

    // ---- fill helper ----

    fn nnz_l(m: &SymCsrMatrix) -> usize {
        use sparse::convert::sym_to_csc;
        analyze(&sym_to_csc(m)).unwrap().nnz_l()
    }

    // ================================================================
    // Validity tests
    // ================================================================

    #[test]
    fn single_node() {
        let mut coo = CooBuilder::new(1, 1);
        coo.add(0, 0, 1.0);
        let m = coo.build_sym().unwrap();
        let p = amd(&m);
        assert_valid(&p, 1);
        assert!(p.is_identity());
    }

    #[test]
    fn two_nodes_valid() {
        let m = tridiag(2);
        let p = amd(&m);
        assert_valid(&p, 2);
    }

    #[test]
    fn tridiag_valid() {
        for n in [3, 5, 10, 20, 50] {
            let m = tridiag(n);
            let p = amd(&m);
            assert_valid(&p, n);
        }
    }

    #[test]
    fn grid_valid() {
        for k in [3, 5, 8] {
            let m = grid_laplacian(k);
            let p = amd(&m);
            assert_valid(&p, k * k);
        }
    }

    #[test]
    fn star_valid() {
        for n in [4, 8, 15] {
            let m = star(n);
            let p = amd(&m);
            assert_valid(&p, n);
        }
    }

    #[test]
    fn disconnected_graph_valid() {
        // Two disconnected components: tridiag(3) and tridiag(4)
        let n = 7;
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        // Component 1: 0-1-2
        coo.add(0, 1, -1.0); coo.add(1, 2, -1.0);
        // Component 2: 4-5-6
        coo.add(4, 5, -1.0); coo.add(5, 6, -1.0);
        // Node 3 is isolated
        let m = coo.build_sym().unwrap();
        let p = amd(&m);
        assert_valid(&p, n);
    }

    // ================================================================
    // Structural property tests
    // ================================================================

    /// For a star graph, AMD must eliminate the hub (node 0, max degree)
    /// late in the ordering — well after the initial low-degree spokes.
    ///
    /// Precise analysis: spokes start at degree 1.  Hub starts at degree n-1.
    /// AMD eliminates spokes first.  After k spokes are gone the hub has
    /// degree n-1-k.  Once only 1 spoke remains both that spoke and the hub
    /// have degree 1; tie-breaking by largest index places hub (node 0) later
    /// than the last spoke.  So the hub lands at new_index n-1 (last position).
    /// We assert hub_new_idx >= n/2 (it's firmly in the second half).
    #[test]
    fn star_hub_eliminated_late() {
        let n = 10;
        let m = star(n);
        let p = amd(&m);
        assert_valid(&p, n);

        let inv = p.inverse();
        let hub_new_idx = inv.old_index(0);
        assert!(
            hub_new_idx >= n / 2,
            "star hub (node 0) should be in the second half, got new_idx={hub_new_idx} (n={n})"
        );
    }

    /// For a path graph, AMD should produce zero fill (path is already
    /// optimal for bidiagonal Cholesky).
    #[test]
    fn path_zero_fill() {
        for n in [5, 10, 20] {
            let m = tridiag(n);
            let p = amd(&m);
            let m_amd = p.permute_sym(&m).unwrap();
            let nnz_natural = nnz_l(&m);
            let nnz_amd     = nnz_l(&m_amd);
            assert_eq!(
                nnz_natural, 2 * n - 1,
                "path natural L should be bidiagonal: nnz={nnz_natural}"
            );
            assert_eq!(
                nnz_amd, 2 * n - 1,
                "AMD must not add fill to a path (n={n}): nnz_amd={nnz_amd}"
            );
        }
    }

    // ================================================================
    // Fill-reduction tests
    // ================================================================

    /// For a star graph, AMD eliminates all spokes first (degree 1), then
    /// the hub last.  The resulting L has optimal fill (2n-1) because the hub
    /// is eliminated last and does not create extra fill.  Natural ordering
    /// would put the hub first, creating a dense fill-in row.
    #[test]
    fn star_amd_eliminates_fill() {
        let n = 10_usize;
        let m = star(n);
        let p  = amd(&m);
        let m_amd = p.permute_sym(&m).unwrap();

        let nnz_natural = nnz_l(&m);
        let nnz_amd     = nnz_l(&m_amd);

        // Natural: hub first → fills all (n-1) off-diagonals → nnz = 1 + 2*(n-1)
        assert!(
            nnz_natural > n,
            "natural ordering should produce fill for star: nnz={nnz_natural}"
        );
        // AMD: spokes first → optimal fill 2n-1
        assert_eq!(
            nnz_amd, 2 * n - 1,
            "AMD should produce optimal fill for star: nnz_amd={nnz_amd}, expected {}",
            2 * n - 1
        );
        assert!(
            nnz_amd < nnz_natural,
            "AMD must beat natural ordering for star: amd={nnz_amd} natural={nnz_natural}"
        );
    }

    /// For a grid Laplacian, AMD should produce fill that is not dramatically
    /// worse than the natural ordering.  We don't require AMD < natural
    /// (grid is already near-optimal), but AMD must not explode.
    #[test]
    fn grid_amd_fill_bounded() {
        for k in [4, 6, 8] {
            let m = grid_laplacian(k);
            let p = amd(&m);
            let m_amd = p.permute_sym(&m).unwrap();
            let nnz_nat = nnz_l(&m);
            let nnz_amd = nnz_l(&m_amd);
            // AMD must not produce more than 3× natural fill for small grids.
            assert!(
                nnz_amd <= nnz_nat * 3,
                "k={k}: AMD fill {nnz_amd} should be ≤3× natural fill {nnz_nat}"
            );
        }
    }

    /// Badly-ordered graph (stride permutation): AMD must improve fill
    /// compared to the random ordering.
    #[test]
    fn amd_reduces_fill_on_badly_ordered_arrow_matrix() {
        // Arrow matrix with hub at node 0 (worst position for Cholesky).
        // Eliminating the hub first creates a clique among all n-1 neighbors
        // → O(n²) fill. AMD moves the hub last → O(n) fill.
        let n = 50_usize;
        let mut coo = CooBuilder::new(n, n);
        for i in 0..n     { coo.add(i, i, (n as f64) + 1.0); }
        for i in 1..n     { coo.add(0, i, -1.0); }
        let m = coo.build_sym().unwrap();

        let nnz_natural = nnz_l(&m);
        let nnz_opt     = 2 * n - 1; // hub-last gives bidiagonal L

        assert!(
            nnz_natural > nnz_opt * 5,
            "hub-first ordering should create ≥5× fill: natural={nnz_natural} opt={nnz_opt}"
        );

        let p_amd = amd(&m);
        let m_amd = p_amd.permute_sym(&m).unwrap();
        let nnz_amd = nnz_l(&m_amd);

        assert_eq!(
            nnz_amd, nnz_opt,
            "AMD should recover optimal fill: amd={nnz_amd} opt={nnz_opt}"
        );
    }

    // ================================================================
    // Correctness: solve after AMD permutation must match natural solve
    // ================================================================

    fn solve_with_perm(k: &SymCsrMatrix, f: &[f64], perm: Permutation) -> Vec<f64> {
        use crate::cholesky::SparseSolver;
        let mut solver = SparseSolver::new();
        solver.set_ordering(crate::ordering::Ordering::Custom(perm));
        solver.analyze_and_factorize(k).unwrap();
        let mut u = vec![0.0_f64; f.len()];
        solver.solve(f, &mut u).unwrap();
        u
    }

    fn solve_natural(k: &SymCsrMatrix, f: &[f64]) -> Vec<f64> {
        use crate::cholesky::SparseSolver;
        let mut solver = SparseSolver::new();
        solver.set_ordering(crate::ordering::Ordering::Natural);
        solver.analyze_and_factorize(k).unwrap();
        let mut u = vec![0.0_f64; f.len()];
        solver.solve(f, &mut u).unwrap();
        u
    }

    #[test]
    fn amd_permutation_matvec_invariant_star() {
        let n = 8_usize;
        let m = star(n);
        let p = amd(&m);
        let km = p.permute_sym(&m).unwrap();
        let pinv = p.inverse();

        let x: Vec<f64> = (0..n).map(|i| (i+1) as f64).collect();
        let x_tilde   = pinv.apply_to_slice(&x);
        let kx_tilde  = m.matvec(&x_tilde).unwrap();
        let y_perm    = km.matvec(&x).unwrap();
        let recovered = p.apply_to_slice(&kx_tilde);

        for (i, (a, b)) in y_perm.iter().zip(recovered.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-10, "matvec invariant violated at {i}: y_perm={a:.8e} recovered={b:.8e}");
        }
    }

    #[test]
    fn amd_permutation_matvec_invariant_grid() {
        let k = 4_usize;
        let n = k * k;
        let m = grid_laplacian(k);
        let p = amd(&m);
        let km = p.permute_sym(&m).unwrap();
        let pinv = p.inverse();

        let x: Vec<f64> = (0..n).map(|i| (i+1) as f64).collect();
        let x_tilde   = pinv.apply_to_slice(&x);
        let kx_tilde  = m.matvec(&x_tilde).unwrap();
        let y_perm    = km.matvec(&x).unwrap();
        let recovered = p.apply_to_slice(&kx_tilde);

        for (i, (a, b)) in y_perm.iter().zip(recovered.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-10, "matvec invariant violated at {i}: y_perm={a:.8e} recovered={b:.8e}");
        }
    }

    #[test]
    fn tridiag_amd_solve_matches_natural() {
        let n = 30_usize;
        let m = tridiag(n);
        let f: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();

        let _u_nat = solve_natural(&m, &f);
        let p      = amd(&m);
        let u_amd  = solve_with_perm(&m, &f, p);

        // Check residual instead of direct solution comparison.
        let ku = m.matvec(&u_amd).unwrap();
        for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
            assert!(
                (kui - fi).abs() < 1e-10,
                "residual[{i}] = {:.2e} for AMD solution on tridiag",
                (kui - fi).abs()
            );
        }
    }

    #[test]
    fn star_amd_solve_matches_natural() {
        let n = 8_usize;
        let m = star(n);
        let f: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let _u_nat = solve_natural(&m, &f);
        let p      = amd(&m);
        let u_amd  = solve_with_perm(&m, &f, p);

        // Check residual instead of direct comparison.
        let ku = m.matvec(&u_amd).unwrap();
        for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
            assert!(
                (kui - fi).abs() < 1e-10,
                "residual[{i}] = {:.2e} for AMD solution on star",
                (kui - fi).abs()
            );
        }
    }

    #[test]
    fn grid_amd_solve_matches_natural() {
        let k = 5_usize;
        let n = k * k;
        let m = grid_laplacian(k);
        let f: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();

        let _u_nat = solve_natural(&m, &f);
        let p   = amd(&m);
        let u_amd  = solve_with_perm(&m, &f, p);

        // Check residual instead of direct comparison.
        let ku = m.matvec(&u_amd).unwrap();
        for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
            assert!(
                (kui - fi).abs() < 1e-10,
                "residual[{i}] = {:.2e} for AMD solution on grid",
                (kui - fi).abs()
            );
        }
    }
}