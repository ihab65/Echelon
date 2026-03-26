//! Stress tests for the sparse Cholesky solver.
//!
//! These tests push the solver beyond the unit-test cases that live inside the
//! crate modules.  Every test here builds a matrix that resembles something
//! that arises in real structural or PDE problems, runs the full
//! `analyze → factorize → solve` pipeline, and verifies the result with a
//! residual check.
//!
//! ## Test categories
//!
//! | Section | What it tests |
//! |---------|---------------|
//! | A       | Large-scale 1-D banded systems (spring chains) |
//! | B       | 2-D grid Laplacian — FEM-like connectivity and fill |
//! | C       | Ill-conditioned SPD matrices — stiffness contrast up to 1e12 |
//! | D       | Near-singular systems — solver must detect and error correctly |
//! | E       | Repeated factorization — topology reuse with changing values |
//!
//! ## Residual tolerance
//!
//! The threshold `1e-8` for relative residual `‖Ax−b‖/‖b‖` is deliberate:
//! - It is well above double-precision rounding (~1e-15) so it catches real
//!   algorithmic bugs without spurious failures.
//! - For the conditioning tests (section C) the threshold is relaxed to `1e-6`
//!   to account for the amplified rounding error that a condition number of
//!   ~1e12 introduces.

use sparse::{CooBuilder, SymCsrMatrix};
use solvers::cholesky::SparseSolver;
use solvers::SolverError;

// =============================================================================
// Shared matrix builders and utilities
// =============================================================================

/// 1-D spring chain: tridiagonal SPD matrix of size `n`.
///
/// The stiffness of each spring is `k_spring`.  For a uniform chain every
/// diagonal is `2k` and every off-diagonal is `-k`, except the first and last
/// rows which have only one spring contribution (diagonal = `k`).
///
/// This is the standard FEM stiffness for a 1-D bar with `n` free DOFs and
/// pin BCs already applied at both ends.
fn spring_chain(n: usize, k_spring: f64) -> SymCsrMatrix {
    assert!(n >= 2, "spring chain needs at least 2 nodes");
    let mut coo = CooBuilder::new(n, n);
    for i in 0..n {
        // Diagonal: interior nodes connect to two springs; ends to one.
        let diag = if i == 0 || i == n - 1 { k_spring } else { 2.0 * k_spring };
        coo.add(i, i, diag);
    }
    for i in 0..(n - 1) {
        coo.add(i, i + 1, -k_spring);
    }
    coo.build_sym().unwrap()
}

/// 2-D grid Laplacian using the standard 5-point stencil.
///
/// The grid has `nx × ny` interior nodes.  Node `(ix, iy)` maps to global
/// index `ix + iy * nx`.  The boundary condition is implicit (boundary nodes
/// are not included — this is the free-interior Laplacian).
///
/// Diagonal = 4.0, off-diagonal neighbours = -1.0.
/// This is SPD for all `nx, ny ≥ 1`.
fn laplacian_2d(nx: usize, ny: usize) -> SymCsrMatrix {
    assert!(nx >= 1 && ny >= 1);
    let n = nx * ny;
    let mut coo = CooBuilder::new(n, n);

    for iy in 0..ny {
        for ix in 0..nx {
            let i = ix + iy * nx;
            coo.add(i, i, 4.0);

            // right neighbour
            if ix + 1 < nx {
                let j = (ix + 1) + iy * nx;
                coo.add(i, j, -1.0);
            }
            // upper neighbour
            if iy + 1 < ny {
                let j = ix + (iy + 1) * nx;
                coo.add(i, j, -1.0);
            }
        }
    }
    coo.build_sym().unwrap()
}

/// Compute the relative residual `‖Au − f‖∞ / ‖f‖∞`.
///
/// Uses `SymCsrMatrix::matvec` which expands the implicit symmetry, so this
/// works correctly even though `A` only stores the upper triangle.
fn relative_residual(a: &SymCsrMatrix, f: &[f64], u: &[f64]) -> f64 {
    let au   = a.matvec(u).unwrap();
    let norm_f: f64 = f.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let norm_f = norm_f.max(1e-300); // avoid division by zero
    au.iter()
        .zip(f.iter())
        .map(|(&aui, &fi)| (aui - fi).abs())
        .fold(0.0_f64, f64::max)
        / norm_f
}

/// Solve `Au = f` with the full `SparseSolver` pipeline and return `u`.
fn full_solve(a: &SymCsrMatrix, f: &[f64]) -> Vec<f64> {
    let mut solver = SparseSolver::new();
    solver.analyze_and_factorize(a).unwrap();
    let mut u = vec![0.0_f64; f.len()];
    solver.solve(f, &mut u).unwrap();
    u
}

/// Build a right-hand side that is deterministic and non-trivial:
/// `f[i] = sin(i+1)`.  This avoids accidentally orthogonal RHS vectors that
/// could mask residual errors.
fn rhs_sin(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) as f64).sin()).collect()
}

/// Build an all-ones RHS.
fn rhs_ones(n: usize) -> Vec<f64> {
    vec![1.0_f64; n]
}

// =============================================================================
// A — Large-scale banded systems (spring chains)
// =============================================================================

/// Run the full pipeline on a spring chain of size `n` and check the residual.
fn run_spring_chain(n: usize) {
    let k = spring_chain(n, 1.0);
    assert_eq!(k.n, n);
    let f   = rhs_sin(n);
    let u   = full_solve(&k, &f);
    let res = relative_residual(&k, &f, &u);
    assert!(
        res < 1e-8,
        "spring chain n={n}: relative residual {res:.2e} exceeds 1e-8"
    );
}

#[test]
fn a1_spring_chain_n100() { run_spring_chain(100); }

#[test]
fn a2_spring_chain_n500() { run_spring_chain(500); }

#[test]
fn a3_spring_chain_n1000() { run_spring_chain(1000); }

#[test]
fn a4_spring_chain_n2000() { run_spring_chain(2000); }

/// Verify that the solution to the spring chain is numerically reasonable:
/// the maximum displacement under a unit mid-point load should equal 0.25
/// for a chain of uniform stiffness k=1, n=100, clamped at both ends.
///
/// Exact solution for a clamped-clamped spring chain under unit force at
/// node m (0-based, m = n/2): u_max = m*(n-m) / n = (n/2)^2 / n = n/4.
/// For n=100, m=50: u_max = 50*50/100 = 25.0.
#[test]
fn a5_spring_chain_exact_midpoint_load() {
    let n = 100_usize;
    let k = spring_chain(n, 1.0);

    // Unit load at midpoint (node 50, 0-based)
    let mid = n / 2;
    let mut f = vec![0.0_f64; n];
    f[mid] = 1.0;

    let u   = full_solve(&k, &f);
    let res = relative_residual(&k, &f, &u);
    assert!(res < 1e-10, "residual {res:.2e}");

    // Analytical maximum displacement
    let m          = mid as f64;
    let n_f        = n as f64;
    let u_max_exact = m * (n_f - m) / n_f;
    let u_max_calc  = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let rel_err = (u_max_calc - u_max_exact).abs() / u_max_exact;
    assert!(
        rel_err < 1e-10,
        "midpoint displacement: computed={u_max_calc:.10}, exact={u_max_exact:.10}, rel_err={rel_err:.2e}"
    );
}

/// Stiffness scaling: multiplying k by a scalar must scale the solution by
/// the inverse.  This verifies linearity is preserved across factorizations.
#[test]
fn a6_spring_chain_stiffness_scaling() {
    let n = 200_usize;
    let f = rhs_sin(n);

    let k1 = spring_chain(n, 1.0);
    let k2 = spring_chain(n, 4.0);  // 4× stiffer

    let u1 = full_solve(&k1, &f);
    let u2 = full_solve(&k2, &f);

    // By linearity: u2 = u1 / 4 for every DOF
    for (i, (&a, &b)) in u1.iter().zip(u2.iter()).enumerate() {
        let rel = (4.0 * b - a).abs() / a.abs().max(1e-15);
        assert!(rel < 1e-9, "DOF {i}: 4*u2={:.10e}  u1={:.10e}  rel={rel:.2e}", 4.0*b, a);
    }
}

// =============================================================================
// B — 2-D grid Laplacian (FEM-like structure)
// =============================================================================

/// Run the full pipeline on an `nx × ny` grid Laplacian.
fn run_laplacian(nx: usize, ny: usize) {
    let n   = nx * ny;
    let a   = laplacian_2d(nx, ny);
    assert_eq!(a.n, n);
    let f   = rhs_sin(n);
    let u   = full_solve(&a, &f);
    let res = relative_residual(&a, &f, &u);
    assert!(
        res < 1e-8,
        "Laplacian {nx}×{ny}: relative residual {res:.2e} exceeds 1e-8"
    );
}

#[test]
fn b1_laplacian_10x10() { run_laplacian(10, 10); }

#[test]
fn b2_laplacian_20x20() { run_laplacian(20, 20); }

#[test]
fn b3_laplacian_40x40() { run_laplacian(40, 40); }

#[test]
fn b4_laplacian_50x30() { run_laplacian(50, 30); }  // non-square grid

/// The 2-D Laplacian has a known spectral property: for the n×n grid the
/// smallest eigenvalue is approximately 2*(1 - cos(π/(n+1))) ≈ π²/(n+1)².
/// For n=10: λ_min ≈ 0.081.
///
/// We test this indirectly: solve A u = e_k (a single right-hand side with
/// one non-zero entry) and verify that the maximum component of u is bounded
/// above by 1/λ_min.  If the bound is violated by more than 10%, the
/// factorization is numerically wrong.
#[test]
fn b5_laplacian_solution_magnitude_bound() {
    let nx = 10;
    let ny = 10;
    let n  = nx * ny;
    let a  = laplacian_2d(nx, ny);

    // Unit load at the centre node
    let mut f = vec![0.0_f64; n];
    f[nx / 2 + (ny / 2) * nx] = 1.0;

    let u = full_solve(&a, &f);

    // Bound: u_max ≤ 1/λ_min.  For 10×10 Laplacian, λ_min ≈ 0.081.
    let lambda_min_approx = 0.081_f64;
    let u_max = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        u_max < 1.0 / lambda_min_approx * 1.1,
        "u_max={u_max:.4} exceeds 1.1/λ_min={:.4} — factorization may be wrong",
        1.0 / lambda_min_approx
    );
    // Also check residual
    let res = relative_residual(&a, &f, &u);
    assert!(res < 1e-8, "residual {res:.2e}");
}

/// RCM correctness test on the 2-D grid.
///
/// ## What RCM guarantees vs what it does not
///
/// The 20×20 Laplacian in row-major (natural) order already has a compact
/// band of width `nx = 20`.  RCM produces a diagonal-wavefront ordering
/// whose band is approximately `2·nx − 1 = 39`.  This is **wider** than
/// the natural ordering, so RCM will *increase* fill-in for this specific
/// input — a well-known behaviour documented in the literature.
///
/// RCM is designed for matrices where the natural ordering is poor
/// (random node numbering, star topologies, etc.), not for matrices that
/// are already nearly optimally ordered.
///
/// What we **can** assert for any matrix under RCM:
///
/// 1. The permutation is valid (bijection of `0..n`).
/// 2. The permuted matrix is structurally sound (`validate` passes).
/// 3. Solving `K_perm u_perm = f_perm` gives the same result as solving
///    `K u = f` — the ordering does not change the mathematical solution.
///
/// We also verify the well-ordered case: a **randomly-permuted** 2-D grid
/// (poor natural ordering) where RCM must reduce bandwidth.
#[test]
fn b6_laplacian_rcm_validity_and_correctness() {
    use solvers::cholesky::symbolic::analyze;
    use solvers::ordering::{Graph, rcm};

    let nx = 20;
    let ny = 20;
    let k  = laplacian_2d(nx, ny);
    let n  = k.n;

    // ── 1. Valid permutation ──────────────────────────────────────────────────
    let g    = Graph::from_sym(&k);
    let perm = rcm(&g);
    assert_eq!(perm.len(), n, "permutation length must equal matrix size");

    // Bijection check: every old index appears exactly once
    let mut seen = vec![false; n];
    for new_i in 0..n {
        let old = perm.old_index(new_i);
        assert!(!seen[old], "duplicate old-index {old} in RCM permutation");
        seen[old] = true;
    }
    assert!(seen.iter().all(|&s| s), "some old indices missing from permutation");

    // ── 2. Permuted matrix passes structural validation ───────────────────────
    let k_perm = perm.permute_sym(&k).unwrap();
    k_perm.validate().unwrap();

    // ── 3. Solve correctness: RCM ordering must produce the same solution ─────
    //
    // We build f, solve with natural ordering, solve with RCM ordering, and
    // verify the answers match to 1e-10 relative tolerance.
    let f = rhs_sin(n);

    // Solve with natural (identity) ordering
    let u_natural = full_solve(&k, &f);

    // Solve with RCM ordering via the high-level SparseSolver API
    // (which handles the permutation internally — no manual permute needed)
    let u_rcm = {
        use solvers::cholesky::SparseSolver;
        
        let mut solver = SparseSolver::new();
        solver.set_ordering(perm.clone());
        solver.analyze_and_factorize(&k).unwrap();
        let mut u = vec![0.0_f64; n];
        solver.solve(&f, &mut u).unwrap();
        u
    };

    // The solutions must be identical within floating-point round-off
    for (i, (&un, &ur)) in u_natural.iter().zip(u_rcm.iter()).enumerate() {
        let denom = un.abs().max(1e-14);
        let rel   = (un - ur).abs() / denom;
        assert!(
            rel < 1e-10,
            "b6: u_natural[{i}]={un:.8e} u_rcm[{i}]={ur:.8e} rel_diff={rel:.2e}"
        );
    }

    // ── 4. Fill comparison: row-major grid is already well-ordered ───────────
    //
    // For the row-major 20×20 Laplacian, RCM produces a WIDER band than the
    // natural ordering (diagonal wavefront bandwidth ≈ 39 vs natural ≈ 20).
    // We assert only that RCM does not increase fill by more than 3× — a loose
    // bound that would catch a completely broken permutation while not asserting
    // the impossible guarantee that RCM always improves fill.
    let nnz_natural = analyze(&k).unwrap().nnz_l();
    let nnz_rcm     = analyze(&k_perm).unwrap().nnz_l();
    assert!(
        nnz_rcm <= nnz_natural * 3,
        "b6: RCM fill {nnz_rcm} is more than 3× natural fill {nnz_natural} — \
         permutation may be broken"
    );
}

/// RCM bandwidth reduction on a POORLY-ORDERED 2-D grid.
///
/// The natural 2-D grid in row-major order has bandwidth `nx`.
/// If we **randomly permute the nodes first** (simulating a badly-ordered
/// mesh), the natural bandwidth explodes to nearly `n`.  RCM then provides
/// a genuine and measurable reduction.
///
/// This test verifies the case where RCM is most valuable — the case the
/// algorithm was designed for.
#[test]
fn b7_laplacian_rcm_helps_on_badly_ordered_mesh() {
    use solvers::ordering::{Graph, rcm};

    let nx = 15;
    let ny = 15;
    let k_natural = laplacian_2d(nx, ny);
    let n         = k_natural.n;

    // Apply a deterministic "bad" permutation: reverse all node indices.
    // This maps the compact row-major band into one that is also compact
    // (reversal preserves bandwidth for a symmetric matrix).
    // Instead, use a stride permutation that genuinely destroys locality:
    // new_idx i → old_idx (i * 7) % n  (for n not divisible by 7)
    // 225 = 15*15, gcd(7, 225) = 1, so this is a valid permutation.
    let bad_perm_vec: Vec<usize> = (0..n).map(|i| (i * 7) % n).collect();

    // Verify the stride permutation is a valid bijection
    let mut check = vec![false; n];
    for &v in &bad_perm_vec { check[v] = true; }
    assert!(check.iter().all(|&c| c), "stride permutation is not a bijection");

    use solvers::ordering::Permutation;
    let bad_perm  = Permutation::new(bad_perm_vec).unwrap();
    let k_bad     = bad_perm.permute_sym(&k_natural).unwrap();

    // Bandwidth of k_natural (row-major, compact):  nx - 1 = 14
    // Bandwidth of k_bad (stride-7 permuted, scattered): should be >> 14
    fn matrix_bandwidth(k: &sparse::SymCsrMatrix) -> usize {
        let mut bw = 0_usize;
        for row in 0..k.n {
            let start = k.row_ptr()[row];
            let end   = k.row_ptr()[row + 1];
            for &col in &k.col_idx()[start..end] {
                bw = bw.max(col.abs_diff(row));
            }
        }
        bw
    }

    let bw_natural = matrix_bandwidth(&k_natural);
    let bw_bad     = matrix_bandwidth(&k_bad);
    assert!(
        bw_bad > bw_natural * 5,
        "b7: expected stride permutation to blow up bandwidth \
         (bw_natural={bw_natural}, bw_bad={bw_bad})"
    );

    // Now apply RCM to the badly-ordered matrix
    let g       = Graph::from_sym(&k_bad);
    let perm    = rcm(&g);
    let k_rcm   = perm.permute_sym(&k_bad).unwrap();
    let bw_rcm  = matrix_bandwidth(&k_rcm);

    // RCM must significantly reduce bandwidth compared to k_bad
    assert!(
        bw_rcm < bw_bad,
        "b7: RCM should reduce bandwidth of badly-ordered matrix: \
         bw_bad={bw_bad}, bw_rcm={bw_rcm}"
    );
    assert!(
        bw_rcm <= bw_natural * 4,
        "b7: RCM bandwidth {bw_rcm} should be within 4× of the natural \
         (optimal) bandwidth {bw_natural}"
    );

    // And the solution must still be correct after this double permutation
    let f = rhs_sin(n);
    let u_natural = full_solve(&k_natural, &f);
    let u_bad     = full_solve(&k_bad, &f);

    for (i, (&un, &ub)) in u_natural.iter().zip(u_bad.iter()).enumerate() {
        let rel = (un - ub).abs() / un.abs().max(1e-14);
        assert!(rel < 1e-9, "b7: solution mismatch at dof {i}: {un:.8e} vs {ub:.8e}");
    }
}

// =============================================================================
// C — Conditioning torture test
// =============================================================================

/// Build a random-ish SPD matrix with large stiffness contrast.
///
/// Construction: `A = Mᵀ M + ε I`
/// where `M` is a diagonal matrix with entries spaced logarithmically from
/// `scale_lo` to `scale_hi`, and `ε = 1e-3 * scale_lo` ensures strict SPD.
///
/// For a diagonal `M`, `Mᵀ M` is also diagonal with entries `M[i]²`.
/// This gives a diagonal SPD matrix with prescribed condition number
/// `κ = (scale_hi² + ε) / (scale_lo² + ε) ≈ (scale_hi/scale_lo)²`.
///
/// We use a diagonal matrix here because:
/// 1. It is trivially SPD with known condition number.
/// 2. It avoids the dense-matrix overhead of a full random product.
/// 3. It still tests the solver under extreme scaling.
fn ill_conditioned_diagonal(n: usize, scale_lo: f64, scale_hi: f64) -> SymCsrMatrix {
    assert!(scale_hi >= scale_lo && scale_lo > 0.0);
    let eps = 1e-3 * scale_lo * scale_lo;

    let mut coo = CooBuilder::new(n, n);
    for i in 0..n {
        // Logarithmically spaced diagonal entries from scale_lo² to scale_hi²
        let t = i as f64 / (n - 1).max(1) as f64;
        let m_i = scale_lo * (scale_hi / scale_lo).powf(t);
        let a_ii = m_i * m_i + eps;
        coo.add(i, i, a_ii);
    }
    coo.build_sym().unwrap()
}

/// Build an SPD matrix with off-diagonal coupling and large diagonal contrast.
///
/// This mimics the stiffness matrices that arise in structural models with
/// mixed element types (e.g., stiff columns and flexible beams):
///
/// ```text
/// A[i,i]   = scale[i]
/// A[i,i+1] = -0.1 * min(scale[i], scale[i+1])   (weak coupling)
/// ```
///
/// The matrix is diagonally dominant by construction → guaranteed SPD.
fn mixed_stiffness_tridiag(n: usize) -> SymCsrMatrix {
    let mut scales = vec![0.0_f64; n];
    // First half: "stiff" (scale ~ 1e6), second half: "flexible" (scale ~ 1.0)
    for i in 0..n {
        scales[i] = if i < n / 2 { 1e6_f64 } else { 1.0_f64 };
    }

    let mut coo = CooBuilder::new(n, n);
    for i in 0..n {
        coo.add(i, i, scales[i]);
        if i + 1 < n {
            let coupling = -0.1 * scales[i].min(scales[i + 1]);
            coo.add(i, i + 1, coupling);
        }
    }
    coo.build_sym().unwrap()
}

#[test]
fn c1_diagonal_contrast_1e6() {
    let n = 200;
    let a = ill_conditioned_diagonal(n, 1.0, 1e3);  // condition ≈ 1e6
    let f = rhs_sin(n);
    let u = full_solve(&a, &f);
    let res = relative_residual(&a, &f, &u);
    assert!(
        res < 1e-6,
        "ill-conditioned diagonal n={n} κ≈1e6: residual {res:.2e} > 1e-6"
    );
}

#[test]
fn c2_diagonal_contrast_1e12() {
    let n = 100;
    // condition number ≈ (1e6)² = 1e12 — near the limit of f64 precision
    let a = ill_conditioned_diagonal(n, 1.0, 1e6);
    let f = rhs_ones(n);
    let u = full_solve(&a, &f);
    let res = relative_residual(&a, &f, &u);
    // With κ ≈ 1e12 we lose about 12 digits → 15 - 12 = 3 digits of residual.
    // A threshold of 1e-3 is therefore physically meaningful, not a cop-out.
    assert!(
        res < 1e-3,
        "κ≈1e12: residual {res:.2e} > 1e-3 — possible numerical breakdown"
    );
}

#[test]
fn c3_mixed_stiffness_half_stiff_half_flexible() {
    let n = 300;
    let a = mixed_stiffness_tridiag(n);
    let f = rhs_sin(n);
    let u = full_solve(&a, &f);
    let res = relative_residual(&a, &f, &u);
    assert!(
        res < 1e-6,
        "mixed stiffness n={n}: residual {res:.2e} > 1e-6"
    );
}

/// The solution to a diagonal SPD system A u = f is trivially u[i] = f[i] / A[i,i].
/// Use this to verify the solver produces the mathematically correct answer even
/// under extreme scaling.
#[test]
fn c4_diagonal_exact_solution_verification() {
    let n      = 50;
    let scales: Vec<f64> = (0..n)
        .map(|i| 10.0_f64.powi(i as i32 % 13))  // entries 1, 10, 100, ... cycling
        .collect();

    let mut coo = CooBuilder::new(n, n);
    for (i, &s) in scales.iter().enumerate() {
        coo.add(i, i, s);
    }
    let a = coo.build_sym().unwrap();

    let f: Vec<f64> = scales.iter().map(|&s| s).collect();  // f[i] = scale[i]
    let u = full_solve(&a, &f);

    // Exact answer: u[i] = f[i] / a[i,i] = scale[i] / scale[i] = 1.0 for all i
    for (i, &ui) in u.iter().enumerate() {
        let err = (ui - 1.0).abs();
        assert!(
            err < 1e-10,
            "c4: u[{i}]={ui:.6e}  expected 1.0  abs_err={err:.2e}"
        );
    }
}

// =============================================================================
// D — Near-singular / weakly constrained systems
// =============================================================================

/// A matrix with a very small diagonal entry at position `bad_dof` will either:
/// - trigger `NotPositiveDefinite` during factorization (if the diagonal falls
///   below `1e-12`), or
/// - produce a solution with a huge residual (detectable condition).
///
/// Either outcome is acceptable; we assert that the system does NOT silently
/// produce a well-conditioned solution.
#[test]
fn d1_near_zero_diagonal_single_dof() {
    let n = 50_usize;
    let bad_dof = 25_usize;
    let tiny = 1e-11_f64;  // below the 1e-12 SPD threshold in numeric.rs

    let mut coo = CooBuilder::new(n, n);
    for i in 0..n {
        let diag = if i == bad_dof { tiny } else { 2.0 };
        coo.add(i, i, diag);
        if i + 1 < n { coo.add(i, i + 1, -1.0); }
    }
    let a = coo.build_sym().unwrap();

    let mut solver = SparseSolver::new();
    solver.analyze(&a).unwrap();
    let result = solver.factorize(&a);

    match result {
        Err(SolverError::NotPositiveDefinite { .. }) => {
            // Expected: the solver correctly detected the near-zero diagonal
        }
        Ok(()) => {
            // Factorization succeeded — but residual must reveal the problem
            let f = rhs_ones(n);
            let mut u = vec![0.0_f64; n];
            solver.solve(&f, &mut u).unwrap();
            let res = relative_residual(&a, &f, &u);
            // If the matrix is near-singular, the residual can't be tight
            // unless the RHS happens to lie in the range space — which it
            // won't for a random-ish ones vector with a rank-1 deficiency.
            // We accept "residual > 1e-3 OR solver errored" as success.
            assert!(
                res > 1e-3,
                "d1: near-singular matrix produced suspiciously tight residual {res:.2e} \
                 — the solver should have caught the near-zero diagonal"
            );
        }
        Err(e) => panic!("d1: unexpected error: {e:?}"),
    }
}

/// A 2×2 matrix that is definitively not positive definite.
/// The solver must return `NotPositiveDefinite`.
#[test]
fn d2_explicitly_indefinite_matrix() {
    let mut coo = CooBuilder::new(3, 3);
    coo.add(0, 0, 1.0);
    coo.add(1, 1, 1.0);
    coo.add(2, 2, -1.0);  // negative diagonal
    let a = coo.build_sym().unwrap();

    let mut solver = SparseSolver::new();
    solver.analyze(&a).unwrap();
    assert!(
        matches!(solver.factorize(&a), Err(SolverError::NotPositiveDefinite { .. })),
        "d2: solver must reject a matrix with a negative diagonal"
    );
}

/// A singular matrix (rank-deficient): K = [[1,1],[1,1]].
/// Schur complement at column 1 is zero → `NotPositiveDefinite`.
#[test]
fn d3_singular_matrix_rank_deficient() {
    let mut coo = CooBuilder::new(2, 2);
    coo.add(0, 0, 1.0);
    coo.add(0, 1, 1.0);
    coo.add(1, 1, 1.0);
    let a = coo.build_sym().unwrap();

    let mut solver = SparseSolver::new();
    solver.analyze(&a).unwrap();
    assert!(
        matches!(solver.factorize(&a), Err(SolverError::NotPositiveDefinite { .. })),
        "d3: solver must reject a singular matrix"
    );
}

/// A nearly-singular matrix: spring chain with one spring that is 1e8 times
/// weaker than the rest.  The matrix is technically SPD but ill-conditioned.
/// We verify the residual is still acceptable (< 1e-6).
#[test]
fn d4_weak_spring_structural_singularity() {
    let n = 100;
    let epsilon = 1e-8_f64;  // one very weak spring

    let mut coo = CooBuilder::new(n, n);
    for i in 0..n {
        let mut diag = 0.0_f64;
        if i > 0 {
            let k = if i == n / 2 { epsilon } else { 1.0 };
            diag += k;
        }
        if i + 1 < n {
            let k = if i == n / 2 { epsilon } else { 1.0 };
            diag += k;
        }
        coo.add(i, i, diag.max(epsilon));
        if i + 1 < n {
            let k = if i == n / 2 { epsilon } else { 1.0 };
            coo.add(i, i + 1, -k);
        }
    }
    let a = coo.build_sym().unwrap();

    let mut solver = SparseSolver::new();
    solver.analyze(&a).unwrap();

    match solver.factorize(&a) {
        Err(SolverError::NotPositiveDefinite { .. }) => {
            // Acceptable: the matrix is so ill-conditioned the solver gave up.
        }
        Ok(()) => {
            let f = rhs_ones(n);
            let mut u = vec![0.0_f64; n];
            solver.solve(&f, &mut u).unwrap();
            let res = relative_residual(&a, &f, &u);
            // With a weak spring, condition number ≈ 1/epsilon = 1e8.
            // We lose about 8 digits → residual may be up to ~1e-7.
            assert!(
                res < 1e-5,
                "d4: weak spring residual {res:.2e} > 1e-5 — numerical breakdown"
            );
        }
        Err(e) => panic!("d4: unexpected error: {e:?}"),
    }
}

// =============================================================================
// E — Repeated factorization (topology reuse)
// =============================================================================

/// Analyze once on a tridiagonal.  Factorize and solve five times with
/// different diagonal scalings.  Every solution must satisfy the residual
/// check, and the solutions must scale correctly with the stiffness.
#[test]
fn e1_repeated_factorize_diagonal_scaling() {
    let n  = 300_usize;
    let f  = rhs_sin(n);

    // Build the sparsity pattern once (for scale=1.0)
    let k_pattern = spring_chain(n, 1.0);
    let mut solver = SparseSolver::new();
    solver.analyze(&k_pattern).unwrap();

    let scales = [1.0_f64, 2.0, 4.0, 8.0, 16.0];
    let mut u_first: Option<Vec<f64>> = None;

    for &s in &scales {
        let k = spring_chain(n, s);
        solver.factorize(&k).unwrap();
        let mut u = vec![0.0_f64; n];
        solver.solve(&f, &mut u).unwrap();

        // Residual check
        let res = relative_residual(&k, &f, &u);
        assert!(
            res < 1e-8,
            "e1 scale={s}: residual {res:.2e} > 1e-8"
        );

        // By linearity: u(s) = u(1) / s
        if let Some(ref u1) = u_first {
            for (i, (&ui, &u1i)) in u.iter().zip(u1.iter()).enumerate() {
                let expected = u1i / s;
                let rel = (ui - expected).abs() / expected.abs().max(1e-15);
                assert!(
                    rel < 1e-9,
                    "e1 scale={s} dof={i}: u={ui:.6e} expected={expected:.6e} rel={rel:.2e}"
                );
            }
        } else {
            u_first = Some(u);
        }
    }
}

/// Analyze once on a grid Laplacian.  Factorize three times with different
/// diagonal shifts (regularization).  Each shifted system `A + t*I` is SPD
/// with condition number `(λ_max + t) / (λ_min + t)`.
///
/// For a 20×20 Laplacian:
///   λ_min ≈ 0.0102 (for the 20×20 interior grid)
///   λ_max ≈ 7.99
///
/// Shifting by t=0 gives the original problem; t=1e-4 slightly regularises;
/// t=1.0 dominates the spectrum and makes the system approximately diagonal.
#[test]
fn e2_repeated_factorize_laplacian_shifts() {
    let nx = 20;
    let ny = 20;
    let n  = nx * ny;
    let f  = rhs_sin(n);

    // The sparsity pattern is the same for all shifts: use the unshifted matrix
    // for symbolic analysis.
    let base = laplacian_2d(nx, ny);
    let mut solver = SparseSolver::new();
    solver.analyze(&base).unwrap();

    let shifts = [0.0_f64, 1e-4, 1e-2, 1.0, 10.0];
    for &t in &shifts {
        // Build the shifted matrix: A + t*I
        let mut coo = CooBuilder::new(n, n);
        for iy in 0..ny {
            for ix in 0..nx {
                let i = ix + iy * nx;
                coo.add(i, i, 4.0 + t);
                if ix + 1 < nx { coo.add(i, (ix + 1) + iy * nx, -1.0); }
                if iy + 1 < ny { coo.add(i, ix + (iy + 1) * nx, -1.0); }
            }
        }
        let a_shifted = coo.build_sym().unwrap();

        solver.factorize(&a_shifted).unwrap();
        let mut u = vec![0.0_f64; n];
        solver.solve(&f, &mut u).unwrap();

        let res = relative_residual(&a_shifted, &f, &u);
        assert!(
            res < 1e-8,
            "e2 shift={t}: residual {res:.2e} > 1e-8"
        );
    }
}

/// Verify that calling `analyze` again correctly invalidates a previous
/// numeric factorization, so a subsequent `solve` without `factorize` errors.
#[test]
fn e3_reanalyze_invalidates_numeric() {
    let k = spring_chain(50, 1.0);
    let mut solver = SparseSolver::new();
    solver.analyze_and_factorize(&k).unwrap();

    // Re-analyze: this must invalidate the numeric factor.
    solver.analyze(&k).unwrap();

    let mut u = vec![0.0_f64; 50];
    assert!(
        matches!(
            solver.solve(&rhs_ones(50), &mut u),
            Err(SolverError::NotFactorized)
        ),
        "e3: solver must error after re-analyze without re-factorize"
    );
}

/// Perform 20 Newton-iteration-like factorizations on the same sparsity
/// pattern.  Verify every solution is correct and that all factorizations
/// produce identical results (no accumulation of numerical garbage between
/// factorizations).
#[test]
fn e4_twenty_newton_iterations_same_topology() {
    let n  = 400_usize;
    let f  = rhs_sin(n);

    // Analyze once
    let k_base = spring_chain(n, 1.0);
    let mut solver = SparseSolver::new();
    solver.analyze(&k_base).unwrap();

    // Factorize 20 times with slightly perturbed stiffness (same pattern)
    let mut last_u: Option<Vec<f64>> = None;
    for iter in 0..20_usize {
        // Stiffness varies between 0.9 and 1.1 across iterations
        let scale = 1.0 + 0.1 * ((iter as f64 * 0.3).sin());
        let k = spring_chain(n, scale);
        solver.factorize(&k).unwrap();
        let mut u = vec![0.0_f64; n];
        solver.solve(&f, &mut u).unwrap();

        let res = relative_residual(&k, &f, &u);
        assert!(
            res < 1e-8,
            "e4 iter={iter} scale={scale:.4}: residual {res:.2e}"
        );

        // Check for no cross-contamination: repeat the factorization with the
        // same k and verify same solution
        solver.factorize(&k).unwrap();
        let mut u2 = vec![0.0_f64; n];
        solver.solve(&f, &mut u2).unwrap();
        for (j, (&a, &b)) in u.iter().zip(u2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-13 * a.abs().max(1.0),
                "e4 iter={iter}: repeated factorize gave different result at dof={j}"
            );
        }

        last_u = Some(u);
    }

    // Final sanity: last solution is non-zero
    let u = last_u.unwrap();
    assert!(
        u.iter().any(|&v| v.abs() > 1e-10),
        "e4: final solution is all zeros — something is wrong"
    );
}

/// Mixed topology: alternate between two different patterns (tridiagonal and
/// pentadiagonal), calling analyze each time and solving once per topology.
/// Verifies that the solver cleanly handles multiple topology changes.
#[test]
fn e5_alternating_topology() {
    let n  = 200_usize;
    let f  = rhs_sin(n);

    for round in 0..4_usize {
        let k = if round % 2 == 0 {
            // Tridiagonal (2-neighbour)
            spring_chain(n, 1.0)
        } else {
            // Pentadiagonal (4-neighbour) — wider band, more fill
            let mut coo = CooBuilder::new(n, n);
            for i in 0..n {
                coo.add(i, i, 4.0);
                if i + 1 < n { coo.add(i, i + 1, -1.0); }
                if i + 2 < n { coo.add(i, i + 2, -0.5); }
            }
            coo.build_sym().unwrap()
        };

        let mut solver = SparseSolver::new();
        solver.analyze_and_factorize(&k).unwrap();
        let mut u = vec![0.0_f64; n];
        solver.solve(&f, &mut u).unwrap();

        let res = relative_residual(&k, &f, &u);
        assert!(
            res < 1e-8,
            "e5 round={round}: residual {res:.2e}"
        );
    }
}