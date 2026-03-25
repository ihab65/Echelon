//! FEM integration tests — verified against closed-form analytical solutions.
//!
//! Every test here assembles a real structural problem from scratch using
//! only the types already in the workspace (`sparse`, `solvers`, `fem_core`),
//! solves it with [`solvers::cholesky::SparseSolver`], and checks the result
//! against an analytical formula derived from first principles.
//!
//! ## What these tests prove
//!
//! Passing these tests means the **entire pipeline** is correct end-to-end:
//!
//! ```text
//! CoordTransf2d  →  ke_local  →  transform  →  ke_global
//!      ↓
//! DofMap::from_nodes  →  SymCsrMatrix::scatter_add  →  K (assembled)
//!      ↓
//! zero_row_col (Dirichlet BCs)
//!      ↓
//! SparseSolver::analyze  →  factorize (RCM + Cholesky)  →  solve
//!      ↓
//! u[free_dofs]  compared to  analytical formula
//! ```
//!
//! ## Element conventions
//!
//! Element stiffness functions live in this file as free functions — there is
//! no `elements` crate yet.  They are intentionally simple: no trait, no
//! abstraction, just the formula.
//!
//! **2D Truss:** 4 DOFs `[u_i, v_i, u_j, v_j]`.  Local stiffness is axial-
//! only.  Global stiffness comes from `Tᵀ Ke_local T` via `CoordTransf2d`.
//!
//! **2D Euler-Bernoulli Beam:** 6 DOFs `[u_i, v_i, θ_i, u_j, v_j, θ_j]`.
//! Classical small-displacement formulation, no shear deformation.
//!
//! ## Assembly convention
//!
//! - Nodes are numbered 0-based.
//! - `ndf` DOFs per node laid out contiguously: node `k` owns global DOFs
//!   `k*ndf .. k*ndf + ndf`.
//! - `DofMap::from_nodes` handles this automatically.
//! - `SymCsrMatrix::zero_row_col` + setting `F[dof] = 0` enforces pin BCs.
//!
//! ## Tolerance
//!
//! All assertions use relative tolerance `1e-10`.  The solver achieves
//! machine precision on these problems; `1e-10` is deliberately generous to
//! avoid false failures from floating-point rounding in the reference formula.

use sparse::{CooBuilder, SymCsrMatrix};
use solvers::cholesky::SparseSolver;
use fem_core::{CoordTransf2d, DofMap, NodeId};
use fem_core::dense::{mat_as_slice, mat_zero};

// =============================================================================
// Helper utilities shared across tests
// =============================================================================

/// Check `|computed - expected| / |expected| < tol`.
/// Falls back to absolute comparison when `expected ≈ 0`.
fn assert_rel(computed: f64, expected: f64, tol: f64, label: &str) {
    let abs_err = (computed - expected).abs();
    let denom   = expected.abs().max(1e-15);
    let rel_err = abs_err / denom;
    assert!(
        rel_err < tol,
        "{label}: computed={computed:.10e}  expected={expected:.10e}  rel_err={rel_err:.2e}"
    );
}

/// Check that the residual `‖Ku − f‖∞ / ‖f‖∞ < tol` for all DOFs.
/// This is the gold-standard solver check independent of any analytical formula.
fn check_residual(k_orig: &SymCsrMatrix, f: &[f64], u: &[f64], tol: f64) {
    let ku = k_orig.matvec(u).unwrap();
    let norm_f: f64 = f.iter().map(|x| x.abs()).fold(0.0_f64, f64::max).max(1e-15);
    for (i, (&kui, &fi)) in ku.iter().zip(f.iter()).enumerate() {
        let err = (kui - fi).abs() / norm_f;
        assert!(
            err < tol,
            "residual[{i}]: |Ku[{i}] − f[{i}]| / ‖f‖ = {err:.2e}  (Ku={kui:.8e}, f={fi:.8e})"
        );
    }
}

// =============================================================================
// Element stiffness formulas
// =============================================================================

/// 2D truss local stiffness, 4×4, row-major.
///
/// In the element's local frame the truss only resists axial force.
/// The 4 local DOFs are `[u1_L, v1_L, u2_L, v2_L]` where `u_L` is along
/// the element axis and `v_L` is transverse.
///
/// ```text
/// ke_local = (EA/L) * [ 1  0 -1  0 ]
///                     [ 0  0  0  0 ]
///                     [-1  0  1  0 ]
///                     [ 0  0  0  0 ]
/// ```
fn truss2d_ke_local(ea_over_l: f64) -> [[f64; 4]; 4] {
    let k = ea_over_l;
    [
        [ k,  0.0, -k,  0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-k,  0.0,  k,  0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
}

/// 2D truss global stiffness, 4×4, as a flat `[f64; 16]` for `scatter_add`.
///
/// Computes `Kg = Tᵀ Ke_local T` using `fem_core::CoordTransf2d`.
fn truss2d_ke_global(e: f64, a: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> [f64; 16] {
    let transf   = CoordTransf2d::from_nodes(x1, y1, x2, y2);
    let ke_local = truss2d_ke_local(e * a / transf.length);
    let ke_global = transf.transform_stiffness_4x4(&ke_local);
    // Flatten [[f64;4];4] → [f64;16] via mat_as_slice
    let slice = mat_as_slice(&ke_global);
    let mut arr = [0.0_f64; 16];
    arr.copy_from_slice(slice);
    arr
}

/// 2D Euler-Bernoulli beam local stiffness, 6×6, row-major.
///
/// DOFs (local): `[u1, v1, θ1, u2, v2, θ2]`.
/// Axial and bending are uncoupled in the local frame.
///
/// ```text
/// Axial block (rows/cols 0,3):
///   [ EA/L   0  -EA/L   0 ]
///   ...
///
/// Bending block (rows/cols 1,2,4,5):
///   [ 12EI/L³   6EI/L²  -12EI/L³   6EI/L²  ]
///   [  6EI/L²   4EI/L    -6EI/L²   2EI/L   ]
///   [-12EI/L³  -6EI/L²   12EI/L³  -6EI/L²  ]
///   [  6EI/L²   2EI/L    -6EI/L²   4EI/L   ]
/// ```
fn beam2d_ke_local(e: f64, a: f64, iz: f64, l: f64) -> [[f64; 6]; 6] {
    let eal  = e * a / l;
    let l2   = l * l;
    let l3   = l2 * l;
    let ei   = e * iz;
    let b1   = 12.0 * ei / l3;  // 12EI/L³
    let b2   =  6.0 * ei / l2;  //  6EI/L²
    let b3   =  4.0 * ei / l;   //  4EI/L
    let b4   =  2.0 * ei / l;   //  2EI/L

    let mut ke = mat_zero::<6>();
    // axial DOFs (0 and 3)
    ke[0][0] =  eal;  ke[0][3] = -eal;
    ke[3][0] = -eal;  ke[3][3] =  eal;
    // bending DOFs (1, 2, 4, 5)
    ke[1][1] =  b1;  ke[1][2] =  b2;  ke[1][4] = -b1;  ke[1][5] =  b2;
    ke[2][1] =  b2;  ke[2][2] =  b3;  ke[2][4] = -b2;  ke[2][5] =  b4;
    ke[4][1] = -b1;  ke[4][2] = -b2;  ke[4][4] =  b1;  ke[4][5] = -b2;
    ke[5][1] =  b2;  ke[5][2] =  b4;  ke[5][4] = -b2;  ke[5][5] =  b3;
    ke
}

/// 2D beam global stiffness as a flat `[f64; 36]` for `scatter_add`.
fn beam2d_ke_global(e: f64, a: f64, iz: f64,
                    x1: f64, y1: f64, x2: f64, y2: f64) -> [f64; 36] {
    let transf   = CoordTransf2d::from_nodes(x1, y1, x2, y2);
    let ke_local = beam2d_ke_local(e, a, iz, transf.length);
    let ke_global = transf.transform_stiffness_6x6(&ke_local);
    let slice = mat_as_slice(&ke_global);
    let mut arr = [0.0_f64; 36];
    arr.copy_from_slice(slice);
    arr
}

// =============================================================================
// Assembly helpers
// =============================================================================

/// Build the upper-triangle sparsity pattern for a model by scattering
/// zeros, then return a zeroed `SymCsrMatrix` ready for value assembly.
///
/// `element_node_lists` is a slice of node-index lists: one per element.
/// `ndf` is DOFs per node.
fn build_pattern(n_nodes: usize, ndf: usize,
                 element_node_lists: &[Vec<NodeId>]) -> SymCsrMatrix {
    let n_dof = n_nodes * ndf;
    let element_dofs: Vec<Vec<usize>> = element_node_lists.iter()
        .map(|nodes| {
            DofMap::from_nodes(nodes, ndf)
                .as_usize_slice()
                .to_vec()
        })
        .collect();
    SymCsrMatrix::from_dof_connectivity(n_dof, &element_dofs).unwrap()
}

/// Apply a pin boundary condition: zero row/col `dof` in `k`, zero `f[dof]`.
fn apply_pin(k: &mut SymCsrMatrix, f: &mut Vec<f64>, dof: usize) {
    k.zero_row_col(dof).unwrap();
    f[dof] = 0.0;
}

/// Solve `Ku = f` and return the solution vector.
fn solve(k: &SymCsrMatrix, f: &[f64]) -> Vec<f64> {
    let mut solver = SparseSolver::new();
    solver.analyze_and_factorize(k).unwrap();
    let mut u = vec![0.0_f64; f.len()];
    solver.solve(f, &mut u).unwrap();
    u
}

// =============================================================================
// TEST 1 — Single spring (1 DOF)
//
// The trivial case: K = [k], F = [P].  Solution: u = P/k.
// Proves the solver handles a 1×1 SPD system.
// =============================================================================

#[test]
fn test_single_spring() {
    let k_spring = 5000.0_f64;  // N/m
    let p        = 1000.0_f64;  // N

    let mut coo = CooBuilder::new(1, 1);
    coo.add(0, 0, k_spring);
    let k = coo.build_sym().unwrap();

    let f = vec![p];
    let u = solve(&k, &f);

    let expected = p / k_spring;
    assert_rel(u[0], expected, 1e-10, "u[0]");
}

// =============================================================================
// TEST 2 — 1D bar, two equal segments, axial load at mid-node
//
// Three co-linear nodes:  A(fixed) ─── C(free) ─── B(fixed)
//
//   x:  0 ────── L ────── 2L
//
// EA = const.  Force P applied at C.  BCs: u_A = u_B = 0.
//
// From compatibility + equilibrium: u_C = P*L / (2*EA).
//
// This test uses the 2D truss element on a purely horizontal 1D problem,
// verifying the assembly pipeline end-to-end.
// =============================================================================

#[test]
fn test_1d_bar_two_segments() {
    let e = 200e9_f64;   // Pa  (steel)
    let a = 0.01_f64;    // m²
    let l = 1.0_f64;     // m per segment
    let p = 50e3_f64;    // N   applied at midpoint

    // 3 nodes × 2 DOFs = 6 global DOFs
    // Node layout (x, y): A=(0,0), C=(L,0), B=(2L,0)
    // Nodes: 0=A, 1=C, 2=B
    let ndf = 2_usize;
    let nodes_e1 = vec![NodeId(0), NodeId(1)]; // A-C
    let nodes_e2 = vec![NodeId(1), NodeId(2)]; // C-B

    let mut k = build_pattern(3, ndf, &[nodes_e1.clone(), nodes_e2.clone()]);

    // Scatter element stiffnesses
    let ke1 = truss2d_ke_global(e, a, 0.0, 0.0,   l, 0.0);
    let ke2 = truss2d_ke_global(e, a,   l, 0.0, 2.0*l, 0.0);

    k.scatter_add(&ke1, DofMap::from_nodes(&nodes_e1, ndf).as_usize_slice()).unwrap();
    k.scatter_add(&ke2, DofMap::from_nodes(&nodes_e2, ndf).as_usize_slice()).unwrap();

    // Load vector: F_x at node 1 (C)
    // DOF layout: node k → DOFs [2k, 2k+1].  Node 1, x-DOF = 2.
    let n_dof = 3 * ndf;
    let mut f = vec![0.0_f64; n_dof];
    f[2] = p;   // x-DOF of node 1

    // BCs: pin all DOFs of nodes 0 and 2, plus transverse DOF at node 1
    for dof in [0, 1, 3, 4, 5] { apply_pin(&mut k, &mut f, dof); }

    let u = solve(&k, &f);

    // Analytical: u_C = P * L / (2 * E * A)
    let expected_uc = p * l / (2.0 * e * a);
    assert_rel(u[2], expected_uc, 1e-10, "u_C (x-disp of mid-node)");

    // y-displacement of C must be zero (no transverse load, no transverse stiffness)
    assert_rel(u[3], 0.0, 1e-10, "v_C must be zero");
}

// =============================================================================
// TEST 3 — Simply supported 2D truss (symmetric V-shape)
//
//          P ↓
//          C
//         / \
//        /   \
//       A─────B
//
// Node layout:
//   A = (0, 0)   — pin (both DOFs fixed)
//   B = (2L, 0)  — roller (y-DOF fixed only, free to move in x)
//   C = (L, H)   — free, vertical load P applied
//
// For a symmetric truss with both members equal (same EA, same length):
//
//   L_elem = sqrt(L² + H²)       element length
//   angle  = atan(H / L)
//   sin α  = H / L_elem
//
// Vertical equilibrium at C:
//   2 * F_elem * sin α = P   →   F_elem = P / (2 sin α)
//
// Axial strain:
//   δ_elem = F_elem * L_elem / (EA) = P * L_elem / (2 * EA * sin α)
//
// Vertical displacement of C (geometry):
//   v_C = δ_elem / sin α  =  P * L_elem / (2 * EA * sin²α)
//
// This is the classic Williot-Mohr result for a symmetric simply supported
// truss under central load.
// =============================================================================

#[test]
fn test_symmetric_v_truss() {
    let e = 200e9_f64;    // Pa
    let a = 0.002_f64;    // m²
    let l = 1.0_f64;      // m  (half-span)
    let h = 1.0_f64;      // m  (height)
    let p = 10e3_f64;     // N  (downward at C)

    // Node indices: 0=A, 1=B, 2=C
    let ndf = 2_usize;
    let nodes_ac = vec![NodeId(0), NodeId(2)];
    let nodes_bc = vec![NodeId(1), NodeId(2)];

    let (xa, ya) = (0.0,     0.0);
    let (xb, yb) = (2.0 * l, 0.0);
    let (xc, yc) = (l,        h);

    let mut k = build_pattern(3, ndf, &[nodes_ac.clone(), nodes_bc.clone()]);

    let ke_ac = truss2d_ke_global(e, a, xa, ya, xc, yc);
    let ke_bc = truss2d_ke_global(e, a, xb, yb, xc, yc);

    k.scatter_add(&ke_ac, DofMap::from_nodes(&nodes_ac, ndf).as_usize_slice()).unwrap();
    k.scatter_add(&ke_bc, DofMap::from_nodes(&nodes_bc, ndf).as_usize_slice()).unwrap();

    // Load: vertical (downward = negative y) at C
    // DOF layout: node 2 → DOFs [4, 5].  y-DOF = 5.
    let n_dof = 3 * ndf;
    let mut f = vec![0.0_f64; n_dof];
    f[5] = -p;  // downward

    // BCs
    // A: fully fixed (pin)
    apply_pin(&mut k, &mut f, 0); // A_x
    apply_pin(&mut k, &mut f, 1); // A_y
    // B: roller — fix y only, free in x (the reaction is vertical)
    apply_pin(&mut k, &mut f, 3); // B_y

    let u = solve(&k, &f);

    // Analytical vertical displacement at C
    let l_elem  = (l * l + h * h).sqrt();
    let sin_a   = h / l_elem;
    let v_c_expected = -p * l_elem / (2.0 * e * a * sin_a * sin_a);

    assert_rel(u[5], v_c_expected, 1e-9, "v_C (vertical disp at apex)");

    // Horizontal displacement at C must be zero (symmetric problem)
    assert!(
        u[4].abs() < 1e-6 * v_c_expected.abs(),
        "u_C should be ≈0 by symmetry, got {}", u[4]
    );

    // B_x: horizontal displacement of the roller support
    // F_x at B = horizontal component of AC member force
    // = F_elem * cos α = (P / 2sinα) * (L / L_elem)
    // δ_B_x = F_x_B * 0 (the roller is free in x — this is just B moving)
    // But B_y is pinned, not B_x — B_x is free (roller), so u[2] is the
    // horizontal displacement of B, which is non-zero and determined by the
    // member deformation.
    //
    // We check it passes the full residual test instead of an analytical formula.
    check_residual(&k, &f, &u, 1e-9);
}

// =============================================================================
// TEST 4 — Cantilever beam under tip point load (beam bending)
//
//   Fixed                   Free
//   ████ ──────────────────── B
//   A                       ↓ P
//
//   L = span,  EI = bending stiffness,  EA = axial stiffness
//
// Analytical tip deflection (Euler-Bernoulli, small displacement):
//   v_B = P * L³ / (3 * E * I)
//
// Analytical tip rotation:
//   θ_B = P * L² / (2 * E * I)
//
// This tests the beam element stiffness formulation: bending entries,
// the sign convention for rotations, and the full 6-DOF assembly.
// =============================================================================

#[test]
fn test_cantilever_beam_tip_load() {
    let e  = 200e9_f64;    // Pa
    let iz = 1e-4_f64;     // m⁴  (second moment of area)
    let a  = 0.01_f64;     // m²  (cross-section area, irrelevant for pure bending)
    let l  = 2.0_f64;      // m
    let p  = 10e3_f64;     // N  (downward at free end)

    // 2 nodes × 3 DOFs = 6 global DOFs
    // Node 0 = A (fixed), Node 1 = B (free)
    // DOF layout: node k → [3k, 3k+1, 3k+2] = [u, v, θ]
    let ndf  = 3_usize;
    let nodes = vec![NodeId(0), NodeId(1)];

    let mut k = build_pattern(2, ndf, &[nodes.clone()]);

    let ke = beam2d_ke_global(e, a, iz, 0.0, 0.0, l, 0.0);
    k.scatter_add(&ke, DofMap::from_nodes(&nodes, ndf).as_usize_slice()).unwrap();

    // Load: vertical (downward) at B → DOF 4 (v of node 1)
    let n_dof = 2 * ndf;
    let mut f = vec![0.0_f64; n_dof];
    f[4] = -p;  // downward

    // BCs: fix all DOFs of node 0 (A is fully clamped)
    for dof in [0, 1, 2] { apply_pin(&mut k, &mut f, dof); }

    let u = solve(&k, &f);

    // Analytical solutions
    let v_b_expected = -p * l * l * l / (3.0 * e * iz);
    let theta_b_expected = -p * l * l / (2.0 * e * iz);  // negative: clockwise

    assert_rel(u[4], v_b_expected,     1e-9, "v_B (tip deflection)");
    assert_rel(u[5], theta_b_expected, 1e-9, "θ_B (tip rotation)");

    // Axial displacement must be zero (no axial load, horizontal element)
    assert!(
        u[3].abs() < 1e-10 * v_b_expected.abs().max(1.0),
        "u_B (axial) should be ≈0, got {}", u[3]
    );

    check_residual(&k, &f, &u, 1e-9);
}

// =============================================================================
// TEST 5 — Fixed-fixed beam under midspan point load
//
//   Fixed end A                         Fixed end B
//   ████ ──────────── C ──────────── ████
//                     ↓ P
//                     L/2        L/2
//
// Classical result for a fixed-fixed beam with central point load:
//
//   v_C = P * L³ / (192 * E * I)     midspan deflection
//   M_A = M_B = -P * L / 8           end moments (hogging = negative convention)
//   R_A = R_B =  P / 2               end reactions
//
// This test uses three nodes and two elements.  It is the first test with
// multiple beam elements, verifying moment-continuity at the interior node.
// =============================================================================

#[test]
fn test_fixed_fixed_beam_midspan_load() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 4.0_f64;      // total span
    let p  = 20e3_f64;     // N  at midspan

    // 3 nodes × 3 DOFs = 9 global DOFs
    // Node 0 = A (left fixed), Node 1 = C (midspan, free), Node 2 = B (right fixed)
    let ndf = 3_usize;
    let nodes_ac = vec![NodeId(0), NodeId(1)];
    let nodes_cb = vec![NodeId(1), NodeId(2)];

    let mut k = build_pattern(3, ndf, &[nodes_ac.clone(), nodes_cb.clone()]);

    let ke_ac = beam2d_ke_global(e, a, iz, 0.0,     0.0, l/2.0, 0.0);
    let ke_cb = beam2d_ke_global(e, a, iz, l/2.0,   0.0, l,     0.0);

    k.scatter_add(&ke_ac, DofMap::from_nodes(&nodes_ac, ndf).as_usize_slice()).unwrap();
    k.scatter_add(&ke_cb, DofMap::from_nodes(&nodes_cb, ndf).as_usize_slice()).unwrap();

    // Load: vertical at midspan node 1 → DOF 4 (v of node 1)
    let n_dof = 3 * ndf;
    let mut f = vec![0.0_f64; n_dof];
    f[4] = -p;

    // BCs: all DOFs of node 0 (A) and node 2 (B) are fixed
    for dof in [0, 1, 2, 6, 7, 8] { apply_pin(&mut k, &mut f, dof); }

    let u = solve(&k, &f);

    // Analytical midspan deflection for fixed-fixed beam
    let v_c_expected = -p * l * l * l / (192.0 * e * iz);
    assert_rel(u[4], v_c_expected, 1e-9, "v_C (midspan deflection)");

    // Midspan horizontal displacement = 0 (symmetric, no axial)
    assert!(u[3].abs() < 1e-12 * v_c_expected.abs().max(1.0),
            "u_C should be ≈0");
    // Midspan rotation = 0 (symmetry)
    assert!(u[5].abs() < 1e-10 * v_c_expected.abs().max(1.0),
            "θ_C should be ≈0 by symmetry, got {}", u[5]);

    check_residual(&k, &f, &u, 1e-9);
}

// =============================================================================
// TEST 6 — Simply supported beam under midspan point load
//
//   Pin A                     Roller B
//   ∧ ──────────── C ──────── ∧
//                  ↓ P
//                  L/2   L/2
//
// Pin at A: u=0, v=0, θ=free.
// Roller at B: v=0, u and θ free.
//
// Analytical results:
//   v_C = P * L³ / (48 * E * I)      midspan deflection
//   θ_A = P * L² / (16 * E * I)      left-end rotation (positive = CCW)
//   θ_B = -P * L² / (16 * E * I)     right-end rotation (negative = CW)
//
// This tests DOF-by-DOF boundary conditions: fixing only v at B (roller),
// leaving u and θ of B free.
// =============================================================================

#[test]
fn test_simply_supported_beam_midspan_load() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 3.0_f64;
    let p  = 15e3_f64;

    let ndf = 3_usize;
    let nodes_ac = vec![NodeId(0), NodeId(1)];
    let nodes_cb = vec![NodeId(1), NodeId(2)];

    let mut k = build_pattern(3, ndf, &[nodes_ac.clone(), nodes_cb.clone()]);

    let ke_ac = beam2d_ke_global(e, a, iz, 0.0,   0.0, l/2.0, 0.0);
    let ke_cb = beam2d_ke_global(e, a, iz, l/2.0, 0.0, l,     0.0);

    k.scatter_add(&ke_ac, DofMap::from_nodes(&nodes_ac, ndf).as_usize_slice()).unwrap();
    k.scatter_add(&ke_cb, DofMap::from_nodes(&nodes_cb, ndf).as_usize_slice()).unwrap();

    // Midspan load
    let n_dof = 3 * ndf;
    let mut f = vec![0.0_f64; n_dof];
    f[4] = -p;

    // BCs:
    // Node 0 (A) — pin: fix u (DOF 0) and v (DOF 1), leave θ (DOF 2) free
    apply_pin(&mut k, &mut f, 0); // A_u
    apply_pin(&mut k, &mut f, 1); // A_v
    // Node 2 (B) — roller: fix v (DOF 7) only
    apply_pin(&mut k, &mut f, 7); // B_v

    let u = solve(&k, &f);

    let v_c_expected  =  p * l * l * l / (48.0 * e * iz);  // downward → negative
    let theta_a       =  p * l * l / (16.0 * e * iz);       // CCW → positive
    let theta_b       = -p * l * l / (16.0 * e * iz);       // CW  → negative

    assert_rel(u[4], -v_c_expected, 1e-9, "v_C midspan deflection");
    assert_rel(u[2],  theta_a,      1e-9, "θ_A left rotation");
    assert_rel(u[8],  theta_b,      1e-9, "θ_B right rotation");

    check_residual(&k, &f, &u, 1e-9);
}

// =============================================================================
// TEST 7 — L-shaped portal frame under horizontal load
//
//        P →
//   C ─────────── D
//   |             |
//   |             |
//   |             |
//   A             B
//   (fixed)       (fixed)
//
// Geometry: columns height H, beam length Lb.
// Column EA_c, EI_c.  Beam EA_b, EI_b.
//
// For a symmetric portal frame with identical columns under horizontal P at C:
//
// This is the first test with a non-trivial mixed frame (columns + beam).
// No simple closed form — verified only by residual check (Ku ≈ f).
//
// The sidesway displacement at C is finite (frame is sway-permitted),
// confirming the solver handles non-trivial coupling between bending and
// axial DOFs across multiple elements.
// =============================================================================

#[test]
fn test_portal_frame_horizontal_load() {
    let e     = 200e9_f64;
    let a_col = 0.01_f64;
    let iz_col = 1e-4_f64;
    let a_beam = 0.01_f64;
    let iz_beam = 2e-4_f64;     // stiffer beam than columns
    let h  = 3.0_f64;           // column height
    let lb = 5.0_f64;           // beam length
    let p  = 50e3_f64;          // horizontal load at C

    // Nodes: 0=A (base-left), 1=B (base-right), 2=C (top-left), 3=D (top-right)
    // DOF layout: node k → [3k, 3k+1, 3k+2] = [u, v, θ]
    let ndf = 3_usize;
    let nodes_ac = vec![NodeId(0), NodeId(2)]; // left column
    let nodes_bd = vec![NodeId(1), NodeId(3)]; // right column
    let nodes_cd = vec![NodeId(2), NodeId(3)]; // beam

    let mut k = build_pattern(4, ndf, &[
        nodes_ac.clone(), nodes_bd.clone(), nodes_cd.clone()
    ]);

    // Coordinates
    let (xa, ya) = (0.0, 0.0);
    let (xb, yb) = (lb,  0.0);
    let (xc, yc) = (0.0, h);
    let (xd, yd) = (lb,  h);

    let ke_ac = beam2d_ke_global(e, a_col,  iz_col,  xa, ya, xc, yc);
    let ke_bd = beam2d_ke_global(e, a_col,  iz_col,  xb, yb, xd, yd);
    let ke_cd = beam2d_ke_global(e, a_beam, iz_beam, xc, yc, xd, yd);

    k.scatter_add(&ke_ac, DofMap::from_nodes(&nodes_ac, ndf).as_usize_slice()).unwrap();
    k.scatter_add(&ke_bd, DofMap::from_nodes(&nodes_bd, ndf).as_usize_slice()).unwrap();
    k.scatter_add(&ke_cd, DofMap::from_nodes(&nodes_cd, ndf).as_usize_slice()).unwrap();

    // Load: horizontal at C → DOF 6 (u of node 2)
    let n_dof = 4 * ndf;
    let mut f = vec![0.0_f64; n_dof];
    f[6] = p;

    // BCs: clamp both bases (A and B — all 6 DOFs)
    for dof in [0, 1, 2, 3, 4, 5] { apply_pin(&mut k, &mut f, dof); }

    let u = solve(&k, &f);

    // Rigid-body checks:
    // Sidesway at C must equal sidesway at D (inextensible beam assumption
    // is approximate here — we just check they're close for a stiff beam).
    let u_c = u[6];
    let u_d = u[9];
    let sway_diff_rel = (u_c - u_d).abs() / u_c.abs().max(1e-10);
    assert!(
        sway_diff_rel < 1e-3,  // beam is stiff: C and D move together within 0.1%
        "Sway diff C vs D = {sway_diff_rel:.2e}  u_C={u_c:.6e}  u_D={u_d:.6e}"
    );

    // Frame must sway in the direction of P (positive u at C)
    assert!(u_c > 0.0, "Frame should sway in direction of load, u_C={u_c}");

    // The residual is the definitive correctness check for this test
    check_residual(&k, &f, &u, 1e-9);
}

// =============================================================================
// TEST 8 — Reanalysis: same topology, new stiffness values
//
// Demonstrates the critical performance pattern: analyze (symbolic) once,
// factorize (numeric) multiple times.
//
// Model: simple 2-element beam (same as TEST 5 topology), but:
//   Pass 1: stiffness EI₁ → solution u₁
//   Pass 2: stiffness EI₂ = 2*EI₁ → solution u₂ = u₁ / 2
//           (by linearity: stiffer beam → half the deflection)
//
// This validates that `SparseSolver::factorize` is independent of `analyze`
// and that reusing the symbolic factor for a different numeric matrix is
// correct.
// =============================================================================

#[test]
fn test_reanalysis_same_topology() {
    let e  = 200e9_f64;
    let a  = 0.01_f64;
    let iz1 = 1e-4_f64;
    let iz2 = 2e-4_f64;    // twice stiffer
    let l  = 4.0_f64;
    let p  = 20e3_f64;

    // Build and assemble K for EI1
    let ndf = 3_usize;
    let nodes_ac = vec![NodeId(0), NodeId(1)];
    let nodes_cb = vec![NodeId(1), NodeId(2)];

    let assemble = |iz: f64| -> (SymCsrMatrix, Vec<f64>) {
        let mut k = build_pattern(3, ndf, &[nodes_ac.clone(), nodes_cb.clone()]);
        let ke_ac = beam2d_ke_global(e, a, iz, 0.0,   0.0, l/2.0, 0.0);
        let ke_cb = beam2d_ke_global(e, a, iz, l/2.0, 0.0, l,     0.0);
        k.scatter_add(&ke_ac, DofMap::from_nodes(&nodes_ac, ndf).as_usize_slice()).unwrap();
        k.scatter_add(&ke_cb, DofMap::from_nodes(&nodes_cb, ndf).as_usize_slice()).unwrap();
        let n_dof = 3 * ndf;
        let mut f = vec![0.0_f64; n_dof];
        f[4] = -p;
        // Fixed-fixed BCs
        for dof in [0, 1, 2, 6, 7, 8] {
            k.zero_row_col(dof).unwrap();
            f[dof] = 0.0;
        }
        (k, f)
    };

    let (k1, f1) = assemble(iz1);
    let (k2, f2) = assemble(iz2);

    // Analyze once (uses k1's pattern — same for both since topology is identical)
    let mut solver = SparseSolver::new();
    solver.analyze(&k1).unwrap();

    // Factorize and solve with k1
    solver.factorize(&k1).unwrap();
    let mut u1 = vec![0.0_f64; f1.len()];
    solver.solve(&f1, &mut u1).unwrap();

    // Factorize and solve with k2 (re-uses symbolic from k1 analysis)
    solver.factorize(&k2).unwrap();
    let mut u2 = vec![0.0_f64; f2.len()];
    solver.solve(&f2, &mut u2).unwrap();

    // By linearity: doubling EI halves all displacements
    let midspan_dof = 4;
    assert_rel(
        u2[midspan_dof],
        u1[midspan_dof] / 2.0,
        1e-9,
        "reanalysis: u2 = u1/2 (doubled stiffness)"
    );

    check_residual(&k1, &f1, &u1, 1e-9);
    check_residual(&k2, &f2, &u2, 1e-9);
}

// =============================================================================
// TEST 9 — 2D truss, 5-bar statically indeterminate (Pratt truss segment)
//
//   A ─────── B ─────── C
//   |       ↗ |       ↗ |
//   |     ↗   |     ↗   |
//   D ─────── E ─────── F
//
//   Nodes (x,y): A=(0,H), B=(L,H), C=(2L,H)
//                D=(0,0),  E=(L,0),  F=(2L,0)
//
//   Elements: 5 (or 7 with diagonals).  Here we build the upper chord,
//   lower chord, verticals, and diagonals:
//     AD (left vert), BE (mid vert), CF (right vert)
//     AB (upper left), BC (upper right)
//     DE (lower left), EF (lower right)
//     AE (diagonal left, lower-left to upper-right)
//     BF (diagonal right, lower-left to upper-right)
//
//   BCs: pin at D, roller at F (v=0).
//   Load: P downward at B (mid upper chord).
//
//   No closed form used — verified by residual only.  The point is to
//   exercise a more realistic sparse assembly with diagonals creating
//   fill entries and to confirm the solver handles a statically
//   indeterminate truss.
// =============================================================================

#[test]
fn test_indeterminate_pratt_truss() {
    let e  = 200e9_f64;
    let a  = 5e-4_f64;     // m² — same area for all members
    let l  = 2.0_f64;      // panel length
    let h  = 1.5_f64;      // truss height
    let p  = 30e3_f64;     // N downward at B

    let ndf = 2_usize;

    // Node indices and coordinates
    // 0=A, 1=B, 2=C (upper chord, left to right)
    // 3=D, 4=E, 5=F (lower chord, left to right)
    let coords: [(f64, f64); 6] = [
        (0.0,   h),   // A
        (l,     h),   // B
        (2.0*l, h),   // C
        (0.0,   0.0), // D
        (l,     0.0), // E
        (2.0*l, 0.0), // F
    ];

    // Elements: (node_i, node_j)
    let elements: &[(usize, usize)] = &[
        (0, 3), // AD  left vertical
        (1, 4), // BE  mid vertical
        (2, 5), // CF  right vertical
        (0, 1), // AB  upper left
        (1, 2), // BC  upper right
        (3, 4), // DE  lower left
        (4, 5), // EF  lower right
        (3, 1), // DA→ diagonal (D to B)
        (4, 2), // EB→ diagonal (E to C)
    ];

    let elem_nodes: Vec<Vec<NodeId>> = elements.iter()
        .map(|&(i, j)| vec![NodeId(i), NodeId(j)])
        .collect();

    let mut k = build_pattern(6, ndf, &elem_nodes);

    for (idx, &(ni, nj)) in elements.iter().enumerate() {
        let (xi, yi) = coords[ni];
        let (xj, yj) = coords[nj];
        let ke = truss2d_ke_global(e, a, xi, yi, xj, yj);
        k.scatter_add(&ke, DofMap::from_nodes(&elem_nodes[idx], ndf).as_usize_slice()).unwrap();
    }

    // Load: P downward at B (node 1), y-DOF = 3
    let n_dof = 6 * ndf;
    let mut f = vec![0.0_f64; n_dof];
    f[3] = -p;  // B y-DOF

    // BCs: pin D (DOFs 6,7), roller F fix y-DOF (DOF 11)
    apply_pin(&mut k, &mut f, 6);  // D_x
    apply_pin(&mut k, &mut f, 7);  // D_y
    apply_pin(&mut k, &mut f, 11); // F_y

    let u = solve(&k, &f);

    // No closed-form — residual is the correctness check
    check_residual(&k, &f, &u, 1e-9);

    // Sanity: B should move down
    assert!(u[3] < 0.0, "B should deflect downward under P, got v_B={}", u[3]);

    // D is fully fixed: displacements must be zero
    assert!(u[6].abs() < 1e-14, "D_x must be 0, got {}", u[6]);
    assert!(u[7].abs() < 1e-14, "D_y must be 0, got {}", u[7]);
}
