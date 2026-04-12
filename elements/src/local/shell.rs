//! MITC4 shell stiffness assembly — local coordinate frame.
//!
//! This module is the shell analogue of `local::beam::ke_local`: it assembles
//! the full 24×24 local stiffness matrix for an `ElasticShell4` element by
//! numerical integration via the 2×2 Gauss rule.
//!
//! ## Assembly breakdown
//!
//! ```text
//! Ke_local = Ke_membrane + Ke_bending + Ke_shear + Ke_drill
//! ```
//!
//! | Part | B-matrix | Material matrix | Notes |
//! |------|----------|-----------------|-------|
//! | Membrane | B_m (3×24) | A = Et/(1−ν²)·C_ps | In-plane stretching |
//! | Bending | B_b (3×24) | D = Et³/12(1−ν²)·C_ps | Out-of-plane bending |
//! | Shear | B_s (2×24) | H = κ·G·t·I₂ | MITC4 transverse shear |
//! | Drill | diagonal | α·diag | Hughes-Brezzi penalty on θ_z |
//!
//! ## Stack allocation budget
//!
//! The largest temporary is the 24×24 stiffness accumulator `[f64; 576]`
//! (= 4.5 KiB). All B-matrices and intermediate products are fixed-size
//! stack arrays. No heap allocation occurs during stiffness computation.

use crate::local::gauss::{gauss_2x2, mitc4_tying_points};
use crate::local::isopar::{
    b_bending, b_membrane, b_shear_mitc4, jacobian, jacobian_inv, physical_derivs,
    shape_fn_derivs, shape_fns, tying_vector_r, tying_vector_s,
};

// -----------------------------------------------------------------
// ShellSectionStiffness
// -----------------------------------------------------------------

/// Pre-computed section-integrated material matrices for an isotropic shell.
///
/// Computed once at construction from `(E, ν, t)` and reused for every stiffness
/// or residual evaluation. Avoids recomputing material constants on the Gauss-point
/// hot path.
///
/// # Material matrices
///
/// The isotropic plane-stress tensor `C` (3×3, Voigt plane stress):
/// ```text
/// C = E/(1−ν²) · |  1    ν     0    |
///                 |  ν    1     0    |
///                 |  0    0   (1−ν)/2|
/// ```
///
/// - `a_membrane = t · C` — membrane stiffness (force/length)
/// - `d_bending  = (t³/12) · C` — bending stiffness (moment·length)
/// - `h_shear    = κ · G · t · I₂` — transverse shear stiffness (force/length)
/// - `alpha_drill = γ · G · t` — Hughes-Brezzi drilling penalty (force·length)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellSectionStiffness {
    /// 3×3 membrane stiffness matrix, row-major `[f64; 9]`.
    pub a_membrane: [f64; 9],
    /// 3×3 bending stiffness matrix, row-major `[f64; 9]`.
    pub d_bending: [f64; 9],
    /// 2×2 transverse shear stiffness matrix, row-major `[f64; 4]`.
    pub h_shear: [f64; 4],
    /// Hughes-Brezzi drilling penalty scalar `α = γ · G · t`.
    pub alpha_drill: f64,
}

impl ShellSectionStiffness {
    /// Compute section stiffness from isotropic material parameters.
    ///
    /// # Arguments
    /// * `e`           — Young's modulus (Pa)
    /// * `nu`          — Poisson's ratio (−1 < ν < 0.5)
    /// * `t`           — Shell thickness (m)
    /// * `gamma_drill` — Dimensionless drilling penalty (recommended: 0.001)
    /// * `kappa_shear` — Shear correction factor (recommended: 5/6 ≈ 0.8333)
    pub fn from_material(e: f64, nu: f64, t: f64, gamma_drill: f64, kappa_shear: f64) -> Self {
        let g     = e / (2.0 * (1.0 + nu));
        let factor = e / (1.0 - nu * nu);
        let half_nu_comp = 0.5 * (1.0 - nu);

        // Plane-stress C tensor (3×3 row-major)
        let c: [f64; 9] = [
            factor,         factor * nu,    0.0,
            factor * nu,    factor,         0.0,
            0.0,            0.0,            factor * half_nu_comp,
        ];

        // A_membrane = t · C
        let a_membrane = scale_mat3(&c, t);

        // D_bending = (t³/12) · C
        let d_bending = scale_mat3(&c, t * t * t / 12.0);

        // H_shear = κ · G · t · I₂
        let ks = kappa_shear * g * t;
        let h_shear = [ks, 0.0, 0.0, ks];

        // Hughes-Brezzi drilling penalty
        let alpha_drill = gamma_drill * g * t;

        Self { a_membrane, d_bending, h_shear, alpha_drill }
    }
}

/// Scale a 3×3 matrix (row-major flat) by a scalar.
#[inline]
fn scale_mat3(m: &[f64; 9], s: f64) -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    for i in 0..9 { out[i] = m[i] * s; }
    out
}

// -----------------------------------------------------------------
// B^T C B accumulation helper
// -----------------------------------------------------------------

/// Accumulate `Ke += B^T · C · B · scale` into the 24×24 stiffness.
///
/// * `ke`    — 24×24 stiffness accumulator, flat row-major `[f64; 576]`
/// * `b`     — m×24 B-matrix, flat row-major (m is 2 or 3)
/// * `c`     — m×m material matrix, flat row-major
/// * `m`     — number of strain components (2 or 3)
/// * `scale` — `det_J · weight`
///
/// The 24-DOF count `n = 24` is fixed at compile time.
fn accumulate_btcb(ke: &mut [f64; 576], b: &[f64], c: &[f64], m: usize, scale: f64) {
    const N: usize = 24;
    // T[p][j] = (C · B)[p][j] = Σ_q C[p][q] · B[q][j]
    let mut t = [0.0_f64; 3 * N]; // max m=3; only first m*N entries used
    for p in 0..m {
        for j in 0..N {
            let mut s = 0.0;
            for q in 0..m {
                s += c[p * m + q] * b[q * N + j];
            }
            t[p * N + j] = s;
        }
    }
    // ke[i][j] += Σ_p B^T[i][p] · T[p][j] · scale = Σ_p B[p][i] · T[p][j] · scale
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for p in 0..m {
                s += b[p * N + i] * t[p * N + j];
            }
            ke[i * N + j] += s * scale;
        }
    }
}

// -----------------------------------------------------------------
// Pre-compute tying vectors
// -----------------------------------------------------------------

/// Pre-computed MITC4 covariant shear tying vectors at all 4 tying points.
///
/// Stored separately for r (A, C) and s (B, D) tying directions.
struct TyingVectors {
    h_a: [f64; 24], // covariant r-shear at A=(0,-1)
    h_b: [f64; 24], // covariant s-shear at B=(+1,0)
    h_c: [f64; 24], // covariant r-shear at C=(0,+1)
    h_d: [f64; 24], // covariant s-shear at D=(-1,0)
}

fn compute_tying_vectors(xy_local: &[[f64; 2]; 4]) -> TyingVectors {
    let tying = mitc4_tying_points();
    // A=(0,-1): r-direction
    let (j_a, _) = jacobian(xy_local, tying[0].0, tying[0].1);
    let dn_a = shape_fn_derivs(tying[0].0, tying[0].1);
    let n_a  = shape_fns(tying[0].0, tying[0].1);
    let h_a  = tying_vector_r(&n_a, &dn_a[0], j_a[0][0], j_a[0][1]);

    // B=(+1,0): s-direction
    let (j_b, _) = jacobian(xy_local, tying[1].0, tying[1].1);
    let dn_b = shape_fn_derivs(tying[1].0, tying[1].1);
    let n_b  = shape_fns(tying[1].0, tying[1].1);
    let h_b  = tying_vector_s(&n_b, &dn_b[1], j_b[1][0], j_b[1][1]);

    // C=(0,+1): r-direction
    let (j_c, _) = jacobian(xy_local, tying[2].0, tying[2].1);
    let dn_c = shape_fn_derivs(tying[2].0, tying[2].1);
    let n_c  = shape_fns(tying[2].0, tying[2].1);
    let h_c  = tying_vector_r(&n_c, &dn_c[0], j_c[0][0], j_c[0][1]);

    // D=(-1,0): s-direction
    let (j_d, _) = jacobian(xy_local, tying[3].0, tying[3].1);
    let dn_d = shape_fn_derivs(tying[3].0, tying[3].1);
    let n_d  = shape_fns(tying[3].0, tying[3].1);
    let h_d  = tying_vector_s(&n_d, &dn_d[1], j_d[1][0], j_d[1][1]);

    TyingVectors { h_a, h_b, h_c, h_d }
}

// -----------------------------------------------------------------
// Local stiffness assembly
// -----------------------------------------------------------------

/// Compute the 24×24 local stiffness matrix for the MITC4 shell element.
///
/// All computations are performed in the **local shell coordinate frame**
/// (origin at centroid, x-y in the shell plane, z = normal direction).
///
/// # Arguments
/// * `xy_local` — 2D coordinates of the 4 nodes in the local plane
/// * `sec`      — pre-computed section stiffness matrices
///
/// # Returns
/// Flat row-major `[f64; 576]` — the 24×24 local stiffness.
///
/// # Stack allocation
/// No heap allocations are made. The largest stack temporary is the
/// 576-element stiffness accumulator itself (≈ 4.5 KiB).
pub fn ke_local_mitc4(xy_local: &[[f64; 2]; 4], sec: &ShellSectionStiffness) -> [f64; 576] {
    let mut ke = [0.0_f64; 576];
    let tv = compute_tying_vectors(xy_local);

    for gp in gauss_2x2() {
        let (j, det_j) = jacobian(xy_local, gp.r, gp.s);
        let j_inv = jacobian_inv(&j, det_j);
        let dn_drs = shape_fn_derivs(gp.r, gp.s);
        let dn_phys = physical_derivs(&j_inv, &dn_drs);
        let scale = det_j * gp.weight;

        // ---- Membrane contribution ----
        let b_m = b_membrane(&dn_phys[0], &dn_phys[1]);
        accumulate_btcb(&mut ke, &b_m, &sec.a_membrane, 3, scale);

        // ---- Bending contribution ----
        let b_b = b_bending(&dn_phys[0], &dn_phys[1]);
        accumulate_btcb(&mut ke, &b_b, &sec.d_bending, 3, scale);

        // ---- MITC4 transverse shear contribution ----
        let b_s = b_shear_mitc4(
            &tv.h_a, &tv.h_b, &tv.h_c, &tv.h_d,
            gp.r, gp.s, &j_inv,
        );
        accumulate_btcb(&mut ke, &b_s, &sec.h_shear, 2, scale);
    }

    // ---- Drilling DOF stabilisation (Hughes-Brezzi penalty) ----
    // Add α to the diagonal entry of each node's θ_z DOF (index 6I+5).
    // This is a lumped penalty: ∫ α · N_I² dA ≈ α · A / 4 per node,
    // but the simplest form (just diagonal) is standard practice.
    // We integrate α · N_I² over the element for each node.
    for gp in gauss_2x2() {
        let (_, det_j) = jacobian(xy_local, gp.r, gp.s);
        let n = shape_fns(gp.r, gp.s);
        let scale = det_j * gp.weight * sec.alpha_drill;
        for i in 0..4 {
            let col = 6 * i + 5; // θ_z drill DOF index
            ke[col * 24 + col] += scale * n[i] * n[i];
        }
    }

    ke
}

/// Compute the 24-component internal force vector in the local shell frame.
///
/// `f_int_local = Ke_local_mitc4 · u_local`
///
/// This is a simple matrix-vector product using the provided stiffness.
///
/// # Arguments
/// * `ke_local` — pre-computed 24×24 local stiffness (from [`ke_local_mitc4`])
/// * `u_local`  — 24-component displacement vector in the local frame
///
/// # Returns
/// `[f64; 24]` internal force vector in the local frame.
pub fn f_int_local_mitc4(ke_local: &[f64; 576], u_local: &[f64; 24]) -> [f64; 24] {
    let mut f = [0.0_f64; 24];
    for i in 0..24 {
        for j in 0..24 {
            f[i] += ke_local[i * 24 + j] * u_local[j];
        }
    }
    f
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn square_xy() -> [[f64; 2]; 4] {
        // Unit square centred at origin: nodes at (±1, ±1)
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    }

    fn steel_section() -> ShellSectionStiffness {
        ShellSectionStiffness::from_material(
            200e9,  // E (Pa)
            0.3,    // ν
            0.01,   // t (m)
            0.001,  // γ_drill
            5.0 / 6.0, // κ_shear
        )
    }

    // ---- ShellSectionStiffness ----

    #[test]
    fn section_stiffness_membrane_diagonal() {
        let sec = steel_section();
        // A_membrane[0][0] = E·t/(1-ν²)
        let expected = 200e9 * 0.01 / (1.0 - 0.3_f64.powi(2));
        assert!((sec.a_membrane[0] - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn section_stiffness_bending_is_t3_over_12_times_membrane() {
        let sec = steel_section();
        // D = t³/12 · C, A = t · C  →  D[i] = t²/12 · A[i]
        let ratio = 0.01_f64.powi(2) / 12.0;
        for i in 0..9 {
            assert!(
                (sec.d_bending[i] - sec.a_membrane[i] * ratio).abs() < 1e4,
                "D[{i}]={} vs A[{i}]*ratio={}", sec.d_bending[i], sec.a_membrane[i]*ratio
            );
        }
    }

    #[test]
    fn section_stiffness_shear_diagonal() {
        let sec = steel_section();
        let g = 200e9 / (2.0 * (1.0 + 0.3));
        let expected = (5.0 / 6.0) * g * 0.01;
        assert!((sec.h_shear[0] - expected).abs() / expected < 1e-10);
        assert!((sec.h_shear[3] - expected).abs() / expected < 1e-10);
        assert_eq!(sec.h_shear[1], 0.0);
        assert_eq!(sec.h_shear[2], 0.0);
    }

    // ---- ke_local_mitc4 ----

    #[test]
    fn ke_local_symmetric() {
        let ke = ke_local_mitc4(&square_xy(), &steel_section());
        for i in 0..24 {
            for j in 0..24 {
                assert!(
                    (ke[i * 24 + j] - ke[j * 24 + i]).abs() < 1.0,
                    "ke not symmetric at ({i},{j}): {} vs {}", ke[i*24+j], ke[j*24+i]
                );
            }
        }
    }

    #[test]
    fn ke_local_size() {
        let ke = ke_local_mitc4(&square_xy(), &steel_section());
        assert_eq!(ke.len(), 576);
    }

    #[test]
    fn ke_local_diagonal_positive() {
        // All diagonal entries should be positive
        let ke = ke_local_mitc4(&square_xy(), &steel_section());
        for i in 0..24 {
            assert!(
                ke[i * 24 + i] > 0.0,
                "ke[{i},{i}] = {} is not positive", ke[i*24+i]
            );
        }
    }

    #[test]
    fn f_int_local_zero_displacement() {
        let ke = ke_local_mitc4(&square_xy(), &steel_section());
        let u = [0.0_f64; 24];
        let f = f_int_local_mitc4(&ke, &u);
        assert!(f.iter().all(|&v| v == 0.0), "f_int for zero u should be zero");
    }

    #[test]
    fn f_int_local_consistent_with_ke() {
        // f_int = Ke · u for linear elastic
        let ke = ke_local_mitc4(&square_xy(), &steel_section());
        let mut u = [0.0_f64; 24];
        u[2] = 1e-3; // small transverse displacement of node 0
        let f = f_int_local_mitc4(&ke, &u);
        // Verify manually: f[i] = sum_j ke[i][j] * u[j]
        for i in 0..24 {
            let expected = ke[i * 24 + 2] * 1e-3;
            assert!(
                (f[i] - expected).abs() < 1e-6,
                "f[{i}]={} expected {expected}", f[i]
            );
        }
    }

    #[test]
    fn drill_dof_has_positive_stiffness() {
        let ke = ke_local_mitc4(&square_xy(), &steel_section());
        // Drilling DOF indices: 5, 11, 17, 23
        for node in 0..4 {
            let i = 6 * node + 5;
            assert!(
                ke[i * 24 + i] > 0.0,
                "drill stiffness at node {node} (dof {i}) must be > 0: {}", ke[i*24+i]
            );
        }
    }
}
