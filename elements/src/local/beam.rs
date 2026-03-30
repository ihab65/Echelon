//! Euler-Bernoulli beam kinematics — local stiffness and internal forces.
//!
//! DOF order (local frame): `[u1, v1, θ1, u2, v2, θ2]`
//!
//! The local frame has its x-axis along the element, so u is axial
//! and v is transverse.  The global-to-local transformation is handled
//! by `fem_core::CoordTransf2d`.
//!
//! ## Stiffness sub-blocks
//!
//! Axial (DOFs 0, 3):
//! ```text
//! EA/L * [ 1 -1; -1 1]
//! ```
//! Bending (DOFs 1, 2, 4, 5) — Euler-Bernoulli:
//! ```text
//! EI/L³ * [ 12   6L  -12   6L ]
//!          [  6L  4L²  -6L  2L²]
//!          [-12  -6L   12  -6L ]
//!          [  6L  2L²  -6L  4L²]
//! ```

/// Compute the 6×6 local (element-frame) stiffness matrix.
///
/// Returns a flat row-major `[f64; 36]`.
///
/// # Arguments
/// * `e`  — Young's modulus (Pa)
/// * `a`  — cross-section area (m²)
/// * `iz` — second moment of area about z (m⁴)
/// * `l`  — element length (m)
pub fn ke_local(e: f64, a: f64, iz: f64, l: f64) -> [f64; 36] {
    let eal  = e * a / l;
    let l2   = l * l;
    let l3   = l2 * l;
    let ei   = e * iz;
    let b1   = 12.0 * ei / l3;
    let b2   =  6.0 * ei / l2;
    let b3   =  4.0 * ei / l;
    let b4   =  2.0 * ei / l;

    // Row-major 6×6, DOF order: [u1, v1, θ1, u2, v2, θ2]
    [
        // row 0 (u1)
         eal,  0.0,  0.0, -eal,  0.0,  0.0,
        // row 1 (v1)
         0.0,  b1,   b2,   0.0, -b1,   b2,
        // row 2 (θ1)
         0.0,  b2,   b3,   0.0, -b2,   b4,
        // row 3 (u2)
        -eal,  0.0,  0.0,  eal,  0.0,  0.0,
        // row 4 (v2)
         0.0, -b1,  -b2,   0.0,  b1,  -b2,
        // row 5 (θ2)
         0.0,  b2,   b4,   0.0, -b2,   b3,
    ]
}

/// Internal force vector in local coordinates: `f_int_local = Ke_local * u_local`.
///
/// # Arguments
/// * `ke` — local stiffness from [`ke_local`], flat row-major `[f64; 36]`
/// * `u`  — local displacements `[u1, v1, θ1, u2, v2, θ2]`
pub fn f_int_local_from_ke(ke: &[f64; 36], u: &[f64]) -> [f64; 6] {
    debug_assert_eq!(u.len(), 6);
    let mut f = [0.0_f64; 6];
    for i in 0..6 {
        for j in 0..6 {
            f[i] += ke[i * 6 + j] * u[j];
        }
    }
    f
}

/// Scalar axial strain at the centroid (for adjoint / stress recovery).
#[inline]
pub fn axial_strain(u: &[f64], l: f64) -> f64 {
    // ε = (u2 - u1) / L  (axial DOFs are 0 and 3 in local frame)
    (u[3] - u[0]) / l
}

/// Mid-span curvature κ = d²v/dx² at x = L/2 (Euler-Bernoulli).
///
/// Useful for damage and ductility assessment post-analysis.
///
/// At the midpoint x = L/2, the Hermite shape function second derivatives are:
/// ```text
///   N1''(L/2) = 0        (transverse displacement node 1)
///   N2''(L/2) = -1/L     (rotation node 1)
///   N3''(L/2) = 0        (transverse displacement node 2)
///   N4''(L/2) = +1/L     (rotation node 2)
/// ```
/// So κ(L/2) = (-θ1 + θ2) / L.
pub fn mid_curvature(u: &[f64], l: f64) -> f64 {
    let t1 = u[2]; // rotation at node 1
    let t2 = u[5]; // rotation at node 2
    (t2 - t1) / l
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: a={a:.6e} b={b:.6e} diff={:.2e}", (a-b).abs());
    }

    #[test]
    fn ke_local_symmetric() {
        let ke = ke_local(200e9, 0.01, 1e-4, 2.0);
        for i in 0..6 {
            for j in 0..6 {
                approx_eq(ke[i*6+j], ke[j*6+i], 1e-6, &format!("ke[{i},{j}]"));
            }
        }
    }

    #[test]
    fn ke_local_axial_entries() {
        let e = 200e9_f64; let a = 0.01_f64; let l = 2.0_f64;
        let ke = ke_local(e, a, 1e-4, l);
        let eal = e * a / l;
        approx_eq(ke[0],  eal, 1e-3, "ke[0,0]");
        approx_eq(ke[3], -eal, 1e-3, "ke[0,3]");
        approx_eq(ke[18], -eal, 1e-3, "ke[3,0]");
        approx_eq(ke[21],  eal, 1e-3, "ke[3,3]");
    }

    #[test]
    fn ke_local_bending_diagonal() {
        let e = 200e9_f64; let iz = 1e-4_f64; let l = 2.0_f64;
        let ke = ke_local(e, 0.01, iz, l);
        let ei = e * iz;
        let b1 = 12.0 * ei / (l * l * l);
        let b3 = 4.0 * ei / l;
        approx_eq(ke[7],  b1, 1e-3, "ke[1,1]");
        approx_eq(ke[14], b3, 1e-3, "ke[2,2]");
    }

    #[test]
    fn f_int_local_rigid_body_axial_translation_zero() {
        // Rigid body axial translation: u1 = u2 = Δ → ε = 0 → f_int = 0
        let ke = ke_local(200e9, 0.01, 1e-4, 2.0);
        let delta = 0.5_f64;
        let u = [delta, 0.0, 0.0, delta, 0.0, 0.0]; // both nodes move equally in x
        let f = f_int_local_from_ke(&ke, &u);
        for (i, &fi) in f.iter().enumerate() {
            assert!(fi.abs() < 1e-6, "f_int[{i}]={fi:.3e} — should be ≈0 for rigid translation");
        }
    }

    #[test]
    fn f_int_local_consistent_with_ke() {
        let ke = ke_local(200e9, 0.01, 1e-4, 2.0);
        let u = [0.0, 0.0, 0.0, 0.001, 0.0, 0.0]; // pure axial elongation
        let f = f_int_local_from_ke(&ke, &u);
        let eal = 200e9 * 0.01 / 2.0;
        approx_eq(f[0], -eal * 0.001, 1e-3, "f_int[0]");
        approx_eq(f[3],  eal * 0.001, 1e-3, "f_int[3]");
    }

    #[test]
    fn axial_strain_elongation() {
        let u = [0.0_f64, 0.0, 0.0, 0.002, 0.0, 0.0]; // u2 = 0.002, L=1
        approx_eq(axial_strain(&u, 1.0), 0.002, 1e-15, "axial strain");
    }
}