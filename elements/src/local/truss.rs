//! Axial kinematics for 2D truss elements.
//!
//! A 2D truss element has 4 local DOFs: `[u1_x, u1_y, u2_x, u2_y]` in
//! global coordinates.  All functions here are generic over `T` so they
//! are callable from both the f64 Newton-Raphson path and the dual-number
//! `energy<T>` path.
//!
//! ## Coordinate convention
//!
//! - Local x-axis points from node 1 to node 2.
//! - `c = cos θ`, `s = sin θ` where `θ` is the element inclination.
//! - Axial displacement: `u_L = c * u_x + s * u_y` (scalar projection).
//! - Axial strain: `ε = (u2_L - u1_L) / L`.

use std::ops::{Add, Div, Mul, Sub};
use num_traits::{One, Zero};

/// Project global DOF pair `(u_x, u_y)` onto the element's local axis.
///
/// `u_local = c * u_x + s * u_y`
#[inline]
pub fn project_axial<T>(u_x: T, u_y: T, cos: T, sin: T) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    cos * u_x + sin * u_y
}

/// Compute the axial strain in a 2D truss element.
///
/// # Arguments
/// * `u`   — slice `[u1_x, u1_y, u2_x, u2_y]` of length 4
/// * `cos` — element cos θ (pre-computed from geometry)
/// * `sin` — element sin θ
/// * `l`   — element length
///
/// # Returns
/// Axial strain `ε = (u2_L - u1_L) / L`
#[inline]
pub fn axial_strain<T>(u: &[T], cos: T, sin: T, l: T) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + Zero + One,
{
    debug_assert_eq!(u.len(), 4, "truss DOF vector must have length 4");
    let u1_l = project_axial(u[0], u[1], cos, sin);
    let u2_l = project_axial(u[2], u[3], cos, sin);
    (u2_l - u1_l) / l
}

/// Closed-form 4×4 truss stiffness in global coordinates.
///
/// `Kg = (EA/L) * [[c² cs -c² -cs], [cs s² -cs -s²], [-c² -cs c² cs], [-cs -s² cs s²]]`
///
/// This is the analytic result of `Tᵀ Ke_local T` expanded.  It is faster
/// than calling `transform_stiffness_4x4` from `fem_core` because it avoids
/// two 4×4 matrix multiplications.
///
/// # Returns
/// Flat row-major `[f64; 16]` — pass to `scatter_add`.
pub fn stiffness_global(ea_over_l: f64, cos: f64, sin: f64) -> [f64; 16] {
    let c2 = cos * cos;
    let s2 = sin * sin;
    let cs = cos * sin;
    let k  = ea_over_l;

    [
         k*c2,  k*cs, -k*c2, -k*cs,
         k*cs,  k*s2, -k*cs, -k*s2,
        -k*c2, -k*cs,  k*c2,  k*cs,
        -k*cs, -k*s2,  k*cs,  k*s2,
    ]
}

/// Internal force vector for a 2D truss in global coordinates.
///
/// `f = (EA/L) * axial_strain * [-c, -s, c, s]`
pub fn f_int_global(ea_over_l: f64, cos: f64, sin: f64, strain: f64) -> [f64; 4] {
    let force = ea_over_l * strain;
    [-force * cos, -force * sin, force * cos, force * sin]
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_strain_horizontal_elongation() {
        // Horizontal element, node 2 moves +delta in x
        let delta = 0.01_f64;
        let l = 1.0_f64;
        let u = [0.0_f64, 0.0, delta, 0.0];
        let eps = axial_strain(&u, 1.0, 0.0, l);
        assert!((eps - delta / l).abs() < 1e-15);
    }

    #[test]
    fn axial_strain_vertical_elongation() {
        // Vertical element (θ=90°), node 2 moves +delta in y
        let delta = 0.02_f64;
        let l = 2.0_f64;
        let u = [0.0_f64, 0.0, 0.0, delta];
        let eps = axial_strain(&u, 0.0, 1.0, l);
        assert!((eps - delta / l).abs() < 1e-15);
    }

    #[test]
    fn axial_strain_zero_for_rigid_body_translation() {
        // Rigid translation: both nodes move the same amount
        let u = [0.1_f64, 0.2, 0.1, 0.2];
        let l = 1.5_f64;
        let eps = axial_strain(&u, 1.0, 0.0, l);
        assert!(eps.abs() < 1e-15);
    }

    #[test]
    fn stiffness_horizontal_correct() {
        let k = stiffness_global(1.0, 1.0, 0.0); // EA/L=1, horizontal
        // Should be [ 1 0 -1 0 / 0 0 0 0 / -1 0 1 0 / 0 0 0 0 ]
        assert!((k[0] -  1.0).abs() < 1e-15);
        assert!((k[2] - -1.0).abs() < 1e-15);
        assert!((k[8] - -1.0).abs() < 1e-15);
        assert!((k[10] - 1.0).abs() < 1e-15);
        assert!(k[1].abs() < 1e-15);
    }

    #[test]
    fn stiffness_vertical_correct() {
        let k = stiffness_global(1.0, 0.0, 1.0); // vertical
        // Stiffness in y-direction only
        assert!((k[5]  -  1.0).abs() < 1e-15); // [1][1]
        assert!((k[7]  - -1.0).abs() < 1e-15); // [1][3]
        assert!((k[13] - -1.0).abs() < 1e-15); // [3][1]
        assert!((k[15] -  1.0).abs() < 1e-15); // [3][3]
        assert!(k[0].abs() < 1e-15);
    }

    #[test]
    fn stiffness_symmetric() {
        let k = stiffness_global(500.0, 0.6, 0.8);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (k[i * 4 + j] - k[j * 4 + i]).abs() < 1e-12,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn f_int_horizontal_tension() {
        // EA/L=1, horizontal, strain=0.01 → force=0.01 in x direction
        let f = f_int_global(1.0, 1.0, 0.0, 0.01);
        assert!((f[0] - -0.01).abs() < 1e-15); // node 1 pulled left
        assert!((f[2] -  0.01).abs() < 1e-15); // node 2 pulled right
        assert!(f[1].abs() < 1e-15);
        assert!(f[3].abs() < 1e-15);
    }
}