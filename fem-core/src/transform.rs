//! Coordinate transforms for 2D structural elements.
//!
//! [`CoordTransf2d`] holds the geometric properties of a single element
//! and provides:
//! - The 6×6 rotation matrix `T` that maps global DOFs to local DOFs.
//! - The full transformed stiffness `Kg = Tᵀ Ke_local T` via
//!   [`CoordTransf2d::transform_stiffness`].
//!
//! ## Convention
//!
//! For a 2D frame element connecting nodes `(x1,y1)` and `(x2,y2)`:
//!
//! - Local x-axis points from node 1 to node 2.
//! - Local y-axis is 90° counter-clockwise from local x.
//! - DOF order (local): `[u1_x, u1_y, θ1, u2_x, u2_y, θ2]`.
//! - DOF order (global): same labelling but in the global x-y frame.
//!
//! The rotation matrix relating local to global displacement components is:
//! ```text
//!   [ c  s  0 ]
//!   [-s  c  0 ]
//!   [ 0  0  1 ]
//! ```
//! where `c = cos θ = (x2-x1)/L` and `s = sin θ = (y2-y1)/L`.
//!
//! The full 6×6 `T` matrix is block-diagonal with two copies of this 3×3:
//! ```text
//! T = diag( R, R )   (6×6)
//! ```
//!
//! ## For 2D truss elements (4×4 T)
//!
//! Use [`CoordTransf2d::t_matrix_4x4`] which produces the 4×4 version
//! for elements with 2 DOFs/node (no rotation DOF).

use sparse::SparseScalar;

use crate::{dense::transform_stiffness as dense_transform};
use crate::error::{Result, CoreError};

/// Geometric properties of a 2D frame or truss element.
///
/// Computed once at construction from the node coordinates and reused
/// throughout the analysis (the geometry does not change for linear analysis).
///
/// For nonlinear / corotational analysis, this struct would be extended with
/// methods to update `cos` and `sin` from the current deformed configuration.
/// That logic belongs in the `elements` crate, not here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordTransf2d<T: SparseScalar> {
    /// `cos θ = (x2 - x1) / L`
    pub cos: T,
    /// `sin θ = (y2 - y1) / L`
    pub sin: T,
    /// Element length `L = sqrt((x2-x1)² + (y2-y1)²)`
    pub length: T,
}

impl<T: SparseScalar> CoordTransf2d<T> {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Compute from node coordinates.
    ///
    /// # Errors
    /// - [`CoreError::DegenerateGeometry`] if the element has length zero (nodes are coincident).
    pub fn from_nodes(x1: T, y1: T, x2: T, y2: T) -> Result<Self> {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let length = (dx * dx + dy * dy).scalar_sqrt();
        
        if length.real_part() <= 0.0 {
            return Err(CoreError::DegenerateGeometry {
                x1: x1.real_part(),
                y1: y1.real_part(),
                z1_str: String::new(),
                x2: x2.real_part(),
                y2: y2.real_part(),
                z2_str: String::new(),
                length: length.real_part(),
            });
        }

        Ok(Self {
            cos: dx / length,
            sin: dy / length,
            length,
        })
    }

    /// Construct directly from known `cos`, `sin`, and `length`.
    ///
    /// Useful in tests or when the geometric properties are already computed.
    /// The caller is responsible for ensuring `cos² + sin² ≈ 1`.
    ///
    /// # Errors
    /// - [`CoreError::NonOrthogonalTransform`] if `cos² + sin²` deviates from 1.0 by more than `1e-12`.
    pub fn from_cos_sin_length(cos: T, sin: T, length: T) -> Result<Self> {
        let norm_sq = cos * cos + sin * sin;
        let deviation = (norm_sq - T::one()).abs();
        if deviation.real_part() > 1e-12 {
            return Err(CoreError::NonOrthogonalTransform {
                norm_sq: norm_sq.real_part(),
                deviation: deviation.real_part(),
            });
        }
        Ok(Self { cos, sin, length })
    }

    // -----------------------------------------------------------------
    // The 3×3 rotation block
    // -----------------------------------------------------------------

    /// The 3×3 rotation matrix for one end of a 2D frame element:
    ///
    /// ```text
    /// R = [ c   s   0 ]
    ///     [-s   c   0 ]
    ///     [ 0   0   1 ]
    /// ```
    ///
    /// Maps global `[ux, uy, θ]` to local `[u_L, v_L, θ_L]`.
    #[inline]
    pub fn rotation_3x3(&self) -> [[T; 3]; 3] {
        let c = self.cos;
        let s = self.sin;
        let o = T::zero();
        let i = T::one();
        [
            [ c,  s, o],
            [-s,  c, o],
            [o,  o, i],
        ]
    }

    // -----------------------------------------------------------------
    // 6×6 transform for 2D frame elements (3 DOF/node)
    // -----------------------------------------------------------------

    /// The 6×6 global→local transform matrix `T` for a 2D frame element.
    ///
    /// ```text
    /// T = [ R  0 ]   where R = 3×3 rotation block
    ///     [ 0  R ]
    /// ```
    ///
    /// Use this with [`transform_stiffness`](CoordTransf2d::transform_stiffness)
    /// to rotate `Ke_local` into global coordinates.
    pub fn t_matrix_6x6(&self) -> [[T; 6]; 6] {
        let c = self.cos;
        let s = self.sin;
        let o = T::zero();
        let i = T::one();

        [
            [ c,  s, o, o, o, o],
            [-s,  c, o, o, o, o],
            [ o,  o, i, o, o, o],
            [ o,  o, o, c, s, o],
            [ o,  o, o,-s, c, o],
            [ o,  o, o, o, o, i],
        ]
    }

    /// Transform a 6×6 local stiffness to global coordinates.
    ///
    /// Computes `Kg = Tᵀ Ke_local T` in two matrix multiplications.
    ///
    /// # Example
    ///
    /// ```
    /// use fem_core::transform::CoordTransf2d;
    ///
    /// let t = CoordTransf2d::from_nodes(0.0, 0.0, 1.0, 0.0); // horizontal
    /// let ke_local = [[0.0_f64; 6]; 6]; // zero stiffness for illustration
    /// let ke_global = t.unwrap().transform_stiffness_6x6(&ke_local);
    /// assert_eq!(ke_global, [[0.0; 6]; 6]);
    /// ```
    pub fn transform_stiffness_6x6(&self, ke_local: &[[T; 6]; 6]) -> [[T; 6]; 6] {
        let t = self.t_matrix_6x6();
        dense_transform(ke_local, &t)
    }

    // -----------------------------------------------------------------
    // 4×4 transform for 2D truss elements (2 DOF/node)
    // -----------------------------------------------------------------

    /// The 4×4 global→local transform matrix `T` for a 2D truss element.
    ///
    /// ```text
    /// T = [ c   s   0   0 ]
    ///     [-s   c   0   0 ]
    ///     [ 0   0   c   s ]
    ///     [ 0   0  -s   c ]
    /// ```
    pub fn t_matrix_4x4(&self) -> [[T; 4]; 4] {
        let c = self.cos;
        let s = self.sin;
        let o = T::zero();
        [
            [ c,  s, o, o],
            [-s,  c, o, o],
            [o, o,  c,  s],
            [o, o, -s,  c],
        ]
    }

    /// Transform a 4×4 local stiffness to global coordinates.
    ///
    /// Computes `Kg = Tᵀ Ke_local T`.
    pub fn transform_stiffness_4x4(&self, ke_local: &[[T; 4]; 4]) -> [[T; 4]; 4] {
        let t = self.t_matrix_4x4();
        dense_transform(ke_local, &t)
    }

    // -----------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------

    /// Element inclination angle in radians, in `[-π, π]`.
    #[inline]
    pub fn angle_rad(&self) -> f64 {
        self.sin.real_part().atan2(self.cos.real_part())
    }

    /// Element inclination angle in degrees.
    #[inline]
    pub fn angle_deg(&self) -> f64 {
        self.angle_rad().to_degrees()
    }

    /// Returns a new `CoordTransf2d` for the reversed element
    /// (node 2 → node 1).  Flips the sign of `sin` and `cos`.
    #[inline]
    pub fn reversed(&self) -> Self {
        Self { cos: -self.cos, sin: -self.sin, length: self.length }
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::{matmul, transpose, mat_zero};
    use std::f64::consts::{FRAC_1_SQRT_2};

    fn approx_eq_6x6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6], tol: f64) {
        for i in 0..6 {
            for j in 0..6 {
                let d = (a[i][j] - b[i][j]).abs();
                assert!(d <= tol, "a[{i}][{j}]={} b[{i}][{j}]={}  diff={d:.2e}", a[i][j], b[i][j]);
            }
        }
    }

    fn approx_eq_4x4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], tol: f64) {
        for i in 0..4 {
            for j in 0..4 {
                let d = (a[i][j] - b[i][j]).abs();
                assert!(d <= tol, "a[{i}][{j}]={} b[{i}][{j}]={}  diff={d:.2e}", a[i][j], b[i][j]);
            }
        }
    }

    // ---- from_nodes ----

    #[test]
    fn horizontal_element() {
        let t = CoordTransf2d::from_nodes(0.0, 0.0, 3.0, 0.0).unwrap();
        assert!((t.cos - 1.0).real_part().abs() < 1e-14);
        assert!((t.sin - 0.0).real_part().abs() < 1e-14);
        assert!((t.length - 3.0).real_part().abs() < 1e-14);
    }

    #[test]
    fn vertical_element() {
        let t = CoordTransf2d::from_nodes(0.0, 0.0, 0.0, 4.0).unwrap();
        assert!((t.cos - 0.0).real_part().abs() < 1e-14);
        assert!((t.sin - 1.0).real_part().abs() < 1e-14);
        assert!((t.length - 4.0).real_part().abs() < 1e-14);
    }

    #[test]
    fn diagonal_45_element() {
        let t = CoordTransf2d::from_nodes(0.0, 0.0, 1.0, 1.0).unwrap();
        assert!((t.cos - FRAC_1_SQRT_2).abs() < 1e-14);
        assert!((t.sin - FRAC_1_SQRT_2).abs() < 1e-14);
        assert!((t.length - 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn angle_horizontal() {
        let t = CoordTransf2d::from_nodes(0.0, 0.0, 5.0, 0.0).unwrap();
        assert!((t.angle_rad() - 0.0).real_part().abs() < 1e-14);
        assert!((t.angle_deg() - 0.0).real_part().abs() < 1e-14);
    }

    #[test]
    fn angle_45_degrees() {
        let t = CoordTransf2d::from_nodes(0.0, 0.0, 1.0, 1.0).unwrap();
        assert!((t.angle_deg() - 45.0).real_part().abs() < 1e-12);
    }

    #[test]
    fn reversed_flips_direction() {
        let t = CoordTransf2d::from_nodes(0.0, 0.0, 3.0, 4.0).unwrap();
        let r = t.reversed();
        assert!((r.cos + t.cos).real_part().abs() < 1e-14);
        assert!((r.sin + t.sin).real_part().abs() < 1e-14);
        assert!((r.length - t.length).real_part().abs() < 1e-14);
    }

    // ---- 6×6 T matrix properties ----

    #[test]
    fn t6x6_is_orthogonal() {
        // T * Tᵀ = I  (T is orthogonal → Tᵀ = T⁻¹)
        let t_mat = CoordTransf2d::from_nodes(0.0, 0.0, 3.0, 4.0).unwrap().t_matrix_6x6();
        let tt = transpose(&t_mat);
        let product = matmul(&t_mat, &tt);
        let eye: [[f64; 6]; 6] = {
            let mut m = mat_zero::<6, f64>();
            for i in 0..6 { m[i][i] = 1.0; }
            m
        };
        approx_eq_6x6(&product, &eye, 1e-13);
    }

    #[test]
    fn t6x6_horizontal_is_identity() {
        let t_mat = CoordTransf2d::from_nodes(0.0, 0.0, 1.0, 0.0).unwrap().t_matrix_6x6();
        let eye: [[f64; 6]; 6] = {
            let mut m = mat_zero::<6, f64>();
            for i in 0..6 { m[i][i] = 1.0; }
            m
        };
        approx_eq_6x6(&t_mat, &eye, 1e-14);
    }

    // ---- 4×4 T matrix properties ----

    #[test]
    fn t4x4_is_orthogonal() {
        let t_mat = CoordTransf2d::from_nodes(0.0, 0.0, 3.0, 4.0).unwrap().t_matrix_4x4();
        let tt = transpose(&t_mat);
        let product = matmul(&t_mat, &tt);
        let eye: [[f64; 4]; 4] = {
            let mut m = mat_zero::<4, f64>();
            for i in 0..4 { m[i][i] = 1.0; }
            m
        };
        approx_eq_4x4(&product, &eye, 1e-13);
    }

    // ---- transform_stiffness_6x6 horizontal == local ----

    #[test]
    fn transform_6x6_horizontal_unchanged() {
        // For a horizontal element, T = I and Tᵀ K T = K
        let transf = CoordTransf2d::from_nodes(0.0, 0.0, 1.0, 0.0).unwrap();
        // Simple symmetric 6×6 with non-trivial values
        let mut ke: [[f64; 6]; 6] = mat_zero();
        ke[0][0] =  1.0; ke[0][3] = -1.0;
        ke[3][0] = -1.0; ke[3][3] =  1.0;
        ke[1][1] =  0.0; ke[4][4] =  0.0;
        ke[2][2] =  0.0; ke[5][5] =  0.0;
        let kg = transf.transform_stiffness_6x6(&ke);
        approx_eq_6x6(&kg, &ke, 1e-14);
    }

    // ---- transform_stiffness_4x4 vertical ----

    #[test]
    fn transform_4x4_vertical_truss() {
        // Vertical element: stiffness in y-direction after transform
        let transf = CoordTransf2d::from_nodes(0.0, 0.0, 0.0, 1.0).unwrap();
        let ke_local: [[f64; 4]; 4] = [
            [ 1.0, 0.0, -1.0, 0.0],
            [ 0.0, 0.0,  0.0, 0.0],
            [-1.0, 0.0,  1.0, 0.0],
            [ 0.0, 0.0,  0.0, 0.0],
        ];
        let kg = transf.transform_stiffness_4x4(&ke_local);
        // Stiffness should be in y-direction now
        assert!((kg[1][1] -  1.0).abs() < 1e-13, "kg[1][1]={}", kg[1][1]);
        assert!((kg[1][3] - -1.0).abs() < 1e-13, "kg[1][3]={}", kg[1][3]);
        assert!((kg[3][1] - -1.0).abs() < 1e-13, "kg[3][1]={}", kg[3][1]);
        assert!((kg[3][3] -  1.0).abs() < 1e-13, "kg[3][3]={}", kg[3][3]);
        // x-direction: zero stiffness
        assert!((kg[0][0]).abs() < 1e-13, "kg[0][0]={}", kg[0][0]);
    }

    // ---- rotation_3x3 block ----

    #[test]
    fn rotation_3x3_preserves_rotation_dof() {
        // The θ DOF (index 2) should be unchanged by the rotation
        let r = CoordTransf2d::from_nodes(0.0, 0.0, 3.0, 4.0).unwrap().rotation_3x3();
        assert!((r[2][2] - 1.0).real_part().abs() < 1e-14);
        assert!((r[0][2]).real_part().abs() < 1e-14);
        assert!((r[1][2]).real_part().abs() < 1e-14);
        assert!((r[2][0]).real_part().abs() < 1e-14);
        assert!((r[2][1]).real_part().abs() < 1e-14);
    }

    #[test]
    fn rotation_3x3_is_orthogonal() {
        let r = CoordTransf2d::from_nodes(1.0, 0.0, 4.0, 4.0).unwrap().rotation_3x3();
        let rt = transpose(&r);
        let rrt = matmul(&r, &rt);
        let eye: [[f64; 3]; 3] = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((rrt[i][j] - eye[i][j]).abs() < 1e-13);
            }
        }
    }
}