//! Coordinate transforms for 3D structural elements.
//!
//! [`CoordTransf3d`] holds the geometric properties of a single 3D element
//! and provides:
//! - The 3×3 rotation matrix `R` that maps global DOFs to local DOFs.
//! - The full 12×12 transformed stiffness `Kg = Tᵀ Ke_local T` for
//!   3D frame elements (6 DOF/node).
//! - The 6×6 version for 3D truss elements (3 DOF/node).
//!
//! ## Convention
//!
//! For a 3D element connecting nodes `(x1,y1,z1)` and `(x2,y2,z2)`:
//!
//! - **Local x-axis** (`e_x`): unit vector from node 1 to node 2.
//! - **Local z-axis** (`e_z`): `e_x × v_ref`, normalised.
//! - **Local y-axis** (`e_y`): `e_z × e_x` (right-hand rule).
//!
//! The 3×3 rotation matrix maps global → local:
//! ```text
//!   R = [ e_x ]     ← each row is a local axis in global coordinates
//!       [ e_y ]
//!       [ e_z ]
//! ```
//!
//! ## Reference vector
//!
//! The reference vector `v_ref` defines the plane containing the local
//! x- and y-axes. The default is `(0, 1, 0)` (global Y-up). When the
//! element is nearly parallel to global Y (`|e_x · v_ref| > 1 − ε`),
//! `from_nodes` automatically falls back to `(0, 0, 1)`.
//!
//! Use [`CoordTransf3d::from_nodes_with_ref`] for explicit control.
//!
//! ## DOF ordering
//!
//! **3D frame** (6 DOF/node → 12 total):
//! `[u1_x, u1_y, u1_z, θ1_x, θ1_y, θ1_z, u2_x, u2_y, u2_z, θ2_x, θ2_y, θ2_z]`
//!
//! **3D truss** (3 DOF/node → 6 total):
//! `[u1_x, u1_y, u1_z, u2_x, u2_y, u2_z]`

use sparse::SparseScalar;

use crate::dense::transform_stiffness as dense_transform;
use crate::error::{Result, CoreError};

// -----------------------------------------------------------------
// CoordTransf3d
// -----------------------------------------------------------------

/// Geometric properties of a 3D frame or truss element.
///
/// Stores the 3×3 rotation matrix `R` and the element length.
/// Computed once at construction from the node coordinates and a reference
/// vector, then reused throughout the analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordTransf3d<T: SparseScalar> {
    /// 3×3 rotation matrix: `R[i][j]` = component `j` (global) of local axis `i`.
    ///
    /// - Row 0: local x-axis direction (element axis).
    /// - Row 1: local y-axis direction.
    /// - Row 2: local z-axis direction.
    pub rotation: [[T; 3]; 3],
    /// Element length `L = ‖P2 − P1‖`.
    pub length: T,
}

impl<T: SparseScalar> CoordTransf3d<T> {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Compute from node coordinates using the default reference vector.
    ///
    /// Uses `v_ref = (0, 1, 0)` (global Y-up). If the element is nearly
    /// parallel to global Y, automatically falls back to `v_ref = (0, 0, 1)`.
    ///
    /// # Errors
    /// - [`CoreError::DegenerateGeometry`] if the element has zero length.
    pub fn from_nodes(
        x1: T, y1: T, z1: T,
        x2: T, y2: T, z2: T,
    ) -> Result<Self> {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dz = z2 - z1;
        let length = (dx * dx + dy * dy + dz * dz).scalar_sqrt();

        if length.real_part() <= 0.0 {
            return Err(CoreError::DegenerateGeometry {
                x1: x1.real_part(),
                y1: y1.real_part(),
                z1_str: format!(", {:.6e}", z1.real_part()),
                x2: x2.real_part(),
                y2: y2.real_part(),
                z2_str: format!(", {:.6e}", z2.real_part()),
                length: length.real_part(),
            });
        }

        let ex = [dx / length, dy / length, dz / length];

        // Default reference vector: global Y
        let v_ref_y = [T::zero(), T::one(), T::zero()];
        // Fallback reference vector: global Z
        let v_ref_z = [T::zero(), T::zero(), T::one()];

        // Check if element is nearly parallel to global Y
        let dot_y = dot3(&ex, &v_ref_y);
        let v_ref = if (T::one() - dot_y.abs()).real_part() < 1e-6 {
            v_ref_z
        } else {
            v_ref_y
        };

        let rotation = build_rotation(&ex, &v_ref)?;
        Ok(Self { rotation, length })
    }

    /// Compute from node coordinates with an explicit reference vector.
    ///
    /// The reference vector `(vx, vy, vz)` defines the plane containing the
    /// local x- and y-axes. It must NOT be parallel to the element axis.
    ///
    /// # Errors
    /// - [`CoreError::DegenerateGeometry`] if the element has zero length.
    /// - [`CoreError::ParallelReferenceVector`] if `v_ref` is parallel to the element axis.
    pub fn from_nodes_with_ref(
        x1: T, y1: T, z1: T,
        x2: T, y2: T, z2: T,
        vx: T, vy: T, vz: T,
    ) -> Result<Self> {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dz = z2 - z1;
        let length = (dx * dx + dy * dy + dz * dz).scalar_sqrt();

        if length.real_part() <= 0.0 {
            return Err(CoreError::DegenerateGeometry {
                x1: x1.real_part(),
                y1: y1.real_part(),
                z1_str: format!(", {:.6e}", z1.real_part()),
                x2: x2.real_part(),
                y2: y2.real_part(),
                z2_str: format!(", {:.6e}", z2.real_part()),
                length: length.real_part(),
            });
        }

        let ex = [dx / length, dy / length, dz / length];
        let v_ref = [vx, vy, vz];

        let rotation = build_rotation(&ex, &v_ref)?;
        Ok(Self { rotation, length })
    }

    /// Construct directly from a known rotation matrix and length.
    ///
    /// The caller is responsible for ensuring `R` is orthogonal (`R Rᵀ = I`).
    ///
    /// # Errors
    /// - [`CoreError::NonOrthogonalTransform`] if any row of `R` deviates
    ///   from unit norm by more than `1e-12`.
    pub fn from_rotation_length(rotation: [[T; 3]; 3], length: T) -> Result<Self> {
        // Validate each row is unit-length
        for (i, row) in rotation.iter().enumerate() {
            let norm_sq = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
            let deviation = (norm_sq - T::one()).abs();
            if deviation.real_part() > 1e-12 {
                return Err(CoreError::NonOrthogonalTransform {
                    norm_sq: norm_sq.real_part(),
                    deviation: deviation.real_part(),
                });
            }
            let _ = i;
        }
        Ok(Self { rotation, length })
    }

    // -----------------------------------------------------------------
    // Rotation matrix access
    // -----------------------------------------------------------------

    /// The 3×3 rotation matrix `R` mapping global → local.
    #[inline]
    pub fn rotation_3x3(&self) -> [[T; 3]; 3] {
        self.rotation
    }

    /// Local x-axis direction (element axis) in global coordinates.
    #[inline]
    pub fn local_x_axis(&self) -> [T; 3] {
        self.rotation[0]
    }

    /// Local y-axis direction in global coordinates.
    #[inline]
    pub fn local_y_axis(&self) -> [T; 3] {
        self.rotation[1]
    }

    /// Local z-axis direction in global coordinates.
    #[inline]
    pub fn local_z_axis(&self) -> [T; 3] {
        self.rotation[2]
    }

    // -----------------------------------------------------------------
    // 12×12 transform for 3D frame elements (6 DOF/node)
    // -----------------------------------------------------------------

    /// The 12×12 global→local transform matrix `T` for a 3D frame element.
    ///
    /// ```text
    /// T = diag(R, R, R, R)     (four 3×3 blocks)
    /// ```
    ///
    /// DOF order: `[u1_x, u1_y, u1_z, θ1_x, θ1_y, θ1_z,
    ///              u2_x, u2_y, u2_z, θ2_x, θ2_y, θ2_z]`
    pub fn t_matrix_12x12(&self) -> [[T; 12]; 12] {
        let r = &self.rotation;
        let o = T::zero();
        let mut t = [[o; 12]; 12];

        // Place R in four 3×3 blocks along the diagonal
        for block in 0..4 {
            let off = block * 3;
            for i in 0..3 {
                for j in 0..3 {
                    t[off + i][off + j] = r[i][j];
                }
            }
        }
        t
    }

    /// Transform a 12×12 local stiffness to global coordinates.
    ///
    /// Computes `Kg = Tᵀ Ke_local T`.
    pub fn transform_stiffness_12x12(&self, ke_local: &[[T; 12]; 12]) -> [[T; 12]; 12] {
        let t = self.t_matrix_12x12();
        dense_transform(ke_local, &t)
    }

    // -----------------------------------------------------------------
    // 6×6 transform for 3D truss elements (3 DOF/node)
    // -----------------------------------------------------------------

    /// The 6×6 global→local transform matrix `T` for a 3D truss element.
    ///
    /// ```text
    /// T = diag(R, R)     (two 3×3 blocks)
    /// ```
    ///
    /// DOF order: `[u1_x, u1_y, u1_z, u2_x, u2_y, u2_z]`
    pub fn t_matrix_6x6(&self) -> [[T; 6]; 6] {
        let r = &self.rotation;
        let o = T::zero();
        let mut t = [[o; 6]; 6];

        // Two 3×3 blocks on the diagonal
        for block in 0..2 {
            let off = block * 3;
            for i in 0..3 {
                for j in 0..3 {
                    t[off + i][off + j] = r[i][j];
                }
            }
        }
        t
    }

    /// Transform a 6×6 local stiffness to global coordinates.
    ///
    /// Computes `Kg = Tᵀ Ke_local T`.
    pub fn transform_stiffness_6x6(&self, ke_local: &[[T; 6]; 6]) -> [[T; 6]; 6] {
        let t = self.t_matrix_6x6();
        dense_transform(ke_local, &t)
    }

    // -----------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------

    /// Returns a new `CoordTransf3d` for the reversed element
    /// (node 2 → node 1). Negates the x-axis, preserves z-axis
    /// handedness by also negating y-axis.
    pub fn reversed(&self) -> Self {
        let r = &self.rotation;
        Self {
            rotation: [
                [-r[0][0], -r[0][1], -r[0][2]], // negate e_x
                [-r[1][0], -r[1][1], -r[1][2]], // negate e_y to keep RH
                [ r[2][0],  r[2][1],  r[2][2]], // e_z unchanged
            ],
            length: self.length,
        }
    }
}

// -----------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------

/// Dot product of two 3-vectors.
#[inline]
fn dot3<T: SparseScalar>(a: &[T; 3], b: &[T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors.
#[inline]
fn cross3<T: SparseScalar>(a: &[T; 3], b: &[T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean norm of a 3-vector.
#[inline]
fn norm3<T: SparseScalar>(v: &[T; 3]) -> T {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).scalar_sqrt()
}

/// Build the 3×3 rotation matrix from a unit element axis `e_x` and a
/// reference vector `v_ref`.
///
/// ```text
/// e_z = normalise(e_x × v_ref)
/// e_y = e_z × e_x
/// R = [e_x; e_y; e_z]
/// ```
///
/// # Errors
/// - [`CoreError::ParallelReferenceVector`] if `‖e_x × v_ref‖ < 1e-10`.
fn build_rotation<T: SparseScalar>(
    ex: &[T; 3],
    v_ref: &[T; 3],
) -> Result<[[T; 3]; 3]> {
    let ez_raw = cross3(ex, v_ref);
    let ez_mag = norm3(&ez_raw);

    if ez_mag.real_part() < 1e-10 {
        return Err(CoreError::ParallelReferenceVector {
            vx: v_ref[0].real_part(),
            vy: v_ref[1].real_part(),
            vz: v_ref[2].real_part(),
            cross_mag: ez_mag.real_part(),
        });
    }

    let ez = [ez_raw[0] / ez_mag, ez_raw[1] / ez_mag, ez_raw[2] / ez_mag];
    let ey = cross3(&ez, ex);

    Ok([*ex, ey, ez])
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::{matmul, transpose, mat_zero};

    const TOL: f64 = 1e-13;

    fn approx_eq_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) {
        for i in 0..3 {
            for j in 0..3 {
                let d = (a[i][j] - b[i][j]).abs();
                assert!(d <= TOL, "a[{i}][{j}]={} b[{i}][{j}]={}  diff={d:.2e}", a[i][j], b[i][j]);
            }
        }
    }

    fn approx_eq_12x12(a: &[[f64; 12]; 12], b: &[[f64; 12]; 12]) {
        for i in 0..12 {
            for j in 0..12 {
                let d = (a[i][j] - b[i][j]).abs();
                assert!(d <= TOL, "a[{i}][{j}]={} b[{i}][{j}]={}  diff={d:.2e}", a[i][j], b[i][j]);
            }
        }
    }

    fn approx_eq_6x6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) {
        for i in 0..6 {
            for j in 0..6 {
                let d = (a[i][j] - b[i][j]).abs();
                assert!(d <= TOL, "a[{i}][{j}]={} b[{i}][{j}]={}  diff={d:.2e}", a[i][j], b[i][j]);
            }
        }
    }

    fn eye_12() -> [[f64; 12]; 12] {
        let mut m = mat_zero::<12, f64>();
        for i in 0..12 { m[i][i] = 1.0; }
        m
    }

    fn eye_6() -> [[f64; 6]; 6] {
        let mut m = mat_zero::<6, f64>();
        for i in 0..6 { m[i][i] = 1.0; }
        m
    }

    fn eye_3() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    // ---- Construction ----

    #[test]
    fn horizontal_element_x_axis() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 5.0, 0.0, 0.0).unwrap();
        assert!((t.length - 5.0_f64).abs() < TOL);
        // Local x = global x → R[0] = (1, 0, 0)
        assert!((t.rotation[0][0] - 1.0_f64).abs() < TOL);
        assert!(t.rotation[0][1].abs() < TOL);
        assert!(t.rotation[0][2].abs() < TOL);
    }

    #[test]
    fn horizontal_element_is_identity_rotation() {
        // Element along global X with v_ref = (0,1,0):
        // e_x = (1,0,0), e_z = (1,0,0)×(0,1,0) = (0,0,1), e_y = (0,0,1)×(1,0,0) = (0,1,0)
        // R = I
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 3.0, 0.0, 0.0).unwrap();
        approx_eq_3x3(&t.rotation, &eye_3());
    }

    #[test]
    fn vertical_element_auto_fallback() {
        // Element along global Y — auto-fallback to v_ref = (0,0,1)
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 0.0, 5.0, 0.0).unwrap();
        // e_x = (0,1,0)
        assert!(t.rotation[0][0].abs() < TOL);
        assert!((t.rotation[0][1] - 1.0_f64).abs() < TOL);
        assert!(t.rotation[0][2].abs() < TOL);
        // R should still be orthogonal
        let rrt = matmul(&t.rotation, &transpose(&t.rotation));
        approx_eq_3x3(&rrt, &eye_3());
    }

    #[test]
    fn element_along_z_axis() {
        // Element along global Z
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 0.0, 0.0, 4.0).unwrap();
        assert!((t.length - 4.0_f64).abs() < TOL);
        // e_x = (0,0,1)
        assert!((t.rotation[0][2] - 1.0_f64).abs() < TOL);
        // R should be orthogonal
        let rrt = matmul(&t.rotation, &transpose(&t.rotation));
        approx_eq_3x3(&rrt, &eye_3());
    }

    #[test]
    fn diagonal_3d_element() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap();
        let expected_length = 3.0_f64.sqrt();
        assert!((t.length - expected_length).abs() < TOL);
        // R should be orthogonal
        let rrt = matmul(&t.rotation, &transpose(&t.rotation));
        approx_eq_3x3(&rrt, &eye_3());
    }

    #[test]
    fn degenerate_geometry_errors() {
        let result = CoordTransf3d::<f64>::from_nodes(1.0, 2.0, 3.0, 1.0, 2.0, 3.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CoreError::DegenerateGeometry { .. }
        ));
    }

    #[test]
    fn parallel_ref_vector_errors() {
        // Element along global Y, explicit v_ref = (0,1,0) → parallel
        let result = CoordTransf3d::<f64>::from_nodes_with_ref(
            0.0, 0.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 1.0, 0.0,
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CoreError::ParallelReferenceVector { .. }
        ));
    }

    #[test]
    fn custom_ref_vector() {
        // Element along X, custom v_ref = (0,0,1)
        // e_x = (1,0,0), e_z = (1,0,0)×(0,0,1) = (0,-1,0) normalised = (0,-1,0)
        // e_y = (0,-1,0)×(1,0,0) = (0,0,-1)
        let t = CoordTransf3d::<f64>::from_nodes_with_ref(
            0.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ).unwrap();
        // R should be orthogonal
        let rrt = matmul(&t.rotation, &transpose(&t.rotation));
        approx_eq_3x3(&rrt, &eye_3());
        // e_x should still be (1,0,0)
        assert!((t.rotation[0][0] - 1.0_f64).abs() < TOL);
    }

    #[test]
    fn from_rotation_length_validates() {
        // Valid: identity rotation
        let r = eye_3();
        assert!(CoordTransf3d::from_rotation_length(r, 5.0_f64).is_ok());

        // Invalid: non-unit row
        let bad_r: [[f64; 3]; 3] = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(CoordTransf3d::from_rotation_length(bad_r, 5.0).is_err());
    }

    // ---- Rotation matrix properties ----

    #[test]
    fn rotation_is_orthogonal_various_orientations() {
        let cases: Vec<(f64, f64, f64)> = vec![
            (1.0, 0.0, 0.0),  // along X
            (0.0, 1.0, 0.0),  // along Y
            (0.0, 0.0, 1.0),  // along Z
            (1.0, 1.0, 0.0),  // XY plane 45°
            (1.0, 0.0, 1.0),  // XZ plane 45°
            (0.0, 1.0, 1.0),  // YZ plane 45°
            (1.0, 1.0, 1.0),  // space diagonal
            (3.0, 4.0, 0.0),  // general in XY plane
            (1.0, 2.0, 3.0),  // general 3D
        ];

        for (dx, dy, dz) in cases {
            let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, dx, dy, dz).unwrap();
            let rrt = matmul(&t.rotation, &transpose(&t.rotation));
            approx_eq_3x3(&rrt, &eye_3());
        }
    }

    #[test]
    fn rotation_det_is_positive_one() {
        // det(R) = +1 for proper rotation (not a reflection)
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).unwrap();
        let r = &t.rotation;
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.0_f64).abs() < TOL, "det(R) = {det}, expected 1.0");
    }

    // ---- T matrix properties ----

    #[test]
    fn t12x12_horizontal_is_identity() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        approx_eq_12x12(&t.t_matrix_12x12(), &eye_12());
    }

    #[test]
    fn t12x12_is_orthogonal() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).unwrap();
        let t_mat = t.t_matrix_12x12();
        let tt = transpose(&t_mat);
        let product = matmul(&t_mat, &tt);
        approx_eq_12x12(&product, &eye_12());
    }

    #[test]
    fn t6x6_horizontal_is_identity() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        approx_eq_6x6(&t.t_matrix_6x6(), &eye_6());
    }

    #[test]
    fn t6x6_is_orthogonal() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 3.0, 4.0, 5.0).unwrap();
        let t_mat = t.t_matrix_6x6();
        let tt = transpose(&t_mat);
        let product = matmul(&t_mat, &tt);
        approx_eq_6x6(&product, &eye_6());
    }

    // ---- Transform stiffness ----

    #[test]
    fn transform_12x12_horizontal_unchanged() {
        let transf = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        // Simple symmetric 12×12 with some non-trivial values
        let mut ke = mat_zero::<12, f64>();
        ke[0][0] =  1.0; ke[0][6] = -1.0;
        ke[6][0] = -1.0; ke[6][6] =  1.0;
        let kg = transf.transform_stiffness_12x12(&ke);
        approx_eq_12x12(&kg, &ke);
    }

    #[test]
    fn transform_6x6_horizontal_unchanged() {
        let transf = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        let mut ke = mat_zero::<6, f64>();
        ke[0][0] =  1.0; ke[0][3] = -1.0;
        ke[3][0] = -1.0; ke[3][3] =  1.0;
        let kg = transf.transform_stiffness_6x6(&ke);
        approx_eq_6x6(&kg, &ke);
    }

    #[test]
    fn transform_preserves_trace() {
        // For orthogonal T: tr(Tᵀ K T) = tr(K)
        let transf = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).unwrap();
        let mut ke = mat_zero::<12, f64>();
        // Fill with a symmetric pattern
        for i in 0..12 {
            ke[i][i] = (i + 1) as f64;
            if i + 1 < 12 {
                ke[i][i + 1] = 0.5;
                ke[i + 1][i] = 0.5;
            }
        }
        let kg = transf.transform_stiffness_12x12(&ke);
        let trace_ke: f64 = (0..12).map(|i| ke[i][i]).sum();
        let trace_kg: f64 = (0..12).map(|i| kg[i][i]).sum();
        assert!(
            (trace_ke - trace_kg).abs() < 1e-10,
            "trace_ke={trace_ke} trace_kg={trace_kg}"
        );
    }

    #[test]
    fn transform_preserves_symmetry() {
        let transf = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).unwrap();
        let mut ke = mat_zero::<12, f64>();
        for i in 0..12 {
            ke[i][i] = (i + 1) as f64;
            if i + 1 < 12 {
                ke[i][i + 1] = 0.5;
                ke[i + 1][i] = 0.5;
            }
        }
        let kg = transf.transform_stiffness_12x12(&ke);
        for i in 0..12 {
            for j in i + 1..12 {
                let d = (kg[i][j] - kg[j][i]).abs();
                assert!(d < 1e-12, "kg[{i}][{j}]={} kg[{j}][{i}]={}  diff={d:.2e}", kg[i][j], kg[j][i]);
            }
        }
    }

    // ---- Axis accessors ----

    #[test]
    fn axis_accessors_match_rotation_rows() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).unwrap();
        assert_eq!(t.local_x_axis(), t.rotation[0]);
        assert_eq!(t.local_y_axis(), t.rotation[1]);
        assert_eq!(t.local_z_axis(), t.rotation[2]);
    }

    // ---- Reversed ----

    #[test]
    fn reversed_flips_x_axis() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).unwrap();
        let r = t.reversed();
        // e_x negated
        for j in 0..3 {
            assert!((r.rotation[0][j] + t.rotation[0][j]).abs() < TOL);
        }
        // e_y negated (to preserve handedness)
        for j in 0..3 {
            assert!((r.rotation[1][j] + t.rotation[1][j]).abs() < TOL);
        }
        // e_z preserved
        for j in 0..3 {
            assert!((r.rotation[2][j] - t.rotation[2][j]).abs() < TOL);
        }
        // length preserved
        assert!((r.length - t.length).abs() < TOL);
    }

    #[test]
    fn reversed_rotation_is_still_orthogonal() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 3.0, 4.0, 5.0).unwrap();
        let r = t.reversed();
        let rrt = matmul(&r.rotation, &transpose(&r.rotation));
        approx_eq_3x3(&rrt, &eye_3());
    }

    #[test]
    fn reversed_det_is_positive_one() {
        let t = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 1.0, 2.0, 3.0).unwrap();
        let r_rev = &t.reversed().rotation;
        let det = r_rev[0][0] * (r_rev[1][1] * r_rev[2][2] - r_rev[1][2] * r_rev[2][1])
                - r_rev[0][1] * (r_rev[1][0] * r_rev[2][2] - r_rev[1][2] * r_rev[2][0])
                + r_rev[0][2] * (r_rev[1][0] * r_rev[2][1] - r_rev[1][1] * r_rev[2][0]);
        assert!((det - 1.0_f64).abs() < TOL, "det(R_rev) = {det}, expected 1.0");
    }

    // ---- 3D truss vertical transform ----

    #[test]
    fn transform_6x6_vertical_truss() {
        // Element along global Y → axial stiffness moves to y-DOFs
        let transf = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 0.0, 1.0, 0.0).unwrap();
        let ke_local: [[f64; 6]; 6] = {
            let mut k = mat_zero::<6, f64>();
            // Axial stiffness in local x-direction
            k[0][0] =  1.0; k[0][3] = -1.0;
            k[3][0] = -1.0; k[3][3] =  1.0;
            k
        };
        let kg = transf.transform_stiffness_6x6(&ke_local);
        // Stiffness should be in global y-direction now
        assert!((kg[1][1] -  1.0_f64).abs() < TOL, "kg[1][1]={}", kg[1][1]);
        assert!((kg[1][4] - -1.0_f64).abs() < TOL, "kg[1][4]={}", kg[1][4]);
        assert!((kg[4][1] - -1.0_f64).abs() < TOL, "kg[4][1]={}", kg[4][1]);
        assert!((kg[4][4] -  1.0_f64).abs() < TOL, "kg[4][4]={}", kg[4][4]);
        // x-direction: zero stiffness
        assert!(kg[0][0].abs() < TOL, "kg[0][0]={}", kg[0][0]);
    }

    #[test]
    fn transform_6x6_z_axis_truss() {
        // Element along global Z → axial stiffness moves to z-DOFs
        let transf = CoordTransf3d::<f64>::from_nodes(0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();
        let ke_local: [[f64; 6]; 6] = {
            let mut k = mat_zero::<6, f64>();
            k[0][0] =  1.0; k[0][3] = -1.0;
            k[3][0] = -1.0; k[3][3] =  1.0;
            k
        };
        let kg = transf.transform_stiffness_6x6(&ke_local);
        // Stiffness should be in global z-direction now (indices 2 and 5)
        assert!((kg[2][2] -  1.0_f64).abs() < TOL, "kg[2][2]={}", kg[2][2]);
        assert!((kg[2][5] - -1.0_f64).abs() < TOL, "kg[2][5]={}", kg[2][5]);
        assert!((kg[5][2] - -1.0_f64).abs() < TOL, "kg[5][2]={}", kg[5][2]);
        assert!((kg[5][5] -  1.0_f64).abs() < TOL, "kg[5][5]={}", kg[5][5]);
    }
}
