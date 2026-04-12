//! Linear elastic isotropic ND material.
//!
//! `ElasticIsotropic` is the multi-dimensional analogue of [`ElasticUniaxial`]:
//! it models a linear, isotropic, elastic solid characterised by Young's
//! modulus `E` and Poisson's ratio `ν`.
//!
//! The material is constructed with a specific [`NdOrder`], which selects
//! the Voigt convention (plane-stress, plane-strain, or full 3-D) and
//! pre-computes the corresponding elastic stiffness tensor at construction
//! time.  All hot-path methods — [`stress`] and [`tangent`] — are then
//! simple matrix-vector or block-copy operations with **zero allocations**.
//!
//! # Parameters
//!
//! | Symbol | Description              | Admissibility     |
//! |--------|--------------------------|-------------------|
//! | `E`    | Young's modulus (Pa)      | `E > 0`          |
//! | `ν`    | Poisson's ratio           | `-1 < ν < 0.5`   |
//!
//! # Supported formulations
//!
//! | [`NdOrder`] variant | `order()` | Description       |
//! |---------------------|-----------|-------------------|
//! | `PlaneStress`       | 3         | σzz = 0           |
//! | `PlaneStrain`       | 4         | εzz = 0           |
//! | `ThreeDimensional`  | 6         | General 3-D solid |

use crate::error::{MaterialError, Result};
use crate::traits::NdMaterial;

// ---- Maximum order (full 3-D = 6 Voigt components) ----

/// Maximum Voigt order supported by any ND formulation.
const MAX_ORDER: usize = 6;

/// Maximum tangent size: `MAX_ORDER * MAX_ORDER = 36`.
const MAX_TANGENT: usize = MAX_ORDER * MAX_ORDER;

/// Maximum strain vector size (same as `MAX_ORDER`).
const MAX_STRAIN: usize = MAX_ORDER;

// -----------------------------------------------------------------
// Formulation selector
// -----------------------------------------------------------------

/// Selects the Voigt convention for [`ElasticIsotropic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdOrder {
    /// 2-D plane stress: σzz = τxz = τyz = 0.
    /// Voigt vector: `[εxx, εyy, γxy]` — order 3.
    PlaneStress,
    /// 2-D plane strain: εzz = γxz = γyz = 0 (σzz ≠ 0).
    /// Voigt vector: `[εxx, εyy, εzz, γxy]` — order 4.
    PlaneStrain,
    /// Full 3-D solid.
    /// Voigt vector: `[εxx, εyy, εzz, γxy, γyz, γxz]` — order 6.
    ThreeDimensional,
}

impl NdOrder {
    /// Number of Voigt components for this formulation.
    #[inline]
    pub const fn len(self) -> usize {
        match self {
            NdOrder::PlaneStress => 3,
            NdOrder::PlaneStrain => 4,
            NdOrder::ThreeDimensional => 6,
        }
    }
}

// -----------------------------------------------------------------
// Struct
// -----------------------------------------------------------------

/// Linear elastic isotropic material: `σ = C : ε`.
///
/// This is the simplest ND material and serves as the reference
/// implementation for the [`NdMaterial`] trait, in the same way that
/// [`ElasticUniaxial`] is the reference for [`UniaxialMaterial`].
///
/// The elastic stiffness tensor `C` is pre-computed at construction
/// time and stored as a flat row-major array.  The maximum allocation
/// is `[f64; 36]` (6×6 for full 3-D), which lives entirely on the
/// stack inside the struct.
///
/// # Example
///
/// ```rust
/// use materials::{ElasticIsotropic, NdMaterial, NdOrder};
///
/// // Plane-stress steel: E = 200 GPa, ν = 0.3
/// let mat = ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, None).unwrap();
/// assert_eq!(mat.order(), 3);
///
/// let strain = [0.001, -0.0003, 0.0];
/// let mut stress = [0.0; 3];
/// mat.stress(&strain, &mut stress);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticIsotropic {
    /// Young's modulus (Pa).
    pub e: f64,
    /// Poisson's ratio (dimensionless).
    pub nu: f64,
    /// Formulation / Voigt order.
    pub nd_order: NdOrder,
    /// Optional mass density (kg/m³) for self-weight calculations.
    pub rho: Option<f64>,
    /// Pre-computed elastic stiffness tensor (row-major, flat).
    /// Only the first `order * order` entries are meaningful.
    c_flat: [f64; MAX_TANGENT],
    /// Committed strain vector.
    /// Only the first `order` entries are meaningful.
    committed_strain: [f64; MAX_STRAIN],
}

impl ElasticIsotropic {
    /// Construct a new `ElasticIsotropic` material.
    ///
    /// # Arguments
    /// * `e`  — Young's modulus (Pa), must be `> 0`.
    /// * `nu` — Poisson's ratio, must satisfy `-1 < ν < 0.5`.
    /// * `nd_order` — Voigt formulation to use.
    /// * `rho` — Optional mass density (kg/m³), must be `>= 0` if present.
    ///
    /// # Errors
    /// - [`MaterialError::InadmissibleParameter`] if `e <= 0`, `nu` is
    ///   outside the open interval `(-1, 0.5)`, or `rho < 0`.
    pub fn new(e: f64, nu: f64, nd_order: NdOrder, rho: Option<f64>) -> Result<Self> {
        if e <= 0.0 {
            return Err(MaterialError::InadmissibleParameter {
                parameter: "E (Young's modulus)",
                value: e,
                requirement: "E > 0",
            });
        }
        if nu <= -1.0 || nu >= 0.5 {
            return Err(MaterialError::InadmissibleParameter {
                parameter: "ν (Poisson's ratio)",
                value: nu,
                requirement: "-1 < ν < 0.5",
            });
        }
        if let Some(r) = rho {
            if r < 0.0 {
                return Err(MaterialError::InadmissibleParameter {
                    parameter: "rho (mass density)",
                    value: r,
                    requirement: "rho >= 0",
                });
            }
        }

        let mut c_flat = [0.0; MAX_TANGENT];
        Self::build_tangent(e, nu, nd_order, &mut c_flat);

        Ok(Self {
            e,
            nu,
            nd_order,
            rho,
            c_flat,
            committed_strain: [0.0; MAX_STRAIN],
        })
    }

    // ---- Internal: build the elastic stiffness tensor ----

    /// Fill `c` with the elastic stiffness tensor in Voigt notation.
    fn build_tangent(e: f64, nu: f64, nd_order: NdOrder, c: &mut [f64; MAX_TANGENT]) {
        c.fill(0.0);
        match nd_order {
            NdOrder::PlaneStress => Self::build_plane_stress(e, nu, c),
            NdOrder::PlaneStrain => Self::build_plane_strain(e, nu, c),
            NdOrder::ThreeDimensional => Self::build_3d(e, nu, c),
        }
    }

    /// Plane-stress elastic stiffness: 3×3.
    ///
    /// ```text
    /// C = E / (1 - ν²) * | 1   ν   0           |
    ///                     | ν   1   0           |
    ///                     | 0   0   (1-ν)/2     |
    /// ```
    fn build_plane_stress(e: f64, nu: f64, c: &mut [f64; MAX_TANGENT]) {
        let n = 3; // order
        let factor = e / (1.0 - nu * nu);
        let g = 0.5 * (1.0 - nu);
        // Row 0: [1, ν, 0]
        c[0 * n + 0] = factor;
        c[0 * n + 1] = factor * nu;
        // Row 1: [ν, 1, 0]
        c[1 * n + 0] = factor * nu;
        c[1 * n + 1] = factor;
        // Row 2: [0, 0, (1-ν)/2]
        c[2 * n + 2] = factor * g;
    }

    /// Plane-strain elastic stiffness: 4×4.
    ///
    /// Voigt order: `[εxx, εyy, εzz, γxy]`
    ///
    /// ```text
    /// C = E / ((1+ν)(1-2ν)) * | 1-ν   ν    ν    0             |
    ///                          |  ν   1-ν   ν    0             |
    ///                          |  ν    ν   1-ν   0             |
    ///                          |  0    0    0    (1-2ν)/2       |
    /// ```
    fn build_plane_strain(e: f64, nu: f64, c: &mut [f64; MAX_TANGENT]) {
        let n = 4; // order
        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let diag = factor * (1.0 - nu);
        let off = factor * nu;
        let shear = factor * (1.0 - 2.0 * nu) / 2.0;

        // Normal block (3×3 upper-left)
        c[0 * n + 0] = diag;
        c[0 * n + 1] = off;
        c[0 * n + 2] = off;
        c[1 * n + 0] = off;
        c[1 * n + 1] = diag;
        c[1 * n + 2] = off;
        c[2 * n + 0] = off;
        c[2 * n + 1] = off;
        c[2 * n + 2] = diag;
        // Shear
        c[3 * n + 3] = shear;
    }

    /// Full 3-D elastic stiffness: 6×6.
    ///
    /// Voigt order: `[εxx, εyy, εzz, γxy, γyz, γxz]`
    ///
    /// ```text
    /// C = E / ((1+ν)(1-2ν)) * | 1-ν   ν    ν    0          0          0          |
    ///                          |  ν   1-ν   ν    0          0          0          |
    ///                          |  ν    ν   1-ν   0          0          0          |
    ///                          |  0    0    0    (1-2ν)/2   0          0          |
    ///                          |  0    0    0    0          (1-2ν)/2   0          |
    ///                          |  0    0    0    0          0          (1-2ν)/2   |
    /// ```
    fn build_3d(e: f64, nu: f64, c: &mut [f64; MAX_TANGENT]) {
        let n = 6; // order
        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let diag = factor * (1.0 - nu);
        let off = factor * nu;
        let shear = factor * (1.0 - 2.0 * nu) / 2.0;

        // Normal block (3×3 upper-left)
        c[0 * n + 0] = diag;
        c[0 * n + 1] = off;
        c[0 * n + 2] = off;
        c[1 * n + 0] = off;
        c[1 * n + 1] = diag;
        c[1 * n + 2] = off;
        c[2 * n + 0] = off;
        c[2 * n + 1] = off;
        c[2 * n + 2] = diag;
        // Shear diagonal (3 entries)
        c[3 * n + 3] = shear;
        c[4 * n + 4] = shear;
        c[5 * n + 5] = shear;
    }

    /// Read-only access to the pre-computed elastic stiffness tensor.
    ///
    /// Returns a slice of length `order() * order()`.
    #[inline]
    pub fn tangent_tensor(&self) -> &[f64] {
        let n = self.nd_order.len();
        &self.c_flat[..n * n]
    }
}

// -----------------------------------------------------------------
// NdMaterial — f64 Newton-Raphson interface
// -----------------------------------------------------------------

impl NdMaterial for ElasticIsotropic {
    #[inline]
    fn order(&self) -> usize {
        self.nd_order.len()
    }

    fn stress(&self, strain: &[f64], out: &mut [f64]) {
        let n = self.nd_order.len();
        debug_assert_eq!(strain.len(), n, "strain length must equal order()");
        debug_assert_eq!(out.len(), n, "stress output length must equal order()");

        // σ = C · ε   (matrix-vector product, row-major C)
        for i in 0..n {
            let mut acc = 0.0;
            let row_start = i * n;
            for j in 0..n {
                acc += self.c_flat[row_start + j] * strain[j];
            }
            out[i] = acc;
        }
    }

    fn tangent(&self, strain: &[f64], out: &mut [f64]) {
        let n = self.nd_order.len();
        debug_assert_eq!(strain.len(), n, "strain length must equal order()");
        debug_assert_eq!(
            out.len(),
            n * n,
            "tangent output length must equal order()²"
        );

        // Linear elastic: tangent is the constant elastic stiffness tensor.
        out[..n * n].copy_from_slice(&self.c_flat[..n * n]);
    }

    fn commit_state(&mut self, strain: &[f64]) -> Result<()> {
        let n = self.nd_order.len();
        debug_assert_eq!(strain.len(), n, "strain length must equal order()");
        self.committed_strain[..n].copy_from_slice(strain);
        Ok(())
    }

    fn revert_to_last_commit(&mut self) {
        // No internal history to revert — state IS the committed strain,
        // which is already stored.  Nothing to do.
    }

    fn clone_box(&self) -> Box<dyn NdMaterial> {
        Box::new(self.clone())
    }

    fn name(&self) -> &'static str {
        "ElasticIsotropic"
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn steel_ps() -> ElasticIsotropic {
        ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, None).unwrap()
    }

    fn steel_pe() -> ElasticIsotropic {
        ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStrain, None).unwrap()
    }

    fn steel_3d() -> ElasticIsotropic {
        ElasticIsotropic::new(200e9, 0.3, NdOrder::ThreeDimensional, None).unwrap()
    }

    // ---- Construction ----

    #[test]
    fn invalid_e_zero() {
        assert!(ElasticIsotropic::new(0.0, 0.3, NdOrder::PlaneStress, None).is_err());
    }

    #[test]
    fn invalid_e_negative() {
        assert!(ElasticIsotropic::new(-1.0, 0.3, NdOrder::PlaneStress, None).is_err());
    }

    #[test]
    fn invalid_nu_at_half() {
        assert!(ElasticIsotropic::new(200e9, 0.5, NdOrder::PlaneStress, None).is_err());
    }

    #[test]
    fn invalid_nu_below_neg_one() {
        assert!(ElasticIsotropic::new(200e9, -1.0, NdOrder::PlaneStress, None).is_err());
    }

    #[test]
    fn invalid_rho_negative() {
        assert!(ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, Some(-1.0)).is_err());
    }

    #[test]
    fn valid_construction() {
        assert!(steel_ps().order() == 3);
        assert!(steel_pe().order() == 4);
        assert!(steel_3d().order() == 6);
    }

    // ---- Tangent symmetry ----

    fn assert_tangent_symmetric(mat: &ElasticIsotropic) {
        let n = mat.order();
        let c = mat.tangent_tensor();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[i * n + j] - c[j * n + i]).abs() < 1e-6,
                    "tangent not symmetric at ({i},{j}): {} vs {}",
                    c[i * n + j],
                    c[j * n + i],
                );
            }
        }
    }

    #[test]
    fn tangent_symmetric_plane_stress() {
        assert_tangent_symmetric(&steel_ps());
    }

    #[test]
    fn tangent_symmetric_plane_strain() {
        assert_tangent_symmetric(&steel_pe());
    }

    #[test]
    fn tangent_symmetric_3d() {
        assert_tangent_symmetric(&steel_3d());
    }

    // ---- Stress correctness ----

    #[test]
    fn stress_plane_stress_uniaxial() {
        // Apply εxx = 0.001 with εyy = γxy = 0 → σxx should be E/(1-ν²) * 0.001
        let mat = steel_ps();
        let strain = [0.001, 0.0, 0.0];
        let mut stress = [0.0; 3];
        mat.stress(&strain, &mut stress);

        let expected_xx = 200e9 / (1.0 - 0.3 * 0.3) * 0.001;
        assert!(
            (stress[0] - expected_xx).abs() < 1.0,
            "σxx = {}, expected {}",
            stress[0],
            expected_xx
        );
        // σyy = ν * E / (1-ν²) * εxx
        let expected_yy = 0.3 * 200e9 / (1.0 - 0.3 * 0.3) * 0.001;
        assert!(
            (stress[1] - expected_yy).abs() < 1.0,
            "σyy = {}, expected {}",
            stress[1],
            expected_yy
        );
        assert!(stress[2].abs() < 1e-6, "τxy should be zero");
    }

    #[test]
    fn stress_3d_hydrostatic() {
        // Apply equal strain in all normal directions:
        // εxx = εyy = εzz = e, γxy = γyz = γxz = 0
        // σxx = E/(1+ν)(1-2ν) * [(1-ν)e + νe + νe] = E·e/(1-2ν)
        let mat = steel_3d();
        let e = 0.001;
        let strain = [e, e, e, 0.0, 0.0, 0.0];
        let mut stress = [0.0; 6];
        mat.stress(&strain, &mut stress);

        let expected = 200e9 * e / (1.0 - 2.0 * 0.3);
        for i in 0..3 {
            assert!(
                (stress[i] - expected).abs() < 1.0,
                "σ[{i}] = {}, expected {expected}",
                stress[i],
            );
        }
        for i in 3..6 {
            assert!(stress[i].abs() < 1e-6, "τ[{i}] should be zero");
        }
    }

    #[test]
    fn stress_pure_shear_plane_stress() {
        let mat = steel_ps();
        let gamma = 0.002;
        let strain = [0.0, 0.0, gamma];
        let mut stress = [0.0; 3];
        mat.stress(&strain, &mut stress);

        // τxy = G * γxy where G = E / (2(1+ν))
        let g = 200e9 / (2.0 * (1.0 + 0.3));
        let expected = g * gamma;
        assert!(
            (stress[2] - expected).abs() < 1.0,
            "τxy = {}, expected {expected}",
            stress[2],
        );
        assert!(stress[0].abs() < 1e-6);
        assert!(stress[1].abs() < 1e-6);
    }

    // ---- Zero strain → zero stress ----

    #[test]
    fn zero_strain_zero_stress_ps() {
        let mat = steel_ps();
        let mut stress = [0.0; 3];
        mat.stress(&[0.0; 3], &mut stress);
        assert!(stress.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn zero_strain_zero_stress_3d() {
        let mat = steel_3d();
        let mut stress = [0.0; 6];
        mat.stress(&[0.0; 6], &mut stress);
        assert!(stress.iter().all(|&v| v == 0.0));
    }

    // ---- Tangent output matches stored tensor ----

    #[test]
    fn tangent_method_matches_stored() {
        let mat = steel_3d();
        let n = mat.order();
        let mut out = [0.0; MAX_TANGENT];
        mat.tangent(&[0.0; 6], &mut out[..n * n]);
        assert_eq!(&out[..n * n], mat.tangent_tensor());
    }

    // ---- State management ----

    #[test]
    fn commit_and_revert() {
        let mut mat = steel_ps();
        let strain = [0.001, -0.0003, 0.0];
        mat.commit_state(&strain).unwrap();
        mat.revert_to_last_commit(); // no-op for elastic
        // Committed strain should still be stored
        assert_eq!(&mat.committed_strain[..3], &strain);
    }

    // ---- clone_box ----

    #[test]
    fn clone_box_produces_equal_material() {
        let mat = steel_3d();
        let cloned = mat.clone_box();
        assert_eq!(cloned.name(), "ElasticIsotropic");
        assert_eq!(cloned.order(), 6);

        let strain = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut s1 = [0.0; 6];
        let mut s2 = [0.0; 6];
        mat.stress(&strain, &mut s1);
        cloned.stress(&strain, &mut s2);
        for i in 0..6 {
            assert!((s1[i] - s2[i]).abs() < 1e-6);
        }
    }

    // ---- name ----

    #[test]
    fn name_is_correct() {
        assert_eq!(steel_ps().name(), "ElasticIsotropic");
    }

    // ---- Plane-stress shear modulus consistency ----

    #[test]
    fn plane_stress_shear_entry() {
        // C[2][2] for plane-stress should be G = E / (2(1+ν))
        let mat = steel_ps();
        let g = 200e9 / (2.0 * (1.0 + 0.3));
        let c = mat.tangent_tensor();
        assert!(
            (c[2 * 3 + 2] - g).abs() < 1.0,
            "C[2,2] = {} expected G = {g}",
            c[2 * 3 + 2],
        );
    }
}
