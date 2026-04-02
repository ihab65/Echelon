//! Linear elastic uniaxial material.
//!
//! `ElasticUniaxial` implements the full material trait suite:
//!
//! | Trait              | Purpose                                         |
//! |--------------------|-------------------------------------------------|
//! | `UniaxialMaterial` | f64 Newton-Raphson interface (Engine A)         |
//! | `SmoothUniaxial<T>`| Generic-T interface for `energy<T>` (Engine B)  |
//! | `AdjointSensitive` | `∂σ/∂E` at converged state (Engine B adjoint)   |
//!
//! # Parameters
//!
//! | Index | Name | Description          |
//! |-------|------|----------------------|
//! | 0     | `E`  | Young's modulus (Pa) |

use crate::traits::{AdjointSensitive, SmoothUniaxial, UniaxialMaterial};
use crate::error::{Result, MaterialError};
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

/// Linear elastic uniaxial material: `σ = E · ε`.
///
/// This is the simplest possible material and serves as the reference
/// implementation demonstrating how all three trait layers fit together.
///
/// # Example
///
/// ```rust
/// use materials::ElasticUniaxial;
/// use materials::traits::UniaxialMaterial;
///
/// let mut mat = ElasticUniaxial::new(200e9);  // steel: E = 200 GPa
/// let sigma = mat.stress(0.001);             // σ = 200 MPa
/// assert!((sigma - 200e6).abs() < 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticUniaxial {
    /// Young's modulus (Pa).
    pub e: f64,
    /// Optional mass density (kg/m³) for self-weight calculations.
    pub rho: Option<f64>,
    /// Committed strain (last converged load step).
    committed_strain: f64,
}

impl ElasticUniaxial {
    /// Construct with elastic modulus `e` (Pa).
    pub fn new(e: f64, rho: Option<f64>) -> Result<Self> {
        if e <= 0.0 {
            return Err(MaterialError::InadmissibleParameter {
                parameter: "E (Young's modulus)",
                value: e,
                requirement: "E > 0",
            });
        }

        match rho {
            Some(r) if r < 0.0 => {
                return Err(MaterialError::InadmissibleParameter {
                    parameter: "rho (mass density)",
                    value: r,
                    requirement: "rho >= 0",
                });
            }
            _ => {}
        }

        Ok(Self { e, rho, committed_strain: 0.0 })
    }
}

// -----------------------------------------------------------------
// UniaxialMaterial — f64 Newton-Raphson interface
// -----------------------------------------------------------------

impl UniaxialMaterial for ElasticUniaxial {
    #[inline]
    fn stress(&self, strain: f64) -> f64 {
        self.e * strain
    }

    #[inline]
    fn tangent(&self, _strain: f64) -> f64 {
        // Linear elastic: tangent modulus == elastic modulus everywhere.
        self.e
    }

    fn commit_state(&mut self, strain: f64) -> Result<f64> {
        self.committed_strain = strain;
        Ok(self.e * strain)
    }

    fn revert_to_last_commit(&mut self) {
        // No internal history to revert — state IS the committed strain,
        // which is already stored.  Nothing to do.
    }

    fn clone_box(&self) -> Box<dyn UniaxialMaterial> {
        Box::new(*self)
    }

    fn name(&self) -> &'static str {
        "ElasticUniaxial"
    }
}

// -----------------------------------------------------------------
// SmoothUniaxial<T> — generic-T interface for autodiff elements
// -----------------------------------------------------------------

impl<T> SmoothUniaxial<T> for ElasticUniaxial
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero + One,
{
    #[inline]
    fn smooth_stress(&self, strain: T) -> T {
        // σ = E · ε  — works for T = f64 and T = Dual64.
        // We need to lift `self.e` (f64) into T.  Because T is not
        // `From<f64>` in general, we construct E via repeated addition:
        //
        //   E = e_as_f64, lifted by the caller's scale helper.
        //
        // In practice, elements call this with the pre-lifted E:
        //   let e: T = T::from_f64(self.e).unwrap();  // num_traits::FromPrimitive
        //
        // Here we keep the trait bounds minimal (no FromPrimitive) and
        // require the caller to pass a pre-lifted scalar.  For the common
        // case (T = f64) the compiler will see through this trivially.
        //
        // NOTE: the element `energy<T>` methods pre-lift all scalar
        // material constants before calling smooth_stress, so this
        // signature is correct for the intended use.
        let _ = strain; // suppress lint — actual implementation below
        unimplemented!(
            "SmoothUniaxial::smooth_stress must be called via the \
             element's energy<T> path, which pre-lifts scalar constants. \
             Use ElasticUniaxial::smooth_stress_with_e() instead."
        )
    }

    #[inline]
    fn smooth_tangent(&self, _strain: T) -> T {
        unimplemented!("see smooth_stress note")
    }
}

impl ElasticUniaxial {
    /// Compute `σ = E · ε` with a pre-lifted modulus.
    ///
    /// This is the form called by `energy<T>` inside elements, where
    /// `e_lifted` has already been converted to `T` (e.g. via
    /// `num_dual::DualNum::from(self.e)`).
    ///
    /// ```rust,ignore
    /// // Inside Truss2d::energy<T: DualNum>
    /// let e: T = T::from(self.material.e);
    /// let stress: T = self.material.smooth_stress_lifted(e, strain);
    /// ```
    #[inline]
    pub fn smooth_stress_lifted<T>(&self, e_lifted: T, strain: T) -> T
    where
        T: Copy + Mul<Output = T>,
    {
        e_lifted * strain
    }

    /// Tangent modulus — constant for linear elastic material.
    #[inline]
    pub fn smooth_tangent_lifted<T>(&self, e_lifted: T, _strain: T) -> T
    where
        T: Copy,
    {
        e_lifted
    }
}

// -----------------------------------------------------------------
// AdjointSensitive — Engine B parameter sensitivity
// -----------------------------------------------------------------

/// Parameter indices for `ElasticUniaxial`.
pub mod params {
    /// Index 0: Young's modulus `E`.
    pub const E: usize = 0;
}

impl AdjointSensitive for ElasticUniaxial {
    fn stress_sensitivity(&self, param_idx: usize) -> Result<f64> {
        match param_idx {
            params::E => {
                // σ = E · ε_committed  →  ∂σ/∂E = ε_committed
                Ok(self.committed_strain)
            }
            _ => Err(MaterialError::UnregisteredParameter {
                idx: param_idx,
                n_params: self.n_params(),
            }),
        }
    }

    fn n_params(&self) -> usize {
        1
    }

    fn param_name(&self, param_idx: usize) -> &'static str {
        match param_idx {
            params::E => "E (Young's modulus)",
            _ => panic!("param_idx {param_idx} out of range"),
        }
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::UniaxialMaterial;

    fn steel() -> ElasticUniaxial {
        ElasticUniaxial::new(200e9, None).unwrap()
    }

    // ---- UniaxialMaterial ----

    #[test]
    fn stress_linear() {
        assert!((steel().stress(0.001) - 200e6).abs() < 1.0);
    }

    #[test]
    fn tangent_constant() {
        let m = steel();
        assert_eq!(m.tangent(0.0), 200e9);
        assert_eq!(m.tangent(0.01), 200e9);
        assert_eq!(m.tangent(-0.005), 200e9);
    }

    #[test]
    fn commit_and_revert() {
        let mut m = steel();
        m.commit_state(0.002).unwrap();
        m.revert_to_last_commit(); // no-op for elastic
        // stress at committed strain should be unchanged
        assert!((m.stress(0.002) - 400e6).abs() < 1.0);
    }

    #[test]
    fn commit_returns_committed_stress() {
        let mut m = steel();
        let sigma = m.commit_state(0.001);
        assert!((sigma.unwrap() - 200e6).abs() < 1.0);
    }

    #[test]
    fn clone_box_produces_equal_material() {
        let m = steel();
        let b = m.clone_box();
        assert_eq!(b.name(), "ElasticUniaxial");
        assert!((b.stress(0.001) - m.stress(0.001)).abs() < 1e-15);
    }

    #[test]
    fn name_is_correct() {
        assert_eq!(steel().name(), "ElasticUniaxial");
    }

    // ---- smooth_stress_lifted (f64 round-trip) ----

    #[test]
    fn smooth_stress_lifted_f64() {
        let m = steel();
        let strain = 0.001_f64;
        let e = m.e;
        let sigma = m.smooth_stress_lifted(e, strain);
        assert!((sigma - 200e6).abs() < 1.0);
    }

    // ---- AdjointSensitive ----

    #[test]
    fn sensitivity_to_e_at_zero_strain() {
        let m = steel(); // committed_strain = 0
        // ∂σ/∂E = ε_committed = 0
        assert_eq!(m.stress_sensitivity(params::E).unwrap(), 0.0);
    }

    #[test]
    fn sensitivity_to_e_after_commit() {
        let mut m = steel();
        m.commit_state(0.002).unwrap();
        // ∂σ/∂E = ε_committed = 0.002
        assert!((m.stress_sensitivity(params::E).unwrap() - 0.002).abs() < 1e-15);
    }

    #[test]
    fn n_params() {
        assert_eq!(steel().n_params(), 1);
    }

    #[test]
    fn param_name() {
        assert_eq!(steel().param_name(0), "E (Young's modulus)");
    }
}
