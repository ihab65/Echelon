//! Convergence test based on the incremental energy (virtual work).
//!
//! [`EnergyIncrement`] is the gold-standard convergence criterion for nonlinear
//! structural analysis. It measures the virtual work done by the residual forces
//! through the displacement increment — an energetically consistent quantity
//! that is invariant to the choice of force and displacement units.
//!
//! ## Formula
//!
//! Convergence is declared at iteration `k` when:
//!
//! ```text
//! 0.5 |Δu_k · R_k| ≤ ε
//! ```
//!
//! where `·` denotes the Euclidean inner product, `Δu_k` is the displacement
//! increment from the current Newton solve, and `R_k` is the residual force
//! vector before the solve.
//!
//! ## Why 0.5?
//!
//! The factor of `0.5` arises from the work-energy theorem for a linear
//! increment: the work done against the linear restoring force is half the
//! force times the displacement. This makes the criterion consistent with
//! the elastic strain energy recovered in each step.
//!
//! ## Advantages over force and displacement norms
//!
//! - **Dimensionally consistent**: for a system with forces in kN and
//!   displacements in mm, the energy unit (kN·mm = J) combines them
//!   naturally without arbitrary scaling.
//! - **Sensitive to both**: a large displacement with a small force (near
//!   a flat plateau) and a small displacement with a large force (near
//!   a steep elastic wall) both contribute proportionally.
//! - **Post-peak accuracy**: particularly important when `K_T` is nearly
//!   singular and the residual norm is no longer a reliable indicator.
//!
//! ## Typical tolerances
//!
//! Because the energy has units of force × displacement, the appropriate
//! tolerance depends on the problem scale:
//!
//! - Forces in N, displacements in m:  `ε ≈ 1e-8` N·m = `1e-8` J
//! - Forces in kN, displacements in mm: `ε ≈ 1e-6` kN·mm = `1e-3` J
//! - Normalised (dimensionless):        `ε ≈ 1e-12`

use crate::system::GlobalSystem;
use crate::convergence::ConvergenceTest;

// -----------------------------------------------------------------
// EnergyIncrement
// -----------------------------------------------------------------

/// Convergence criterion: `0.5 |Δu · R| ≤ tolerance`.
///
/// The most rigorous and dimensionally consistent stopping criterion for
/// Newton-Raphson. Measures the virtual work of the residual forces through
/// the displacement increment.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::tests::energy::EnergyIncrement;
///
/// // Converge when virtual work increment < 1e-8 N·m
/// let test = EnergyIncrement::new(1e-8);
/// ```
#[derive(Debug, Clone)]
pub struct EnergyIncrement {
    /// Absolute convergence tolerance on `0.5 |Δu · R|`.
    pub tolerance: f64,
}

impl EnergyIncrement {
    /// Create a new `EnergyIncrement` criterion with the given absolute tolerance.
    ///
    /// # Panics
    /// Panics if `tolerance ≤ 0.0`.
    pub fn new(tolerance: f64) -> Self {
        assert!(tolerance > 0.0, "EnergyIncrement tolerance must be positive");
        Self { tolerance }
    }
}

impl ConvergenceTest for EnergyIncrement {
    /// Return `true` if `0.5 |Δu · R| ≤ self.tolerance`.
    ///
    /// Skips the check on iteration 0 — the first residual vector is the
    /// applied load (not an out-of-balance correction), so the energy
    /// product is not yet meaningful.
    fn check(&self, system: &GlobalSystem, iter: usize) -> bool {
        if iter == 0 {
            return false;
        }
        system.energy_increment() <= self.tolerance
    }

    fn name(&self) -> &'static str {
        "EnergyIncrement"
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;
    use crate::system::GlobalSystem;

    fn tiny_system() -> GlobalSystem {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 1.0);
        coo.add(1, 1, 1.0);
        coo.add(2, 2, 1.0);
        GlobalSystem::new(coo.build_sym().unwrap())
    }

    #[test]
    fn converged_when_energy_small() {
        let mut sys = tiny_system();
        // 0.5 * |1e-7 * 1e-7| = 0.5 * 1e-14 < 1e-8
        sys.delta_u = vec![1e-7, 0.0, 0.0];
        sys.r       = vec![1e-7, 0.0, 0.0];
        let test = EnergyIncrement::new(1e-8);
        assert!(test.check(&sys, 1));
    }

    #[test]
    fn not_converged_when_energy_large() {
        let mut sys = tiny_system();
        // 0.5 * |1.0 * 1.0| = 0.5 >> 1e-8
        sys.delta_u = vec![1.0, 0.0, 0.0];
        sys.r       = vec![1.0, 0.0, 0.0];
        let test = EnergyIncrement::new(1e-8);
        assert!(!test.check(&sys, 1));
    }

    #[test]
    fn skips_iter_zero() {
        let sys = tiny_system();
        let test = EnergyIncrement::new(1e-8);
        assert!(!test.check(&sys, 0));
    }

    #[test]
    fn exact_threshold_is_converged() {
        let mut sys = tiny_system();
        // 0.5 * |2.0 * 1.0| = 1.0 exactly
        sys.delta_u = vec![2.0, 0.0, 0.0];
        sys.r       = vec![1.0, 0.0, 0.0];
        let test = EnergyIncrement::new(1.0);
        assert!(test.check(&sys, 1));
    }
}