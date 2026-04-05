//! Convergence test based on the Euclidean norm of the residual force vector.
//!
//! [`NormUnbalance`] is the most common convergence criterion in structural FEA.
//! It measures the magnitude of the out-of-balance forces at the current
//! Newton iterate and declares convergence when that magnitude drops below
//! an absolute or relative tolerance.
//!
//! ## Formula
//!
//! At Newton iteration `k`, convergence is declared when:
//!
//! ```text
//! ‖R_k‖₂ ≤ ε_abs
//! ```
//!
//! or (if a reference norm `‖R_0‖₂` is used for relative checking):
//!
//! ```text
//! ‖R_k‖₂ / ‖R_0‖₂ ≤ ε_rel
//! ```
//!
//! ## Physical interpretation
//!
//! The residual `R = F_ext − F_int` is the net unbalanced force at each DOF.
//! When `‖R‖₂` is small, the structure is near a state of equilibrium: the
//! forces the elements are resisting internally (`F_int`) match the externally
//! applied forces (`F_ext`) to within the tolerance.
//!
//! ## Choosing a tolerance
//!
//! - For steel structures with forces in Newtons: `1e-3` to `1e-6` N is typical.
//! - For normalised (dimensionless) systems: `1e-8` to `1e-12`.
//! - When in doubt, use `NormUnbalance::new(1e-6)` and tighten if the
//!   displacement increment norm is still large.

use crate::system::GlobalSystem;
use crate::convergence::ConvergenceTest;

// -----------------------------------------------------------------
// NormUnbalance
// -----------------------------------------------------------------

/// Convergence criterion: `‖R‖₂ ≤ tolerance`.
///
/// Checks that the Euclidean norm of the residual (unbalanced force) vector
/// is below the configured absolute tolerance.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::tests::unbalance::NormUnbalance;
///
/// let test = NormUnbalance::new(1e-6);  // converge when ‖R‖ < 1e-6 N
/// ```
#[derive(Debug, Clone)]
pub struct NormUnbalance {
    /// Absolute convergence tolerance on `‖R‖₂`.
    ///
    /// The Newton loop is declared converged when the Euclidean norm of
    /// the residual vector drops below this value.
    pub tolerance: f64,
}

impl NormUnbalance {
    /// Create a new `NormUnbalance` criterion with the given absolute tolerance.
    ///
    /// # Panics
    /// Panics if `tolerance ≤ 0.0`.
    pub fn new(tolerance: f64) -> Self {
        assert!(tolerance > 0.0, "NormUnbalance tolerance must be positive");
        Self { tolerance }
    }
}

impl ConvergenceTest for NormUnbalance {
    /// Return `true` if `‖R‖₂ ≤ self.tolerance`.
    ///
    /// The check is skipped on iteration 0 (before the first solve) to avoid
    /// spurious convergence on the initial zero state.
    fn check(&self, system: &GlobalSystem, iter: usize) -> bool {
        if iter == 0 {
            // The first residual (before any Δu has been applied) is the
            // initial out-of-balance. Skip the check: it may be zero if
            // u_global starts at the exact solution, but in general it is
            // the unscaled external load — which is almost never below tol.
            return false;
        }
        system.residual_norm() <= self.tolerance
    }

    fn name(&self) -> &'static str {
        "NormUnbalance"
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
    fn converged_when_residual_small() {
        let mut sys = tiny_system();
        sys.r = vec![1e-8, 0.0, 0.0];
        let test = NormUnbalance::new(1e-6);
        assert!(test.check(&sys, 1));
    }

    #[test]
    fn not_converged_when_residual_large() {
        let mut sys = tiny_system();
        sys.r = vec![1.0, 0.0, 0.0];
        let test = NormUnbalance::new(1e-6);
        assert!(!test.check(&sys, 1));
    }

    #[test]
    fn skips_check_on_iter_zero() {
        // Even with zero residual, iter=0 must return false
        let sys = tiny_system(); // r is all zero
        let test = NormUnbalance::new(1e-6);
        assert!(!test.check(&sys, 0));
    }

    #[test]
    #[should_panic]
    fn panics_on_zero_tolerance() {
        let _ = NormUnbalance::new(0.0);
    }
}