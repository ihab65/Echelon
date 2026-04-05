//! Convergence test based on the Euclidean norm of the displacement increment.
//!
//! [`NormDispIncr`] measures how much the displacement field is still
//! changing between Newton iterations. When `‖Δu‖₂` is small, the iterates
//! have stabilised — the solver is no longer making meaningful corrections.
//!
//! ## Formula
//!
//! Convergence is declared at iteration `k` when:
//!
//! ```text
//! ‖Δu_k‖₂ ≤ ε
//! ```
//!
//! ## When to use this criterion
//!
//! `NormDispIncr` complements `NormUnbalance` for softening structures.
//! Near a limit point, the tangent stiffness becomes very small, so even
//! a large displacement increment produces a very small force residual. In
//! this regime, `NormUnbalance` may declare premature convergence while
//! displacements are still evolving. `NormDispIncr` catches this.
//!
//! Combine with [`crate::tests::unbalance::NormUnbalance`] via
//! [`crate::tests::AndTest`] for the most reliable stopping criterion.

use crate::system::GlobalSystem;
use crate::convergence::ConvergenceTest;

// -----------------------------------------------------------------
// NormDispIncr
// -----------------------------------------------------------------

/// Convergence criterion: `‖Δu‖₂ ≤ tolerance`.
///
/// Checks that the Euclidean norm of the displacement increment vector
/// is below the configured absolute tolerance.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::tests::displacement::NormDispIncr;
///
/// // Converge when incremental displacements are smaller than 1 micrometre
/// let test = NormDispIncr::new(1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct NormDispIncr {
    /// Absolute convergence tolerance on `‖Δu‖₂`.
    pub tolerance: f64,
}

impl NormDispIncr {
    /// Create a new `NormDispIncr` criterion with the given absolute tolerance.
    ///
    /// # Panics
    /// Panics if `tolerance ≤ 0.0`.
    pub fn new(tolerance: f64) -> Self {
        assert!(tolerance > 0.0, "NormDispIncr tolerance must be positive");
        Self { tolerance }
    }
}

impl ConvergenceTest for NormDispIncr {
    /// Return `true` if `‖Δu‖₂ ≤ self.tolerance`.
    ///
    /// Unlike `NormUnbalance`, this test is meaningful even on iteration 0:
    /// a zero initial displacement increment is a valid (trivially converged)
    /// state. However, in practice the first `Δu` from the initial solve is
    /// non-trivial, so `iter == 0` skipping is not needed here.
    fn check(&self, system: &GlobalSystem, iter: usize) -> bool {
        if iter == 0 {
            return false;
        }
        system.delta_u_norm() <= self.tolerance
    }

    fn name(&self) -> &'static str {
        "NormDispIncr"
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
    fn converged_when_delta_u_small() {
        let mut sys = tiny_system();
        sys.delta_u = vec![1e-9, 0.0, 0.0];
        let test = NormDispIncr::new(1e-6);
        assert!(test.check(&sys, 1));
    }

    #[test]
    fn not_converged_when_delta_u_large() {
        let mut sys = tiny_system();
        sys.delta_u = vec![0.01, 0.0, 0.0];
        let test = NormDispIncr::new(1e-6);
        assert!(!test.check(&sys, 1));
    }

    #[test]
    fn skips_iter_zero() {
        let sys = tiny_system();
        let test = NormDispIncr::new(1e-6);
        assert!(!test.check(&sys, 0));
    }
}