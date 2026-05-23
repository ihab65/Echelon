//! Convergence criteria for the Newton-Raphson inner loop.
//!
//! A [`ConvergenceTest`] answers one question per Newton iteration:
//! **"Has the system reached equilibrium?"**
//!
//! The test inspects the current state of the [`GlobalSystem`] — the residual
//! vector `r`, the displacement increment `Δu`, and the energy increment
//! `0.5 Δu·R` — and returns `true` when a norm-based criterion is satisfied.
//!
//! ## Available criteria
//!
//! | Module | Criterion | Formula | Notes |
//! |--------|-----------|---------|-------|
//! | [`unbalance`] | `NormUnbalance` | `‖R‖₂ ≤ tol` | Fast; good for force-controlled |
//! | [`displacement`] | `NormDispIncr` | `‖Δu‖₂ ≤ tol` | Good for soft materials |
//! | [`energy`] | `EnergyIncrement` | `0.5 ‖Δu·R‖ ≤ tol` | Gold standard; dimensionally consistent |
//!
//! ## Choosing a criterion
//!
//! **`NormUnbalance`** is the simplest and most widely used. It is suitable
//! for most structural analyses and corresponds to the physical requirement
//! that the out-of-balance forces are negligible.
//!
//! **`NormDispIncr`** is useful when the force residual is small but
//! displacements are still evolving (e.g., softening materials with low
//! post-peak stiffness).
//!
//! **`EnergyIncrement`** is the most rigorous: it checks virtual work and
//! is insensitive to the scaling of force and displacement units. It is the
//! preferred criterion for highly nonlinear or multi-physics analyses.
//!
//! ## Composing criteria
//!
//! For safety-critical applications, use [`AndTest`] to require that
//! multiple criteria are satisfied simultaneously, or [`OrTest`] to
//! accept convergence when any one is met.

pub mod displacement;
pub mod energy;
pub mod unbalance;

use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// ConvergenceTest trait
// -----------------------------------------------------------------

/// Determines whether the Newton-Raphson inner loop has converged.
///
/// Implementors inspect the current [`GlobalSystem`] and return `true` when
/// the relevant norm has dropped below the configured tolerance.
///
/// The `iter` argument is the **0-based** Newton iteration count. It can be
/// used to skip the convergence check on the first iteration (before the
/// first solve) or to apply stricter tolerances as iterations accumulate.
///
/// # Contract
///
/// - If `check` returns `true`, the algorithm **must** call
///   [`assembly::state::commit_state`] and exit its inner loop.
/// - If `check` returns `false`, the algorithm continues iterating.
/// - The test **must not** mutate the `GlobalSystem`.
/// - The test **must** be stateless: identical inputs always produce the
///   same output.
pub trait ConvergenceTest: Send + Sync {
    /// Return `true` if the system has converged.
    ///
    /// # Arguments
    /// * `system` — current analysis buffers (read-only).
    /// * `iter`   — 0-based Newton iteration index.
    fn check(&self, system: &GlobalSystem, iter: usize) -> bool;

    /// Human-readable name of this criterion, used in diagnostic messages.
    fn name(&self) -> &'static str;
}

// -----------------------------------------------------------------
// AndTest — both criteria must be satisfied
// -----------------------------------------------------------------

/// Composite test: requires **all** inner criteria to be satisfied.
///
/// This is the most conservative option — useful when you want to guarantee
/// both small residuals and small displacement increments before declaring
/// convergence.
///
/// # Example
///
/// ```rust,ignore
/// use crate::convergence::{AndTest, ConvergenceTest};
/// use crate::convergence::unbalance::NormUnbalance;
/// use crate::convergence::displacement::NormDispIncr;
///
/// let test = AndTest::new(vec![
///     Box::new(NormUnbalance::new(1e-6)),
///     Box::new(NormDispIncr::new(1e-8)),
/// ]);
/// ```
pub struct AndTest {
    inner: Vec<Box<dyn ConvergenceTest>>,
}

impl AndTest {
    /// Create a new `AndTest` wrapping the given criteria.
    ///
    /// # Panics
    /// Panics if `inner` is empty (a vacuously true convergence test would
    /// terminate Newton-Raphson on the first iteration regardless of state).
    pub fn new(inner: Vec<Box<dyn ConvergenceTest>>) -> Self {
        assert!(!inner.is_empty(), "AndTest requires at least one inner criterion");
        Self { inner }
    }
}

impl ConvergenceTest for AndTest {
    fn check(&self, system: &GlobalSystem, iter: usize) -> bool {
        self.inner.iter().all(|t| t.check(system, iter))
    }

    fn name(&self) -> &'static str {
        "AndTest"
    }
}

// -----------------------------------------------------------------
// OrTest — any one criterion is sufficient
// -----------------------------------------------------------------

/// Composite test: converged when **any** inner criterion is satisfied.
///
/// This is the most lenient option — useful when you trust any one of
/// multiple independent indicators.
pub struct OrTest {
    inner: Vec<Box<dyn ConvergenceTest>>,
}

impl OrTest {
    /// Create a new `OrTest` wrapping the given criteria.
    ///
    /// # Panics
    /// Panics if `inner` is empty.
    pub fn new(inner: Vec<Box<dyn ConvergenceTest>>) -> Self {
        assert!(!inner.is_empty(), "OrTest requires at least one inner criterion");
        Self { inner }
    }
}

impl ConvergenceTest for OrTest {
    fn check(&self, system: &GlobalSystem, iter: usize) -> bool {
        self.inner.iter().any(|t| t.check(system, iter))
    }

    fn name(&self) -> &'static str {
        "OrTest"
    }
}