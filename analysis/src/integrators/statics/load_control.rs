//! Load-controlled static integrator.
//!
//! [`LoadControl`] advances the analysis by incrementing the global load
//! factor λ by a fixed amount `Δλ` at each step. The external force vector
//! is assembled at `pseudo_time = current_lambda` by calling
//! [`assembly::assemble_load_vector`].
//!
//! ## When to use
//!
//! Load control is appropriate for:
//! - Gravity loading (monotonically increasing from 0 to 1).
//! - Monotonic pushover analyses on structures with a stable ascending
//!   load-displacement curve.
//! - Any problem where the structure does not pass through a limit point
//!   (snap-through or snap-back).
//!
//! ## Limitations
//!
//! Load control cannot trace a post-peak descending branch. At a limit point
//! the load increment cannot be satisfied at any finite displacement, and
//! Newton-Raphson diverges. For post-peak analysis, switch to
//! [`crate::integrators::statics::disp_control::DispControl`] or an
//! arc-length method.
//!
//! ## Load factor semantics
//!
//! `current_lambda` is passed as `pseudo_time` to
//! [`assembly::assemble_load_vector`]. Every [`assembly::loads::pattern::LoadPattern`]
//! in the model evaluates its `TimeSeries` at `pseudo_time` to determine the
//! scale factor for its reference loads.
//!
//! A `ConstantSeries` ignores `pseudo_time` entirely, making the load fully
//! applied from step 1. A `LinearSeries` ramps from 0 to 1 as `pseudo_time`
//! goes from 0 to 1, making it ideal for a load-controlled pushover with
//! `delta_lambda = 0.1` over 10 steps.

use assembly::{assemble_load_vector, Model};

use crate::error::Result;
use crate::integrators::Integrator;
use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// LoadControl
// -----------------------------------------------------------------

/// Load-controlled static integrator: increments the load factor λ by `Δλ`.
///
/// # Example — 10-step pushover from 0 to full load
///
/// ```rust,ignore
/// use analysis::integrators::statics::load_control::LoadControl;
///
/// // 10 equal steps: λ goes 0.1, 0.2, …, 1.0
/// let integrator = LoadControl::new(0.1);
/// ```
///
/// # Example — gravity preload (0 → 1 in a single step)
///
/// ```rust,ignore
/// let integrator = LoadControl::new(1.0);
/// // Then analyze for 1 step.
/// ```
#[derive(Debug, Clone)]
pub struct LoadControl {
    /// The load factor increment applied at each step: `Δλ`.
    pub delta_lambda: f64,

    /// The current (post-increment) load factor. Starts at 0.
    ///
    /// After `new_step()` is called, this equals the load factor for the
    /// current step. After `revert()`, it returns to its pre-step value.
    current_lambda: f64,

    /// The committed load factor from the last successful step.
    ///
    /// Used by `revert()` to roll back after a failed Newton loop.
    committed_lambda: f64,
}

impl LoadControl {
    /// Create a new `LoadControl` integrator starting at λ = 0.
    ///
    /// # Arguments
    /// * `delta_lambda` — load factor increment per step.
    ///
    /// # Panics
    /// Panics if `delta_lambda ≤ 0`. Zero or negative increments would
    /// produce a degenerate analysis (no load applied, or load reduction).
    pub fn new(delta_lambda: f64) -> Self {
        assert!(delta_lambda > 0.0, "LoadControl: delta_lambda must be positive");
        Self {
            delta_lambda,
            current_lambda:   0.0,
            committed_lambda: 0.0,
        }
    }

    /// Create a `LoadControl` resuming from an existing committed load level.
    ///
    /// Useful when chaining analyses (e.g., continuing a pushover from a
    /// previously saved state rather than starting from zero).
    pub fn starting_from(delta_lambda: f64, committed_lambda: f64) -> Self {
        assert!(delta_lambda > 0.0, "LoadControl: delta_lambda must be positive");
        Self {
            delta_lambda,
            current_lambda:   committed_lambda,
            committed_lambda,
        }
    }

    /// The current load factor (after the most recent `new_step` call).
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.current_lambda
    }

    /// The committed load factor from the last successful step.
    #[inline]
    pub fn committed_lambda(&self) -> f64 {
        self.committed_lambda
    }
}

impl Integrator for LoadControl {
    /// Increment λ by `Δλ` and assemble the external load vector.
    ///
    /// 1. Advances `current_lambda += delta_lambda`.
    /// 2. Calls [`assemble_load_vector`] at `pseudo_time = current_lambda`,
    ///    writing the scaled external loads into `system.f_ext`.
    ///
    /// # Errors
    /// Propagates any error from [`assemble_load_vector`] (rare in practice —
    /// the function returns `Result` for future extensibility).
    fn new_step(&mut self, system: &mut GlobalSystem, model: &mut Model) -> Result<()> {
        self.current_lambda += self.delta_lambda;
        assemble_load_vector(model, self.current_lambda, &mut system.f_ext)?;
        Ok(())
    }

    /// Record `current_lambda` as the new committed state.
    ///
    /// Called by the driver after a successful Newton convergence.
    fn commit(&mut self) {
        self.committed_lambda = self.current_lambda;
    }

    /// Roll back to the last committed load level.
    ///
    /// Called when the Newton loop fails. After revert, `new_step` can be
    /// called again with a smaller `delta_lambda` (if the driver supports
    /// adaptive stepping).
    fn revert(&mut self) {
        self.current_lambda = self.committed_lambda;
    }

    fn name(&self) -> &'static str {
        "LoadControl"
    }

    fn current_time(&self) -> f64 {
        self.current_lambda
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_lambda_is_zero() {
        let lc = LoadControl::new(0.1);
        assert_eq!(lc.lambda(), 0.0);
        assert_eq!(lc.committed_lambda(), 0.0);
    }

    #[test]
    fn commit_advances_committed_lambda() {
        let mut lc = LoadControl::new(0.25);
        // Simulate new_step without a real model
        lc.current_lambda += lc.delta_lambda;
        lc.commit();
        assert!((lc.committed_lambda() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn revert_restores_lambda() {
        let mut lc = LoadControl::new(0.1);
        lc.current_lambda += lc.delta_lambda;
        assert!((lc.lambda() - 0.1).abs() < 1e-15);
        lc.revert();
        assert_eq!(lc.lambda(), 0.0);
    }

    #[test]
    fn starting_from_sets_base() {
        let lc = LoadControl::starting_from(0.1, 0.5);
        assert!((lc.committed_lambda() - 0.5).abs() < 1e-15);
        assert!((lc.current_time() - 0.5).abs() < 1e-15);
    }

    #[test]
    #[should_panic]
    fn zero_delta_lambda_panics() {
        let _ = LoadControl::new(0.0);
    }

    #[test]
    #[should_panic]
    fn negative_delta_lambda_panics() {
        let _ = LoadControl::new(-0.1);
    }
}