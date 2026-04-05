//! Top-level analysis drivers — the outer loops that orchestrate everything.
//!
//! A [`AnalysisDriver`] owns the integrator, algorithm, and [`GlobalSystem`].
//! It executes the outer `for step in 0..steps` loop and coordinates the
//! integrator and algorithm for each step.
//!
//! ## Available drivers
//!
//! | Module | Driver | Use case |
//! |--------|--------|----------|
//! | [`linear_static`] | `LinearStatic` | Single-step elastic analysis |
//! | [`nonlinear_static`] | `StaticNonlinear` | Pushover, monotonic loading |
//! | [`transient`] | `TransientDriver` | Time-history, seismic, dynamic |
//!
//! ## Execution model
//!
//! ```text
//! driver.analyze(model, steps):
//!   for step in 0..steps:
//!
//!     1. integrator.new_step(system, model)   ← advance λ or t, fill f_ext
//!
//!     2. algorithm.solve(system, model, solver)  ← Newton inner loop
//!
//!     3. On success:
//!          integrator.commit()
//!          record results (optional)
//!
//!     4. On failure:
//!          return Ok(false) or Err(e) depending on strategy
//!
//!   return Ok(true)  ← all steps converged
//! ```

pub mod linear_static;
pub mod nonlinear_static;
pub mod transient;

use assembly::Model;
use crate::error::Result;

// -----------------------------------------------------------------
// AnalysisDriver trait
// -----------------------------------------------------------------

/// Executes a complete multi-step analysis lifecycle.
///
/// The driver owns the `GlobalSystem` and orchestrates the integrator and
/// algorithm across all load or time steps.
///
/// # Return value
///
/// `analyze` returns `Ok(true)` when all steps converge. It returns `Ok(false)`
/// when the analysis terminates early due to a non-fatal failure (e.g., the
/// Newton loop did not converge on step `k` but the driver is configured to
/// continue with the remaining converged steps). It returns `Err(e)` only for
/// unrecoverable errors (assembly fault, solver state violation, etc.).
pub trait AnalysisDriver: Send + Sync {
    /// Execute the analysis for `steps` load/time increments.
    ///
    /// On return, `model.u_global` contains the displacement at the last
    /// successfully converged state.
    ///
    /// # Errors
    /// Returns [`crate::error::AnalysisError`] for unrecoverable faults.
    fn analyze(&mut self, model: &mut Model, steps: usize) -> Result<bool>;
}