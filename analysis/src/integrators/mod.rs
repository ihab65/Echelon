//! Load stepping and time stepping schemes — the Integrator layer.
//!
//! An [`Integrator`] controls how the analysis advances from one converged
//! state to the next. In statics it controls the load parameter λ; in
//! dynamics it controls the time step Δt and augments the system with
//! inertia and damping contributions.
//!
//! ## Available integrators
//!
//! ### Static
//!
//! | Module | Integrator | Controls | Use case |
//! |--------|-----------|----------|----------|
//! | [`statics::load_control`] | `LoadControl` | Load factor λ | Monotonic pushover, gravity |
//! | [`statics::disp_control`] | `DispControl` | DOF displacement | Post-peak, snap-through |
//!
//! ### Transient
//!
//! | Module | Integrator | Method | Use case |
//! |--------|-----------|--------|----------|
//! | [`transient::newmark`] | `Newmark` | Newmark-β | General time-history, seismic |
//! | [`transient::hht`] | `HHT` | Hilber-Hughes-Taylor-α | Seismic with numerical damping |
//!
//! ## Role in the analysis lifecycle
//!
//! ```text
//! for each load/time step:
//!
//!   integrator.new_step(system, model)?     ← advance λ or t, set f_ext
//!
//!   algorithm.solve(system, model, solver)  ← Newton inner loop
//!
//!   integrator.commit()                     ← save converged integrator state
//! ```
//!
//! The integrator is responsible for populating `system.f_ext` before the
//! algorithm runs. For load control this is simply scaling the reference
//! load by the new λ. For transient analysis it includes the effective force
//! from inertia and damping: `F_eff = F_ext − M·ü − C·u̇`.

pub mod statics;
pub mod transient;

use assembly::Model;
use crate::error::Result;
use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// Integrator trait
// -----------------------------------------------------------------

/// Controls how the analysis advances from one equilibrium state to the next.
///
/// The integrator is called **once per load/time step** by the driver, before
/// the algorithm's Newton loop. It is responsible for:
///
/// 1. Advancing the internal state (λ, t, ü_prev, u̇_prev).
/// 2. Populating `system.f_ext` with the effective external force for this step.
/// 3. After successful convergence, saving the converged state via `commit`.
///
/// # Contract
///
/// - `new_step` is always called before `algorithm.solve`.
/// - `commit` is called only if `algorithm.solve` returns `Ok(())`.
/// - If the algorithm returns an error, `commit` is **not** called —
///   the integrator should remain at its pre-step state so the driver
///   can retry with a reduced step size.
pub trait Integrator: Send + Sync {
    /// Advance to the next step and populate `system.f_ext`.
    ///
    /// For load-controlled statics: increments λ and assembles the scaled
    /// external load vector into `system.f_ext`.
    ///
    /// For transient analysis: advances t by Δt, computes the effective load
    /// vector (including inertia and damping terms), and writes it to
    /// `system.f_ext`.
    ///
    /// # Errors
    /// Returns [`crate::error::AnalysisError::InvalidConfiguration`] if the
    /// integrator is in an invalid state (e.g., called before initialization).
    fn new_step(&mut self, system: &mut GlobalSystem, model: &mut Model) -> Result<()>;

    /// Save the converged integrator state after a successful Newton loop.
    ///
    /// For load control: records the current λ as the committed load level.
    /// For Newmark/HHT: saves velocity and acceleration for the next step.
    ///
    /// This is a no-op for stateless integrators like `LoadControl`.
    fn commit(&mut self);

    /// Revert the integrator to its state before the last `new_step` call.
    ///
    /// Called when the Newton loop fails and the driver wants to retry with a
    /// smaller step. For load control: decrements λ back to the previous value.
    fn revert(&mut self);

    /// Human-readable name of this integrator, used in diagnostic messages.
    fn name(&self) -> &'static str;

    /// Current load factor or pseudo-time (for diagnostic reporting).
    fn current_time(&self) -> f64;
}