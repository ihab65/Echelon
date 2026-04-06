//! Post-processing recorders — capture simulation state at every converged step.
//!
//! A [`Recorder`] is notified after every successful Newton convergence via
//! [`Recorder::record`]. It receives the current `pseudo_time` and a
//! read-only reference to the model, from which it can extract any result
//! quantity and store it in an internal buffer for later retrieval.
//!
//! ## Design
//!
//! Recorders are owned by the analysis driver as `Vec<Box<dyn Recorder>>`.
//! They are triggered in the `Ok(())` branch of the driver's step loop,
//! after `commit_state` has been called. This guarantees they always see
//! the converged, committed state — never a trial state.
//!
//! ## Available recorders
//!
//! | Type | Records |
//! |------|---------|
//! | [`NodeRecorder`] | Displacement DOFs at specified nodes |
//! | [`ElementRecorder`] | Internal forces (N, V, M) at element ends |
//!
//! ## Usage example
//!
//! ```rust,ignore
//! use analysis::recorder::{NodeRecorder, ElementRecorder};
//!
//! let mut driver = StaticNonlinear::new(algo, integrator, &model)?;
//!
//! // Record roof displacement (node 5, DOF 1 = UY)
//! driver.add_recorder(Box::new(NodeRecorder::new(5, vec![1])));
//!
//! // Record internal forces at element 0
//! driver.add_recorder(Box::new(ElementRecorder::new(0)));
//!
//! driver.analyze(&mut model, 100)?;
//!
//! let roof_disp = driver.recorder::<NodeRecorder>(0).unwrap().data();
//! ```

pub mod node;
pub mod element;

pub use node::NodeRecorder;
pub use element::ElementRecorder;

use assembly::Model;
use std::any::Any;

// -----------------------------------------------------------------
// Recorder trait
// -----------------------------------------------------------------

/// Observer interface for capturing simulation results at each converged step.
///
/// Implementors accumulate results in an internal `Vec` or similar structure.
/// Call the appropriate accessor after `driver.analyze()` returns to retrieve
/// the recorded history.
pub trait Recorder: Send + Sync {
    /// Called by the driver after every successful Newton convergence.
    ///
    /// # Arguments
    /// * `pseudo_time` — the current load factor (static) or simulation time (dynamic)
    /// * `model`       — the structural model at the converged state
    fn record(&mut self, pseudo_time: f64, model: &Model);

    /// Human-readable description of what this recorder captures.
    fn description(&self) -> String;
    /// For downcasting to concrete recorder types after analysis.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}