//! [`AdjointSensitive`] — Engine B hook for history-dependent materials.
//!
//! ## Why not just thread `Dual64` through everything?
//!
//! For smooth elastic materials, threading `Dual64` through the `energy<T>`
//! function is exactly right: it gives exact, machine-precision gradients at
//! zero extra implementation cost.
//!
//! For history-dependent materials (Steel01, Concrete01, Bouc-Wen, etc.) the
//! situation is fundamentally different.  These materials are implemented
//! through return-mapping algorithms: their stress is determined by a
//! sequence of conditionals and iterative projections back onto the yield
//! surface.  If you thread `Dual64` through this algorithm, you get the
//! derivative of the *algorithmic* stress map — which is only correct at
//! smooth points far from the yield surface, and wrong (or undefined) at
//! the kink.
//!
//! The adjoint method avoids this problem entirely.  At a converged solution:
//!
//! ```text
//! K_T λ = ∂J/∂u                    (one linear solve — K_T already factored)
//! dJ/dθ = -λᵀ (∂f_int/∂θ)          (assembled from element contributions)
//! ∂f_int/∂θ = B^T · (∂σ/∂θ)        (element geometry × material sensitivity)
//! ```
//!
//! The term `∂σ/∂θ` is the derivative of the *physically correct committed
//! stress* with respect to a material parameter `θ` — not the algorithmic
//! tangent.  For plasticity models this is the derivative of the yield
//! criterion with respect to the parameter, evaluated at the committed
//! plastic state.  It is analytic and exact.
//!
//! ## Implementing this trait
//!
//! Implement [`AdjointSensitive`] on any [`crate::UniaxialMaterial`] that you want
//! to include in Engine B sensitivity analysis.  The trait has a single method:
//! `stress_sensitivity`, which returns `∂σ/∂θ_i` at the **committed** strain.
//!
//! The parameter index `i` is material-model-specific.  Each model documents
//! which index corresponds to which parameter.

use crate::error::Result;

/// Extension trait for materials that can provide exact parameter sensitivities
/// for use in the adjoint method (Engine B).
///
/// # Contract
///
/// `stress_sensitivity` is called **after** Engine A has converged and
/// `commit_state` has been called.  The sensitivity must be evaluated at
/// the *committed* strain state, not a trial state.
///
/// Implementations are free to cache the committed stress gradient during
/// `commit_state` so that `stress_sensitivity` is O(1).
pub trait AdjointSensitive {
    /// Return `∂σ/∂θ_i` at the current committed strain.
    ///
    /// # Arguments
    /// * `param_idx` — index of the material parameter.
    ///
    /// # Parameter index convention
    ///
    /// Each material implementation documents its own parameter ordering.
    /// The `MaterialParam` enum in `materials/mod.rs` provides named
    /// constants for common parameters (`E`, `Fy`, `fc28`, etc.).
    ///
    /// # Panics
    /// May panic if `param_idx` is out of range for this material.
    fn stress_sensitivity(&self, param_idx: usize) -> Result<f64>;

    /// Number of parameters this material exposes for sensitivity analysis.
    fn n_params(&self) -> usize;

    /// Human-readable name for parameter `i`.  Used in output and diagnostics.
    fn param_name(&self, param_idx: usize) -> &'static str;
}
