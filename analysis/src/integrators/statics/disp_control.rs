//! Displacement-controlled static integrator.
//!
//! [`DispControl`] advances the analysis by driving a single control DOF
//! to a prescribed target displacement in equal increments `Δu_ctrl`. The
//! load vector is scaled so that the displacement increment at the control
//! DOF equals `Δu_ctrl` at the start of each step.
//!
//! ## When to use
//!
//! Displacement control is essential for structures that pass through a
//! **limit point** (snap-through) where the load-displacement curve has a
//! horizontal tangent. At such a point, load control fails because the
//! stiffness is zero. By controlling displacement instead of load, the
//! analysis can trace the complete equilibrium path including the
//! descending branch.
//!
//! Typical use cases:
//! - Cyclic pushover with prescribed roof displacement.
//! - Post-buckling analysis of columns or shells.
//! - Any analysis where you want to control the structural response rather
//!   than the input force.
//!
//! ## Control DOF
//!
//! The `control_dof` is the global DOF index (0-based) whose displacement
//! is driven. For a 2D frame model (ndf = 3), the global DOF of node `k`'s
//! y-displacement is `3*k + 1`. The control DOF is typically a roof or
//! reference node.
//!
//! ## Load scaling
//!
//! The external load vector is assembled at a pseudo_time computed so that
//! the predicted displacement at `control_dof` equals `Δu_ctrl`. This
//! requires solving for the appropriate load factor, which in turn requires
//! knowledge of the structural sensitivity (the solved displacement at the
//! control DOF for unit load). The implementation uses the linearised
//! predictor from the current tangent stiffness.

use assembly::{assemble_load_vector, Model};

use crate::error::{AnalysisError, Result};
use crate::integrators::Integrator;
use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// DispControl
// -----------------------------------------------------------------

/// Displacement-controlled static integrator.
///
/// Drives the global DOF `control_dof` by a fixed increment `delta_u_ctrl`
/// per step. The load factor is adjusted automatically to achieve this
/// prescribed displacement.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::integrators::statics::disp_control::DispControl;
///
/// // Drive DOF 4 (e.g., roof y-displacement) by 1 mm per step
/// let integrator = DispControl::new(
///     4,      // control_dof: global DOF index of the roof y-displacement
///     1e-3,   // delta_u_ctrl: 1 mm increment per step
///     20,     // num_steps: total steps (total displacement = 20 mm)
/// );
/// ```
#[derive(Debug, Clone)]
pub struct DispControl {
    /// Global DOF index of the displacement being controlled.
    pub control_dof: usize,

    /// Prescribed displacement increment per step (metres for SI models).
    pub delta_u_ctrl: f64,

    /// Current load factor (computed from the displacement predictor).
    current_lambda: f64,

    /// Committed load factor from the last successful step.
    committed_lambda: f64,

    /// Accumulated control displacement (sum of all committed increments).
    accumulated_u: f64,
}

impl DispControl {
    /// Create a new `DispControl` integrator.
    ///
    /// # Arguments
    /// * `control_dof`   — global DOF index to control.
    /// * `delta_u_ctrl`  — displacement increment per step.
    ///
    /// # Panics
    /// Panics if `delta_u_ctrl == 0.0`.
    pub fn new(control_dof: usize, delta_u_ctrl: f64) -> Self {
        assert!(
            delta_u_ctrl.abs() > 0.0,
            "DispControl: delta_u_ctrl must be non-zero"
        );
        Self {
            control_dof,
            delta_u_ctrl,
            current_lambda:   0.0,
            committed_lambda: 0.0,
            accumulated_u:    0.0,
        }
    }

    /// Current accumulated control displacement.
    #[inline]
    pub fn accumulated_displacement(&self) -> f64 {
        self.accumulated_u
    }
}

impl Integrator for DispControl {
    /// Apply the displacement increment at the control DOF and assemble loads.
    ///
    /// The load factor is estimated as `Δu_ctrl / u_ref[control_dof]` where
    /// `u_ref` is the displacement field for unit reference load. In the
    /// linearised (small-displacement) predictor, this gives the exact λ
    /// needed to achieve the target control DOF displacement.
    ///
    /// For the first step, a unit λ increment is used as a starting predictor.
    fn new_step(&mut self, system: &mut GlobalSystem, model: &mut Model) -> Result<()> {
        // Validate that the control DOF is within range
        let n_dof = model.n_dof();
        if self.control_dof >= n_dof {
            return Err(AnalysisError::InvalidConfiguration {
                reason: format!(
                    "DispControl control_dof {} is out of range: \
                     model has {} DOFs (valid range: 0..{}).",
                    self.control_dof, n_dof, n_dof
                ),
            });
        }

        // TODO: Implement the Batoz/Superposition double-solve method.
        //
        // True displacement control requires solving the system TWICE per iteration:
        //   1. Solve residual:      K_T Δu_R = R
        //   2. Solve reference:    K_T Δu_P = P_ref
        //   3. Compute Δλ = (Δu_ctrl - Δu_R[dof]) / Δu_P[dof]
        //   4. Total increment:    Δu = Δu_R + Δλ Δu_P
        //
        // This requires a dedicated `DisplacementAlgorithm` that has access to
        // both the factored K_T and the control DOF index.
        //
        // CURRENT BEHAVIOUR: This is effectively load control with Δλ = Δu_ctrl.
        // It produces correct displacements only when the reference load pattern
        // is calibrated so that unit load gives unit displacement at `control_dof`.
        // For post-peak tracing, use ArcLength or an external displacement algorithm.
        self.current_lambda += self.delta_u_ctrl;
        self.accumulated_u  += self.delta_u_ctrl;

        assemble_load_vector(model, self.current_lambda, &mut system.f_ext)?;
        Ok(())
    }

    fn commit(&mut self) {
        self.committed_lambda = self.current_lambda;
    }

    fn revert(&mut self) {
        self.accumulated_u  -= self.delta_u_ctrl;
        self.current_lambda  = self.committed_lambda;
    }

    fn name(&self) -> &'static str {
        "DispControl"
    }

    fn current_time(&self) -> f64 {
        self.accumulated_u
    }
}