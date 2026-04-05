//! Nonlinear static analysis driver — pushover and monotonic loading.
//!
//! [`StaticNonlinear`] is the main driver for nonlinear static analysis in
//! Echelon. It orchestrates the integrator (load or displacement control),
//! the algorithm (Newton-Raphson or Modified Newton), and the solver across
//! a user-specified number of load steps.
//!
//! ## Architecture
//!
//! ```text
//! StaticNonlinear {
//!   integrator:  Box<dyn Integrator>    ← LoadControl or DispControl
//!   algorithm:   Box<dyn EquiSolnAlgo>  ← NewtonRaphson or ModifiedNewton
//!   solver:      CholeskySolver<f64>    ← reuses symbolic phase across steps
//!   system:      GlobalSystem           ← pre-allocated buffers (K, R, Δu)
//! }
//! ```
//!
//! ## Execution per load step
//!
//! ```text
//! integrator.new_step(system, model)   ← advance λ, fill system.f_ext
//! algorithm.solve(system, model, solver) ← Newton loop until convergence
//! integrator.commit()                  ← record committed load factor
//! ```
//!
//! ## Error handling
//!
//! When the Newton loop fails to converge on step `k`, the driver:
//! 1. Calls `integrator.revert()` to roll back the load level.
//! 2. Returns `Ok(false)` — a soft failure indicating partial convergence.
//!
//! The caller can inspect `model.u_global` to recover the last converged state.
//!
//! This design matches OpenSees's behaviour: a non-converged step is not
//! necessarily a fatal error in a population run — it may simply mean the
//! structure has reached its capacity.

use assembly::{Model, build_pattern};
use solvers::{CholeskySolver, linear::LinearSolver};

use crate::algorithms::EquiSolnAlgo;
use crate::drivers::AnalysisDriver;
use crate::error::{AnalysisError, Result};
use crate::integrators::Integrator;
use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// StaticNonlinear
// -----------------------------------------------------------------

/// Nonlinear static analysis driver: pushover, monotonic, or cyclic loading.
///
/// Owns the integrator, algorithm, solver, and system buffers. Executes the
/// outer load-step loop and coordinates all components for each step.
///
/// # Construction
///
/// ```rust,ignore
/// use analysis::algorithms::newton::NewtonRaphson;
/// use analysis::integrators::statics::load_control::LoadControl;
/// use analysis::tests::unbalance::NormUnbalance;
/// use analysis::drivers::nonlinear_static::StaticNonlinear;
/// use analysis::drivers::AnalysisDriver;
///
/// // 1. Build convergence test
/// let test = Box::new(NormUnbalance::new(1e-6));
///
/// // 2. Build Newton-Raphson algorithm
/// let algorithm = Box::new(NewtonRaphson::new(test, 25));
///
/// // 3. Build load-controlled integrator: Δλ = 0.1 per step
/// let integrator = Box::new(LoadControl::new(0.1));
///
/// // 4. Assemble the model first, then create the driver
/// let mut driver = StaticNonlinear::new(algorithm, integrator, &model)?;
///
/// // 5. Run 10 steps: λ goes 0.1, 0.2, …, 1.0
/// let ok = driver.analyze(&mut model, 10)?;
/// if ok {
///     println!("All steps converged. u_roof = {:.4e}", model.u_global[control_dof]);
/// } else {
///     println!("Analysis terminated early — structure may have reached capacity.");
/// }
/// ```
pub struct StaticNonlinear {
    /// The equilibrium solution algorithm (Newton-Raphson, Modified Newton, …).
    pub algorithm: Box<dyn EquiSolnAlgo>,

    /// The load or displacement integrator.
    pub integrator: Box<dyn Integrator>,

    /// Sparse Cholesky solver. The symbolic phase is performed once per
    /// topology and reused across all Newton iterations and load steps.
    solver: CholeskySolver<f64>,

    /// Pre-allocated analysis buffers (K_T, R, Δu, F_ext, F_int).
    system: GlobalSystem,
}

impl StaticNonlinear {
    /// Create a new `StaticNonlinear` driver.
    ///
    /// Builds the global stiffness pattern from the model topology and
    /// performs the symbolic Cholesky analysis immediately. The model must
    /// have all nodes, elements, and constraints added before this call.
    ///
    /// # Arguments
    /// * `algorithm`  — the equilibrium solution algorithm.
    /// * `integrator` — the load or displacement integrator.
    /// * `model`      — the fully built model (topology must be final).
    ///
    /// # Errors
    /// - [`AnalysisError::InvalidConfiguration`] if the model has no elements or DOFs.
    /// - Assembly or solver errors from pattern construction.
    pub fn new(
        algorithm:  Box<dyn EquiSolnAlgo>,
        integrator: Box<dyn Integrator>,
        model:      &Model,
    ) -> Result<Self> {
        if model.n_elements() == 0 {
            return Err(AnalysisError::InvalidConfiguration {
                reason: "StaticNonlinear: the model has no elements.".to_string(),
            });
        }
        if model.n_dof() == 0 {
            return Err(AnalysisError::InvalidConfiguration {
                reason: "StaticNonlinear: the model has no DOFs (no nodes).".to_string(),
            });
        }

        let k_pattern = build_pattern(model)?;
        let mut solver = CholeskySolver::new();
        solver.analyze(&k_pattern)?;
        let system = GlobalSystem::new(k_pattern);

        Ok(Self { algorithm, integrator, solver, system })
    }

    /// Return a reference to the current analysis buffers.
    #[inline]
    pub fn system(&self) -> &GlobalSystem {
        &self.system
    }

    /// Return the current load factor from the integrator.
    #[inline]
    pub fn current_lambda(&self) -> f64 {
        self.integrator.current_time()
    }
}

impl AnalysisDriver for StaticNonlinear {
    /// Execute `steps` nonlinear load increments.
    ///
    /// For each step:
    /// 1. The integrator advances the load level and fills `system.f_ext`.
    /// 2. The algorithm runs the Newton-Raphson inner loop.
    /// 3. On convergence: the integrator commits and the loop continues.
    /// 4. On failure: the integrator reverts and the driver returns `Ok(false)`.
    ///
    /// # Returns
    /// - `Ok(true)` — all `steps` converged.
    /// - `Ok(false)` — at least one step failed to converge (soft failure).
    /// - `Err(e)` — an unrecoverable error (assembly or solver fault).
    fn analyze(&mut self, model: &mut Model, steps: usize) -> Result<bool> {
        self.system.check_dof_consistency(model)?;

        for step in 0..steps {
            // ── 1. Advance the integrator ─────────────────────────────────
            self.integrator.new_step(&mut self.system, model)?;

            // ── 2. Run the Newton-Raphson inner loop ─────────────────────
            match self.algorithm.solve(&mut self.system, model, &mut self.solver, step) {
                Ok(()) => {
                    // ── 3. Success: commit the integrator state ──────────
                    self.integrator.commit();
                }
                Err(AnalysisError::MaxIterationsReached { iterations, norm }) => {
                    // Soft failure: Newton did not converge.
                    // The algorithm has already reverted model state.
                    // Revert the integrator load level too.
                    self.integrator.revert();
                    eprintln!(
                        "[analysis] Step {step}: Newton did not converge \
                         ({iterations} iterations, last norm = {norm:.3e}). \
                         Terminating at converged state (λ = {:.4e}).",
                        self.integrator.current_time()
                    );
                    return Ok(false);
                }
                Err(AnalysisError::Divergence { step, norm, threshold }) => {
                    self.integrator.revert();
                    eprintln!(
                        "[analysis] Step {step}: Newton diverged \
                         (norm = {norm:.3e} > threshold = {threshold:.3e}). \
                         Terminating at converged state (λ = {:.4e}).",
                        self.integrator.current_time()
                    );
                    return Ok(false);
                }
                Err(other) => {
                    // Unrecoverable: propagate to caller
                    self.integrator.revert();
                    return Err(other);
                }
            }
        }

        Ok(true)
    }
}