//! Transient (dynamic) analysis driver — time-history analysis.
//!
//! [`TransientDriver`] orchestrates a Newmark or HHT time integrator with
//! a Newton-Raphson algorithm to solve the equations of motion:
//!
//! ```text
//! M ü(t) + C u̇(t) + K_T(u) u(t) = F_ext(t)
//! ```
//!
//! at each time step. The driver is structurally identical to
//! [`crate::drivers::nonlinear_static::StaticNonlinear`] but holds a
//! transient integrator (Newmark or HHT) instead of a static one.
//!
//! ## Typical workflow
//!
//! ```rust,ignore
//! use analysis::algorithms::newton::NewtonRaphson;
//! use analysis::integrators::transient::newmark::Newmark;
//! use analysis::tests::unbalance::NormUnbalance;
//! use analysis::drivers::transient::TransientDriver;
//! use analysis::drivers::AnalysisDriver;
//! use assembly::assemble_mass;
//!
//! // Assemble mass matrix from model
//! let mut mass_mat = build_pattern(&model)?;
//! assemble_mass(&model, &mut mass_mat)?;
//!
//! // Build Newmark integrator (average acceleration, no damping)
//! let integrator = Box::new(Newmark::average_acceleration(0.01, mass_mat, None));
//!
//! // Build algorithm
//! let test      = Box::new(NormUnbalance::new(1e-4));
//! let algorithm = Box::new(NewtonRaphson::new(test, 25));
//!
//! // Create driver and run for 100 time steps (1 second at dt=0.01)
//! let mut driver = TransientDriver::new(algorithm, integrator, &model)?;
//! let ok = driver.analyze(&mut model, 100)?;
//! ```

use assembly::{Model, build_pattern};
use solvers::{CholeskySolver, linear::LinearSolver};

use crate::algorithms::EquiSolnAlgo;
use crate::drivers::AnalysisDriver;
use crate::error::{AnalysisError, Result};
use crate::integrators::Integrator;
use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// TransientDriver
// -----------------------------------------------------------------

/// Dynamic time-history analysis driver.
///
/// Owns a transient integrator (Newmark or HHT), an equilibrium algorithm,
/// and the pre-allocated system buffers. Executes the outer time-step loop.
pub struct TransientDriver {
    /// The equilibrium solution algorithm (Newton-Raphson recommended).
    pub algorithm: Box<dyn EquiSolnAlgo>,

    /// The transient integrator (Newmark or HHT).
    pub integrator: Box<dyn Integrator>,

    /// Sparse Cholesky solver. The symbolic phase is performed once per
    /// topology and reused across all time steps and Newton iterations.
    solver: CholeskySolver<f64>,

    /// Pre-allocated analysis buffers.
    system: GlobalSystem,

    /// Post-processing recorders — triggered after each converged step.
    recorders: Vec<Box<dyn crate::recorder::Recorder>>,
}

impl TransientDriver {
    /// Create a new `TransientDriver`.
    ///
    /// Builds the stiffness pattern and performs the symbolic Cholesky
    /// analysis immediately. The model topology must be final at this point.
    ///
    /// # Errors
    /// - [`AnalysisError::InvalidConfiguration`] if model has no elements or DOFs.
    /// - Assembly or solver errors from pattern construction.
    pub fn new(
        algorithm:  Box<dyn EquiSolnAlgo>,
        integrator: Box<dyn Integrator>,
        model:      &Model,
    ) -> Result<Self> {
        if model.n_elements() == 0 {
            return Err(AnalysisError::InvalidConfiguration {
                reason: "TransientDriver: model has no elements.".to_string(),
            });
        }
        if model.n_dof() == 0 {
            return Err(AnalysisError::InvalidConfiguration {
                reason: "TransientDriver: model has no DOFs.".to_string(),
            });
        }

        let k_pattern = build_pattern(model)?;
        let mut solver = CholeskySolver::new();
        solver.analyze(&k_pattern)?;
        let system = GlobalSystem::new(k_pattern);
        let recorders = Vec::new();

        Ok(Self { algorithm, integrator, solver, system, recorders })
    }

    /// Current simulation time.
    #[inline]
    pub fn current_time(&self) -> f64 {
        self.integrator.current_time()
    }

    /// Register a recorder to be triggered after each converged step.
    pub fn add_recorder(&mut self, recorder: Box<dyn crate::recorder::Recorder>) {
        self.recorders.push(recorder);
    }

    /// Access a recorder by index (for retrieving results after analysis).
    pub fn recorder(&self, index: usize) -> Option<&dyn crate::recorder::Recorder> {
        self.recorders.get(index).map(|r| r.as_ref())
    }

    /// Access a recorder by index, downcast to the concrete type `R`.
    ///
    /// Returns `None` if the index is out of range or the type doesn't match.
    pub fn recorder_as<R: crate::recorder::Recorder + 'static>(&self, index: usize)
        -> Option<&R>
    {
        self.recorders.get(index)?.as_any().downcast_ref::<R>()
    }
}

impl AnalysisDriver for TransientDriver {
    /// Execute `steps` time steps.
    ///
    /// Each step advances the integrator by Δt, runs Newton-Raphson to
    /// dynamic equilibrium, and commits the state. Soft failures
    /// (non-convergence) return `Ok(false)` with the model left at the
    /// last committed state.
    ///
    /// # Errors
    /// - [`AnalysisError::InvalidConfiguration`] if system buffers are inconsistently sized.
    /// - [`AnalysisError::SolverError`] if matrix factorization fails.
    fn analyze(&mut self, model: &mut Model, steps: usize) -> Result<bool> {
        self.system.check_dof_consistency(model)?;

        for step in 0..steps {
            self.integrator.new_step(&mut self.system, model)?;

            match self.algorithm.solve(
                &mut self.system, 
                model, 
                &mut self.solver, 
                self.integrator.as_ref(),
                step
            ) {
                Ok(()) => {
                    self.integrator.commit();
                    // Trigger all recorders at the committed state
                    let t = self.integrator.current_time();
                    for rec in &mut self.recorders {
                        rec.record(t, model);
                    }
                }
                Err(AnalysisError::MaxIterationsReached { iterations, norm }) => {
                    self.integrator.revert();
                    eprintln!(
                        "[transient] t = {:.4e}: Newton did not converge \
                         ({iterations} iterations, last norm = {norm:.3e}).",
                        self.integrator.current_time()
                    );
                    return Ok(false);
                }
                Err(AnalysisError::Divergence { step, norm, threshold }) => {
                    self.integrator.revert();
                    eprintln!(
                        "[transient] Step {step}: diverged \
                         (norm={norm:.3e} > {threshold:.3e})."
                    );
                    return Ok(false);
                }
                Err(other) => {
                    self.integrator.revert();
                    return Err(other);
                }
            }
        }

        Ok(true)
    }
}