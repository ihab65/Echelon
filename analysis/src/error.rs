//! Analysis-level error types.
//!
//! Every error in the `analysis` crate carries:
//! - A dot-separated diagnostic code (`echelon::analysis::…`) for programmatic
//!   matching in population sampling scripts.
//! - A `help` message that identifies the most likely *structural* cause of the
//!   failure, not just the numerical symptom.
//! - Structured context fields so that a population runner can log, filter, and
//!   triage realisations without parsing string messages.
//!
//! ## Error taxonomy
//!
//! ```text
//! AnalysisError
//! ├── MaxIterationsReached  ← Newton loop hit the wall
//! ├── Divergence            ← residual exploded (ill-conditioned or unstable)
//! ├── SingularSystem        ← Cholesky failed (mechanism or zero stiffness)
//! ├── StepTooSmall          ← arc-length / adaptive step refused to proceed
//! ├── InvalidConfiguration  ← logic error in driver construction
//! ├── Assembly(…)           ← propagated from the `assembly` crate
//! └── Solver(…)             ← propagated from the `solvers` crate
//! ```

use miette::Diagnostic;
use thiserror::Error;

use assembly::error::AssemblyError;
use solvers::error::SolverError;

/// Errors that can arise during an Echelon analysis run.
///
/// These errors represent both convergence failures and logic faults.
/// They are designed to be caught and inspected in population-scale
/// Monte Carlo loops: a `MaxIterationsReached` is a soft failure
/// (log and continue), while a `SingularSystem` signals a kinematically
/// inadmissible realisation that should be discarded.
///
/// All variants are `non_exhaustive` so that adding new error kinds in
/// future crate versions does not break existing match arms.
#[derive(Debug, Error, Diagnostic)]
#[non_exhaustive]
pub enum AnalysisError {
    // ---- Convergence failures --------------------------------------------

    /// The Newton-Raphson inner loop exhausted its iteration budget without
    /// satisfying any convergence criterion.
    ///
    /// The `iterations` field records how many Newton steps were taken.
    /// The `norm` field records the last observed convergence measure
    /// (residual norm, displacement norm, or energy increment, depending on
    /// the `ConvergenceTest` in use).
    #[error(
        "Newton-Raphson failed to converge: \
         {iterations} iteration(s) completed without satisfying the \
         convergence criterion (last norm = {norm:.6e})."
    )]
    #[diagnostic(
        code(echelon::analysis::newton::max_iterations_reached),
        help(
            "Possible remedies (in order of effort):\n\
             (1) Increase `max_iterations` — 25 is a conservative default; \
                 softening structures sometimes need 50–100.\n\
             (2) Reduce the load increment Δλ — a smaller step is easier to \
                 converge, especially near a yield plateau or snap-through.\n\
             (3) Switch from `NewtonRaphson` to `ModifiedNewton` — re-forming \
                 K_T every iteration is expensive; a stale tangent is acceptable \
                 far from the singularity.\n\
             (4) Check the convergence tolerance — `NormUnbalance(1e-6)` may be \
                 too tight for a softening material; try `1e-4` first.\n\
             (5) Verify that Dirichlet BCs are applied correctly: a missing \
                 restraint causes the residual to drift without bound."
        )
    )]
    MaxIterationsReached {
        /// Number of Newton iterations attempted.
        iterations: usize,
        /// Last convergence measure observed before giving up.
        norm: f64,
    },

    /// The residual norm grew beyond the divergence threshold during Newton
    /// iteration, indicating the solver is moving away from equilibrium.
    ///
    /// Divergence typically signals:
    /// - A load increment so large that the linearised system is invalid.
    /// - A structural instability (limit point reached, snap-through).
    /// - A material model in tension softening without regularisation.
    #[error(
        "Newton-Raphson diverged at load step {step}: \
         residual norm grew to {norm:.6e} (divergence threshold: {threshold:.6e})."
    )]
    #[diagnostic(
        code(echelon::analysis::newton::divergence),
        help(
            "Divergence means the Newton update is moving away from equilibrium. \n\
             (1) Halve the load increment Δλ and retry — if the structure is \
                 near a limit point, smaller steps are essential.\n\
             (2) Inspect the deformed shape at the last converged step: \
                 are any members in tension softening or snap-through?\n\
             (3) For post-peak analysis, switch to displacement control \
                 (`DispControl`) or arc-length method instead of load control.\n\
             (4) If this occurs at step 1, verify that gravity loads are not \
                 exceeding the elastic capacity in a single increment."
        )
    )]
    Divergence {
        /// Load step index at which divergence was detected.
        step: usize,
        /// Residual norm at the point of divergence.
        norm: f64,
        /// Threshold above which the norm was declared diverged.
        threshold: f64,
    },

    // ---- Solver failures -------------------------------------------------

    /// The sparse Cholesky factorization failed because the tangent stiffness
    /// matrix is not positive-definite.
    ///
    /// This wraps [`SolverError::NotPositiveDefinite`] with the additional
    /// context of which analysis step and Newton iteration triggered the fault.
    #[error(
        "Tangent stiffness matrix is singular or indefinite at load step {step}, \
         Newton iteration {iteration}. \
         The Cholesky factorization encountered a non-positive pivot."
    )]
    #[diagnostic(
        code(echelon::analysis::system::singular_stiffness),
        help(
            "A non-positive pivot in the Cholesky factor indicates one of:\n\
             (1) Kinematic instability — the structure is a mechanism because \
                 one or more Dirichlet BCs are missing. For a 2D frame, at \
                 minimum fix UX, UY at one node and UY at another non-colinear node.\n\
             (2) Material instability — the tangent modulus of a nonlinear \
                 material has gone negative (post-peak softening). The structure \
                 has passed its limit load.\n\
             (3) Near-zero element stiffness — an element has E ≤ 0, A ≤ 0, \
                 or Iz ≤ 0. Check the parameter sample that produced this realisation.\n\
             In Monte Carlo runs, log this realisation as 'kinematically unstable' \
             or 'limit-load exceeded' and continue sampling."
        )
    )]
    SingularSystem {
        /// Load step index when singularity was detected.
        step: usize,
        /// Newton iteration index when singularity was detected.
        iteration: usize,
    },

    // ---- Step size control -----------------------------------------------

    /// The adaptive step size controller could not find a step small enough
    /// to achieve convergence, or the minimum allowed step size was reached.
    ///
    /// This is typically triggered by the arc-length method or an adaptive
    /// load step controller when repeated halving has reduced the step below
    /// the minimum threshold.
    #[error(
        "Analysis step {step} aborted: step size {current_step:.6e} is below \
         the minimum allowed value {min_step:.6e} without achieving convergence."
    )]
    #[diagnostic(
        code(echelon::analysis::integrator::step_too_small),
        help(
            "The step-size controller could not find a step small enough to converge.\n\
             (1) Lower the `min_step` threshold if physically meaningful — \
                 near a snap-through, steps of 1e-6 × the reference load may be required.\n\
             (2) Increase the maximum number of step halvings.\n\
             (3) Consider switching to displacement control or arc-length \
                 to navigate past limit points.\n\
             (4) If this occurs in a population run, flag the realisation as \
                 'snap-through / highly nonlinear' for separate treatment."
        )
    )]
    StepTooSmall {
        /// Load step index.
        step: usize,
        /// Step size at the point of failure.
        current_step: f64,
        /// Minimum step size the controller is allowed to use.
        min_step: f64,
    },

    // ---- Logic / configuration errors ------------------------------------

    /// The driver, algorithm, or integrator was constructed or called in a
    /// state that is logically invalid.
    ///
    /// Examples:
    /// - `StaticNonlinear::new()` called with `num_dofs = 0`.
    /// - `analyze()` called before the model has any elements.
    /// - An integrator's `commit()` called before `form_unbalance()`.
    #[error("Invalid analysis configuration: {reason}")]
    #[diagnostic(
        code(echelon::analysis::config::invalid),
        help(
            "This is a programming error in the analysis setup, not a \
             structural failure. Review the driver construction code and \
             ensure all preconditions are met before calling `analyze()`."
        )
    )]
    InvalidConfiguration {
        /// Human-readable description of the configuration fault.
        reason: String,
    },

    // ---- Transparent passthroughs ----------------------------------------

    /// A fault propagated from the `assembly` crate (model topology, scatter
    /// operations, load application).
    #[error(transparent)]
    #[diagnostic(transparent)]
    Assembly(#[from] AssemblyError),

    /// A fault propagated from the `solvers` crate (Cholesky symbolic/numeric
    /// phase, state sequencing violation, RHS size mismatch).
    #[error(transparent)]
    #[diagnostic(transparent)]
    Solver(#[from] SolverError),
}

/// Alias for `Result<T, AnalysisError>`.
///
/// Returned by all fallible functions in the `analysis` crate.
pub type Result<T> = std::result::Result<T, AnalysisError>;