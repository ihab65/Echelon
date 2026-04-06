//! Linear static analysis driver — single-step elastic baseline.
//!
//! [`LinearStatic`] is the simplest possible driver. It executes a single
//! analysis step: assemble `K`, apply loads, apply BCs, factorize, solve.
//! No Newton-Raphson loop, no convergence check, no load stepping.
//!
//! ## When to use
//!
//! - All materials are linear elastic.
//! - Small-displacement assumption holds.
//! - A single load case (gravity, wind, seismic equivalent static).
//!
//! ## Performance
//!
//! Because the stiffness matrix pattern is fixed for a given mesh topology,
//! `LinearStatic` performs the symbolic Cholesky analysis **once** at
//! construction time (or on the first `analyze` call) and only the numeric
//! factorization on each call. For population sampling with a fixed topology
//! this means the expensive ordering step is paid only once across all
//! realisations.
//!
//! ## Multi-step usage
//!
//! Calling `analyze(model, steps)` with `steps > 1` repeats the single-pass
//! solve `steps` times. For elastic structures this is equivalent to calling
//! it once (the solution is the same), but it can be used to apply different
//! load patterns by modifying `model.loads` between driver calls.

use assembly::{
    Model,
    build_pattern,
    assemble_load_vector,
};
use solvers::{CholeskySolver, linear::LinearSolver};

use crate::algorithms::EquiSolnAlgo;
use crate::algorithms::linear::LinearAlgorithm;
use crate::drivers::AnalysisDriver;
use crate::error::{AnalysisError, Result};
use crate::system::GlobalSystem;
use crate::integrators::statics::load_control::LoadControl;

// -----------------------------------------------------------------
// LinearStatic
// -----------------------------------------------------------------

/// Single-step linear static analysis driver.
///
/// Constructs `K`, applies loads and BCs, factorizes with sparse Cholesky,
/// and solves for displacements. The symbolic phase is performed lazily on
/// the first `analyze` call and cached for subsequent calls.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::drivers::linear_static::LinearStatic;
/// use analysis::drivers::AnalysisDriver;
///
/// let mut driver = LinearStatic::new();
/// let ok = driver.analyze(&mut model, 1)?;
/// assert!(ok);
/// // model.u_global now holds the solution
/// ```
pub struct LinearStatic {
    /// The underlying single-pass algorithm.
    algorithm: LinearAlgorithm,

    /// The sparse Cholesky solver — reuses symbolic analysis across calls.
    solver: CholeskySolver<f64>,

    /// Pre-allocated analysis buffers. Constructed on the first call.
    system: Option<GlobalSystem>,

    /// Dummy static integrator — form_tangent is a no-op for linear static.
    integrator:  LoadControl
}

impl LinearStatic {
    /// Create a new `LinearStatic` driver with AMD ordering (default).
    pub fn new() -> Self {
        Self {
            algorithm: LinearAlgorithm,
            solver:    CholeskySolver::new(),
            system:    None,
            integrator: LoadControl::new(1.0), // Δλ = 1.0 for load-controlled
        }
    }
}

impl Default for LinearStatic {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisDriver for LinearStatic {
    /// Execute the linear static solve.
    ///
    /// On the first call, builds the sparsity pattern from the model topology
    /// and performs the symbolic Cholesky analysis. On all subsequent calls
    /// (or if the topology is unchanged), only the numeric factorization is
    /// repeated.
    ///
    /// # Steps
    ///
    /// For each step (typically just 1):
    /// 1. Assemble `f_ext` at `pseudo_time = 1.0` (full load).
    /// 2. [`LinearAlgorithm::solve`]: assemble `K`, form `R`, apply BCs,
    ///    factorize, solve, update `u_global`, commit state.
    ///
    /// # Errors
    /// - [`AnalysisError::InvalidConfiguration`] if the model has no elements
    ///   or no DOFs.
    /// - [`AnalysisError::SingularSystem`] if `K` is singular (missing BCs).
    /// - Transparent assembly and solver errors.
    fn analyze(&mut self, model: &mut Model, steps: usize) -> Result<bool> {
        // Validate model
        if model.n_elements() == 0 {
            return Err(AnalysisError::InvalidConfiguration {
                reason: "LinearStatic: the model has no elements. \
                         Add elements before calling analyze()."
                    .to_string(),
            });
        }
        if model.n_dof() == 0 {
            return Err(AnalysisError::InvalidConfiguration {
                reason: "LinearStatic: the model has no DOFs (no nodes). \
                         Add nodes before calling analyze()."
                    .to_string(),
            });
        }

        // Build system lazily on first call
        if self.system.is_none() {
            let k_pattern = build_pattern(model)?;
            self.solver.analyze(&k_pattern)?;
            self.system = Some(GlobalSystem::new(k_pattern));
        }

        let system = self.system.as_mut().unwrap();
        system.check_dof_consistency(model)?;

        for step in 0..steps {
            // Assemble full load (pseudo_time = 1.0 for load-controlled)
            assemble_load_vector(model, 1.0, &mut system.f_ext)?;

            // Single-pass solve (no Newton loop)
            self.algorithm.solve(system, model, &mut self.solver, &self.integrator, step)?;
        }

        Ok(true)
    }
}