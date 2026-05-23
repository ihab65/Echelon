//! Modified Newton-Raphson equilibrium solution algorithm.
//!
//! [`ModifiedNewton`] forms and factorizes the tangent stiffness matrix `K_T`
//! only **once** at the start of each load step, then reuses the same
//! factorization for every subsequent Newton iteration within that step.
//! Only the residual `R = F_ext − F_int` is updated on each iteration.
//!
//! ## Comparison with standard Newton-Raphson
//!
//! | Property | `NewtonRaphson` | `ModifiedNewton` |
//! |----------|-----------------|------------------|
//! | Tangent updates | Every iteration | Once per step |
//! | Convergence rate | Quadratic | Linear |
//! | Cost per iteration | Assemble + Factorize + Solve | Assemble + Solve |
//! | Iterations to converge | 3–10 | 10–50 |
//! | Best for | Small models, high nonlinearity | Large models, mild nonlinearity |
//!
//! The crossover point depends on the ratio of factorization cost to
//! solve cost. For models with thousands of DOFs and many elements, modified
//! Newton can be significantly faster despite requiring more iterations,
//! because triangular solve is much cheaper than sparse Cholesky factorization.
//!
//! ## Stiffness evaluation point
//!
//! The default is to evaluate `K_T` at the **beginning of the step**
//! (the last converged state). An alternative is to evaluate it at a
//! midpoint (secant stiffness), which can improve convergence without the
//! full cost of tangent updates.
//!
//! ## Algorithm
//!
//! ```text
//! // Once per load step:
//! assemble K_T(u_committed)         — stiffness at the committed state
//! apply Dirichlet BCs to K_T
//! factorize K_T                     — the expensive step, done only ONCE
//!
//! // Per Newton iteration k:
//! for k = 0, 1, 2, …:
//!   assemble F_int(u_k)             — internal forces at current u
//!   form R_k = F_ext − F_int
//!   apply Dirichlet BCs to R_k
//!   solve K_T Δu = R_k             — cheap: reuse the factor
//!   u_{k+1} = u_k + Δu
//!   check convergence
//!
//! commit_state(model)
//! ```

use assembly::{
    assemble_stiffness,
    assemble_internal_force,
    apply_dirichlet_bcs,
    state::{commit_state, revert_state},
};
use solvers::{linear::LinearSolver, error::SolverError};

use crate::algorithms::EquiSolnAlgo;
use crate::error::{AnalysisError, Result};
use crate::system::GlobalSystem;
use crate::integrators::Integrator;
use crate::convergence::ConvergenceTest;
use assembly::Model;

// -----------------------------------------------------------------
// ModifiedNewton
// -----------------------------------------------------------------

/// Modified Newton-Raphson: re-forms `K_T` once per step, reuses factorization.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::algorithms::modified::ModifiedNewton;
/// use analysis::tests::unbalance::NormUnbalance;
///
/// let test = NormUnbalance::new(1e-4);   // looser tol for more iters
/// let algo = ModifiedNewton::new(test, 50);         // allow more iterations
/// ```
pub struct ModifiedNewton {
    /// The convergence criterion.
    pub test: Box<dyn ConvergenceTest>,

    /// Maximum number of Newton iterations per load step.
    pub max_iterations: usize,

    /// Residual norm threshold above which the iteration is declared diverged.
    pub divergence_threshold: f64,
}

impl ModifiedNewton {
    /// Create a new `ModifiedNewton` with the given convergence test and
    /// iteration limit.
    ///
    /// # Panics
    /// Panics if `max_iterations == 0`.
    pub fn new<T: ConvergenceTest + 'static>(test: T, max_iterations: usize) -> Self {
        assert!(max_iterations > 0, "max_iterations must be at least 1");
        Self {
            test: Box::new(test),
            max_iterations,
            divergence_threshold: 1e15,
        }
    }
}

impl EquiSolnAlgo for ModifiedNewton {
    /// Run the modified Newton loop for the current load level.
    ///
    /// The tangent stiffness `K_T` is assembled and factorized **once**
    /// at the start of the step (at the current `model.u_global`, which
    /// holds the last committed state). Subsequent iterations only re-assemble
    /// `F_int` and update `R`, reusing the cached factorization.
    fn solve(
        &mut self,
        system: &mut GlobalSystem,
        model:  &mut Model,
        solver: &mut dyn LinearSolver<f64>,
        integrator: &dyn Integrator,
        step:   usize,
    ) -> Result<()> {
        // ── Form and factorize K_T once ───────────────────────────────────
        system.zero_out();
        assemble_stiffness(model, &mut system.k_t)
            .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;
        // Augment with inertia/damping terms (no-op for static integrators).
        integrator.form_tangent(system)
            .map_err(|e| { revert_state(model); e })?;
        apply_dirichlet_bcs(&model.constraints, &mut system.k_t, &mut system.r)
            .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;

        solver.factorize(&system.k_t).map_err(|e| {
            revert_state(model);
            match e {
                SolverError::NotPositiveDefinite { .. } => {
                    AnalysisError::SingularSystem { step, iteration: 0 }
                }
                other => AnalysisError::from(other),
            }
        })?;

        // ── Inner iteration loop ──────────────────────────────────────────
        for iter in 0..self.max_iterations {
            // Only F_int and R change each iteration; K_T and its factor are fixed.
            system.f_int.fill(0.0);
            system.r.fill(0.0);
            system.delta_u.fill(0.0);

            assemble_internal_force(model, &mut system.f_int)
                .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;

            system.form_residual();

            // Zero constrained DOFs in R (K_T is already BC-applied)
            for c in &model.constraints {
                system.r[c.global_dof] = 0.0;
            }

            // Divergence guard
            let norm = system.residual_norm();
            if iter > 0 && norm > self.divergence_threshold {
                revert_state(model);
                return Err(AnalysisError::Divergence {
                    step,
                    norm,
                    threshold: self.divergence_threshold,
                });
            }

            solver.solve(&system.r, &mut system.delta_u)
                .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;

            for (u, &du) in model.u_global.iter_mut().zip(system.delta_u.iter()) {
                *u += du;
            }

            if self.test.check(system, iter) {
                commit_state(model)
                    .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;
                return Ok(());
            }
        }

        let last_norm = system.residual_norm();
        revert_state(model);
        Err(AnalysisError::MaxIterationsReached {
            iterations: self.max_iterations,
            norm:       last_norm,
        })
    }

    fn name(&self) -> &'static str {
        "ModifiedNewton"
    }
}