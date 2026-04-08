//! Standard Newton-Raphson equilibrium solution algorithm.
//!
//! [`NewtonRaphson`] is the workhorse algorithm for nonlinear structural
//! analysis. At each iteration it re-forms the tangent stiffness matrix
//! from the current displacement state and uses it to compute the next
//! displacement increment.
//!
//! ## Algorithm
//!
//! Given the current displacement state `u_k` and external load `F_ext`:
//!
//! ```text
//! for k = 0, 1, 2, … (up to max_iterations):
//!
//!   1. Assemble K_T(u_k)          — tangent stiffness at current state
//!   2. Assemble F_int(u_k)        — internal resisting forces
//!   3. Form R_k = F_ext − F_int   — unbalanced force (residual)
//!   4. Apply Dirichlet BCs        — zero constrained rows/cols of K_T and R
//!   5. Factorize K_T              — sparse Cholesky L Lᵀ = K_T
//!   6. Solve  K_T Δu = R_k        — triangular solve for displacement increment
//!   7. Update u_{k+1} = u_k + Δu
//!   8. Check convergence(R_k, Δu) — if converged, commit state and exit
//!
//! commit_state(model)             — advance material histories
//! ```
//!
//! ## Convergence properties
//!
//! Standard Newton-Raphson exhibits **quadratic convergence** near the exact
//! solution when the tangent stiffness is exact. For smooth elastic problems
//! this means 3–5 iterations suffice. For highly nonlinear problems (large
//! yield zones, softening) more iterations are needed near the limit load.
//!
//! ## When to prefer Modified Newton
//!
//! Re-forming and factorizing `K_T` every iteration is expensive (it dominates
//! the cost for large models). If the problem is mildly nonlinear and the
//! iteration count per step is already low (< 5), standard Newton is optimal.
//! If the model has many elements and convergence is slow, try
//! [`crate::algorithms::modified::ModifiedNewton`] which re-uses the stiffness
//! factorization within a step.
//!
//! ## Symbolic / numeric phase reuse
//!
//! The solver's **symbolic phase** (sparsity pattern analysis, fill-reduction
//! ordering) is performed only **once per topology change** by the driver.
//! The algorithm only calls `solver.factorize()` (numeric phase) on each
//! iteration. This makes the per-iteration cost dominated by the numeric
//! factorization, not the expensive symbolic ordering.

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
// NewtonRaphson
// -----------------------------------------------------------------

/// Standard Newton-Raphson algorithm for nonlinear equilibrium.
///
/// Re-forms the tangent stiffness matrix `K_T` and factorizes it on every
/// Newton iteration. Achieves quadratic convergence near the exact solution.
///
/// # Construction
///
/// ```rust,ignore
/// use analysis::algorithms::newton::NewtonRaphson;
/// use analysis::tests::unbalance::NormUnbalance;
///
/// let test = Box::new(NormUnbalance::new(1e-6));
/// let nr   = NewtonRaphson::new(test, 25);
/// ```
///
/// # Field documentation
///
/// | Field | Description |
/// |-------|-------------|
/// | `test` | The convergence criterion (norm of residual, Δu, or energy) |
/// | `max_iterations` | Maximum Newton steps before declaring failure |
/// | `divergence_threshold` | Residual norm above which the iteration is considered diverged |
pub struct NewtonRaphson {
    /// The convergence criterion used to decide when the inner loop exits.
    ///
    /// Stored as a boxed trait object so the caller can hot-swap criteria
    /// at construction time without touching the algorithm's logic.
    pub test: Box<dyn ConvergenceTest>,

    /// Maximum number of Newton iterations per load step.
    ///
    /// If convergence is not achieved within this many iterations, the
    /// algorithm reverts all material states and returns
    /// [`AnalysisError::MaxIterationsReached`].
    ///
    /// Typical values: 10–25 for mildly nonlinear, 50–100 for highly
    /// nonlinear or post-peak analyses.
    pub max_iterations: usize,

    /// Residual norm threshold above which the iteration is declared diverged.
    ///
    /// If `‖R_k‖ > divergence_threshold` at any iteration after the first,
    /// the algorithm immediately reverts and returns
    /// [`AnalysisError::Divergence`] without exhausting the iteration budget.
    ///
    /// Default: `1e15` (effectively disabled — divergence is detected by
    /// `MaxIterationsReached` instead). Set lower for early-exit in
    /// population sampling loops where divergent realisations should be
    /// discarded quickly.
    pub divergence_threshold: f64,
}

impl NewtonRaphson {
    /// Create a new `NewtonRaphson` with the given convergence test and
    /// iteration limit. Uses the default divergence threshold (`1e15`).
    ///
    /// # Arguments
    /// * `test`           — the convergence criterion.
    /// * `max_iterations` — iteration budget per load step.
    ///
    /// # Panics
    /// Panics if `max_iterations == 0`.
    pub fn new<T: ConvergenceTest + 'static>(test: T, max_iterations: usize) -> Self {
        assert!(max_iterations > 0, "max_iterations must be at least 1");
        Self {
            test : Box::new(test),
            max_iterations,
            divergence_threshold: 1e15,
        }
    }

    /// Create a `NewtonRaphson` with an explicit divergence threshold.
    ///
    /// The algorithm will return [`AnalysisError::Divergence`] early if the
    /// residual norm exceeds `divergence_threshold` at any iteration after
    /// the first. This is useful in population runs to avoid spending the
    /// full iteration budget on clearly divergent realisations.
    ///
    /// # Panics
    /// Panics if `max_iterations == 0` or `divergence_threshold ≤ 0`.
    pub fn with_divergence_threshold(
        test:                 Box<dyn ConvergenceTest>,
        max_iterations:       usize,
        divergence_threshold: f64,
    ) -> Self {
        assert!(max_iterations > 0, "max_iterations must be at least 1");
        assert!(divergence_threshold > 0.0, "divergence_threshold must be positive");
        Self { test, max_iterations, divergence_threshold }
    }
}

impl EquiSolnAlgo for NewtonRaphson {
    /// Run the Newton-Raphson loop for the current load level.
    ///
    /// ## Step-by-step execution
    ///
    /// For each iteration `k` from 0 to `max_iterations − 1`:
    ///
    /// 1. **Zero** `system.k_t`, `system.r`, `system.delta_u`, `system.f_int`
    ///    via [`GlobalSystem::zero_out`].
    /// 2. **Assemble** `K_T` from all elements at the current `model.u_global`.
    /// 3. **Assemble** `F_int` from all elements at the current `model.u_global`.
    /// 4. **Form residual** `R = F_ext − F_int` in `system.r`.
    /// 5. **Apply Dirichlet BCs** — zero constrained rows/cols of `K_T`
    ///    and zero corresponding entries of `R`.
    /// 6. **Check divergence** — if `‖R‖ > divergence_threshold` abort early.
    /// 7. **Factorize** `K_T` (numeric Cholesky, reuses symbolic pattern).
    /// 8. **Solve** `K_T Δu = R` → `system.delta_u`.
    /// 9. **Update** `model.u_global += system.delta_u`.
    /// 10. **Check convergence** via `self.test.check(system, k)`.
    ///     - If converged: call `commit_state` and return `Ok(())`.
    ///     - If not: continue.
    ///
    /// If the loop exhausts `max_iterations` without convergence:
    /// - `revert_state` is called on the model.
    /// - `AnalysisError::MaxIterationsReached` is returned.
    ///
    /// ## Error recovery
    ///
    /// On any error (assembly, solver, divergence), `revert_state` is called
    /// before returning the error so the model is always left in the last
    /// committed (valid) state.
    fn solve(
        &mut self,
        system: &mut GlobalSystem,
        model:  &mut Model,
        solver: &mut dyn LinearSolver<f64>,
        integrator: &dyn Integrator,
        step:   usize,
    ) -> Result<()> {
        for iter in 0..self.max_iterations {
            // ── 1. Zero buffers (preserves f_ext) ────────────────────
            system.zero_out();

            // ── 2. Assemble tangent stiffness K_T(u_k) ───────────────
            assemble_stiffness(model, &mut system.k_t)
                .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;

            // ── 2b. Augment K_T with inertia/damping (no-op for statics)
            integrator.form_tangent(system)
                .map_err(|e| { revert_state(model); e })?;

            // ── 3. Assemble internal force F_int(u_k) ────────────────
            assemble_internal_force(model, &mut system.f_int)
                .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;

            // ── 4. Form residual R = F_ext − F_int ───────────────────
            system.form_residual();

            // ── 5. Apply Dirichlet boundary conditions ────────────────
            apply_dirichlet_bcs(&model.constraints, &mut system.k_t, &mut system.r)
                .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;

            // ── 6. Divergence guard ───────────────────────────────────
            let norm = system.residual_norm();
            if iter > 0 && norm > self.divergence_threshold {
                revert_state(model);
                return Err(AnalysisError::Divergence {
                    step,
                    norm,
                    threshold: self.divergence_threshold,
                });
            }

            // ── 7. Numeric factorization K_T = L Lᵀ ──────────────────
            solver.factorize(&system.k_t).map_err(|e| {
                revert_state(model);
                match e {
                    SolverError::NotPositiveDefinite { .. } => {
                        AnalysisError::SingularSystem { step, iteration: iter }
                    }
                    other => AnalysisError::from(other),
                }
            })?;

            // ── 8. Triangular solve  K_T Δu = R ──────────────────────
            solver.solve(&system.r, &mut system.delta_u)
                .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;

            // ── 9. Update displacements  u ← u + Δu ──────────────────
            for (u, &du) in model.u_global.iter_mut().zip(system.delta_u.iter()) {
                *u += du;
            }

            // ── 10. Convergence check ─────────────────────────────────
            if self.test.check(system, iter) {
                commit_state(model)
                    .map_err(|e| { revert_state(model); AnalysisError::from(e) })?;
                return Ok(());
            }
        }

        // Exhausted iteration budget without convergence.
        let last_norm = system.residual_norm();
        revert_state(model);
        Err(AnalysisError::MaxIterationsReached {
            iterations: self.max_iterations,
            norm:       last_norm,
        })
    }

    fn name(&self) -> &'static str {
        "NewtonRaphson"
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convergence::unbalance::NormUnbalance;

    #[test]
    #[should_panic]
    fn zero_max_iterations_panics() {
        let test = NormUnbalance::new(1e-6);
        let _ = NewtonRaphson::new(test, 0);
    }

    #[test]
    fn name_is_correct() {
        let nr = NewtonRaphson::new(NormUnbalance::new(1e-6), 10);
        assert_eq!(nr.name(), "NewtonRaphson");
    }

    #[test]
    fn default_divergence_threshold_is_large() {
        let nr = NewtonRaphson::new(NormUnbalance::new(1e-6), 10);
        assert!(nr.divergence_threshold > 1e10);
    }

    #[test]
    fn with_divergence_threshold_stores_value() {
        let nr = NewtonRaphson::with_divergence_threshold(
            Box::new(NormUnbalance::new(1e-6)),
            20,
            1e8,
        );
        assert!((nr.divergence_threshold - 1e8).abs() < 1.0);
        assert_eq!(nr.max_iterations, 20);
    }
}