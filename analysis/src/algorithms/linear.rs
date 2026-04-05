//! Single-pass linear algorithm for purely elastic analysis.
//!
//! [`LinearAlgorithm`] bypasses the Newton-Raphson loop entirely and solves
//! the system in exactly one pass: assemble `K` and `R` once, factorize,
//! solve, update, commit. It is correct *only* for small-displacement
//! linear elastic models where `K_T = K` is constant (independent of `u`).
//!
//! ## When to use
//!
//! - All materials are linear elastic (constant tangent modulus).
//! - Deformations are small enough that geometric nonlinearity is negligible.
//! - No yielding, cracking, contact, or other path-dependent behaviour.
//!
//! In these conditions, the Newton-Raphson loop converges in exactly one
//! iteration anyway — `LinearAlgorithm` skips the iteration machinery
//! and checks, making it slightly faster and conceptually clearer.
//!
//! ## When NOT to use
//!
//! Do not use `LinearAlgorithm` with:
//! - Nonlinear materials (plasticity, softening, hysteresis).
//! - Large-deformation / corotational elements.
//! - Geometric stiffness (P-Δ, buckling analysis).
//!
//! In these cases, use [`crate::algorithms::newton::NewtonRaphson`] or
//! [`crate::algorithms::modified::ModifiedNewton`].

use assembly::{
    assemble_stiffness,
    assemble_internal_force,
    apply_dirichlet_bcs,
    state::commit_state,
};
use solvers::linear::LinearSolver;

use crate::algorithms::EquiSolnAlgo;
use crate::error::{AnalysisError, Result};
use crate::system::GlobalSystem;
use assembly::Model;

// -----------------------------------------------------------------
// LinearAlgorithm
// -----------------------------------------------------------------

/// Single-pass solver for purely linear elastic structural analysis.
///
/// Assembles `K`, `F_int`, forms `R = F_ext − F_int`, applies BCs,
/// factorizes, solves, updates displacements, and commits — all in
/// one pass with no convergence checking.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::algorithms::linear::LinearAlgorithm;
/// use analysis::drivers::linear_static::LinearStatic;
///
/// // LinearStatic uses LinearAlgorithm internally — you usually
/// // do not need to construct it directly.
/// let algo = LinearAlgorithm;
/// ```
pub struct LinearAlgorithm;

impl EquiSolnAlgo for LinearAlgorithm {
    /// Solve the linear elastic system in a single pass.
    ///
    /// Sequence:
    /// 1. Zero system buffers.
    /// 2. Assemble `K_T` (= `K` for linear elastic).
    /// 3. Assemble `F_int` (= 0 at zero displacement for linear elastic,
    ///    but computed correctly for incremental loading with non-zero `u_global`).
    /// 4. Form `R = F_ext − F_int`.
    /// 5. Apply Dirichlet BCs.
    /// 6. Factorize `K_T`.
    /// 7. Solve `K_T Δu = R` → `system.delta_u`.
    /// 8. Update `model.u_global += Δu`.
    /// 9. Commit state (advances material histories, no-op for elastic materials).
    ///
    /// # Errors
    /// - [`AnalysisError::SingularSystem`] if `K` is singular (missing BCs).
    /// - Transparent errors from assembly or the solver.
    fn solve(
        &mut self,
        system: &mut GlobalSystem,
        model:  &mut Model,
        solver: &mut dyn LinearSolver<f64>,
        step:   usize,
    ) -> Result<()> {
        system.zero_out();

        assemble_stiffness(model, &mut system.k_t)?;
        assemble_internal_force(model, &mut system.f_int)?;
        system.form_residual();
        apply_dirichlet_bcs(&model.constraints, &mut system.k_t, &mut system.r)?;

        solver.factorize(&system.k_t).map_err(|e| {
            use solvers::error::SolverError;
            match e {
                SolverError::NotPositiveDefinite { .. } => {
                    AnalysisError::SingularSystem { step, iteration: 0 }
                }
                other => AnalysisError::from(other),
            }
        })?;

        solver.solve(&system.r, &mut system.delta_u)?;

        for (u, &du) in model.u_global.iter_mut().zip(system.delta_u.iter()) {
            *u += du;
        }

        commit_state(model)?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "LinearAlgorithm"
    }
}