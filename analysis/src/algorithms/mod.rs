//! Nonlinear equilibrium solution algorithms.
//!
//! An [`EquiSolnAlgo`] is responsible for driving the inner Newton loop:
//! given the current load level (encoded in the residual `R` set by the
//! integrator), it iterates toward a displacement state that satisfies
//! global equilibrium `F_ext = F_int` to within the convergence tolerance.
//!
//! ## Available algorithms
//!
//! | Module | Algorithm | Cost per iteration | When to use |
//! |--------|-----------|--------------------|-------------|
//! | [`linear`] | `LinearAlgorithm` | 1 assemble + 1 solve | Purely elastic, small-deformation |
//! | [`newton`]  | `NewtonRaphson`   | 1 assemble + 1 factorize + 1 solve | General nonlinear |
//! | [`modified`] | `ModifiedNewton` | 1 factorize (first iter) + `n` solves | High iteration count, cheap per iter |
//!
//! ## Design
//!
//! All algorithms share the same [`EquiSolnAlgo`] trait. The driver holds its
//! algorithm as `Box<dyn EquiSolnAlgo>`, so switching from Newton to Modified
//! Newton is a one-line change at construction time.
//!
//! The algorithm receives:
//! - A mutable reference to [`GlobalSystem`] — for writing `K_T`, `R`, `ΔU`.
//! - A mutable reference to [`assembly::Model`] — for assembly and state updates.
//! - A shared reference to a [`solvers::linear::LinearSolver`] — for
//!   factorization and triangular solve.
//!
//! The algorithm must *not* advance the load level — that is the integrator's
//! job. It only drives the inner equilibrium loop for the current load state.

pub mod linear;
pub mod modified;
pub mod newton;

use assembly::Model;
use crate::error::Result;
use crate::system::GlobalSystem;
use crate::integrators::Integrator;
use solvers::linear::LinearSolver;

// -----------------------------------------------------------------
// EquiSolnAlgo trait
// -----------------------------------------------------------------

/// Interface for nonlinear equilibrium solution algorithms.
///
/// Implementors drive the Newton-Raphson (or Newton-like) inner loop for a
/// single load or time step. After this method returns, `model.u_global`
/// should contain the converged displacement state for the current load level.
///
/// # Responsibilities of the implementor
///
/// 1. Loop until convergence or the iteration limit is reached.
/// 2. On each iteration: zero the system, assemble `K_T` and `F_int`,
///    form `R = F_ext - F_int`, apply BCs, factorize, solve for `Δu`.
/// 3. Accumulate `model.u_global += Δu` after each solve.
/// 4. Check convergence via the configured [`crate::tests::ConvergenceTest`].
/// 5. On convergence: call [`assembly::state::commit_state`] and return `Ok(())`.
/// 6. On failure: call [`assembly::state::revert_state`] and return an error.
///
/// # What the algorithm must NOT do
///
/// - Advance the load level (λ or t) — that is the integrator's job.
/// - Allocate inside the loop — use the pre-allocated [`GlobalSystem`] buffers.
/// - Swallow errors from assembly or the solver — propagate them as
///   [`crate::error::AnalysisError`].
pub trait EquiSolnAlgo: Send + Sync {
    /// Solve the nonlinear equilibrium problem for the current load level.
    ///
    /// On entry:
    /// - `system.f_ext` contains the external load for this step (set by the integrator).
    /// - `model.u_global` contains the last converged displacement state.
    ///
    /// On exit (success):
    /// - `model.u_global` contains the converged displacement state.
    /// - All element material histories are committed.
    ///
    /// On exit (error):
    /// - `model.u_global` and material histories are **reverted** to the state
    ///   on entry (the last converged step).
    ///
    /// # Errors
    /// - [`crate::error::AnalysisError::MaxIterationsReached`] — iteration limit hit.
    /// - [`crate::error::AnalysisError::Divergence`] — residual exploded.
    /// - [`crate::error::AnalysisError::SingularSystem`] — Cholesky factorization failed.
    /// - Transparent errors from assembly or the solver.
    fn solve(
        &mut self,
        system:  &mut GlobalSystem,
        model:   &mut Model,
        solver:  &mut dyn LinearSolver<f64>,
        integrator: &dyn Integrator,
        step:    usize,
    ) -> Result<()>;

    /// Human-readable name of this algorithm, used in diagnostic messages.
    fn name(&self) -> &'static str;
}