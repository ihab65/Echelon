//! Pre-allocated analysis buffers — the memory core of every Newton iteration.
//!
//! [`GlobalSystem`] holds the three large vectors that every Newton-Raphson
//! iteration writes into:
//!
//! - **`k_t`** — the global tangent stiffness matrix (symmetric CSR).
//! - **`r`** — the global residual (unbalanced force) vector.
//! - **`delta_u`** — the incremental displacement vector solved from `K_T Δu = R`.
//!
//! ## Why a dedicated struct?
//!
//! In structural FEA the three most expensive operations per Newton step are:
//!
//! 1. **Assembly** — writing into `k_t` and `r` from element contributions.
//! 2. **Factorization** — the Cholesky factor of `k_t`.
//! 3. **Triangular solve** — back-substitution to obtain `delta_u`.
//!
//! None of these *need* to allocate. If the stiffness matrix pattern is fixed
//! (which it is for a given mesh topology), the only work per iteration is
//! *filling pre-existing memory with new values*. Allocating inside the loop
//! would be a gratuitous performance regression.
//!
//! `GlobalSystem` is constructed **once per analysis** from the mesh topology.
//! The algorithm and integrator receive mutable references to the same struct
//! and zero it at the start of each Newton step with [`GlobalSystem::zero_out`].
//!
//! ## Memory layout
//!
//! ```text
//! GlobalSystem {
//!   k_t:     SymCsrMatrix<f64>  — n_dof × n_dof (upper triangle, CSR)
//!   r:       Vec<f64>           — length n_dof
//!   delta_u: Vec<f64>           — length n_dof
//!   f_ext:   Vec<f64>           — length n_dof  (external force, reused across Newton iters)
//!   f_int:   Vec<f64>           — length n_dof  (internal force, reused across Newton iters)
//! }
//! ```
//!
//! `f_ext` and `f_int` are scratch buffers reused across Newton iterations.
//! Within a load step, `f_ext` is assembled once (before the loop) and held
//! constant. `f_int` is re-assembled every Newton iteration as displacements
//! evolve.

use sparse::{SparseMatrix, SymCsrMatrix};
use assembly::Model;

use crate::error::{AnalysisError, Result};

// -----------------------------------------------------------------
// GlobalSystem
// -----------------------------------------------------------------

/// Pre-allocated memory buffers for a complete analysis step.
///
/// Construct once from the global stiffness pattern (obtained via
/// [`assembly::topology::build_pattern`]) and reuse across all load steps
/// and Newton iterations.
///
/// # Example
///
/// ```rust,ignore
/// use assembly::{Model, build_pattern};
/// use analysis::system::GlobalSystem;
///
/// let mut model = build_my_model();
/// let k_pattern = build_pattern(&model).unwrap();
/// let mut system = GlobalSystem::new(k_pattern);
///
/// // Each Newton iteration:
/// system.zero_out();
/// // ... assembly writes into system.k_t, system.r ...
/// // ... solver writes into system.delta_u ...
/// ```
pub struct GlobalSystem {
    /// Global tangent stiffness matrix.
    ///
    /// The sparsity pattern is fixed at construction time (from the mesh
    /// topology) and never changes. Only the numerical values are overwritten
    /// on each Newton iteration by [`assembly::assemble_stiffness`].
    pub k_t: SymCsrMatrix<f64>,

    /// Global residual (unbalanced force) vector.
    ///
    /// After assembly: `r = f_ext - f_int`.
    /// After applying Dirichlet BCs: constrained DOFs are zeroed.
    /// After solve: `K_T Δu = r`, so `r` is overwritten by the solve result
    /// in some solvers. Use `delta_u` for the increment.
    pub r: Vec<f64>,

    /// Incremental displacement vector for the current Newton step.
    ///
    /// Set by the solver: `delta_u = K_T⁻¹ r`.
    /// Added to `model.u_global` after each Newton iteration:
    /// `u_global += delta_u`.
    pub delta_u: Vec<f64>,

    /// External load vector for the current load step.
    ///
    /// Assembled once per load step (before the Newton loop) by
    /// [`assembly::assemble_load_vector`]. Held constant across all Newton
    /// iterations of that step.
    pub f_ext: Vec<f64>,

    /// Internal (resisting) force vector.
    ///
    /// Re-assembled on every Newton iteration from the current
    /// `model.u_global` by [`assembly::assemble_internal_force`].
    pub f_int: Vec<f64>,
}

impl GlobalSystem {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Create a new `GlobalSystem` from the global stiffness pattern.
    ///
    /// The stiffness pattern must have been built by
    /// [`assembly::topology::build_pattern`] for the current mesh topology.
    /// All vectors are initialised to zero.
    ///
    /// # Arguments
    /// * `k_pattern` — the pattern-only stiffness matrix (values may be zero).
    ///   Its sparsity structure is reused for the lifetime of this system.
    pub fn new(k_pattern: SymCsrMatrix<f64>) -> Self {
        let n = k_pattern.nrows();
        Self {
            k_t:     k_pattern,
            r:       vec![0.0; n],
            delta_u: vec![0.0; n],
            f_ext:   vec![0.0; n],
            f_int:   vec![0.0; n],
        }
    }

    // -----------------------------------------------------------------
    // Per-step reset
    // -----------------------------------------------------------------

    /// Zero all mutable buffers for the start of a new Newton step.
    ///
    /// Specifically:
    /// - `k_t` values are set to zero (pattern is preserved).
    /// - `r`, `delta_u`, `f_int` are filled with zeros.
    ///
    /// `f_ext` is **not** zeroed here — it is assembled once per load step
    /// (before the Newton loop) and intentionally held constant.
    ///
    /// # When to call
    ///
    /// Call at the start of each Newton iteration, before assembly:
    ///
    /// ```text
    /// // Outer loop — load steps
    /// assemble_load_vector(&model, t, &mut system.f_ext)?;
    ///
    /// // Inner loop — Newton iterations
    /// loop {
    ///     system.zero_out();              // ← here
    ///     assemble_stiffness(&model, &mut system.k_t)?;
    ///     assemble_internal_force(&model, &mut system.f_int)?;
    ///     // form residual, apply BCs, solve ...
    /// }
    /// ```
    #[inline]
    pub fn zero_out(&mut self) {
        self.k_t.zero();
        self.r.fill(0.0);
        self.delta_u.fill(0.0);
        self.f_int.fill(0.0);
        // f_ext is intentionally left intact
    }

    /// Zero the external force buffer.
    ///
    /// Call before re-assembling `f_ext` at the start of each load step,
    /// or use [`assembly::assemble_load_vector`] which zeros it internally.
    #[inline]
    pub fn zero_f_ext(&mut self) {
        self.f_ext.fill(0.0);
    }

    // -----------------------------------------------------------------
    // Sizing helpers
    // -----------------------------------------------------------------

    /// Total number of global degrees of freedom.
    #[inline]
    pub fn n_dof(&self) -> usize {
        self.r.len()
    }

    // -----------------------------------------------------------------
    // Residual formation
    // -----------------------------------------------------------------

    /// Form the residual vector `r = f_ext - f_int` in-place.
    ///
    /// Must be called after both `f_ext` and `f_int` have been assembled.
    /// The result is written into `self.r` and replaces any previous content.
    ///
    /// # Panics
    /// Panics if `f_ext` and `f_int` have different lengths (i.e. the system
    /// was constructed inconsistently). This cannot happen in normal usage.
    #[inline]
    pub fn form_residual(&mut self) {
        debug_assert_eq!(self.f_ext.len(), self.f_int.len());
        for i in 0..self.r.len() {
            self.r[i] = self.f_ext[i] - self.f_int[i];
        }
    }

    /// Compute the Euclidean norm of the current residual vector `r`.
    ///
    /// This is the primary convergence measure for [`crate::convergence::unbalance::NormUnbalance`].
    #[inline]
    pub fn residual_norm(&self) -> f64 {
        self.r.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Compute the Euclidean norm of the displacement increment `Δu`.
    ///
    /// This is the primary convergence measure for
    /// [`crate::convergence::displacement::NormDispIncr`].
    #[inline]
    pub fn delta_u_norm(&self) -> f64 {
        self.delta_u.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Compute the incremental energy `0.5 * Δu · R`.
    ///
    /// This is the convergence measure for [`crate::convergence::energy::EnergyIncrement`].
    /// It represents the virtual work done by the residual forces through the
    /// displacement increment — an energetically consistent stopping criterion.
    #[inline]
    pub fn energy_increment(&self) -> f64 {
        let dot: f64 = self
            .delta_u
            .iter()
            .zip(self.r.iter())
            .map(|(&du, &ri)| du * ri)
            .sum();
        dot.abs() * 0.5
    }

    // -----------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------

    /// Verify that the system is consistent with the given model.
    ///
    /// Checks that the number of DOFs in this system matches `model.n_dof()`.
    /// Returns an error if they disagree, which would indicate that the
    /// `GlobalSystem` was built from a different (or modified) model.
    ///
    /// # Errors
    /// Returns [`AnalysisError::InvalidConfiguration`] if sizes do not match.
    pub fn check_dof_consistency(&self, model: &Model) -> Result<()> {
        let n_sys   = self.n_dof();
        let n_model = model.n_dof();
        if n_sys != n_model {
            return Err(AnalysisError::InvalidConfiguration {
                reason: format!(
                    "GlobalSystem has {n_sys} DOFs but the model has {n_model} DOFs. \
                     Rebuild the GlobalSystem after modifying the model's topology."
                ),
            });
        }
        Ok(())
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;

    /// Build a tiny 3×3 symmetric pattern for testing.
    fn tiny_pattern() -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0, 1.0);
        coo.add(1, 1, 1.0);
        coo.add(2, 2, 1.0);
        coo.add(0, 1, 0.5);
        coo.add(1, 2, 0.5);
        coo.build_sym().unwrap()
    }

    #[test]
    fn new_allocates_correct_sizes() {
        let pat = tiny_pattern();
        let sys = GlobalSystem::new(pat);
        assert_eq!(sys.n_dof(), 3);
        assert_eq!(sys.r.len(), 3);
        assert_eq!(sys.delta_u.len(), 3);
        assert_eq!(sys.f_ext.len(), 3);
        assert_eq!(sys.f_int.len(), 3);
    }

    #[test]
    fn zero_out_clears_r_and_delta_u_and_f_int() {
        let pat = tiny_pattern();
        let mut sys = GlobalSystem::new(pat);
        sys.r[0] = 99.0;
        sys.delta_u[1] = -7.0;
        sys.f_int[2] = 42.0;
        sys.f_ext[0] = 100.0; // should survive zero_out

        sys.zero_out();

        assert!(sys.r.iter().all(|&v| v == 0.0));
        assert!(sys.delta_u.iter().all(|&v| v == 0.0));
        assert!(sys.f_int.iter().all(|&v| v == 0.0));
        assert_eq!(sys.f_ext[0], 100.0, "f_ext must not be zeroed by zero_out");
    }

    #[test]
    fn form_residual_subtracts_correctly() {
        let pat = tiny_pattern();
        let mut sys = GlobalSystem::new(pat);
        sys.f_ext = vec![10.0, 20.0, 30.0];
        sys.f_int = vec![3.0,  5.0,  7.0];
        sys.form_residual();
        assert!((sys.r[0] - 7.0).abs() < 1e-15);
        assert!((sys.r[1] - 15.0).abs() < 1e-15);
        assert!((sys.r[2] - 23.0).abs() < 1e-15);
    }

    #[test]
    fn residual_norm_correct() {
        let pat = tiny_pattern();
        let mut sys = GlobalSystem::new(pat);
        sys.r = vec![3.0, 4.0, 0.0];
        assert!((sys.residual_norm() - 5.0).abs() < 1e-14);
    }

    #[test]
    fn delta_u_norm_correct() {
        let pat = tiny_pattern();
        let mut sys = GlobalSystem::new(pat);
        sys.delta_u = vec![0.0, 3.0, 4.0];
        assert!((sys.delta_u_norm() - 5.0).abs() < 1e-14);
    }

    #[test]
    fn energy_increment_correct() {
        let pat = tiny_pattern();
        let mut sys = GlobalSystem::new(pat);
        // 0.5 * (1*2 + 2*4 + 0*0) = 0.5 * 10 = 5
        sys.delta_u = vec![1.0, 2.0, 0.0];
        sys.r       = vec![2.0, 4.0, 0.0];
        assert!((sys.energy_increment() - 5.0).abs() < 1e-14);
    }
}