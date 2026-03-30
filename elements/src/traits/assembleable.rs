//! [`Assembleable`] — bridges elements to the global system.
//!
//! This trait provides the two pieces of information the assembler needs:
//!
//! 1. **`dof_map`** — maps local element DOF indices to global DOF indices.
//!    The assembler calls `scatter_add(ke_flat, dof_map.as_usize_slice())`.
//!
//! 2. **`partial_residual_wrt_param`** — the element-level contribution to
//!    `∂f_int/∂θ` used by Engine B's adjoint method.
//!
//! ## Why a separate trait?
//!
//! `Element` defines *what an element computes* (stiffness, residual).
//! `Assembleable` defines *where the computation goes* (which global DOFs)
//! and *what Engine B needs* (parameter sensitivity hooks).
//!
//! Keeping these separate allows:
//! - Mock elements in tests (implement `Element` without `Assembleable`).
//! - Different assembly strategies (e.g. p-adaptive, multi-point constraints)
//!   without changing element implementations.

use fem_core::DofMap;

/// Bridges an element to the global assembly loop and to Engine B.
///
/// Implement this trait on all concrete elements that participate in a
/// complete FEM analysis.
pub trait Assembleable: crate::traits::Element {
    /// The DOF map: `dof_map[local] = GlobalDof`.
    ///
    /// Constructed during model build from `DofMap::from_nodes`.
    /// Immutable after construction.
    fn dof_map(&self) -> &DofMap;

    /// Element-level contribution to `∂f_int/∂θ` for Engine B adjoint.
    ///
    /// For a scalar material parameter `θ` (e.g. `E`, `Fy`):
    ///
    /// ```text
    /// (∂f_int/∂θ)[element] = B^T · (∂σ/∂θ) · volume_weight
    /// ```
    ///
    /// where `B` is the strain-displacement matrix evaluated at the converged
    /// state, and `∂σ/∂θ` comes from the material's `AdjointSensitive` impl.
    ///
    /// # Arguments
    /// * `u_local`   — converged local displacements (global coord), length `n_dof()`
    /// * `param_idx` — index of the parameter within the element's parameter space
    ///
    /// # Returns
    /// `Vec<f64>` of length `n_dof()` — the local `∂f_int/∂θ` vector.
    /// The assembler scatters this into the global `∂F/∂θ` using `dof_map`.
    ///
    /// # Note
    /// For geometric parameters (e.g. element length, cross-section area) the
    /// derivative is computed analytically from the element formulation.
    /// For material parameters the element delegates to its material's
    /// `AdjointSensitive::stress_sensitivity`.
    fn partial_residual_wrt_param(&self, u_local: &[f64], param_idx: usize) -> Vec<f64>;

    /// Number of parameters this element exposes for adjoint sensitivity.
    fn n_params(&self) -> usize;

    /// Human-readable name for parameter `i`.
    fn param_name(&self, param_idx: usize) -> &'static str;
}