//! [`NdMaterial`] — the multi-dimensional constitutive interface.
//!
//! This is the N-dimensional analogue of [`UniaxialMaterial`].  Where the
//! uniaxial trait maps a scalar strain to a scalar stress, `NdMaterial`
//! maps a strain **vector** to a stress **vector** and provides the
//! material tangent **matrix** (Voigt notation, row-major flat storage).
//!
//! The trait is parameterised by [`NdMaterial::order`], which returns the
//! number of independent strain/stress components:
//!
//! | Formulation      | `order()` | Voigt components                      |
//! |------------------|-----------|---------------------------------------|
//! | Plane Stress     | 3         | εxx, εyy, γxy                         |
//! | Plane Strain     | 4         | εxx, εyy, εzz, γxy                    |
//! | Axisymmetric     | 4         | εrr, εzz, εθθ, γrz                    |
//! | 3-D              | 6         | εxx, εyy, εzz, γxy, γyz, γxz          |
//! | Plate/Shell      | 5 or 8    | (formulation-dependent)               |
//!
//! ## Zero-allocation contract
//!
//! All output methods write into caller-provided `&mut [f64]` slices.
//! The element that owns the material is responsible for allocating the
//! work buffers **once** at construction time and reusing them across
//! every Newton-Raphson iteration.  This guarantees zero heap allocations
//! on the hot loop.
//!
//! ## State lifecycle
//!
//! ```text
//! loop over load steps:
//!     trial_strain ← element kinematics (Voigt vector, length = order())
//!     material.stress(trial_strain, &mut σ)        // σ: length order()
//!     material.tangent(trial_strain, &mut C)        // C: length order()²
//!
//!     if converged:
//!         material.commit_state(trial_strain)
//!     else if diverged:
//!         material.revert_to_last_commit()
//! ```
//!
//! The material is free to cache any internal state (back-stress tensor,
//! plastic strain tensor, damage variables) between `stress`/`tangent`
//! calls and `commit_state`.  After `revert_to_last_commit`, all internal
//! state must be identical to the last committed state.

use crate::error::Result;

/// The core interface for a multi-dimensional constitutive model.
///
/// All methods operate in **Voigt-notation strain space**: the element
/// passes the current trial strain vector and the material returns the
/// stress vector and tangent stiffness matrix.
///
/// # Slice length conventions
///
/// | Argument / output | Length              |
/// |-------------------|---------------------|
/// | `strain`          | `order()`           |
/// | stress `out`      | `order()`           |
/// | tangent `out`     | `order() * order()` |
///
/// The tangent matrix is stored **row-major** so that `C[i][j]` lives at
/// index `i * order() + j` in the flat slice — consistent with the
/// row-major convention used by `ke_flat` in the [`Element`] trait.
///
/// # Thread safety
///
/// Like [`UniaxialMaterial`], implementations must be `Send + Sync` to
/// support population-parallel analysis.
pub trait NdMaterial: Send + Sync {
    /// Number of independent strain/stress components in the Voigt vector.
    ///
    /// - `3` — plane stress (εxx, εyy, γxy)
    /// - `4` — plane strain / axisymmetric
    /// - `6` — full 3-D
    fn order(&self) -> usize;

    /// Compute the stress vector for the given trial strain.
    ///
    /// For linear elastic materials this is `σ = C : ε` (matrix-vector
    /// product of the elastic stiffness tensor with the strain vector).
    /// For inelastic materials it involves the full return-mapping
    /// algorithm in Voigt space.
    ///
    /// # Arguments
    /// * `strain` — trial strain vector (Voigt), length `order()`
    /// * `out`    — stress output buffer, length `order()`
    ///
    /// # Panics (debug)
    /// Debug-asserts that `strain.len() == order()` and
    /// `out.len() == order()`.
    fn stress(&self, strain: &[f64], out: &mut [f64]);

    /// Compute the material tangent stiffness matrix for the given trial
    /// strain.
    ///
    /// For linear elastic materials this is the constant elastic stiffness
    /// tensor `C`.  For inelastic materials this is the algorithmic
    /// (consistent) tangent, not the continuum tangent.
    ///
    /// # Arguments
    /// * `strain` — the same trial strain as passed to [`stress`]
    /// * `out`    — tangent output buffer (row-major), length `order()²`
    ///
    /// # Panics (debug)
    /// Debug-asserts that `strain.len() == order()` and
    /// `out.len() == order() * order()`.
    fn tangent(&self, strain: &[f64], out: &mut [f64]);

    /// Commit the current strain as the new converged state.
    ///
    /// Called once per load step after Newton-Raphson convergence.
    /// The material should:
    /// 1. Store `strain` as the committed strain vector.
    /// 2. Store any accumulated internal variables (plastic strain
    ///    tensor, back-stress tensor, damage variables) as committed.
    ///
    /// # Arguments
    /// * `strain` — the converged trial strain vector, length `order()`
    ///
    /// # Errors
    /// - [`MaterialError::StrainDomainViolation`] if any component of
    ///   `strain` falls outside the valid domain of this constitutive
    ///   model.
    fn commit_state(&mut self, strain: &[f64]) -> Result<()>;

    /// Revert all internal state to the last committed state.
    ///
    /// Called when Newton-Raphson fails to converge and the solver must
    /// retry the load step with a smaller increment.  After this call
    /// the material behaves exactly as it did at the last `commit_state`.
    fn revert_to_last_commit(&mut self);

    /// Clone into a boxed trait object.
    ///
    /// Required because `Clone` is not object-safe, but we need to
    /// duplicate material objects for population-parallel analysis
    /// (each analysis instance owns its own material state).
    fn clone_box(&self) -> Box<dyn NdMaterial>;

    /// Human-readable name for this material model.
    fn name(&self) -> &'static str;
}
