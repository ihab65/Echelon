//! [`UniaxialMaterial`] — the fundamental f64 material interface.
//!
//! Every material in Echelon — elastic, plastic, nonlinear — implements
//! this trait.  It is intentionally narrow: the three methods that the
//! Newton-Raphson loop actually calls, plus the state management pair
//! required for path-dependent constitutive models.
//!
//! Materials that are smooth and parameter-generic should *also* implement
//! [`SmoothUniaxial`] (see `smooth.rs`) so they can be used inside
//! energy-based elements with automatic differentiation.
//!
//! Materials that have history dependence and want to participate in
//! Engine B adjoint sensitivity analysis should also implement
//! [`AdjointSensitive`] (see `adjoint.rs`).

/// The core interface for a one-dimensional constitutive model.
///
/// All methods operate in **strain space**: the element passes the current
/// trial strain and the material returns stress and tangent modulus.
///
/// # State lifecycle
///
/// ```text
/// loop over load steps:
///     trial_strain ← element kinematics
///     σ  = material.stress(trial_strain)
///     Eₜ = material.tangent(trial_strain)
///
///     if converged:
///         material.commit_state(trial_strain)
///     else if diverged:
///         material.revert_to_last_commit()
/// ```
///
/// The material is free to cache any internal state (back-stress,
/// plastic strain, damage variable) between `stress`/`tangent` calls
/// and `commit_state`.  After `revert_to_last_commit`, all internal
/// state must be identical to the last committed state.
pub trait UniaxialMaterial: Send + Sync {
    /// Compute the stress at the given trial strain.
    ///
    /// For elastic materials this is simply `E * strain`.
    /// For inelastic materials it involves the full return-mapping.
    ///
    /// # Arguments
    /// * `strain` — total engineering strain (dimensionless)
    fn stress(&self, strain: f64) -> f64;

    /// Compute the consistent tangent modulus at the given trial strain.
    ///
    /// For elastic materials this equals the elastic modulus `E`.
    /// For inelastic materials this is the algorithmic (consistent) tangent,
    /// not the continuum tangent.
    ///
    /// # Arguments
    /// * `strain` — the same trial strain as passed to [`stress`]
    fn tangent(&self, strain: f64) -> f64;

    /// Commit the current strain as the new converged state.
    ///
    /// Called once per load step after Newton-Raphson convergence.
    /// The material should:
    /// 1. Store `strain` as the committed strain.
    /// 2. Store any accumulated internal variables as committed values.
    /// 3. Return the committed stress (for output purposes).
    ///
    /// # Arguments
    /// * `strain` — the converged trial strain
    fn commit_state(&mut self, strain: f64) -> f64;

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
    fn clone_box(&self) -> Box<dyn UniaxialMaterial>;

    /// Human-readable name for this material model.
    fn name(&self) -> &'static str;
}
