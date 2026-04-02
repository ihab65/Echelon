use miette::Diagnostic;
use thiserror::Error;

/// Errors arising in the `materials` constitutive layer.
///
/// These errors represent violations of physical admissibility bounds,
/// state-variable consistency, and adjoint sensitivity parameter indexing —
/// the faults that arise when the material model receives physically
/// impossible inputs or is queried outside its defined parameter space.
#[derive(Debug, Error, Diagnostic)]
#[non_exhaustive]
pub enum MaterialError {
    /// A constitutive parameter violates its physical admissibility bounds.
    ///
    /// For example, Young's modulus E must be strictly positive to ensure
    /// positive strain energy density; a cross-section area A ≤ 0 has no
    /// physical meaning.
    ///
    /// The `parameter` field names the violated parameter, `value` is the
    /// supplied value, and `requirement` describes the admissible range.
    #[error(
        "Constitutive parameter '{parameter}' violates physical admissibility \
         bounds (provided: {value:.6e}; required: {requirement})."
    )]
    #[diagnostic(
        code(echelon::materials::admissibility::parameter_violation),
        help(
            "Material parameter '{parameter}' received a value of {value:.6e}, \
             which violates the requirement: {requirement}. \n\
             Physical reasoning: \n\
             - Young's modulus E must be strictly positive (E > 0) to ensure \
               positive strain energy density W = ½ E ε² > 0 for all ε ≠ 0. \
             - Cross-section area A must be strictly positive to resist axial load. \
             - Second moment of area Iz must be strictly positive to resist bending. \
             In probabilistic sampling, add a lower bound (e.g. a LogNormal \
             distribution with μ ≫ 0) to prevent the sampler from drawing \
             physically inadmissible realisations."
        )
    )]
    InadmissibleParameter {
        parameter: &'static str,
        value: f64,
        requirement: &'static str,
    },

    /// The adjoint sensitivity was requested for a parameter index that is
    /// not registered in this material model's parameter space.
    ///
    /// The `idx` field is the requested index; `n_params` is the number of
    /// parameters actually registered.
    #[error(
        "Adjoint sensitivity requested for unregistered parameter index {idx}. \
         This material model has {n_params} parameter(s) (indices 0..{n_params})."
    )]
    #[diagnostic(
        code(echelon::materials::adjoint::unregistered_parameter),
        help(
            "The requested parameter index {idx} is out of range for this \
             constitutive model, which registers {n_params} parameter(s) \
             (valid indices: 0..{n_params}). \n\
             Check the parameter mapping in the adjoint sensitivity driver: \
             the global parameter vector must be partitioned correctly among \
             all element types, and each element must offset its local parameter \
             index by the correct base index. \
             Consult the `param_name()` method or the per-element `params` module \
             for the canonical parameter ordering."
        )
    )]
    UnregisteredParameter { idx: usize, n_params: usize },

    /// The material was asked to commit a strain state that is outside the
    /// valid domain of the constitutive model.
    ///
    /// For example, a damage model may have a maximum compressive strain
    /// beyond which its equations are undefined, or a gap element requires
    /// non-negative gap opening.
    #[error(
        "Constitutive state update failed: trial strain {strain:.6e} is \
         outside the valid domain [{domain_lo:.6e}, {domain_hi:.6e}] for \
         material '{material_name}'."
    )]
    #[diagnostic(
        code(echelon::materials::state::strain_domain_violation),
        help(
            "The trial strain {strain:.6e} passed to `commit_state` exceeds the \
             valid domain of the '{material_name}' constitutive model. \n\
             In a Newton-Raphson loop, this typically indicates an excessively \
             large load increment that drove the material past its defined \
             response envelope. Consider: \n\
             (1) Reducing the load step size via a finer load increment. \n\
             (2) Switching to arc-length control to traverse limit points \
                 in the equilibrium path. \n\
             (3) Verifying that the structural geometry and boundary conditions \
                 are physically consistent with the applied loading."
        )
    )]
    StrainDomainViolation {
        strain: f64,
        domain_lo: f64,
        domain_hi: f64,
        material_name: &'static str,
    },
}

pub type Result<T> = std::result::Result<T, MaterialError>;