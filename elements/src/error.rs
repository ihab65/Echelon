use miette::Diagnostic;
use thiserror::Error;
use fem_core::error::CoreError;
use materials::error::MaterialError;

/// Errors arising in the `elements` kinematic and formulation layer.
///
/// These errors represent faults in element construction, geometric
/// transformations, numerical integration, and parameter sensitivity
/// evaluation — the layer between the constitutive model and the global
/// assembly loop.
///
/// Transparent passthrough variants ensure clean error bubbling from
/// the material and fem_core layers into the assembly loop, preserving
/// the original diagnostic codes.
#[derive(Debug, Error, Diagnostic)]
#[non_exhaustive]
pub enum ElementError {
    /// The element formulation was constructed with parameter values that
    /// violate the kinematic or section admissibility requirements of
    /// the specific element type.
    ///
    /// This is distinct from `MaterialError::InadmissibleParameter`, which
    /// concerns the constitutive law; this error concerns the element
    /// geometric and section parameters (length, area, moment of inertia)
    /// independent of the material.
    #[error(
        "Element section parameter '{parameter}' violates kinematic admissibility \
         (provided: {value:.6e}; required: {requirement}). \
         Element type: {element_type}."
    )]
    #[diagnostic(
        code(echelon::elements::kinematics::inadmissible_section),
        help(
            "The {element_type} element received a section parameter '{parameter}' \
             with value {value:.6e}, which violates the requirement: {requirement}. \n\
             Section parameters govern the element's kinematic response and must \
             satisfy physical admissibility: \n\
             - Cross-section area A > 0 is required for axial stiffness EA/L > 0. \
             - Second moment of area Iz > 0 is required for bending stiffness EI/L³ > 0. \
             In probabilistic sampling, apply lower bounds to section property \
             distributions to prevent inadmissible realisations."
        )
    )]
    InadmissibleSection {
        element_type: &'static str,
        parameter: &'static str,
        value: f64,
        requirement: &'static str,
    },

    /// An adjoint sensitivity was requested for a parameter index that
    /// exceeds this element's registered parameter count.
    #[error(
        "Element adjoint sensitivity requested for unregistered parameter \
         index {idx}. Element type '{element_type}' has {n_params} \
         parameter(s) (indices 0..{n_params})."
    )]
    #[diagnostic(
        code(echelon::elements::adjoint::unregistered_parameter),
        help(
            "The requested parameter index {idx} is out of range for '{element_type}', \
             which exposes {n_params} parameters (valid indices: 0..{n_params}). \n\
             Consult the `params` sub-module in the element's source file for the \
             canonical parameter ordering (e.g., `elements::truss2d::params::E = 0`, \
             `elements::truss2d::params::A = 1`). \n\
             In the sensitivity assembly driver, verify that the global parameter \
             offset for each element type is correctly computed from the element \
             connectivity and the model's parameter index map."
        )
    )]
    UnregisteredParameter {
        element_type: &'static str,
        idx: usize,
        n_params: usize,
    },

    // ---- Transparent passthroughs for clean error bubbling ----

    /// A constitutive fault propagated up from the material layer.
    ///
    /// The original `MaterialError` diagnostic code is preserved
    /// transparently so that population sampling scripts can distinguish
    /// `echelon::materials::admissibility::parameter_violation` from
    /// `echelon::elements::kinematics::inadmissible_section`.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Material(#[from] MaterialError),

    /// A geometric or topological fault propagated up from `fem_core`.
    ///
    /// The original `CoreError` diagnostic code is preserved transparently.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Core(#[from] CoreError),
}

pub type Result<T> = std::result::Result<T, ElementError>;