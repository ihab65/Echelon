use miette::Diagnostic;
use thiserror::Error;

use fem_core::error::CoreError;
use sparse::SparseError;
use elements::error::ElementError;

/// Errors arising in the `assembly` crate.
///
/// These errors represent faults in model construction, DOF topology,
/// load application, and the matrix/vector scatter operations.
///
/// All variants carry structured context fields so that a population
/// sampling script can distinguish a badly-constrained model from a
/// density-missing element from a scatter index fault — without parsing
/// string messages.
///
/// Transparent passthrough variants preserve the original diagnostic codes
/// from `sparse`, `fem_core`, and `elements` so the full error chain is
/// visible through `miette`'s `fancy` renderer.
#[derive(Debug, Error, Diagnostic)]
#[non_exhaustive]
pub enum AssemblyError {
    // ---- Model topology --------------------------------------------------

    /// A load pattern or element references a node index that has not been
    /// registered in the model's node list.
    #[error(
        "Unresolved node reference: node {node_id} is referenced by an \
         element or load pattern but has not been added to the model."
    )]
    #[diagnostic(
        code(echelon::assembly::topology::unresolved_node),
        help(
            "Ensure that every node referenced by elements and load patterns \
             is added to the model via `Model::add_node` before the first \
             call to `build_pattern` or any assembly function. \
             Node indices are 0-based and must be contiguous."
        )
    )]
    UnresolvedNode { node_id: usize },

    /// A load or constraint targets a DOF index that exceeds the number of
    /// DOFs per node declared by `ModelDim`.
    ///
    /// For example, applying a moment (DOF 2) to a node in a 2D truss model
    /// (ndf = 2) would trigger this error.
    #[error(
        "DOF index overflow: local DOF {local_dof} was requested for node \
         {node_id}, but the model declares only {ndf} DOF(s) per node \
         (valid indices: 0..{ndf})."
    )]
    #[diagnostic(
        code(echelon::assembly::kinematics::dof_overflow),
        help(
            "The local DOF index must be in the range 0..ndf where ndf is \
             the number of degrees of freedom per node declared in `ModelDim`. \
             For a 2D frame model (ModelDim::frame_2d): DOF 0 = UX, DOF 1 = UY, \
             DOF 2 = RZ. For a 2D truss model (ModelDim::truss_2d): DOF 0 = UX, \
             DOF 1 = UY — moment DOFs do not exist."
        )
    )]
    DofOverflow {
        node_id:   usize,
        local_dof: usize,
        ndf:       usize,
    },

    // ---- Mass assembly ---------------------------------------------------

    /// `assemble_mass` was called but element {element_idx} returned an
    /// all-zero mass matrix because its material has no density (`rho`).
    ///
    /// Every element participating in a dynamic or self-weight analysis
    /// must have `rho` set on its material.
    #[error(
        "Missing mass density: element at index {element_idx} returned an \
         all-zero mass matrix. Mass assembly requires every element to have \
         a density parameter `rho` defined on its material."
    )]
    #[diagnostic(
        code(echelon::assembly::mass::missing_density),
        help(
            "Set the `rho` field on the element's material before requesting \
             a mass matrix or self-weight load vector. \
             For `ElasticUniaxial`, pass `rho: Some(value)` to the constructor. \
             If only a static analysis is intended, do not call \
             `assemble_mass` or `assemble_self_weight` — use explicit \
             nodal loads for gravity instead."
        )
    )]
    MissingDensity { element_idx: usize },

    // ---- Transparent passthroughs ----------------------------------------

    /// A sparse matrix operation (pattern construction, scatter, BC
    /// application) failed in the `sparse` crate.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Sparse(#[from] SparseError),

    /// A geometric or topological fault propagated from `fem_core`.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Core(#[from] CoreError),

    /// An element-level fault (inadmissible section, unregistered parameter,
    /// material error) propagated from the `elements` crate.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Element(#[from] ElementError),
}

pub type Result<T> = std::result::Result<T, AssemblyError>;