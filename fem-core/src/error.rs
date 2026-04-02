use miette::Diagnostic;
use thiserror::Error;

/// Errors arising in the `fem_core` geometric and topological layer.
///
/// These errors represent faults in element geometry, DOF mapping
/// construction, and model dimensionality — the foundations that all
/// element formulations build upon.
#[derive(Debug, Error, Diagnostic)]
#[non_exhaustive]
pub enum CoreError {
    /// Two nodes defining an element are coincident: the element has
    /// zero characteristic length, producing a singular Jacobian for
    /// all strain-displacement relations.
    ///
    /// The `x1/y1` and `x2/y2` fields are the coordinates of the two
    /// coincident (or near-coincident) nodes.
    #[error(
        "Degenerate element geometry: nodes at ({x1:.6e}, {y1:.6e}) and \
         ({x2:.6e}, {y2:.6e}) are coincident (computed length = {length:.6e})."
    )]
    #[diagnostic(
        code(echelon::fem_core::geometry::degenerate_element),
        help(
            "The characteristic length of the element evaluates to {length:.6e}, \
             which causes a division by zero in the strain-displacement matrix B \
             and a singular coordinate transformation T. \n\
             Likely causes: \n\
             (1) A probabilistic geometry parameter sampled a near-zero \
                 inter-nodal distance. Add a lower bound to the node-separation \
                 distribution. \n\
             (2) A mesh generation error placed two nodes at the same location. \
                 Check the nodal coordinate array for duplicate entries. \n\
             (3) If modelling a pin or hinge at a single point, use a \
                 ZeroLength element formulation rather than a finite-domain \
                 element with coincident end nodes."
        )
    )]
    DegenerateGeometry {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        length: f64,
    },

    /// The coordinate transform rotation matrix is not orthogonal to within
    /// a numerical tolerance: `cos²θ + sin²θ` deviates from 1.
    #[error(
        "Malformed coordinate transformation: cos²θ + sin²θ = {norm_sq:.10e} \
         (deviation from 1: {deviation:.3e}). The rotation matrix is not orthogonal."
    )]
    #[diagnostic(
        code(echelon::fem_core::geometry::non_orthogonal_transform),
        help(
            "A coordinate transformation matrix constructed from (cos, sin, length) \
             values that do not satisfy the unit-norm constraint will produce \
             incorrect stiffness assembly and incorrect internal force recovery. \
             Use `CoordTransf2d::from_nodes(x1, y1, x2, y2)` to compute (cos, sin) \
             from nodal coordinates rather than supplying them directly via \
             `from_cos_sin_length`."
        )
    )]
    NonOrthogonalTransform { norm_sq: f64, deviation: f64 },

    /// A DOF map construction attempted to use a node ID that exceeds the
    /// range implied by the model's `ndf` (DOFs per node) and the total
    /// allocated DOF count.
    #[error(
        "DOF map construction error: node {node_id} with {ndf} DOFs per node \
         would occupy global DOF {last_dof}, exceeding the allocated dimension {n_dof}."
    )]
    #[diagnostic(
        code(echelon::fem_core::topology::dof_map_overflow),
        help(
            "The global DOF index for node {node_id} evaluates to \
             node_id x ndf + (ndf - 1) = {last_dof}, which exceeds the \
             total allocated DOF count ({n_dof}). \
             Ensure that the mesh node indices are 0-based and contiguous, \
             and that `n_dof = n_nodes x ndf` was computed from the complete \
             nodal set before assembling element DOF maps."
        )
    )]
    DofMapOverflow {
        node_id: usize,
        ndf: usize,
        last_dof: usize,
        n_dof: usize,
    },
}

pub type Result<T> = std::result::Result<T, CoreError>;