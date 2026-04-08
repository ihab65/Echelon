//! DOF topology — counting and validating the global equation system size.
//!
//! In Echelon the DOF layout is fully determined by the `NodeId` ordering
//! and the model's `ndf` (DOFs per node). There is no separate DOF numbering
//! pass because `DofMap::from_nodes` in `fem_core` already encodes the
//! convention:
//!
//! ```text
//! global_dof(node k, local dof j) = k * ndf + j
//! ```
//!
//! This module provides two functions:
//! - `count_dofs`: returns the total system size `n_nodes * ndf`.
//! - `validate_dof_maps`: checks every element's `DofMap` against the
//!   system size and reports the first out-of-range index.

use crate::error::{AssemblyError, Result};
use crate::model::Model;

// -----------------------------------------------------------------
// count_dofs
// -----------------------------------------------------------------

/// Return the total number of global DOFs for this model.
///
/// Equal to `n_nodes × ndf`. This is the dimension of the global stiffness
/// matrix K, the load vector F, and `model.u_global`.
///
/// # Example
///
/// ```rust,ignore
/// let n = count_dofs(&model);
/// assert_eq!(n, model.n_dof());
/// ```
#[inline]
pub fn count_dofs(model: &Model) -> usize {
    model.n_dof()
}

// -----------------------------------------------------------------
// validate_dof_maps
// -----------------------------------------------------------------

/// Verify that every element's `DofMap` refers only to valid global DOF
/// indices for this model.
///
/// This is a defence-in-depth check. In release builds it is O(nnz_elements)
/// and is typically called once after the model is fully built, before the
/// first `build_pattern` call.
///
/// # Errors
/// Returns [`AssemblyError::UnresolvedNode`] with the offending `node_id`
/// (back-computed from the DOF index) if any element's DOF map references a
/// global DOF `>= n_dof`.
pub fn validate_dof_maps(model: &Model) -> Result<()> {
    let n_dof = model.n_dof();
    let ndf   = model.dim.ndf();

    for element in &model.elements {
        element.dof_map()
            .validate_against(n_dof, ndf)
            .map_err(|_core_err| {
                // Convert a CoreError::DofMapOverflow into our assembly error.
                // The CoreError carries the node_id already; we re-derive it
                // to stay within the assembly error type.
                let bad_dof = element.dof_map()
                    .as_usize_slice()
                    .iter()
                    .copied()
                    .find(|&d| d >= n_dof)
                    .unwrap_or(usize::MAX);
                let bad_node = bad_dof / ndf.max(1);
                AssemblyError::UnresolvedNode { node_id: bad_node }
            })?;
    }
    Ok(())
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{ModelDim, NodeId};
    use materials::ElasticUniaxial;
    use elements::Truss2d;
    use crate::model::{Model, Node};

    fn two_node_truss_model() -> Model {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 1.0, 0.0, 0.0)).unwrap();
        let mat  = ElasticUniaxial::new(200e9, None).unwrap();
        let elem = Truss2d::new(
            NodeId(0), NodeId(1),
            0.0, 0.0, 1.0, 0.0,
            mat, 0.01,
        ).unwrap();
        m.add_element(elem);
        m.build_state();
        m
    }

    #[test]
    fn count_dofs_correct() {
        let m = two_node_truss_model();
        assert_eq!(count_dofs(&m), 4); // 2 nodes × 2 DOF
    }

    #[test]
    fn validate_dof_maps_passes_for_valid_model() {
        let m = two_node_truss_model();
        validate_dof_maps(&m).unwrap();
    }

    #[test]
    fn empty_model_zero_dofs() {
        let m = Model::new(ModelDim::frame_2d());
        assert_eq!(count_dofs(&m), 0);
    }
}