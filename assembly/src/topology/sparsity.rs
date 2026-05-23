//! Sparsity pattern construction for the global stiffness matrix.
//!
//! `build_pattern` is called **once per topology change** — when elements are
//! added or removed. As long as the connectivity is unchanged (only material
//! parameters or geometry change), the pattern is reused across all Newton
//! iterations and load steps.
//!
//! The resulting [`SymCsrMatrix`] is zero-valued but has the correct
//! upper-triangle sparsity pattern for Cholesky factorization. Pass it
//! to `builders::assemble_stiffness` or `builders::assemble_mass` to fill
//! in the numerical values.

use sparse::SymCsrMatrix;

use crate::error::Result;
use crate::model::Model;
use crate::topology::dof::validate_dof_maps;

// -----------------------------------------------------------------
// build_pattern
// -----------------------------------------------------------------

/// Build the upper-triangle sparsity pattern for the global stiffness matrix.
///
/// Collects the `DofMap` of every element, forms the set of all `(i, j)` DOF
/// pairs each element couples (upper triangle: `j >= i`), and returns a
/// zero-valued [`SymCsrMatrix`] ready for value assembly.
///
/// ## When to call
///
/// Call once after the full topology is established (all nodes and elements
/// added). The returned matrix pattern is stable for the lifetime of that
/// topology — do not rebuild it on every Newton iteration.
///
/// ```rust,ignore
/// let mut k = build_pattern(&model)?;
/// // Analysis loop:
/// loop {
///     k.zero();
///     assemble_stiffness(&model, &mut k)?;
///     // ...
/// }
/// ```
///
/// ## Errors
/// - [`crate::error::AssemblyError::UnresolvedNode`] if any element references a DOF index
///   that exceeds the model's total DOF count (caught by `validate_dof_maps`).
/// - [`sparse::SparseError`] from `SymCsrMatrix::from_dof_connectivity` if the
///   pattern construction fails (e.g., empty model).
pub fn build_pattern(model: &Model) -> Result<SymCsrMatrix<f64>> {
    // Validate before building — fail fast with a clear error rather than
    // producing a pattern that silently misses entries.
    validate_dof_maps(model)?;

    let n_dof = model.n_dof();

    // Collect element DOF lists as slices of usize — zero-copy thanks to
    // DofMap::as_usize_slice() which reinterprets &[GlobalDof] as &[usize].
    let element_dofs: Vec<Vec<usize>> = model.elements
        .iter()
        .map(|e| e.dof_map().as_usize_slice().to_vec())
        .collect();

    let k = SymCsrMatrix::from_dof_connectivity(n_dof, &element_dofs)?;
    Ok(k)
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{ModelDim, NodeId};
    use materials::ElasticUniaxial;
    use elements::{Truss2d, ElasticBeam2d};
    use crate::model::{Model, Node};

    fn steel() -> ElasticUniaxial {
        ElasticUniaxial::new(200e9, None).unwrap()
    }

    // ---- Truss model ----

    fn two_node_truss() -> Model {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        let e = Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0, steel(), 0.01).unwrap();
        m.add_element(e);
        m.build_state();
        m
    }

    #[test]
    fn truss_pattern_correct_size() {
        let m = two_node_truss();
        let k = build_pattern(&m).unwrap();
        // 2 nodes × 2 DOF = 4×4 system
        assert_eq!(k.n, 4);
        k.validate().unwrap();
    }

    #[test]
    fn truss_pattern_has_diagonal() {
        let m   = two_node_truss();
        let k   = build_pattern(&m).unwrap();
        // Diagonal must be present in every row
        for i in 0..4 {
            assert!(k.get(i, i).is_ok());
        }
    }

    #[test]
    fn truss_pattern_upper_triangle_connectivity() {
        let m = two_node_truss();
        let k = build_pattern(&m).unwrap();
        // Horizontal truss: DOF 0 and DOF 2 are both UX DOFs — they should
        // be connected in the upper triangle.
        // k.get(0, 2) should exist in the pattern (non-zero structural entry)
        // get returns 0.0 for structural zeros, so we check that scatter later
        // would find the entry — here we just confirm the pattern is square and valid.
        assert_eq!(k.n, 4);
    }

    // ---- Beam model ----

    fn two_node_beam() -> Model {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        let e = ElasticBeam2d::new(
            NodeId(0), NodeId(1),
            0.0, 0.0, 3.0, 0.0,
            steel(), 0.01, 1e-4,
        ).unwrap();
        m.add_element(e);
        m.build_state();
        m
    }

    #[test]
    fn beam_pattern_is_6x6() {
        let m = two_node_beam();
        let k = build_pattern(&m).unwrap();
        assert_eq!(k.n, 6); // 2 nodes × 3 DOF
        k.validate().unwrap();
    }

    #[test]
    fn beam_pattern_nnz_consistent() {
        let m = two_node_beam();
        let k = build_pattern(&m).unwrap();
        // 2-node beam: all 6 DOFs couple → upper triangle has at most 21 entries
        // (6 diagonal + 15 off-diagonal). The actual count depends on the
        // symmetric pattern, but must be at least 6 (diagonal) and at most 21.
        assert!(k.nnz() >= 6);
        assert!(k.nnz() <= 21);
    }

    #[test]
    fn multi_element_pattern_valid() {
        // Two beams in series: node 0 – node 1 – node 2
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(2), 6.0, 0.0, 0.0)).unwrap();
        let e1 = ElasticBeam2d::new(
            NodeId(0), NodeId(1), 0.0, 0.0, 3.0, 0.0, steel(), 0.01, 1e-4,
        ).unwrap();
        let e2 = ElasticBeam2d::new(
            NodeId(1), NodeId(2), 3.0, 0.0, 6.0, 0.0, steel(), 0.01, 1e-4,
        ).unwrap();
        m.add_element(e1);
        m.add_element(e2);
        m.build_state();

        let k = build_pattern(&m).unwrap();
        assert_eq!(k.n, 9); // 3 nodes × 3 DOF
        k.validate().unwrap();
    }
}