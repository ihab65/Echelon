//! Global stiffness matrix assembly.
//!
//! `assemble_stiffness` is the innermost loop of every Newton iteration.
//! It iterates over all elements, extracts their local displacements from
//! `model.u_global`, computes the tangent stiffness via `element.ke_flat`,
//! and scatters the result into the global `SymCsrMatrix`.
//!
//! ## Stateless kinematics
//!
//! Elements never store trial displacements. The assembly crate is
//! responsible for extracting `u_local` and passing it to the element,
//! which avoids the classic bug of assembling stiffness from a stale state.
//!
//! ## Zero before assemble
//!
//! `assemble_stiffness` calls `k.zero()` at the start of every call so the
//! caller does not need to remember to do it. This is deliberate: always
//! zeroing inside the function eliminates Trap 1 from the design document.

use sparse::SymCsrMatrix;

use crate::error::Result;
use crate::kinematics::extract::extract_local_u;
use crate::model::Model;

// -----------------------------------------------------------------
// assemble_stiffness
// -----------------------------------------------------------------

/// Assemble the global tangent stiffness matrix.
///
/// For each element:
/// 1. Extract its local displacement vector from `model.u_global`.
/// 2. Call `element.ke_flat(&u_local)` → dense row-major `n²` stiffness.
/// 3. Scatter the upper triangle into `k` via `k.scatter_add`.
///
/// `k` is zeroed at the start of every call — do not pre-zero it.
///
/// # Arguments
/// * `model` — read-only model (elements + current `u_global`)
/// * `k`     — the global stiffness matrix (must have the pattern from
///             `topology::build_pattern`; only values are overwritten)
///
/// # Errors
/// Propagates [`SparseError::IndexOutOfBounds`] if any element DOF map
/// references a position absent from `k`'s pattern (topology mismatch).
///
/// # Calling convention
///
/// ```text
/// // Per Newton iteration:
/// k.zero();  ← done internally by assemble_stiffness
/// assemble_stiffness(&model, &mut k)?;
/// assemble_load_vector(&model, pseudo_time, &mut f_ext)?;
/// assemble_internal_force(&model, &mut f_int)?;
/// apply_dirichlet_bcs(&model.constraints, &mut k, &mut f_residual)?;
/// solver.factorize(&k)?;
/// solver.solve(&f_residual, &mut delta_u)?;
/// ```
pub fn assemble_stiffness(model: &Model, k: &mut SymCsrMatrix<f64>) -> Result<()> {
    k.zero();

    for element in &model.elements {
        let dof_map = element.dof_map();
        let u_local = extract_local_u(&model.u_global, dof_map);

        let ke = element.ke_flat(&u_local);
        k.scatter_add(&ke, dof_map.as_usize_slice())?;
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
    use elements::{Truss2d, ElasticBeam2d};

    use crate::model::{Model, Node};
    use crate::topology::sparsity::build_pattern;

    fn steel() -> ElasticUniaxial {
        ElasticUniaxial::new(200e9, None).unwrap()
    }

    // ---- 2-node horizontal truss ----

    fn two_node_truss() -> (Model, SymCsrMatrix<f64>) {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element(
            Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0, steel(), 0.01).unwrap()
        );
        m.build_state();
        let k = build_pattern(&m).unwrap();
        (m, k)
    }

    #[test]
    fn truss_stiffness_correct_axial_diagonal() {
        let (m, mut k) = two_node_truss();
        assemble_stiffness(&m, &mut k).unwrap();

        let ea_over_l = 200e9 * 0.01 / 2.0; // 1e9
        // Horizontal truss: k[0,0] = EA/L, k[2,2] = EA/L
        assert!((k.get(0, 0).unwrap() - ea_over_l).abs() < 1e3);
        assert!((k.get(2, 2).unwrap() - ea_over_l).abs() < 1e3);
        // Off-diagonal k[0,2] = -EA/L (upper triangle)
        assert!((k.get(0, 2).unwrap() + ea_over_l).abs() < 1e3);
    }

    #[test]
    fn truss_stiffness_is_symmetric() {
        let (m, mut k) = two_node_truss();
        assemble_stiffness(&m, &mut k).unwrap();
        // SymCsrMatrix always mirrors; check the get() accessor for both triangles
        assert!((k.get(0, 2).unwrap() - k.get(2, 0).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn second_call_zeros_and_reassembles() {
        let (m, mut k) = two_node_truss();
        assemble_stiffness(&m, &mut k).unwrap();
        let v1 = k.get(0, 0).unwrap();

        // Second call should give identical result (not doubled)
        assemble_stiffness(&m, &mut k).unwrap();
        let v2 = k.get(0, 0).unwrap();

        assert!((v1 - v2).abs() < 1e-10);
    }

    // ---- 2-node beam ----

    fn two_node_beam() -> (Model, SymCsrMatrix<f64>) {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                steel(), 0.01, 1e-4,
            ).unwrap()
        );
        m.build_state();
        let k = build_pattern(&m).unwrap();
        (m, k)
    }

    #[test]
    fn beam_stiffness_correct_bending_diagonal() {
        let (m, mut k) = two_node_beam();
        assemble_stiffness(&m, &mut k).unwrap();

        let e = 200e9_f64;
        let iz = 1e-4_f64;
        let l = 2.0_f64;
        let b1 = 12.0 * e * iz / (l * l * l); // K[1,1]
        let b3 =  4.0 * e * iz / l;            // K[2,2]

        assert!((k.get(1, 1).unwrap() - b1).abs() / b1 < 1e-10);
        assert!((k.get(2, 2).unwrap() - b3).abs() / b3 < 1e-10);
    }

    #[test]
    fn beam_stiffness_matvec_rigid_body_zero() {
        // Rigid-body axial translation: all nodes move +1 in x
        // → K * u_rigid = 0 (zero net force on an unconstrained structure)
        let (m, mut k) = two_node_beam();
        assemble_stiffness(&m, &mut k).unwrap();

        let u_rigid = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let ku = k.matvec(&u_rigid).unwrap();
        for (i, &v) in ku.iter().enumerate() {
            assert!(v.abs() < 1e-6, "Ku[{i}] = {v:.3e} for rigid body — should be ≈0");
        }
    }
}