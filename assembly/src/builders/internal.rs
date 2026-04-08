//! Internal (resisting) force vector assembly.
//!
//! `assemble_internal_force` computes the vector of element resisting forces
//! at the current displacement state `model.u_global`. Together with the
//! external load vector it forms the unbalanced force (residual) used by the
//! Newton-Raphson solver:
//!
//! ```text
//! r = f_ext - f_int
//! ```
//!
//! For a linear elastic model at equilibrium, `r ≈ 0`. During Newton
//! iterations `r` drives the incremental displacement update `Δu`.
//!
//! ## Stateless kinematics
//!
//! Like `assemble_stiffness`, this function extracts `u_local` from
//! `model.u_global` and passes it explicitly to `element.f_int(&u_local)`.
//! Elements never store trial displacements internally.

use crate::error::Result;
use crate::kinematics::extract::extract_local_u;
use crate::model::Model;

// -----------------------------------------------------------------
// assemble_internal_force
// -----------------------------------------------------------------

/// Assemble the global internal (resisting) force vector.
///
/// Zeros `f_int`, then for each element:
/// 1. Extracts `u_local` from `model.u_global`.
/// 2. Calls `element.f_int(&u_local)` → local resisting force vector.
/// 3. Scatters by direct DOF-indexed addition into `f_int`.
///
/// # Arguments
/// * `model` — read-only model (elements + current `u_global`)
/// * `f_int` — mutable global internal force vector, length `model.n_dof()`
///
/// # Errors
/// None expected in normal operation. The function returns `Result` for
/// future compatibility (e.g., nonlinear elements that can detect ill-posed
/// states during `f_int` evaluation).
pub fn assemble_internal_force(model: &Model, f_int: &mut [f64]) -> Result<()> {
    f_int.fill(0.0);

    for element in &model.elements {
        let dof_map = element.dof_map();
        let u_local = extract_local_u(&model.u_global, dof_map);

        let fe = element.f_int(&u_local);
        let global_dofs = dof_map.as_usize_slice();

        // Manual scatter — same logic as the general CSR scatter_add but
        // operating on a plain slice rather than a sparse matrix.
        for (local_idx, &val) in fe.iter().enumerate() {
            let global_dof = global_dofs[local_idx];
            f_int[global_dof] += val;
        }
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

    fn steel() -> ElasticUniaxial {
        ElasticUniaxial::new(200e9, None).unwrap()
    }

    fn two_node_truss() -> Model {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element(
            Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0, steel(), 0.01).unwrap()
        );
        m.build_state();
        m
    }

    fn two_node_beam() -> Model {
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
        m
    }

    // ---- Zero displacement → zero internal force ----

    #[test]
    fn zero_displacement_gives_zero_internal_force_truss() {
        let m = two_node_truss();
        let mut f_int = vec![0.0_f64; 4];
        assemble_internal_force(&m, &mut f_int).unwrap();
        assert!(f_int.iter().all(|&v| v.abs() < 1e-14));
    }

    #[test]
    fn zero_displacement_gives_zero_internal_force_beam() {
        let m = two_node_beam();
        let mut f_int = vec![0.0_f64; 6];
        assemble_internal_force(&m, &mut f_int).unwrap();
        assert!(f_int.iter().all(|&v| v.abs() < 1e-14));
    }

    // ---- Axial extension → correct resisting force ----

    #[test]
    fn axial_elongation_correct_resisting_force() {
        let mut m = two_node_truss();
        // Elongate node 1 UX by 1 mm
        m.u_global[2] = 1e-3; // DOF 2 = node1 UX

        let mut f_int = vec![0.0_f64; 4];
        assemble_internal_force(&m, &mut f_int).unwrap();

        let ea_over_l = 200e9 * 0.01 / 2.0;
        let force     = ea_over_l * 1e-3;

        // Node 0 UX: pushed left (negative)
        assert!((f_int[0] + force).abs() < 1e3,
            "f_int[0]={:.4e} expected {:.4e}", f_int[0], -force);
        // Node 1 UX: pushed right (positive)
        assert!((f_int[2] - force).abs() < 1e3,
            "f_int[2]={:.4e} expected {:.4e}", f_int[2], force);
        // Transverse forces zero for horizontal truss
        assert!(f_int[1].abs() < 1e-6);
        assert!(f_int[3].abs() < 1e-6);
    }

    // ---- f_int accumulates across multiple elements ----

    #[test]
    fn two_elements_accumulate_internal_forces() {
        // Two trusses in series: node 0 – node 1 – node 2
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 1.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(2), 2.0, 0.0, 0.0)).unwrap();
        let ea = 200e9 * 0.01;
        m.add_element(
            Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, 1.0, 0.0, steel(), 0.01).unwrap()
        );
        m.add_element(
            Truss2d::new(NodeId(1), NodeId(2), 1.0, 0.0, 2.0, 0.0, steel(), 0.01).unwrap()
        );
        m.build_state();

        // Apply 1 mm extension at node 1 UX
        m.u_global[2] = 1e-3;

        let mut f_int = vec![0.0_f64; 6];
        assemble_internal_force(&m, &mut f_int).unwrap();

        // Node 0 UX: only element 0 contributes → -EA*u/L = -ea * 1e-3
        let force0 = ea * 1e-3;
        assert!((f_int[0] + force0).abs() < 1e3,
            "f_int[0]={:.4e}", f_int[0]);

        // Node 1 UX receives contributions from both elements:
        // elem0 (nodes 0→1, L=1): ε = (1e-3 - 0)/1 = +1e-3 → f_node1 = +EA/L * 1e-3 = +ea*1e-3
        // elem1 (nodes 1→2, L=1): u_local = [1e-3, 0, 0, 0], ε = (0 - 1e-3)/1 = -1e-3
        //   → f_node1 (local DOF 0, which is node1) = -EA/L * (-1e-3) ... but sign convention:
        //   f_int = EA/L * ε * [-c, -s, c, s] with c=1,s=0:
        //   f = EA * (-1e-3) * [-1, 0, 1, 0] → node1_ux (local 0) = +EA*1e-3
        // Total at node1 UX: +EA*1e-3 + EA*1e-3 = 2*EA*1e-3
        let ea_over_l = 200e9 * 0.01 / 1.0; // L=1 for both elements
        let expected_node1 = 2.0 * ea_over_l * 1e-3;
        assert!((f_int[2] - expected_node1).abs() < 1e3,
            "f_int[2] (node1 UX)={:.4e} — expected {:.4e}", f_int[2], expected_node1);
    }

    // ---- f_int is always re-zeroed on each call ----

    #[test]
    fn second_call_does_not_accumulate() {
        let mut m = two_node_truss();
        m.u_global[2] = 1e-3;

        let mut f_int = vec![0.0_f64; 4];
        assemble_internal_force(&m, &mut f_int).unwrap();
        let v1 = f_int[0];

        assemble_internal_force(&m, &mut f_int).unwrap();
        let v2 = f_int[0];

        assert!((v1 - v2).abs() < 1e-10,
            "Second call gave different result: {v1:.4e} vs {v2:.4e}");
    }
}