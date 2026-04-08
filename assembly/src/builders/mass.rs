//! Mass matrix assembly and gravity load vector.
//!
//! This module provides two functions:
//!
//! - **`assemble_mass`** — scatters lumped element mass matrices into the global
//!   `SymCsrMatrix`. Required for eigenvalue analysis (`Eigen` driver) and
//!   Newmark time integration (`Transient` driver).
//!
//! - **`assemble_self_weight`** — computes `F_gravity = M_e × g` for each
//!   element and scatters the result directly into a load vector. This is
//!   more convenient than writing distributed gravity load patterns by hand,
//!   and is exact for a lumped mass formulation.
//!
//! ## Density requirement
//!
//! Both functions require that every element's material has `rho: Some(value)`
//! set. If an element returns an all-zero mass vector, `assemble_mass` throws
//! [`AssemblyError::MissingDensity`] with the element's index, pointing the
//! user directly to the missing parameter.
//!
//! ## Lumped vs consistent mass
//!
//! Elements currently return lumped mass matrices (diagonal). A consistent
//! mass formulation is architecturally supported — elements would override
//! `mass_flat()` to return the full consistent matrix — but is not yet
//! implemented for the standard 2D elements.

use sparse::SymCsrMatrix;

use crate::error::{AssemblyError, Result};
use crate::model::Model;

// -----------------------------------------------------------------
// assemble_mass
// -----------------------------------------------------------------

/// Assemble the global mass matrix from element lumped mass contributions.
///
/// Zeros `m_global`, then for each element:
/// 1. Calls `element.mass_flat()` → flat `n²` mass matrix.
/// 2. Checks that the mass is non-zero (requires `rho` to be set).
/// 3. Scatters the upper triangle via `m_global.scatter_add`.
///
/// `m_global` must have the same sparsity pattern as the global stiffness
/// matrix (from `topology::build_pattern`).
///
/// # Errors
/// - [`AssemblyError::MissingDensity`] if element `element_idx` has an
///   all-zero mass matrix (material `rho` not set).
/// - [`SparseError`] from `scatter_add` on pattern mismatch.
pub fn assemble_mass(model: &Model, m_global: &mut SymCsrMatrix<f64>) -> Result<()> {
    m_global.zero();

    for (elem_idx, element) in model.elements.iter().enumerate() {
        let me = element.mass_flat();

        // Guard: if all entries are zero the element has no density.
        // This is always a user error when mass assembly is requested.
        let all_zero = me.iter().all(|&v| v.abs() < f64::EPSILON);
        if all_zero {
            return Err(AssemblyError::MissingDensity { element_idx: elem_idx });
        }

        let dof_map = element.dof_map();
        m_global.scatter_add(&me, dof_map.as_usize_slice())?;
    }

    Ok(())
}

// -----------------------------------------------------------------
// assemble_self_weight
// -----------------------------------------------------------------

/// Compute gravity loads `F = M_e × g` and scatter into `f_ext`.
///
/// For each element:
/// 1. Fetches the lumped mass vector (diagonal of `mass_flat()`).
/// 2. Multiplies the vertical (y-direction) mass entries by `gravity_accel`.
/// 3. Adds the result to `f_ext` at the corresponding vertical DOFs.
///
/// `gravity_accel` should be **negative** if downward loads are expressed as
/// negative forces in the global Y axis (structural convention). Typically:
/// `gravity_accel = -9.81` m/s².
///
/// This eliminates the need for explicit distributed gravity load patterns
/// for uniform self-weight loading. For non-uniform or partial self-weight,
/// use explicit `NodalLoad` patterns instead.
///
/// # DOF convention
///
/// For a 2D frame model (ndf = 3): DOF 1 of each node is UY.
/// For a 2D truss model (ndf = 2): DOF 1 of each node is UY.
/// The Y-DOF is always local DOF index 1 in the 2D models supported here.
///
/// # Errors
/// - [`AssemblyError::MissingDensity`] if any element has no density.
pub fn assemble_self_weight(
    model:         &Model,
    gravity_accel: f64,
    f_ext:         &mut [f64],
) -> Result<()> {
    let ndf = model.dim.ndf();

    for (elem_idx, element) in model.elements.iter().enumerate() {
        let me = element.mass_flat();
        let n  = element.n_dof();

        // Extract diagonal of the mass matrix (lumped mass per DOF)
        let n_local = (me.len() as f64).sqrt() as usize;
        let diagonal: Vec<f64> = (0..n_local)
            .map(|i| me[i * n_local + i])
            .collect();

        // Detect missing density by checking only the Y-direction (translational)
        // diagonal entries — rotation DOFs legitimately carry near-zero mass
        // (stub values like 1e-9) even without density, so checking all entries
        // would give false negatives for beam elements.
        let y_mass_zero = (0..n).all(|local_dof| {
            let node_local_dof = local_dof % ndf;
            if node_local_dof == 1 {
                diagonal[local_dof].abs() < f64::EPSILON
            } else {
                true // non-Y DOFs don't count for this check
            }
        });
        if y_mass_zero {
            return Err(AssemblyError::MissingDensity { element_idx: elem_idx });
        }

        let dof_map    = element.dof_map();
        let global_dofs = dof_map.as_usize_slice();

        // Scatter gravity force: F_y = mass_at_dof × gravity_accel
        // Only vertical (Y) DOFs carry gravity — local DOF 1 within each node.
        for local_dof in 0..n {
            let node_local_dof = local_dof % ndf; // which DOF within the node
            if node_local_dof == 1 {
                // This is a Y-displacement DOF
                let global_dof = global_dofs[local_dof];
                if global_dof < f_ext.len() {
                    f_ext[global_dof] += diagonal[local_dof] * gravity_accel;
                }
            }
        }

        let _ = n;
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

    fn steel_with_rho() -> ElasticUniaxial {
        // Steel: E = 200 GPa, rho = 7850 kg/m³
        ElasticUniaxial::new(200e9, Some(7850.0)).unwrap()
    }

    fn steel_no_rho() -> ElasticUniaxial {
        ElasticUniaxial::new(200e9, None).unwrap()
    }

    fn two_node_truss_with_rho() -> (Model, SymCsrMatrix<f64>) {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element_typed(
            Truss2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                steel_with_rho(), 0.01,
            ).unwrap()
        );
        m.build_state();
        let k = build_pattern(&m).unwrap();
        (m, k)
    }

    fn two_node_beam_with_rho() -> (Model, SymCsrMatrix<f64>) {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element_typed(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                steel_with_rho(), 0.01, 1e-4,
            ).unwrap()
        );
        m.build_state();
        let k = build_pattern(&m).unwrap();
        (m, k)
    }

    // ---- assemble_mass ----

    #[test]
    fn mass_truss_positive_diagonal() {
        let (m, mut mg) = two_node_truss_with_rho();
        assemble_mass(&m, &mut mg).unwrap();
        // All diagonal entries must be positive
        for i in 0..4 {
            assert!(mg.get(i, i).unwrap() > 0.0, "m[{i},{i}] should be > 0");
        }
    }

    #[test]
    fn mass_truss_total_mass_conserved() {
        let (m, mut mg) = two_node_truss_with_rho();
        assemble_mass(&m, &mut mg).unwrap();
        // Lumped truss mass: m_half = rho*A*L/2 assigned to each of 4 DOFs
        // (UX and UY at each of 2 nodes).
        // Sum of diagonal = 4 * m_half = 2 * rho * A * L
        // This is physically correct: each translational DOF carries half the element mass.
        let m_total  = 7850.0 * 0.01 * 2.0; // rho * A * L = 157 kg
        let expected = 2.0 * m_total;        // 4 DOFs × m_half = 2 × m_total = 314 kg
        let total_mass: f64 = (0..4).map(|i| mg.get(i, i).unwrap()).sum();
        assert!((total_mass - expected).abs() / expected < 1e-10,
            "total mass {total_mass:.4} != expected {expected:.4}");
    }

    #[test]
    fn mass_missing_density_errors() {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element_typed(
            Truss2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                steel_no_rho(), 0.01,
            ).unwrap()
        );
        m.build_state();
        let mut mg = build_pattern(&m).unwrap();

        let err = assemble_mass(&m, &mut mg).unwrap_err();
        assert!(matches!(err, AssemblyError::MissingDensity { element_idx: 0 }));
    }

    #[test]
    fn mass_second_call_does_not_double() {
        let (m, mut mg) = two_node_truss_with_rho();
        assemble_mass(&m, &mut mg).unwrap();
        let v1 = mg.get(0, 0).unwrap();
        assemble_mass(&m, &mut mg).unwrap();
        let v2 = mg.get(0, 0).unwrap();
        assert!((v1 - v2).abs() < 1e-10);
    }

    // ---- assemble_self_weight ----

    #[test]
    fn self_weight_beam_y_dofs_negative() {
        let (m, _) = two_node_beam_with_rho();
        let mut f = vec![0.0_f64; 6];
        assemble_self_weight(&m, -9.81, &mut f).unwrap();

        // Y DOFs (indices 1 and 4) should be negative (downward gravity)
        assert!(f[1] < 0.0, "f[1]={} — Y force at node 0 should be negative", f[1]);
        assert!(f[4] < 0.0, "f[4]={} — Y force at node 1 should be negative", f[4]);
        // X and rotation DOFs should be zero
        assert_eq!(f[0], 0.0); // node 0 UX
        assert_eq!(f[2], 0.0); // node 0 RZ
        assert_eq!(f[3], 0.0); // node 1 UX
        assert_eq!(f[5], 0.0); // node 1 RZ
    }

    #[test]
    fn self_weight_total_force_equals_total_weight() {
        let (m, _) = two_node_beam_with_rho();
        let mut f = vec![0.0_f64; 6];
        assemble_self_weight(&m, -9.81, &mut f).unwrap();

        let total_weight = f.iter().sum::<f64>();
        // Total gravity load = -rho * A * L * g = -7850 * 0.01 * 2 * 9.81
        let expected = -7850.0 * 0.01 * 2.0 * 9.81;
        assert!((total_weight - expected).abs() / expected.abs() < 1e-10,
            "total weight {total_weight:.4} != expected {expected:.4}");
    }

    #[test]
    fn self_weight_missing_density_errors() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element_typed(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                steel_no_rho(), 0.01, 1e-4,
            ).unwrap()
        );
        m.build_state();
        let mut f = vec![0.0_f64; 6];
        let err = assemble_self_weight(&m, -9.81, &mut f).unwrap_err();
        assert!(matches!(err, AssemblyError::MissingDensity { element_idx: 0 }));
    }
}