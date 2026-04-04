//! Adjoint sensitivity assembly — Engine B parameter gradient vector.
//!
//! `assemble_partial_residual` constructs the vector `∂F_int/∂θ` for a
//! single scalar parameter `θ` (e.g. Young's modulus `E` of element `k`,
//! or cross-section area `A`). This vector is the right-hand side of the
//! adjoint equation:
//!
//! ```text
//! K_T λ = ∂J/∂u                     (adjoint solve — K_T already factored)
//! dJ/dθ = -λᵀ · (∂F_int/∂θ)         (scalar sensitivity)
//! ```
//!
//! where `J` is a scalar response quantity (e.g. tip displacement, max drift).
//!
//! ## Parameter index convention
//!
//! Each element exposes `n_params()` parameters, numbered locally starting
//! at 0. The *global* parameter index `param_idx` passed to this function
//! is an offset into a flat parameter vector that spans all elements:
//!
//! ```text
//! global_param_offset[elem k] = Σ_{i<k} element_i.n_params()
//! ```
//!
//! `assemble_partial_residual` iterates every element, skips those whose
//! local parameter range does not contain `param_idx`, and calls
//! `element.partial_residual_wrt_param(&u_local, local_param_idx)` on the
//! matching element.
//!
//! ## Why not thread `Dual64` through everything?
//!
//! For smooth elastic elements the dual-number path through `energy<T>` is
//! elegant. For history-dependent materials (Steel01, Concrete01) the
//! return-mapping algorithm is non-smooth — threading duals through it gives
//! wrong gradients at yield. The adjoint method computes the exact gradient
//! using the already-factored `K_T`, making it correct for both smooth and
//! inelastic materials.
//!
//! See `elements/src/traits/assembleable.rs` for the element-level contract.

use crate::error::{AssemblyError, Result};
use crate::kinematics::extract::extract_local_u;
use crate::model::Model;

// -----------------------------------------------------------------
// assemble_partial_residual
// -----------------------------------------------------------------

/// Assemble `∂F_int/∂θ` for the global parameter at index `param_idx`.
///
/// Zeros `dp_dtheta`, finds which element owns `param_idx`, extracts its
/// local displacements, calls `element.partial_residual_wrt_param`, and
/// scatters the result into the global vector.
///
/// # Arguments
/// * `model`       — read-only model at the converged displacement state
/// * `param_idx`   — global parameter index (flat across all elements)
/// * `dp_dtheta`   — output vector `∂F_int/∂θ`, length `model.n_dof()`
///
/// # Global parameter layout
///
/// The global parameter vector is formed by concatenating each element's
/// local parameter block in element order:
///
/// ```text
/// [E_0, A_0 | E_1, A_1, Iz_1 | E_2 | ...]
///  ← elem 0 →← elem 1 (beam) →← elem 2 →
/// ```
///
/// `param_idx = 0` → `E` of element 0.
/// `param_idx = 2` → `E` of element 1 (beam, which has 3 params: E, A, Iz).
///
/// # Errors
/// - [`AssemblyError::Element`] wrapping an [`ElementError::UnregisteredParameter`]
///   if `param_idx` is out of range for the entire model (no element owns it).
/// - Any other [`ElementError`] from `partial_residual_wrt_param`.
pub fn assemble_partial_residual(
    model:      &Model,
    param_idx:  usize,
    dp_dtheta:  &mut [f64],
) -> Result<()> {
    dp_dtheta.fill(0.0);

    let mut offset = 0_usize;

    for element in &model.elements {
        let n_params = element.n_params();
        let end      = offset + n_params;

        // Check whether this element owns `param_idx`
        if param_idx >= offset && param_idx < end {
            let local_param = param_idx - offset;

            let dof_map    = element.dof_map();
            let u_local    = extract_local_u(&model.u_global, dof_map);
            let global_dofs = dof_map.as_usize_slice();

            // Delegate to the element's analytical ∂f_int/∂θ
            let df_local = element.partial_residual_wrt_param(&u_local, local_param)?;

            // Scatter into dp_dtheta
            for (local_i, &val) in df_local.iter().enumerate() {
                let g = global_dofs[local_i];
                if g < dp_dtheta.len() {
                    dp_dtheta[g] += val;
                }
            }

            // Found the owning element — no need to continue
            return Ok(());
        }

        offset = end;
    }

    // param_idx was beyond the last element's parameter range
    Err(AssemblyError::Element(
        elements::error::ElementError::UnregisteredParameter {
            element_type: "model",
            idx:          param_idx,
            n_params:     offset, // total params across all elements
        }
    ))
}

// -----------------------------------------------------------------
// total_n_params helper
// -----------------------------------------------------------------

/// Return the total number of sensitivity parameters across all elements.
///
/// This is the length of the flat global parameter vector that
/// `assemble_partial_residual` indexes into.
pub fn total_n_params(model: &Model) -> usize {
    model.elements.iter().map(|e| e.n_params()).sum()
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

    // ---- single truss (2 params: E, A) ----

    fn single_truss_model() -> Model {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0)).unwrap();
        m.add_element_typed(
            Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0, steel(), 0.01).unwrap()
        );
        m.build_state();
        m
    }

    #[test]
    fn total_params_single_truss() {
        let m = single_truss_model();
        assert_eq!(total_n_params(&m), 2); // E and A
    }

    #[test]
    fn zero_displacement_gives_zero_sensitivity() {
        // At zero displacement ε = 0 → ∂f/∂E = 0 and ∂f/∂A = 0
        let m = single_truss_model();
        let mut dp = vec![0.0_f64; 4];

        assemble_partial_residual(&m, 0, &mut dp).unwrap(); // ∂/∂E
        assert!(dp.iter().all(|&v| v.abs() < 1e-20),
            "∂f/∂E at zero strain should be zero: {dp:?}");

        assemble_partial_residual(&m, 1, &mut dp).unwrap(); // ∂/∂A
        assert!(dp.iter().all(|&v| v.abs() < 1e-20),
            "∂f/∂A at zero strain should be zero: {dp:?}");
    }

    #[test]
    fn sensitivity_e_direction_correct_under_axial_strain() {
        // Apply 1 mm axial elongation at node 1
        let mut m = single_truss_model();
        m.u_global[2] = 1e-3; // node 1 UX

        let mut dp = vec![0.0_f64; 4];
        assemble_partial_residual(&m, 0, &mut dp).unwrap(); // ∂/∂E

        // For horizontal truss, ∂f/∂E = (A/L)*ε * [-1, 0, 1, 0]
        // A = 0.01, L = 2, ε = 1e-3/2 = 5e-4
        // scale = (0.01/2) * 5e-4 = 2.5e-6
        let expected_axial = 0.01 * (1e-3 / 2.0);
        assert!((dp[0] + expected_axial).abs() < 1e-18,
            "dp[0]={:.4e} expected {:.4e}", dp[0], -expected_axial);
        assert!((dp[2] - expected_axial).abs() < 1e-18,
            "dp[2]={:.4e} expected {:.4e}", dp[2],  expected_axial);
        // Transverse components are zero
        assert!(dp[1].abs() < 1e-20);
        assert!(dp[3].abs() < 1e-20);
    }

    #[test]
    fn out_of_range_param_idx_errors() {
        let m = single_truss_model();
        let mut dp = vec![0.0_f64; 4];
        // Truss has 2 params (indices 0 and 1); index 2 is out of range
        let err = assemble_partial_residual(&m, 2, &mut dp).unwrap_err();
        assert!(matches!(err, AssemblyError::Element(_)));
    }

    // ---- two elements: truss (2 params) + beam (3 params) ----
    // Global layout: [E_truss, A_truss | E_beam, A_beam, Iz_beam]
    //                 idx 0     idx 1     idx 2   idx 3   idx 4

    fn truss_beam_model() -> Model {
        let mut m = Model::new(ModelDim::frame_2d()); // beam needs 3 DOF/node
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(2), 4.0, 0.0)).unwrap();

        // Beam connecting node 0 and node 1 (3 params: E, A, Iz)
        m.add_element_typed(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                steel(), 0.01, 1e-4,
            ).unwrap()
        );
        // Beam connecting node 1 and node 2 (another 3 params)
        m.add_element_typed(
            ElasticBeam2d::new(
                NodeId(1), NodeId(2), 2.0, 0.0, 4.0, 0.0,
                steel(), 0.01, 1e-4,
            ).unwrap()
        );

        m.build_state();
        m
    }

    #[test]
    fn total_params_two_beams() {
        let m = truss_beam_model();
        assert_eq!(total_n_params(&m), 6); // 3 + 3
    }

    #[test]
    fn param_routing_second_element() {
        // param_idx = 3 should route to element 1 (beam), local param 0 (E)
        let mut m = truss_beam_model();
        // Apply tip displacement so sensitivity is non-zero
        m.u_global[3] = 1e-3; // node 1 UX

        let mut dp_elem1_e = vec![0.0_f64; 9];
        let mut dp_elem0_e = vec![0.0_f64; 9];

        assemble_partial_residual(&m, 0, &mut dp_elem0_e).unwrap(); // elem 0 E
        assemble_partial_residual(&m, 3, &mut dp_elem1_e).unwrap(); // elem 1 E

        // At the same displacement state both elements see the same axial strain.
        // The magnitudes should both be non-trivially small (ε ≈ 1e-3/2 = 5e-4).
        // The key check is that the routing correctly identifies different elements:
        // elem 0 affects DOFs 0..5 (nodes 0,1), elem 1 affects DOFs 3..8 (nodes 1,2).
        // DOFs 6..8 (node 2) are only affected by elem 1.
        assert!(dp_elem0_e[6].abs() < 1e-20,
            "elem 0 E should not affect node 2 DOFs");
        // elem 1 *does* affect node 2 DOFs (if there's a displacement-driven sensitivity)
        // At zero node-2 displacement and non-zero node-1 displacement both elements
        // are strained — but their scatter targets differ by element DOF map.
        // The test is that the two vectors are not identical (different elements)
        // and that no panic occurred.
        let different = dp_elem0_e.iter().zip(dp_elem1_e.iter())
            .any(|(a, b)| (a - b).abs() > 1e-30);
        assert!(different, "sensitivity of param 0 and param 3 should differ");
    }

    #[test]
    fn dp_dtheta_zeroed_on_each_call() {
        let mut m = single_truss_model();
        m.u_global[2] = 1e-3;

        let mut dp = vec![0.0_f64; 4];
        assemble_partial_residual(&m, 0, &mut dp).unwrap();
        let v1 = dp[0];

        assemble_partial_residual(&m, 0, &mut dp).unwrap();
        let v2 = dp[0];

        assert!((v1 - v2).abs() < 1e-25,
            "second call gave different result: {v1:.4e} vs {v2:.4e}");
    }
}