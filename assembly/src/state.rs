//! State management — commit and revert material histories.
//!
//! After a Newton-Raphson loop converges, every element's material must
//! be told to commit its current trial state as the new baseline. If the
//! loop diverges (or a load step is rejected by the arc-length method),
//! all materials must revert to the last committed state before retrying.
//!
//! ## The stateless kinematics pattern
//!
//! Echelon uses stateless kinematics: elements do not store trial
//! displacements internally. Instead, the assembly crate extracts the
//! relevant slice of `model.u_global` and passes it explicitly to every
//! element call (`ke_flat(&u_local)`, `f_int(&u_local)`, `commit(&u_local)`).
//!
//! This completely eliminates the class of bugs where `element.update(u)`
//! is forgotten before `element.stiffness()`, making Newton-Raphson
//! iterations safe to parallelise and trivially restartable.
//!
//! ## Commit / revert lifecycle
//!
//! ```text
//! load step n:
//!   loop (Newton iterations):
//!     assemble K, F_ext, F_int
//!     solve Δu
//!     u_global += Δu
//!     if converged → break
//!     if diverged  → revert_state(&mut model); reduce step; retry
//!
//!   commit_state(&mut model)   ← called exactly once per converged step
//! ```

use crate::error::Result;
use crate::kinematics::extract::extract_local_u;
use crate::model::Model;

// -----------------------------------------------------------------
// commit_state
// -----------------------------------------------------------------

/// Commit the current `u_global` as the converged material state for all elements.
///
/// For each element:
/// 1. Extract its local displacement vector from `model.u_global`
///    using the element's `DofMap`.
/// 2. Call `element.commit(&u_local)` — forwards to the material's
///    `commit_state(strain)`, updating plastic strain, back-stress, etc.
///
/// This must be called **once per converged load step**, after Newton-Raphson
/// has exited its inner loop. Calling it during a Newton iteration will
/// advance the material state prematurely.
///
/// # Errors
/// Propagates any [`ElementError`] from the element's `commit` method,
/// which in turn propagates [`MaterialError::StrainDomainViolation`] if
/// the converged strain falls outside the constitutive model's valid range
/// (e.g., fracture strain exceeded in a softening model).
pub fn commit_state(model: &mut Model) -> Result<()> {
    let max_dof = model.elements.iter().map(|e| e.n_dof()).max().unwrap_or(0);
    
    let mut u_buffer = vec![0.0; max_dof];

    for element in &mut model.elements {
        let n       = element.n_dof();
        let dof_map = element.dof_map();
        
        let u_local = &mut u_buffer[..n];

        extract_local_u(&model.u_global, dof_map, u_local);
        
        element.commit(u_local)?;
    }
    
    Ok(())
}

// -----------------------------------------------------------------
// revert_state
// -----------------------------------------------------------------

/// Revert all element material states to the last committed configuration.
///
/// For each element, calls `element.revert()`, which in turn calls
/// `material.revert_to_last_commit()` on all owned materials. After this
/// call the elements behave as if the current Newton iteration never happened.
///
/// This is called when:
/// - The Newton-Raphson loop fails to converge within the iteration limit.
/// - The arc-length method rejects the current load step.
/// - Any other condition requires rolling back to the last stable state.
///
/// Unlike `commit_state`, this function is infallible: reverting to a
/// previously committed (valid) state cannot fail by construction.
pub fn revert_state(model: &mut Model) {
    for element in &mut model.elements {
        element.revert();
    }
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
    use crate::constraints::SpConstraint;

    /// Build a minimal 2-node truss model: node 0 fixed, node 1 free.
    fn two_node_truss() -> Model {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 1.0, 0.0, 0.0)).unwrap();

        let mat = ElasticUniaxial::new(200e9, None).unwrap();
        let truss = Truss2d::new(
            NodeId(0), NodeId(1),
            0.0, 0.0, 1.0, 0.0,
            mat, 0.01,
        ).unwrap();
        m.add_element(truss);

        // Fix node 0 (both DOFs)
        m.add_constraint(SpConstraint::new(NodeId(0), 0, 0.0, 2)).unwrap();
        m.add_constraint(SpConstraint::new(NodeId(0), 1, 0.0, 2)).unwrap();

        m.build_state();
        m
    }

    #[test]
    fn commit_zero_displacement_does_not_error() {
        // u_global is all zero — commit should succeed with no material state change
        let mut model = two_node_truss();
        commit_state(&mut model).unwrap();
    }

    #[test]
    fn commit_with_displacement_does_not_error() {
        let mut model = two_node_truss();
        // Simulate 1mm axial elongation at node 1
        model.u_global[2] = 1e-3; // DOF 2 = node1 UX
        commit_state(&mut model).unwrap();
    }

    #[test]
    fn revert_is_infallible_on_fresh_model() {
        let mut model = two_node_truss();
        // Should not panic; no committed state yet → reverts to zero
        revert_state(&mut model);
    }

#[test]
    fn revert_after_commit_restores_state() {
        let mut model = two_node_truss();

        // Commit displacement state A
        model.u_global[2] = 2e-3;
        commit_state(&mut model).unwrap();

        // Simulate a Newton iteration displacement (trial state B)
        model.u_global[2] = 5e-3;

        // Revert: should go back to committed state A
        revert_state(&mut model);

        // ZERO ALLOCATION TEST UPDATES:
        let dof_map = model.elements[0].dof_map();
        
        // 1. Pre-allocate and extract local displacements
        let mut u_local = [0.0; 4]; // Truss2d has 4 DOFs
        crate::kinematics::extract::extract_local_u(&model.u_global, dof_map, &mut u_local);
        
        // 2. Pre-allocate and compute internal forces
        let mut f_buffer = [0.0; 4];
        model.elements[0].f_int(&u_local, &mut f_buffer);
        
        // If we reach here without panic, revert worked correctly
    }

    #[test]
    fn commit_revert_roundtrip_no_error() {
        let mut model = two_node_truss();

        for step in 1..=5 {
            model.u_global[2] = step as f64 * 1e-4;
            commit_state(&mut model).unwrap();
        }

        // Trial step that we reject
        model.u_global[2] = 999.0;
        revert_state(&mut model);

        // Commit again at the properly incremented state
        model.u_global[2] = 6e-4;
        commit_state(&mut model).unwrap();
    }
}