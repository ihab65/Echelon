//! Stateless kinematics — extract local displacements from the global vector.
//!
//! In Echelon elements never store trial displacements internally. Instead,
//! every builder function extracts the relevant slice of `u_global` and
//! passes it explicitly to the element. This eliminates the class of bugs
//! where `element.update(u)` is forgotten before `element.stiffness()`.
//!
//! ## The gathering operation
//!
//! Given a global displacement vector `u_global[0..n_dof]` and an element's
//! `DofMap` (which maps local DOF index to global DOF index), the local
//! displacement vector is:
//!
//! ```text
//! u_local[i] = u_global[ dof_map[i] ]   for i in 0..n_local
//! ```
//!
//! This is a pure gather — no arithmetic, no allocation beyond the returned
//! `Vec`. For elements with 4–6 DOFs the returned `Vec` is tiny and stack-
//! allocated by the optimiser in most cases.

use fem_core::DofMap;

// -----------------------------------------------------------------
// extract_local_u
// -----------------------------------------------------------------

/// Gather the local displacement vector for one element from `u_global`.
///
/// # Arguments
/// * `u_global` — the full global displacement vector, length `n_dof`
/// * `dof_map`  — the element's DOF map (maps local index → global index)
///
/// # Returns
/// A `Vec<f64>` of length `dof_map.n_local()` containing the displacements
/// at each of the element's DOFs in local index order.
///
/// # Panics
/// In debug mode, panics if any global DOF index in `dof_map` is out of
/// range for `u_global`. In release mode this is a silent bounds check.
///
/// # Example
///
/// ```rust,ignore
/// let u_local = extract_local_u(&model.u_global, element.dof_map());
/// let ke = element.ke_flat(&u_local);
/// ```
pub fn extract_local_u(u_global: &[f64], dof_map: &DofMap, out: &mut [f64]) {
    debug_assert_eq!(out.len(), dof_map.n_local());

    for (i, &global_idx) in dof_map.as_usize_slice().iter().enumerate() {
        debug_assert!(
            global_idx < u_global.len(),
            "DOF index {global_idx} out of range for u_global of length {}",
            u_global.len()
        );
        out[i] = u_global[global_idx];
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{DofMap, NodeId};

    #[test]
    fn extract_contiguous_block() {
        // Node 1 in a 3-DOF model: global DOFs 3, 4, 5
        let u_global = vec![0.0, 0.1, 0.2, 1.0, 2.0, 3.0];
        let dof_map  = DofMap::from_nodes(&[NodeId(1)], 3);
        
        let mut u_local = vec![0.0; dof_map.n_local()];
        extract_local_u(&u_global, &dof_map, &mut u_local);
        
        assert_eq!(u_local, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn extract_two_node_element() {
        // 2D truss: node 0 (DOFs 0,1) and node 2 (DOFs 4,5)
        let u_global = vec![0.1, 0.2, 0.0, 0.0, 0.3, 0.4];
        let dof_map  = DofMap::from_nodes(&[NodeId(0), NodeId(2)], 2);
        
        let mut u_local = vec![0.0; dof_map.n_local()];
        extract_local_u(&u_global, &dof_map, &mut u_local);
        
        assert_eq!(u_local, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn extract_all_zeros_initial_state() {
        let u_global = vec![0.0; 6];
        let dof_map  = DofMap::from_nodes(&[NodeId(0), NodeId(1)], 3);
        
        let mut u_local = vec![0.0; dof_map.n_local()];
        extract_local_u(&u_global, &dof_map, &mut u_local);
        
        assert!(u_local.iter().all(|&v| v == 0.0));
        assert_eq!(u_local.len(), 6);
    }

    #[test]
    fn extract_single_dof_element() {
        let u_global = vec![5.0, 10.0, 15.0];
        let dof_map  = DofMap::from_nodes(&[NodeId(1)], 1);
        
        // You can also use a fixed-size stack array for the buffer
        let mut u_local = [0.0; 1];
        extract_local_u(&u_global, &dof_map, &mut u_local);
        
        assert_eq!(u_local, [10.0]);
    }

    #[test]
    fn length_matches_n_local() {
        let u_global = vec![0.0; 12];
        let dof_map  = DofMap::from_nodes(&[NodeId(0), NodeId(1)], 3);
        
        let mut u_local = vec![0.0; dof_map.n_local()];
        extract_local_u(&u_global, &dof_map, &mut u_local);
        
        assert_eq!(u_local.len(), dof_map.n_local());
    }
}