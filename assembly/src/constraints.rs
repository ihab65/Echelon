//! Dirichlet boundary conditions — constraint definitions and K/F modification.
//!
//! ## Design
//!
//! A [`SpConstraint`] ("support constraint") encodes a single prescribed
//! displacement: which node, which local DOF, and what value (usually 0.0
//! for a fixed support, non-zero for a settlement).
//!
//! The `global_dof` field is computed during model construction from the
//! `NodeId` and `local_dof` using the model's `ndf`:
//!
//! ```text
//! global_dof = node_id * ndf + local_dof
//! ```
//!
//! This is the same convention used everywhere in `fem_core::DofMap`, so
//! no separate DOF numbering pass is needed.
//!
//! ## Application order
//!
//! Per the assembly sequencing rules:
//! 1. `k.zero()` + full element scatter (`assemble_stiffness`)
//! 2. `assemble_load_vector` → `f_ext`
//! 3. `assemble_internal_force` → `f_int`
//! 4. **`apply_dirichlet_bcs`** — only after all scatter is complete
//! 5. `solve`
//!
//! Applying BCs before scatter is complete would be silently overwritten
//! by subsequent element contributions.

use fem_core::NodeId;
use sparse::SymCsrMatrix;

use crate::error::Result;

// -----------------------------------------------------------------
// SpConstraint
// -----------------------------------------------------------------

/// A single prescribed-displacement boundary condition.
///
/// "Sp" stands for "Single Point" — the OpenSees terminology for a
/// constraint that fixes one DOF at one node to a specified value.
///
/// # Example: fix UX, UY, and RZ at node 0 (fully clamped 2D frame)
///
/// ```rust,ignore
/// use assembly::constraints::SpConstraint;
/// use fem_core::NodeId;
///
/// let ndf = 3; // frame_2d
/// for local_dof in 0..ndf {
///     model.add_constraint(SpConstraint::new(NodeId(0), local_dof, 0.0, ndf)).unwrap();
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpConstraint {
    /// The user-facing node being constrained.
    pub node_id: NodeId,

    /// Local DOF index within the node (0 = UX, 1 = UY, 2 = RZ for 2D frame).
    /// Must be in `0..ndf`. Validated by `Model::add_constraint`.
    pub local_dof: usize,

    /// Target displacement value. `0.0` for a standard fixed support;
    /// non-zero for a prescribed settlement or imposed displacement.
    pub prescribed_value: f64,

    /// Global equation number: `node_id.0 * ndf + local_dof`.
    ///
    /// Computed at construction time and cached here so that
    /// `apply_dirichlet_bcs` can operate in O(n_constraints) without
    /// looking up the node each time.
    pub global_dof: usize,
}

impl SpConstraint {
    /// Construct a support constraint and compute the `global_dof` from
    /// the node's position and the model's `ndf`.
    ///
    /// # Arguments
    /// * `node_id`          — node being constrained
    /// * `local_dof`        — local DOF index within the node (0-based)
    /// * `prescribed_value` — target displacement (usually 0.0)
    /// * `ndf`              — DOFs per node from `ModelDim::ndf()`
    pub fn new(
        node_id:          NodeId,
        local_dof:        usize,
        prescribed_value: f64,
        ndf:              usize,
    ) -> Self {
        let global_dof = node_id.0 * ndf + local_dof;
        Self { node_id, local_dof, prescribed_value, global_dof }
    }
}

// -----------------------------------------------------------------
// apply_dirichlet_bcs
// -----------------------------------------------------------------

/// Apply all Dirichlet boundary conditions to the assembled system `(K, f)`.
///
/// For each constraint:
/// 1. Call `K.zero_row_col(global_dof)` — zeros the row and column in the
///    symmetric stiffness matrix and sets the diagonal to `1.0`.
/// 2. Set `f[global_dof] = prescribed_value` — replaces the load (which
///    accumulated contributions from load patterns) with the target
///    displacement value.
///
/// ## Must be called after all scatter is complete
///
/// Any element contribution scattered after this call would corrupt the BC.
/// Always apply BCs as the last step before solving:
///
/// ```text
/// assemble_stiffness(...)
/// assemble_load_vector(...)
/// assemble_internal_force(...)
/// apply_dirichlet_bcs(...)   ← here
/// solve(...)
/// ```
///
/// ## Residual convention
///
/// For the Newton-Raphson residual `r = f_ext - f_int`, BCs are applied
/// to the *assembled* K and the assembled residual vector `r`. Setting
/// `r[dof] = 0.0` (prescribed_value = 0.0) forces convergence at the
/// constrained DOF while leaving the rest of the system intact.
///
/// For non-zero prescribed displacements (settlements), `r[dof]` is set
/// to the target minus the current displacement, following the standard
/// penalty / substitution approach.
///
/// # Errors
/// Propagates [`sparse::SparseError`] from `SymCsrMatrix::zero_row_col` if the
/// DOF index is out of range for the pattern (indicates a topology mismatch).
pub fn apply_dirichlet_bcs(
    constraints: &[SpConstraint],
    k:           &mut SymCsrMatrix<f64>,
    f:           &mut [f64],
) -> Result<()> {
    for c in constraints {
        // Zero the row and column; set K[dof, dof] = 1.0
        k.zero_row_col(c.global_dof)?;
        // Set the RHS to the prescribed displacement
        f[c.global_dof] = c.prescribed_value;
    }
    Ok(())
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;

    fn tridiag_3() -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(3, 3);
        coo.add(0, 0,  2.0); coo.add(0, 1, -1.0);
        coo.add(1, 1,  2.0); coo.add(1, 2, -1.0);
        coo.add(2, 2,  2.0);
        coo.build_sym().unwrap()
    }

    // ---- SpConstraint construction ----

    #[test]
    fn global_dof_computed_correctly() {
        // node 2, local DOF 1, ndf = 3  →  global = 2*3 + 1 = 7
        let c = SpConstraint::new(NodeId(2), 1, 0.0, 3);
        assert_eq!(c.global_dof, 7);
    }

    #[test]
    fn zero_local_dof_zero_node() {
        let c = SpConstraint::new(NodeId(0), 0, 5.0, 2);
        assert_eq!(c.global_dof, 0);
        assert_eq!(c.prescribed_value, 5.0);
    }

    // ---- apply_dirichlet_bcs ----

    #[test]
    fn fix_middle_dof_zeros_row_and_col() {
        let mut k = tridiag_3();
        let mut f = vec![1.0_f64, 2.0, 3.0];

        let c = SpConstraint::new(NodeId(0), 1, 0.0, 1); // global_dof = 1
        apply_dirichlet_bcs(&[c], &mut k, &mut f).unwrap();

        // Diagonal must be 1
        assert_eq!(k.get(1, 1).unwrap(), 1.0);
        // Row 1: all off-diagonals zeroed
        assert_eq!(k.get(1, 0).unwrap(), 0.0);
        assert_eq!(k.get(1, 2).unwrap(), 0.0);
        // Column 1: all off-diagonals zeroed (symmetric)
        assert_eq!(k.get(0, 1).unwrap(), 0.0);
        // RHS at constrained DOF set to prescribed value
        assert_eq!(f[1], 0.0);
        // Other RHS entries unchanged
        assert_eq!(f[0], 1.0);
        assert_eq!(f[2], 3.0);
    }

    #[test]
    fn non_zero_settlement() {
        let mut k = tridiag_3();
        let mut f = vec![0.0_f64; 3];

        // Prescribe 0.005 m settlement at DOF 0
        let c = SpConstraint::new(NodeId(0), 0, 0.005, 1);
        apply_dirichlet_bcs(&[c], &mut k, &mut f).unwrap();

        assert_eq!(k.get(0, 0).unwrap(), 1.0);
        assert_eq!(k.get(0, 1).unwrap(), 0.0);
        assert_eq!(f[0], 0.005);
    }

    #[test]
    fn multiple_constraints_applied() {
        let mut k = tridiag_3();
        let mut f = vec![10.0_f64, 20.0, 30.0];

        let c0 = SpConstraint::new(NodeId(0), 0, 0.0, 1); // DOF 0
        let c2 = SpConstraint::new(NodeId(0), 2, 0.0, 1); // DOF 2
        apply_dirichlet_bcs(&[c0, c2], &mut k, &mut f).unwrap();

        assert_eq!(k.get(0, 0).unwrap(), 1.0);
        assert_eq!(k.get(2, 2).unwrap(), 1.0);
        assert_eq!(f[0], 0.0);
        assert_eq!(f[2], 0.0);
        // Middle DOF untouched
        assert_eq!(f[1], 20.0);
    }

    #[test]
    fn empty_constraints_no_op() {
        let mut k = tridiag_3();
        let mut f = vec![1.0_f64, 2.0, 3.0];
        let k_before = k.get(0, 1).unwrap();

        apply_dirichlet_bcs(&[], &mut k, &mut f).unwrap();

        // Nothing should change
        assert_eq!(k.get(0, 1).unwrap(), k_before);
        assert_eq!(f, vec![1.0, 2.0, 3.0]);
    }
}