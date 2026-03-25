//! Index newtypes — typed wrappers around `usize` that prevent accidental
//! confusion of structurally distinct index spaces.
//!
//! In FEM code it is trivially easy to write `K[node_id]` when you meant
//! `K[global_dof]`.  These four newtypes make the compiler catch that mistake
//! at zero runtime cost.
//!
//! # Layout guarantee
//!
//! All four types are `#[repr(transparent)]`, meaning they have the same
//! memory layout as a bare `usize`.  [`GlobalDof`] additionally exposes this
//! via [`DofMap::as_usize_slice`](crate::DofMap::as_usize_slice), which
//! reinterprets a `&[GlobalDof]` as `&[usize]` without copying — the safety
//! requirement is that `GlobalDof` is `repr(transparent)` over `usize`,
//! which is enforced by the layout tests below.

/// Index of a node in the mesh.
///
/// Nodes are geometric points; their positions are stored separately in the
/// coordinate arrays.  A `NodeId` is not a DOF index — a single node can
/// own 2 or 3 DOFs depending on the model dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeId(pub usize);

/// Index of an element in the mesh.
///
/// Elements connect nodes and define the stiffness contribution.
/// `ElemId` is distinct from `NodeId` even though both are `usize` wrappers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ElemId(pub usize);

/// Index into the global DOF vector — the row/column of the global stiffness
/// matrix `K` and the position in the global load vector `F`.
///
/// `GlobalDof` is the index you pass to `scatter_add`.  It is `repr(transparent)`
/// so that `&[GlobalDof]` can be safely reinterpreted as `&[usize]` for the
/// `scatter_add` hot path (see [`DofMap::as_usize_slice`](crate::DofMap::as_usize_slice)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct GlobalDof(pub usize);

/// Index of a DOF within a single element's local stiffness matrix.
///
/// A 2D beam element has 6 local DOFs (2 nodes × 3 DOF/node).
/// `LocalDof` is used when indexing into `ke` before it is assembled
/// into the global `K`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LocalDof(pub usize);

// -----------------------------------------------------------------
// Display
// -----------------------------------------------------------------

impl std::fmt::Display for NodeId   { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "N{}",  self.0) } }
impl std::fmt::Display for ElemId   { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "E{}",  self.0) } }
impl std::fmt::Display for GlobalDof{ fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "G{}",  self.0) } }
impl std::fmt::Display for LocalDof { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "L{}",  self.0) } }

// -----------------------------------------------------------------
// Arithmetic helpers
// -----------------------------------------------------------------

impl GlobalDof {
    /// Offset this DOF index by `n` — useful when computing the first DOF
    /// for node `i`: `base + i * ndf`.
    #[inline]
    pub fn offset(self, n: usize) -> Self {
        Self(self.0 + n)
    }
}

impl NodeId {
    /// Compute the first `GlobalDof` owned by this node given `ndf`
    /// DOFs per node.
    ///
    /// # Example
    /// ```
    /// use fem_core::{NodeId, GlobalDof};
    /// assert_eq!(NodeId(3).first_dof(3), GlobalDof(9));
    /// ```
    #[inline]
    pub fn first_dof(self, ndf: usize) -> GlobalDof {
        GlobalDof(self.0 * ndf)
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    // ---- layout guarantees ----

    #[test]
    fn global_dof_same_size_as_usize() {
        assert_eq!(mem::size_of::<GlobalDof>(), mem::size_of::<usize>());
    }

    #[test]
    fn global_dof_same_align_as_usize() {
        assert_eq!(mem::align_of::<GlobalDof>(), mem::align_of::<usize>());
    }

    #[test]
    fn all_newtypes_same_layout_as_usize() {
        assert_eq!(mem::size_of::<NodeId>(),    mem::size_of::<usize>());
        assert_eq!(mem::size_of::<ElemId>(),    mem::size_of::<usize>());
        assert_eq!(mem::size_of::<LocalDof>(),  mem::size_of::<usize>());
    }

    // ---- type incompatibility (compile-time, verified by existence) ----
    // These functions will not compile if the types are confused.

    fn _takes_node_id(_: NodeId)    {}
    fn _takes_elem_id(_: ElemId)    {}
    fn _takes_global(_: GlobalDof)  {}
    fn _takes_local(_: LocalDof)    {}

    #[test]
    fn display_formats() {
        assert_eq!(format!("{}", NodeId(5)),    "N5");
        assert_eq!(format!("{}", ElemId(2)),    "E2");
        assert_eq!(format!("{}", GlobalDof(7)), "G7");
        assert_eq!(format!("{}", LocalDof(1)),  "L1");
    }

    #[test]
    fn first_dof() {
        assert_eq!(NodeId(0).first_dof(3), GlobalDof(0));
        assert_eq!(NodeId(1).first_dof(3), GlobalDof(3));
        assert_eq!(NodeId(2).first_dof(3), GlobalDof(6));
        assert_eq!(NodeId(5).first_dof(2), GlobalDof(10));
    }

    #[test]
    fn global_dof_offset() {
        assert_eq!(GlobalDof(9).offset(0), GlobalDof(9));
        assert_eq!(GlobalDof(9).offset(1), GlobalDof(10));
        assert_eq!(GlobalDof(9).offset(2), GlobalDof(11));
    }

    #[test]
    fn ordering() {
        assert!(NodeId(0) < NodeId(1));
        assert!(GlobalDof(5) > GlobalDof(3));
    }

    #[test]
    fn hash_and_eq() {
        use std::collections::HashSet;
        let mut s: HashSet<GlobalDof> = HashSet::new();
        s.insert(GlobalDof(1));
        s.insert(GlobalDof(2));
        s.insert(GlobalDof(1)); // duplicate
        assert_eq!(s.len(), 2);
    }
}