//! DOF map — the bridge between local element DOF indices and the global
//! stiffness matrix.
//!
//! Every element owns a [`DofMap`] that records which global DOF corresponds
//! to each of its local DOFs.  At assembly time, `scatter_add` is called with
//! `dof_map.as_usize_slice()` as the index array.
//!
//! ## Zero-cost `as_usize_slice`
//!
//! [`GlobalDof`] is `#[repr(transparent)]` over `usize`, so a `&[GlobalDof]`
//! has exactly the same memory layout as a `&[usize]`.  We exploit this in
//! [`DofMap::as_usize_slice`] via a pointer cast — no copying, no allocation,
//! no runtime overhead in the assembly hot path.
//!
//! This is the same technique used by `std` for `Path` / `OsStr` / `str`.
//! It is sound because:
//! 1. `GlobalDof` is `repr(transparent)` (enforced by the layout tests in
//!    [`crate::ids`]).
//! 2. We only produce a shared reference, not a mutable one.
//! 3. The lifetime of the output is tied to `&self`.

use crate::ids::{GlobalDof, LocalDof, NodeId};

/// Maps local element DOF indices `[0..n_local)` to global DOF indices.
///
/// Built during the DOF-numbering phase (once per model topology) and stored
/// per element.  Immutable after construction.
///
/// # Example
///
/// ```
/// use fem_core::{DofMap, GlobalDof, LocalDof, NodeId};
///
/// // A 2D truss element: 2 nodes, 2 DOFs per node → 4 local DOFs
/// // Node 0 owns global DOFs 0,1; Node 3 owns global DOFs 6,7
/// let dof_map = DofMap::from_global_dofs(vec![
///     GlobalDof(0), GlobalDof(1),   // node 0: ux, uy
///     GlobalDof(6), GlobalDof(7),   // node 3: ux, uy
/// ]);
///
/// assert_eq!(dof_map.n_local(), 4);
/// assert_eq!(dof_map[LocalDof(2)], GlobalDof(6));
/// assert_eq!(dof_map.as_usize_slice(), &[0, 1, 6, 7]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DofMap {
    /// Internal storage: `indices[local_dof] = global_dof`.
    indices: Vec<GlobalDof>,
}

impl DofMap {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build from an explicit ordered list of global DOF indices.
    ///
    /// The order must match the element's local DOF numbering convention.
    /// For a 2D beam element this is typically:
    /// `[u1_x, u1_y, θ1, u2_x, u2_y, θ2]` (node 1 first, then node 2).
    pub fn from_global_dofs(indices: Vec<GlobalDof>) -> Self {
        Self { indices }
    }

    /// Build from a list of `(node, ndf)` pairs: the DOFs owned by each node
    /// in connectivity order.
    ///
    /// This is the most common construction path:
    /// ```
    /// use fem_core::{DofMap, NodeId};
    ///
    /// // 2D truss: node 0 and node 3, 2 DOFs each
    /// let dof_map = DofMap::from_nodes(&[NodeId(0), NodeId(3)], 2);
    /// // → GlobalDofs: [0, 1, 6, 7]
    /// assert_eq!(dof_map.as_usize_slice(), &[0, 1, 6, 7]);
    /// ```
    pub fn from_nodes(nodes: &[NodeId], ndf: usize) -> Self {
        let mut indices = Vec::with_capacity(nodes.len() * ndf);
        for &node in nodes {
            let base = node.first_dof(ndf);
            for k in 0..ndf {
                indices.push(base.offset(k));
            }
        }
        Self { indices }
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Number of local DOFs (length of the map).
    #[inline]
    pub fn n_local(&self) -> usize {
        self.indices.len()
    }

    /// The global DOF for local DOF `i` (zero-indexed).
    ///
    /// # Panics
    /// Panics if `i >= self.n_local()`.
    #[inline]
    pub fn get(&self, i: LocalDof) -> GlobalDof {
        self.indices[i.0]
    }

    /// Slice of global DOF indices as raw `usize` — zero-cost, zero-copy.
    ///
    /// Pass this directly to `CsrMatrix::scatter_add` or
    /// `SymCsrMatrix::scatter_add`.
    ///
    /// # Safety rationale
    ///
    /// `GlobalDof` is `repr(transparent)` over `usize` (verified by tests in
    /// [`crate::ids`]).  Therefore `&[GlobalDof]` and `&[usize]` have identical
    /// memory layout, alignment, and length.  The cast via `std::slice::from_raw_parts`
    /// is sound because:
    /// - The pointer comes from a valid, live `Vec<GlobalDof>`.
    /// - The length is unchanged.
    /// - We produce a `&[usize]` tied to `'_` (the lifetime of `self`).
    /// - We never produce a `&mut [usize]`.
    #[inline]
    pub fn as_usize_slice(&self) -> &[usize] {
        // SAFETY: see doc comment above.
        unsafe {
            std::slice::from_raw_parts(
                self.indices.as_ptr() as *const usize,
                self.indices.len(),
            )
        }
    }

    /// Iterator over `(LocalDof, GlobalDof)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (LocalDof, GlobalDof)> + '_ {
        self.indices
            .iter()
            .enumerate()
            .map(|(i, &g)| (LocalDof(i), g))
    }
}

// -----------------------------------------------------------------
// Index operator
// -----------------------------------------------------------------

impl std::ops::Index<LocalDof> for DofMap {
    type Output = GlobalDof;

    #[inline]
    fn index(&self, local: LocalDof) -> &GlobalDof {
        &self.indices[local.0]
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- from_global_dofs ----

    #[test]
    fn from_global_dofs_roundtrip() {
        let dofs = vec![GlobalDof(0), GlobalDof(1), GlobalDof(6), GlobalDof(7)];
        let m = DofMap::from_global_dofs(dofs.clone());
        assert_eq!(m.n_local(), 4);
        for (i, &expected) in dofs.iter().enumerate() {
            assert_eq!(m.get(LocalDof(i)), expected);
        }
    }

    // ---- from_nodes ----

    #[test]
    fn from_nodes_2d_truss() {
        // node 0 → DOFs 0,1;  node 3 → DOFs 6,7
        let m = DofMap::from_nodes(&[NodeId(0), NodeId(3)], 2);
        assert_eq!(m.n_local(), 4);
        assert_eq!(m[LocalDof(0)], GlobalDof(0));
        assert_eq!(m[LocalDof(1)], GlobalDof(1));
        assert_eq!(m[LocalDof(2)], GlobalDof(6));
        assert_eq!(m[LocalDof(3)], GlobalDof(7));
    }

    #[test]
    fn from_nodes_2d_beam() {
        // node 1 → DOFs 3,4,5;  node 2 → DOFs 6,7,8
        let m = DofMap::from_nodes(&[NodeId(1), NodeId(2)], 3);
        assert_eq!(m.n_local(), 6);
        assert_eq!(m[LocalDof(0)], GlobalDof(3));
        assert_eq!(m[LocalDof(1)], GlobalDof(4));
        assert_eq!(m[LocalDof(2)], GlobalDof(5));
        assert_eq!(m[LocalDof(3)], GlobalDof(6));
        assert_eq!(m[LocalDof(4)], GlobalDof(7));
        assert_eq!(m[LocalDof(5)], GlobalDof(8));
    }

    #[test]
    fn from_nodes_node0_is_always_dof0() {
        let m = DofMap::from_nodes(&[NodeId(0), NodeId(1)], 3);
        assert_eq!(m[LocalDof(0)], GlobalDof(0));
        assert_eq!(m[LocalDof(3)], GlobalDof(3));
    }

    // ---- as_usize_slice ----

    #[test]
    fn as_usize_slice_correct_values() {
        let m = DofMap::from_nodes(&[NodeId(0), NodeId(3)], 2);
        assert_eq!(m.as_usize_slice(), &[0usize, 1, 6, 7]);
    }

    #[test]
    fn as_usize_slice_no_copy_same_address() {
        // Verify the pointer is the same as the underlying data —
        // i.e. it is a zero-copy view.
        let m = DofMap::from_nodes(&[NodeId(0), NodeId(1)], 2);
        let slice = m.as_usize_slice();
        let internal_ptr = m.indices.as_ptr() as *const usize;
        assert_eq!(slice.as_ptr(), internal_ptr);
    }

    #[test]
    fn as_usize_slice_empty() {
        let m = DofMap::from_global_dofs(vec![]);
        assert_eq!(m.as_usize_slice(), &[] as &[usize]);
    }

    // ---- iter ----

    #[test]
    fn iter_yields_correct_pairs() {
        let m = DofMap::from_nodes(&[NodeId(2)], 2);
        let pairs: Vec<_> = m.iter().collect();
        assert_eq!(pairs, vec![
            (LocalDof(0), GlobalDof(4)),
            (LocalDof(1), GlobalDof(5)),
        ]);
    }

    // ---- index operator ----

    #[test]
    fn index_operator() {
        let m = DofMap::from_global_dofs(vec![GlobalDof(10), GlobalDof(20)]);
        assert_eq!(m[LocalDof(0)], GlobalDof(10));
        assert_eq!(m[LocalDof(1)], GlobalDof(20));
    }
}