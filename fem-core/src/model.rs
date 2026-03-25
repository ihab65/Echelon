//! Model dimensionality — the single top-level declaration that determines
//! how many spatial dimensions and DOFs per node the analysis uses.
//!
//! This mirrors OpenSeesPy's `model('basic', '-ndm', 2, '-ndf', 3)` but as
//! a typed Rust value rather than a string-keyed function call.
//!
//! `ModelDim` is not just metadata — it governs:
//! - How many DOFs are assigned per node (used by [`crate::DofMap`]).
//! - Which coordinate transforms are valid.
//! - Which element formulations can be instantiated.
//!
//! The `elements` crate will validate at construction time that elements are
//! compatible with the declared `ModelDim`.

/// Spatial dimensionality and DOFs-per-node for a structural model.
///
/// # Variants
///
/// | Variant | `ndm` | Typical `ndf` | Use case |
/// |---------|-------|---------------|----------|
/// | `Dim2 { ndf: 2 }` | 2D | 2 | Truss (ux, uy per node) |
/// | `Dim2 { ndf: 3 }` | 2D | 3 | Frame (ux, uy, θz per node) |
/// | `Dim3 { ndf: 3 }` | 3D | 3 | 3D truss (ux, uy, uz) |
/// | `Dim3 { ndf: 6 }` | 3D | 6 | 3D frame (ux,uy,uz,θx,θy,θz) |
///
/// # Example
/// ```
/// use fem_core::ModelDim;
///
/// let model = ModelDim::frame_2d();
/// assert_eq!(model.ndm(), 2);
/// assert_eq!(model.ndf(), 3);
/// assert_eq!(model.dofs_per_element(2), 6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelDim {
    /// Two-dimensional model.
    Dim2 {
        /// Degrees of freedom per node.
        /// 2 for 2D truss (ux, uy); 3 for 2D frame (ux, uy, θz).
        ndf: usize,
    },
    /// Three-dimensional model.
    Dim3 {
        /// Degrees of freedom per node.
        /// 3 for 3D truss; 6 for 3D frame.
        ndf: usize,
    },
}

impl ModelDim {
    // -----------------------------------------------------------------
    // Common configurations
    // -----------------------------------------------------------------

    /// 2D truss model: 2 DOFs per node (ux, uy).
    pub const fn truss_2d() -> Self {
        Self::Dim2 { ndf: 2 }
    }

    /// 2D frame model: 3 DOFs per node (ux, uy, θz).
    /// This is the most common 2D structural analysis configuration.
    pub const fn frame_2d() -> Self {
        Self::Dim2 { ndf: 3 }
    }

    /// 3D truss model: 3 DOFs per node (ux, uy, uz).
    pub const fn truss_3d() -> Self {
        Self::Dim3 { ndf: 3 }
    }

    /// 3D frame model: 6 DOFs per node (ux, uy, uz, θx, θy, θz).
    pub const fn frame_3d() -> Self {
        Self::Dim3 { ndf: 6 }
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Number of spatial dimensions (2 or 3).
    #[inline]
    pub const fn ndm(&self) -> usize {
        match self {
            Self::Dim2 { .. } => 2,
            Self::Dim3 { .. } => 3,
        }
    }

    /// Degrees of freedom per node.
    #[inline]
    pub const fn ndf(&self) -> usize {
        match self {
            Self::Dim2 { ndf } | Self::Dim3 { ndf } => *ndf,
        }
    }

    /// Returns `true` if this is a 2D model.
    #[inline]
    pub const fn is_2d(&self) -> bool {
        matches!(self, Self::Dim2 { .. })
    }

    /// Returns `true` if this is a 3D model.
    #[inline]
    pub const fn is_3d(&self) -> bool {
        matches!(self, Self::Dim3 { .. })
    }

    /// Total local DOFs for an element connecting `n_nodes` nodes.
    ///
    /// For a 2D beam element (2 nodes, 3 DOF/node): `dofs_per_element(2) == 6`.
    #[inline]
    pub const fn dofs_per_element(&self, n_nodes: usize) -> usize {
        self.ndf() * n_nodes
    }

    /// Total number of global DOFs for a mesh with `n_nodes` nodes.
    #[inline]
    pub const fn total_dofs(&self, n_nodes: usize) -> usize {
        self.ndf() * n_nodes
    }
}

impl std::fmt::Display for ModelDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}D/{}-DOF", self.ndm(), self.ndf())
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_constructors() {
        let td2 = ModelDim::truss_2d();
        assert_eq!(td2.ndm(), 2);
        assert_eq!(td2.ndf(), 2);

        let fr2 = ModelDim::frame_2d();
        assert_eq!(fr2.ndm(), 2);
        assert_eq!(fr2.ndf(), 3);

        let td3 = ModelDim::truss_3d();
        assert_eq!(td3.ndm(), 3);
        assert_eq!(td3.ndf(), 3);

        let fr3 = ModelDim::frame_3d();
        assert_eq!(fr3.ndm(), 3);
        assert_eq!(fr3.ndf(), 6);
    }

    #[test]
    fn is_2d_is_3d() {
        assert!( ModelDim::frame_2d().is_2d());
        assert!(!ModelDim::frame_2d().is_3d());
        assert!(!ModelDim::frame_3d().is_2d());
        assert!( ModelDim::frame_3d().is_3d());
    }

    #[test]
    fn dofs_per_element() {
        // 2D beam: 2 nodes × 3 DOF/node = 6
        assert_eq!(ModelDim::frame_2d().dofs_per_element(2), 6);
        // 2D truss: 2 nodes × 2 DOF/node = 4
        assert_eq!(ModelDim::truss_2d().dofs_per_element(2), 4);
        // 3D frame: 2 nodes × 6 DOF/node = 12
        assert_eq!(ModelDim::frame_3d().dofs_per_element(2), 12);
    }

    #[test]
    fn total_dofs() {
        assert_eq!(ModelDim::frame_2d().total_dofs(5), 15);
        assert_eq!(ModelDim::truss_2d().total_dofs(4), 8);
    }

    #[test]
    fn custom_ndf() {
        let m = ModelDim::Dim2 { ndf: 6 };
        assert_eq!(m.ndf(), 6);
        assert_eq!(m.ndm(), 2);
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", ModelDim::frame_2d()), "2D/3-DOF");
        assert_eq!(format!("{}", ModelDim::frame_3d()), "3D/6-DOF");
    }

    #[test]
    fn copy_clone_eq() {
        let a = ModelDim::frame_2d();
        let b = a;    // Copy
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    #[test]
    fn const_constructors_are_const() {
        // Verify these can be used in const context
        const _M: ModelDim = ModelDim::frame_2d();
        const _N: usize = ModelDim::frame_2d().ndf();
        assert_eq!(_N, 3);
    }
}