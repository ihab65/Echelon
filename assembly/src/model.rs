//! The `Model` struct — the single owned container for a complete structural model.
//!
//! A `Model` holds nodes, elements, boundary conditions, and loads.
//! It is the Echelon equivalent of OpenSees's `Domain` class, but without
//! global mutable state: every `Model` is an independent owned value that
//! can be cloned and sent across threads for population-parallel analysis.
//!
//! ## Lifecycle
//!
//! ```text
//! Model::new(dim)
//!   │
//!   ├── add_node / add_element / add_constraint / add_load
//!   │         (build phase — topology fixed here)
//!   │
//!   ├── topology::build_pattern  →  SymCsrMatrix (once per topology)
//!   │
//!   ├── [analysis loop]
//!   │     builders::assemble_stiffness  →  K
//!   │     builders::assemble_load_vector →  F_ext
//!   │     builders::assemble_internal_force → F_int
//!   │     constraints::apply_dirichlet_bcs
//!   │     solver::solve → Δu
//!   │     model.u_global += Δu
//!   │     state::commit_state / revert_state
//!   │
//!   └── lock_loads  (between analysis phases, e.g. gravity → pushover)
//! ```
//!
//! ## No global state
//!
//! Multiple `Model` instances coexist simultaneously. Every function that
//! operates on a model takes `&Model` or `&mut Model` — there is no global
//! domain, no global tag registry, and no global counter anywhere in this crate.

use fem_core::{ModelDim, NodeId};
use elements::Assembleable;

use crate::constraints::SpConstraint;
use crate::error::{AssemblyError, Result};
use crate::loads::pattern::LoadPattern;

// -----------------------------------------------------------------
// Node
// -----------------------------------------------------------------

/// A geometric point in the mesh.
///
/// Nodes are numbered 0-based and contiguously. The `id` field mirrors the
/// node's position in `Model::nodes` — it is stored explicitly so that
/// elements and load patterns can hold a `NodeId` and the model can quickly
/// verify its existence.
///
/// DOF indices for this node in the global system are computed on demand
/// via `node.id.first_dof(model.dim.ndf())` — there is no stored global
/// DOF table because the layout is fully determined by the `NodeId` and `ndf`.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Index of this node in the mesh (0-based, equals its position in `Model::nodes`).
    pub id: NodeId,
    /// X coordinate (m).
    pub x: f64,
    /// Y coordinate (m).
    pub y: f64,
}

impl Node {
    /// Create a 2D node at `(x, y)`.
    pub fn new(id: NodeId, x: f64, y: f64) -> Self {
        Self { id, x, y }
    }
}

// -----------------------------------------------------------------
// Model
// -----------------------------------------------------------------

/// The complete structural model: nodes, elements, BCs, loads, and state.
///
/// # Example
///
/// ```rust,ignore
/// use assembly::model::{Model, Node};
/// use fem_core::{ModelDim, NodeId};
///
/// let mut model = Model::new(ModelDim::frame_2d());
/// model.add_node(Node::new(NodeId(0), 0.0, 0.0));
/// model.add_node(Node::new(NodeId(1), 3.0, 0.0));
/// // ... add elements, constraints, loads ...
/// ```
pub struct Model {
    // ---- Geometry / topology ----

    /// Model dimensionality and DOFs-per-node declaration.
    pub dim: ModelDim,

    /// Ordered list of mesh nodes. `nodes[k].id == NodeId(k)` is enforced
    /// by `add_node`.
    pub nodes: Vec<Node>,

    /// Heterogeneous element container. Each element implements `Assembleable`,
    /// which provides the DOF map and the stiffness/residual/adjoint hooks.
    ///
    /// Elements are stored as boxed trait objects so that a single model can
    /// contain trusses, beams, and future element types simultaneously.
    pub elements: Vec<Box<dyn Assembleable>>,

    /// Dirichlet boundary conditions. Populated by `add_constraint`.
    /// Applied to K and F by `constraints::apply_dirichlet_bcs`.
    pub constraints: Vec<SpConstraint>,

    // ---- Active state ----

    /// Load patterns for the current analysis phase.
    /// Consumed (and cleared) by `lock_loads` when transitioning phases.
    pub loads: Vec<Box<dyn LoadPattern>>,

    /// Global displacement vector — updated by the solver at each Newton
    /// iteration. Length = `n_dof()`. Initialised to zero by `build_state`.
    pub u_global: Vec<f64>,

    // ---- Baked state (load locking) ----

    /// The "frozen" gravity/pre-load vector, set by `lock_loads`.
    ///
    /// When present, `assemble_load_vector` seeds `f_ext` with these values
    /// before adding the active load patterns. This replicates OpenSees's
    /// `loadConst` / `setLoadConst` pattern without any global state.
    pub p_base: Option<Vec<f64>>,
}

impl Model {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Create an empty model with the given dimensionality.
    pub fn new(dim: ModelDim) -> Self {
        Self {
            dim,
            nodes:       Vec::new(),
            elements:    Vec::new(),
            constraints: Vec::new(),
            loads:       Vec::new(),
            u_global:    Vec::new(),
            p_base:      None,
        }
    }

    // -----------------------------------------------------------------
    // Build phase — add nodes / elements / constraints / loads
    // -----------------------------------------------------------------

    /// Register a node in the mesh.
    ///
    /// Nodes must be added in ascending `NodeId` order without gaps.
    /// The node's `id` must equal the current length of `self.nodes`.
    ///
    /// # Errors
    /// Returns [`AssemblyError::UnresolvedNode`] if the node's `id` does not
    /// match the expected next index (gap or duplicate detection).
    pub fn add_node(&mut self, node: Node) -> Result<()> {
        let expected = NodeId(self.nodes.len());
        if node.id != expected {
            return Err(AssemblyError::UnresolvedNode { node_id: node.id.0 });
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Add a structural element to the model.
    ///
    /// The element must have been constructed with `NodeId`s that are already
    /// present in `self.nodes`. Validation happens lazily at assembly time
    /// (via `DofMap::validate_against`) rather than here, to avoid the cost
    /// of checking every element at add-time in population runs.
    pub fn add_element(&mut self, element: Box<dyn Assembleable>) {
        self.elements.push(element);
    }

    /// Convenience wrapper: add an element from any concrete type that
    /// implements `Assembleable`.
    pub fn add_element_typed<E: Assembleable + 'static>(&mut self, element: E) {
        self.elements.push(Box::new(element));
    }

    /// Add a Dirichlet boundary condition.
    ///
    /// # Errors
    /// Returns [`AssemblyError::DofOverflow`] if `local_dof >= self.dim.ndf()`.
    pub fn add_constraint(&mut self, constraint: SpConstraint) -> Result<()> {
        let ndf = self.dim.ndf();
        if constraint.local_dof >= ndf {
            return Err(AssemblyError::DofOverflow {
                node_id:   constraint.node_id.0,
                local_dof: constraint.local_dof,
                ndf,
            });
        }
        self.constraints.push(constraint);
        Ok(())
    }

    /// Add a load pattern to the active load queue.
    pub fn add_load(&mut self, load: Box<dyn LoadPattern>) {
        self.loads.push(load);
    }

    /// Convenience wrapper: add a load from any concrete type.
    pub fn add_load_typed<L: LoadPattern + 'static>(&mut self, load: L) {
        self.loads.push(Box::new(load));
    }

    // -----------------------------------------------------------------
    // State initialisation
    // -----------------------------------------------------------------

    /// Allocate and zero `u_global` for the current node count.
    ///
    /// Must be called after the topology is fully built (all nodes added)
    /// and before the first analysis step. Calling this again resets all
    /// displacements to zero.
    pub fn build_state(&mut self) {
        self.u_global = vec![0.0_f64; self.n_dof()];
    }

    // -----------------------------------------------------------------
    // Load locking
    // -----------------------------------------------------------------

    /// Bake the current active loads at `pseudo_time` into `p_base` and
    /// clear the active load queue for the next analysis phase.
    ///
    /// This replicates OpenSees's `loadConst` pattern: after calling this,
    /// `assemble_load_vector` will always seed `f_ext` with `p_base`
    /// (the gravity / pre-load vector) before layering the new phase's loads.
    ///
    /// # Typical use
    /// ```rust,ignore
    /// // Phase 1: gravity analysis
    /// model.add_load_typed(gravity_load);
    /// // ... run linear static to convergence ...
    ///
    /// // Transition: lock gravity, set up pushover
    /// model.lock_loads(1.0);          // bakes gravity at pseudo_time=1.0
    /// model.add_load_typed(lateral);  // fresh lateral load for pushover
    /// // ... run pushover ...
    /// ```
    ///
    /// # Errors
    /// Returns [`AssemblyError::EmptyLoadLock`] if there are no active loads
    /// to bake, as this almost always indicates a user logic error.
    pub fn lock_loads(&mut self, pseudo_time: f64) -> Result<()> {
        if self.loads.is_empty() {
            return Err(AssemblyError::EmptyLoadLock);
        }

        let n = self.n_dof();
        let mut f = vec![0.0_f64; n];

        // Seed from existing p_base (supports chained lock calls)
        if let Some(ref base) = self.p_base {
            f.iter_mut().zip(base.iter()).for_each(|(fi, &bi)| *fi = bi);
        }

        // Apply all active patterns at the given pseudo_time
        for load in &self.loads {
            load.apply_to_global_vector(pseudo_time, self, &mut f);
        }

        self.p_base = Some(f);
        self.loads.clear();
        Ok(())
    }

    // -----------------------------------------------------------------
    // Sizing helpers
    // -----------------------------------------------------------------

    /// Total number of global DOFs: `n_nodes × ndf`.
    #[inline]
    pub fn n_dof(&self) -> usize {
        self.nodes.len() * self.dim.ndf()
    }

    /// Number of nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of elements.
    #[inline]
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// `true` if `node_id` refers to a valid node in this model.
    #[inline]
    pub fn has_node(&self, node_id: NodeId) -> bool {
        node_id.0 < self.nodes.len()
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::ModelDim;

    #[test]
    fn new_model_empty() {
        let m = Model::new(ModelDim::frame_2d());
        assert_eq!(m.n_nodes(), 0);
        assert_eq!(m.n_elements(), 0);
        assert_eq!(m.n_dof(), 0);
        assert!(m.p_base.is_none());
    }

    #[test]
    fn add_node_sequential_ok() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0)).unwrap();
        assert_eq!(m.n_nodes(), 2);
        assert_eq!(m.n_dof(), 6); // 2 nodes × 3 DOF
    }

    #[test]
    fn add_node_gap_errors() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        // Skipping NodeId(1) — should error
        let err = m.add_node(Node::new(NodeId(2), 6.0, 0.0)).unwrap_err();
        assert!(matches!(err, AssemblyError::UnresolvedNode { node_id: 2 }));
    }

    #[test]
    fn add_constraint_dof_overflow() {
        let mut m = Model::new(ModelDim::truss_2d()); // ndf = 2
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        let c = SpConstraint {
            node_id:          NodeId(0),
            local_dof:        2, // DOF 2 doesn't exist in a 2-DOF truss
            prescribed_value: 0.0,
            global_dof:       2,
        };
        let err = m.add_constraint(c).unwrap_err();
        assert!(matches!(
            err,
            AssemblyError::DofOverflow { local_dof: 2, ndf: 2, .. }
        ));
    }

    #[test]
    fn build_state_sizes_u_global() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0)).unwrap();
        m.build_state();
        assert_eq!(m.u_global.len(), 6);
        assert!(m.u_global.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lock_loads_empty_errors() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        m.build_state();
        let err = m.lock_loads(1.0).unwrap_err();
        assert!(matches!(err, AssemblyError::EmptyLoadLock));
    }

    #[test]
    fn has_node_correct() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        assert!(m.has_node(NodeId(0)));
        assert!(!m.has_node(NodeId(1)));
    }
}