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

use fem_core::{ElemId, ModelDim, NodeId};
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
    /// Z coordinate (m).
    pub z: f64,
}

impl Node {
    /// Create a node at `(x, y, z)`.
    pub fn new(id: NodeId, x: f64, y: f64, z: f64) -> Self {
        Self { id, x, y, z }
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

    /// Support reactions at constrained DOFs, populated by [`compute_reactions`].
    ///
    /// Empty until [`compute_reactions`] is called after a converged step.
    pub reactions: Vec<f64>,
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
            reactions:   Vec::new(),
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
    pub fn add_element(&mut self, element: Box<dyn Assembleable>) -> ElemId {
        let id = self.elements.len();
        self.elements.push(element);
        ElemId(id)
    }

    /// Convenience wrapper: add an element from any concrete type that
    /// implements `Assembleable`.
    pub fn add_element_typed<E: Assembleable + 'static>(&mut self, element: E) -> ElemId {
        let id = self.elements.len();
        self.elements.push(Box::new(element));
        ElemId(id)
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
    // Load baking — Phase 4
    // -----------------------------------------------------------------

    /// Bake a specific load pattern permanently into the pre-load vector.
    ///
    /// Evaluates `load` at `pseudo_time` and **accumulates** the result into
    /// `self.p_base`. Subsequent calls accumulate further — call
    /// [`clear_baked_loads`] first to start fresh.
    ///
    /// Unlike the removed `lock_loads`, this method does **not** touch
    /// `self.loads` (the active load queue). The caller decides what gets
    /// baked and what stays dynamic.
    ///
    /// # Typical use — seismic pre-load
    ///
    /// ```rust,ignore
    /// // Bake the expected seismic weight (1.0·DL + 0.25·LL) permanently.
    /// model.bake_load(&seismic_combo, 1.0);
    ///
    /// // Active loads for the pushover (untouched by bake_load).
    /// model.add_load_typed(lateral_load);
    /// ```
    pub fn bake_load(&mut self, load: &dyn LoadPattern, pseudo_time: f64) {
        if self.p_base.is_none() {
            self.p_base = Some(vec![0.0_f64; self.n_dof()]);
        }
        // borrow split: p_base is separate from the rest of self
        let n = self.n_dof();
        // We need to pass `self` to `apply_to_global_vector`, but we also
        // hold a mutable reference to `p_base`. Use a temporary scratch
        // buffer and accumulate, to avoid the split-borrow problem.
        let mut scratch = vec![0.0_f64; n];
        load.apply_to_global_vector(pseudo_time, self, &mut scratch);
        let p_base = self.p_base.as_mut().unwrap();
        for (b, s) in p_base.iter_mut().zip(scratch.iter()) {
            *b += s;
        }
    }

    /// Returns `true` if the model has a non-empty baked base load vector.
    #[inline]
    pub fn has_baked_loads(&self) -> bool {
        self.p_base.is_some()
    }

    /// Clear the baked base load vector (`p_base = None`).
    ///
    /// Essential for multi-stage analyses where the pre-load state needs
    /// to be reset between analysis phases.
    #[inline]
    pub fn clear_baked_loads(&mut self) {
        self.p_base = None;
    }

    /// Clear the active dynamic load queue without touching `p_base`.
    ///
    /// Equivalent to OpenSees's `loadConst` when called after
    /// [`bake_load`]: first bake the gravity state, then clear the
    /// active queue before adding the next analysis phase's loads.
    #[inline]
    pub fn clear_active_loads(&mut self) {
        self.loads.clear();
    }

    // -----------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------

    /// Print a structured diagnostic report to stdout.
    ///
    /// Checks for common modelling errors and reports the active load tree.
    /// Call this before the first `driver.analyze()` to verify the model.
    pub fn diagnose(&self) {
        println!("╔═══════════════════════════════════════╗");
        println!("║       Echelon Model Diagnostics       ║");
        println!("╚═══════════════════════════════════════╝");
        println!("  Nodes:    {}", self.n_nodes());
        println!("  Elements: {}", self.n_elements());
        println!("  DOFs:     {}", self.n_dof());
        println!("  Constraints: {}", self.constraints.len());

        // Baked load status
        if self.has_baked_loads() {
            println!("    Baked base loads (p_base) are ACTIVE.");
            println!("    These will be present in all subsequent analyses.");
            println!("    Call model.clear_baked_loads() to reset if needed.");
        } else {
            println!("  ✓ No baked base loads.");
        }

        // Constraint check: warn if no constraints at all (likely mechanism)
        if self.constraints.is_empty() && !self.elements.is_empty() {
            println!("    WARNING: no Dirichlet constraints — the stiffness");
            println!("    matrix will be singular (rigid-body mechanism).");
        }

        // Active load tree
        println!();
        self.print_load_tree();
    }

    /// Print the active load patterns as a hierarchical tree.
    pub fn print_load_tree(&self) {
        println!("=== Active Load Tree ===");
        if self.loads.is_empty() {
            println!("  (no active loads)");
            return;
        }
        let n = self.loads.len();
        for (i, load) in self.loads.iter().enumerate() {
            print!("{}", load.format_tree("  ", i == n - 1));
        }
    }

    /// Compute support reactions at all constrained DOFs.
    ///
    /// After a converged analysis step, the reaction at a constrained DOF is
    /// the internal resisting force that the structure exerts on the support.
    /// It equals the sum of all element internal-force contributions at that
    /// DOF, evaluated at the current `u_global`.
    ///
    /// The result is stored in `self.reactions` and also returned as a slice
    /// reference for immediate use.
    ///
    /// # Calculation
    ///
    /// `R = F_int(u) |_{constrained DOFs}`
    ///
    /// where `F_int` is the global internal force vector assembled from all
    /// elements at the current displacement state.
    ///
    /// # Panics
    /// Panics if `build_state` has not been called (i.e. `u_global` is empty).
    pub fn compute_reactions(&mut self) -> &[f64] {
        let n = self.n_dof();
        self.reactions.resize(n, 0.0);
        self.reactions.fill(0.0);

        // Assemble F_int from all elements
        for element in &self.elements {
            let dof_map = element.dof_map();
            let u_local: Vec<f64> = dof_map
                .as_usize_slice()
                .iter()
                .map(|&g| self.u_global[g])
                .collect();
            let f_int = element.f_int(&u_local);
            for (local_i, &global_dof) in dof_map.as_usize_slice().iter().enumerate() {
                self.reactions[global_dof] += f_int[local_i];
            }
        }

        // Zero out unconstrained DOFs — reactions only exist at supports
        // Build a mask of constrained DOFs
        let mut is_constrained = vec![false; n];
        for c in &self.constraints {
            if c.global_dof < n {
                is_constrained[c.global_dof] = true;
            }
        }
        for (r, &constrained) in self.reactions.iter_mut().zip(is_constrained.iter()) {
            if !constrained {
                *r = 0.0;
            }
        }

        &self.reactions
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
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        assert_eq!(m.n_nodes(), 2);
        assert_eq!(m.n_dof(), 6); // 2 nodes × 3 DOF
    }

    #[test]
    fn add_node_gap_errors() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        // Skipping NodeId(1) — should error
        let err = m.add_node(Node::new(NodeId(2), 6.0, 0.0, 0.0)).unwrap_err();
        assert!(matches!(err, AssemblyError::UnresolvedNode { node_id: 2 }));
    }

    #[test]
    fn add_constraint_dof_overflow() {
        let mut m = Model::new(ModelDim::truss_2d()); // ndf = 2
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
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
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        m.build_state();
        assert_eq!(m.u_global.len(), 6);
        assert!(m.u_global.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn has_node_correct() {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        assert!(m.has_node(NodeId(0)));
        assert!(!m.has_node(NodeId(1)));
    }
}