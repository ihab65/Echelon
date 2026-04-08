//! Composite load pattern — scales and sums a collection of child patterns.
//!
//! [`LoadCombo`] implements the Composite design pattern over [`LoadPattern`].
//! Because it is itself a `LoadPattern`, combos nest arbitrarily:
//! a ULS combination can contain dead-load and live-load sub-combos, each
//! of which can contain nodal, element, and gravity loads.
//!
//! ## Example
//!
//! ```rust,ignore
//! use assembly::loads::combo::LoadCombo;
//!
//! let mut dead = LoadCombo::new();
//! dead.add(gravity_load));
//!
//! let mut live = LoadCombo::new();
//! live.add(floor_load));
//!
//! // ULS: 1.35·DL + 1.5·LL
//! let mut uls = LoadCombo::new();
//! uls.add(1.35, dead);
//! uls.add(1.50, live);
//!
//! model.add_load_typed(uls);
//! ```

use crate::loads::pattern::LoadPattern;
use crate::model::Model;

/// A composite load that applies a collection of child patterns scaled by a
/// common factor.
///
/// The contribution to `f_ext` is:
/// `Σ child_i.apply(...) * scale_factor`
pub struct LoadCombo {
    /// Child load patterns (may themselves be [`LoadCombo`]s).
    pub children: Vec<(f64, Box<dyn LoadPattern>)>,
}

impl LoadCombo {
    /// Create an empty combo with the given scale factor.
    pub fn new() -> Self {
        Self { children: Vec::new() }
    }

    /// Add a child load pattern to this combination.
    pub fn add<T: LoadPattern + 'static>(&mut self, factor: f64, load: T) {
        self.children.push((factor, Box::new(load)));
    }

    /// Number of direct children.
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Returns `true` if this combo has no children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}

impl LoadPattern for LoadCombo {
    fn apply_to_global_vector(
        &self,
        pseudo_time: f64,
        model:       &Model,
        f_ext:       &mut [f64],
        alpha:       f64
    ) {
        if self.children.is_empty() {
            return;
        }
        // This avoids aliasing: we never write to f_ext while children read it.
        for (factor, child) in &self.children {
            child.apply_to_global_vector(pseudo_time, model, f_ext, alpha * factor);
        }
    }

    fn clone_box(&self) -> Box<dyn LoadPattern> {
        let mut copy = LoadCombo::new();
        // Unpack the tuple here
        for (factor, child) in &self.children {
            // We push directly to the vector rather than using copy.add() 
            // because .add() expects a generic T, not an already-boxed trait object.
            copy.children.push((*factor, child.clone_box()));
        }
        Box::new(copy)
    }

    fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let branch = if is_last { "└── " } else { "├── " };
        
        // LoadCombo no longer has a global scale, just print its name
        let mut out = format!("{prefix}{branch}LoadCombo\n");
        
        let child_prefix = format!(
            "{prefix}{}",
            if is_last { "    " } else { "│   " }
        );
        
        let n = self.children.len();
        for (i, (factor, child)) in self.children.iter().enumerate() {
            let is_last_child = i == n - 1;
            let child_str = child.format_tree(&child_prefix, is_last_child);
            
            // Ninja move: Inject the load factor directly into the child's branch output
            // so the diagnostic tree still looks beautiful.
            let child_branch = if is_last_child { "└── " } else { "├── " };
            let injected_str = child_str.replacen(
                child_branch, 
                &format!("{}[×{:.2}] ", child_branch, factor), 
                1
            );
            
            out.push_str(&injected_str);
        }
        out
    }
}

// LoadCombo contains Box<dyn LoadPattern> which is Send + Sync.
unsafe impl Send for LoadCombo {}
unsafe impl Sync for LoadCombo {}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{ModelDim, NodeId};
    use crate::loads::pattern::NodalLoad;
    use crate::loads::series::ConstantSeries;
    use crate::model::{Model, Node};

    fn two_node_model() -> Model {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        m.build_state();
        m
    }

    #[test]
    fn empty_combo_contributes_zero() {
        let m = two_node_model();
        let c = LoadCombo::new();
        let mut f = vec![1.0_f64; 6];
        c.apply_to_global_vector(1.0, &m, &mut f, 1.0);
        assert!(f.iter().all(|&v| v == 1.0), "empty combo must not modify f_ext");
    }

    #[test]
    fn scale_factor_applied_correctly() {
        let m = two_node_model();
        let mut c = LoadCombo::new(); // double everything
        c.add(
            2.0,
            NodalLoad {
                node_id:         NodeId(1),
                reference_loads: vec![10.0, 0.0, 0.0],
                series:          Box::new(ConstantSeries),
            }
        );
        let mut f = vec![0.0_f64; 6];
        c.apply_to_global_vector(1.0, &m, &mut f, 1.0);
        // Node 1 UX = DOF 3; scaled by 2 → 20
        assert!((f[3] - 20.0).abs() < 1e-12, "f[3]={}", f[3]);
    }

    #[test]
    fn nested_combo_sums_correctly() {
        let m = two_node_model();
        // Inner combo: scale=1.35, load=10 → contributes 13.5
        let mut inner = LoadCombo::new();
        inner.add(
            1.35,
            NodalLoad {
                node_id: NodeId(1),
                reference_loads: vec![10.0, 0.0, 0.0],
                series: Box::new(ConstantSeries),
            }
        );
        // Outer combo: scale=1.0, wraps inner
        let mut outer = LoadCombo::new();
        outer.add(1.0, inner);
        let mut f = vec![0.0_f64; 6];
        outer.apply_to_global_vector(1.0, &m, &mut f, 1.0);
        assert!((f[3] - 13.5).abs() < 1e-10, "f[3]={}", f[3]);
    }

    #[test]
    fn clone_box_produces_independent_copy() {
        let m = two_node_model();
        let mut c = LoadCombo::new();
        c.add(
            3.0,
            NodalLoad {
                node_id: NodeId(0),
                reference_loads: vec![5.0, 0.0, 0.0],
                series: Box::new(ConstantSeries),
            }
        );
        let cloned = c.clone_box();

        let mut f1 = vec![0.0_f64; 6];
        let mut f2 = vec![0.0_f64; 6];
        c.apply_to_global_vector(1.0, &m, &mut f1, 1.0);
        cloned.apply_to_global_vector(1.0, &m, &mut f2, 1.0);
        assert_eq!(f1, f2, "clone must produce identical result");
    }

    #[test]
    fn format_tree_contains_scale() {
        let mut c = LoadCombo::new();
        c.add(
            1.5, 
            NodalLoad {
                node_id: NodeId(0),
                reference_loads: vec![0.0, -50e3, 0.0],
                series: Box::new(ConstantSeries),
            }
        );
        let tree = c.format_tree("", true);
        assert!(tree.contains("1.50"), "scale missing from tree: {tree}");
        assert!(tree.contains("NodalLoad"), "child missing from tree: {tree}");
    }
}