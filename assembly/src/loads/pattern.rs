//! Load patterns — spatial distribution of forces on the structural model.
//!
//! A [`LoadPattern`] encodes *where* loads are applied and *what* their
//! reference magnitudes are. A companion [`TimeSeries`] encodes *how* those
//! loads scale with the current `pseudo_time`.
//!
//! ## Heterogeneous load list
//!
//! `Model::loads` holds a `Vec<Box<dyn LoadPattern>>`, so a single model can
//! simultaneously carry:
//! - Nodal point loads (gravity, applied forces)
//! - (Future) element distributed loads (wind pressure, snow)
//! - (Future) ground motion acceleration patterns
//!
//! ## Scatter convention
//!
//! `apply_to_global_vector` writes *into* a pre-allocated `f_ext` buffer
//! using `+=`. The buffer is pre-seeded by `assemble_load_vector` (from
//! `p_base` if present), so patterns accumulate additively — consistent with
//! superposition.
//!
//! Fixed DOFs receive whatever value is scattered here; `apply_dirichlet_bcs`
//! will overwrite those entries with the prescribed displacement immediately
//! after, so over-writing is harmless.

use fem_core::{ElemId, NodeId};
use elements::traits::ElementLoadParams;

use crate::loads::series::TimeSeries;
use crate::model::Model;

// -----------------------------------------------------------------
// LoadPattern trait
// -----------------------------------------------------------------

/// Spatial distribution of a load, with a time series for scaling.
///
/// Implementors must be `Send + Sync` so that models can be cloned and
/// evaluated concurrently across population-parallel analyses.
pub trait LoadPattern: Send + Sync {
    /// Evaluate the load at `pseudo_time` and scatter it into `f_ext`.
    ///
    /// # Arguments
    /// * `pseudo_time` — current load parameter or simulation time
    /// * `model`       — read-only access to the model (for `dim.ndf()`, etc.)
    /// * `f_ext`       — mutable global external force vector (length = `model.n_dof()`)
    fn apply_to_global_vector(
        &self,
        pseudo_time: f64,
        model:       &Model,
        f_ext:       &mut [f64],
        alpha:       f64
    );

        /// Clone this load pattern into a heap-allocated trait object.
    ///
    /// Required by [`LoadCombo`] and the `load_combo!` macro so that load
    /// cases can be composed into combinations without consuming ownership.
    fn clone_box(&self) -> Box<dyn LoadPattern>;

    /// Format this pattern as a tree node for diagnostic output.
    ///
    /// `prefix` is the indentation string for this level.
    /// `is_last` controls whether `└──` or `├──` is drawn.
    ///
    /// The default implementation produces a generic one-line entry.
    fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let branch = if is_last { "└── " } else { "├── " };
        format!("{prefix}{branch}LoadPattern\n")
    }
}

// -----------------------------------------------------------------
// NodalLoad
// -----------------------------------------------------------------

/// A concentrated force (and/or moment) applied at a single node.
///
/// The `reference_loads` vector holds the force components in global
/// coordinates, ordered by local DOF: `[Fx, Fy]` for a 2D truss node,
/// `[Fx, Fy, Mz]` for a 2D frame node. Its length must equal `model.dim.ndf()`.
///
/// # Example: vertical load P at node 2 of a 2D frame
///
/// ```rust,ignore
/// use assembly::loads::pattern::NodalLoad;
/// use assembly::loads::series::ConstantSeries;
/// use fem_core::NodeId;
///
/// let load = NodalLoad {
///     node_id:         NodeId(2),
///     reference_loads: vec![0.0, -50e3, 0.0],  // Fy = -50 kN
///     series:          Box::new(ConstantSeries),
/// };
/// model.add_load_typed(load);
/// ```
pub struct NodalLoad {
    /// The node at which the load is applied.
    pub node_id: NodeId,

    /// Reference force components in global coordinates, one per DOF.
    /// Length must match `model.dim.ndf()` at apply time.
    pub reference_loads: Vec<f64>,

    /// Temporal scaling rule.
    pub series: Box<dyn TimeSeries>,
}

impl NodalLoad {
    pub fn new<S: TimeSeries + 'static>(
        node_id: NodeId,
        reference_loads: Vec<f64>,
        series: S,
    ) -> Self 
    {
        Self { 
            node_id, 
            reference_loads, 
            series: Box::new(series) 
        }
    }
}

impl LoadPattern for NodalLoad {
    fn apply_to_global_vector(
        &self,
        pseudo_time: f64,
        model:       &Model,
        f_ext:       &mut [f64],
        alpha:       f64
    ) {
        let scale   = self.series.factor_at(pseudo_time) * alpha;
        let ndf     = model.dim.ndf();
        let base    = self.node_id.first_dof(ndf);

        // Scatter each load component into its global DOF
        let n = self.reference_loads.len().min(ndf);
        for i in 0..n {
            let global_dof = base.offset(i).0;
            if global_dof < f_ext.len() {
                f_ext[global_dof] += self.reference_loads[i] * scale;
            }
        }
    }

    fn clone_box(&self) -> Box<dyn LoadPattern> {
        // We need NodalLoad's fields to be Clone. TimeSeries is a trait object;
        // we delegate to a new impl on it.
        Box::new(NodalLoad {
            node_id:         self.node_id,
            reference_loads: self.reference_loads.clone(),
            series:          self.series.clone_box(),
        })
    }

    fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let branch = if is_last { "└── " } else { "├── " };
        let fx = self.reference_loads.first().copied().unwrap_or(0.0);
        let fy = self.reference_loads.get(1).copied().unwrap_or(0.0);
        format!(
            "{prefix}{branch}NodalLoad (node={}, Fx={fx:.3e}, Fy={fy:.3e})\n",
            self.node_id.0
        )
    }

}

// Make NodalLoad Send + Sync — both NodeId and Vec<f64> are already Send + Sync.
// The Box<dyn TimeSeries> bound enforces the same on the series.
unsafe impl Send for NodalLoad {}
unsafe impl Sync for NodalLoad {}

// -----------------------------------------------------------------
// ElementLoad
// -----------------------------------------------------------------

/// A distributed or point load applied along the span of a specific element.
///
/// The element is identified by the index returned from
/// [`Model::add_element`]. The load is converted to equivalent global
/// nodal forces via [`Element::equivalent_nodal_forces`] and scattered into
/// `f_ext` at the element's global DOFs.
///
/// # Example — uniform gravity load on beam element 2
///
/// ```rust,ignore
/// use assembly::loads::pattern::ElementLoad;
/// use assembly::loads::series::ConstantSeries;
/// use elements::ElementLoadParams;
///
/// model.add_load_typed(ElementLoad {
///     elem_id: 2,
///     params:  ElementLoadParams::Uniform { wx: 0.0, wy: -20e3 }, // 20 kN/m downward
///     series:  Box::new(ConstantSeries),
/// });
/// ```
pub struct ElementLoad {
    /// Index of the target element (returned by [`Model::add_element`]).
    pub elem_id: ElemId,
    /// Load type and magnitude.
    pub params: ElementLoadParams,
    /// Temporal scaling rule.
    pub series: Box<dyn TimeSeries>,
}

impl ElementLoad {
    pub fn new<S: TimeSeries + 'static>(
        elem_id: ElemId,
        params: ElementLoadParams,
        series: S
    ) -> Self
    {
        Self {
            elem_id,
            params,
            series: Box::new(series)
        }
    }
}

impl LoadPattern for ElementLoad {
    fn apply_to_global_vector(
        &self,
        pseudo_time: f64,
        model:       &Model,
        f_ext:       &mut [f64],
        alpha:       f64
    ) {
        let Some(elem) = model.elements.get(self.elem_id.0) else {
            // Element index out of range — silently skip rather than panic.
            return;
        };

        let scale   = self.series.factor_at(pseudo_time) * alpha;
        let n_local = elem.n_dof();
        
        // 1. Create a buffer perfectly sized for this element
        let mut f_enq = vec![0.0; n_local];
        
        // 2. Compute the equivalent forces directly into the buffer!
        elem.equivalent_nodal_forces(&self.params, &mut f_enq);
        
        let dof_map = elem.dof_map();

        // 3. Scatter into the global external force vector
        for (local_i, &global_dof) in dof_map.as_usize_slice().iter().enumerate() {
            if global_dof < f_ext.len() {
                f_ext[global_dof] += f_enq[local_i] * scale;
            }
        }
    }

    fn clone_box(&self) -> Box<dyn LoadPattern> {
        Box::new(ElementLoad {
            elem_id: self.elem_id,
            params:  self.params.clone(),
            series:  self.series.clone_box(),
        })
    }

    fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let branch = if is_last { "└── " } else { "├── " };
        let desc = match &self.params {
            ElementLoadParams::Uniform { wx, wy } =>
                format!("Uniform(wx={wx:.3e}, wy={wy:.3e})"),
            ElementLoadParams::Point { px, py, xi } =>
                format!("Point(px={px:.3e}, py={py:.3e}, xi={xi:.2})"),
            ElementLoadParams::Trapezoidal { wy_i, wy_j, .. } =>
                format!("Trapezoidal(wy_i={wy_i:.3e}, wy_j={wy_j:.3e})"),
        };
        format!("{prefix}{branch}ElementLoad (elem={}, {desc})\n", self.elem_id)
    }
}

unsafe impl Send for ElementLoad {}
unsafe impl Sync for ElementLoad {}


// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{ModelDim, NodeId};
    use crate::loads::series::{ConstantSeries, LinearSeries};
    use crate::model::{Model, Node};

    fn frame_model_2nodes() -> Model {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        m.build_state();
        m
    }

    fn truss_model_2nodes() -> Model {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        m.build_state();
        m
    }

    // ---- NodalLoad with ConstantSeries ----

    #[test]
    fn nodal_load_constant_scatters_at_node1() {
        let m = frame_model_2nodes(); // ndf = 3
        let load = NodalLoad {
            node_id:         NodeId(1),
            reference_loads: vec![10.0, -20.0, 5.0],
            series:          Box::new(ConstantSeries),
        };
        let mut f = vec![0.0_f64; 6];
        load.apply_to_global_vector(1.0, &m, &mut f, 1.0);

        // Node 1: global DOFs 3, 4, 5
        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);
        assert!((f[3] -  10.0).abs() < 1e-14);
        assert!((f[4] - -20.0).abs() < 1e-14);
        assert!((f[5] -   5.0).abs() < 1e-14);
    }

    #[test]
    fn nodal_load_accumulates_additively() {
        let m = frame_model_2nodes();
        let load1 = NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![1.0, 0.0, 0.0],
            series:          Box::new(ConstantSeries),
        };
        let load2 = NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![2.0, 0.0, 0.0],
            series:          Box::new(ConstantSeries),
        };
        let mut f = vec![0.0_f64; 6];
        load1.apply_to_global_vector(1.0, &m, &mut f, 1.0);
        load2.apply_to_global_vector(1.0, &m, &mut f, 1.0);
        assert!((f[0] - 3.0).abs() < 1e-14); // 1 + 2 = 3
    }

    // ---- NodalLoad with LinearSeries ----

    #[test]
    fn nodal_load_linear_scales_with_pseudo_time() {
        let m = frame_model_2nodes();
        let load = NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![100.0, 0.0, 0.0],
            series:          Box::new(LinearSeries),
        };
        let mut f = vec![0.0_f64; 6];
        load.apply_to_global_vector(0.3, &m, &mut f, 1.0);
        assert!((f[0] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn nodal_load_zero_pseudo_time_gives_zero() {
        let m = frame_model_2nodes();
        let load = NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![100.0, 0.0, 0.0],
            series:          Box::new(LinearSeries),
        };
        let mut f = vec![0.0_f64; 6];
        load.apply_to_global_vector(0.0, &m, &mut f, 1.0);
        assert!(f.iter().all(|&v| v == 0.0));
    }

    // ---- 2D truss (ndf = 2) ----

    #[test]
    fn nodal_load_truss_ndf2() {
        let m = truss_model_2nodes(); // ndf = 2
        let load = NodalLoad {
            node_id:         NodeId(1),
            reference_loads: vec![0.0, -50e3],
            series:          Box::new(ConstantSeries),
        };
        let mut f = vec![0.0_f64; 4]; // 2 nodes × 2 DOF
        load.apply_to_global_vector(1.0, &m, &mut f, 1.0);
        // Node 1: DOFs 2 and 3
        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);
        assert!((f[3] + 50e3).abs() < 1e-3);
    }
}