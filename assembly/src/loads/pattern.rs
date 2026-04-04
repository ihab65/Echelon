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

use fem_core::NodeId;

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
    );
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

impl LoadPattern for NodalLoad {
    fn apply_to_global_vector(
        &self,
        pseudo_time: f64,
        model:       &Model,
        f_ext:       &mut [f64],
    ) {
        let scale   = self.series.factor_at(pseudo_time);
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
}

// Make NodalLoad Send + Sync — both NodeId and Vec<f64> are already Send + Sync.
// The Box<dyn TimeSeries> bound enforces the same on the series.
unsafe impl Send for NodalLoad {}
unsafe impl Sync for NodalLoad {}

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
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0)).unwrap();
        m.build_state();
        m
    }

    fn truss_model_2nodes() -> Model {
        let mut m = Model::new(ModelDim::truss_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0)).unwrap();
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
        load.apply_to_global_vector(1.0, &m, &mut f);

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
        load1.apply_to_global_vector(1.0, &m, &mut f);
        load2.apply_to_global_vector(1.0, &m, &mut f);
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
        load.apply_to_global_vector(0.3, &m, &mut f);
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
        load.apply_to_global_vector(0.0, &m, &mut f);
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
        load.apply_to_global_vector(1.0, &m, &mut f);
        // Node 1: DOFs 2 and 3
        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);
        assert!((f[3] + 50e3).abs() < 1e-3);
    }
}