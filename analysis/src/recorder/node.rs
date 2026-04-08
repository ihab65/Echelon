//! [`NodeRecorder`] — capture nodal displacements at specified DOFs.

use assembly::Model;
use super::Recorder;

/// Records the displacement at specified global DOF indices after each step.
///
/// The `times` and `data` vectors grow by one entry per converged step.
///
/// # Example — track roof drift (node 10, UY = DOF 31 in a 3-DOF model)
///
/// ```rust,ignore
/// let rec = NodeRecorder::new(31, "roof_uy");
/// ```
pub struct NodeRecorder {
    /// Global DOF indices to record.
    dofs: Vec<usize>,
    /// Pseudo-time at each recorded step.
    pub times: Vec<f64>,
    /// Recorded displacements: `data[step][i]` = displacement at `dofs[i]`.
    pub data: Vec<Vec<f64>>,
    /// Human-readable label.
    label: String,
}

impl NodeRecorder {
    /// Create a recorder for the given global DOF indices.
    ///
    /// # Arguments
    /// * `dofs`  — global DOF indices to track (e.g. `vec![3, 4]` for node 1 UX and UY)
    /// * `label` — identifier used in [`description`]
    pub fn new(dofs: Vec<usize>, label: impl Into<String>) -> Self {
        Self {
            dofs,
            times: Vec::new(),
            data:  Vec::new(),
            label: label.into(),
        }
    }

    /// Convenience: record a single DOF.
    pub fn single(dof: usize, label: impl Into<String>) -> Self {
        Self::new(vec![dof], label)
    }

    /// All recorded displacement snapshots.
    ///
    /// `data()[step]` is a slice of length `dofs.len()`.
    pub fn data(&self) -> &[Vec<f64>] {
        &self.data
    }

    /// Flatten into a single `Vec<f64>` for a single-DOF recorder.
    ///
    /// Useful when tracking one DOF (e.g. roof displacement vs. base shear).
    ///
    /// # Panics
    /// Panics if this recorder was constructed with more than one DOF.
    pub fn flatten(&self) -> Vec<f64> {
        assert_eq!(self.dofs.len(), 1,
            "flatten() is only valid for single-DOF recorders");
        self.data.iter().map(|snap| snap[0]).collect()
    }

    /// The pseudo-time series corresponding to `data()`.
    pub fn times(&self) -> &[f64] {
        &self.times
    }
}

impl Recorder for NodeRecorder {
    fn record(&mut self, pseudo_time: f64, model: &Model) {
        self.times.push(pseudo_time);
        let snap: Vec<f64> = self.dofs.iter()
            .map(|&dof| {
                if dof < model.u_global.len() {
                    model.u_global[dof]
                } else {
                    0.0
                }
            })
            .collect();
        self.data.push(snap);
    }

    fn description(&self) -> String {
        format!("NodeRecorder[{}] dofs={:?}", self.label, self.dofs)
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{ModelDim, NodeId};
    use assembly::model::{Model, Node};

    fn tiny_model() -> Model {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 1.0, 0.0, 0.0)).unwrap();
        m.build_state();
        m
    }

    #[test]
    fn records_correct_dof() {
        let mut m = tiny_model();
        m.u_global[3] = 0.005; // node 1 UX
        m.u_global[4] = -0.01; // node 1 UY

        let mut rec = NodeRecorder::new(vec![3, 4], "node1");
        rec.record(0.5, &m);

        assert_eq!(rec.times(), &[0.5]);
        assert_eq!(rec.data().len(), 1);
        assert!((rec.data()[0][0] - 0.005).abs() < 1e-15);
        assert!((rec.data()[0][1] - (-0.01)).abs() < 1e-15);
    }

    #[test]
    fn multiple_steps_accumulate() {
        let mut m = tiny_model();
        let mut rec = NodeRecorder::single(4, "uy");

        for step in 1..=5 {
            m.u_global[4] = step as f64 * 0.001;
            rec.record(step as f64 * 0.1, &m);
        }

        assert_eq!(rec.times().len(), 5);
        let flat = rec.flatten();
        assert!((flat[4] - 0.005).abs() < 1e-15);
    }

    #[test]
    fn out_of_range_dof_gives_zero() {
        let m = tiny_model();
        let mut rec = NodeRecorder::single(999, "out");
        rec.record(1.0, &m);
        assert_eq!(rec.data()[0][0], 0.0);
    }
}