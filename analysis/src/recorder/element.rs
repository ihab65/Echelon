//! [`ElementRecorder`] — capture element internal forces at each step.
//!
//! For beam elements, the internal force vector (in global coordinates)
//! contains `[N_i, V_i, M_i, N_j, V_j, M_j]`. This recorder stores
//! the raw `f_int` vector at each step, from which shear/moment diagrams
//! can be reconstructed in post-processing.

use assembly::Model;
use super::Recorder;

/// Records the internal force vector of a specific element after each step.
///
/// The internal force vector `f_int` (in global coordinates, length = n_dof)
/// is computed from the element's current local displacement state.
/// For a 2D beam: `[Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]`.
pub struct ElementRecorder {
    /// Index of the element to record.
    elem_id: usize,
    /// Pseudo-time at each recorded step.
    pub times: Vec<f64>,
    /// `f_int` snapshot per step; inner `Vec` length = element n_dof.
    pub data: Vec<Vec<f64>>,
}

impl ElementRecorder {
    /// Record element internal forces for `elem_id`.
    pub fn new(elem_id: usize) -> Self {
        Self { elem_id, times: Vec::new(), data: Vec::new() }
    }

    /// All recorded `f_int` snapshots.
    pub fn data(&self) -> &[Vec<f64>] {
        &self.data
    }

    /// The pseudo-time history.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Extract the axial force history at node I (DOF 0 of local vector).
    ///
    /// For a horizontal beam, DOF 0 is `Fx` at node I (positive = tension
    /// at the left end from Newton's 3rd law: element pushes node I to the
    /// left, so positive `f_int[0]` means the element is in compression).
    pub fn axial_at_i(&self) -> Vec<f64> {
        self.data.iter().map(|snap| snap.first().copied().unwrap_or(0.0)).collect()
    }

    /// Extract the shear force history at node I (DOF 1 for beam, DOF 1 for truss).
    pub fn shear_at_i(&self) -> Vec<f64> {
        self.data.iter().map(|snap| snap.get(1).copied().unwrap_or(0.0)).collect()
    }

    /// Extract the bending moment history at node I (DOF 2 for beam).
    pub fn moment_at_i(&self) -> Vec<f64> {
        self.data.iter().map(|snap| snap.get(2).copied().unwrap_or(0.0)).collect()
    }
}

impl Recorder for ElementRecorder {
    fn record(&mut self, pseudo_time: f64, model: &Model) {
        self.times.push(pseudo_time);

        let Some(elem) = model.elements.get(self.elem_id) else {
            self.data.push(Vec::new());
            return;
        };

        let dof_map = elem.dof_map();
        let u_local: Vec<f64> = dof_map
            .as_usize_slice()
            .iter()
            .map(|&g| if g < model.u_global.len() { model.u_global[g] } else { 0.0 })
            .collect();

        self.data.push(elem.f_int(&u_local));
    }

    fn description(&self) -> String {
        format!("ElementRecorder[elem={}]", self.elem_id)
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
    use materials::ElasticUniaxial;
    use elements::ElasticBeam2d;
    use assembly::model::{Model, Node};

    fn cantilever_model() -> Model {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                ElasticUniaxial::new(200e9, None).unwrap(),
                0.01, 1e-4,
            ).unwrap()
        );
        m.build_state();
        m
    }

    #[test]
    fn zero_displacement_gives_zero_f_int() {
        let m = cantilever_model();
        let mut rec = ElementRecorder::new(0);
        rec.record(0.0, &m);
        assert_eq!(rec.data().len(), 1);
        assert!(rec.data()[0].iter().all(|&v| v.abs() < 1e-14));
    }

    #[test]
    fn nonzero_displacement_nonzero_f_int() {
        let mut m = cantilever_model();
        m.u_global[4] = -0.01; // tip deflection
        let mut rec = ElementRecorder::new(0);
        rec.record(1.0, &m);
        let f = &rec.data()[0];
        // With tip deflection, shear at node I (index 1) must be non-zero
        assert!(f[1].abs() > 0.0, "shear should be non-zero: {f:?}");
    }

    #[test]
    fn invalid_element_gives_empty_snapshot() {
        let m = cantilever_model();
        let mut rec = ElementRecorder::new(999);
        rec.record(1.0, &m);
        assert_eq!(rec.data()[0].len(), 0);
    }

    #[test]
    fn multiple_steps() {
        let mut m = cantilever_model();
        let mut rec = ElementRecorder::new(0);
        for i in 1..=3 {
            m.u_global[4] = -(i as f64) * 0.001;
            rec.record(i as f64 * 0.1, &m);
        }
        assert_eq!(rec.times().len(), 3);
        assert_eq!(rec.data().len(), 3);
    }
}