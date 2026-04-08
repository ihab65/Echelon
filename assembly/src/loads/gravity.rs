//! Self-weight load pattern computed from element mass matrices.
//!
//! [`GravityLoad`] evaluates `F = mass × g` for every element at
//! assembly time — no static vector is stored. This lets it participate
//! in [`LoadCombo`](crate::loads::combo::LoadCombo) with arbitrary scale
//! factors, which is essential for code-compliant load combinations such
//! as `1.35·DL + 1.5·LL`.
//!
//! ## Requirements
//!
//! Every element whose self-weight should be included must have a density
//! `rho` set on its material. Elements with `rho = None` contribute zero
//! (their lumped mass is all-zero).

use crate::loads::pattern::LoadPattern;
use crate::loads::series::TimeSeries;
use crate::model::Model;

/// Load pattern that computes element self-weight on the fly.
///
/// The gravity acceleration vector has one component per model DOF type.
/// For a 2D frame model (`ndf = 3`, DOFs: UX, UY, RZ):
/// `gravity = [0.0, -9.81, 0.0]` applies downward gravity to the Y DOFs.
///
/// # Example
///
/// ```rust,ignore
/// use assembly::loads::gravity::GravityLoad;
/// use assembly::loads::series::ConstantSeries;
///
/// model.add_load_typed(GravityLoad::new(
///     vec![0.0, -9.81, 0.0],  // 2D frame
///     Box::new(ConstantSeries),
/// ));
/// ```
pub struct GravityLoad {
    /// Acceleration vector, one component per local DOF index within a node.
    /// Length must equal `model.dim.ndf()` at apply time.
    pub gravity: Vec<f64>,
    /// Temporal scale (use `ConstantSeries` for static self-weight).
    pub series: Box<dyn TimeSeries>,
}

impl GravityLoad {
    /// Construct from an explicit gravity vector and time series.
    pub fn new<S: TimeSeries + 'static>(gravity: Vec<f64>, series: S) -> Self {
        Self { 
            gravity, 
            series: Box::new(series) 
        }
    }

    /// Convenience: constant downward gravity for a 2D frame model.
    ///
    /// Equivalent to `GravityLoad::new(vec![0.0, -g, 0.0], Box::new(ConstantSeries))`.
    pub fn frame_2d(g: f64) -> Self {
        use crate::loads::series::ConstantSeries;
        Self::new(vec![0.0, -g, 0.0], ConstantSeries)
    }

    /// Convenience: constant downward gravity for a 2D truss model.
    pub fn truss_2d(g: f64) -> Self {
        use crate::loads::series::ConstantSeries;
        Self::new(vec![0.0, -g], ConstantSeries)
    }
}

impl LoadPattern for GravityLoad {
    fn apply_to_global_vector(
        &self,
        pseudo_time: f64,
        model:       &Model,
        f_ext:       &mut [f64],
    ) {
        let scale = self.series.factor_at(pseudo_time);
        let ndf   = model.dim.ndf();

        for elem in model.elements.iter() {
            let mass    = elem.mass_flat(); // flat n×n lumped mass matrix
            let n_local = elem.n_dof();
            let dof_map = elem.dof_map();
            let globals = dof_map.as_usize_slice();

            // For a lumped mass matrix, only the diagonal matters.
            // diagonal[i] = mass[i * n_local + i]
            for local_i in 0..n_local {
                let m_ii       = mass[local_i * n_local + local_i];
                let dof_type   = local_i % ndf; // which DOF within the node
                let global_dof = globals[local_i];

                if global_dof < f_ext.len() && dof_type < self.gravity.len() {
                    // F_gravity = m * g  (g is already signed, e.g. -9.81)
                    f_ext[global_dof] += m_ii * self.gravity[dof_type] * scale;
                }
            }
        }
    }

    fn clone_box(&self) -> Box<dyn LoadPattern> {
        Box::new(GravityLoad {
            gravity: self.gravity.clone(),
            series:  self.series.clone_box(),
        })
    }

    fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let branch = if is_last { "└── " } else { "├── " };
        format!("{prefix}{branch}GravityLoad (g={:?})\n", self.gravity)
    }
}

unsafe impl Send for GravityLoad {}
unsafe impl Sync for GravityLoad {}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{ModelDim, NodeId};
    use materials::ElasticUniaxial;
    use elements::ElasticBeam2d;
    use crate::model::{Model, Node};

    fn beam_model_with_density() -> Model {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 2.0, 0.0, 0.0)).unwrap();
        m.add_element(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                ElasticUniaxial::new(200e9, Some(7850.0)).unwrap(),
                0.01, 1e-4,
            ).unwrap()
        );
        m.build_state();
        m
    }

    #[test]
    fn gravity_y_dofs_are_negative() {
        let m = beam_model_with_density();
        let g = GravityLoad::frame_2d(9.81);
        let mut f = vec![0.0_f64; 6];
        g.apply_to_global_vector(1.0, &m, &mut f);
        // UY DOFs (indices 1 and 4) must be negative (downward)
        assert!(f[1] < 0.0, "f[1]={}", f[1]);
        assert!(f[4] < 0.0, "f[4]={}", f[4]);
        // UX and RZ must be zero
        assert_eq!(f[0], 0.0);
        assert_eq!(f[2], 0.0);
        assert_eq!(f[3], 0.0);
        assert_eq!(f[5], 0.0);
    }

    #[test]
    fn gravity_total_force_equals_weight() {
        let m = beam_model_with_density();
        let g = GravityLoad::frame_2d(9.81);
        let mut f = vec![0.0_f64; 6];
        g.apply_to_global_vector(1.0, &m, &mut f);
        let total: f64 = f.iter().sum();
        // rho*A*L*g = 7850 * 0.01 * 2 * 9.81
        let expected = -7850.0 * 0.01 * 2.0 * 9.81;
        assert!((total - expected).abs() / expected.abs() < 1e-10,
            "total={total:.4} expected={expected:.4}");
    }

    #[test]
    fn gravity_zero_for_no_density() {
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
        let g = GravityLoad::frame_2d(9.81);
        let mut f = vec![0.0_f64; 6];
        g.apply_to_global_vector(1.0, &m, &mut f);
        assert!(f.iter().all(|&v| v == 0.0), "no density → zero gravity");
    }

    #[test]
    fn clone_box_produces_same_result() {
        let m = beam_model_with_density();
        let g       = GravityLoad::frame_2d(9.81);
        let g_clone = g.clone_box();
        let mut f1 = vec![0.0_f64; 6];
        let mut f2 = vec![0.0_f64; 6];
        g.apply_to_global_vector(1.0, &m, &mut f1);
        g_clone.apply_to_global_vector(1.0, &m, &mut f2);
        assert_eq!(f1, f2);
    }
}