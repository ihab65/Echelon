//! Ground motion input and uniform excitation load pattern.
//!
//! [`GroundMotion`] holds a discretely-sampled acceleration record and
//! provides linear interpolation at any simulation time. [`UniformExcitation`]
//! wraps it as a [`LoadPattern`] that computes `P_eff(t) = -M·ι·ü_g(t)` and
//! scatters it into `f_ext`.
//!
//! ## Supported input
//!
//! Acceleration records are provided as `(dt, accelerations)` pairs.
//! Typical sources: PEER NGA-West2 `.AT2` files (parsed externally),
//! NGA-East, CESMD `.csv` exports.
//!
//! ## Sign convention
//!
//! Positive acceleration in the excitation direction produces a **negative**
//! (inertial) force on the structure: `P_eff = -M·ü_g`. Pass the raw
//! record values without sign reversal — the struct handles the physics.
//!
//! ## Example
//!
//! ```rust,ignore
//! // 0.005 s record, e.g. from PEER AT2 after parsing
//! let gm  = GroundMotion::new(0.005, accel_g_values);
//! let ux  = UniformExcitation::new(gm, 0, 9.81); // UX direction, scale by g
//!
//! model.add_load_typed(ux);
//! // TransientDriver calls assemble_load_vector(t) each step — done!
//! ```

use crate::loads::pattern::LoadPattern;
use crate::model::Model;

// -----------------------------------------------------------------
// GroundMotion
// -----------------------------------------------------------------

/// A discretely-sampled earthquake acceleration record.
///
/// Provides `accel_at(t)` via linear interpolation between samples.
/// Outside the record window the acceleration clamps to zero.
#[derive(Debug, Clone)]
pub struct GroundMotion {
    /// Uniform time step between samples (seconds).
    pub dt: f64,
    /// Acceleration samples (any consistent unit — m/s², g, etc.).
    pub accelerations: Vec<f64>,
}

impl GroundMotion {
    /// Build from a time step and a vector of acceleration values.
    ///
    /// # Panics
    /// Panics if `dt <= 0` or `accelerations` is empty.
    pub fn new(dt: f64, accelerations: Vec<f64>) -> Self {
        assert!(dt > 0.0, "GroundMotion: dt must be positive");
        assert!(!accelerations.is_empty(), "GroundMotion: no acceleration data");
        Self { dt, accelerations }
    }

    /// Total duration of the record (seconds).
    pub fn duration(&self) -> f64 {
        self.dt * (self.accelerations.len() - 1) as f64
    }

    /// Number of samples.
    pub fn n_samples(&self) -> usize {
        self.accelerations.len()
    }

    /// Ground acceleration at time `t` via linear interpolation.
    ///
    /// Returns `0.0` outside `[0, duration]`.
    pub fn accel_at(&self, t: f64) -> f64 {
        if t <= 0.0 || t >= self.duration() {
            return 0.0;
        }
        let idx_f = t / self.dt;
        let i0    = idx_f.floor() as usize;
        let i1    = (i0 + 1).min(self.accelerations.len() - 1);
        let alpha = idx_f - i0 as f64;
        self.accelerations[i0] * (1.0 - alpha) + self.accelerations[i1] * alpha
    }
}

// -----------------------------------------------------------------
// UniformExcitation
// -----------------------------------------------------------------

/// Load pattern that applies ground-motion inertial forces to the structure.
///
/// At each time step, the effective load is:
/// ```text
/// P_eff(t) = -M·ι·ü_g(t)
/// ```
/// where `ι` is the influence vector (selects DOFs in the excitation direction).
///
/// The mass matrix is extracted from element `mass_flat()` diagonals,
/// consistent with the lumped mass formulation used by `assemble_mass`.
pub struct UniformExcitation {
    /// The ground acceleration record.
    pub ground_motion: GroundMotion,
    /// Local DOF index within each node that corresponds to the excitation
    /// direction (0 = UX, 1 = UY for 2D models).
    pub dof_dir: usize,
    /// Scale factor applied to the raw acceleration (e.g. `9.81` to convert
    /// from *g* to m/s²).
    pub accel_scale: f64,
}

impl UniformExcitation {
    /// Create a new uniform excitation pattern.
    ///
    /// # Arguments
    /// * `ground_motion` — parsed acceleration record
    /// * `dof_dir`       — local DOF index for the excitation direction (0=UX, 1=UY)
    /// * `accel_scale`   — multiplier applied to raw accelerations (use `9.81` for g → m/s²)
    pub fn new(ground_motion: GroundMotion, dof_dir: usize, accel_scale: f64) -> Self {
        Self { ground_motion, dof_dir, accel_scale }
    }
}

impl LoadPattern for UniformExcitation {
    fn apply_to_global_vector(
        &self,
        pseudo_time: f64,
        model:       &Model,
        f_ext:       &mut [f64],
    ) {
        let accel = self.ground_motion.accel_at(pseudo_time) * self.accel_scale;
        if accel == 0.0 {
            return; // short-circuit for out-of-record times
        }

        let ndf = model.dim.ndf();

        for elem in model.elements.iter() {
            let mass    = elem.mass_flat();
            let n_local = elem.n_dof();
            let dof_map = elem.dof_map();
            let globals = dof_map.as_usize_slice();

            for local_i in 0..n_local {
                let dof_type   = local_i % ndf;
                let global_dof = globals[local_i];

                if dof_type == self.dof_dir && global_dof < f_ext.len() {
                    let m_ii = mass[local_i * n_local + local_i];
                    // P_eff = -M * a_g  (inertial force opposes ground acceleration)
                    f_ext[global_dof] -= m_ii * accel;
                }
            }
        }
    }

    fn clone_box(&self) -> Box<dyn LoadPattern> {
        Box::new(UniformExcitation {
            ground_motion: self.ground_motion.clone(),
            dof_dir:       self.dof_dir,
            accel_scale:   self.accel_scale,
        })
    }

    fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let branch = if is_last { "└── " } else { "├── " };
        format!(
            "{prefix}{branch}UniformExcitation (dir={}, duration={:.2}s, n={} samples)\n",
            self.dof_dir,
            self.ground_motion.duration(),
            self.ground_motion.n_samples(),
        )
    }
}

unsafe impl Send for UniformExcitation {}
unsafe impl Sync for UniformExcitation {}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_record(n: usize, dt: f64, freq: f64) -> GroundMotion {
        let accels: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 * dt).sin())
            .collect();
        GroundMotion::new(dt, accels)
    }

    // ---- GroundMotion ----

    #[test]
    fn accel_at_zero_is_zero_for_sine() {
        let gm = sine_record(1000, 0.005, 1.0);
        assert!(gm.accel_at(0.0).abs() < 1e-14);
    }

    #[test]
    fn accel_at_clamps_outside_record() {
        let gm = GroundMotion::new(0.01, vec![1.0, 2.0, 3.0]);
        assert_eq!(gm.accel_at(-1.0), 0.0);
        assert_eq!(gm.accel_at(999.0), 0.0);
    }

    #[test]
    fn accel_at_interpolates_linearly() {
        let gm = GroundMotion::new(1.0, vec![0.0, 10.0]);
        // At t = 0.5: halfway between 0 and 10 = 5
        assert!((gm.accel_at(0.5) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn duration_correct() {
        let gm = GroundMotion::new(0.01, vec![0.0; 201]); // 200 intervals
        assert!((gm.duration() - 2.0).abs() < 1e-12);
    }

    // ---- UniformExcitation ----

    #[test]
    fn zero_accel_contributes_nothing() {
        use fem_core::{ModelDim, NodeId};
        use materials::ElasticUniaxial;
        use elements::ElasticBeam2d;
        use crate::model::{Model, Node};

        let mut model = Model::new(ModelDim::frame_2d());
        model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        model.add_node(Node::new(NodeId(1), 2.0, 0.0)).unwrap();
        model.add_element_typed(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                ElasticUniaxial::new(200e9, Some(7850.0)).unwrap(),
                0.01, 1e-4,
            ).unwrap()
        );
        model.build_state();

        // Record that returns 0 at t=0 and t>=duration
        let gm  = GroundMotion::new(1.0, vec![0.0, 1.0, 0.0]);
        let exc = UniformExcitation::new(gm, 0, 9.81);

        let mut f = vec![0.0_f64; 6];
        exc.apply_to_global_vector(0.0, &model, &mut f); // t=0 → 0
        assert!(f.iter().all(|&v| v == 0.0));
        exc.apply_to_global_vector(2.0, &model, &mut f); // t=duration → 0
        assert!(f.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn nonzero_accel_produces_negative_inertial_force() {
        use fem_core::{ModelDim, NodeId};
        use materials::ElasticUniaxial;
        use elements::ElasticBeam2d;
        use crate::model::{Model, Node};

        let mut model = Model::new(ModelDim::frame_2d());
        model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        model.add_node(Node::new(NodeId(1), 2.0, 0.0)).unwrap();
        model.add_element_typed(
            ElasticBeam2d::new(
                NodeId(0), NodeId(1), 0.0, 0.0, 2.0, 0.0,
                ElasticUniaxial::new(200e9, Some(7850.0)).unwrap(),
                0.01, 1e-4,
            ).unwrap()
        );
        model.build_state();

        // Record: 0 → 1g at t=0.5s, → 0 at t=1.0s
        let gm  = GroundMotion::new(0.5, vec![0.0, 1.0, 0.0]);
        let exc = UniformExcitation::new(gm, 0, 9.81); // UX direction

        let mut f = vec![0.0_f64; 6];
        exc.apply_to_global_vector(0.5, &model, &mut f); // at peak

        // UX DOFs (0 and 3) must be negative (inertial = -M*a, a > 0)
        assert!(f[0] < 0.0, "f[0]={}", f[0]); // node 0 UX
        assert!(f[3] < 0.0, "f[3]={}", f[3]); // node 1 UX
        // UY and RZ must be zero (excitation is in UX only)
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);
        assert_eq!(f[4], 0.0);
        assert_eq!(f[5], 0.0);
    }
}