//! 2D linear elastic truss element.
//!
//! `Truss2d` connects two nodes in 2D space and resists only axial force.
//! It is the simplest energy-based element and serves as the reference
//! implementation demonstrating how `Element`, `DifferentiableElement`, and
//! `Assembleable` fit together.
//!
//! ## DOF convention
//!
//! Local DOF order (global coordinates): `[u1_x, u1_y, u2_x, u2_y]`
//!
//! ## Strain energy
//!
//! ```text
//! W = ½ · (EA/L) · ε²   where   ε = (u2_L - u1_L) / L
//!                                 u_L = cos·u_x + sin·u_y
//! ```
//!
//! ## Parameter index convention (for `Assembleable::partial_residual_wrt_param`)
//!
//! | Index | Symbol | Description                |
//! |-------|--------|----------------------------|
//! | 0     | `E`    | Young's modulus (Pa)       |
//! | 1     | `A`    | Cross-section area (m²)    |

use fem_core::{CoordTransf2d, DofMap, NodeId};
use materials::ElasticUniaxial;
use materials::UniaxialMaterial;

use crate::local::truss::{axial_strain, f_int_global, stiffness_global};
use crate::traits::{Assembleable, DifferentiableElement, Element};
use crate::error::{Result, ElementError};

// -----------------------------------------------------------------
// Struct
// -----------------------------------------------------------------

/// 2D linear elastic truss element.
///
/// # Example
///
/// ```rust
/// use elements::Element;
/// use materials::ElasticUniaxial;
/// use elements::Truss2d;
/// use fem_core::NodeId;
///
/// // Horizontal truss element from (0,0) to (2,0), steel-like properties
/// let truss = Truss2d::new(
///     NodeId(0), NodeId(1),
///     0.0, 0.0,   // node 1 coordinates
///     2.0, 0.0,   // node 2 coordinates
///     ElasticUniaxial::new(200e9, None).unwrap(),      // E (Pa)
///     0.01,       // A (m²)
/// );
/// assert_eq!(truss.unwrap().n_dof(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct Truss2d {
    /// Global DOF map: `dof_map[local] = GlobalDof`.
    dof_map: DofMap,
    /// Geometric properties (cos, sin, length) — immutable after construction.
    transf: CoordTransf2d<f64>,
    /// Elastic material (owns E; shared for Engine A and Engine B paths).
    material: ElasticUniaxial,
    /// Pre-computed `EA/L` — used in every stiffness / residual call.
    ea_over_l: f64,
}

impl Truss2d {
    /// Construct a `Truss2d` from node coordinates and section properties.
    ///
    /// # Arguments
    /// * `node1`, `node2` — node indices in the global mesh
    /// * `x1, y1` — coordinates of node 1 (m)
    /// * `x2, y2` — coordinates of node 2 (m)
    /// * `e` — Young's modulus (Pa)
    /// * `a` — cross-section area (m²)
    ///
    /// # Errors
    /// - [`ElementError::InadmissibleSection`] if `a <= 0.0`.
    /// - [`CoreError::DegenerateGeometry`] if `node1` and `node2` are coincident.
    pub fn new(
        node1: NodeId, node2: NodeId,
        x1: f64, y1: f64,
        x2: f64, y2: f64,
        material: ElasticUniaxial,
        a: f64,
    ) -> Result<Self> {
        if a <= 0.0 {
            return Err(ElementError::InadmissibleSection {
                element_type: "Truss2d",
                parameter: "A (cross-section area)",
                value: a,
                requirement: "must be > 0",
            });
        }

        let transf   = CoordTransf2d::from_nodes(x1, y1, x2, y2)?;
        let ea_over_l = material.e * a / transf.length;

        // 2D truss: 2 DOFs per node (ndf = 2)
        let dof_map = DofMap::from_nodes(&[node1, node2], 2);

        Ok(Self { dof_map, transf, material, ea_over_l })
    }

    // ---- Geometry accessors ----

    /// Element length (m).
    #[inline]
    pub fn length(&self) -> f64 { self.transf.length }

    /// cos θ of element inclination.
    #[inline]
    pub fn cos(&self) -> f64 { self.transf.cos }

    /// sin θ of element inclination.
    #[inline]
    pub fn sin(&self) -> f64 { self.transf.sin }

    /// Young's modulus (Pa).
    #[inline]
    pub fn e(&self) -> f64 { self.material.e }

    /// Cross-section area (m²).
    pub fn a(&self) -> f64 { self.ea_over_l * self.transf.length / self.material.e }

    /// EA/L (axial stiffness).
    #[inline]
    pub fn ea_over_l(&self) -> f64 { self.ea_over_l }
}

// -----------------------------------------------------------------
// Element trait
// -----------------------------------------------------------------

impl Element for Truss2d {
    #[inline]
    fn n_dof(&self) -> usize { 4 }

    fn ke_flat(&self, _u: &[f64], out: &mut [f64]) {
        // Linear elastic: stiffness is displacement-independent.
        debug_assert_eq!(out.len(), 16);
        out.copy_from_slice(
            &stiffness_global(self.ea_over_l, self.transf.cos, self.transf.sin)
        );
    }

    fn mass_flat(&self, out: &mut [f64]) {
        // If the user didn't define rho, the mass is zero.
        // The assembly/analysis (Eigen or Transiant) crate 
        // will intercept all-zero mass matrices if needed.
        let rho = self.material.rho.unwrap_or(0.0);
        let m_total = rho * self.a() * self.length();
        let m_half = m_total / 2.0;

        debug_assert_eq!(out.len(), 16);
        out.fill(0.0); // 4x4 matrix for 2D truss
        
        out[0] = m_half;  // node 1 ux
        out[5] = m_half;  // node 1 uy
        out[10] = m_half; // node 2 ux
        out[15] = m_half; // node 2 uy
    }

    fn f_int(&self, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(u.len(), 4);
        let eps = axial_strain(u, self.transf.cos, self.transf.sin, self.transf.length);
        out.copy_from_slice(
            &f_int_global(
                self.ea_over_l, 
                self.transf.cos, 
                self.transf.sin, 
                eps * self.transf.length
            )
        );
    }

    fn commit(&mut self, u: &[f64]) -> Result<()> {
        debug_assert_eq!(u.len(), 4);
        let eps = axial_strain(u, self.transf.cos, self.transf.sin, self.transf.length);
        self.material.commit_state(eps)?;
        Ok(())
    }

    fn revert(&mut self) {
        self.material.revert_to_last_commit();
    }

    fn clone_box(&self) -> Box<dyn Element> {
        Box::new(self.clone())
    }

    fn type_name(&self) -> &'static str { "Truss2d" }

    fn equivalent_nodal_forces(&self, params: &crate::traits::ElementLoadParams, out: &mut [f64]) {
        use crate::traits::ElementLoadParams;

        let l = self.transf.length;
        let c = self.transf.cos;
        let s = self.transf.sin;

        // Trusses resist only axial force. Project the load onto the local
        // axis and apply simple lever-arm interpolation. Transverse components
        // produce zero axial force and are silently ignored — a transverse
        // load on a pin-ended truss member is not physically meaningful.
        match *params {
            ElementLoadParams::Uniform { wx, wy } => {
                // Axial projection: wx_local = wx·cos + wy·sin
                let wx_l = wx * c + wy * s;
                let n    = wx_l * l / 2.0;
                // Global force vector: [-c, -s] at node I, [c, s] at node J
                out.copy_from_slice(&[-n * c, -n * s, n * c, n * s])
            }

            ElementLoadParams::Point { px, py, xi } => {
                let xi   = xi.clamp(0.0, 1.0);
                let a    = xi * l;
                let b    = l - a;
                let px_l = px * c + py * s; // axial projection
                let n_i  = px_l * (b / l);
                let n_j  = px_l * (a / l);
                out.copy_from_slice(&[-n_i * c, -n_i * s, n_j * c, n_j * s])
            }

            ElementLoadParams::Trapezoidal { wx_i, wx_j, wy_i, wy_j } => {
                let wx_l_i = wx_i * c + wy_i * s;
                let wx_l_j = wx_j * c + wy_j * s;
                let dwx    = wx_l_j - wx_l_i;
                let n_i    = wx_l_i * l / 2.0 + dwx * l / 6.0;
                let n_j    = wx_l_i * l / 2.0 + dwx * l / 3.0;
                out.copy_from_slice(&[-n_i * c, -n_i * s, n_j * c, n_j * s])
            }
        }
    }
}

// -----------------------------------------------------------------
// DifferentiableElement trait
// -----------------------------------------------------------------

impl DifferentiableElement for Truss2d {
    /// Scalar strain energy: `W = ½ · (EA/L) · ε²`.
    ///
    /// Written in global-coordinate DOFs so the trigonometric projection
    /// (cos/sin) is included.  When this is evaluated with dual numbers,
    /// the full gradient (including geometric sensitivity) flows through.
    fn energy_f64(&self, u: &[f64]) -> f64 {
        debug_assert_eq!(u.len(), 4);
        let eps = axial_strain(u, self.transf.cos, self.transf.sin, self.transf.length);
        let delta = eps * self.transf.length;
        0.5 * self.ea_over_l * delta * delta
    }

    /// Override the default finite-difference stiffness with the closed-form
    /// expression — exact and fast.
    fn ke_flat_from_energy(&self, u: &[f64], out: &mut [f64]) {
        // For a linear element ke_flat_from_energy == ke_flat.
        self.ke_flat(u, out)
    }

    /// Override the default finite-difference residual with the closed-form
    /// expression.
    fn f_int_from_energy(&self, u: &[f64], out: &mut [f64]) {
        self.f_int(u, out)
    }
}

// -----------------------------------------------------------------
// Assembleable trait
// -----------------------------------------------------------------

/// Parameter indices for `Truss2d`.
pub mod params {
    /// Index 0: Young's modulus `E` (Pa).
    pub const E: usize = 0;
    /// Index 1: Cross-section area `A` (m²).
    pub const A: usize = 1;
}

impl Assembleable for Truss2d {
    fn dof_map(&self) -> &DofMap {
        &self.dof_map
    }

    fn partial_residual_wrt_param(&self, u_local: &[f64], param_idx: usize, out: &mut [f64]) -> Result<()> {
        debug_assert_eq!(u_local.len(), 4);
        let c   = self.transf.cos;
        let s   = self.transf.sin;
        let l   = self.transf.length;
        let eps = axial_strain(u_local, c, s, l);

        // f_int = (EA/L) * ε * [-c, -s, c, s]
        // ∂f_int/∂E = (A/L) * ε * [-c, -s, c, s]
        // ∂f_int/∂A = (E/L) * ε * [-c, -s, c, s]

        let scale = match param_idx {
            params::E => {
                // f_int = (EA/L) * ε * direction
                // ∂f_int/∂E = (A/L) * ε * direction
                let a_over_l = self.ea_over_l / self.material.e; // = A/L
                a_over_l * eps * l
            }
            params::A => {
                // ∂(EA/L)/∂A = E/L → sensitivity = E * ε
                self.material.e * eps
            }
            _ => return Err(ElementError::UnregisteredParameter {
                element_type: "Truss2d",
                idx: param_idx,
                n_params: self.n_params(),
            }),
        };

        out.copy_from_slice(&[-scale * c, -scale * s, scale * c, scale * s]);

        Ok(())
    }

    fn n_params(&self) -> usize { 2 }

    fn param_name(&self, param_idx: usize) -> &'static str {
        match param_idx {
            params::E => "E (Young's modulus)",
            params::A => "A (cross-section area)",
            _ => panic!("param_idx {param_idx} out of range"),
        }
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Horizontal truss element: E=200 GPa, A=0.01 m², L=2 m
    fn horizontal() -> Truss2d {
        Truss2d::new(
            NodeId(0), NodeId(1), 
            0.0, 0.0, 2.0, 0.0, 
            ElasticUniaxial::new(200e9, None).unwrap(), 
            0.01
        ).unwrap()
    }

    /// 45° truss element: from (0,0) to (1,1)
    fn diagonal() -> Truss2d {
        Truss2d::new(
            NodeId(0), NodeId(1), 
            0.0, 0.0, 1.0, 1.0, 
            ElasticUniaxial::new(200e9, None).unwrap(), 
            0.01
        ).unwrap()
    }

    // ---- Construction ----

    #[test]
    fn horizontal_geometry() {
        let t = horizontal();
        assert!((t.length() - 2.0).abs() < 1e-15);
        assert!((t.cos() - 1.0).abs() < 1e-15);
        assert!(t.sin().abs() < 1e-15);
    }

    #[test]
    fn diagonal_geometry() {
        let t = diagonal();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((t.length() - std::f64::consts::SQRT_2).abs() < 1e-14);
        assert!((t.cos() - s).abs() < 1e-14);
        assert!((t.sin() - s).abs() < 1e-14);
    }

    #[test]
    fn n_dof_is_four() {
        assert_eq!(horizontal().n_dof(), 4);
    }

    // ---- ke_flat ----

    #[test]
    fn ke_flat_horizontal_correct() {
        let t  = horizontal();
        let mut ke = [0.0; 16];
        t.ke_flat(&[0.0; 4], &mut ke);
        let k  = t.ea_over_l();
        // [0,0] = k, [0,2] = -k, [2,0] = -k, [2,2] = k
        assert!((ke[0]  -  k).abs() < 1e-3, "ke[0,0]");
        assert!((ke[2]  - -k).abs() < 1e-3, "ke[0,2]");
        assert!((ke[8]  - -k).abs() < 1e-3, "ke[2,0]");
        assert!((ke[10] -  k).abs() < 1e-3, "ke[2,2]");
    }

    #[test]
    fn ke_flat_symmetric() {
        let mut ke = [0.0; 16];
        diagonal().ke_flat(&[0.0; 4], &mut ke);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (ke[i * 4 + j] - ke[j * 4 + i]).abs() < 1e-10,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    // ---- f_int ----

    #[test]
    fn f_int_zero_displacement_is_zero() {
        let mut f = [0.0; 4];
        horizontal().f_int(&[0.0; 4], &mut f);
        assert!(f.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f_int_axial_tension() {
        let t = horizontal();
        // Elongate node 2 by 1mm in x
        let u = [0.0, 0.0, 1e-3, 0.0];
        let mut f = [0.0; 4];
        t.f_int(&u, &mut f);
        
        let force = t.ea_over_l() * 1e-3;
        assert!((f[0] - -force).abs() < 1.0); // node 1 pulled left
        assert!((f[2] -  force).abs() < 1.0); // node 2 pulled right
        assert!(f[1].abs() < 1e-10);
        assert!(f[3].abs() < 1e-10);
    }

    // ---- energy ----

    #[test]
    fn energy_zero_displacement_is_zero() {
        assert_eq!(horizontal().energy_f64(&[0.0; 4]), 0.0);
    }

    #[test]
    fn energy_positive_for_nonzero_strain() {
        let u = [0.0, 0.0, 1e-3, 0.0];
        assert!(horizontal().energy_f64(&u) > 0.0);
    }

    #[test]
    fn energy_hessian_matches_ke_flat() {
        // ∂²W/∂uᵢ∂uⱼ must equal ke[i,j] for a linear element
        let t  = horizontal();
        
        let mut ke = [0.0; 16];
        t.ke_flat(&[0.0; 4], &mut ke);
        
        let mut ke_from_energy = [0.0; 16];
        t.ke_flat_from_energy(&[0.0; 4], &mut ke_from_energy);
        
        for (i, (a, b)) in ke.iter().zip(ke_from_energy.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e3,  // relaxed tol: EA/L ≈ 1e9, FD accurate to ~1e6
                "ke[{i}]: direct={a:.6e} from_energy={b:.6e}"
            );
        }
    }

    #[test]
    fn f_int_from_energy_matches_f_int() {
        let t = horizontal();
        let u = [0.0, 0.0, 5e-4, 1e-4];
        
        let mut f1 = [0.0; 4];
        t.f_int(&u, &mut f1);
        
        let mut f2 = [0.0; 4];
        t.f_int_from_energy(&u, &mut f2);
        
        for (i, (a, b)) in f1.iter().zip(f2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e4,
                "f_int[{i}]: direct={a:.6e} from_energy={b:.6e}"
            );
        }
    }

    // ---- commit / revert ----

    #[test]
    fn commit_does_not_change_stiffness() {
        let mut t = horizontal();
        
        let mut ke_before = [0.0; 16];
        t.ke_flat(&[0.0; 4], &mut ke_before);
        
        t.commit(&[0.0, 0.0, 1e-3, 0.0]).unwrap();
        
        let mut ke_after = [0.0; 16];
        t.ke_flat(&[0.0; 4], &mut ke_after);
        
        assert_eq!(ke_before, ke_after);
    }

    // ---- Assembleable ----

    #[test]
    fn dof_map_correct_for_horizontal() {
        use fem_core::{GlobalDof, LocalDof};
        let t = horizontal(); // NodeId(0), NodeId(1), ndf=2
        let dm = t.dof_map();
        assert_eq!(dm[LocalDof(0)], GlobalDof(0));
        assert_eq!(dm[LocalDof(1)], GlobalDof(1));
        assert_eq!(dm[LocalDof(2)], GlobalDof(2));
        assert_eq!(dm[LocalDof(3)], GlobalDof(3));
    }

    #[test]
    fn partial_residual_e_direction_correct() {
        let t = horizontal();
        let u = [0.0, 0.0, 1e-3, 0.0]; // ε = 1e-3/2 = 5e-4
        let mut dr = [0.0; 4];
        t.partial_residual_wrt_param(&u, params::E, &mut dr).unwrap();
        // ∂f/∂E = (A/L) * ε * [-c, -s, c, s]
        // A = ea_over_l / E * L = (200e9 * 0.01 / 2) / 200e9 * 2 = 0.01
        let a = 0.01_f64;
        let l = 2.0_f64;
        let eps = 1e-3 / l;
        let scale = a * eps;
        assert!((dr[0] - -scale).abs() < 1e-20, "dr[0]");
        assert!((dr[2] -  scale).abs() < 1e-20, "dr[2]");
        assert!(dr[1].abs() < 1e-25);
        assert!(dr[3].abs() < 1e-25);
    }

    #[test]
    fn n_params_is_two() {
        assert_eq!(horizontal().n_params(), 2);
    }

    #[test]
    fn clone_box_independent_state() {
        let mut t1 = horizontal();
        let mut t2_box = t1.clone_box();
        // Commit different strains
        t1.commit(&[0.0, 0.0, 1e-3, 0.0]).unwrap();
        t2_box.commit(&[0.0, 0.0, 2e-3, 0.0]).unwrap();
        
        // Stiffness matrices should still be equal (linear elastic)
        let mut ke1 = [0.0; 16];
        t1.ke_flat(&[0.0; 4], &mut ke1);
        
        let mut ke2 = [0.0; 16];
        t2_box.ke_flat(&[0.0; 4], &mut ke2);
        
        assert_eq!(ke1, ke2);
    }
}
