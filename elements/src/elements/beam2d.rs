//! 2D Euler-Bernoulli elastic beam element.
//!
//! `ElasticBeam2d` connects two nodes with full frame DOFs:
//! `[u_x, u_y, θ]` per node → 6 local DOFs.
//!
//! ## Stiffness
//!
//! Uses the closed-form local stiffness from `local::beam::ke_local` and
//! transforms it to global coordinates via `CoordTransf2d::transform_stiffness_6x6`.
//! No numerical integration is performed — the element is exact for
//! linear Euler-Bernoulli bending.
//!
//! ## Energy function
//!
//! The beam energy is:
//! ```text
//! W = ½ uᵀ Ke u   (expressed in local coordinates)
//! ```
//! For the autodiff path, `energy_f64` computes this via the local
//! stiffness matrix evaluated at the local displacements.
//!
//! ## Parameter index convention
//!
//! | Index | Symbol | Description                         |
//! |-------|--------|-------------------------------------|
//! | 0     | `E`    | Young's modulus (Pa)                |
//! | 1     | `A`    | Cross-section area (m²)             |
//! | 2     | `Iz`   | Second moment of area about z (m⁴)  |

use fem_core::{CoordTransf2d, DofMap, NodeId};
use fem_core::dense::mat_as_slice;
use materials::ElasticUniaxial;
use materials::UniaxialMaterial;

use crate::local::beam::{axial_strain, f_int_local_from_ke, ke_local};
use crate::traits::{Assembleable, DifferentiableElement, Element};
use crate::error::{Result, ElementError};

// -----------------------------------------------------------------
// Struct
// -----------------------------------------------------------------

/// 2D Euler-Bernoulli beam element with linear elastic material.
///
/// # Example
///
/// ```rust
/// use fem_core::NodeId;
/// use elements::Element;
/// use materials::ElasticUniaxial;
/// use elements::ElasticBeam2d;
///
/// // Horizontal 2m cantilever beam, steel-like properties
/// let beam = ElasticBeam2d::new(
///     NodeId(0), NodeId(1),
///     0.0, 0.0,   // node 1
///     2.0, 0.0,   // node 2
///     ElasticUniaxial::new(200e9, None).unwrap(),      // E (Pa)
///     0.01,       // A (m²)
///     1e-4,       // Iz (m⁴)
/// );
/// assert_eq!(beam.unwrap().n_dof(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct ElasticBeam2d {
    /// Global DOF map.
    dof_map: DofMap,
    /// Geometric transformation (cos, sin, length).
    transf: CoordTransf2d<f64>,
    /// Elastic material (owns E, used for commit/revert and adjoint).
    material: ElasticUniaxial,
    /// Cross-section area (m²).
    a: f64,
    /// Second moment of area about z (m⁴).
    iz: f64,
    /// Pre-computed local stiffness — constant for linear elastic element.
    ke_local_cached: [f64; 36],
}

impl ElasticBeam2d {
    /// Construct from node coordinates and section properties.
    ///
    /// # Arguments
    /// * `node1`, `node2` — node indices
    /// * `x1, y1` — coordinates of node 1 (m)
    /// * `x2, y2` — coordinates of node 2 (m)
    /// * `e`  — Young's modulus (Pa)
    /// * `a`  — cross-section area (m²)
    /// * `iz` — second moment of area about z (m⁴)
    pub fn new(
        node1: NodeId, node2: NodeId,
        x1: f64, y1: f64,
        x2: f64, y2: f64,
        material: ElasticUniaxial,
        a: f64, iz: f64,
    ) -> Result<Self> {    
        if a <= 0.0 {
            return Err(ElementError::InadmissibleSection {
                element_type: "ElasticBeam2d",
                parameter: "A",
                value: a,
                requirement: "A > 0",
            });
        }

        if iz <= 0.0 {
            return Err(ElementError::InadmissibleSection {
                element_type: "ElasticBeam2d",
                parameter: "Iz",
                value: iz,
                requirement: "Iz > 0",
            });
        }

        let transf = CoordTransf2d::from_nodes(x1, y1, x2, y2)?;
        let ke_local_cached = ke_local(material.e, a, iz, transf.length);
        // 2D frame: 3 DOFs per node
        let dof_map = DofMap::from_nodes(&[node1, node2], 3);

        Ok(Self { dof_map, transf, material, a, iz, ke_local_cached })
    }

    // ---- Geometry / section accessors ----

    /// Element length (m).
    #[inline]
    pub fn length(&self) -> f64 { self.transf.length }

    /// Young's modulus (Pa).
    #[inline]
    pub fn e(&self) -> f64 { self.material.e }

    /// Cross-section area (m²).
    #[inline]
    pub fn a(&self) -> f64 { self.a }

    /// Second moment of area (m⁴).
    #[inline]
    pub fn iz(&self) -> f64 { self.iz }

    // ---- Internal helpers ----

    /// Transform local displacements (global coords → local frame).
    ///
    /// Applies `T * u_global` using the 6×6 rotation matrix from `CoordTransf2d`.
    fn u_local(&self, u_global: &[f64]) -> [f64; 6] {
        debug_assert_eq!(u_global.len(), 6);
        let t = self.transf.t_matrix_6x6();
        let mut ul = [0.0_f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                ul[i] += t[i][j] * u_global[j];
            }
        }
        ul
    }

    /// Global stiffness as a flat `[f64; 36]`.
    ///
    /// `Kg = Tᵀ Ke_local T`
    fn ke_global_flat(&self) -> [f64; 36] {
        // Build [[f64; 6]; 6] from the cached flat array
        let mut ke_arr = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                ke_arr[i][j] = self.ke_local_cached[i * 6 + j];
            }
        }
        let kg_arr = self.transf.transform_stiffness_6x6(&ke_arr);
        let mut kg_flat = [0.0_f64; 36];
        kg_flat.copy_from_slice(mat_as_slice(&kg_arr));
        kg_flat
    }

    /// Rotate a local 6-DOF force vector to global coordinates: `f_global = Tᵀ f_local`.
    fn rotate_to_global(&self, f_local: &[f64; 6]) -> Vec<f64> {
        let t = self.transf.t_matrix_6x6();
        let mut fg = [0.0_f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                // Tᵀ[i,j] = T[j,i]
                fg[i] += t[j][i] * f_local[j];
            }
        }
        fg.to_vec()
    }
}

// -----------------------------------------------------------------
// Element trait
// -----------------------------------------------------------------

impl Element for ElasticBeam2d {
    #[inline]
    fn n_dof(&self) -> usize { 6 }

    fn ke_flat(&self, _u: &[f64]) -> Vec<f64> {
        // Linear elastic: stiffness is displacement-independent.
        self.ke_global_flat().to_vec()
    }

    fn mass_flat(&self) -> Vec<f64> {
        // If the user didn't define rho, the mass is zero.
        // The assembly/analysis (Eigen or Transiant) crate 
        // will intercept all-zero mass matrices if needed.
        let rho = self.material.rho.unwrap_or(0.0);
        let m_total = rho * self.a() * self.length();
        let m_half = m_total / 2.0;

        let mut lumped_mass = vec![0.0_f64; 36];
        // Node 1 (UX, UY, RZ)
        lumped_mass[0]  = m_half;       // Row 0, Col 0: u1_x
        lumped_mass[7]  = m_half;       // Row 1, Col 1: u1_y
        lumped_mass[14] = 1e-9;         // Row 2, Col 2: u1_theta
        
        // Node 2 (UX, UY, RZ)
        lumped_mass[21] = m_half;       // Row 3, Col 3: u2_x
        lumped_mass[28] = m_half;       // Row 4, Col 4: u2_y
        lumped_mass[35] = 1e-9;         // Row 5, Col 5: u2_theta
        
        lumped_mass
    }

    fn f_int(&self, u: &[f64]) -> Vec<f64> {
        debug_assert_eq!(u.len(), 6);
        // Compute in local frame, then rotate to global.
        let ul = self.u_local(u);
        let fl = f_int_local_from_ke(&self.ke_local_cached, &ul);

        // Rotate back: f_global = Tᵀ * f_local
        let t = self.transf.t_matrix_6x6();
        let mut fg = [0.0_f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                // Tᵀ[i,j] = T[j,i]
                fg[i] += t[j][i] * fl[j];
            }
        }
        fg.to_vec()
    }

    fn commit(&mut self, u: &[f64]) -> Result<()> {
        debug_assert_eq!(u.len(), 6);
        let ul = self.u_local(u);
        let eps = axial_strain(&ul, self.transf.length);
        self.material.commit_state(eps)?;
        Ok(())
    }

    fn revert(&mut self) {
        self.material.revert_to_last_commit();
    }

    fn clone_box(&self) -> Box<dyn Element> {
        Box::new(self.clone())
    }

    fn type_name(&self) -> &'static str { "ElasticBeam2d" }

    fn equivalent_nodal_forces(&self, params: &crate::traits::ElementLoadParams) -> Vec<f64> {
        use crate::traits::ElementLoadParams;

        let l  = self.transf.length;
        let l2 = l * l;
        let l3 = l2 * l;

        // All formulas produce forces in LOCAL coordinates:
        //   [N_i, V_i, M_i, N_j, V_j, M_j]
        // where N = axial, V = shear, M = moment (CCW positive).
        //
        // These are the forces the load applies TO THE NODES (not the
        // fixed-end reactions — we keep the sign that loads the structure).
        //
        // The load components (wx, wy) are in GLOBAL coordinates, so we must
        // project them onto the local axes before applying the FER formulas.
        let c = self.transf.cos;
        let s = self.transf.sin;

        let f_local: [f64; 6] = match *params {
            ElementLoadParams::Uniform { wx, wy } => {
                // Project global (wx, wy) onto local (axial x_L, transverse y_L).
                // Local x-axis: (c, s);  local y-axis: (-s, c)
                let wx_l = wx * c + wy * s;   // axial component
                let wy_l = -wx * s + wy * c;  // transverse component

                let n = wx_l * l / 2.0;
                let v = wy_l * l / 2.0;
                let m = wy_l * l2 / 12.0;

                [n, v, m, n, v, -m]
            }

            ElementLoadParams::Point { px, py, xi } => {
                let xi = xi.clamp(0.0, 1.0);
                let a  = xi * l;         // distance from node I
                let b  = l - a;          // distance from node J

                // Project to local frame
                let px_l =  px * c + py * s;
                let py_l = -px * s + py * c;

                // Axial: simple lever-arm interpolation
                let n_i = px_l * (b / l);
                let n_j = px_l * (a / l);

                // Transverse: Euler-Bernoulli fixed-end reactions
                let v_i = py_l * (b * b * (3.0 * a + b)) / l3;
                let v_j = py_l * (a * a * (a + 3.0 * b)) / l3;

                // Moments (CCW positive)
                let m_i =  py_l * (a * b * b) / l2;
                let m_j = -py_l * (a * a * b) / l2;

                [n_i, v_i, m_i, n_j, v_j, m_j]
            }

            ElementLoadParams::Trapezoidal { wx_i, wx_j, wy_i, wy_j } => {
                // Project both endpoints to local frame
                let wx_l_i =  wx_i * c + wy_i * s;
                let wy_l_i = -wx_i * s + wy_i * c;
                let wx_l_j =  wx_j * c + wy_j * s;
                let wy_l_j = -wx_j * s + wy_j * c;

                // Axial: trapezoidal rule
                let dwx = wx_l_j - wx_l_i;
                let n_i = wx_l_i * l / 2.0 + dwx * l / 6.0;
                let n_j = wx_l_i * l / 2.0 + dwx * l / 3.0;

                // Transverse: superpose uniform (at wy_l_i) + triangular (Δwy)
                let dwy = wy_l_j - wy_l_i;

                // Uniform part (magnitude wy_l_i)
                let v_i_u =  wy_l_i * l  / 2.0;
                let v_j_u =  wy_l_i * l  / 2.0;
                let m_i_u =  wy_l_i * l2 / 12.0;
                let m_j_u = -wy_l_i * l2 / 12.0;

                // Triangular part (peaking at node J with magnitude dwy)
                let v_i_t =  dwy * 3.0 * l  / 20.0;
                let v_j_t =  dwy * 7.0 * l  / 20.0;
                let m_i_t =  dwy * l2        / 30.0;
                let m_j_t = -dwy * l2        / 20.0;

                [
                    n_i,
                    v_i_u + v_i_t,
                    m_i_u + m_i_t,
                    n_j,
                    v_j_u + v_j_t,
                    m_j_u + m_j_t,
                ]
            }
        };

        self.rotate_to_global(&f_local)
    }
}

// -----------------------------------------------------------------
// DifferentiableElement trait
// -----------------------------------------------------------------

impl DifferentiableElement for ElasticBeam2d {
    /// Strain energy `W = ½ uᵀ K u` evaluated in the local frame.
    ///
    /// Note: for the linear beam the energy-derived stiffness equals the
    /// analytic stiffness exactly.  The `DifferentiableElement` implementation
    /// here serves as a correctness check and as the foundation for future
    /// nonlinear extensions.
    fn energy_f64(&self, u: &[f64]) -> f64 {
        debug_assert_eq!(u.len(), 6);
        // W = ½ u_local^T * Ke_local * u_local
        let ul = self.u_local(u);
        let fl = f_int_local_from_ke(&self.ke_local_cached, &ul);
        // W = ½ uᵀ f_int  (since f_int = Ke u for linear elastic)
        0.5 * ul.iter().zip(fl.iter()).map(|(u, f)| u * f).sum::<f64>()
    }

    /// Override with the closed-form global stiffness — exact for linear beam.
    fn ke_flat_from_energy(&self, u: &[f64]) -> Vec<f64> {
        self.ke_flat(u)
    }

    /// Override with the closed-form residual.
    fn f_int_from_energy(&self, u: &[f64]) -> Vec<f64> {
        self.f_int(u)
    }
}

// -----------------------------------------------------------------
// Assembleable trait
// -----------------------------------------------------------------

/// Parameter indices for `ElasticBeam2d`.
pub mod params {
    /// Index 0: Young's modulus `E` (Pa).
    pub const E: usize = 0;
    /// Index 1: Cross-section area `A` (m²).
    pub const A: usize = 1;
    /// Index 2: Second moment of area `Iz` (m⁴).
    pub const IZ: usize = 2;
}

impl Assembleable for ElasticBeam2d {
    fn dof_map(&self) -> &DofMap {
        &self.dof_map
    }

    fn partial_residual_wrt_param(&self, u_global: &[f64], param_idx: usize) -> Result<Vec<f64>> {
        debug_assert_eq!(u_global.len(), 6);

        let e  = self.material.e;
        let a  = self.a;
        let iz = self.iz;
        let l  = self.transf.length;

        // The local internal force is: f_int_local = Ke_local * u_local
        // Differentiating with respect to a parameter θ:
        //   ∂f_int_global/∂θ = Tᵀ * (∂Ke_local/∂θ) * u_local
        //
        // We compute (∂Ke_local/∂θ) analytically and apply it to u_local.

        let ul = self.u_local(u_global);

        // Compute ∂Ke_local/∂θ * u_local directly
        let dfl_local: [f64; 6] = match param_idx {
            params::E => {
                // ∂Ke/∂E: replace E with 1 in the stiffness formula
                let dke = ke_local(1.0, a, iz, l);
                f_int_local_from_ke(&dke, &ul)
            }
            params::A => {
                // Axial part only: ∂Ke/∂A = (E/L) * axial_block
                // ke_axial = (E*A/L) block → ∂/∂A gives (E/L) block
                let dke = ke_local(e, 1.0, 0.0, l); // A=1, Iz=0 → axial only
                f_int_local_from_ke(&dke, &ul)
            }
            params::IZ => {
                // Bending part only: ∂Ke/∂Iz = bending_block / Iz
                // ke_bending = E*Iz*bending_shape → ∂/∂Iz = E*bending_shape
                let dke = ke_local(e, 0.0, 1.0, l); // A=0, Iz=1 → bending only
                f_int_local_from_ke(&dke, &ul)
            }
            _ => return Err(ElementError::UnregisteredParameter {
                element_type: "ElasticBeam2d",
                idx: param_idx,
                n_params: self.n_params(),
            }),
        };

        // Rotate back to global: ∂f_global/∂θ = Tᵀ * ∂f_local/∂θ
        let t = self.transf.t_matrix_6x6();
        let mut dfg = [0.0_f64; 6];
        for i in 0..6 {
            for j in 0..6 {
                dfg[i] += t[j][i] * dfl_local[j];
            }
        }
        Ok(dfg.to_vec())
    }

    fn n_params(&self) -> usize { 3 }

    fn param_name(&self, param_idx: usize) -> &'static str {
        match param_idx {
            params::E  => "E (Young's modulus)",
            params::A  => "A (cross-section area)",
            params::IZ => "Iz (second moment of area)",
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

    fn cantilever() -> ElasticBeam2d {
        // 2m horizontal beam, steel-like properties
        ElasticBeam2d::new(
            NodeId(0), NodeId(1),
            0.0, 0.0, 2.0, 0.0, 
            ElasticUniaxial::new(200e9, None).unwrap(), 
            0.01, 1e-4
        ).unwrap()
    }

    // ---- Construction ----

    #[test]
    fn n_dof_is_six() {
        assert_eq!(cantilever().n_dof(), 6);
    }

    #[test]
    fn geometry_correct() {
        let b = cantilever();
        assert!((b.length() - 2.0).abs() < 1e-15);
        assert!((b.e() - 200e9).abs() < 1.0);
        assert!((b.a() - 0.01).abs() < 1e-15);
        assert!((b.iz() - 1e-4).abs() < 1e-20);
    }

    // ---- ke_flat ----

    #[test]
    fn ke_flat_symmetric() {
        let ke = cantilever().ke_flat(&[0.0; 6]);
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (ke[i * 6 + j] - ke[j * 6 + i]).abs() < 1e-6,
                    "not symmetric at ({i},{j}): {} vs {}",
                    ke[i * 6 + j], ke[j * 6 + i]
                );
            }
        }
    }

    #[test]
    fn ke_flat_axial_diagonal() {
        let b = cantilever();
        let ke = b.ke_flat(&[0.0; 6]);
        let eal = b.e() * b.a() / b.length();
        // Horizontal element: ke[0,0] = EA/L, ke[3,3] = EA/L
        assert!((ke[0]  - eal).abs() < 1e-3);
        assert!((ke[21] - eal).abs() < 1e-3);
    }

    #[test]
    fn ke_flat_bending_diagonal() {
        let b = cantilever();
        let ke = b.ke_flat(&[0.0; 6]);
        let l  = b.length();
        let ei = b.e() * b.iz();
        let b1 = 12.0 * ei / (l * l * l);
        let b3 =  4.0 * ei / l;
        // Horizontal element: v-DOFs at indices 1 and 4, θ-DOFs at 2 and 5
        assert!((ke[7]  - b1).abs() < 1e-3, "ke[1,1]={} expected {b1}", ke[7]);
        assert!((ke[14] - b3).abs() < 1e-3, "ke[2,2]={} expected {b3}", ke[14]);
    }

    // ---- f_int ----

    #[test]
    fn f_int_zero_displacement_is_zero() {
        let f = cantilever().f_int(&[0.0; 6]);
        assert!(f.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn f_int_axial_elongation() {
        let b = cantilever();
        let delta = 1e-3;
        // Pure axial elongation: u2 = delta
        let u = [0.0, 0.0, 0.0, delta, 0.0, 0.0];
        let f = b.f_int(&u);
        let eal = b.e() * b.a() / b.length();
        assert!((f[0] - -eal * delta).abs() < 1.0);
        assert!((f[3] -  eal * delta).abs() < 1.0);
    }

    // ---- energy ----

    #[test]
    fn energy_zero_displacement_is_zero() {
        assert!(cantilever().energy_f64(&[0.0; 6]).abs() < 1e-20);
    }

    #[test]
    fn energy_positive_for_nonzero_displacement() {
        let u = [0.0, 0.0, 0.0, 1e-3, 0.0, 0.0];
        assert!(cantilever().energy_f64(&u) > 0.0);
    }

    #[test]
    fn energy_equals_half_u_f_int() {
        // W = ½ uᵀ f_int for linear elastic
        let b = cantilever();
        let u = [0.0, 0.0, 0.0, 1e-3, 1e-4, 5e-4];
        let w    = b.energy_f64(&u);
        let f    = b.f_int(&u);
        let w_kf = 0.5 * u.iter().zip(f.iter()).map(|(ui, fi)| ui * fi).sum::<f64>();
        assert!((w - w_kf).abs() / w_kf.abs() < 1e-10, "W={w:.6e} W_kf={w_kf:.6e}");
    }

    // ---- DOF map ----

    #[test]
    fn dof_map_six_entries() {
        assert_eq!(cantilever().dof_map().n_local(), 6);
    }

    #[test]
    fn dof_map_node0_node1_frame() {
        use fem_core::{GlobalDof, LocalDof};
        let b = cantilever(); // NodeId(0), NodeId(1), ndf=3
        let dm = b.dof_map();
        assert_eq!(dm[LocalDof(0)], GlobalDof(0)); // node0 ux
        assert_eq!(dm[LocalDof(1)], GlobalDof(1)); // node0 uy
        assert_eq!(dm[LocalDof(2)], GlobalDof(2)); // node0 θ
        assert_eq!(dm[LocalDof(3)], GlobalDof(3)); // node1 ux
        assert_eq!(dm[LocalDof(4)], GlobalDof(4)); // node1 uy
        assert_eq!(dm[LocalDof(5)], GlobalDof(5)); // node1 θ
    }

    // ---- Assembleable ----

    #[test]
    fn n_params_is_three() {
        assert_eq!(cantilever().n_params(), 3);
    }

    #[test]
    fn partial_residual_e_length_correct() {
        let b = cantilever();
        // Pure axial elongation
        let u = [0.0, 0.0, 0.0, 1e-3, 0.0, 0.0];
        let dr = b.partial_residual_wrt_param(&u, params::E).unwrap();
        assert_eq!(dr.len(), 6);
        // For axial strain ε = 1e-3/2:
        // ∂f[0]/∂E = -(A/L) * ε = -(0.01/2) * 5e-4 = -1.25e-6
        let a = b.a(); let l = b.length();
        let eps = 1e-3 / l;
        let expected_axial = -a * eps;
        assert!(
            (dr[0] - expected_axial).abs() < 1e-15,
            "dr[0]={:.6e} expected {expected_axial:.6e}", dr[0]
        );
    }

    #[test]
    fn clone_box_type_name() {
        let b = cantilever();
        let b2 = b.clone_box();
        assert_eq!(b2.type_name(), "ElasticBeam2d");
    }
}
