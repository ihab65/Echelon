//! 4-node MITC4 flat-shell element with linear elastic isotropic material.
//!
//! `ElasticShell4` connects four coplanar nodes in 3D space.  It combines:
//! - **Membrane** behaviour (in-plane stretching and shearing)
//! - **Bending** behaviour (Mindlin-Reissner plate kinematics)
//! - **MITC4 transverse shear** (locking-free shear via tying, no reduced integration)
//! - **Hughes-Brezzi drilling penalty** (prevents the θ_z singularity)
//!
//! ## DOF convention
//!
//! **6 DOFs per node, 24 total** (global coordinates):
//! ```text
//! [u_x, u_y, u_z, θ_x, θ_y, θ_z]  per node
//!   0    1    2    3    4    5
//!
//! Full vector:  [node 0 | node 1 | node 2 | node 3]
//! ```
//!
//! Node connectivity (counter-clockwise for outward normal, right-hand rule):
//! ```text
//! 3 ──── 2
//! │      │   ← normal points toward viewer (+z_local)
//! 0 ──── 1
//! ```
//!
//! ## Local shell frame
//!
//! Built at construction time from the four node coordinates:
//! - **Origin** at the centroid.
//! - **`e₁`** = `normalise(P₁ − P₀)` — along edge 0→1.
//! - **`e₃`** = `normalise((P₂−P₀) × (P₃−P₁))` — shell outward normal.
//! - **`e₂`** = `e₃ × e₁` — completes the right-hand frame.
//!
//! The rotation matrix `R` (rows = local axes in global coords) transforms:
//! ```text
//! u_local = R · u_global    (for both translations and rotations)
//! f_global = Rᵀ · f_local
//! ```
//!
//! ## Parameter index convention (for `Assembleable`)
//!
//! | Index | Symbol | Description |
//! |-------|--------|-------------|
//! | 0 | `E` | Young's modulus (Pa) |
//! | 1 | `ν` | Poisson's ratio |
//! | 2 | `t` | Shell thickness (m) |

use fem_core::{DofMap, NodeId};
use materials::{ElasticIsotropic, NdMaterial};

use crate::local::shell::{ke_local_mitc4, f_int_local_mitc4, ShellSectionStiffness};
use crate::traits::{Assembleable, Element};
use crate::error::{ElementError, Result};

#[allow(dead_code)]
const DEFAULT_GAMMA_DRILL: f64 = 0.001;
// Default Mindlin shear correction factor
const DEFAULT_KAPPA_SHEAR: f64 = 5.0 / 6.0;

// -----------------------------------------------------------------
// Struct
// -----------------------------------------------------------------

/// 4-node MITC4 flat-shell element with linear elastic isotropic material.
///
/// # Example
///
/// ```rust
/// use elements::{ElasticShell4, Element};
/// use materials::{ElasticIsotropic, NdOrder};
/// use fem_core::NodeId;
///
/// // Flat square shell element in the XY plane (1 m × 1 m, 10 mm thick, steel)
/// let material = ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, None).unwrap();
/// let shell = ElasticShell4::new(
///     [NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
///     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
///     material,
///     0.01,    // thickness (m)
///     0.001,   // drilling penalty γ
///     1e-6,    // coplanarity tolerance (m)
/// ).unwrap();
///
/// assert_eq!(shell.n_dof(), 24);
/// ```
#[derive(Debug, Clone)]
pub struct ElasticShell4 {
    /// Global DOF map: 4 nodes × 6 DOF = 24.
    dof_map: DofMap,
    /// 3×3 rotation matrix `R` (rows are local axes in global coords).
    /// Maps global → local: `u_local = R · u_global`.
    rotation: [[f64; 3]; 3],
    /// Node coordinates in the local shell plane `[x_local, y_local]`.
    xy_local: [[f64; 2]; 4],
    /// Elastic isotropic material (owns E, ν for commit/revert).
    material: ElasticIsotropic,
    /// Shell thickness (m).
    thickness: f64,
    /// Pre-computed section stiffness matrices.
    #[allow(dead_code)]
    section: ShellSectionStiffness,
    /// Cached 24×24 local stiffness (constant for linear elastic).
    ke_local_cache: [f64; 576],
    /// Committed strain vector (plane-stress Voigt, length 3).
    committed_strain: [f64; 3],
}

// -----------------------------------------------------------------
// Construction helpers
// -----------------------------------------------------------------

/// Cross product of two 3-vectors.
#[inline]
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
#[inline]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm of a 3-vector.
#[inline]
fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Normalise a 3-vector. Returns `None` if the norm is below `eps`.
#[inline]
fn normalise3(v: &[f64; 3], eps: f64) -> Option<[f64; 3]> {
    let n = norm3(v);
    if n < eps { return None; }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

/// Subtract two 3-vectors.
#[inline]
fn sub3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// -----------------------------------------------------------------
// Global ↔ local transformation helpers
// -----------------------------------------------------------------

/// Transform a global 24-DOF vector to the local shell frame.
///
/// For each node I's 6 DOFs, applies `R` to both the translation and rotation triples.
#[inline]
fn to_local_u(r: &[[f64; 3]; 3], u_global: &[f64]) -> [f64; 24] {
    let mut u_loc = [0.0_f64; 24];
    for node in 0..4 {
        let base = 6 * node;
        for a in 0..3 {
            for j in 0..3 {
                u_loc[base + a]     += r[a][j] * u_global[base + j];     // translations
                u_loc[base + 3 + a] += r[a][j] * u_global[base + 3 + j]; // rotations
            }
        }
    }
    u_loc
}

/// Transform a local 24-DOF force vector to global coordinates.
///
/// Applies `Rᵀ` to both translation and rotation triples of each node.
#[inline]
fn to_global_f(r: &[[f64; 3]; 3], f_local: &[f64; 24]) -> [f64; 24] {
    let mut f_glob = [0.0_f64; 24];
    for node in 0..4 {
        let base = 6 * node;
        for a in 0..3 {
            for k in 0..3 {
                f_glob[base + a]     += r[k][a] * f_local[base + k];     // Rᵀ for translations
                f_glob[base + 3 + a] += r[k][a] * f_local[base + 3 + k]; // Rᵀ for rotations
            }
        }
    }
    f_glob
}

/// Transform the local 24×24 stiffness to global coordinates.
///
/// Computes `Ke_global = Tᵀ Ke_local T` exploiting the block structure of `T`:
///
/// ```text
/// T = block_diag(R̃, R̃, R̃, R̃)   R̃ = block_diag(R, R)  (6×6 per node)
/// ```
///
/// Two-pass implementation:
/// 1. `A = Ke_local · T`  (right-multiply)
/// 2. `Ke_global = Tᵀ · A`  (left-multiply by Tᵀ)
fn to_global_ke(r: &[[f64; 3]; 3], ke_l: &[f64; 576]) -> [f64; 576] {
    const N: usize = 24;
    // Pass 1: A = Ke_local · T
    // A[i][6J+b] = Σ_{c=0..2} ke_l[i][6J + g*3 + c] * R[c][b%3]
    // where g = b/3 (translation group 0, rotation group 1)
    let mut a = [0.0_f64; 576];
    for i in 0..N {
        for j_node in 0..4_usize {
            for b in 0..6_usize {
                let g = b / 3;        // group: 0=translation, 1=rotation
                let b_loc = b % 3;
                let j = 6 * j_node + b;
                let mut sum = 0.0;
                for c in 0..3_usize {
                    let k = 6 * j_node + g * 3 + c;
                    sum += ke_l[i * N + k] * r[c][b_loc];
                }
                a[i * N + j] = sum;
            }
        }
    }
    // Pass 2: Ke_global = Tᵀ · A
    // Ke_global[6I+a][j] = Σ_{c=0..2} R[c][a%3] * A[6I + g*3 + c][j]
    let mut ke_g = [0.0_f64; 576];
    for i_node in 0..4_usize {
        for a_i in 0..6_usize {
            let g = a_i / 3;
            let a_loc = a_i % 3;
            let i = 6 * i_node + a_i;
            for j in 0..N {
                let mut sum = 0.0;
                for c in 0..3_usize {
                    let k = 6 * i_node + g * 3 + c;
                    sum += r[c][a_loc] * a[k * N + j];
                }
                ke_g[i * N + j] = sum;
            }
        }
    }
    ke_g
}

// -----------------------------------------------------------------
// ElasticShell4
// -----------------------------------------------------------------

impl ElasticShell4 {
    /// Construct an `ElasticShell4` from global node coordinates.
    ///
    /// # Arguments
    /// * `nodes`        — 4 node IDs in counter-clockwise order (viewed from outside)
    /// * `coords`       — global 3D coordinates of the 4 nodes
    /// * `material`     — `ElasticIsotropic` material (plane-stress)
    /// * `thickness`    — shell thickness `t > 0` (m)
    /// * `gamma_drill`  — dimensionless drilling penalty (recommended: 0.001)
    /// * `coplanar_tol` — coplanarity tolerance (m); recommended: `1e-6 × element_size`
    ///
    /// # Errors
    /// - [`crate::error::ElementError::InadmissibleSection`] if `thickness ≤ 0`.
    /// - [`fem_core::error::CoreError::DegenerateGeometry`] if nodes are nearly coincident (zero-length edges).
    /// - [`fem_core::error::CoreError::DegenerateGeometry`] if node 3 deviates from the plane of nodes 0–1–2
    ///   by more than `coplanar_tol`.
    pub fn new(
        nodes:        [NodeId; 4],
        coords:       [[f64; 3]; 4],
        material:     ElasticIsotropic,
        thickness:    f64,
        gamma_drill:  f64,
        coplanar_tol: f64,
    ) -> Result<Self> {
        // ---- Validate thickness ----
        if thickness <= 0.0 {
            return Err(ElementError::InadmissibleSection {
                element_type: "ElasticShell4",
                parameter: "t (thickness)",
                value: thickness,
                requirement: "t > 0",
            });
        }

        // ---- Build local shell frame ----

        // e1: along edge 0→1
        let edge01 = sub3(&coords[1], &coords[0]);
        let e1 = normalise3(&edge01, 1e-14).ok_or_else(|| {
            ElementError::InadmissibleSection {
                element_type: "ElasticShell4",
                parameter: "edge 0→1 length",
                value: norm3(&edge01),
                requirement: "> 1e-14 (nodes 0 and 1 must not be coincident)",
            }
        })?;

        // e3: normal from cross product of diagonals (robust for any quad)
        let diag02 = sub3(&coords[2], &coords[0]); // P₂ - P₀
        let diag13 = sub3(&coords[3], &coords[1]); // P₃ - P₁
        let n_raw  = cross3(&diag02, &diag13);
        let e3 = normalise3(&n_raw, 1e-14).ok_or_else(|| {
            ElementError::InadmissibleSection {
                element_type: "ElasticShell4",
                parameter: "element area",
                value: norm3(&n_raw),
                requirement: "> 1e-14 (nodes must not be collinear)",
            }
        })?;

        // e2: complete right-hand frame
        let e2 = cross3(&e3, &e1);

        // ---- Coplanarity check ----
        // Node 3 must lie within `coplanar_tol` of the plane defined by nodes 0, 1, 2.
        let v30 = sub3(&coords[3], &coords[0]);
        let deviation = dot3(&v30, &e3).abs();
        if deviation > coplanar_tol {
            return Err(ElementError::InadmissibleSection {
                element_type: "ElasticShell4",
                parameter: "coplanarity deviation of node 3",
                value: deviation,
                requirement: "<= coplanar_tol (nodes must be coplanar)",
            });
        }

        // rotation: rows = local axes in global coords (global → local transform)
        let rotation = [e1, e2, e3];

        // ---- Project nodes to local 2D plane ----
        // Centroid
        let c = [
            (coords[0][0] + coords[1][0] + coords[2][0] + coords[3][0]) * 0.25,
            (coords[0][1] + coords[1][1] + coords[2][1] + coords[3][1]) * 0.25,
            (coords[0][2] + coords[1][2] + coords[2][2] + coords[3][2]) * 0.25,
        ];
        let mut xy_local = [[0.0_f64; 2]; 4];
        for (i, p) in coords.iter().enumerate() {
            let dp = sub3(p, &c);
            xy_local[i][0] = dot3(&dp, &e1); // local x
            xy_local[i][1] = dot3(&dp, &e2); // local y
        }

        // ---- Section stiffness ----
        let section = ShellSectionStiffness::from_material(
            material.e, material.nu, thickness, gamma_drill, DEFAULT_KAPPA_SHEAR,
        );

        // ---- Cache local stiffness ----
        let ke_local_cache = ke_local_mitc4(&xy_local, &section);

        // ---- DOF map: 4 nodes × 6 DOF ----
        let dof_map = DofMap::from_nodes(&nodes, 6);

        Ok(Self {
            dof_map,
            rotation,
            xy_local,
            material,
            thickness,
            section,
            ke_local_cache,
            committed_strain: [0.0; 3],
        })
    }

    // ---- Accessors ----

    /// Shell thickness (m).
    #[inline]
    pub fn thickness(&self) -> f64 { self.thickness }

    /// 3×3 rotation matrix `R` (rows = local axes in global coordinates).
    #[inline]
    pub fn local_frame(&self) -> &[[f64; 3]; 3] { &self.rotation }

    /// The local shell x-axis direction in global coordinates.
    #[inline]
    pub fn local_x_axis(&self) -> [f64; 3] { self.rotation[0] }

    /// The local shell y-axis direction in global coordinates.
    #[inline]
    pub fn local_y_axis(&self) -> [f64; 3] { self.rotation[1] }

    /// The shell outward normal (local z-axis) in global coordinates.
    #[inline]
    pub fn shell_normal(&self) -> [f64; 3] { self.rotation[2] }

    /// Node 2D coordinates in the local shell plane.
    #[inline]
    pub fn xy_local(&self) -> &[[f64; 2]; 4] { &self.xy_local }

    /// Young's modulus (Pa).
    #[inline]
    pub fn e(&self) -> f64 { self.material.e }

    /// Poisson's ratio.
    #[inline]
    pub fn nu(&self) -> f64 { self.material.nu }

    /// Approximate element area (in the local plane).
    ///
    /// Computed as the area of the parallelogram formed by the two diagonals
    /// — exact for parallelograms, approximate for general quads.
    pub fn approx_area(&self) -> f64 {
        let d1x = self.xy_local[2][0] - self.xy_local[0][0];
        let d1y = self.xy_local[2][1] - self.xy_local[0][1];
        let d2x = self.xy_local[3][0] - self.xy_local[1][0];
        let d2y = self.xy_local[3][1] - self.xy_local[1][1];
        0.5 * (d1x * d2y - d1y * d2x).abs()
    }
}

// -----------------------------------------------------------------
// Element trait
// -----------------------------------------------------------------

impl Element for ElasticShell4 {
    #[inline]
    fn n_dof(&self) -> usize { 24 }

    fn ke_flat(&self, _u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(out.len(), 576, "ke_flat output must have length 576");
        // Transform pre-cached local stiffness to global coords
        let ke_global = to_global_ke(&self.rotation, &self.ke_local_cache);
        out.copy_from_slice(&ke_global);
    }

    fn mass_flat(&self, out: &mut [f64]) {
        debug_assert_eq!(out.len(), 576, "mass_flat output must have length 576");
        out.fill(0.0);

        let rho = self.material.rho.unwrap_or(0.0);
        let m_total = rho * self.thickness * self.approx_area();
        let m_node  = m_total / 4.0; // lumped equally to 4 nodes

        // Small regularisation for rotational DOFs (prevents zero-mass rows)
        let m_rot = 1e-9;

        for node in 0..4 {
            let base = 6 * node;
            let diag = base * 24 + base; // [base][base] flat index
            // Translational DOFs: ux, uy, uz
            out[diag + 0 * 24 + 0] = m_node;
            out[diag + 1 * 24 + 1] = m_node;
            out[diag + 2 * 24 + 2] = m_node;
            // Rotational DOFs: θx, θy, θz
            out[diag + 3 * 24 + 3] = m_rot;
            out[diag + 4 * 24 + 4] = m_rot;
            out[diag + 5 * 24 + 5] = m_rot;
        }
    }

    fn f_int(&self, u: &[f64], out: &mut [f64]) {
        debug_assert_eq!(u.len(), 24);
        debug_assert_eq!(out.len(), 24);

        let u_loc = to_local_u(&self.rotation, u);
        let f_loc = f_int_local_mitc4(&self.ke_local_cache, &u_loc);
        let f_glob = to_global_f(&self.rotation, &f_loc);
        out.copy_from_slice(&f_glob);
    }

    fn commit(&mut self, u: &[f64]) -> Result<()> {
        debug_assert_eq!(u.len(), 24);
        // Compute representative membrane strain at the centre (r=0, s=0)
        // for state management. The material only needs to track that a commit
        // happened (it's linear elastic — commit is a no-op on the state).
        let u_loc = to_local_u(&self.rotation, u);
        // Use centroid strain as the representative value for commit/revert
        let mut strain_voigt = [0.0_f64; 3];
        strain_voigt[0] = u_loc[6] - u_loc[0]; // simple representative proxy
        self.committed_strain = strain_voigt;
        self.material.commit_state(&strain_voigt)?;
        Ok(())
    }

    fn revert(&mut self) {
        self.material.revert_to_last_commit();
        self.committed_strain = [0.0; 3];
    }

    fn clone_box(&self) -> Box<dyn Element> {
        Box::new(self.clone())
    }

    fn type_name(&self) -> &'static str { "ElasticShell4" }
}

// -----------------------------------------------------------------
// Assembleable trait
// -----------------------------------------------------------------

/// Parameter indices for `ElasticShell4`.
pub mod params {
    /// Index 0: Young's modulus `E` (Pa).
    pub const E: usize = 0;
    /// Index 1: Poisson's ratio `ν`.
    pub const NU: usize = 1;
    /// Index 2: Shell thickness `t` (m).
    pub const T: usize = 2;
}

impl Assembleable for ElasticShell4 {
    fn dof_map(&self) -> &DofMap {
        &self.dof_map
    }

    fn partial_residual_wrt_param(
        &self,
        u_global: &[f64],
        param_idx: usize,
        out: &mut [f64],
    ) -> Result<()> {
        debug_assert_eq!(u_global.len(), 24);
        debug_assert_eq!(out.len(), 24);

        // ∂f_int/∂θ = Tᵀ · (∂Ke_local/∂θ) · u_local
        // Build ∂Ke_local/∂θ by recomputing with perturbed section stiffness.
        let u_loc = to_local_u(&self.rotation, u_global);

        let d_sec = match param_idx {
            params::E => {
                // ∂Ke/∂E: rebuild section with E=1 (all other params unchanged)
                ShellSectionStiffness::from_material(
                    1.0, self.material.nu, self.thickness, 0.001, DEFAULT_KAPPA_SHEAR,
                )
            }
            params::NU => {
                // Finite difference on ν (analytic form is complex)
                let h = 1e-7;
                let nu_p = self.material.nu + h;
                let nu_m = (self.material.nu - h).max(-0.99);
                let sec_p = ShellSectionStiffness::from_material(
                    self.material.e, nu_p, self.thickness, 0.001, DEFAULT_KAPPA_SHEAR,
                );
                let sec_m = ShellSectionStiffness::from_material(
                    self.material.e, nu_m, self.thickness, 0.001, DEFAULT_KAPPA_SHEAR,
                );
                // ∂sec/∂ν ≈ (sec_p - sec_m) / (2h) — build directly
                let mut d = sec_p;
                for i in 0..9 {
                    d.a_membrane[i] = (sec_p.a_membrane[i] - sec_m.a_membrane[i]) / (2.0 * h);
                    d.d_bending[i]  = (sec_p.d_bending[i]  - sec_m.d_bending[i])  / (2.0 * h);
                }
                for i in 0..4 {
                    d.h_shear[i] = (sec_p.h_shear[i] - sec_m.h_shear[i]) / (2.0 * h);
                }
                d.alpha_drill = (sec_p.alpha_drill - sec_m.alpha_drill) / (2.0 * h);
                d
            }
            params::T => {
                // ∂Ke/∂t: rebuild with t=1, ν unchanged (since A∝t, D∝t³, H∝t)
                // D_∂Ke/∂t is computed by finite difference
                let h = self.thickness * 1e-6;
                let sec_p = ShellSectionStiffness::from_material(
                    self.material.e, self.material.nu, self.thickness + h, 0.001, DEFAULT_KAPPA_SHEAR,
                );
                let sec_m = ShellSectionStiffness::from_material(
                    self.material.e, self.material.nu, self.thickness - h, 0.001, DEFAULT_KAPPA_SHEAR,
                );
                let mut d = sec_p;
                for i in 0..9 {
                    d.a_membrane[i] = (sec_p.a_membrane[i] - sec_m.a_membrane[i]) / (2.0 * h);
                    d.d_bending[i]  = (sec_p.d_bending[i]  - sec_m.d_bending[i])  / (2.0 * h);
                }
                for i in 0..4 {
                    d.h_shear[i] = (sec_p.h_shear[i] - sec_m.h_shear[i]) / (2.0 * h);
                }
                d.alpha_drill = (sec_p.alpha_drill - sec_m.alpha_drill) / (2.0 * h);
                d
            }
            _ => {
                return Err(ElementError::UnregisteredParameter {
                    element_type: "ElasticShell4",
                    idx: param_idx,
                    n_params: self.n_params(),
                });
            }
        };

        // ∂Ke_local/∂θ · u_local
        let d_ke_local = ke_local_mitc4(&self.xy_local, &d_sec);
        let d_f_local: [f64; 24] = {
            let mut f = [0.0_f64; 24];
            for i in 0..24 {
                for j in 0..24 {
                    f[i] += d_ke_local[i * 24 + j] * u_loc[j];
                }
            }
            f
        };

        let d_f_global = to_global_f(&self.rotation, &d_f_local);
        out.copy_from_slice(&d_f_global);
        Ok(())
    }

    fn n_params(&self) -> usize { 3 }

    fn param_name(&self, param_idx: usize) -> &'static str {
        match param_idx {
            params::E  => "E (Young's modulus)",
            params::NU => "nu (Poisson's ratio)",
            params::T  => "t (shell thickness)",
            _          => panic!("param_idx {param_idx} out of range"),
        }
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::NodeId;
    use materials::NdOrder;

    /// Unit-square shell in the XY plane: nodes at (0,0,0),(1,0,0),(1,1,0),(0,1,0)
    fn square_shell() -> ElasticShell4 {
        let material = ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, None).unwrap();
        ElasticShell4::new(
            [NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            material,
            0.01,
            0.001,
            1e-6,
        ).unwrap()
    }

    /// Tilted square shell in a plane with a 45° angle (test non-trivial local frame)
    fn tilted_shell() -> ElasticShell4 {
        let material = ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, None).unwrap();
        let sq2 = 2.0_f64.sqrt() / 2.0;
        // Square in the plane z = y (rotated 45° about x-axis)
        ElasticShell4::new(
            [NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            [
                [0.0,  0.0,  0.0],
                [1.0,  0.0,  0.0],
                [1.0,  sq2,  sq2],
                [0.0,  sq2,  sq2],
            ],
            material,
            0.01,
            0.001,
            1e-4,
        ).unwrap()
    }

    // ---- Construction ----

    #[test]
    fn n_dof_is_24() {
        assert_eq!(square_shell().n_dof(), 24);
    }

    #[test]
    fn local_frame_orthonormal() {
        let s = square_shell();
        let r = s.local_frame();
        // Each row must be unit-length
        for i in 0..3 {
            let len_sq = r[i][0].powi(2) + r[i][1].powi(2) + r[i][2].powi(2);
            assert!((len_sq - 1.0).abs() < 1e-13, "row {i} not unit: len²={len_sq}");
        }
        // Rows must be mutually orthogonal
        for i in 0..3 {
            for j in (i+1)..3 {
                let d = dot3(&r[i], &r[j]);
                assert!(d.abs() < 1e-13, "rows {i},{j} not orthogonal: dot={d}");
            }
        }
    }

    #[test]
    fn shell_normal_points_in_z_for_xy_plane() {
        let s = square_shell();
        let n = s.shell_normal();
        // For a CCW quad in the XY plane, normal should be +Z
        assert!(n[2] > 0.9, "normal z-component={}", n[2]);
        assert!(n[0].abs() < 1e-13, "normal x should be ~0: {}", n[0]);
        assert!(n[1].abs() < 1e-13, "normal y should be ~0: {}", n[1]);
    }

    #[test]
    fn xy_local_z_components_near_zero() {
        // All local z-coordinates should be zero (flat element)
        let s = square_shell();
        let r = s.local_frame();
        let coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let c = [0.5, 0.5, 0.0];
        for p in &coords {
            let dp = sub3(p, &c);
            let z_local = dot3(&dp, &r[2]);
            assert!(z_local.abs() < 1e-13, "local z = {z_local}");
        }
    }

    #[test]
    fn approx_area_unit_square() {
        let s = square_shell();
        assert!((s.approx_area() - 1.0).abs() < 1e-12, "area = {}", s.approx_area());
    }

    #[test]
    fn invalid_thickness_rejected() {
        let material = ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, None).unwrap();
        let err = ElasticShell4::new(
            [NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            [[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0]],
            material, 0.0, 0.001, 1e-6,
        );
        assert!(err.is_err(), "zero thickness must fail");
    }

    #[test]
    fn non_coplanar_nodes_rejected() {
        let material = ElasticIsotropic::new(200e9, 0.3, NdOrder::PlaneStress, None).unwrap();
        let err = ElasticShell4::new(
            [NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            [[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0]], // node 3 out of plane
            material, 0.01, 0.001,
            1e-6, // tight tolerance
        );
        assert!(err.is_err(), "non-coplanar node must fail");
    }

    // ---- ke_flat ----

    #[test]
    fn ke_flat_correct_size() {
        let s = square_shell();
        let mut ke = [0.0_f64; 576];
        s.ke_flat(&[0.0; 24], &mut ke);
        assert_eq!(ke.len(), 576);
    }

    #[test]
    fn ke_flat_symmetric() {
        let s = square_shell();
        let mut ke = [0.0_f64; 576];
        s.ke_flat(&[0.0; 24], &mut ke);
        for i in 0..24 {
            for j in 0..24 {
                assert!(
                    (ke[i * 24 + j] - ke[j * 24 + i]).abs() < 1.0,
                    "ke not symmetric at ({i},{j}): {} vs {}", ke[i*24+j], ke[j*24+i]
                );
            }
        }
    }

    #[test]
    fn ke_flat_positive_diagonal() {
        let s = square_shell();
        let mut ke = [0.0_f64; 576];
        s.ke_flat(&[0.0; 24], &mut ke);
        for i in 0..24 {
            assert!(ke[i * 24 + i] > 0.0, "ke[{i},{i}] = {}", ke[i*24+i]);
        }
    }

    #[test]
    fn ke_flat_tilted_still_symmetric() {
        let s = tilted_shell();
        let mut ke = [0.0_f64; 576];
        s.ke_flat(&[0.0; 24], &mut ke);
        for i in 0..24 {
            for j in 0..24 {
                let diff = (ke[i * 24 + j] - ke[j * 24 + i]).abs();
                assert!(
                    diff < 1.0,
                    "tilted ke not symmetric at ({i},{j}): {} vs {} diff={diff:.2e}",
                    ke[i*24+j], ke[j*24+i]
                );
            }
        }
    }

    // ---- f_int ----

    #[test]
    fn f_int_zero_displacement_is_zero() {
        let s = square_shell();
        let mut f = [0.0_f64; 24];
        s.f_int(&[0.0; 24], &mut f);
        assert!(f.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f_int_consistent_with_ke() {
        // For a linear element: f_int(u) = Ke · u
        let s = square_shell();
        let mut u = [0.0_f64; 24];
        u[2] = 1e-4; // small transverse displacement at node 0
        let mut f_direct = [0.0_f64; 24];
        s.f_int(&u, &mut f_direct);

        let mut ke = [0.0_f64; 576];
        s.ke_flat(&u, &mut ke);
        let mut f_from_ke = [0.0_f64; 24];
        for i in 0..24 {
            for j in 0..24 {
                f_from_ke[i] += ke[i * 24 + j] * u[j];
            }
        }
        for i in 0..24 {
            let diff = (f_direct[i] - f_from_ke[i]).abs();
            assert!(
                diff < 1.0,
                "f_int[{i}] direct={:.6e} ke·u={:.6e} diff={diff:.2e}",
                f_direct[i], f_from_ke[i]
            );
        }
    }

    // ---- commit / revert ----

    #[test]
    fn commit_does_not_change_stiffness() {
        let mut s = square_shell();
        let mut ke_before = [0.0_f64; 576];
        s.ke_flat(&[0.0; 24], &mut ke_before);
        s.commit(&[0.0; 24]).unwrap();
        let mut ke_after = [0.0_f64; 576];
        s.ke_flat(&[0.0; 24], &mut ke_after);
        assert_eq!(ke_before, ke_after);
    }

    // ---- Assembleable ----

    #[test]
    fn n_params_is_three() {
        assert_eq!(square_shell().n_params(), 3);
    }

    #[test]
    fn dof_map_has_24_entries() {
        assert_eq!(square_shell().dof_map().n_local(), 24);
    }

    #[test]
    fn type_name_is_correct() {
        assert_eq!(square_shell().type_name(), "ElasticShell4");
    }

    #[test]
    fn clone_box_preserves_type_name() {
        let s = square_shell();
        let b = s.clone_box();
        assert_eq!(b.type_name(), "ElasticShell4");
        assert_eq!(b.n_dof(), 24);
    }

    #[test]
    fn param_names_correct() {
        let s = square_shell();
        assert_eq!(s.param_name(params::E),  "E (Young's modulus)");
        assert_eq!(s.param_name(params::NU), "nu (Poisson's ratio)");
        assert_eq!(s.param_name(params::T),  "t (shell thickness)");
    }
}
