//! Isoparametric kinematics for the 4-node quadrilateral shell element.
//!
//! This module provides the pure-math building blocks for MITC4:
//!
//! | Function | Output | Description |
//! |----------|--------|-------------|
//! | [`shape_fns`] | `[f64; 4]` | Bilinear shape functions N_I(r,s) |
//! | [`shape_fn_derivs`] | `[[f64;4];2]` | dN/dr and dN/ds rows |
//! | [`jacobian`] | `([[f64;2];2], f64)` | 2×2 Jacobian and determinant |
//! | [`jacobian_inv`] | `[[f64;2];2]` | Inverse of the 2×2 Jacobian |
//! | [`physical_derivs`] | `[[f64;4];2]` | dN/dx, dN/dy in physical coords |
//! | [`b_membrane`] | `[f64; 72]` | 3×24 membrane B-matrix |
//! | [`b_bending`] | `[f64; 72]` | 3×24 bending B-matrix |
//! | [`tying_vector_r`] | `[f64; 24]` | 24-component covariant r-shear row at a tying point |
//! | [`tying_vector_s`] | `[f64; 24]` | 24-component covariant s-shear row at a tying point |
//! | [`b_shear_mitc4`] | `[f64; 48]` | 2×24 MITC4 physical shear B-matrix at a Gauss point |
//!
//! ## DOF ordering convention (24 total, 6 per node)
//!
//! ```text
//! Node I local DOFs:  [u_I, v_I, w_I, θ_xI, θ_yI, θ_zI]
//!                       0    1    2     3      4      5
//!
//! Full 24-DOF vector (node 0 first, node 3 last):
//!   [u0 v0 w0 θx0 θy0 θz0 | u1 v1 w1 θx1 θy1 θz1 | u2 ... | u3 ...]
//!     0  1  2   3   4   5     6  7  8   9  10  11   12 ... | 18 ...
//! ```
//!
//! ## Rotation convention
//!
//! - `θ_x`: right-hand rotation about the local x-axis (in-plane, causing
//!   bending in the y-z plane of the shell).
//! - `θ_y`: right-hand rotation about the local y-axis (in-plane, causing
//!   bending in the x-z plane of the shell).
//! - `θ_z` (drilling): rotation about the shell normal — stabilised by a
//!   small Hughes-Brezzi penalty; does not enter the B-matrices here.
//!
//! ## Curvature and shear conventions
//!
//! Curvatures (bending strains):
//! ```text
//! κxx =  ∂θ_y/∂x
//! κyy = -∂θ_x/∂y
//! κxy =  ∂θ_y/∂y − ∂θ_x/∂x
//! ```
//!
//! Physical transverse shear strains (Mindlin-Reissner):
//! ```text
//! γ_xz = θ_y + ∂w/∂x
//! γ_yz = −θ_x + ∂w/∂y
//! ```
//!
//! All output arrays are stack-allocated. No heap allocation occurs.

// -----------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------

/// Bilinear shape function values `N_I(r, s)` for all 4 nodes.
///
/// Node ordering (counter-clockwise, natural coords):
/// ```text
/// 3(-1,+1) ──── 2(+1,+1)
///    │                │
///    │                │
/// 0(-1,-1) ──── 1(+1,-1)
/// ```
///
/// ```text
/// N0 = ¼(1−r)(1−s)
/// N1 = ¼(1+r)(1−s)
/// N2 = ¼(1+r)(1+s)
/// N3 = ¼(1−r)(1+s)
/// ```
///
/// Returns `[N0, N1, N2, N3]`.
#[inline]
pub fn shape_fns(r: f64, s: f64) -> [f64; 4] {
    [
        0.25 * (1.0 - r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 + s),
        0.25 * (1.0 - r) * (1.0 + s),
    ]
}

/// Natural-coordinate derivatives of the bilinear shape functions.
///
/// Returns a `[[f64; 4]; 2]` where:
/// - Row 0: `[dN0/dr, dN1/dr, dN2/dr, dN3/dr]`
/// - Row 1: `[dN0/ds, dN1/ds, dN2/ds, dN3/ds]`
///
/// ```text
/// dN0/dr = −¼(1−s),  dN0/ds = −¼(1−r)
/// dN1/dr = +¼(1−s),  dN1/ds = −¼(1+r)
/// dN2/dr = +¼(1+s),  dN2/ds = +¼(1+r)
/// dN3/dr = −¼(1+s),  dN3/ds = +¼(1−r)
/// ```
#[inline]
pub fn shape_fn_derivs(r: f64, s: f64) -> [[f64; 4]; 2] {
    [
        // dN/dr row
        [
            -0.25 * (1.0 - s),
             0.25 * (1.0 - s),
             0.25 * (1.0 + s),
            -0.25 * (1.0 + s),
        ],
        // dN/ds row
        [
            -0.25 * (1.0 - r),
            -0.25 * (1.0 + r),
             0.25 * (1.0 + r),
             0.25 * (1.0 - r),
        ],
    ]
}

// -----------------------------------------------------------------
// Jacobian
// -----------------------------------------------------------------

/// 2×2 Jacobian of the isoparametric mapping at `(r, s)`.
///
/// `xy_local` is the 2D coordinates of the 4 nodes in the shell's local
/// plane (origin typically at the centroid).
///
/// ```text
/// J = [∂x/∂r  ∂y/∂r]   = Σ_I [ dN_I/dr · x_I   dN_I/dr · y_I ]
///     [∂x/∂s  ∂y/∂s]         [ dN_I/ds · x_I   dN_I/ds · y_I ]
/// ```
///
/// Returns `(J, det_J)`.
///
/// # Panics (debug)
/// Asserts `det_J > 0` — a non-positive determinant indicates a degenerate
/// or inverted element, which should have been caught at construction time.
#[inline]
pub fn jacobian(xy_local: &[[f64; 2]; 4], r: f64, s: f64) -> ([[f64; 2]; 2], f64) {
    let dn = shape_fn_derivs(r, s);
    let mut j = [[0.0_f64; 2]; 2];
    for i in 0..4 {
        j[0][0] += dn[0][i] * xy_local[i][0]; // ∂x/∂r
        j[0][1] += dn[0][i] * xy_local[i][1]; // ∂y/∂r
        j[1][0] += dn[1][i] * xy_local[i][0]; // ∂x/∂s
        j[1][1] += dn[1][i] * xy_local[i][1]; // ∂y/∂s
    }
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    debug_assert!(det > 0.0, "Jacobian determinant is non-positive: {det:.6e}");
    (j, det)
}

/// Inverse of the 2×2 Jacobian.
///
/// `det` must be the determinant from [`jacobian`].
#[inline]
pub fn jacobian_inv(j: &[[f64; 2]; 2], det: f64) -> [[f64; 2]; 2] {
    let inv_det = 1.0 / det;
    [
        [ j[1][1] * inv_det, -j[0][1] * inv_det],
        [-j[1][0] * inv_det,  j[0][0] * inv_det],
    ]
}

/// Physical derivatives `dN/dx` and `dN/dy` from the inverse Jacobian.
///
/// ```text
/// [dN_I/dx]   [J⁻¹₀₀  J⁻¹₀₁] [dN_I/dr]
/// [dN_I/dy] = [J⁻¹₁₀  J⁻¹₁₁] [dN_I/ds]
/// ```
///
/// Returns `[[dN/dx for each node], [dN/dy for each node]]`.
#[inline]
pub fn physical_derivs(j_inv: &[[f64; 2]; 2], dn_drs: &[[f64; 4]; 2]) -> [[f64; 4]; 2] {
    let mut out = [[0.0_f64; 4]; 2];
    for i in 0..4 {
        out[0][i] = j_inv[0][0] * dn_drs[0][i] + j_inv[0][1] * dn_drs[1][i]; // dN/dx
        out[1][i] = j_inv[1][0] * dn_drs[0][i] + j_inv[1][1] * dn_drs[1][i]; // dN/dy
    }
    out
}

// -----------------------------------------------------------------
// Membrane B-matrix (3 × 24)
// -----------------------------------------------------------------

/// 3×24 membrane B-matrix, flat row-major `[f64; 72]`.
///
/// Maps in-plane membrane DOFs `[u_I, v_I]` to strains `[εxx, εyy, γxy]`:
///
/// ```text
/// B_m = [ dN0/dx   0       dN1/dx   0       dN2/dx   0       dN3/dx   0      | zeros (w,θ,drill) ]
///        [   0    dN0/dy     0     dN1/dy     0     dN2/dy     0     dN3/dy  | zeros             ]
///        [ dN0/dy  dN0/dx  dN1/dy  dN1/dx  dN2/dy  dN2/dx  dN3/dy  dN3/dx  | zeros             ]
/// ```
///
/// Columns for `u_I` are at index `6I + 0`; for `v_I` at `6I + 1`.
/// All other columns (w, θ_x, θ_y, θ_z) are zero.
pub fn b_membrane(dn_dx: &[f64; 4], dn_dy: &[f64; 4]) -> [f64; 72] {
    let mut b = [0.0_f64; 72]; // 3 rows × 24 cols
    for i in 0..4 {
        let col_u = 6 * i;     // u_I column
        let col_v = 6 * i + 1; // v_I column
        // Row 0: εxx = Σ dN_I/dx · u_I
        b[0 * 24 + col_u] = dn_dx[i];
        // Row 1: εyy = Σ dN_I/dy · v_I
        b[1 * 24 + col_v] = dn_dy[i];
        // Row 2: γxy = Σ (dN_I/dy · u_I + dN_I/dx · v_I)
        b[2 * 24 + col_u] = dn_dy[i];
        b[2 * 24 + col_v] = dn_dx[i];
    }
    b
}

// -----------------------------------------------------------------
// Bending B-matrix (3 × 24)
// -----------------------------------------------------------------

/// 3×24 bending B-matrix, flat row-major `[f64; 72]`.
///
/// Maps rotation DOFs `[θ_xI, θ_yI]` to curvatures `[κxx, κyy, κxy]`
/// using the Mindlin-Reissner sign convention:
///
/// ```text
/// κxx =  ∂θ_y/∂x
/// κyy = -∂θ_x/∂y
/// κxy =  ∂θ_y/∂y − ∂θ_x/∂x
/// ```
///
/// ```text
/// B_b = [  0        dN0/dx    0        dN1/dx    0        dN2/dx    0        dN3/dx  | zeros ]
///        [-dN0/dy   0        -dN1/dy   0        -dN2/dy   0        -dN3/dy   0       | zeros ]
///        [-dN0/dx   dN0/dy  -dN1/dx   dN1/dy  -dN2/dx   dN2/dy  -dN3/dx   dN3/dy   | zeros ]
/// ```
///
/// Columns for `θ_xI` are at index `6I + 3`; for `θ_yI` at `6I + 4`.
/// All other columns (u, v, w, θ_z) are zero.
pub fn b_bending(dn_dx: &[f64; 4], dn_dy: &[f64; 4]) -> [f64; 72] {
    let mut b = [0.0_f64; 72]; // 3 rows × 24 cols
    for i in 0..4 {
        let col_tx = 6 * i + 3; // θ_xI column
        let col_ty = 6 * i + 4; // θ_yI column
        // Row 0: κxx = Σ dN_I/dx · θ_yI
        b[0 * 24 + col_ty] =  dn_dx[i];
        // Row 1: κyy = Σ -dN_I/dy · θ_xI
        b[1 * 24 + col_tx] = -dn_dy[i];
        // Row 2: κxy = Σ (dN_I/dy · θ_yI - dN_I/dx · θ_xI)
        b[2 * 24 + col_tx] = -dn_dx[i];
        b[2 * 24 + col_ty] =  dn_dy[i];
    }
    b
}

// -----------------------------------------------------------------
// MITC4 shear tying
// -----------------------------------------------------------------

/// 24-component covariant r-shear tying vector at a tying point.
///
/// At tying point `(r_t, s_t)`, the covariant r-shear strain is a linear
/// functional of the DOF vector `u`:
///
/// ```text
/// γ^cov_r = h_r · u_local
/// ```
///
/// where `h_r` is the returned 24-vector.
///
/// ```text
/// γ^cov_r = Σ_I (dN_I/dr · w_I)  +  J₀₀ · Σ_I (N_I · θ_yI)  −  J₀₁ · Σ_I (N_I · θ_xI)
/// ```
///
/// Non-zero entries only in columns `6I+2` (w), `6I+3` (θ_x), `6I+4` (θ_y).
///
/// # Arguments
/// * `n_vals` — shape function values `N_I` at the tying point
/// * `dn_dr`  — shape function r-derivatives `dN_I/dr` at the tying point
/// * `j00`    — Jacobian entry `J[0][0] = ∂x/∂r` at the tying point
/// * `j01`    — Jacobian entry `J[0][1] = ∂y/∂r` at the tying point
pub fn tying_vector_r(
    n_vals: &[f64; 4],
    dn_dr:  &[f64; 4],
    j00: f64,
    j01: f64,
) -> [f64; 24] {
    let mut h = [0.0_f64; 24];
    for i in 0..4 {
        h[6 * i + 2] = dn_dr[i];           // ∂N_I/∂r · w_I
        h[6 * i + 3] = -j01 * n_vals[i];   // -J₀₁ · N_I · θ_xI
        h[6 * i + 4] =  j00 * n_vals[i];   //  J₀₀ · N_I · θ_yI
    }
    h
}

/// 24-component covariant s-shear tying vector at a tying point.
///
/// ```text
/// γ^cov_s = Σ_I (dN_I/ds · w_I)  +  J₁₀ · Σ_I (N_I · θ_yI)  −  J₁₁ · Σ_I (N_I · θ_xI)
/// ```
///
/// Non-zero entries only in columns `6I+2` (w), `6I+3` (θ_x), `6I+4` (θ_y).
///
/// # Arguments
/// * `n_vals` — shape function values `N_I` at the tying point
/// * `dn_ds`  — shape function s-derivatives `dN_I/ds` at the tying point
/// * `j10`    — Jacobian entry `J[1][0] = ∂x/∂s`
/// * `j11`    — Jacobian entry `J[1][1] = ∂y/∂s`
pub fn tying_vector_s(
    n_vals: &[f64; 4],
    dn_ds:  &[f64; 4],
    j10: f64,
    j11: f64,
) -> [f64; 24] {
    let mut h = [0.0_f64; 24];
    for i in 0..4 {
        h[6 * i + 2] = dn_ds[i];           // ∂N_I/∂s · w_I
        h[6 * i + 3] = -j11 * n_vals[i];   // -J₁₁ · N_I · θ_xI
        h[6 * i + 4] =  j10 * n_vals[i];   //  J₁₀ · N_I · θ_yI
    }
    h
}

/// 2×24 MITC4 physical transverse shear B-matrix at a Gauss point, flat row-major `[f64; 48]`.
///
/// Interpolates the covariant shear strains from the four tying vectors and
/// transforms to physical shear strains `[γ_xz, γ_yz]` using the inverse Jacobian:
///
/// ```text
/// γ̃_rz(r,s) = ½(1−s)·h_A·u + ½(1+s)·h_C·u
/// γ̃_sz(r,s) = ½(1+r)·h_B·u + ½(1−r)·h_D·u
///
/// [γ_xz]   [J⁻¹₀₀  J⁻¹₀₁] [γ̃_rz]
/// [γ_yz] = [J⁻¹₁₀  J⁻¹₁₁] [γ̃_sz]
/// ```
///
/// So `B_shear = J⁻¹ · [h_tying_r; h_tying_s]` where tying vectors are
/// weighted by the MITC4 interpolation functions.
///
/// # Arguments
/// * `h_a`, `h_b`, `h_c`, `h_d` — tying vectors at points A, B, C, D
/// * `r`, `s` — natural coordinates of the Gauss point
/// * `j_inv` — inverse Jacobian at the Gauss point
pub fn b_shear_mitc4(
    h_a: &[f64; 24],
    h_b: &[f64; 24],
    h_c: &[f64; 24],
    h_d: &[f64; 24],
    r: f64,
    s: f64,
    j_inv: &[[f64; 2]; 2],
) -> [f64; 48] {
    let wa = 0.5 * (1.0 - s); // MITC4 weight for h_A (r-direction)
    let wc = 0.5 * (1.0 + s); // MITC4 weight for h_C (r-direction)
    let wb = 0.5 * (1.0 + r); // MITC4 weight for h_B (s-direction)
    let wd = 0.5 * (1.0 - r); // MITC4 weight for h_D (s-direction)

    let mut b = [0.0_f64; 48]; // 2 rows × 24 cols
    for j in 0..24 {
        // Interpolated covariant shear components at this Gauss point
        let gamma_r = wa * h_a[j] + wc * h_c[j]; // γ̃_rz row component
        let gamma_s = wb * h_b[j] + wd * h_d[j]; // γ̃_sz row component

        // Transform to physical: [γ_xz, γ_yz] = J⁻¹ · [γ̃_r, γ̃_s]
        b[0 * 24 + j] = j_inv[0][0] * gamma_r + j_inv[0][1] * gamma_s; // Row 0: γ_xz
        b[1 * 24 + j] = j_inv[1][0] * gamma_r + j_inv[1][1] * gamma_s; // Row 1: γ_yz
    }
    b
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Shape functions ----

    #[test]
    fn shape_fns_partition_of_unity() {
        // At any (r,s) the shape functions must sum to 1
        for &(r, s) in &[(0.0, 0.0), (0.5, -0.3), (-0.7, 0.9), (1.0, 1.0), (-1.0, -1.0)] {
            let n = shape_fns(r, s);
            let sum: f64 = n.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "partition sum={sum} at ({r},{s})");
        }
    }

    #[test]
    fn shape_fns_at_corner_nodes() {
        // N_I at node I's natural coords should be 1, 0 elsewhere
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        for (i, &(r, s)) in corners.iter().enumerate() {
            let n = shape_fns(r, s);
            for (j, &nj) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (nj - expected).abs() < 1e-15,
                    "N_{j}({r},{s}) = {nj}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn shape_fn_derivs_consistency() {
        // Verify ∂N_I/∂r is correct via finite difference
        let r = 0.3;
        let s = -0.4;
        let h = 1e-7;
        let dn = shape_fn_derivs(r, s);
        let n_p = shape_fns(r + h, s);
        let n_m = shape_fns(r - h, s);
        for i in 0..4 {
            let fd = (n_p[i] - n_m[i]) / (2.0 * h);
            assert!(
                (dn[0][i] - fd).abs() < 1e-8,
                "dN{i}/dr FD mismatch: analytic={:.10e} fd={fd:.10e}", dn[0][i]
            );
        }
    }

    #[test]
    fn shape_fn_derivs_sum_to_zero() {
        // Σ_I dN_I/dr = 0 and Σ_I dN_I/ds = 0 (partition of unity → constant derivative = 0)
        let dn = shape_fn_derivs(0.2, -0.5);
        let sum_r: f64 = dn[0].iter().sum();
        let sum_s: f64 = dn[1].iter().sum();
        assert!(sum_r.abs() < 1e-15, "sum dN/dr = {sum_r}");
        assert!(sum_s.abs() < 1e-15, "sum dN/ds = {sum_s}");
    }

    // ---- Jacobian ----

    #[test]
    fn jacobian_unit_square_is_identity() {
        // Square with side 2 centered at origin: nodes at (±1, ±1)
        let xy = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let (j, det) = jacobian(&xy, 0.0, 0.0);
        assert!((j[0][0] - 1.0).abs() < 1e-14, "J[0][0]={}",j[0][0]);
        assert!((j[1][1] - 1.0).abs() < 1e-14, "J[1][1]={}",j[1][1]);
        assert!(j[0][1].abs() < 1e-14);
        assert!(j[1][0].abs() < 1e-14);
        assert!((det - 1.0).abs() < 1e-14, "det={det}");
    }

    #[test]
    fn jacobian_scaled_square() {
        // Rectangle: nodes at (0,0),(a,0),(a,b),(0,b)
        let a = 3.0_f64;
        let b = 2.0_f64;
        let xy = [[0.0, 0.0], [a, 0.0], [a, b], [0.0, b]];
        let (j, det) = jacobian(&xy, 0.0, 0.0);
        // Jacobian should be [[a/2, 0], [0, b/2]]
        assert!((j[0][0] - a / 2.0).abs() < 1e-13, "J[0][0]={}", j[0][0]);
        assert!((j[1][1] - b / 2.0).abs() < 1e-13, "J[1][1]={}", j[1][1]);
        assert!((det - a * b / 4.0).abs() < 1e-13, "det={det}");
    }

    #[test]
    fn jacobian_inv_roundtrip() {
        let xy = [[0.0, 0.0], [2.0, 0.5], [2.5, 2.0], [0.3, 1.8]];
        let (j, det) = jacobian(&xy, 0.2, -0.3);
        let ji = jacobian_inv(&j, det);
        // J * J_inv should be identity
        let m00 = j[0][0] * ji[0][0] + j[0][1] * ji[1][0];
        let m01 = j[0][0] * ji[0][1] + j[0][1] * ji[1][1];
        let m10 = j[1][0] * ji[0][0] + j[1][1] * ji[1][0];
        let m11 = j[1][0] * ji[0][1] + j[1][1] * ji[1][1];
        assert!((m00 - 1.0).abs() < 1e-13, "m00={m00}");
        assert!(m01.abs() < 1e-13, "m01={m01}");
        assert!(m10.abs() < 1e-13, "m10={m10}");
        assert!((m11 - 1.0).abs() < 1e-13, "m11={m11}");
    }

    // ---- B membrane ----

    #[test]
    fn b_membrane_rigid_body_zero_strain() {
        // Uniform u translation: u_I = 1 for all I, v_I = 0
        // → εxx = Σ dN/dx · 1 = ∂(Σ N_I)/∂x = 0 (partition of unity)
        let dn_dx = [0.1, 0.2, 0.3, -0.6_f64]; // these sum to 0 — partition of unity derivative
        let dn_dy = [0.05, -0.1, 0.2, -0.15_f64];
        let b = b_membrane(&dn_dx, &dn_dy);

        // DOF vector: u=1 everywhere, v=0, others=0
        let mut u = [0.0_f64; 24];
        for i in 0..4 { u[6 * i] = 1.0; } // u_I = 1

        // Strain = B·u
        let mut strain = [0.0_f64; 3];
        for row in 0..3 {
            for col in 0..24 {
                strain[row] += b[row * 24 + col] * u[col];
            }
        }
        // εxx = Σ dN/dx_I · 1 = 0 (sum of dN/dx = 0)
        assert!(strain[0].abs() < 1e-14, "εxx={}", strain[0]);
    }

    #[test]
    fn b_membrane_shape_is_3x24() {
        let b = b_membrane(&[0.1; 4], &[0.2; 4]);
        assert_eq!(b.len(), 72);
    }

    // ---- B bending ----

    #[test]
    fn b_bending_shape_is_3x24() {
        let b = b_bending(&[0.1; 4], &[0.2; 4]);
        assert_eq!(b.len(), 72);
    }

    #[test]
    fn b_bending_rigid_rotation_zero_curvature() {
        // If θ_xI = c (constant) for all I, then κyy = -∂θ_x/∂y = -c Σ dN/dy = 0
        let dn_dx = [0.1, 0.2, 0.3, -0.6_f64]; // sum to 0
        let dn_dy = [0.05, -0.1, 0.2, -0.15_f64]; // sum to 0
        let b = b_bending(&dn_dx, &dn_dy);

        let c = 0.5_f64;
        let mut u = [0.0_f64; 24];
        for i in 0..4 { u[6 * i + 3] = c; } // θ_xI = c

        let mut kappa = [0.0_f64; 3];
        for row in 0..3 {
            for col in 0..24 {
                kappa[row] += b[row * 24 + col] * u[col];
            }
        }
        // All curvatures should be zero (rigid body rotation)
        for (k, &kk) in kappa.iter().enumerate() {
            assert!(kk.abs() < 1e-13, "κ[{k}]={kk}");
        }
    }

    // ---- Tying vectors ----

    #[test]
    fn tying_vectors_nonzero_in_correct_columns() {
        let n = [0.25_f64; 4];
        let dn = [0.1; 4];
        let h = tying_vector_r(&n, &dn, 1.0, 0.5);
        for i in 0..4 {
            assert_eq!(h[6 * i + 0], 0.0, "u column must be zero");
            assert_eq!(h[6 * i + 1], 0.0, "v column must be zero");
            assert_eq!(h[6 * i + 5], 0.0, "drill column must be zero");
        }
    }

    #[test]
    fn b_shear_mitc4_shape_is_2x24() {
        let zero = [0.0_f64; 24];
        let j_inv = [[1.0, 0.0], [0.0, 1.0]];
        let b = b_shear_mitc4(&zero, &zero, &zero, &zero, 0.0, 0.0, &j_inv);
        assert_eq!(b.len(), 48);
    }
}
