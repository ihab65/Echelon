//! 2D Gauss–Legendre quadrature rules for isoparametric elements.
//!
//! All rules operate on the reference square `[-1, 1]²`.
//! The `2×2` rule is the primary integration rule for MITC4 and other Q4
//! elements; it integrates polynomials of degree ≤ 3 exactly in each natural
//! coordinate direction.
//!
//! No heap allocations are made — every return type is a fixed-size stack
//! array.

// -----------------------------------------------------------------
// Types
// -----------------------------------------------------------------

/// A single 2D Gauss–Legendre integration point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussPoint2d {
    /// Natural coordinate `r ∈ [-1, 1]`.
    pub r: f64,
    /// Natural coordinate `s ∈ [-1, 1]`.
    pub s: f64,
    /// Integration weight.
    pub weight: f64,
}

// -----------------------------------------------------------------
// 2×2 Gauss rule
// -----------------------------------------------------------------

/// 2×2 Gauss–Legendre quadrature rule on `[-1, 1]²`.
///
/// Returns the 4 integration points in the order:
/// `(-, -)`, `(+, -)`, `(-, +)`, `(+, +)` where `±` denotes `±1/√3`.
///
/// Each weight is `1.0`. The rule integrates polynomials of degree ≤ 3
/// exactly in each natural coordinate direction — sufficient for bilinear
/// Q4 elements.
///
/// # Example
///
/// ```
/// use elements::local::gauss::gauss_2x2;
///
/// let pts = gauss_2x2();
/// let total_weight: f64 = pts.iter().map(|p| p.weight).sum();
/// assert!((total_weight - 4.0).abs() < 1e-15);
/// ```
pub fn gauss_2x2() -> [GaussPoint2d; 4] {
    let a = 1.0 / 3.0_f64.sqrt();
    [
        GaussPoint2d { r: -a, s: -a, weight: 1.0 },
        GaussPoint2d { r:  a, s: -a, weight: 1.0 },
        GaussPoint2d { r: -a, s:  a, weight: 1.0 },
        GaussPoint2d { r:  a, s:  a, weight: 1.0 },
    ]
}

// -----------------------------------------------------------------
// MITC4 tying points
// -----------------------------------------------------------------

/// MITC4 transverse shear tying point coordinates in natural space.
///
/// The four tying points are the **mid-points of the element edges** in
/// natural coordinates.  The MITC4 formulation evaluates the covariant
/// transverse shear strains at these points and interpolates them across
/// the element to eliminate shear locking without reduced integration.
///
/// ```text
///     s
///     │        C = (0, +1)
///     │      ┌────┬────┐
///     │      │    │    │
///  D=(─1,0)──┼────┼────┼──B=(+1,0)
///     │      │    │    │
///     │      └────┴────┘
///     │        A = (0, ─1)
///     └──────────────────── r
/// ```
///
/// | Index | Point | Position | Used for |
/// |-------|-------|----------|----------|
/// | 0 | A | `(0, −1)` | r-direction covariant shear `γ^cov_r` |
/// | 1 | B | `(+1, 0)` | s-direction covariant shear `γ^cov_s` |
/// | 2 | C | `(0, +1)` | r-direction covariant shear `γ^cov_r` |
/// | 3 | D | `(−1, 0)` | s-direction covariant shear `γ^cov_s` |
///
/// The tying interpolation at a Gauss point `(r, s)` is:
/// ```text
/// γ̃_rz(r,s) = ½(1−s)·γ^A_rz + ½(1+s)·γ^C_rz
/// γ̃_sz(r,s) = ½(1+r)·γ^B_sz + ½(1−r)·γ^D_sz
/// ```
pub fn mitc4_tying_points() -> [(f64, f64); 4] {
    [
        ( 0.0, -1.0), // A — mid-point of edge 0→1
        ( 1.0,  0.0), // B — mid-point of edge 1→2
        ( 0.0,  1.0), // C — mid-point of edge 2→3
        (-1.0,  0.0), // D — mid-point of edge 3→0
    ]
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- 2×2 Gauss rule ----

    #[test]
    fn gauss_2x2_weight_sum() {
        let pts = gauss_2x2();
        let total: f64 = pts.iter().map(|p| p.weight).sum();
        assert!((total - 4.0).abs() < 1e-15, "weight sum = {total}");
    }

    #[test]
    fn gauss_2x2_integrates_constant_exactly() {
        // ∫₋₁¹ ∫₋₁¹ 1 dr ds = 4
        let pts = gauss_2x2();
        let integral: f64 = pts.iter().map(|p| p.weight).sum();
        assert!((integral - 4.0).abs() < 1e-15);
    }

    #[test]
    fn gauss_2x2_integrates_quadratic_exactly() {
        // ∫₋₁¹ ∫₋₁¹ r² dr ds = (2/3)·2 = 4/3
        let pts = gauss_2x2();
        let integral: f64 = pts.iter().map(|p| p.weight * p.r * p.r).sum();
        assert!((integral - 4.0 / 3.0).abs() < 1e-14, "integral = {integral}");
    }

    #[test]
    fn gauss_2x2_integrates_bilinear_exactly() {
        // ∫₋₁¹ ∫₋₁¹ r·s dr ds = 0
        let pts = gauss_2x2();
        let integral: f64 = pts.iter().map(|p| p.weight * p.r * p.s).sum();
        assert!(integral.abs() < 1e-15, "integral = {integral}");
    }

    #[test]
    fn gauss_2x2_integrates_degree3_exactly() {
        // ∫₋₁¹ ∫₋₁¹ r³ dr ds = 0  (odd function)
        let pts = gauss_2x2();
        let integral: f64 = pts.iter().map(|p| p.weight * p.r.powi(3)).sum();
        assert!(integral.abs() < 1e-14, "integral = {integral}");
    }

    #[test]
    fn gauss_2x2_symmetric_about_origin() {
        let pts = gauss_2x2();
        // Sum of r-coords = 0 (symmetric about r=0)
        let sum_r: f64 = pts.iter().map(|p| p.r).sum();
        assert!(sum_r.abs() < 1e-15, "sum r = {sum_r}");
        // Sum of s-coords = 0
        let sum_s: f64 = pts.iter().map(|p| p.s).sum();
        assert!(sum_s.abs() < 1e-15, "sum s = {sum_s}");
    }

    // ---- Tying points ----

    #[test]
    fn tying_points_at_edge_midpoints() {
        let pts = mitc4_tying_points();
        // A: mid of bottom edge
        assert_eq!(pts[0], (0.0, -1.0));
        // B: mid of right edge
        assert_eq!(pts[1], (1.0,  0.0));
        // C: mid of top edge
        assert_eq!(pts[2], (0.0,  1.0));
        // D: mid of left edge
        assert_eq!(pts[3], (-1.0, 0.0));
    }

    #[test]
    fn tying_points_on_boundary() {
        // All tying points should lie on the boundary |r|=1 or |s|=1
        for (r, s) in mitc4_tying_points() {
            let on_boundary = r.abs() == 1.0 || s.abs() == 1.0;
            assert!(on_boundary, "({r},{s}) is not on the boundary");
        }
    }
}
