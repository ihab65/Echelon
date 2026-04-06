//! Rayleigh proportional damping matrix construction.
//!
//! Rayleigh damping is the standard approach for introducing physical damping
//! into an implicit dynamic analysis without having to measure individual modal
//! damping ratios for every mode.
//!
//! ## Formula
//!
//! ```text
//! C = α_M · M  +  β_K · K
//! ```
//!
//! where:
//! - `M` is the global mass matrix (assembled via `assemble_mass`)
//! - `K` is the global stiffness matrix (assembled at the initial undeformed state)
//! - `α_M` (mass-proportional coefficient) — damps low-frequency modes
//! - `β_K` (stiffness-proportional coefficient) — damps high-frequency modes
//!
//! ## Choosing coefficients
//!
//! For target damping ratio `ξ` at two circular frequencies `ω₁` and `ω₂`:
//!
//! ```text
//! α_M = 2·ω₁·ω₂·ξ / (ω₁ + ω₂)
//! β_K = 2·ξ / (ω₁ + ω₂)
//! ```
//!
//! Typical seismic analysis uses `ξ = 0.05` (5% critical damping) and
//! targets the first and third natural frequencies of the structure.
//!
//! ## References
//!
//! Chopra, A.K. (2012). *Dynamics of Structures*, 4th ed. §11.4.

use sparse::SymCsrMatrix;
use crate::error::{AssemblyError, Result};

// -----------------------------------------------------------------
// build_rayleigh_damping
// -----------------------------------------------------------------

/// Compute the Rayleigh damping matrix `C = α_M · M + β_K · K`.
///
/// Both `mass` and `stiffness` must have **identical sparsity patterns**
/// (built from the same mesh topology via `build_pattern` and `assemble_mass`).
/// The result has the same pattern.
///
/// # Arguments
/// * `mass`      — global consistent or lumped mass matrix
/// * `stiffness` — global elastic stiffness matrix at the reference state
///                 (typically the undeformed geometry, full load applied)
/// * `alpha_m`   — mass-proportional Rayleigh coefficient (`α_M ≥ 0`)
/// * `beta_k`    — stiffness-proportional Rayleigh coefficient (`β_K ≥ 0`)
///
/// # Errors
/// Returns [`AssemblyError::Sparse`] if the matrices have different sizes.
///
/// # Example
///
/// ```rust,ignore
/// // 5% critical damping at ω₁ = 10 rad/s and ω₂ = 50 rad/s
/// let xi = 0.05;
/// let (w1, w2) = (10.0, 50.0);
/// let alpha_m = 2.0 * w1 * w2 * xi / (w1 + w2);
/// let beta_k  = 2.0 * xi / (w1 + w2);
///
/// let c = build_rayleigh_damping(&mass, &stiffness, alpha_m, beta_k)?;
/// let integrator = Newmark::average_acceleration(dt, mass, Some(c));
/// ```
pub fn build_rayleigh_damping(
    mass:      &SymCsrMatrix<f64>,
    stiffness: &SymCsrMatrix<f64>,
    alpha_m:   f64,
    beta_k:    f64,
) -> Result<SymCsrMatrix<f64>> {
    if mass.n != stiffness.n {
        return Err(AssemblyError::Sparse(sparse::SparseError::DimensionMismatch {
            expected: mass.n,
            got:      stiffness.n,
        }));
    }

    // Clone the mass matrix to start; we will accumulate α_M·M + β_K·K into it.
    let mut c = mass.clone();

    // Scale the clone by α_M in place.
    c.scale(alpha_m);

    // Add β_K · K: iterate every stored entry of stiffness and accumulate.
    for (row, col, val) in stiffness.iter_upper() {
        c.add_value(row, col, val * beta_k)
            .map_err(|e| AssemblyError::Sparse(e))?;
    }

    Ok(c)
}

/// Compute the Rayleigh coefficients for a target damping ratio `xi`
/// at two circular frequencies (rad/s).
///
/// Returns `(alpha_m, beta_k)` suitable for passing to
/// [`build_rayleigh_damping`].
///
/// # Panics
/// Panics if `omega1 >= omega2` or either is non-positive.
pub fn rayleigh_coefficients(xi: f64, omega1: f64, omega2: f64) -> (f64, f64) {
    assert!(omega1 > 0.0 && omega2 > omega1,
        "rayleigh_coefficients: require 0 < omega1 < omega2, got ({omega1}, {omega2})");
    let alpha_m = 2.0 * omega1 * omega2 * xi / (omega1 + omega2);
    let beta_k  = 2.0 * xi / (omega1 + omega2);
    (alpha_m, beta_k)
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;

    fn diag_2x2(a: f64, b: f64) -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, a);
        coo.add(1, 1, b);
        coo.build_sym().unwrap()
    }

    fn tridiag_2x2(diag: f64, off: f64) -> SymCsrMatrix<f64> {
        let mut coo = CooBuilder::new(2, 2);
        coo.add(0, 0, diag);
        coo.add(1, 1, diag);
        coo.add(0, 1, off);
        coo.build_sym().unwrap()
    }

    #[test]
    fn zero_coefficients_gives_zero_matrix() {
        let m = diag_2x2(2.0, 3.0);
        let k = diag_2x2(100.0, 200.0);
        let c = build_rayleigh_damping(&m, &k, 0.0, 0.0).unwrap();
        assert_eq!(c.get(0, 0).unwrap(), 0.0);
        assert_eq!(c.get(1, 1).unwrap(), 0.0);
    }

    #[test]
    fn mass_only_rayleigh() {
        // C = 2.0 * M, K contribution = 0
        let m = diag_2x2(10.0, 20.0);
        let k = diag_2x2(100.0, 200.0);
        let c = build_rayleigh_damping(&m, &k, 2.0, 0.0).unwrap();
        assert!((c.get(0, 0).unwrap() - 20.0).abs() < 1e-12);
        assert!((c.get(1, 1).unwrap() - 40.0).abs() < 1e-12);
    }

    #[test]
    fn stiffness_only_rayleigh() {
        // C = 0.5 * K, M contribution = 0
        let m = diag_2x2(10.0, 20.0);
        let k = diag_2x2(100.0, 200.0);
        let c = build_rayleigh_damping(&m, &k, 0.0, 0.5).unwrap();
        assert!((c.get(0, 0).unwrap() - 50.0).abs() < 1e-12);
        assert!((c.get(1, 1).unwrap() - 100.0).abs() < 1e-12);
    }

    #[test]
    fn combined_rayleigh_correct() {
        // C = 1.0*M + 0.01*K
        let m = diag_2x2(5.0, 8.0);
        let k = diag_2x2(1000.0, 2000.0);
        let c = build_rayleigh_damping(&m, &k, 1.0, 0.01).unwrap();
        // C[0,0] = 1*5 + 0.01*1000 = 15
        // C[1,1] = 1*8 + 0.01*2000 = 28
        assert!((c.get(0, 0).unwrap() - 15.0).abs() < 1e-10);
        assert!((c.get(1, 1).unwrap() - 28.0).abs() < 1e-10);
    }

    #[test]
    fn combined_rayleigh_off_diagonal() {
        let m = tridiag_2x2(4.0, -1.0);
        let k = tridiag_2x2(8.0, -2.0);
        // C = 2*M + 3*K
        // diag:  2*4 + 3*8 = 32
        // off:   2*(-1) + 3*(-2) = -8
        let c = build_rayleigh_damping(&m, &k, 2.0, 3.0).unwrap();
        assert!((c.get(0, 0).unwrap() - 32.0).abs() < 1e-10);
        assert!((c.get(0, 1).unwrap() - (-8.0)).abs() < 1e-10);
    }

    #[test]
    fn size_mismatch_errors() {
        let m = diag_2x2(1.0, 1.0);
        let k = {
            let mut coo = CooBuilder::new(3, 3);
            coo.add(0, 0, 1.0); coo.add(1, 1, 1.0); coo.add(2, 2, 1.0);
            coo.build_sym().unwrap()
        };
        assert!(build_rayleigh_damping(&m, &k, 1.0, 1.0).is_err());
    }

    #[test]
    fn rayleigh_coefficients_5pct_at_10_50() {
        let (a, b) = rayleigh_coefficients(0.05, 10.0, 50.0);
        // α_M = 2*10*50*0.05/(10+50) = 50/60 ≈ 0.8333
        // β_K = 2*0.05/(10+50) = 0.1/60 ≈ 0.001667
        assert!((a - 50.0 / 60.0).abs() < 1e-12, "alpha_m={a}");
        assert!((b - 0.1 / 60.0).abs() < 1e-12, "beta_k={b}");
    }
}