//! Eigenvalue solvers for structural dynamics.
//!
//! The [`EigenSolver`] trait defines the interface for computing the natural
//! frequencies and mode shapes of a structural model from the generalised
//! eigenvalue problem:
//!
//! ```text
//! K φ = ω² M φ
//! ```
//!
//! where:
//! - `K` is the global stiffness matrix (symmetric positive semi-definite)
//! - `M` is the global mass matrix (symmetric positive definite)
//! - `ω²` are the squared natural frequencies (eigenvalues)
//! - `φ` are the mode shape vectors (eigenvectors)
//!
//! ## Implementations
//!
//! | Type | Algorithm | Notes |
//! |------|-----------|-------|
//! | [`LanczosEigenSolver`] | Shift-invert Lanczos | Best for large sparse K, M; few modes needed |
//!
//! Planned implementations:
//!
//! | Type | Algorithm | Notes |
//! |------|-----------|-------|
//! | `SubspaceEigenSolver` | Subspace iteration | Robust alternative for ill-conditioned systems |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use solvers::eigen::{EigenSolver, LanczosEigenSolver};
//!
//! // Solve for the lowest 5 modes
//! let mut solver = LanczosEigenSolver::new();
//! let result = solver.solve_modes(&k, &m, 5)?;
//! println!("ω₁ = {:.4} rad/s", result.frequencies[0].sqrt());
//! for (i, shape) in result.mode_shapes.iter().enumerate() {
//!     println!("Mode {}: {:?}", i + 1, shape);
//! }
//! ```

pub mod lanczos;
pub use lanczos::LanczosEigenSolver;

use sparse::SymCsrMatrix;
use crate::error::Result;

// -----------------------------------------------------------------
// EigenResult
// -----------------------------------------------------------------

/// Output of an eigenvalue solve: natural frequencies and mode shapes.
///
/// Eigenvalues are returned as `ω²` (squared angular frequencies in rad²/s²).
/// Take the square root to get angular frequencies in rad/s, then divide by
/// `2π` for frequencies in Hz.
///
/// Mode shapes are mass-orthonormalised by convention:
/// ```text
/// φᵢᵀ M φⱼ = δᵢⱼ    (Kronecker delta)
/// φᵢᵀ K φⱼ = ωᵢ² δᵢⱼ
/// ```
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Squared natural frequencies `ω²` in ascending order, length `n_modes`.
    pub eigenvalues:  Vec<f64>,

    /// Mode shape vectors, one per eigenvalue.
    /// `mode_shapes[i]` has length `n_dof` and satisfies `φᵢᵀ M φᵢ = 1`.
    pub mode_shapes:  Vec<Vec<f64>>,
}

impl EigenResult {
    /// Angular natural frequencies `ωᵢ = √(ω²ᵢ)` in rad/s.
    pub fn angular_frequencies(&self) -> Vec<f64> {
        self.eigenvalues.iter().map(|&w2| w2.max(0.0).sqrt()).collect()
    }

    /// Natural frequencies `fᵢ = ωᵢ / (2π)` in Hz.
    pub fn frequencies_hz(&self) -> Vec<f64> {
        self.angular_frequencies()
            .into_iter()
            .map(|w| w / (2.0 * std::f64::consts::PI))
            .collect()
    }

    /// Natural periods `Tᵢ = 1 / fᵢ` in seconds.
    pub fn periods(&self) -> Vec<f64> {
        self.frequencies_hz()
            .into_iter()
            .map(|f| if f > 0.0 { 1.0 / f } else { f64::INFINITY })
            .collect()
    }
}

// -----------------------------------------------------------------
// EigenSolver trait
// -----------------------------------------------------------------

/// Interface for generalised eigenvalue solvers.
///
/// Solves the structural dynamics eigenvalue problem `K φ = ω² M φ` for the
/// `n_modes` lowest eigenvalues and their corresponding mode shapes.
///
/// # Two-phase design
///
/// Unlike linear solvers, eigenvalue solvers operate in two phases:
///
/// 1. **`prepare`** — symbolic/structural setup, shift computation, and
///    factorization of the shifted matrix `(K - σ M)` where `σ` is a shift
///    below the smallest eigenvalue of interest. Called once per topology
///    change or when the target frequency band changes.
///
/// 2. **`solve_modes`** — runs the iterative eigenvalue algorithm (Lanczos,
///    subspace iteration, etc.) and returns the [`EigenResult`]. Called once
///    per eigenvalue analysis request.
///
/// # Convergence tolerance
///
/// The solver terminates when the relative residual for each mode satisfies:
/// ```text
/// ‖K φᵢ - ωᵢ² M φᵢ‖ / (ωᵢ² ‖M φᵢ‖) < tol
/// ```
pub trait EigenSolver {
    /// Prepare the solver for a given stiffness and mass matrix pair.
    ///
    /// This phase may factorize `(K - σ M)` for a chosen shift `σ`, build
    /// the Lanczos starting vector, or perform other setup that is independent
    /// of the number of modes requested.
    ///
    /// # Arguments
    /// * `k` — global stiffness matrix (upper triangle, SPD after BCs applied)
    /// * `m` — global mass matrix (upper triangle, SPD)
    ///
    /// # Errors
    /// - [`crate::error::SolverError::NotPositiveDefinite`] if the shift-invert matrix is singular.
    fn prepare(&mut self, k: &SymCsrMatrix<f64>, m: &SymCsrMatrix<f64>) -> Result<()>;

    /// Compute the `n_modes` lowest eigenpairs.
    ///
    /// # Arguments
    /// * `n_modes` — number of eigenvalues/vectors to compute (must be ≥ 1)
    ///
    /// # Returns
    /// An [`EigenResult`] with eigenvalues sorted ascending and mass-normalised
    /// mode shapes.
    ///
    /// # Errors
    /// - Convergence failure if the algorithm does not converge within the
    ///   maximum iteration count.
    /// - Any error from the internal linear solve steps.
    fn solve_modes(&mut self, n_modes: usize) -> Result<EigenResult>;

    /// Convenience: prepare and solve in one call.
    ///
    /// Equivalent to calling `prepare` then `solve_modes`.
    fn compute(
        &mut self,
        k:       &SymCsrMatrix<f64>,
        m:       &SymCsrMatrix<f64>,
        n_modes: usize,
    ) -> Result<EigenResult> {
        self.prepare(k, m)?;
        self.solve_modes(n_modes)
    }
}