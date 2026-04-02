use miette::Diagnostic;
use thiserror::Error;
use sparse::SparseError;

/// Errors that can arise during the sparse Cholesky direct solve pipeline.
///
/// Every variant carries a dot-separated diagnostic code traceable to its
/// exact origin and a `help` attribute that states a structural or
/// mathematical hypothesis for why the fault occurred.
///
/// These error codes are programmatically catchable in probabilistic
/// sampling scripts: an `echelon::solvers::cholesky::indefinite_matrix`
/// code signals a collapsed or kinematically unstable structural
/// realisation that should be logged and discarded, not treated as a
/// fatal crash.
#[derive(Debug, Error, Diagnostic)]
#[non_exhaustive]
pub enum SolverError {
    /// The tangent stiffness matrix is not positive-definite: a zero or
    /// negative Schur complement was encountered at the indicated pivot
    /// equation during Cholesky elimination.
    ///
    /// The `index` field is the equation number (0-indexed) in the
    /// **permuted** DOF ordering. The `value` field is the numerical
    /// value of the failed pivot before the square-root step.
    #[error(
        "Tangent stiffness matrix is not positive-definite: \
         pivot at permuted equation {index} evaluated to {value:.6e} \
         (threshold: 1e-12)."
    )]
    #[diagnostic(
        code(echelon::solvers::cholesky::indefinite_matrix),
        help(
            "A non-positive pivot indicates one of the following structural \
             conditions: \n\
             (1) Kinematic instability — the structure contains a mechanism \
                 (rigid body mode) because insufficient Dirichlet boundary \
                 conditions have been applied. Verify that all rigid body \
                 motions (translations and rotations) are restrained. \
                 For a 2D frame model, at minimum fix ux, uy at one node \
                 and uy at a second non-colinear node. \n\
             (2) Element with zero or negative stiffness — an element has \
                 non-physical material parameters (E ≤ 0, A ≤ 0, or Iz ≤ 0). \
                 Inspect the parameter sample that produced this realisation. \n\
             (3) Numerical ill-conditioning — the stiffness contrast between \
                 elements exceeds approximately 1e12, causing catastrophic \
                 cancellation. Consider scaling or regularisation. \n\
             In Monte Carlo population runs, log this realisation as \
             'kinematically unstable' and continue sampling."
        )
    )]
    NotPositiveDefinite { index: usize, value: f64 },

    /// The sparse storage layer returned an error during matrix preparation
    /// (pattern construction, permutation, or format conversion).
    #[error(transparent)]
    #[diagnostic(transparent)]
    Sparse(#[from] SparseError),

    /// `factorize()` was called before `analyze()`.
    ///
    /// The symbolic Cholesky phase (elimination tree and fill pattern) must
    /// be completed before numerical values can be factored.
    #[error(
        "Solver state violation: `factorize()` called before `analyze()`. \
         The symbolic phase must be completed first."
    )]
    #[diagnostic(
        code(echelon::solvers::state::not_analyzed),
        help(
            "Call `solver.analyze(&K)` once per mesh topology change before \
             calling `solver.factorize(&K)`. The symbolic phase is O(nnz) and \
             computes the elimination tree and fill pattern — it only needs to \
             be repeated when the sparsity pattern of K changes (i.e. when \
             elements are added or removed, not merely when stiffness values \
             are updated). Use `solver.analyze_and_factorize(&K)` for the \
             initial solve, then only `solver.factorize(&K)` for each Newton \
             iteration."
        )
    )]
    NotAnalyzed,

    /// `solve()` was called before `factorize()`.
    ///
    /// The numeric Cholesky factors L must exist before a triangular solve
    /// can be performed.
    #[error(
        "Solver state violation: `solve()` called before `factorize()`. \
         The numeric Cholesky factor L has not been computed."
    )]
    #[diagnostic(
        code(echelon::solvers::state::not_factorized),
        help(
            "Call `solver.factorize(&K)` after `solver.analyze(&K)` and before \
             `solver.solve(&f, &mut u)`. If `solver.analyze()` was called again \
             after a previous factorization (e.g. due to a topology change), \
             `factorize()` must be called again before the next solve. \
             Note: `analyze()` intentionally clears the numeric factor to \
             prevent stale results."
        )
    )]
    NotFactorized,

    /// The right-hand-side vector or solution vector has the wrong length.
    #[error(
        "Right-hand-side size mismatch: system has {expected} degrees of freedom \
         but the supplied vector has {got} entries."
    )]
    #[diagnostic(
        code(echelon::solvers::solve::rhs_size_mismatch),
        help(
            "Both the load vector `f` and the solution vector `u` must have \
             exactly as many entries as the global stiffness matrix has rows \
             ({expected}). Ensure that Dirichlet boundary conditions are applied \
             by zeroing rows/columns of K (via `zero_row_col`) rather than by \
             reducing the matrix size, so the DOF count remains consistent \
             across K, f, and u."
        )
    )]
    RhsSizeMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, SolverError>;