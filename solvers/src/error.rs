use thiserror::Error;
use sparse::SparseError;

/// Errors that can arise during sparse direct solving.
///
/// Wraps [`SparseError`] for storage-level failures encountered during
/// factorization setup.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SolverError {
    /// The matrix passed to the solver has a storage-level problem.
    #[error("sparse storage error: {0}")]
    Sparse(#[from] SparseError),

    /// The matrix is not positive definite.
    /// Contains the index and value of the offending diagonal.
    #[error("matrix is not positive definite: L[{index},{index}]² = {value}")]
    NotPositiveDefinite { index: usize, value: f64 },

    /// `factorize()` called before `analyze()`.
    #[error("call analyze() before factorize()")]
    NotAnalyzed,

    /// `solve()` called before `factorize()`.
    #[error("call factorize() before solve()")]
    NotFactorized,

    /// RHS vector has the wrong length.
    #[error("RHS has {got} entries; matrix has {expected} rows")]
    RhsSizeMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, SolverError>;