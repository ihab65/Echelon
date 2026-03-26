use thiserror::Error;

/// All errors that can arise from sparse matrix storage and operations.
///
/// Marked `#[non_exhaustive]` so that adding a new variant in a future
/// release is not a breaking change for downstream crates that match on it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SparseError {
    // ---- indexing -------------------------------------------------------
    /// `(row, col)` is within matrix bounds but absent from the sparsity
    /// pattern.  Returned by `add_value`, `set_value`, and `scatter_add`.
    #[error("({row}, {col}) is not in the sparsity pattern")]
    IndexOutOfBounds { row: usize, col: usize },

    /// Row index is out of range.
    #[error("row {row} is out of range (matrix has {nrows} rows)")]
    RowOutOfRange { row: usize, nrows: usize },

    /// Column index is out of range.
    #[error("column {col} is out of range (matrix has {ncols} columns)")]
    ColOutOfRange { col: usize, ncols: usize },

    /// A column index in `SymCsrMatrix` is below the row index (lower
    /// triangle entry passed to an upper-triangle-only structure).
    #[error("column {col} < row {row}: lower-triangle entry in symmetric storage")]
    LowerTriangleEntry { row: usize, col: usize },

    // ---- shape ----------------------------------------------------------
    /// Vector or operand dimension does not match what the operation expects.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// `pattern.len() != nrows` in `from_pattern`.
    #[error("pattern has {pattern_len} rows but matrix has {nrows} rows")]
    PatternLengthMismatch { pattern_len: usize, nrows: usize },

    /// An operation requires a square matrix.
    #[error("matrix must be square, but is {nrows}x{ncols}")]
    NotSquare { nrows: usize, ncols: usize },

    // ---- construction ---------------------------------------------------
    /// `ke.len() != dof_map.len()²` in `scatter_add`.
    #[error(
        "element stiffness has {ke_len} entries but dof_map has {n} DOFs \
         (expected {n}² = {expected} entries)"
    )]
    ScatterSizeMismatch { ke_len: usize, n: usize, expected: usize },

    // ---- I/O ------------------------------------------------------------
    /// Errors arising from Matrix Market parsing or file system operations.
    #[error("I/O error: {0}")]
    IoError(String),
}

/// Convenience alias used throughout the `sparse` crate.
pub type Result<T> = std::result::Result<T, SparseError>;

// Manual implementation for std::io::Error to allow '?' operator in to_mtx
impl From<std::io::Error> for SparseError {
    fn from(err: std::io::Error) -> Self {
        SparseError::IoError(err.to_string())
    }
}