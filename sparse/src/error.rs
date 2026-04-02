use miette::Diagnostic;
use thiserror::Error;

/// All errors that can arise from sparse matrix storage and operations.
///
/// Every variant carries a dot-separated diagnostic code traceable to its
/// exact origin in the Echelon stack, and a `help` attribute that states
/// a structural or mathematical hypothesis for why the fault occurred.
///
/// Marked `#[non_exhaustive]` so that adding a new variant in a future
/// release is not a breaking change for downstream crates that match on it.
#[derive(Debug, Error, Diagnostic)]
#[non_exhaustive]
pub enum SparseError {
    // ---- indexing -------------------------------------------------------

    /// A `(row, col)` coordinate is within the matrix bounds but is absent
    /// from the declared sparsity pattern.
    ///
    /// Returned by `add_value`, `set_value`, and `scatter_add`.
    #[error("Sparsity pattern violation: entry ({row}, {col}) is structurally absent from the declared non-zero set.")]
    #[diagnostic(
        code(echelon::sparse::pattern::absent_entry),
        help(
            "The element stiffness matrix or load vector is attempting to \
             write to a global DOF pair ({row}, {col}) that was not included \
             during pattern construction. Verify that the element's DOF map \
             in `from_dof_connectivity` or `from_pattern` correctly enumerates \
             all pairs (i, j) that this element couples. If using `scatter_add`, \
             ensure the sparsity pattern was built from the same connectivity list."
        )
    )]
    IndexOutOfBounds { row: usize, col: usize },

    /// A row index exceeds the allocated matrix dimension.
    #[error("Row index out of range: requested row {row}, but matrix has only {nrows} rows.")]
    #[diagnostic(
        code(echelon::sparse::index::row_out_of_range),
        help(
            "A node or DOF index ({row}) exceeds the total allocated dimension \
             ({nrows}). This typically indicates a mismatch between the number \
             of nodes declared to the assembler and the connectivity list provided. \
             Verify that `n_dof` was computed from the complete nodal set before \
             pattern construction."
        )
    )]
    RowOutOfRange { row: usize, nrows: usize },

    /// A column index exceeds the allocated matrix dimension.
    #[error("Column index out of range: requested column {col}, but matrix has only {ncols} columns.")]
    #[diagnostic(
        code(echelon::sparse::index::col_out_of_range),
        help(
            "A node or DOF index ({col}) exceeds the total allocated column \
             dimension ({ncols}). Verify that `n_dof` was computed from the \
             complete nodal set and that the element connectivity does not \
             reference node IDs created after the pattern was built."
        )
    )]
    ColOutOfRange { col: usize, ncols: usize },

    /// A column index is below the row index, placing the entry in the
    /// lower triangle of a `SymCsrMatrix` that stores only the upper triangle.
    #[error("Lower-triangle entry rejected: column {col} < row {row} in symmetric upper-triangle storage.")]
    #[diagnostic(
        code(echelon::sparse::symmetry::lower_triangle_entry),
        help(
            "SymCsrMatrix stores only the upper triangle (col >= row). \
             The entry ({row}, {col}) lies in the lower triangle and must \
             be provided as its mirror ({col}, {row}) instead. \
             When using `CooBuilder::build_sym`, lower-triangle triplets \
             are automatically reflected — use that interface if you are \
             assembling from both triangles."
        )
    )]
    LowerTriangleEntry { row: usize, col: usize },

    // ---- shape ----------------------------------------------------------

    /// A vector or operand dimension is inconsistent with what the operation
    /// requires.
    #[error("Dimension mismatch: operation requires length {expected}, but received length {got}.")]
    #[diagnostic(
        code(echelon::sparse::shape::dimension_mismatch),
        help(
            "The supplied vector or workspace has {got} entries but the matrix \
             operation requires exactly {expected}. Ensure the right-hand side \
             vector and solution vector are both allocated for the full \
             unconstrained DOF count before applying Dirichlet boundary conditions."
        )
    )]
    DimensionMismatch { expected: usize, got: usize },

    /// The number of row patterns does not match the declared row count.
    #[error("Pattern length mismatch: {pattern_len} row patterns provided for a matrix declared with {nrows} rows.")]
    #[diagnostic(
        code(echelon::sparse::pattern::length_mismatch),
        help(
            "The sparsity pattern must supply exactly one column-index list per \
             row. Provided {pattern_len} lists for a matrix with {nrows} rows. \
             Verify that `n_dof` matches the length of the pattern vector and \
             that no nodes were added to the mesh after the pattern was \
             constructed."
        )
    )]
    PatternLengthMismatch { pattern_len: usize, nrows: usize },

    /// An operation requiring a square matrix received a rectangular one.
    #[error("Non-square matrix: operation requires a square matrix, but received {nrows}×{ncols}.")]
    #[diagnostic(
        code(echelon::sparse::shape::not_square),
        help(
            "Cholesky factorization, symmetric matvec, and DOF-connectivity \
             assembly all require a square matrix. The supplied matrix has \
             {nrows} rows and {ncols} columns. Ensure the DOF count is \
             consistent across row and column dimensions."
        )
    )]
    NotSquare { nrows: usize, ncols: usize },

    // ---- construction ---------------------------------------------------

    /// The flat element stiffness array has the wrong length for the
    /// supplied DOF map.
    #[error(
        "Element stiffness size mismatch: `ke` has {ke_len} entries but \
         the DOF map has {n} DOFs, requiring n²={expected} entries (row-major)."
    )]
    #[diagnostic(
        code(echelon::sparse::assembly::scatter_size_mismatch),
        help(
            "The flat element stiffness matrix passed to `scatter_add` must \
             have exactly n² entries in row-major order, where n = |dof_map|. \
             Received {ke_len} entries for a {n}-DOF element (expected {expected}). \
             Check that the element correctly reports its DOF count via `n_dof()` \
             and that `ke_flat()` returns a vector of that length squared."
        )
    )]
    ScatterSizeMismatch { ke_len: usize, n: usize, expected: usize },

    // ---- I/O ------------------------------------------------------------

    /// Errors arising from Matrix Market file parsing or file-system access.
    #[error("Matrix Market I/O error: {0}")]
    #[diagnostic(
        code(echelon::sparse::io::matrix_market_error),
        help(
            "Verify that the file path is correct, that the file follows the \
             Matrix Market exchange format (%%MatrixMarket header, integer \
             dimension line, coordinate triplets), and that the declared \
             dimensions match the number of data lines present."
        )
    )]
    IoError(String),
}

/// Convenience alias used throughout the `sparse` crate.
pub type Result<T> = std::result::Result<T, SparseError>;

impl From<std::io::Error> for SparseError {
    fn from(err: std::io::Error) -> Self {
        SparseError::IoError(err.to_string())
    }
}