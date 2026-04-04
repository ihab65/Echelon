pub mod symbolic;
pub mod numeric;
pub mod solve;

// -----------------------------------------------------------------
// Backward-compatibility alias
// -----------------------------------------------------------------
//
// `SparseSolver` was the original name of the Cholesky solver. It is kept
// here as a type alias so that all existing import paths remain valid:
//
//   use solvers::cholesky::SparseSolver;       ← old path, still works
//   use solvers::linear::CholeskySolver;       ← new preferred path
//
// The concrete implementation lives in `solvers::linear::cholesky`.
// `SparseSolver` is not re-exported at the crate root to avoid ambiguity —
// new code should import `CholeskySolver` from `solvers::linear`.

/// Backward-compatible alias for [`crate::linear::CholeskySolver`].
///
/// `SparseSolver` was the original name of the sparse Cholesky solver.
/// All existing call sites using `SparseSolver` continue to compile without
/// changes. New code should use [`crate::linear::CholeskySolver`] instead.
///
/// # Migration
///
/// ```rust,ignore
/// // Before (still compiles):
/// use solvers::cholesky::SparseSolver;
/// let mut solver = SparseSolver::<f64>::new();
///
/// // After (preferred):
/// use solvers::linear::{CholeskySolver, LinearSolver};
/// let mut solver = CholeskySolver::<f64>::new();
/// ```
pub type SparseSolver<T> = crate::linear::CholeskySolver<T>;