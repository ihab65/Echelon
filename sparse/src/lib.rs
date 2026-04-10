//! # sparse
//!
//! Compressed sparse matrix storage formats and operations for FEM assembly.
//!
//! ## Formats
//!
//! | Type             | Storage        | Primary use                     |
//! |------------------|----------------|---------------------------------|
//! | [`CsrMatrix`]    | Full, row-major| General assembly and matvec     |
//! | [`SymCsrMatrix`] | Upper triangle | Symmetric K, Cholesky input     |
//! | [`CscMatrix`]    | Full, col-major| Cholesky factorization (solver) |
//!
//! ## Typical workflow
//!
//! ```text
//! CooBuilder  →  CsrMatrix  →  SymCsrMatrix  →  CscMatrix
//!   (input)      (assembly)    (half storage)   (solver)
//! ```
//!
//! ## Crate layout
//!
//! ```text
//! src/
//!   lib.rs         — this file: SparseMatrix trait + re-exports
//!   error.rs       — SparseError, Result
//!   convert.rs     — conversions between formats
//!   coo.rs         — CooBuilder (triplet entry point)
//!   csr/           — CsrMatrix
//!   sym/           — SymCsrMatrix
//!   csc/           — CscMatrix
//! ```

pub mod error;
pub mod convert;
pub mod coo;
pub mod csr;
pub mod sym;
pub mod csc;

pub use error::{SparseError, Result};
pub use coo::CooBuilder;
pub use csr::CsrMatrix;
pub use sym::SymCsrMatrix;
pub use csc::CscMatrix;
pub use csr::ops::MatvecWorkspace;
pub use convert::ConvertWorkspace;

use std::fmt::{Debug, Display};
// -----------------------------------------------------------------
// SparseMatrix trait
//
// Implemented by all three storage types.  The solver crate accepts
// `&impl SparseMatrix` so it doesn't need to know which format it
// has been handed.
// -----------------------------------------------------------------

pub trait SparseScalar:
    Copy + Clone + Debug + PartialEq
    + num_traits::Zero           // additive identity
    + num_traits::One            // multiplicative identity
    + num_traits::Num            // +, -, *, /, %
    + num_traits::NumAssign      // +=, -=, *=, /=
    + num_traits::Signed         // abs, signum — Dual64 implements this
    + Send + Sync
    + Display
    + 'static
{
    /// Extract the real part for SPD checks and convergence tests.
    fn real_part(&self) -> f64;

    /// Square root — delegates to Float::sqrt for f32/f64,
    /// and to Dual64's own sqrt for autodiff.
    fn scalar_sqrt(self) -> Self;
}

impl SparseScalar for f64 {
    fn real_part(&self) -> f64 { *self }
    fn scalar_sqrt(self) -> Self { f64::sqrt(self) }
}

impl SparseScalar for f32 {
    fn real_part(&self) -> f64 { *self as f64 }
    fn scalar_sqrt(self) -> Self { f32::sqrt(self) }
}

#[cfg(feature = "autodiff")]
impl SparseScalar for num_dual::Dual64 {
    fn real_part(&self) -> f64 { self.re }
    fn scalar_sqrt(self) -> Self {
        use num_dual::DualNum;
        self.sqrt() 
    }
}

/// Common interface shared by all sparse matrix types in this crate.
pub trait SparseMatrix {
    /// Number of rows.
    fn nrows(&self) -> usize;

    /// Number of columns.
    fn ncols(&self) -> usize;

    /// Number of structurally non-zero entries (including stored zeros).
    fn nnz(&self) -> usize;

    /// Returns `true` if the matrix is square.
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    /// Density: `nnz / (nrows * ncols)`.
    /// Returns `0.0` for a 0×0 matrix.
    fn density(&self) -> f64 {
        let total = self.nrows() * self.ncols();
        if total == 0 { 0.0 } else { self.nnz() as f64 / total as f64 }
    }

    /// Verify all internal invariants.
    /// Returns `Ok(())` on a well-formed matrix.
    fn validate(&self) -> Result<()>;
}

// -----------------------------------------------------------------
// Compile-time Send + Sync assertions (zero runtime cost).
// If a future change breaks thread-safety the compiler catches it here.
// -----------------------------------------------------------------
#[allow(dead_code)]
fn _assert_send_sync<T: SparseScalar>() {
    fn req<T: Send + Sync>() {}
    req::<CsrMatrix<T>>();
    req::<SymCsrMatrix<T>>();
    req::<CscMatrix<T>>();
    req::<MatvecWorkspace<T>>();
}