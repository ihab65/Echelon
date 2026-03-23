//! Numeric Cholesky factorization: compute `L` from `K` and the symbolic
//! factor pattern.

use crate::error::Result;
use super::symbolic::SymbolicCholesky;
use sparse::SymCsrMatrix;

/// Values of the Cholesky factor `L`.
pub struct NumericCholesky {
    pub values: Vec<f64>,
    pub n:      usize,
    // references the symbolic pattern (col_ptr, row_idx) from SymbolicCholesky
}

/// Compute the numeric Cholesky factorization.
pub fn factorize(_k: &SymCsrMatrix, _sym: &SymbolicCholesky) -> Result<NumericCholesky> {
    // TODO: implement
    // Uses the column-by-column "left-looking" algorithm with a dense
    // column workspace (the standard approach for sparse Cholesky).
    Ok(NumericCholesky { values: Vec::new(), n: _k.n })
}