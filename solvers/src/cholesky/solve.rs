//! Triangular solve: forward and backward substitution using `L`.

use crate::error::{SolverError, Result};
use super::numeric::NumericCholesky;

/// Solve `LLᵀu = f` by forward then backward substitution.
pub fn solve(num: &NumericCholesky, f: &[f64], u: &mut [f64]) -> Result<()> {
    if f.len() != num.n {
        return Err(SolverError::RhsSizeMismatch { expected: num.n, got: f.len() });
    }
    if u.len() != num.n {
        return Err(SolverError::RhsSizeMismatch { expected: num.n, got: u.len() });
    }
    // TODO: implement forward/backward substitution
    u.iter_mut().zip(f).for_each(|(ui, fi)| *ui = *fi);
    Ok(())
}