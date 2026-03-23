//! Permutation vector newtype.
//!
//! A [`Permutation`] maps new (reordered) indices to old (original) indices:
//! `perm[new] = old`.
//!
//! The inverse maps old to new: `inv[old] = new`.
//!
//! ## Applying to a matrix
//!
//! To permute `K` with permutation `p`:
//! ```text
//! K_permuted[i, j] = K[p[i], p[j]]
//! ```
//! This is what the solver does before factorization when the ordering is
//! not the identity.

use sparse::SymCsrMatrix;
use sparse::error::{SparseError, Result};

/// A reordering permutation: `perm[new_index] = old_index`.
///
/// Invariant: `perm` is a valid permutation of `0..n` — every index
/// appears exactly once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    /// `perm[new] = old`
    perm: Vec<usize>,
}

impl Permutation {
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    /// Build from a raw vector.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `perm` is not a valid
    ///   permutation of `0..perm.len()`.
    pub fn new(perm: Vec<usize>) -> Result<Self> {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in &perm {
            if p >= n {
                return Err(SparseError::RowOutOfRange { row: p, nrows: n });
            }
            if seen[p] {
                return Err(SparseError::IndexOutOfBounds { row: p, col: p });
            }
            seen[p] = true;
        }
        Ok(Self { perm })
    }

    /// Identity permutation of size `n` — no reordering.
    pub fn identity(n: usize) -> Self {
        Self { perm: (0..n).collect() }
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Length of the permutation.
    #[inline]
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Returns `true` if the permutation has length 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// `perm[new] = old`: the old index that maps to `new`.
    #[inline]
    pub fn old_index(&self, new: usize) -> usize {
        self.perm[new]
    }

    /// Raw slice: `perm[new] = old`.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.perm
    }

    /// Compute and return the inverse: `inv[old] = new`.
    ///
    /// O(n).  If you need the inverse repeatedly, store it separately.
    pub fn inverse(&self) -> Self {
        let n = self.perm.len();
        let mut inv = vec![0usize; n];
        for (new, &old) in self.perm.iter().enumerate() {
            inv[old] = new;
        }
        Self { perm: inv }
    }

    /// Returns `true` if this is the identity permutation.
    pub fn is_identity(&self) -> bool {
        self.perm.iter().enumerate().all(|(i, &p)| i == p)
    }

    // -----------------------------------------------------------------
    // Applying the permutation
    // -----------------------------------------------------------------

    /// Apply the permutation to a slice: `out[i] = v[perm[i]]`.
    ///
    /// # Panics
    /// Panics if `v.len() != self.len()`.
    pub fn apply_to_slice<T: Copy>(&self, v: &[T]) -> Vec<T> {
        assert_eq!(v.len(), self.perm.len());
        self.perm.iter().map(|&old| v[old]).collect()
    }

    /// Apply the inverse permutation to a slice: `out[perm[i]] = v[i]`.
    ///
    /// Equivalent to `self.inverse().apply_to_slice(v)` but without
    /// allocating the inverse permutation.
    ///
    /// # Panics
    /// Panics if `v.len() != self.len()`.
    pub fn apply_inverse_to_slice<T: Copy + Default>(&self, v: &[T]) -> Vec<T> {
        assert_eq!(v.len(), self.perm.len());
        let mut out = vec![T::default(); self.perm.len()];
        for (i, &old) in self.perm.iter().enumerate() {
            out[old] = v[i];
        }
        out
    }

    /// Permute a `SymCsrMatrix`: `K_permuted[i,j] = K[perm[i], perm[j]]`.
    ///
    /// This rebuilds the matrix with reordered rows and columns.
    /// The result has the same values but potentially much better
    /// fill during Cholesky factorization.
    ///
    /// # Errors
    /// - [`SparseError::DimensionMismatch`] if `perm.len() != k.n`
    pub fn permute_sym(&self, k: &SymCsrMatrix) -> Result<SymCsrMatrix> {
        let n = k.n;
        if self.perm.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                got: self.perm.len(),
            });
        }

        // Build the permuted upper-triangle pattern.
        //
        // For each stored entry (old_i, old_j) with old_j >= old_i:
        //   new_i = inv[old_i],  new_j = inv[old_j]
        //   store as (min(new_i,new_j), max(new_i,new_j)) in the upper triangle
        let inv = self.inverse();
        let inv_perm = inv.perm;

        // Collect (new_row, new_col, val) triples for all upper entries
        use std::collections::BTreeMap;
        // key: (row, col) in permuted space, value: accumulated value
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();

        for old_i in 0..n {
            let end   = k.row_ptr()[old_i + 1];
            let start = k.row_ptr()[old_i];
            for idx in start..end {
                let old_j = k.col_idx()[idx];
                let val   = k.values()[idx];

                let new_i = inv_perm[old_i];
                let new_j = inv_perm[old_j];

                // Store in upper triangle: row <= col
                let (r, c) = if new_i <= new_j {
                    (new_i, new_j)
                } else {
                    (new_j, new_i)
                };
                *entries.entry((r, c)).or_insert(0.0) += val;
            }
        }

        // Build pattern from entries (BTreeMap is sorted so rows are in order)
        let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(r, c) in entries.keys() {
            pattern[r].push(c);
        }

        let mut permuted = SymCsrMatrix::from_pattern(n, &pattern)?;
        for ((r, c), val) in entries {
            permuted.add_value(r, c, val)?;
        }

        Ok(permuted)
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_identity() {
        let p = Permutation::identity(4);
        assert!(p.is_identity());
        assert_eq!(p.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn inverse_roundtrip() {
        let p = Permutation::new(vec![2, 0, 3, 1]).unwrap();
        let pinv = p.inverse();
        // p[pinv[i]] == i for all i
        for i in 0..4 {
            assert_eq!(p.old_index(pinv.old_index(i)), i);
        }
    }

    #[test]
    fn new_rejects_invalid_permutation() {
        // duplicate index
        assert!(Permutation::new(vec![0, 0, 2]).is_err());
        // out of range
        assert!(Permutation::new(vec![0, 1, 5]).is_err());
    }

    #[test]
    fn apply_to_slice() {
        let p = Permutation::new(vec![2, 0, 1]).unwrap();
        let v = vec![10.0_f64, 20.0, 30.0];
        // out[0] = v[p[0]] = v[2] = 30
        // out[1] = v[p[1]] = v[0] = 10
        // out[2] = v[p[2]] = v[1] = 20
        assert_eq!(p.apply_to_slice(&v), vec![30.0, 10.0, 20.0]);
    }

    #[test]
    fn apply_inverse_to_slice() {
        let p = Permutation::new(vec![2, 0, 1]).unwrap();
        let v = vec![10.0_f64, 20.0, 30.0];
        let forward = p.apply_to_slice(&v);
        let back    = p.apply_inverse_to_slice(&forward);
        assert_eq!(back, v);
    }

    #[test]
    fn permute_sym_identity_unchanged() {
        let pattern = vec![vec![0usize, 1], vec![1, 2], vec![2]];
        let mut m = SymCsrMatrix::from_pattern(3, &pattern).unwrap();
        m.set_value(0, 0,  4.0).unwrap();
        m.set_value(0, 1, -1.0).unwrap();
        m.set_value(1, 1,  4.0).unwrap();
        m.set_value(1, 2, -1.0).unwrap();
        m.set_value(2, 2,  4.0).unwrap();

        let p = Permutation::identity(3);
        let permuted = p.permute_sym(&m).unwrap();
        permuted.validate().unwrap();

        // Values should be identical
        assert_eq!(permuted.get(0, 0).unwrap(),  4.0);
        assert_eq!(permuted.get(0, 1).unwrap(), -1.0);
        assert_eq!(permuted.get(1, 1).unwrap(),  4.0);
        assert_eq!(permuted.get(1, 2).unwrap(), -1.0);
        assert_eq!(permuted.get(2, 2).unwrap(),  4.0);
    }

    #[test]
    fn permute_sym_reversal() {
        // Permutation [2,1,0] reverses node order
        // Original tridiag:
        //   [ 4 -1  0]
        //   [-1  4 -1]
        //   [ 0 -1  4]
        // After reversing: same structure (tridiag is symmetric under reversal)
        let pattern = vec![vec![0usize, 1], vec![1, 2], vec![2]];
        let mut m = SymCsrMatrix::from_pattern(3, &pattern).unwrap();
        m.set_value(0, 0,  4.0).unwrap();
        m.set_value(0, 1, -1.0).unwrap();
        m.set_value(1, 1,  4.0).unwrap();
        m.set_value(1, 2, -1.0).unwrap();
        m.set_value(2, 2,  4.0).unwrap();

        let p = Permutation::new(vec![2usize, 1, 0]).unwrap();
        let permuted = p.permute_sym(&m).unwrap();
        permuted.validate().unwrap();

        // Diagonal should still be 4 everywhere
        assert_eq!(permuted.get(0, 0).unwrap(), 4.0);
        assert_eq!(permuted.get(1, 1).unwrap(), 4.0);
        assert_eq!(permuted.get(2, 2).unwrap(), 4.0);
        // Off-diagonal should still be -1
        assert_eq!(permuted.get(0, 1).unwrap(), -1.0);
        assert_eq!(permuted.get(1, 2).unwrap(), -1.0);
    }

    #[test]
    fn permute_sym_matvec_invariant() {
        // K_perm * (P_inv * x) == P_inv * (K * x)  for any x
        // Equivalently: P * K_perm * P^T == K
        // We verify by checking matvec on a known vector
        let pattern = vec![vec![0usize, 1], vec![1, 2], vec![2]];
        let mut m = SymCsrMatrix::from_pattern(3, &pattern).unwrap();
        m.set_value(0, 0,  4.0).unwrap();
        m.set_value(0, 1, -1.0).unwrap();
        m.set_value(1, 1,  4.0).unwrap();
        m.set_value(1, 2, -1.0).unwrap();
        m.set_value(2, 2,  4.0).unwrap();

        let p    = Permutation::new(vec![2usize, 0, 1]).unwrap();
        let pinv = p.inverse();
        let km   = p.permute_sym(&m).unwrap();

        let x     = vec![1.0_f64, 2.0, 3.0];
        
        // y2 = P_inv * (K_perm * (P * x))   should equal y1
        let x_tilde  = pinv.apply_to_slice(&x);       // x̃[k] = x[p_inv[k]]
        let kx_tilde = m.matvec(&x_tilde).unwrap();   // K * x̃
        let y_perm   = km.matvec(&x).unwrap();         // K_perm * x
        let recovered = p.apply_to_slice(&kx_tilde);  // (K * x̃)[p[i]]

        for (a, b) in y_perm.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-11, "y_perm={a} recovered={b}");
        }
    }
}
