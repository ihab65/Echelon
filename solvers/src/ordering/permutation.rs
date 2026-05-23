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

use sparse::{SparseScalar, SymCsrMatrix};
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
    pub fn permute_sym<T>(&self, k: &SymCsrMatrix<T>) -> Result<SymCsrMatrix<T>> where T: SparseScalar {
        self.permute_sym_with_map(k).map(|(k_perm, _)| k_perm)
    }

    /// Permute a symmetric matrix and compute the exact value-index mapping.
    ///
    /// Returns a tuple `(k_perm, index_map)` where:
    /// - `k_perm` is the permuted `SymCsrMatrix`.
    /// - `index_map` is a vector of length `k.values().len()`.
    ///
    /// For any non-zero value at index `i` in the original matrix `k.values()`,
    /// it will be placed at index `index_map[i]` in `k_perm.values()`.
    /// This map is perfectly constant as long as the sparsity pattern of `k`
    /// does not change, enabling zero-allocation permutations via [`Self::permute_sym_into`].
    pub fn permute_sym_with_map<T>(&self, k: &SymCsrMatrix<T>) -> Result<(SymCsrMatrix<T>, Vec<usize>)> 
    where 
        T: SparseScalar 
    {
        let n = k.n;
        if self.perm.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                got: self.perm.len(),
            });
        }

        let inv_perm = self.inverse().perm;

        use std::collections::BTreeMap;
        // Output contains (accumulated_value, original_k_values_index)
        let mut entries: BTreeMap<(usize, usize), (T, usize)> = BTreeMap::new();

        for old_i in 0..n {
            let start = k.row_ptr()[old_i];
            let end   = k.row_ptr()[old_i + 1];
            for idx in start..end {
                let old_j = k.col_idx()[idx];
                let val   = k.values()[idx];

                let new_i = inv_perm[old_i];
                let new_j = inv_perm[old_j];

                let (r, c) = if new_i <= new_j {
                    (new_i, new_j)
                } else {
                    (new_j, new_i)
                };

                // K contains no duplicate coordinates (enforced by SymCsrMatrix)
                let prev = entries.insert((r, c), (val, idx));
                debug_assert!(prev.is_none(), "SymCsrMatrix must not contain duplicate coordinates");
            }
        }

        // Build pattern
        let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(r, c) in entries.keys() {
            pattern[r].push(c);
        }

        let mut permuted = SymCsrMatrix::from_pattern(n, &pattern)?;
        let mut index_map = vec![0; k.values().len()];

        for ((r, c), (val, old_idx)) in entries {
            let new_idx = permuted.add_value_and_return_index(r, c, val)?;
            index_map[old_idx] = new_idx;
        }

        Ok((permuted, index_map))
    }

    /// Zero-allocation symmetric permutation using a pre-computed index map.
    ///
    /// Copies values from `k` into the pre-allocated `k_perm` matrix using
    /// the constant index mapping generated by [`Self::permute_sym_with_map`].
    ///
    /// # Panics
    /// Panics in debug mode if `k.values().len() != index_map.len()`.
    pub fn permute_sym_into<T>(&self, k: &SymCsrMatrix<T>, k_perm: &mut SymCsrMatrix<T>, index_map: &[usize]) 
    where 
        T: SparseScalar 
    {
        debug_assert_eq!(
            k.values().len(), index_map.len(),
            "index_map length must match original k values length"
        );
        debug_assert_eq!(
            k.values().len(), k_perm.values().len(),
            "permuted k values length must match original k values length"
        );

        let k_vals = k.values();
        let k_perm_vals = k_perm.values_mut();

        // O(nnz) direct array assignments — zero allocations, zero lookups
        for i in 0..k_vals.len() {
            k_perm_vals[index_map[i]] = k_vals[i];
        }
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
