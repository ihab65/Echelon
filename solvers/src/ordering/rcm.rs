//! Reverse Cuthill-McKee (RCM) ordering for fill reduction.
//!
//! ## Purpose
//!
//! Without fill reduction, Cholesky factorization of a sparse matrix can
//! produce a factor `L` with many more non-zeros than the original `K`.
//! RCM reorders the DOFs to reduce this fill, making both factorization
//! and triangular solves much faster.
//!
//! ## Algorithm sketch
//!
//! 1. Find a peripheral node (BFS from a low-degree starting node, take
//!    the last node found — this is the "pseudo-peripheral" node).
//! 2. BFS from the peripheral node, adding nodes in level-set order
//!    (within each level, sort by ascending degree).
//! 3. Reverse the resulting ordering — this is the RCM permutation.
//!
//! ## Output
//!
//! Returns a permutation vector `p` such that `p[new_index] = old_index`.
//! To apply the permutation to K: `K_permuted[i,j] = K[p[i], p[j]]`.

/// Compute the RCM permutation for a symmetric sparse matrix.
///
/// `adjacency[i]` is the list of neighbors of node `i` (excluding `i`
/// itself).  This is just the off-diagonal column indices for row `i`
/// in either CSR or SymCSR format.
///
/// Returns `p` where `p[new_index] = old_index`.
///
/// # Panics
/// Panics if `adjacency.len() == 0`.
pub fn rcm_ordering(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    assert!(n > 0, "adjacency list must be non-empty");

    // TODO: implement
    // For now return identity permutation (correct but no fill reduction)
    (0..n).collect()
}

/// Apply a permutation to produce the reordered row/column index mapping.
///
/// Given permutation `p` (new→old), returns `p_inv` (old→new).
pub fn invert_permutation(p: &[usize]) -> Vec<usize> {
    let n = p.len();
    let mut inv = vec![0usize; n];
    for (new_idx, &old_idx) in p.iter().enumerate() {
        inv[old_idx] = new_idx;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_permutation_when_unimplemented() {
        let adj = vec![vec![1usize], vec![0, 2], vec![1]];
        let p = rcm_ordering(&adj);
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn invert_permutation_roundtrip() {
        let p    = vec![2usize, 0, 1];
        let pinv = invert_permutation(&p);
        for i in 0..3 {
            assert_eq!(pinv[p[i]], i);
        }
    }
}