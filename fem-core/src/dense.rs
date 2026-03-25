//! Fixed-size dense matrix operations for element-level computations.
//!
//! Element stiffness matrices are small and fixed-size: a 2D truss element
//! produces a 4×4 matrix; a 2D beam element produces a 6×6.  These are
//! computed thousands of times per analysis and must be fast.
//!
//! We use plain `[[f64; N]; N]` arrays (row-major, stack-allocated) with no
//! dependencies.  They are trivially convertible to `&[f64]` for `scatter_add`
//! via [`mat_as_slice`].
//!
//! ## Provided operations
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`matmul`] | `C = A * B` — general `N×N` multiply |
//! | [`mat_transpose_mul`] | `C = Aᵀ * B` — used in `Tᵀ K` |
//! | [`transform_stiffness`] | `Kg = Tᵀ K T` — full coordinate transform |
//! | [`mat_as_slice`] | Flatten `[[f64; N]; N]` to `&[f64]` (zero-copy) |
//! | [`mat_zero`] | Return zeroed matrix |
//! | [`mat_add_assign`] | In-place `A += B` |
//!
//! ## Why not `nalgebra`?
//!
//! `nalgebra` is excellent but is a substantial dependency.  For the
//! fixed small sizes we use (4×4, 6×6), plain arrays are:
//! - Faster to compile.
//! - Trivially `repr`-compatible with `&[f64]` for `scatter_add`.
//! - Sufficient — these matrices are never inverted or decomposed here.
//!
//! If eigenvalue decomposition or matrix inversion is needed (e.g. for
//! section analysis), add `nalgebra` as an optional dependency then.

// -----------------------------------------------------------------
// Flatten helpers
// -----------------------------------------------------------------

/// Flatten a row-major `N×N` matrix to a `&[f64]` of length `N*N`.
///
/// Zero-copy: the slice points into the same stack memory as the array.
/// This is what you pass to `CsrMatrix::scatter_add`.
///
/// # Example
/// ```
/// use fem_core::dense::mat_as_slice;
/// let ke = [[1.0_f64, 0.0], [0.0, 1.0]];
/// let s = mat_as_slice(&ke);
/// assert_eq!(s, &[1.0, 0.0, 0.0, 1.0]);
/// ```
#[inline]
pub fn mat_as_slice<const N: usize>(m: &[[f64; N]; N]) -> &[f64] {
    // SAFETY: [[f64; N]; N] is a contiguous block of N*N f64 values in
    // row-major order (C layout).  The total size is N*N*size_of::<f64>().
    // We produce a shared reference for the same lifetime as `m`.
    unsafe {
        std::slice::from_raw_parts(m.as_ptr() as *const f64, N * N)
    }
}

/// Return a zeroed `N×N` matrix.
#[inline]
pub fn mat_zero<const N: usize>() -> [[f64; N]; N] {
    [[0.0; N]; N]
}

// -----------------------------------------------------------------
// Arithmetic
// -----------------------------------------------------------------

/// In-place `A += B`.
#[inline]
pub fn mat_add_assign<const N: usize>(a: &mut [[f64; N]; N], b: &[[f64; N]; N]) {
    for i in 0..N {
        for j in 0..N {
            a[i][j] += b[i][j];
        }
    }
}

/// Scale in-place: `A *= s`.
#[inline]
pub fn mat_scale<const N: usize>(a: &mut [[f64; N]; N], s: f64) {
    for i in 0..N {
        for j in 0..N {
            a[i][j] *= s;
        }
    }
}

/// `C = A * B` — standard row-major `N×N` matrix multiply.
///
/// For N = 4 (truss) or N = 6 (beam) this is only 64 or 216 multiply-adds,
/// which the compiler fully unrolls and vectorises.
#[inline]
pub fn matmul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = mat_zero::<N>();
    for i in 0..N {
        for k in 0..N {
            let aik = a[i][k];
            if aik == 0.0 { continue; } // skip structural zeros in T
            for j in 0..N {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// `C = Aᵀ * B` — multiply the transpose of `A` by `B`.
///
/// Equivalent to `matmul(&transpose(A), B)` but avoids materialising the
/// transpose.  Used in the first step of `Tᵀ K T`.
#[inline]
pub fn mat_transpose_mul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = mat_zero::<N>();
    for k in 0..N {  // k indexes columns of Aᵀ = rows of A
        for i in 0..N {
            let aki = a[k][i]; // Aᵀ[i,k] = A[k,i]
            if aki == 0.0 { continue; }
            for j in 0..N {
                c[i][j] += aki * b[k][j];
            }
        }
    }
    c
}

/// `Kg = Tᵀ K T` — coordinate transform of a stiffness matrix.
///
/// This is the full congruence transformation that rotates an element
/// stiffness matrix from local to global coordinates.  For a 2D beam element,
/// `N = 6`.
///
/// Computed in two passes to avoid a temporary `N×N` allocation:
/// ```text
/// step 1:  W = Tᵀ K        (N² multiplications)
/// step 2:  Kg = W T         (N² multiplications)
/// ```
/// Total: `2 N³` multiply-adds.  For N=6: 432 operations.
#[inline]
pub fn transform_stiffness<const N: usize>(
    ke_local: &[[f64; N]; N],
    t: &[[f64; N]; N],
) -> [[f64; N]; N] {
    // step 1: W = Tᵀ * Ke_local
    let w = mat_transpose_mul(t, ke_local);
    // step 2: Kg = W * T
    matmul(&w, t)
}

/// Transpose: `B[i][j] = A[j][i]`.
///
/// Useful for building the full T from its components in tests.
#[inline]
pub fn transpose<const N: usize>(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut b = mat_zero::<N>();
    for i in 0..N {
        for j in 0..N {
            b[i][j] = a[j][i];
        }
    }
    b
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N], tol: f64) {
        for i in 0..N {
            for j in 0..N {
                let diff = (a[i][j] - b[i][j]).abs();
                assert!(
                    diff <= tol,
                    "a[{i}][{j}]={} b[{i}][{j}]={}  diff={:.2e}",
                    a[i][j], b[i][j], diff
                );
            }
        }
    }

    // ---- mat_as_slice ----

    #[test]
    fn mat_as_slice_2x2() {
        let m = [[1.0_f64, 2.0], [3.0, 4.0]];
        assert_eq!(mat_as_slice(&m), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mat_as_slice_zero_copy() {
        let m = [[1.0_f64, 2.0], [3.0, 4.0]];
        let s = mat_as_slice(&m);
        assert_eq!(s.as_ptr(), m.as_ptr() as *const f64);
    }

    #[test]
    fn mat_as_slice_4x4_length() {
        let m = mat_zero::<4>();
        assert_eq!(mat_as_slice(&m).len(), 16);
    }

    #[test]
    fn mat_as_slice_6x6_length() {
        let m = mat_zero::<6>();
        assert_eq!(mat_as_slice(&m).len(), 36);
    }

    // ---- matmul ----

    #[test]
    fn matmul_identity_2x2() {
        let id: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
        let a: [[f64; 2]; 2] = [[3.0, 4.0], [5.0, 6.0]];
        approx_eq(&matmul(&a, &id), &a, 1e-14);
        approx_eq(&matmul(&id, &a), &a, 1e-14);
    }

    #[test]
    fn matmul_2x2_known() {
        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        let a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
        let b: [[f64; 2]; 2] = [[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-13);
        assert!((c[0][1] - 22.0).abs() < 1e-13);
        assert!((c[1][0] - 43.0).abs() < 1e-13);
        assert!((c[1][1] - 50.0).abs() < 1e-13);
    }

    #[test]
    fn matmul_4x4_identity() {
        let id: [[f64; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let k: [[f64; 4]; 4] = [
            [ 1.0,  0.0, -1.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0],
            [-1.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.0],
        ];
        approx_eq(&matmul(&k, &id), &k, 1e-14);
    }

    // ---- mat_transpose_mul ----

    #[test]
    fn mat_transpose_mul_equals_explicit_transpose() {
        let a: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let b: [[f64; 3]; 3] = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
        let at = transpose(&a);
        let expected = matmul(&at, &b);
        let got = mat_transpose_mul(&a, &b);
        approx_eq(&got, &expected, 1e-13);
    }

    #[test]
    fn mat_transpose_mul_identity() {
        let id: [[f64; 3]; 3] = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let a: [[f64; 3]; 3] = [[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]];
        // Iᵀ * A = A
        approx_eq(&mat_transpose_mul(&id, &a), &a, 1e-14);
    }

    // ---- transform_stiffness ----

    #[test]
    fn transform_stiffness_identity_2x2() {
        let id: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
        let k: [[f64; 2]; 2] = [[4.0, -1.0], [-1.0, 4.0]];
        // Iᵀ K I = K
        approx_eq(&transform_stiffness(&k, &id), &k, 1e-14);
    }

    #[test]
    fn transform_stiffness_rotation_preserves_eigenvalues() {
        // Rotating a symmetric matrix by an orthogonal T preserves eigenvalues.
        // For a 2×2 rotation by 45°: Tr(K) must be preserved.
        use std::f64::consts::FRAC_1_SQRT_2 as S;
        let t: [[f64; 2]; 2] = [[S, -S], [S, S]];
        let k: [[f64; 2]; 2] = [[3.0, 1.0], [1.0, 2.0]];
        let kg = transform_stiffness(&k, &t);
        // Trace is preserved under similarity transform
        let trace_k  = k[0][0]  + k[1][1];
        let trace_kg = kg[0][0] + kg[1][1];
        assert!((trace_k - trace_kg).abs() < 1e-13,
                "trace_k={trace_k} trace_kg={trace_kg}");
    }

    #[test]
    fn transform_stiffness_orthogonal_t_preserves_symmetry() {
        // For orthogonal T, Tᵀ K T should be symmetric if K is symmetric.
        use std::f64::consts::FRAC_1_SQRT_2 as S;
        let t: [[f64; 2]; 2] = [[S, -S], [S, S]];
        let k: [[f64; 2]; 2] = [[4.0, 1.0], [1.0, 3.0]];
        let kg = transform_stiffness(&k, &t);
        assert!((kg[0][1] - kg[1][0]).abs() < 1e-13,
                "Kg not symmetric: Kg[0][1]={} Kg[1][0]={}", kg[0][1], kg[1][0]);
    }

    #[test]
    fn transform_stiffness_equals_manual_triple_product() {
        // Verify Tᵀ K T == matmul(transpose(T), matmul(K, T))
        let t: [[f64; 3]; 3] = [[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]];
        let k: [[f64; 3]; 3] = [[5.0, 1.0, 0.0], [1.0, 3.0, 2.0], [0.0, 2.0, 4.0]];
        let manual = matmul(&matmul(&transpose(&t), &k), &t);
        let fast   = transform_stiffness(&k, &t);
        approx_eq(&manual, &fast, 1e-12);
    }

    // ---- mat_add_assign / mat_scale ----

    #[test]
    fn mat_add_assign_2x2() {
        let mut a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
        let b: [[f64; 2]; 2] = [[10.0, 0.0], [0.0, 10.0]];
        mat_add_assign(&mut a, &b);
        assert_eq!(a, [[11.0, 2.0], [3.0, 14.0]]);
    }

    #[test]
    fn mat_scale_2x2() {
        let mut a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
        mat_scale(&mut a, 2.0);
        assert_eq!(a, [[2.0, 4.0], [6.0, 8.0]]);
    }

    // ---- 2D truss stiffness transform (concrete example) ----
    //
    // A 2D truss element at angle θ has local stiffness (in local x-y):
    //   ke_local = EA/L * [[1,0,-1,0],[0,0,0,0],[-1,0,1,0],[0,0,0,0]]
    // and transform T such that Kg = Tᵀ ke_local T.
    // For θ=0 (horizontal), T = I and Kg = ke_local.

    #[test]
    fn truss_2d_horizontal_transform_unchanged() {
        // EA/L = 1 for simplicity
        let ke_local: [[f64; 4]; 4] = [
            [ 1.0, 0.0, -1.0, 0.0],
            [ 0.0, 0.0,  0.0, 0.0],
            [-1.0, 0.0,  1.0, 0.0],
            [ 0.0, 0.0,  0.0, 0.0],
        ];
        // θ=0: c=1, s=0
        let t: [[f64; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let kg = transform_stiffness(&ke_local, &t);
        approx_eq(&kg, &ke_local, 1e-14);
    }

    #[test]
    fn truss_2d_vertical_transform_swaps_dofs() {
        // θ=90°: c=0, s=1
        // Local x-axis is vertical, so horizontal DOFs ↔ vertical DOFs
        let ke_local: [[f64; 4]; 4] = [
            [ 1.0, 0.0, -1.0, 0.0],
            [ 0.0, 0.0,  0.0, 0.0],
            [-1.0, 0.0,  1.0, 0.0],
            [ 0.0, 0.0,  0.0, 0.0],
        ];
        // T for θ=90°: maps global (x,y) to local (x_L, y_L)
        // x_L =  0*x + 1*y,  y_L = -1*x + 0*y
        let c = 0.0_f64;
        let s = 1.0_f64;
        let t: [[f64; 4]; 4] = [
            [ c,  s, 0.0, 0.0],
            [-s,  c, 0.0, 0.0],
            [0.0, 0.0,  c,  s],
            [0.0, 0.0, -s,  c],
        ];
        let kg = transform_stiffness(&ke_local, &t);
        // After 90° rotation, stiffness is in y-direction:
        // Kg should have 1.0 in [1,1], -1.0 in [1,3], -1.0 in [3,1], 1.0 in [3,3]
        assert!((kg[1][1] -  1.0).abs() < 1e-13, "kg[1][1]={}", kg[1][1]);
        assert!((kg[1][3] - -1.0).abs() < 1e-13, "kg[1][3]={}", kg[1][3]);
        assert!((kg[3][1] - -1.0).abs() < 1e-13, "kg[3][1]={}", kg[3][1]);
        assert!((kg[3][3] -  1.0).abs() < 1e-13, "kg[3][3]={}", kg[3][3]);
        // x-direction should be zero stiffness
        assert!((kg[0][0]).abs() < 1e-13, "kg[0][0]={}", kg[0][0]);
    }
}