//! Shift-invert Lanczos eigensolver for the generalised eigenvalue problem.
//!
//! [`LanczosEigenSolver`] implements [`EigenSolver`] to compute the `k` lowest
//! eigenpairs of `K φ = ω² M φ` using a shift-invert Lanczos iteration with
//! implicit QL convergence.
//!
//! ## Algorithm overview
//!
//! 1. **Prepare** — factorize `K` with [`LdltSolver`] (zero shift, so the
//!    shift-invert operator is `K⁻¹ M`).
//!
//! 2. **Lanczos loop** — build an `m = min(N, 2k)` Krylov subspace.
//!    At each step:
//!    - Apply `K⁻¹ M` to the current Lanczos vector.
//!    - Compute the Rayleigh quotient (`alpha_j`).
//!    - Orthogonalise (three-term recurrence).
//!    - Compute the sub-diagonal coefficient (`beta_j`) via the M-norm.
//!    - Advance to the next Lanczos vector.
//!    The Lanczos basis `V` (N × m, column-major) accumulates all vectors.
//!
//! 3. **Solve tridiagonal** — [`implicit_ql`] diagonalises the `m × m`
//!    tridiagonal `T_m` in-place, accumulating orthogonal transformations
//!    directly into `V` to give global mode shapes.
//!
//! 4. **Post-process** — sort by ascending eigenvalue, truncate to `k` modes,
//!    M-normalise each shape, and return an [`EigenResult`].
//!
//! ## Zero-allocation loop invariant
//!
//! All working buffers are allocated **once** before the loop in `solve_modes`.
//! The loop body contains no `vec![]` or other heap allocations except the
//! single unavoidable `rhs_buf.copy_from_slice` required by the
//! `LinearSolver::solve(&f, &mut u)` API signature, which operates on
//! pre-allocated stack buffers.

use sparse::SymCsrMatrix;
use sparse::MatvecWorkspace;

use crate::eigen::{EigenResult, EigenSolver};
use crate::error::{Result, SolverError};
use crate::linear::{LdltSolver, LinearSolver};

// ─────────────────────────────────────────────────────────────────────────────
// Public struct
// ─────────────────────────────────────────────────────────────────────────────

/// Shift-invert Lanczos eigensolver for sparse structural dynamics.
///
/// Solves `K φ = ω² M φ` for the `n_modes` lowest eigenpairs using a
/// Lanczos iteration with shift-invert (`K⁻¹ M`) and implicit QL convergence.
///
/// # Usage
///
/// ```rust,ignore
/// use solvers::eigen::{EigenSolver, LanczosEigenSolver};
///
/// let mut solver = LanczosEigenSolver::new();
/// let result = solver.compute(&k, &m, 5)?;
/// println!("T₁ = {:.4} s", result.periods()[0]);
/// ```
pub struct LanczosEigenSolver {
    /// Factorized stiffness matrix — provides `K⁻¹` via triangular solve.
    ldlt:  Option<LdltSolver<f64>>,
    /// Mass matrix, retained for M-norm normalisation after the QL step.
    m_mat: Option<SymCsrMatrix<f64>>,
    /// Total number of degrees of freedom `N`.
    n_dof: usize,
}

impl LanczosEigenSolver {
    /// Create a new, un-prepared solver.
    pub fn new() -> Self {
        Self { ldlt: None, m_mat: None, n_dof: 0 }
    }
}

impl Default for LanczosEigenSolver {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────────────────────
// EigenSolver implementation
// ─────────────────────────────────────────────────────────────────────────────

impl EigenSolver for LanczosEigenSolver {
    /// Factorize `K` so that the Lanczos loop can apply `K⁻¹` cheaply.
    ///
    /// Retains `M` for the subsequent M-norm normalisation step.
    ///
    /// # Errors
    /// - [`SolverError::NotPositiveDefinite`] if `K` is singular or indefinite
    ///   in a way that causes the LDLᵀ factorization to encounter a zero pivot.
    fn prepare(&mut self, k: &SymCsrMatrix<f64>, m: &SymCsrMatrix<f64>) -> Result<()> {
        debug_assert_eq!(k.n, m.n, "K and M must have the same dimension");

        let mut ldlt = LdltSolver::new();
        ldlt.analyze_and_factorize(k)?;

        self.ldlt  = Some(ldlt);
        self.m_mat = Some(m.clone());
        self.n_dof = k.n;
        Ok(())
    }

    /// Run the Lanczos iteration and return the `n_modes` lowest eigenpairs.
    ///
    /// # Errors
    /// - [`SolverError::NotAnalyzed`] if `prepare` has not been called.
    /// - [`SolverError::NotPositiveDefinite`] (repurposed as convergence error)
    ///   if the implicit QL algorithm fails to converge within 30 sweeps.
    fn solve_modes(&mut self, n_modes: usize) -> Result<EigenResult> {
        // ── Validate state ────────────────────────────────────────────────────
        let ldlt = self.ldlt.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let m    = self.m_mat.as_ref().ok_or(SolverError::NotAnalyzed)?;
        let n    = self.n_dof;

        if n_modes == 0 {
            return Ok(EigenResult { eigenvalues: vec![], mode_shapes: vec![] });
        }

        let n_modes_capped = n_modes.min(n);

        // ── Subspace size: m_sub = min(N, 2k), at least k ────────────────────
        let m_sub = (2 * n_modes_capped).min(n);

        // ── Allocate all working buffers ONCE before the loop ─────────────────
        //
        // V: N × m_sub Lanczos basis, column-major.
        //    Column j occupies v_flat[j*n .. (j+1)*n].
        let mut v_flat  = vec![0.0_f64; n * m_sub];
        let mut alpha   = vec![0.0_f64; m_sub];  // tridiagonal diagonal T_m
        let mut beta    = vec![0.0_f64; m_sub];  // tridiagonal sub-diagonal

        // Working vectors — reused every iteration, no allocation in the loop.
        let mut q_cur   = vec![0.0_f64; n]; // current Lanczos vector q_j
        let mut q_prev  = vec![0.0_f64; n]; // previous vector q_{j-1}
        let mut rhs_buf = vec![0.0_f64; n]; // M q_j (input to K⁻¹ solve)
        let mut kq      = vec![0.0_f64; n]; // K⁻¹ M q_j (output of K⁻¹ solve)
        let mut m_ws    = MatvecWorkspace::new(n); // M matvec output buffer

        // ── Starting vector: q_0 = e_0, M-normalised ─────────────────────────
        q_cur[0] = 1.0;
        m.matvec_into(&q_cur, &mut m_ws).map_err(SolverError::Sparse)?;
        let mnorm = dot(&q_cur, m_ws.as_slice()).max(0.0).sqrt();
        for qi in q_cur.iter_mut() { *qi /= mnorm; }

        // Store q_0 as column 0 of V.
        v_flat[0..n].copy_from_slice(&q_cur);

        // ── Lanczos loop — ZERO allocation inside this block ──────────────────
        for j in 0..m_sub {
            // Step 1: rhs = M q_j
            m.matvec_into(&q_cur, &mut m_ws).map_err(SolverError::Sparse)?;
            rhs_buf.copy_from_slice(m_ws.as_slice());

            // Step 2: kq = K⁻¹ (M q_j)  — shift-invert
            ldlt.solve(&rhs_buf, &mut kq)?;

            // Step 3: alpha_j = (M q_j)ᵀ (K⁻¹ M q_j)
            //         = rhs_buf · kq  (Rayleigh quotient in the M-inner-product)
            alpha[j] = dot(&rhs_buf, &kq);

            // Step 4: orthogonalise  w ← kq - alpha_j * q_j - beta_{j-1} * q_{j-1}
            let b_prev = if j > 0 { beta[j - 1] } else { 0.0 };
            for i in 0..n {
                kq[i] -= alpha[j] * q_cur[i] + b_prev * q_prev[i];
            }
            // `kq` now holds the un-normalised next Lanczos vector `w`.

            // Step 5: beta_j = ‖w‖_M
            m.matvec_into(&kq, &mut m_ws).map_err(SolverError::Sparse)?;
            let b_j = dot(&kq, m_ws.as_slice()).max(0.0).sqrt();
            beta[j] = b_j;

            // Step 6: advance — q_{j+1} = w / beta_j
            if j + 1 < m_sub {
                q_prev.copy_from_slice(&q_cur);

                if b_j > 1e-14 {
                    let inv = 1.0 / b_j;
                    for i in 0..n { q_cur[i] = kq[i] * inv; }
                } else {
                    // Lucky breakdown: restart with a fresh orthogonal vector.
                    restart_vector(&mut q_cur, &v_flat, j + 1, n, m, &mut m_ws);
                    beta[j] = 0.0;
                }

                // Store q_{j+1} as column j+1 of V.
                let col_start = (j + 1) * n;
                v_flat[col_start..col_start + n].copy_from_slice(&q_cur);
            }
        }
        // ── End of Lanczos loop ───────────────────────────────────────────────

        // ── Implicit QL on T_m, rotating eigenvectors into V ─────────────────
        implicit_ql(&mut alpha, &mut beta, &mut v_flat, n, m_sub)
            .map_err(|_| SolverError::NotPositiveDefinite { index: 0, value: 0.0 })?;

        // ── Sort eigenpairs by ascending eigenvalue ───────────────────────────
        let mut order: Vec<usize> = (0..m_sub).collect();
        order.sort_unstable_by(|&a, &b| {
            alpha[b].partial_cmp(&alpha[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── Truncate, M-normalise, and construct EigenResult ──────────────────
        let mut eigenvalues = Vec::with_capacity(n_modes_capped);
        let mut mode_shapes = Vec::with_capacity(n_modes_capped);

        for &col in order.iter().take(n_modes_capped) {
            let mu = alpha[col];

            // lambda = 1.0 / mu
            let omega_sq = 1.0 / mu;

            // Extract the global mode shape from column `col` of V.
            let col_start = col * n;
            let mut phi: Vec<f64> = v_flat[col_start..col_start + n].to_vec();

            // M-normalise: phi /= sqrt(phi^T M phi)
            m.matvec_into(&phi, &mut m_ws).map_err(SolverError::Sparse)?;
            let mnorm = dot(&phi, m_ws.as_slice()).max(0.0).sqrt();
            if mnorm > 1e-14 {
                let inv = 1.0 / mnorm;
                for pi in phi.iter_mut() { *pi *= inv; }
            }

            eigenvalues.push(omega_sq);
            mode_shapes.push(phi);
        }

        Ok(EigenResult { eigenvalues, mode_shapes })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private loop helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean dot product `aᵀ b`.
#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&ai, &bi)| ai * bi).sum()
}

/// Produce a new starting vector after a lucky breakdown.
///
/// Picks the standard basis vector `e_k` for increasing `k` until a vector
/// orthogonal to all stored Lanczos columns `v_flat[:,0..num_vecs]` is found,
/// then M-normalises it.
fn restart_vector(
    out:      &mut [f64],
    v_flat:   &[f64],
    num_vecs: usize,
    n:        usize,
    m:        &SymCsrMatrix<f64>,
    m_ws:     &mut MatvecWorkspace<f64>,
) {
    for k in 0..n {
        out.fill(0.0);
        out[k] = 1.0;

        // Gram-Schmidt against all stored V columns.
        for jj in 0..num_vecs {
            let col = &v_flat[jj * n..(jj + 1) * n];
            let d   = dot(out, col);
            for i in 0..n { out[i] -= d * col[i]; }
        }

        // M-normalise.
        m.matvec_into(out, m_ws).ok();
        let nrm = dot(out, m_ws.as_slice()).max(0.0).sqrt();
        if nrm > 1e-14 {
            let inv = 1.0 / nrm;
            for oi in out.iter_mut() { *oi *= inv; }
            return;
        }
    }
    // Absolute fallback — subspace is complete, zero out.
    out.fill(0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Implicit QL algorithm with Wilkinson shifts
// ─────────────────────────────────────────────────────────────────────────────

/// Diagonalise the `m × m` symmetric tridiagonal matrix `T` given by its
/// diagonal `d` and sub-diagonal `e`, accumulating orthogonal transformations
/// into the `big_n × m` column-major array `z` (the Lanczos basis `V`).
///
/// On entry `z[:,j]` is the `j`-th Lanczos vector.  On exit `z[:,j]` is
/// the `j`-th global eigenvector of `T` in the physical DOF basis.
///
/// # Arguments
/// * `d`      — diagonal, length `m`. Modified in-place to eigenvalues.
/// * `e`      — sub-diagonal, length `m` (`e[m-1]` should be 0).
/// * `z`      — column-major `big_n × m` array. Modified in-place.
/// * `big_n`  — row count of `z` (total DOFs `N`).
/// * `m`      — subspace dimension.
///
/// # Errors
/// Returns `Err(&'static str)` if QL fails to converge within 30 iterations.
pub fn implicit_ql(
    d:     &mut [f64],
    e:     &mut [f64],
    z:     &mut [f64],
    big_n: usize,
    m:     usize,
) -> std::result::Result<(), &'static str> {
    let max_iter = 30;

    for l in 0..m {
        let mut iter = 0;
        loop {
            // Find a small sub-diagonal element that splits the matrix.
            let mut mm = l;
            while mm < m - 1 {
                let tol = f64::EPSILON * (d[mm].abs() + d[mm + 1].abs());
                if e[mm].abs() <= tol { break; }
                mm += 1;
            }

            if mm == l { break; } // eigenvalue `l` has converged

            if iter == max_iter {
                return Err("Implicit QL failed to converge within max iterations.");
            }
            iter += 1;

            // Wilkinson shift
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let     r = (g * g + 1.0).sqrt();
            g = d[mm] - d[l] + e[l] / (g + g.signum() * r);

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;

            // Givens rotations sweeping from mm down to l.
            for i in (l..mm).rev() {
                let f = s * e[i];
                let b = c * e[i];

                if f.abs() >= g.abs() {
                    c = g / f;
                    let r = (c * c + 1.0).sqrt();
                    e[i + 1] = f * r;
                    s = 1.0 / r;
                    c *= s;
                } else {
                    s = f / g;
                    let r = (s * s + 1.0).sqrt();
                    e[i + 1] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }

                g = d[i + 1] - p;
                let r2 = (d[i] - g) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g + p;
                g = c * r2 - b;

                // Accumulate the Givens rotation into V (big_n × m, column-major).
                //
                // Column `i`   starts at offset i * big_n.
                // Column `i+1` starts at offset (i+1) * big_n.
                //
                // For each row k:
                //   z[i+1, k] =  s * z[i, k] + c * z[i+1, k]
                //   z[i,   k] =  c * z[i, k] - s * z[i+1, k]
                //
                // Operated on contiguous memory for cache efficiency.
                let base_i  = i       * big_n;
                let base_i1 = (i + 1) * big_n;
                for k in 0..big_n {
                    let old_i  = z[base_i  + k];
                    let old_i1 = z[base_i1 + k];
                    z[base_i1 + k] =  s * old_i  + c * old_i1;
                    z[base_i  + k] =  c * old_i  - s * old_i1;
                }
            }

            d[l]  -= p;
            e[l]   = g;
            e[mm]  = 0.0;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::CooBuilder;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Diagonal eigenvalue problem. ω² = k_vals[i] / m_vals[i].
    fn diagonal_system(
        k_vals: &[f64],
        m_vals: &[f64],
    ) -> (SymCsrMatrix<f64>, SymCsrMatrix<f64>) {
        let n = k_vals.len();
        let mut k_coo = CooBuilder::new(n, n);
        let mut m_coo = CooBuilder::new(n, n);
        for i in 0..n {
            k_coo.add(i, i, k_vals[i]);
            m_coo.add(i, i, m_vals[i]);
        }
        (k_coo.build_sym().unwrap(), m_coo.build_sym().unwrap())
    }

    /// 2-DOF shear building.
    ///
    /// K = [[10, -4], [-4, 4]],  M = diag(2, 1)
    ///
    /// det(K - ω²M) = 0 → ω⁴ - 9ω² + 12 = 0 → ω² = (9 ± √33) / 2
    fn shear_building_2dof() -> (SymCsrMatrix<f64>, SymCsrMatrix<f64>, [f64; 2]) {
        let k1 = 6.0_f64;
        let k2 = 4.0_f64;
        let m1 = 2.0_f64;
        let m2 = 1.0_f64;

        let mut k_coo = CooBuilder::new(2, 2);
        k_coo.add(0, 0, k1 + k2);
        k_coo.add(0, 1, -k2);
        k_coo.add(1, 1, k2);
        let k = k_coo.build_sym().unwrap();

        let mut m_coo = CooBuilder::new(2, 2);
        m_coo.add(0, 0, m1);
        m_coo.add(1, 1, m2);
        let m = m_coo.build_sym().unwrap();

        // (10 - 2ω²)(4 - ω²) - 16 = 0
        // 2ω⁴ - 18ω² + 24 = 0  →  ω⁴ - 9ω² + 12 = 0
        let disc  = (81.0_f64 - 48.0).sqrt();
        let w2_lo = (9.0 - disc) / 2.0;
        let w2_hi = (9.0 + disc) / 2.0;

        (k, m, [w2_lo, w2_hi])
    }

    // ── Unit: initialisation and error states ─────────────────────────────────

    #[test]
    fn solver_starts_unprepared() {
        let mut solver = LanczosEigenSolver::new();
        assert!(matches!(solver.solve_modes(1), Err(SolverError::NotAnalyzed)));
    }

    #[test]
    fn zero_modes_returns_empty_without_error() {
        let (k, m, _) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(0).unwrap();
        assert!(r.eigenvalues.is_empty());
        assert!(r.mode_shapes.is_empty());
    }

    #[test]
    fn requesting_more_modes_than_dofs_is_safe() {
        // 2-DOF system, request 20 → must not panic, return ≤ 2 modes
        let (k, m, _) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(20).unwrap();
        assert!(r.eigenvalues.len() <= 2,
            "returned {} modes for 2-DOF system", r.eigenvalues.len());
    }

    #[test]
    fn eigenvalues_returned_in_ascending_order() {
        let (k, m, _) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(2).unwrap();
        for w in r.eigenvalues.windows(2) {
            assert!(w[0] <= w[1], "not sorted: {} > {}", w[0], w[1]);
        }
    }

    // ── Integration: diagonal system ──────────────────────────────────────────

    #[test]
    fn diagonal_4dof_eigenvalues_match_exactly() {
        // K = diag(1,4,9,16),  M = I  →  ω² = {1,4,9,16}
        let (k, m) = diagonal_system(&[1.0, 4.0, 9.0, 16.0], &[1.0; 4]);
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(4).unwrap();

        let exact = [1.0_f64, 4.0, 9.0, 16.0];
        for (i, (&computed, &ex)) in r.eigenvalues.iter().zip(&exact).enumerate() {
            let rel = (computed - ex).abs() / ex;
            assert!(rel < 1e-8,
                "mode {i}: ω²={computed:.10} exact={ex:.10} rel={rel:.2e}");
        }
    }

    #[test]
    fn diagonal_4dof_mode_shapes_m_orthonormal() {
        let (k, m) = diagonal_system(&[1.0, 4.0, 9.0, 16.0], &[1.0; 4]);
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(4).unwrap();
        let nd = r.mode_shapes[0].len();
        let nm = r.mode_shapes.len();

        for i in 0..nm {
            for j in 0..nm {
                let dij: f64 = (0..nd)
                    .map(|k| r.mode_shapes[i][k] * r.mode_shapes[j][k])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (dij - expected).abs();
                assert!(err < 1e-7,
                    "φ_{i}ᵀMφ_{j}={dij:.8} expected={expected} err={err:.2e}");
            }
        }
    }

    // ── Integration: 2-DOF shear building (analytical benchmark) ─────────────

    #[test]
    fn shear_building_2dof_frequencies_match_analytical() {
        let (k, m, [w2_lo, w2_hi]) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(2).unwrap();

        assert_eq!(r.eigenvalues.len(), 2,
            "expected 2 eigenvalues, got {}", r.eigenvalues.len());

        let tol = 1e-8;
        for (i, (&w2c, w2e)) in r.eigenvalues.iter().zip([w2_lo, w2_hi]).enumerate() {
            let rel = (w2c - w2e).abs() / w2e;
            assert!(rel < tol,
                "mode {i}: ω²={w2c:.10} exact={w2e:.10} rel={rel:.2e}");
        }
    }

    #[test]
    fn shear_building_2dof_residuals_tight() {
        let (k, m, _) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(2).unwrap();

        for (i, (phi, &w2)) in r.mode_shapes.iter().zip(&r.eigenvalues).enumerate() {
            let kphi = k.matvec(phi).unwrap();
            let mphi = m.matvec(phi).unwrap();
            let res_norm: f64 = kphi.iter().zip(&mphi)
                .map(|(&kp, &mp)| (kp - w2 * mp).powi(2))
                .sum::<f64>()
                .sqrt();
            let phi_norm: f64 = phi.iter().map(|&p| p * p).sum::<f64>().sqrt();
            let rel = res_norm / (w2 * phi_norm);
            assert!(rel < 1e-7,
                "mode {i}: ‖Kφ - ω²Mφ‖/(ω²‖φ‖) = {rel:.2e}");
        }
    }

    #[test]
    fn shear_building_2dof_m_normalised() {
        let (k, m, _) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(2).unwrap();

        for (i, phi) in r.mode_shapes.iter().enumerate() {
            let mphi = m.matvec(phi).unwrap();
            let m_ip: f64 = phi.iter().zip(&mphi).map(|(&p, &mp)| p * mp).sum();
            let err = (m_ip - 1.0).abs();
            assert!(err < 1e-8,
                "mode {i}: φᵀMφ={m_ip:.10} expected 1.0 err={err:.2e}");
        }
    }

    // ── Integration: 3-DOF diagonal, only 1 mode requested ───────────────────

    #[test]
    fn diagonal_3dof_first_mode_only() {
        let (k, m) = diagonal_system(&[1.0, 4.0, 9.0], &[1.0; 3]);
        let mut solver = LanczosEigenSolver::new();
        solver.prepare(&k, &m).unwrap();
        let r = solver.solve_modes(1).unwrap();

        assert_eq!(r.eigenvalues.len(), 1);
        let err = (r.eigenvalues[0] - 1.0).abs();
        assert!(err < 1e-8, "ω₁²={:.10} err={err:.2e}", r.eigenvalues[0]);
    }

    // ── Convenience: compute() ────────────────────────────────────────────────

    #[test]
    fn compute_convenience_matches_prepare_plus_solve() {
        let (k, m, [w2_lo, _]) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        let r = solver.compute(&k, &m, 1).unwrap();
        let rel = (r.eigenvalues[0] - w2_lo).abs() / w2_lo;
        assert!(rel < 1e-8, "compute() rel={rel:.2e}");
    }

    // ── EigenResult helper methods ────────────────────────────────────────────

    #[test]
    fn frequencies_hz_are_positive() {
        let (k, m, _) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        let r = solver.compute(&k, &m, 2).unwrap();
        for (i, f) in r.frequencies_hz().iter().enumerate() {
            assert!(*f > 0.0, "f_{i} = {f}");
        }
    }

    #[test]
    fn periods_are_positive_and_finite() {
        let (k, m, _) = shear_building_2dof();
        let mut solver = LanczosEigenSolver::new();
        let r = solver.compute(&k, &m, 2).unwrap();
        for (i, t) in r.periods().iter().enumerate() {
            assert!(*t > 0.0 && t.is_finite(), "T_{i} = {t}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration into the module tree
// ─────────────────────────────────────────────────────────────────────────────
//
// To wire this file into the build, add the following two lines to
// `solvers/src/eigen/mod.rs`:
//
//   pub mod lanczos;
//   pub use lanczos::LanczosEigenSolver;
//
// And optionally re-export from the crate root in `solvers/src/lib.rs`:
//
//   pub use eigen::LanczosEigenSolver;