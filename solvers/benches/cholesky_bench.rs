use faer::Mat;
use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, BatchSize,
};
use sparse::{CooBuilder, SymCsrMatrix};
use solvers::cholesky::SparseSolver;
use solvers::LinearSolver;

// =============================================================================
// Matrix builders
// =============================================================================

/// 1-D spring chain: tridiagonal SPD n×n matrix.
/// Identical to the one in solver_stress.rs — duplicated here to keep
/// the benchmark file self-contained without a shared helper crate.
fn spring_chain(n: usize) -> SymCsrMatrix<f64> {
    let mut coo = CooBuilder::new(n, n);
    for i in 0..n {
        let diag = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
        coo.add(i, i, diag);
    }
    for i in 0..(n - 2) {
        coo.add(i, i + 1, -1.0);
    }
    coo.build_sym().unwrap()
}

/// 2-D grid Laplacian: 5-point stencil on an `nx × ny` interior grid.
fn laplacian_2d(nx: usize, ny: usize) -> SymCsrMatrix<f64> {
    let n = nx * ny;
    let mut coo = CooBuilder::new(n, n);
    for iy in 0..ny {
        for ix in 0..nx {
            let i = ix + iy * nx;
            coo.add(i, i, 4.0);
            if ix + 1 < nx { coo.add(i, (ix + 1) + iy * nx, -1.0); }
            if iy + 1 < ny { coo.add(i, ix + (iy + 1) * nx, -1.0); }
        }
    }
    coo.build_sym().unwrap()
}

/// Deterministic RHS: `f[i] = (i+1) as f64 % 7 + 1` — avoids cancellations.
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) % 7 + 1) as f64).collect()
}

// =============================================================================
// Pre-solved solver state for benchmarks that need it pre-warmed
// =============================================================================

/// A fully analyzed + factorized solver, ready for `solve` benchmarks.
struct ReadySolver {
    solver: SparseSolver<f64>,
    n:      usize,
}

impl ReadySolver {
    fn new(k: &SymCsrMatrix<f64>) -> Self {
        let mut solver = SparseSolver::new();
        solver.set_ordering(solvers::ordering::Ordering::Amd);
        solver.analyze_and_factorize(k).unwrap();
        Self { solver, n: k.n }
    }
    fn solve(&self, f: &[f64]) -> Vec<f64> {
        let mut u = vec![0.0_f64; self.n];
        self.solver.solve(f, &mut u).unwrap();
        u
    }
}

// =============================================================================
// Benchmark 1 — analyze (symbolic phase) vs matrix size
//
// The symbolic phase runs once per topology change.  For banded matrices
// (tridiagonal) it should scale approximately as O(n) because nnz(L) = O(n)
// and the elimination tree is a simple chain.
//
// Input sizes: 100, 500, 1000, 5000
// Expected: roughly linear scaling in wall time.
// =============================================================================

fn bench_analyze(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze/tridiagonal");
    group.sample_size(20);  // analyze is slow for large n — fewer samples

    for &n in &[100_usize, 500, 1000, 5000] {
        let k = spring_chain(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &k, |b, k| {
            b.iter_batched(
                // Setup: fresh solver per iteration so analyze always starts cold
                SparseSolver::new,
                |mut solver| {
                    solver.analyze(black_box(k)).unwrap();
                    solver  // return to prevent drop inside timing
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// =============================================================================
// Benchmark 2 — factorize (numeric phase) vs matrix size
//
// The numeric phase runs every Newton iteration.  For a tridiagonal, L is
// bidiagonal → nnz(L) = 2n-1 → factorize should be O(n).
// For a 2-D grid, fill is larger → factorize is slower per n.
//
// We bench both matrix types to see the fill-reduction effect of RCM.
// =============================================================================

fn bench_factorize_tridiagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorize/tridiagonal");
    group.sample_size(50);

    for &n in &[100_usize, 500, 1000, 5000] {
        let k = spring_chain(n);

        // Pre-compute symbolic analysis — not part of the timed region.
        let mut solver = SparseSolver::new();
        solver.analyze(&k).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), &k, |b, k| {
            b.iter(|| {
                // Each iteration re-factorizes with the same values.
                // This matches the Newton-Raphson hot path.
                solver.factorize(black_box(k)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_factorize_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorize/grid-laplacian");
    group.sample_size(30);

    // Grid sizes: (10,10)=100, (20,20)=400, (40,40)=1600
    for &(nx, ny) in &[(10_usize, 10_usize), (20, 20), (40, 40)] {
        let k   = laplacian_2d(nx, ny);
        let n   = k.n;
        let id  = BenchmarkId::new(format!("{nx}x{ny}"), n);

        let mut solver = SparseSolver::new();
        solver.analyze(&k).unwrap();

        group.bench_with_input(id, &k, |b, k| {
            b.iter(|| {
                solver.factorize(black_box(k)).unwrap();
            });
        });
    }
    group.finish();
}

// =============================================================================
// Benchmark 3 — solve (triangular solve) vs matrix size
//
// The solve phase runs once per right-hand side.  For a factored system it
// is O(nnz(L)) — same asymptotic cost as factorize but with a smaller constant.
// =============================================================================

fn bench_solve_tridiagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve/tridiagonal");
    group.sample_size(200);  // solve is cheap, more samples give better stats

    for &n in &[100_usize, 500, 1000, 5000] {
        let k = spring_chain(n);
        let f = rhs(n);
        let rs = ReadySolver::new(&k);

        group.bench_with_input(BenchmarkId::from_parameter(n), &f, |b, f| {
            b.iter(|| rs.solve(black_box(f)));
        });
    }
    group.finish();
}

fn bench_solve_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve/grid-laplacian");
    group.sample_size(100);

    for &(nx, ny) in &[(10_usize, 10_usize), (20, 20), (40, 40)] {
        let k   = laplacian_2d(nx, ny);
        let n   = k.n;
        let f   = rhs(n);
        let rs  = ReadySolver::new(&k);
        let id  = BenchmarkId::new(format!("{nx}x{ny}"), n);

        group.bench_with_input(id, &f, |b, f| {
            b.iter(|| rs.solve(black_box(f)));
        });
    }
    group.finish();
}

// =============================================================================
// Benchmark 4 — end-to-end: analyze + factorize + solve
//
// Measures the total cost for a fresh solve (new model, new topology).
// This is the cost seen by the user the first time they solve a system.
// =============================================================================

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end-to-end/tridiagonal");
    group.sample_size(20);

    for &n in &[100_usize, 500, 1000] {
        let k = spring_chain(n);
        let f = rhs(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &(&k, &f), |b, &(k, f)| {
            b.iter(|| {
                let mut solver = SparseSolver::new();
                solver.analyze_and_factorize(black_box(k)).unwrap();
                let mut u = vec![0.0_f64; k.n];
                solver.solve(black_box(f), &mut u).unwrap();
                u  // consumed by criterion's implicit black_box on return
            });
        });
    }
    group.finish();
}

// =============================================================================
// Benchmark 5 — faer comparison (dense Cholesky)
//
// faer is a high-quality pure-Rust linear algebra library.  We compare our
// sparse Cholesky against faer's **dense** Cholesky to show:
//
// 1. For small n (≤ 200), dense and sparse are comparable.
// 2. For large n (≥ 500), our sparse solver is dramatically faster because
//    it avoids O(n²) work on the zeros.
//
// We use faer's dense interface because its sparse API changed significantly
// between 0.14, 0.16, and 0.19.  The dense path is stable and available in
// all versions.  The comparison is still informative: dense Cholesky is the
// naive baseline you would reach for before implementing a sparse solver.
//
// ── faer dense Cholesky API (v0.16) ──────────────────────────────────────────
//
// use faer::prelude::*;
// use faer::Side;
//
// let mat: Mat<f64> = faer::Mat::from_fn(n, n, |i, j| { ... });
// let llt = mat.cholesky(Side::Lower).unwrap();
// let rhs_col: Col<f64> = faer::Col::from_fn(n, |i| f[i]);
// let sol = llt.solve(rhs_col.as_ref());
//
// ─────────────────────────────────────────────────────────────────────────────
// =============================================================================

fn bench_faer_dense_vs_sparse(c: &mut Criterion) {
    use faer::prelude::*;
    use faer::Side;

    let mut group = c.benchmark_group("comparison/sparse-vs-faer-dense");
    group.sample_size(20);

    for &n in &[50_usize, 100, 200, 500] {
        let k_sparse = spring_chain(n);
        let f        = rhs(n);

        // Build the equivalent faer dense matrix once per n.
        // We expand the symmetric tridiagonal to a full n×n dense matrix.
        let k_dense: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == 0 || i == n - 1 {
                if i == j { 1.0 } else { 0.0 }  // identity row
            } else if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        });

        // ── Our sparse solver ───────────────────────────────────────────────
        {
            let id = BenchmarkId::new("echelon-sparse", n);
            let rs = ReadySolver::new(&k_sparse);  // pre-analyzed outside loop

            group.bench_with_input(id, &f, |b, f| {
                b.iter(|| rs.solve(black_box(f)));
            });
        }

        // ── faer dense Cholesky ─────────────────────────────────────────────
        {
            let id  = BenchmarkId::new("faer-dense-solve", n);
            let rhs_mat: Mat<f64> = Mat::from_fn(n, 1, |i, _| f[i]);

            // Pre-factorize outside the loop (fair comparison with our
            // pre-analyzed sparse solver above).
            let llt = k_dense.cholesky(Side::Lower).unwrap();

            group.bench_with_input(id, &rhs_mat, |b, rhs_mat| {
                b.iter(|| llt.solve(black_box(rhs_mat)));
            });
        }
    }
    group.finish();
}

// =============================================================================
// Benchmark 6 — faer dense factorize time vs our sparse factorize time
//
// Separates the factorization cost from the solve cost, making it clear that
// the sparse advantage comes entirely from avoiding O(n²) work in the
// numeric phase — not from the solve.
// =============================================================================

fn bench_faer_dense_factorize_vs_sparse_factorize(c: &mut Criterion) {
    use faer::prelude::*;
    use faer::Side;

    let mut group = c.benchmark_group("comparison/factorize-sparse-vs-faer-dense");
    group.sample_size(20);

    for &n in &[50_usize, 100, 200, 500] {
        let k_sparse = spring_chain(n);

        let k_dense: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == 0 || i == n - 1 {
                if i == j { 1.0 } else { 0.0 }  // identity row
            } else if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        });

        // ── Our sparse: factorize only (analyze pre-computed) ───────────────
        {
            let id = BenchmarkId::new("echelon-sparse-factorize", n);
            let mut solver = SparseSolver::new();
            solver.analyze(&k_sparse).unwrap();

            group.bench_with_input(id, &k_sparse, |b, k| {
                b.iter(|| {
                    solver.factorize(black_box(k)).unwrap();
                });
            });
        }

        // ── faer dense: full Cholesky decomposition ─────────────────────────
        {
            let id = BenchmarkId::new("faer-dense-factorize", n);
            group.bench_with_input(id, &k_dense, |b, mat| {
                b.iter(|| {
                    black_box(mat).cholesky(Side::Lower).unwrap()
                });
            });
        }
    }
    group.finish();
}

// =============================================================================
// Criterion wiring
// =============================================================================

criterion_group!(
    benches_analyze,
    bench_analyze,
);

criterion_group!(
    benches_factorize,
    bench_factorize_tridiagonal,
    bench_factorize_grid,
);

criterion_group!(
    benches_solve,
    bench_solve_tridiagonal,
    bench_solve_grid,
);

criterion_group!(
    benches_end_to_end,
    bench_end_to_end,
);

criterion_group!(
    benches_faer,
    bench_faer_dense_vs_sparse,
    bench_faer_dense_factorize_vs_sparse_factorize,
);

criterion_main!(
    benches_analyze,
    benches_factorize,
    benches_solve,
    benches_end_to_end,
    benches_faer,
);
