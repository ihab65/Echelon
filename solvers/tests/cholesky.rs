//! Integration tests for the Cholesky factorization pipeline.
//!
//! Tests the symbolic phase through the public `SparseSolver` interface
//! and directly through `symbolic::analyze`.

use solvers::cholesky::symbolic::analyze;
use sparse::{CooBuilder, SymCsrMatrix};
use sparse::convert::sym_to_csc;

fn tridiag(n: usize) -> SymCsrMatrix {
    let mut coo = CooBuilder::new(n, n);
    for i in 0..n       { coo.add(i, i,      2.0); }
    for i in 0..(n - 1) { coo.add(i, i + 1, -1.0); }
    coo.build_sym().unwrap()
}

// ---- symbolic phase via public analyze() ----

#[test]
fn symbolic_tridiag_nnz_l_equals_2n_minus_1() {
    // For an n×n tridiagonal: L is bidiagonal → nnz(L) = 2n - 1
    for n in [3, 5, 10, 20] {
        let k   = tridiag(n);
        let sym = analyze(&sym_to_csc(&k)).unwrap();
        assert_eq!(
            sym.nnz_l(), 2 * n - 1,
            "n={n}: expected nnz(L)={}, got {}", 2 * n - 1, sym.nnz_l()
        );
    }
}

#[test]
fn symbolic_col_ptr_monotone_nondecreasing() {
    let sym = analyze(&sym_to_csc(&tridiag(10))).unwrap();
    for w in sym.col_ptr.windows(2) {
        assert!(w[0] <= w[1], "col_ptr not non-decreasing: {:?}", &sym.col_ptr);
    }
}

#[test]
fn symbolic_row_idx_in_lower_triangle() {
    let sym = analyze(&sym_to_csc(&tridiag(8))).unwrap();
    for col in 0..sym.n {
        for &row in &sym.row_idx[sym.col_ptr[col]..sym.col_ptr[col + 1]] {
            assert!(row >= col, "upper-triangle entry in L: ({row}, {col})");
        }
    }
}

#[test]
fn symbolic_diagonal_entries_present_in_all_columns() {
    let sym = analyze(&sym_to_csc(&tridiag(6))).unwrap();
    for col in 0..sym.n {
        let rows = &sym.row_idx[sym.col_ptr[col]..sym.col_ptr[col + 1]];
        assert!(rows.contains(&col), "diagonal missing in col {col}");
    }
}

// ---- SparseSolver interface ----

#[test]
fn sparse_solver_analyze_then_factorize_error() {
    use solvers::{cholesky::SparseSolver, SolverError};
    let k = tridiag(4);
    let mut solver = SparseSolver::new();
    // factorize before analyze should error
    assert!(matches!(
        solver.factorize(&k).unwrap_err(),
        SolverError::NotAnalyzed
    ));
    // after analyze it should succeed (numeric is a stub but doesn't error)
    solver.analyze(&k).unwrap();
    solver.factorize(&k).unwrap();
}

#[test]
fn sparse_solver_solve_before_factorize_errors() {
    use solvers::{cholesky::SparseSolver, SolverError};
    let k = tridiag(3);
    let mut solver = SparseSolver::new();
    solver.analyze(&k).unwrap();
    // factorize not called yet
    let mut u = vec![0.0; 3];
    assert!(matches!(
        solver.solve(&[1.0, 0.0, 0.0], &mut u).unwrap_err(),
        SolverError::NotFactorized
    ));
}
