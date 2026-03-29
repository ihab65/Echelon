//! Integration tests for CsrMatrix — public API only, no internals.

use sparse::{CsrMatrix, SparseError, SparseMatrix, CooBuilder};

fn spring_chain() -> CsrMatrix<f64> {
    // Three springs in series: fixed – u0 – u1 – fixed, k=1 each
    // K = [[2,-1],[-1,2]]
    let mut coo = CooBuilder::new(2, 2);
    coo.add(0, 0,  1.0); // spring 1
    coo.add(0, 0,  1.0); coo.add(0, 1, -1.0);
    coo.add(1, 0, -1.0); coo.add(1, 1,  1.0); // spring 2
    coo.add(1, 1,  1.0); // spring 3
    coo.build_csr().unwrap()
}

#[test]
fn spring_chain_stiffness_values() {
    let k = spring_chain();
    assert_eq!(k.get(0, 0).unwrap(),  2.0);
    assert_eq!(k.get(0, 1).unwrap(), -1.0);
    assert_eq!(k.get(1, 0).unwrap(), -1.0);
    assert_eq!(k.get(1, 1).unwrap(),  2.0);
}

#[test]
fn spring_chain_matvec_checks_solution() {
    // K * u_exact must equal F = [1, 0]
    let k = spring_chain();
    let u = vec![2.0_f64 / 3.0, 1.0 / 3.0];
    let f = k.matvec(&u).unwrap();
    assert!((f[0] - 1.0).abs() < 1e-12);
    assert!((f[1] - 0.0).abs() < 1e-12);
}

#[test]
fn boundary_condition_zeroes_row_and_col() {
    let mut k = spring_chain();
    k.zero_row_col(0).unwrap();
    assert_eq!(k.get(0, 0).unwrap(), 1.0);
    assert_eq!(k.get(0, 1).unwrap(), 0.0);
    assert_eq!(k.get(1, 0).unwrap(), 0.0);
    assert_eq!(k.get(1, 1).unwrap(), 2.0);
}

#[test]
fn sparse_matrix_trait_methods() {
    let k = spring_chain();
    assert_eq!(k.nrows(), 2);
    assert_eq!(k.ncols(), 2);
    assert!(k.is_square());
    assert_eq!(k.nnz(), 4);
}

#[test]
fn validate_passes_on_assembled_matrix() {
    spring_chain().validate().unwrap();
}

#[test]
fn error_add_to_absent_entry() {
    let mut m = CsrMatrix::from_pattern(2, 2, &[vec![0usize], vec![1usize]]).unwrap();
    assert!(matches!(
        m.add_value(0, 1, 1.0).unwrap_err(),
        SparseError::IndexOutOfBounds { row: 0, col: 1 }
    ));
}