//! Integration tests for SymCsrMatrix — public API only.

use sparse::{SymCsrMatrix, SparseError, SparseMatrix, CooBuilder};

fn tridiag() -> SymCsrMatrix<f64> {
    // [ 4 -1  0]
    // [-1  4 -1]
    // [ 0 -1  4]
    let mut coo = CooBuilder::new(3, 3);
    coo.add(0, 0,  4.0); coo.add(0, 1, -1.0);
    coo.add(1, 1,  4.0); coo.add(1, 2, -1.0);
    coo.add(2, 2,  4.0);
    coo.build_sym().unwrap()
}

#[test]
fn matvec_all_ones() {
    // A * [1,1,1]ᵀ = [3,2,3]ᵀ
    let y = tridiag().matvec(&[1.0, 1.0, 1.0]).unwrap();
    assert_eq!(y, vec![3.0, 2.0, 3.0]);
}

#[test]
fn get_both_triangles_equal() {
    let m = tridiag();
    assert_eq!(m.get(0, 1).unwrap(), m.get(1, 0).unwrap());
    assert_eq!(m.get(1, 2).unwrap(), m.get(2, 1).unwrap());
}

#[test]
fn diagonal_correct() {
    assert_eq!(tridiag().extract_diagonal(), vec![4.0, 4.0, 4.0]);
}

#[test]
fn boundary_condition() {
    let mut m = tridiag();
    m.zero_row_col(1).unwrap();
    assert_eq!(m.get(1, 1).unwrap(), 1.0);
    assert_eq!(m.get(0, 1).unwrap(), 0.0); // column zeroed
    assert_eq!(m.get(0, 0).unwrap(), 4.0); // untouched
}

#[test]
fn sparse_matrix_trait() {
    let m = tridiag();
    assert_eq!(m.nrows(), 3);
    assert!(m.is_square());
    assert_eq!(m.nnz(), 5); // 3 diagonal + 2 off-diagonal
}

#[test]
fn lower_triangle_entry_error() {
    let mut m = tridiag();
    assert!(matches!(
        m.add_value(2, 0, 1.0).unwrap_err(),
        SparseError::LowerTriangleEntry { .. }
    ));
}

#[test]
fn validate_passes() {
    tridiag().validate().unwrap();
}

#[test]
fn coo_lower_triangle_mirrored() {
    // Feed lower-triangle entries to CooBuilder — should produce valid SymCsr
    let mut coo = CooBuilder::new(3, 3);
    coo.add(0, 0,  4.0);
    coo.add(1, 0, -1.0); // lower → mirrors to (0,1)
    coo.add(1, 1,  4.0);
    coo.add(2, 1, -1.0); // lower → mirrors to (1,2)
    coo.add(2, 2,  4.0);
    let m = coo.build_sym().unwrap();
    m.validate().unwrap();
    assert_eq!(m.get(0, 1).unwrap(), -1.0);
    assert_eq!(m.get(1, 2).unwrap(), -1.0);
}