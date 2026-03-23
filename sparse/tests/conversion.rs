//! Integration tests for format conversions.

use sparse::{CsrMatrix, SymCsrMatrix, CooBuilder};
use sparse::convert::{csr_to_csc, csc_to_csr, sym_to_csc, csr_to_sym};

fn sym_tridiag() -> SymCsrMatrix {
    let mut coo = CooBuilder::new(3, 3);
    coo.add(0, 0,  4.0); coo.add(0, 1, -1.0);
    coo.add(1, 1,  4.0); coo.add(1, 2, -1.0);
    coo.add(2, 2,  4.0);
    coo.build_sym().unwrap()
}

fn upper_triangular_csr() -> CsrMatrix {
    let mut coo = CooBuilder::new(3, 3);
    coo.add(0, 0, 1.0); coo.add(0, 2, 2.0);
    coo.add(1, 1, 3.0); coo.add(1, 2, 4.0);
    coo.add(2, 2, 5.0);
    coo.build_csr().unwrap()
}

#[test]
fn csr_csc_roundtrip_nnz() {
    let csr = upper_triangular_csr();
    let csc = csr_to_csc(&csr);
    let csr2 = csc_to_csr(&csc);
    assert_eq!(csr.nnz(), csc.nnz());
    assert_eq!(csr.nnz(), csr2.nnz());
    csc.validate().unwrap();
    csr2.validate().unwrap();
}

#[test]
fn csr_csc_roundtrip_matvec() {
    let csr = upper_triangular_csr();
    let csc = csr_to_csc(&csr);
    let x = vec![1.0_f64, 2.0, 3.0];
    assert_eq!(csr.matvec(&x).unwrap(), csc.matvec(&x).unwrap());
}

#[test]
fn sym_to_csc_full_expansion() {
    let sym = sym_tridiag();
    let csc = sym_to_csc(&sym);
    csc.validate().unwrap();
    // full symmetric tridiag has 3 diag + 2*2 off-diag = 7 entries
    assert_eq!(csc.nnz(), 7);
    // symmetry preserved
    assert_eq!(csc.get(0, 1).unwrap(), csc.get(1, 0).unwrap());
}

#[test]
fn sym_to_csc_matvec_matches_sym_matvec() {
    let sym = sym_tridiag();
    let csc = sym_to_csc(&sym);
    let x = vec![1.0_f64, 2.0, 3.0];
    let y_sym = sym.matvec(&x).unwrap();
    let y_csc = csc.matvec(&x).unwrap();
    for (a, b) in y_sym.iter().zip(y_csc.iter()) {
        assert!((a - b).abs() < 1e-14, "sym={a} csc={b}");
    }
}

#[test]
fn csr_to_sym_extracts_upper_triangle() {
    // Build symmetric CSR (both triangles) then convert
    let mut coo = CooBuilder::new(3, 3);
    coo.add(0, 0,  4.0); coo.add(0, 1, -1.0);
    coo.add(1, 0, -1.0); coo.add(1, 1,  4.0); coo.add(1, 2, -1.0);
    coo.add(2, 1, -1.0); coo.add(2, 2,  4.0);
    let csr = coo.build_csr().unwrap();

    let sym = csr_to_sym(&csr).unwrap();
    sym.validate().unwrap();

    // upper values preserved
    assert_eq!(sym.get(0, 0).unwrap(),  4.0);
    assert_eq!(sym.get(0, 1).unwrap(), -1.0);
    assert_eq!(sym.get(1, 2).unwrap(), -1.0);
    // lower reads mirrored
    assert_eq!(sym.get(1, 0).unwrap(), -1.0);
}

#[test]
fn full_pipeline_sym_csc_csr_matvec() {
    let sym  = sym_tridiag();
    let csc  = sym_to_csc(&sym);
    let csr  = csc_to_csr(&csc);
    let x    = vec![1.0_f64, 2.0, 3.0];
    let y_sym = sym.matvec(&x).unwrap();
    let y_csr = csr.matvec(&x).unwrap();
    for (a, b) in y_sym.iter().zip(y_csr.iter()) {
        assert!((a - b).abs() < 1e-14);
    }
}