#[cfg(feature = "io")]
use tempfile::NamedTempFile;
#[cfg(feature = "io")]
use std::io::Write;
#[cfg(feature = "io")]
use sparse::coo::CooBuilder;
#[cfg(feature = "io")]
use sparse::SparseMatrix;

#[cfg(feature = "io")]
#[test]
fn test_from_mtx_general() {
    // Create a temporary general MTX file
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "%%MatrixMarket matrix coordinate real general").unwrap();
    writeln!(file, "3 3 3").unwrap();
    writeln!(file, "1 1 10.0").unwrap();
    writeln!(file, "2 2 20.0").unwrap();
    writeln!(file, "3 3 30.0").unwrap();

    let builder = CooBuilder::from_mtx(file.path()).unwrap();
    let csr = builder.build_csr().unwrap();

    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.get(0, 0).unwrap(), 10.0);
    assert_eq!(csr.get(1, 1).unwrap(), 20.0);
    assert_eq!(csr.get(2, 2).unwrap(), 30.0);
}

#[cfg(feature = "io")]
#[test]
fn test_from_mtx_symmetric_normalization() {
    // Create a symmetric MTX file with ONLY lower triangle entries
    // This tests if our loader correctly flips them to the upper triangle (r <= c)
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "%%MatrixMarket matrix coordinate real symmetric").unwrap();
    writeln!(file, "2 2 3").unwrap();
    writeln!(file, "1 1 2.0").unwrap();
    writeln!(file, "2 1 0.5").unwrap(); // (row 2, col 1) -> should become (0, 1)
    writeln!(file, "2 2 2.0").unwrap();

    let builder = CooBuilder::from_mtx(file.path()).unwrap();
    
    // build_sym requires upper triangle invariants (j >= i)
    let sym = builder.build_sym().expect("Should normalize lower triangle to upper");

    assert_eq!(sym.get(0, 0).unwrap(), 2.0);
    assert_eq!(sym.get(0, 1).unwrap(), 0.5); // Check flipped entry
    assert_eq!(sym.get(1, 0).unwrap(), 0.5); // Check mirrored access
    assert_eq!(sym.get(1, 1).unwrap(), 2.0);
}