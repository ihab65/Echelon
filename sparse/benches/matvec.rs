use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sparse::MatvecWorkspace;

fn bench_matvec(c: &mut Criterion) {
    let n = 1000;
    // Tridiagonal n×n matrix
    let mut coo = sparse::CooBuilder::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i + 1 < n { coo.add(i, i + 1, -1.0); }
        if i > 0     { coo.add(i, i - 1, -1.0); }
    }
    let k = coo.build_csr().unwrap();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    c.bench_function("csr_matvec_allocating", |b| {
        b.iter(|| k.matvec(black_box(&x)).unwrap())
    });

    let mut ws = MatvecWorkspace::new(n);
    c.bench_function("csr_matvec_into_workspace", |b| {
        b.iter(|| k.matvec_into(black_box(&x), &mut ws).unwrap())
    });
}

criterion_group!(benches, bench_matvec);
criterion_main!(benches);