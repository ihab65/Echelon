use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sparse::CsrMatrix;

fn bench_scatter_add(c: &mut Criterion) {
    // 100-element chain — each element touches 2 DOFs, so 2×2 ke
    let n_dof = 101;
    let element_dofs: Vec<Vec<usize>> = (0..100).map(|i| vec![i, i + 1]).collect();
    let mut k = CsrMatrix::from_dof_connectivity(n_dof, &element_dofs).unwrap();
    let ke = vec![1.0_f64, -1.0, -1.0, 1.0];

    c.bench_function("scatter_add_100_elements", |b| {
        b.iter(|| {
            k.zero();
            for dofs in &element_dofs {
                k.scatter_add(black_box(&ke), black_box(dofs)).unwrap();
            }
        })
    });
}

criterion_group!(benches, bench_scatter_add);
criterion_main!(benches);