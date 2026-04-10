<div align="center">

# Echelon

**A structural finite element engine built in Rust — parallel by design, Python-accessible, and AI-native.**

*Currently in active development. The sparse matrix layer and solver ordering are complete. Cholesky factorization, elements, materials, and the Python API are in progress.*

</div>

---

## What Is Echelon?

Echelon is a structural analysis engine that addresses a gap in the existing landscape: there is no tool that is simultaneously **structurally correct**, **parallel by design**, **AI/ML-native**, **Python-accessible**, and **built for population-scale workflows**.

OpenSees is powerful but carries 1990s architecture — single-threaded, global mutable state, not designed for programmatic access at scale. Commercial tools are black boxes with no ML integration. Pure-Python FEM tools are too slow for anything beyond toy problems.

Echelon is built from the ground up for the workflow that modern structural engineering research actually needs:

```python
import echelon as ec

def rc_frame(params):
    model = ec.Model(ndm=2, ndf=3)
    # ... build parameterized frame model ...
    return model

# Run 10,000 analyses in parallel, get structured results
results = ec.Population(
    builder=rc_frame,
    distributions={
        'height': ec.LogNormal(mean=3.0, cov=0.1),
        'E':      ec.LogNormal(mean=200e9, cov=0.05),
    },
    n_samples=10_000,
    n_workers=8,
).run(analysis=ec.StaticAnalysis(integrator=ec.LoadControl(incr=0.1)))

df = results.to_dataframe()
results.to_parquet("rc_frames.parquet")
```

That workflow — define, distribute, run in parallel, get structured data — does not exist as a coherent tool anywhere else.

---

## Architecture

Echelon is a Cargo workspace of focused crates with a strict one-way dependency graph:

```
                     ┌─────────────────┐
                     │  echelon (PyO3) │  ← Python API (OpenSeesPy replacement)
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ assembly │   │  solver  │   │   bevy   │  (optional visualization)
        └─────┬────┘   └────┬─────┘   └──────────┘
              │              │
    ┌─────────┴──────────────┘
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│ elements │ │materials │
└────┬─────┘ └────┬─────┘
     │             │
     └──────┬──────┘
            ▼
       ┌──────────┐
       │   core   │  (index types, small dense matrices, transforms)
       └────┬─────┘
            │
            ▼
       ┌──────────┐
       │ solvers  │  ← Cholesky, RCM ordering (this crate)
       └────┬─────┘
            │
            ▼
       ┌──────────┐
       │  sparse  │  ← CSR, SymCSR, CSC, CooBuilder (foundation)
       └──────────┘
```

The entire analysis stack — from sparse storage through the Python API — is pure Rust with no
external linear algebra dependencies. No LAPACK, no BLAS, no unsafe FFI.

---

## Current Status

### ✅ `sparse` — Complete

The sparse matrix storage layer. Three formats: `CsrMatrix` (full, row-major), `SymCsrMatrix`
(upper triangle, for Cholesky input), `CscMatrix` (column-major, for Cholesky computation).

Everything needed for FEM assembly is implemented:

- `from_dof_connectivity` — build sparsity pattern from element DOF lists
- `scatter_add` — the innermost assembly loop (dense element stiffness → global K)
- `zero_row_col` — Dirichlet BC application
- `matvec_into` — zero-allocation matrix-vector product
- Full conversion pipeline: CSR → SymCSR → CSC
- `CooBuilder` for triplet-based construction with duplicate summation
- Optional `rayon` parallel feature
- Comprehensive test suite with integration tests

### ✅ `solvers` — Ordering, Cholesky, and LDLT Complete

RCM (Reverse Cuthill-McKee) fill-reduction ordering is fully implemented:
- `Graph` — adjacency structure from `SymCsrMatrix`, O(nnz) two-pass construction
- `Permutation` — validated newtype with `permute_sym` for symmetric matrix reordering
- `rcm()` — full algorithm with pseudo-peripheral node detection and degree-ordered BFS

`SparseSolver` interface (three-phase: `analyze` → `factorize` → `solve`) is implemented.
Included are both symmetric positive-definite (Cholesky) and symmetric indefinite (LDLᵀ) direct solvers.

### ✅ `fem-core` — Complete

Foundation types for the finite element domain:
- Node and element ID newtypes.
- Small dense vector/matrix math.
- 2D `CoordTransf2d` and `DofMap` implementations.

### ✅ `materials` — Complete

Architecture supporting smooth autodiff-friendly materials and history-dependent materials.
Includes reference implementations like `ElasticUniaxial`.

### ✅ `elements` — Complete

Implementation of finite elements:
- `Truss2d` (linear, energy-based implementation)
- `ElasticBeam2d` (frame element, linear elastic)

### ✅ `assembly` — Complete

The core system for holding models and assembling sparse matrices.
- `Model` structure for managing nodes, elements, and DOFs.
- Connectivity and topology graph generation to determine matrix non-zero patterns.
- Stiffness, mass, and damping matrix assembly.
- Global load combination and external force application.

### ✅ `analysis` (solver) — Complete
Core algorithmic routines for solving structural equilibrium:
- Linear equation solving.
- Nonlinear `Newton-Raphson` and `ModifiedNewton` solution algorithms.
- Integrators for `StaticNonlinear` (e.g. `LoadControl`, `DispControl`).
- Integrators for `Transient` dynamics (e.g. `Newmark`, `HHT`).

### 🔄 In Progress / Planned

| Crate | Status |
|---|---|
| `echelon` (Python) | Not started — PyO3 bindings, population runner |

---

## Why Build This?

### The Research Workflow Problem

The current workflow for structural engineering researchers is painfully slow:

1. Write OpenSees Tcl scripts by hand
2. Wrap in Python subprocess calls
3. Manage hundreds of output text files
4. Parse text files into numpy arrays
5. Feed into ML pipeline manually
6. Repeat for each parameter variation
7. Spend 60% of time on infrastructure, 40% on research

**Total time to generate 10,000 training samples: weeks.**

With Echelon, the same workflow takes hours. Infrastructure time approaches zero. Research time approaches 100%.

### The Reproducibility Problem

In structural ML research today:
- Paper A trains on their OpenSees data
- Paper B trains on their ETABS data
- Paper C trains on their simplified model

Results are not comparable. Nobody knows if improvement comes from better ML or better training data. This is the reproducibility crisis in structural ML.

With Echelon:
- Cite "echelon v0.3.0, RC frame archetype v2"
- Anyone can reproduce the dataset exactly
- Results are comparable across papers
- The field accumulates knowledge instead of rediscovering it

### The Research Areas Echelon Unlocks

**Structural fragility at scale** — run 10,000 structures × 100 ground motions = 1,000,000
analyses. Statistical confidence that changes what's publishable. Fragility surfaces instead of
fragility curves.

**Physics-informed neural networks** — your exact solver generates ground truth for PINN training.
Researchers focus on network architecture, not FEM plumbing.

**Surrogate models for NLTHA** — standardized data generation means surrogate papers are about
the ML architecture, not about how training data was produced.

**Bayesian structural identification** — MCMC for structural identification needs 100,000+ FEM
evaluations. Echelon makes this feasible with realistic models.

**Automated structural design** — population-based optimization (genetic algorithms, CMA-ES)
with 1,000 design candidates evaluated per generation in seconds, not hours.

---

## Design Principles

**No global state.** Every model is an owned value. Multiple models coexist simultaneously.
Population parallelism is a direct consequence.

**Zero-cost where it matters.** The sparse layer uses binary search, pre-allocated workspaces,
and no per-call heap allocation in hot paths. `matvec_into` makes no allocations.
Cholesky will use the standard dense-column-workspace approach.

**Errors, not panics.** Every fallible operation returns `Result`. The error types carry
context fields (row, column, dimension) so the debugging message is actionable.

**Invariants enforced at the type boundary.** `SymCsrMatrix` physically cannot store a
lower-triangle entry. `Permutation` is validated at construction. `CsrMatrix` has `pub(crate)`
fields so the sorted-column invariant cannot be violated externally.

**Separation of concerns.** Sparse storage knows nothing about FEM. The solver knows nothing
about element types. Elements know nothing about the global system. Each layer has one job.

---

## Getting Started (Development)

### Prerequisites

- Rust (stable, 2024 edition) — [rustup.rs](https://rustup.rs)
- For the Python layer: Python 3.10+, `maturin`

### Build and Test

```bash
git clone https://github.com/ihab65/Echelon
cd Echelon

# build and test everything in the workspace
cargo test --workspace

# test just the sparse crate
cargo test -p sparse

# test with parallel feature enabled
cargo test -p sparse --features parallel

# run benchmarks
cargo bench -p sparse
```

### Workspace Structure

```
Echelon/
  Cargo.toml          ← workspace root
  sparse/
    src/
      lib.rs          ← SparseMatrix trait + re-exports
      error.rs        ← SparseError
      coo.rs          ← CooBuilder
      csr/            ← CsrMatrix (matrix.rs, ops.rs, iter.rs, mod.rs)
      sym/            ← SymCsrMatrix (matrix.rs, ops.rs, iter.rs, mod.rs)
      csc/            ← CscMatrix (matrix.rs, ops.rs, iter.rs, mod.rs)
      convert.rs      ← all format conversions
    tests/
    benches/
  solvers/
    src/
      lib.rs
      error.rs
      ordering/       ← graph.rs, permutation.rs, rcm.rs
      cholesky/       ← mod.rs, symbolic.rs, numeric.rs, solve.rs
    test/
```

---

## Comparison With Existing Tools

| | Echelon | OpenSeesPy | FEniCS | ETABS/SAP |
|---|---|---|---|---|
| Structurally correct | ✅ | ✅ | ✅ | ✅ |
| Parallel by design | ✅ | ❌ | Partial | ❌ |
| AI/ML native | ✅ | ❌ | ❌ | ❌ |
| Python API | ✅ (planned) | ✅ | ✅ | Limited |
| Population scale | ✅ | ❌ | ❌ | ❌ |
| Open source | ✅ | ✅ | ✅ | ❌ |
| Pure Rust core | ✅ | ❌ | ❌ | ❌ |
| Structural element library | ✅ (planned) | ✅ | Partial | ✅ |

---

## Roadmap

### Phase 1 — Working Linear Static Solver (Current Focus)
- [x] Sparse storage layer (`CsrMatrix`, `SymCsrMatrix`, `CscMatrix`)
- [x] COO builder with duplicate summation
- [x] Format conversions
- [x] RCM fill-reduction ordering
- [x] Symbolic Cholesky (elimination tree + L pattern)
- [x] Numeric Cholesky (left-looking, dense column workspace)
- [x] Triangular solve (with permutation)
- [x] `fem-core` crate (index newtypes, transforms)
- [x] `materials`: `ElasticUniaxial`
- [x] `elements`: `Truss2d`, `ElasticBeam2d`
- [x] `assembly`: DOF numbering, stiffness assembly, BC application
- [x] Portal frame integration test (compare with analytical solution)

### Phase 2 — Nonlinear Static Analysis
- [x] Newton-Raphson loop in `analysis`
- [ ] `Steel01`, `Concrete01` material models
- [x] Load patterns with combinations
- [x] RC frame pushover analysis (verified against OpenSees via `analysis_integration` tests)

### Phase 3 — Dynamic Analysis
- [x] Consistent and lumped mass matrices
- [x] Newmark integration
- [x] HHT integration
- [x] Nonlinear time history analysis drivers
- [ ] Ground motion input (`PathSeries`)

### Phase 4 — Python API and Population Runner
- [ ] PyO3 bindings (`echelon` Python package)
- [ ] Population runner with Rayon parallelism
- [ ] DataFrame and numpy result extraction
- [ ] Parquet export

### Phase 5 — Research Platform
- [ ] Structural archetypes (RC frame, steel frame, bridge)
- [ ] Surrogate model interface
- [ ] Verification manual (against analytical solutions and OpenSees)
- [ ] Reference datasets published

---

## Contributing

This is currently a personal research project. If you are a structural engineer or computational
mechanics researcher interested in contributing, open an issue to discuss.

---

## License

To be determined.

---

## References

- Davis, T.A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
- Liu, J.W.H. (1986). "A compact row storage scheme for Cholesky factors using elimination trees." *ACM TOMS* 12(2).
- Cuthill, E. & McKee, J. (1969). "Reducing the bandwidth of sparse symmetric matrices." *ACM proceedings*.
- McKenna, F. (2011). "OpenSees: A Framework for Earthquake Engineering Simulation." *Computing in Science & Engineering*.
- Zhao, J. & Sritharan, S. (2007). "Modeling of strain penetration effects in fiber-based analysis of reinforced concrete structures." *ACI Structural Journal* 104(2).
