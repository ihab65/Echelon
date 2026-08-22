<div align="center">

# Echelon

**A structural finite element engine built in Rust — parallel by design, Python-accessible, and AI-native.**

*Currently in active development. The Rust core — sparse storage, direct solvers, modal analysis, and linear-elastic FEM assembly/analysis — is complete and tested. Nonlinear materials, the Python API, and the population runner are next.*

</div>

---

## What Is Echelon?

Echelon is a structural analysis engine that addresses a gap in the existing landscape: there is no tool that is simultaneously **structurally correct**, **parallel by design**, **AI/ML-native**, **Python-accessible**, and **built for population-scale workflows**.

OpenSees is powerful but carries 1990s architecture — single-threaded, global mutable state, not designed for programmatic access at scale. Commercial tools are black boxes with no ML integration. Pure-Python FEM tools are too slow for anything beyond toy problems.

Echelon is built from the ground up for the workflow that modern structural engineering research actually needs:

> **⚠️ Target workflow — not yet runnable.** The Python package and population runner below are the goal of Phase 4. Today Echelon is a Rust library; you build models and run analyses in Rust directly (see [What You Can Do With It Today](#what-you-can-do-with-it-today)).

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
                     │  echelon (PyO3) │  ← Python API (planned)
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌────────────────┐
        │ assembly │   │ analysis │   │ bevy (planned) │
        └─────┬────┘   └────┬─────┘   │   , optional   │
              │             │         └────────────────┘
    ┌─────────┴─────────────┘
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│ elements │ │materials │
└────┬─────┘ └────┬─────┘
     │             │
     └──────┬──────┘
            ▼
       ┌──────────┐
       │   core   │  (index types, dense math, 2D/3D transforms)
       └────┬─────┘
            │
            ▼
       ┌──────────┐
       │ solvers  │  ← Cholesky, LDLT, RCM ordering, Lanczos eigen
       └────┬─────┘
            │
            ▼
       ┌──────────┐
       │  sparse  │  ← CSR, SymCSR, CSC, COO (foundation)
       └──────────┘
```

A companion crate, `fem-tests`, holds structural-level integration tests (frames, trusses, beams) verified against analytical solutions.

The entire analysis stack — from sparse storage through the solver — is pure Rust with no external linear algebra dependencies. No LAPACK, no BLAS, no unsafe FFI.

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
- Matrix-market I/O and benchmarks
- Comprehensive test suite with integration tests

### ✅ `solvers` — Direct Solvers, Ordering, and Modal Analysis

RCM (Reverse Cuthill-McKee) fill-reduction ordering is fully implemented:
- `Graph` — adjacency structure from `SymCsrMatrix`, O(nnz) two-pass construction
- `Permutation` — validated newtype with `permute_sym` for symmetric matrix reordering
- `rcm()` — full algorithm with pseudo-peripheral node detection and degree-ordered BFS

`SparseSolver` interface (three-phase: `analyze` → `factorize` → `solve`) is implemented.
Included are:
- Symmetric positive-definite **Cholesky** (symbolic via elimination trees, numeric left-looking with dense column workspace)
- Symmetric indefinite **LDLᵀ**
- **Lanczos eigensolver** for modal analysis — eigenvalues, mode shapes (M-orthonormal), frequencies and periods

### ✅ `fem-core` — Complete

Foundation types for the finite element domain:
- Node and element ID newtypes.
- Small dense vector/matrix math (2D and 3D).
- `CoordTransf2d` and `CoordTransf3d`, `DofMap` implementations.

### ✅ `materials` — Complete (elastic only)

Architecture supporting smooth autodiff-friendly materials and history-dependent materials
(commit/revert state model). Reference implementations:
- `ElasticUniaxial`
- `ElasticIsotropic` (ND material)

**Note:** no inelastic material models exist yet (`Steel01`, `Concrete01` are planned). Nonlinear
analysis currently means nonlinear solution algorithms with linear-elastic constitutive response.

### ✅ `elements` — Complete (linear elastic)

Implementation of finite elements:
- `Truss2d` (linear, energy-based implementation)
- `ElasticBeam2d` (frame element, linear elastic)
- `ElasticShell4` (4-node MITC4 flat shell, 24 DOF, ND material, isoparametric formulation with Gauss integration)

### ✅ `assembly` — Complete

The core system for holding models and assembling sparse matrices.
- `Model` structure for managing nodes, elements, constraints, and DOFs.
- Connectivity and topology graph generation to determine matrix non-zero patterns.
- Stiffness, mass, and Rayleigh damping matrix assembly; internal force and self-weight vectors.
- Dirichlet constraints (`SpConstraint`) and reaction recovery.
- Loads: nodal, element, gravity, seismic (ground motion / `UniformExcitation`), time series
  (`ConstantSeries`, `LinearSeries`, `PathSeries`), and load combinations.
- Adjoint sensitivity scaffolding (`assemble_partial_residual`) — experimental, see [TODO](#experimental--at-risk).

### ✅ `analysis` — Complete

Core algorithmic routines for solving structural equilibrium:
- `LinearStatic` driver.
- Nonlinear `NewtonRaphson` and `ModifiedNewton` solution algorithms.
- Convergence criteria: displacement, unbalance (force), energy increment.
- Integrators for `StaticNonlinear` (`LoadControl`, `DispControl`).
- Integrators for `Transient` dynamics (`Newmark`, `HHT-α`) with a nonlinear time-history driver.
- `NodeRecorder` and `ElementRecorder` for structured response output.

### ✅ `fem-tests` — Structural Verification

Integration tests at the structure level, verified against analytical closed-form solutions:
single spring, 1D bar, V-truss, cantilever, fixed-fixed beam, simply supported beam, portal
frame under lateral load, indeterminate Pratt truss, and reanalysis with shared topology.

### Verification status

- ✅ Verified against **analytical closed forms** (beams, trusses, frames, eigen pairs) with tight
  relative tolerances (1e-9).
- ❌ **No OpenSees cross-validation yet.** The earlier README claim of "verified against OpenSees"
  was incorrect — verification is analytical, which is a stronger test where it applies.
- ⚠️ Dynamics and modal analysis are verified at the **algorithm level** (tangent formation, eigen
  residuals vs. hand-computed systems) but not yet **end-to-end** on real structures.

### 🔄 Planned

| Crate | Status |
|---|---|
| `echelon` (Python) | Not started — PyO3 bindings, population runner |
| `bevy` (visualization) | Not started — optional |

---

## What You Can Do With It Today

Everything below is implemented and tested, but is a **Rust API** — there is no Python interface yet.

- **Build structural models in code**: 2D/3D truss, frame, and MITC4 shell models with nodal
  coordinates, elements, elastic materials, and boundary conditions.
- **Assemble** stiffness, lumped mass, and Rayleigh damping matrices; apply nodal/element/gravity
  loads, ground-motion excitation, and load combinations.
- **Run static analysis**: linear static, and nonlinear-elastic static with Newton-Raphson or
  Modified-Newton, load or displacement control, and your choice of convergence criterion.
- **Run modal analysis**: extract natural frequencies, periods, and mode shapes via Lanczos.
- **Time-history scaffolding**: Newmark and HHT-α integrators with a transient driver are coded
  and unit-tested, ready for seismic analysis once end-to-end verification lands.
- **Recover results**: reaction forces, nodal displacement/velocity/acceleration histories via
  recorders, element axial/shear/moment at nodes.
- **Parallelism**: `sparse` ships a `rayon` feature; the design (stateless models, no global
  state) is built to parallelize, but the population-scale runner does not exist yet.

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
and no per-call heap allocation in hot paths. `matvec_into` makes no allocations. Cholesky,
permutations, and the transient integrators use pre-allocated workspaces.

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
- For the (future) Python layer: Python 3.10+, `maturin`

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
      linear/         ← cholesky.rs, ldlt.rs
      eigen/          ← lanczos.rs
      cholesky/       ← mod.rs, symbolic.rs, numeric.rs, solve.rs
    tests/
  fem-core/           ← ids, dense math, transforms, dof maps
  materials/          ← elastic materials + smooth/history traits
  elements/           ← truss2d, beam2d, shell4, local routines
  assembly/           ← model, builders, loads, constraints, topology
  analysis/           ← drivers, algorithms, integrators, recorders
  fem-tests/          ← structural-level integration tests
```

---

## Comparison With Existing Tools

| | Echelon | OpenSeesPy | FEniCS | ETABS/SAP |
|---|---|---|---|---|
| Structurally correct | ✅ | ✅ | ✅ | ✅ |
| Parallel by design | ✅ | ❌ | Partial | ❌ |
| Pure Rust core | ✅ | ❌ | ❌ | ❌ |
| Open source | ✅ | ✅ | ✅ | ❌ |
| Python API | 🚧 (planned) | ✅ | ✅ | Limited |
| AI/ML native | 🚧 (planned) | ❌ | ❌ | ❌ |
| Population scale | 🚧 (planned) | ❌ | ❌ | ❌ |
| Structural element library | ⚠️ (3 elastic elements) | ✅ | Partial | ✅ |

Legend: ✅ implemented today · 🚧 planned · ⚠️ partial

---

## Roadmap

### Phase 1 — Linear Static Core ✅ Complete
- [x] Sparse storage layer (`CsrMatrix`, `SymCsrMatrix`, `CscMatrix`)
- [x] COO builder with duplicate summation
- [x] Format conversions
- [x] RCM fill-reduction ordering
- [x] Symbolic Cholesky (elimination tree + L pattern)
- [x] Numeric Cholesky (left-looking, dense column workspace)
- [x] Triangular solve (with permutation)
- [x] LDLᵀ solver (symmetric indefinite)
- [x] Lanczos eigensolver (modal analysis)
- [x] `fem-core` crate (index newtypes, 2D/3D transforms)
- [x] `materials`: `ElasticUniaxial`, `ElasticIsotropic`
- [x] `elements`: `Truss2d`, `ElasticBeam2d`, `ElasticShell4`
- [x] `assembly`: DOF numbering, stiffness/mass/damping assembly, BC application
- [x] `analysis`: `LinearStatic`, `StaticNonlinear` (Newton/Modified-Newton), load/displacement control
- [x] Loads: nodal, element, gravity, seismic, `PathSeries`, load combinations
- [x] Reaction recovery and recorders
- [x] Structural integration tests (beams, trusses, frames) vs. analytical solutions

### Phase 2 — Nonlinear Static Analysis 🔄 In Progress
- [x] Newton-Raphson and Modified-Newton loops in `analysis`
- [x] Load patterns with combinations
- [x] Incremental load-control analysis (pushover driver) — currently linear-elastic only, verified vs. analytical
- [ ] `Steel01`, `Concrete01` inelastic material models
- [ ] Fiber-section engine + nonlinear beam-column element (distributed plasticity)

### Phase 3 — Dynamic Analysis 🔄 In Progress
- [x] Lumped mass matrices
- [x] Newmark integration
- [x] HHT-α integration
- [x] Nonlinear time-history analysis driver
- [x] Ground motion input (`PathSeries`, `GroundMotion`, `UniformExcitation`)
- [ ] Consistent mass matrices
- [ ] End-to-end dynamic response verification (e.g. free-vibration period, NLTHA vs. known solution)

### Phase 4 — Python API and Population Runner ⏳ Not Started
- [ ] PyO3 bindings (`echelon` Python package)
- [ ] Population runner with Rayon parallelism
- [ ] DataFrame and numpy result extraction
- [ ] Parquet export

### Phase 5 — Research Platform ⏳ Not Started
- [ ] Structural archetypes (RC frame, steel frame, bridge)
- [ ] Surrogate model interface
- [ ] Verification manual (against analytical solutions and OpenSees)
- [ ] Reference datasets published

---

## TODO / Next Steps

### High priority
- [ ] `Steel01`, `Concrete01` inelastic material models
- [ ] Fiber-section engine + nonlinear beam-column element (distributed plasticity)
- [ ] Consistent mass matrices
- [ ] End-to-end dynamic (NLTHA) verification against a known solution
- [ ] End-to-end modal analysis on a real frame structure
- [ ] Shell element structural-level verification test
- [ ] OpenSees cross-validation suite

### Medium priority
- [ ] PyO3 bindings (`echelon` Python package)
- [ ] Population runner with Rayon parallelism
- [ ] DataFrame / numpy result extraction and Parquet export
- [ ] Structural archetypes (RC frame, steel frame, bridge)
- [ ] Verification manual
- [ ] Reference datasets published
- [ ] Optional `bevy` visualization

### Experimental / at-risk
- [ ] **Gradient-based sensitivity (adjoint method)** — the `assemble_partial_residual` /
  `partial_residual_wrt_param` scaffolding is currently a **purely theoretical prototype**.
  Exact gradients through history-dependent materials require a return-mapping-aware adjoint,
  and this path **may be discarded in the future due to complexity**.

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
