//! Iterative solvers for large sparse systems.
//!
//! # Not yet available
//!
//! Iterative solvers (CG, PCG, MINRES, GMRES) are planned for a future release.
//! They are most useful when:
//!
//! - The system is too large for direct factorization to be memory-feasible.
//! - Only a matrix-vector product is available (matrix-free problems).
//! - The system is mildly ill-conditioned and a good preconditioner is known.
//!
//! For all current Echelon analyses (linear static, nonlinear static, transient
//! with moderate DOF counts), [`crate::linear::CholeskySolver`] is the
//! appropriate solver.
//!
//! ## Planned types
//!
//! | Type | Algorithm | Notes |
//! |------|-----------|-------|
//! | `ConjugateGradient` | Preconditioned CG | SPD systems only |
//! | `Minres` | MINRES | Symmetric indefinite (near-buckling) |
//! | `Gmres` | GMRES | General non-symmetric (future unsymmetric elements) |
//!
//! ## Planned preconditioners
//!
//! | Type | Description |
//! |------|-------------|
//! | `DiagonalPreconditioner` | Jacobi / diagonal scaling |
//! | `IncompleteCholesky` | IC(0) fill-level incomplete factorization |
//! | `BlockJacobi` | Independent per-subdomain solve (parallel) |