//! # analysis
//!
//! The nonlinear equation solver and time-stepping layer for the Echelon engine.
//!
//! This crate sits above `assembly` and `solvers` in the dependency graph,
//! orchestrating them into complete static and dynamic analysis workflows.
//!
//! ```text
//!                    ┌───────────────────┐
//!                    │      analysis      │  ← this crate
//!                    │                   │
//!                    │  drivers          │  ← outer load/time loops
//!                    │    └─ algorithms  │  ← Newton-Raphson, Modified Newton
//!                    │    └─ integrators │  ← LoadControl, Newmark, HHT
//!                    │    └─ tests       │  ← convergence criteria
//!                    └────────┬──────────┘
//!                             │
//!               ┌─────────────┼─────────────┐
//!               ▼             ▼             ▼
//!          ┌─────────┐  ┌──────────┐  ┌─────────┐
//!          │assembly │  │ solvers  │  │ sparse  │
//!          └─────────┘  └──────────┘  └─────────┘
//! ```
//!
//! ## What this crate provides
//!
//! | Module             | Contents |
//! |--------------------|----------|
//! | [`error`]          | [`AnalysisError`] — EERS diagnostic codes |
//! | [`system`]         | [`GlobalSystem`] — pre-allocated analysis buffers |
//! | [`tests`]          | [`ConvergenceTest`] — norm-based stopping criteria |
//! | [`algorithms`]     | [`EquiSolnAlgo`] — Newton-Raphson, Modified Newton, Linear |
//! | [`integrators`]    | [`Integrator`] — LoadControl, DispControl, Newmark, HHT |
//! | [`drivers`]        | [`AnalysisDriver`] — top-level linear static, nonlinear static, transient |
//!
//! ## Architecture overview
//!
//! The analysis crate is built around a strict **separation of concerns**:
//!
//! - The **`Driver`** owns the outer loop (`for step in 0..steps`). It is the
//!   only place that knows how many load or time steps to execute.
//! - The **`Integrator`** advances the load level (λ) or time (t) and forms
//!   the effective unbalanced force vector for the current step.
//! - The **`Algorithm`** runs the inner Newton-Raphson loop until either
//!   convergence is declared by the **`ConvergenceTest`** or the iteration
//!   limit is exceeded.
//! - The **`GlobalSystem`** holds the three large pre-allocated buffers
//!   (`K_T`, `R`, `ΔU`) that the assembly and solver write into. Zero
//!   heap allocations occur inside the inner loop.
//!
//! ## Typical usage — linear static (elastic analysis)
//!
//! ```rust,ignore
//! use assembly::{Model, Node, build_pattern, assemble_stiffness,
//!                assemble_load_vector, assemble_internal_force};
//! use assembly::constraints::apply_dirichlet_bcs;
//! use analysis::drivers::linear_static::LinearStatic;
//! use analysis::drivers::AnalysisDriver;
//! use fem_core::{ModelDim, NodeId};
//!
//! let mut model = build_my_model();
//! let mut driver = LinearStatic::new();
//! let ok = driver.analyze(&mut model, 1)?;
//! assert!(ok);
//! // model.u_global now contains the solution
//! ```
//!
//! ## Typical usage — nonlinear static pushover
//!
//! ```rust,ignore
//! use analysis::algorithms::newton::NewtonRaphson;
//! use analysis::integrators::statics::load_control::LoadControl;
//! use analysis::tests::unbalance::NormUnbalance;
//! use analysis::drivers::nonlinear_static::StaticNonlinear;
//! use analysis::drivers::AnalysisDriver;
//!
//! let test      = Box::new(NormUnbalance::new(1e-6));
//! let algorithm = Box::new(NewtonRaphson::new(test, 25));
//! let integrator = Box::new(LoadControl::new(0.1));
//! let mut driver = StaticNonlinear::new(algorithm, integrator, model.n_dof());
//!
//! let ok = driver.analyze(&mut model, 10)?;  // 10 load steps of Δλ = 0.1
//! ```
//!
//! ## Design principles
//!
//! **Zero allocations in the inner loop.** [`GlobalSystem`] pre-allocates
//! every buffer once. The Newton-Raphson loop writes into these buffers
//! in-place, never calling `Vec::new()` or `Box::new()`.
//!
//! **Hot-swappable strategies.** The `Driver` holds its `Algorithm` and
//! `Integrator` as `Box<dyn Trait>`. Switching from Newton-Raphson to
//! Modified Newton requires changing one line of construction code, not
//! re-architecting the analysis.
//!
//! **EERS errors.** Every failure carries a dot-separated diagnostic code
//! and a structured help message explaining the likely structural cause.
//! Divergence in a population run is a catchable `AnalysisError::Divergence`,
//! not a panic.

pub mod error;
pub mod system;
pub mod convergence;
pub mod algorithms;
pub mod integrators;
pub mod drivers;
pub mod recorder;

// -----------------------------------------------------------------
// Flat re-exports — the most commonly imported items
// -----------------------------------------------------------------

pub use error::AnalysisError;
pub use system::GlobalSystem;
pub use convergence::ConvergenceTest;
pub use algorithms::EquiSolnAlgo;
pub use integrators::Integrator;
pub use drivers::AnalysisDriver;
pub use recorder::Recorder;
pub use recorder::NodeRecorder;
pub use recorder::ElementRecorder;