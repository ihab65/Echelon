//! # elements
//!
//! Structural finite element formulations for the Echelon FEM engine.
//!
//! ## Architecture
//!
//! Elements are organized into four layers, each with a dedicated module:
//!
//! ```text
//! src/
//!   lib.rs              — this file: module declarations + flat re-exports
//!   traits/
//!     mod.rs            — re-exports all element traits
//!     element.rs        — Element (Engine A: stiffness + residual)
//!     differentiable.rs — DifferentiableElement (Engine B: energy<T>)
//!     assembleable.rs   — Assembleable (DOF map + adjoint ∂f/∂θ)
//!   local/
//!     mod.rs            — re-exports local kinematics helpers
//!     truss.rs          — 2D axial kinematics (strain, stiffness, f_int)
//!     beam.rs           — Euler-Bernoulli kinematics (local ke, f_int)
//!   elements/
//!     mod.rs            — re-exports all concrete elements
//!     truss2d.rs        — Truss2d: energy-based 2D truss
//!     beam2d.rs         — ElasticBeam2d: closed-form 2D beam
//! ```
//!
//! ## Adding a new element
//!
//! 1. Add any new local kinematics to `local/`.
//! 2. Create `elements/<name>.rs` implementing `Element` and optionally
//!    `DifferentiableElement` + `Assembleable`.
//! 3. Register it in `elements/mod.rs` and re-export below.
//! 4. Add integration tests in `fem-tests/tests/`.
//!
//! ## Dependency position
//!
//! ```text
//! fem-tests (integration tests)
//!     ↓
//! elements          ← this crate
//!     ↓           ↓
//! materials    fem_core
//!                  ↓
//!             sparse / solvers
//! ```

pub mod traits;
pub mod local;
pub mod error;
mod elements;

// ---- Flat re-exports: traits ----
pub use traits::Element;
pub use traits::DifferentiableElement;
pub use traits::Assembleable;
pub use traits::ElementLoadParams;
pub use error::ElementError;

// ---- Flat re-exports: concrete elements ----
pub use elements::Truss2d;
pub use elements::ElasticBeam2d;