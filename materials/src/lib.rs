//! # materials
//!
//! Constitutive models for the Echelon FEM engine.
//!
//! ## Architecture
//!
//! Materials are separated into two implementation paths:
//!
//! **Path A — Smooth materials** (generic over `T: DualNum`):
//! - No internal state; stress is a smooth function of strain.
//! - Implement [`SmoothUniaxial<T>`] so they can be threaded through
//!   element `energy<T>` functions for automatic differentiation.
//! - Example: [`ElasticUniaxial`].
//!
//! **Path B — History-dependent materials** (pure f64):
//! - Internal state (plastic strain, back-stress, damage variable).
//! - Implement [`UniaxialMaterial`] for the Newton-Raphson loop.
//! - Optionally implement [`AdjointSensitive`] for Engine B sensitivity.
//! - Example: `Steel01`, `Concrete01` (planned).
//!
//! ## Dependency position
//!
//! ```text
//! elements
//!     ↓
//! materials    ← this crate
//!     ↓
//! (no sparse / solvers dependency — materials are pure math)
//! ```
//!
//! ## Crate layout
//!
//! ```text
//! src/
//!   lib.rs              — this file: re-exports + module declarations
//!   traits/
//!     mod.rs            — re-exports all traits
//!     uniaxial.rs       — UniaxialMaterial (f64 Newton-Raphson interface)
//!     nd_material.rs    — NdMaterial (multi-dim f64 Newton-Raphson interface)
//!     adjoint.rs        — AdjointSensitive (Engine B parameter sensitivity)
//!     smooth.rs         — SmoothUniaxial<T> (generic autodiff interface)
//!   materials/
//!     mod.rs            — re-exports all concrete materials
//!     elastic.rs        — ElasticUniaxial
//!     elastic_isotropic.rs — ElasticIsotropic (ND linear elastic)
//!     steel01.rs        — Steel01 (planned)
//!     concrete01.rs     — Concrete01 (planned)
//! ```

pub mod error;
mod materials;
pub mod traits;

// ---- Flat re-exports ----

// Traits (users import these to write generic code)
pub use error::MaterialError;
pub use traits::AdjointSensitive;
pub use traits::NdMaterial;
pub use traits::SmoothUniaxial;
pub use traits::UniaxialMaterial;

// Concrete materials
pub use materials::ElasticUniaxial;
pub use materials::{ElasticIsotropic, NdOrder};
