//! Core traits for uniaxial material models.
//!
//! ## Design: two complementary paths
//!
//! | Path | Materials | Engine |
//! |------|-----------|--------|
//! | **Smooth** — generic over `T` | `ElasticUniaxial` | A (f64) + B (Dual64) |
//! | **History-dependent** — pure f64 | `Steel01`, `Concrete01` | A only; Engine B uses adjoint method |
//!
//! Smooth materials can be threaded through the generic element `energy<T>`
//! function to get automatic differentiation for free.  History-dependent
//! materials (those with `commit`/`revert` state management) cannot be
//! auto-differentiated through their return-mapping algorithms; instead
//! Engine B uses the [`AdjointSensitive`] trait to retrieve analytically
//! correct parameter sensitivities.
//!
//! The split is enforced at the type level: implement [`UniaxialMaterial`]
//! for state-based materials, and additionally implement [`AdjointSensitive`]
//! if you want to participate in Engine B sensitivity analysis.

mod uniaxial;
pub use uniaxial::UniaxialMaterial;

mod adjoint;
pub use adjoint::AdjointSensitive;

mod smooth;
pub use smooth::SmoothUniaxial;