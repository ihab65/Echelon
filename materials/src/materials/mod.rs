//! Concrete material model implementations.
//!
//! ## Module layout
//!
//! ```text
//! materials/
//!   elastic.rs   — ElasticUniaxial (linear, smooth, autodiff-ready)
//!   steel01.rs   — Steel01 (bilinear elasto-plastic, history-dependent)  [stub]
//!   concrete01.rs— Concrete01 (Kent-Park, history-dependent)              [stub]
//! ```
//!
//! ## Adding a new material
//!
//! 1. Create `materials/<name>.rs`.
//! 2. Implement [`UniaxialMaterial`] (always required).
//! 3. If the material is smooth (no yield surface), also implement
//!    [`SmoothUniaxial<T>`] for generic `T`.
//! 4. If the material has history dependence and should participate in
//!    Engine B sensitivity analysis, implement [`AdjointSensitive`].
//! 5. Add `pub mod <name>;` and a re-export below.

mod elastic;
pub use elastic::ElasticUniaxial;

// Stubs — uncomment as implemented:
// mod steel01;
// pub use steel01::Steel01;
//
// mod concrete01;
// pub use concrete01::Concrete01;