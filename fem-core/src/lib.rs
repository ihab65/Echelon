//! # fem_core
//!
//! Shared vocabulary types for the Echelon FEM engine.
//!
//! This crate sits between `sparse`/`solvers` and the `elements`/`materials`
//! layer.  It is intentionally thin: it defines types, not algorithms.
//!
//! ## What is here
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`ids`] | [`NodeId`], [`ElemId`], [`GlobalDof`], [`LocalDof`] — typed index newtypes |
//! | [`dof_map`] | [`DofMap`] — maps local element DOFs to global DOF indices |
//! | [`model`] | [`ModelDim`] — declares `ndm` and `ndf` for the analysis |
//! | [`dense`] | Fixed-size matrix ops: `matmul`, `transform_stiffness`, `mat_as_slice` |
//! | [`transform`] | [`CoordTransf2d`] — 2D element coordinate transforms |
//!
//! ## Dependency position
//!
//! ```text
//! elements / materials
//!       ↓
//!   fem_core          ← this crate
//!       ↓
//!  sparse / solvers
//! ```
//!
//! `fem_core` does **not** depend on `sparse` or `solvers`.  Types that need
//! to cross the boundary (e.g. passing `DofMap::as_usize_slice()` to
//! `scatter_add`) do so at the `assembly` layer, not here.

pub mod ids;
pub mod dof_map;
pub mod model;
pub mod dense;
pub mod transform;

// Flat re-exports for the most commonly used types
pub use ids::{NodeId, ElemId, GlobalDof, LocalDof};
pub use dof_map::DofMap;
pub use model::ModelDim;
pub use transform::CoordTransf2d;

// -----------------------------------------------------------------
// Compile-time Send + Sync assertions
//
// Every type in this crate must be Send + Sync — this is the commitment
// that makes population-parallel analysis possible.  If a future change
// accidentally introduces interior mutability, the compiler will catch it
// here.
// -----------------------------------------------------------------
#[allow(dead_code)]
fn _assert_send_sync() {
    fn req<T: Send + Sync>() {}
    req::<NodeId>();
    req::<ElemId>();
    req::<GlobalDof>();
    req::<LocalDof>();
    req::<DofMap>();
    req::<ModelDim>();
    req::<CoordTransf2d>();
}