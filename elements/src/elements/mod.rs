//! Concrete element implementations.
//!
//! ## Module layout
//!
//! ```text
//! elements/
//!   truss2d.rs      — Truss2d: 2D linear elastic truss, energy-based + adjoint
//!   beam2d.rs       — ElasticBeam2d: 2D Euler-Bernoulli beam, closed-form stiffness
//!   shell4.rs       — ElasticShell4: 4-node MITC4 flat-shell, ND material
//! ```
//!
//! ## Adding a new element
//!
//! 1. Create `elements/<name>.rs`.
//! 2. Define the struct (geometry + material reference or owned material).
//! 3. Implement `Element` (always required).
//! 4. If smooth and energy-based, implement `DifferentiableElement`.
//! 5. Implement `Assembleable` to connect to the global system.
//! 6. Add `pub mod <name>;` and re-export below.
//! 7. Add integration tests in `fem-tests/tests/`.

pub mod truss2d;
pub mod beam2d;
pub mod shell4;

pub use truss2d::Truss2d;
pub use beam2d::ElasticBeam2d;
pub use shell4::ElasticShell4;