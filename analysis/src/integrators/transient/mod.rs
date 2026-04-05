//! Transient (dynamic) integrators — Newmark and HHT.
//!
//! Transient integrators advance the analysis through time by approximating
//! the time derivatives of displacement (velocity `u̇` and acceleration `ü`)
//! using the displacement history.
//!
//! All transient integrators augment the static tangent stiffness `K_T`
//! with mass and damping contributions to form an **effective stiffness**:
//!
//! ```text
//! K_eff = K_T + (γ / βΔt) C + (1 / βΔt²) M
//! ```
//!
//! and an **effective load vector** that accounts for the inertia and
//! damping terms carried over from the previous time step.
//!
//! The Echelon transient integrators require a global mass matrix `M` and
//! (optionally) a global damping matrix `C`, both of which are assembled
//! by the `assembly` crate before the first time step.

pub mod hht;
pub mod newmark;