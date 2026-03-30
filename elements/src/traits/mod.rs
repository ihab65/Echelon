//! Core element traits.
//!
//! ## Trait hierarchy
//!
//! ```text
//! Element                  — minimal Engine A interface (stiffness + residual)
//!   └─ DifferentiableElement  — Engine B extension: energy<T> + autodiff hooks
//!        └─ Assembleable       — global DOF map + adjoint ∂f_int/∂θ
//! ```
//!
//! Every concrete element must implement `Element`.
//! Elements that support Engine B (differentiable physics) implement
//! `DifferentiableElement` in addition.
//! The `Assembleable` trait bridges elements to the global system.

mod element;
mod differentiable;
mod assembleable;

pub use element::Element;
pub use differentiable::DifferentiableElement;
pub use assembleable::Assembleable;