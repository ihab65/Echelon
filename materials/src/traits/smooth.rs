//! [`SmoothUniaxial`] — generic-`T` interface for smooth, history-free materials.
//!
//! A smooth material has no internal state and no path dependence.  Its
//! stress–strain relationship is a smooth function of strain (and possibly of
//! parameters), making it fully compatible with automatic differentiation.
//!
//! ## How it fits into the element energy machinery
//!
//! Elements that implement `DifferentiableElement` define an `energy<T>` method
//! that is generic over any numeric type `T` satisfying the `DualNum` bound.
//! When a material participates in that energy computation, it too must be
//! generic over `T`.  [`SmoothUniaxial`] provides exactly this interface.
//!
//! ```rust,ignore
//! // Inside Truss2d::energy<T: DualNum>(…)
//! let strain: T = (u2_local - u1_local) / self.length;
//! let stress: T = self.material.smooth_stress(strain);
//! let energy: T = T::from(0.5) * stress * strain * self.area * self.length;
//! ```
//!
//! For `T = f64` this is ordinary evaluation.
//! For `T = Dual64` the derivative flows through automatically.
//!
//! ## Relationship to `UniaxialMaterial`
//!
//! `ElasticUniaxial` implements *both* traits:
//! - `UniaxialMaterial` (f64 interface) for use in the Newton-Raphson loop.
//! - `SmoothUniaxial<T>` (generic interface) for use inside `energy<T>`.
//!
//! Inelastic materials implement only `UniaxialMaterial` (and optionally
//! `AdjointSensitive`).  They *cannot* implement `SmoothUniaxial` because
//! their stress–strain map is non-smooth.

use num_traits::{One, Zero};
use std::ops::{Add, Mul};

/// A material whose stress is a smooth function of strain, expressible
/// generically over any numeric type `T`.
///
/// # Bounds on `T`
///
/// `T` must support at minimum: addition, multiplication, and conversion
/// from `f64` scalars.  In practice `T` will be `f64` (Engine A) or
/// `num_dual::Dual64` / `num_dual::HyperDual64` (Engine B).
///
/// The bound `T: Copy + Add<Output=T> + Mul<Output=T> + Zero + One` covers
/// all cases without requiring the full `num_dual::DualNum` trait, keeping
/// the `materials` crate independent of `num-dual` by default.
pub trait SmoothUniaxial<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero + One,
{
    /// Compute stress as a smooth function of strain.
    ///
    /// For linear elastic: `σ(ε) = E · ε`.
    /// For nonlinear elastic: any smooth, differentiable function.
    ///
    /// # Arguments
    /// * `strain` — the current strain as a generic numeric value
    fn smooth_stress(&self, strain: T) -> T;

    /// Compute the tangent modulus `dσ/dε` as a smooth function of strain.
    ///
    /// For linear elastic: `E` (constant).
    ///
    /// When `T = Dual64`, calling `smooth_stress` already gives you the
    /// derivative via the dual part, so this method is optional for elements
    /// that use AD.  It is provided as a fast path for f64-only evaluation.
    fn smooth_tangent(&self, strain: T) -> T;
}
