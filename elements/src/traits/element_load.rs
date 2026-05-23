//! Parameters describing a load applied to an element span.
//!
//! [`ElementLoadParams`] is a pure data enum — no methods, no state.
//! It is passed to [`crate::traits::Element::equivalent_nodal_forces`] which converts it
//! into the equivalent nodal force vector in global coordinates.
//!
//! ## Sign convention
//!
//! All load components are in the **global** coordinate system:
//! - `wx` — force per unit length in the global X direction
//! - `wy` — force per unit length in the global Y direction
//! - Positive Y is upward; a downward distributed load uses `wy < 0`.
//!
//! ## Reference
//!
//! Fixed-end reaction formulas follow the standard Euler-Bernoulli
//! beam tables (e.g., Roark's Formulas for Stress and Strain, Table 8).

/// Parameters for a load applied along the span of a structural element.
///
/// Passed to [`crate::traits::Element::equivalent_nodal_forces`] to compute the
/// statically equivalent nodal forces (fixed-end reactions, reversed).
#[derive(Debug, Clone, PartialEq)]
pub enum ElementLoadParams {
    /// Uniformly distributed load along the full element length.
    ///
    /// `wx` and `wy` are force-per-unit-length in global X and Y.
    /// Either component may be zero.
    Uniform {
        /// Distributed load in global X (N/m). Positive → rightward.
        wx: f64,
        /// Distributed load in global Y (N/m). Positive → upward.
        wy: f64,
    },

    /// Concentrated (point) load at a fractional position along the element.
    ///
    /// `xi` is the dimensionless distance from node I: `0.0` is at node I,
    /// `1.0` is at node J, `0.5` is at mid-span.
    Point {
        /// Concentrated force in global X (N). Positive → rightward.
        px: f64,
        /// Concentrated force in global Y (N). Positive → upward.
        py: f64,
        /// Fractional position from node I (clamped to [0.0, 1.0]).
        xi: f64,
    },

    /// Linearly varying (trapezoidal) distributed load.
    ///
    /// The load intensity varies linearly from the values at node I
    /// (`wx_i`, `wy_i`) to the values at node J (`wx_j`, `wy_j`).
    /// A triangular load has one end at zero.
    Trapezoidal {
        /// Distributed load in X at node I (N/m).
        wx_i: f64,
        /// Distributed load in X at node J (N/m).
        wx_j: f64,
        /// Distributed load in Y at node I (N/m).
        wy_i: f64,
        /// Distributed load in Y at node J (N/m).
        wy_j: f64,
    },
}