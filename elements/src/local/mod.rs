//! Local kinematics — geometry and coordinate-transform helpers used by
//! concrete element implementations.
//!
//! This module isolates the small set of per-element geometric calculations
//! (axial strain, curvature, coordinate transformation) so that the element
//! files stay focused on constitutive integration and the trait implementations.
//!
//! ## Contents
//!
//! | Module         | Contents                                           |
//! |----------------|----------------------------------------------------|
//! | `truss.rs`     | Axial strain, local displacements for 2D truss     |
//! | `beam.rs`      | Euler-Bernoulli curvature, beam kinematics         |
//!
//! ## Design note
//!
//! Functions here are generic over `T` where possible, allowing reuse in
//! both the f64 Newton-Raphson path and the dual-number `energy<T>` path.

pub mod truss;
pub mod beam;
pub mod gauss;
pub mod isopar;
pub mod shell;