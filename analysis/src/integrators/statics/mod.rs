//! Static integrators — load and displacement control.
//!
//! Static integrators advance the analysis by incrementing a control
//! parameter: either the load factor λ (load control) or the displacement
//! at a specific DOF (displacement control).

pub mod disp_control;
pub mod load_control;