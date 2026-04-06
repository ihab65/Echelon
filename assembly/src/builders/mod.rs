//! Pure matrix/vector scatter operations.
//!
//! Every function in this module is a pure function:
//! `(Model, &mut GlobalMatrix/Vec) → Result<()>`.
//! They have no hidden dependencies and produce the same output for the
//! same model state — making them trivially parallelisable at the population
//! level.
//!
//! ## Assembly order per Newton iteration
//!
//! ```text
//! 1. assemble_stiffness     → K   (zeros K first)
//! 2. assemble_load_vector   → f_ext
//! 3. assemble_internal_force → f_int
//! 4. constraints::apply_dirichlet_bcs(K, f_residual)  ← after all scatter
//! 5. solver.factorize(K) + solver.solve(f_residual)
//! 6. u_global += delta_u
//! ```
//!
//! ## Mass assembly (once per topology, before Eigen/Transient analysis)
//!
//! ```text
//! assemble_mass(&model, &mut M)
//! assemble_self_weight(&model, -9.81, &mut f_gravity)
//! ```
//!
//! ## Adjoint sensitivity (once per converged load step, Engine B)
//!
//! ```text
//! for param_idx in 0..total_n_params(&model) {
//!     assemble_partial_residual(&model, param_idx, &mut dp_dtheta)?;
//!     dJ_dtheta[param_idx] = -lambda.dot(&dp_dtheta);
//! }
//! ```

pub mod stiffness;
pub mod internal;
pub mod mass;
pub mod external;
pub mod adjoint;
pub mod damping;

pub use stiffness::assemble_stiffness;
pub use internal::assemble_internal_force;
pub use mass::{assemble_mass, assemble_self_weight};
pub use external::assemble_load_vector;
pub use adjoint::{assemble_partial_residual, total_n_params};
pub use damping::{build_rayleigh_damping, rayleigh_coefficients};