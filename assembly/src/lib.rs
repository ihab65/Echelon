//! # assembly
//!
//! The structural model container and FEM assembly layer for the Echelon engine.
//!
//! This crate sits between the element/material definitions and the solver:
//!
//! ```text
//!                    ┌──────────────┐
//!                    │   analysis   │  ← Newton-Raphson, Newmark, Eigen drivers
//!                    └──────┬───────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │   assembly   │  ← this crate
//!                    └──────┬───────┘
//!              ┌────────────┼────────────┐
//!              ▼            ▼            ▼
//!         ┌─────────┐ ┌─────────┐ ┌──────────┐
//!         │elements │ │materials│ │ fem_core │
//!         └─────────┘ └─────────┘ └──────────┘
//!                           │
//!              ┌────────────┴────────────┐
//!              ▼                         ▼
//!         ┌─────────┐             ┌──────────┐
//!         │ solvers │             │  sparse  │
//!         └─────────┘             └──────────┘
//! ```
//!
//! ## What this crate provides
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`model`] | [`Model`], [`Node`] — the central structural container |
//! | [`constraints`] | [`SpConstraint`], [`apply_dirichlet_bcs`] |
//! | [`state`] | [`commit_state`], [`revert_state`] |
//! | [`topology`] | [`build_pattern`], [`count_dofs`], [`validate_dof_maps`] |
//! | [`kinematics`] | [`extract_local_u`] — stateless DOF extraction |
//! | [`loads`] | [`LoadPattern`], [`NodalLoad`], [`TimeSeries`], series impls |
//! | [`builders`] | All five scatter functions (stiffness, internal, mass, external, adjoint) |
//! | [`error`] | [`AssemblyError`], EERS diagnostic codes |
//!
//! ## Typical analysis workflow
//!
//! ```rust,ignore
//! use assembly::{Model, Node, build_pattern};
//! use assembly::builders::*;
//! use assembly::constraints::{SpConstraint, apply_dirichlet_bcs};
//! use assembly::state::{commit_state, revert_state};
//! use assembly::loads::{NodalLoad, ConstantSeries};
//! use fem_core::{ModelDim, NodeId};
//!
//! // 1. Build the model
//! let mut model = Model::new(ModelDim::frame_2d());
//! model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
//! model.add_node(Node::new(NodeId(1), 3.0, 0.0)).unwrap();
//! model.add_element(beam);
//! model.add_constraint(SpConstraint::new(NodeId(0), 0, 0.0, 3)).unwrap();
//! model.add_constraint(SpConstraint::new(NodeId(0), 1, 0.0, 3)).unwrap();
//! model.add_constraint(SpConstraint::new(NodeId(0), 2, 0.0, 3)).unwrap();
//! model.add_load_typed(NodalLoad { node_id: NodeId(1), ... });
//! model.build_state();
//!
//! // 2. Build the sparsity pattern (once per topology)
//! let mut k = build_pattern(&model).unwrap();
//!
//! // 3. Analysis loop
//! let mut f_ext = vec![0.0; model.n_dof()];
//! let mut f_int = vec![0.0; model.n_dof()];
//! let mut r     = vec![0.0; model.n_dof()];
//!
//! for pseudo_time in [0.25, 0.5, 0.75, 1.0] {
//!     assemble_load_vector(&model, pseudo_time, &mut f_ext).unwrap();
//!
//!     // Newton-Raphson inner loop
//!     loop {
//!         assemble_stiffness(&model, &mut k).unwrap();
//!         assemble_internal_force(&model, &mut f_int).unwrap();
//!
//!         // Residual
//!         for i in 0..r.len() { r[i] = f_ext[i] - f_int[i]; }
//!         apply_dirichlet_bcs(&model.constraints, &mut k, &mut r).unwrap();
//!
//!         // Solve Δu (via solvers crate)
//!         solver.factorize(&k).unwrap();
//!         solver.solve(&r, &mut delta_u).unwrap();
//!
//!         for i in 0..model.n_dof() { model.u_global[i] += delta_u[i]; }
//!
//!         if converged { break; }
//!     }
//!     commit_state(&mut model).unwrap();
//! }
//! ```
//!
//! ## Design principles
//!
//! **No global state.** Every `Model` is an independent owned value. Multiple
//! models coexist simultaneously for population-parallel analysis.
//!
//! **Pure function assembly.** Builder functions are `(Model, &mut Output) → Result`.
//! Same inputs always produce the same output.
//!
//! **Stateless kinematics.** Elements never store trial displacements.
//! `u_local` is extracted from `model.u_global` and passed explicitly on
//! every call, eliminating the update-before-stiffness trap.
//!
//! **EERS errors.** All errors use `thiserror` + `miette` with structured
//! context fields and dot-separated diagnostic codes.

// -----------------------------------------------------------------
// Module declarations
// -----------------------------------------------------------------

pub mod error;
pub mod model;
pub mod constraints;
pub mod state;
pub mod topology;
pub mod kinematics;
pub mod loads;
pub mod builders;
pub mod macros;

// -----------------------------------------------------------------
// Flat re-exports — the most commonly imported items
// -----------------------------------------------------------------

// Core model types
pub use model::{Model, Node};

// Constraint type and BC application
pub use constraints::{SpConstraint, apply_dirichlet_bcs};

// State management
pub use state::{commit_state, revert_state};

// Topology
pub use topology::{build_pattern, count_dofs, validate_dof_maps};

// Kinematics
pub use kinematics::extract_local_u;

// Load traits and concrete implementations
pub use loads::pattern::{LoadPattern, NodalLoad, ElementLoad};
pub use loads::series::{TimeSeries, ConstantSeries, LinearSeries, PathSeries};
pub use loads::combo::LoadCombo;
pub use loads::gravity::GravityLoad;
pub use loads::seismic::{GroundMotion, UniformExcitation};

// All five builder functions
pub use builders::{
    assemble_stiffness,
    assemble_internal_force,
    assemble_mass,
    assemble_self_weight,
    assemble_load_vector,
    assemble_partial_residual,
    total_n_params,
    build_rayleigh_damping, 
    rayleigh_coefficients
};

// Error type
pub use error::AssemblyError;