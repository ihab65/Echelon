pub mod dof;
pub mod sparsity;

pub use dof::{count_dofs, validate_dof_maps};
pub use sparsity::build_pattern;