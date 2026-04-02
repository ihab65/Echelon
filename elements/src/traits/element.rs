//! [`Element`] — the fundamental Engine A interface.
//!
//! Every element in Echelon implements this trait.  It is the minimum surface
//! required to drive the Newton-Raphson loop: provide a stiffness matrix and
//! an internal force vector, and manage constitutive state.
//!
//! ## Method responsibilities
//!
//! | Method       | Called by          | Returns                          |
//! |--------------|-------------------|----------------------------------|
//! | `n_dof`      | Assembly           | Local DOF count                  |
//! | `ke_flat`    | Assembly           | Flattened n×n stiffness (row-major) |
//! | `f_int`      | Newton residual    | Internal force vector            |
//! | `commit`     | After convergence  | —                                |
//! | `revert`     | After divergence   | —                                |
//!
//! ## Convention: local vs global
//!
//! `ke_flat` and `f_int` are returned in **global** coordinates.
//! The coordinate transformation (rotation from local to global frame)
//! is the responsibility of the element itself, not the assembler.
//! This keeps the assembly loop a pure scatter-add with no geometric
//! knowledge.

use crate::error::Result;

/// Minimal interface required by the Newton-Raphson assembly loop (Engine A).
///
/// # Thread safety
///
/// Elements must be `Send + Sync` to support population-parallel analysis,
/// where independent model instances are evaluated concurrently on separate
/// threads.
pub trait Element: Send + Sync {
    /// Number of local degrees of freedom.
    ///
    /// - 2D truss: `4` (2 nodes × 2 DOF/node)
    /// - 2D beam:  `6` (2 nodes × 3 DOF/node)
    fn n_dof(&self) -> usize;

    /// Flattened `n×n` tangent stiffness matrix in **global** coordinates,
    /// row-major order, for the current displacement state `u`.
    ///
    /// # Arguments
    /// * `u` — slice of length `n_dof()`, local displacement in global coords
    ///
    /// # Returns
    /// `Vec<f64>` of length `n_dof()²`, row-major.  Pass directly to
    /// `SymCsrMatrix::scatter_add`.
    fn ke_flat(&self, u: &[f64]) -> Vec<f64>;

    /// Returns the element's mass matrix (lumped) as a flat array.
    /// The length of the returned vector should be `n_dof()`.
    fn mass_flat(&self) -> Vec<f64>;

    /// Internal force vector in **global** coordinates, for displacement `u`.
    ///
    /// # Arguments
    /// * `u` — slice of length `n_dof()`, local displacement in global coords
    ///
    /// # Returns
    /// `Vec<f64>` of length `n_dof()`.
    fn f_int(&self, u: &[f64]) -> Vec<f64>;

    /// Commit the current state as converged.
    ///
    /// Forwards the call to all owned materials.  After this call,
    /// `revert` will restore to this committed state.
    ///
    /// # Arguments
    /// * `u` — the converged displacement at which state is committed
    fn commit(&mut self, u: &[f64]) -> Result<()>;

    /// Revert all internal state to the last committed state.
    ///
    /// Forwards the call to all owned materials.
    fn revert(&mut self);

    /// Clone into a boxed trait object.
    ///
    /// Required for population-parallel analysis: each worker thread
    /// gets its own clone of the element with independent material state.
    fn clone_box(&self) -> Box<dyn Element>;

    /// Human-readable element type name for diagnostics.
    fn type_name(&self) -> &'static str;
}