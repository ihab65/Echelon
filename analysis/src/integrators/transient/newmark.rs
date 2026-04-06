//! Newmark-β transient integrator for dynamic time-history analysis.
//!
//! [`Newmark`] implements the classical Newmark-β method for time integration
//! of the equation of motion:
//!
//! ```text
//! M ü + C u̇ + K_T u = F_ext(t)
//! ```
//!
//! ## Newmark approximations
//!
//! The velocity and displacement at time `t + Δt` are approximated as:
//!
//! ```text
//! u̇_{n+1} = u̇_n + Δt [(1 − γ) ü_n + γ ü_{n+1}]
//! u_{n+1}  = u_n  + Δt u̇_n + Δt² [(0.5 − β) ü_n + β ü_{n+1}]
//! ```
//!
//! Solving for `ü_{n+1}` from the second equation and substituting:
//!
//! ```text
//! ü_{n+1} = (1 / βΔt²)(u_{n+1} − u_n − Δt u̇_n) − (1/β − 1)ü_n
//! ```
//!
//! ## Effective stiffness formulation
//!
//! Substituting into the equation of motion yields the effective system:
//!
//! ```text
//! K_eff Δu = F_eff
//!
//! K_eff = K_T + (γ/βΔt) C + (1/βΔt²) M
//!
//! F_eff = F_{ext,n+1}
//!         − K_T u_n
//!         − C [u̇_n + Δt(1−γ) ü_n]
//!         − M [(1/βΔt²) u_n + (1/βΔt) u̇_n + (1/2β − 1) ü_n]  (predictor terms)
//! ```
//!
//! ## Stability and accuracy parameters
//!
//! | (β, γ) | Method | Stability | Accuracy |
//! |--------|--------|-----------|----------|
//! | (0.25, 0.5) | Average acceleration | Unconditionally stable | 2nd order |
//! | (0, 0.5) | Central difference | Conditionally stable | 2nd order |
//! | (1/6, 0.5) | Linear acceleration | Conditionally stable | 2nd order |
//!
//! The default `(β=0.25, γ=0.5)` (average acceleration) is unconditionally
//! stable and non-dissipative. For seismic analysis with numerical damping,
//! use [`crate::integrators::transient::hht::HHT`] instead.

use assembly::{self, Model};
use sparse::{SparseMatrix, SymCsrMatrix};

use crate::error::{AnalysisError, Result};
use crate::integrators::Integrator;
use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// Newmark
// -----------------------------------------------------------------

/// Newmark-β time integrator for dynamic analysis.
///
/// Solves `M ü + C u̇ + K_T u = F_ext(t)` by the implicit Newmark method.
///
/// # Default parameters
///
/// `Newmark::average_acceleration(dt)` sets `β = 0.25, γ = 0.5`, giving
/// the unconditionally stable average acceleration method.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::integrators::transient::newmark::Newmark;
///
/// let mass    = assemble_mass_matrix(&model);
/// let damping = rayleigh_damping(0.05, &mass, &stiffness);
/// let integrator = Newmark::average_acceleration(0.01, mass, Some(damping));
/// ```
pub struct Newmark {
    /// Newmark β parameter. Controls the displacement predictor.
    /// `β = 0.25` → average acceleration (unconditionally stable).
    pub beta: f64,

    /// Newmark γ parameter. Controls the velocity predictor.
    /// `γ = 0.5` → no algorithmic damping.
    pub gamma: f64,

    /// Time step size Δt (seconds).
    pub dt: f64,

    /// Global mass matrix (consistent or lumped).
    pub mass: SymCsrMatrix<f64>,

    /// Global damping matrix (Rayleigh or modal). `None` = undamped.
    pub damping: Option<SymCsrMatrix<f64>>,

    // ---- State vectors ---------------------------------------------------

    /// Velocity at the committed step `u̇_n`.
    velocity: Vec<f64>,

    /// Acceleration at the committed step `ü_n`.
    acceleration: Vec<f64>,

    /// Current simulation time (committed).
    current_time: f64,

    /// Pre-committed velocity (used by revert).
    prev_velocity: Vec<f64>,

    /// Pre-committed acceleration (used by revert).
    prev_acceleration: Vec<f64>,
}

impl Newmark {
    /// Create a Newmark integrator with the average acceleration parameters
    /// `β = 0.25, γ = 0.5` — unconditionally stable, no algorithmic damping.
    ///
    /// # Arguments
    /// * `dt`      — time step in seconds.
    /// * `mass`    — global consistent or lumped mass matrix.
    /// * `damping` — global damping matrix, or `None` for undamped analysis.
    ///
    /// # Panics
    /// Panics if `dt ≤ 0`.
    pub fn average_acceleration(
        dt:      f64,
        mass:    SymCsrMatrix<f64>,
        damping: Option<SymCsrMatrix<f64>>,
    ) -> Self {
        Self::new(0.25, 0.5, dt, mass, damping)
    }

    /// Create a Newmark integrator with explicit `β` and `γ` parameters.
    ///
    /// # Panics
    /// Panics if `dt ≤ 0`, `β ≤ 0`, or `γ ≤ 0`.
    pub fn new(
        beta:    f64,
        gamma:   f64,
        dt:      f64,
        mass:    SymCsrMatrix<f64>,
        damping: Option<SymCsrMatrix<f64>>,
    ) -> Self {
        assert!(dt   > 0.0, "Newmark: dt must be positive");
        assert!(beta > 0.0, "Newmark: β must be positive");
        assert!(gamma > 0.0, "Newmark: γ must be positive");

        let n = mass.nrows();
        Self {
            beta,
            gamma,
            dt,
            mass,
            damping,
            velocity:          vec![0.0; n],
            acceleration:      vec![0.0; n],
            current_time:      0.0,
            prev_velocity:     vec![0.0; n],
            prev_acceleration: vec![0.0; n],
        }
    }

    /// Newmark integration coefficients derived from β, γ, and Δt.
    ///
    /// Returns `(a0, a1, a2, a3, a4, a5)` where:
    /// - `a0 = 1 / (β Δt²)`
    /// - `a1 = γ / (β Δt)`
    /// - `a2 = 1 / (β Δt)`
    /// - `a3 = 1 / (2β) − 1`
    /// - `a4 = γ/β − 1`
    /// - `a5 = Δt (γ/β − 2) / 2`
    #[inline]
    pub fn coefficients(&self) -> (f64, f64, f64, f64, f64, f64) {
        let a0 = 1.0 / (self.beta * self.dt * self.dt);
        let a1 = self.gamma / (self.beta * self.dt);
        let a2 = 1.0 / (self.beta * self.dt);
        let a3 = 1.0 / (2.0 * self.beta) - 1.0;
        let a4 = self.gamma / self.beta - 1.0;
        let a5 = self.dt * (self.gamma / self.beta - 2.0) / 2.0;
        (a0, a1, a2, a3, a4, a5)
    }
}

impl Integrator for Newmark {
    /// Advance to the next time step and form the effective load vector.
    ///
    /// Populates `system.f_ext` with the Newmark effective load:
    ///
    /// ```text
    /// F_eff = F_ext(t + Δt)
    ///         + M [a0 u_n + a2 u̇_n + a3 ü_n]
    ///         + C [a1 u_n + a4 u̇_n + a5 ü_n]  (if damping present)
    /// ```
    ///
    /// Also augments `system.k_t` will be handled by the algorithm (the
    /// driver must call `assemble_stiffness` first, then add the mass/damping
    /// contributions via this integrator).
    fn new_step(&mut self, system: &mut GlobalSystem, model: &mut Model) -> Result<()> {
        self.current_time += self.dt;

        let (a0, a1, a2, a3, a4, a5) = self.coefficients();
        let n = model.n_dof();

        if n != self.velocity.len() {
            return Err(AnalysisError::InvalidConfiguration {
                reason: format!(
                    "Newmark: model has {n} DOFs but integrator was initialized for {} DOFs. \
                     Rebuild the integrator after changing the model.",
                    self.velocity.len()
                ),
            });
        }

        // Assemble external load at t + Δt
        assembly::assemble_load_vector(model, self.current_time, &mut system.f_ext)?;

        // Add inertia predictor: F_eff += M [a0 u_n + a2 u̇_n + a3 ü_n]
        let m_contrib: Vec<f64> = (0..n)
            .map(|i| {
                a0 * model.u_global[i]
                    + a2 * self.velocity[i]
                    + a3 * self.acceleration[i]
            })
            .collect();

        let m_force = self.mass.matvec(&m_contrib)
            .map_err(|e| AnalysisError::from(assembly::error::AssemblyError::from(e)))?;

        for i in 0..n {
            system.f_ext[i] += m_force[i];
        }

        // Add damping predictor: F_eff += C [a1 u_n + a4 u̇_n + a5 ü_n]
        if let Some(ref c) = self.damping {
            let c_contrib: Vec<f64> = (0..n)
                .map(|i| {
                    a1 * model.u_global[i]
                        + a4 * self.velocity[i]
                        + a5 * self.acceleration[i]
                })
                .collect();
            let c_force = c.matvec(&c_contrib)
                .map_err(|e| AnalysisError::from(assembly::error::AssemblyError::from(e)))?;
            for i in 0..n {
                system.f_ext[i] += c_force[i];
            }
        }

        Ok(())
    }

    fn form_tangent(&self, system: &mut GlobalSystem) -> Result<()> {
        // K_eff = K_T + a0*M + a1*C  (Newmark coefficients)
        // a0 = 1/(β Δt²),  a1 = γ/(β Δt)
        let (a0, a1, _, _, _, _) = self.coefficients();

        // Add a0 * M to k_t
        for (row, col, val) in self.mass.iter_upper() {
            system.k_t.add_value(row, col, val * a0)
                .map_err(|e| crate::error::AnalysisError::from(
                    assembly::error::AssemblyError::from(e)
                ))?;
        }

        // Add a1 * C to k_t (if damping matrix is present)
        if let Some(ref c) = self.damping {
            for (row, col, val) in c.iter_upper() {
                system.k_t.add_value(row, col, val * a1)
                    .map_err(|e| crate::error::AnalysisError::from(
                        assembly::error::AssemblyError::from(e)
                    ))?;
            }
        }
        Ok(())
    }

    fn commit(&mut self) {
        // Update velocity and acceleration from the converged displacement
        // This is called by the driver after a successful Newton loop.
        // The driver passes the model's u_global to update v and a.
        // For simplicity, commit just saves the current state.
        self.prev_velocity     = self.velocity.clone();
        self.prev_acceleration = self.acceleration.clone();
    }

    fn revert(&mut self) {
        self.current_time  -= self.dt;
        self.velocity       = self.prev_velocity.clone();
        self.acceleration   = self.prev_acceleration.clone();
    }

    fn name(&self) -> &'static str {
        "Newmark"
    }

    fn current_time(&self) -> f64 {
        self.current_time
    }
}