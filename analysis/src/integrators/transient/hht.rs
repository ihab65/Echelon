//! Hilber-Hughes-Taylor-α (HHT-α) transient integrator.
//!
//! [`HHT`] is a generalization of the Newmark method that introduces a
//! single parameter `α ∈ [−1/3, 0]` to add controlled **algorithmic damping**
//! at high frequencies while preserving second-order accuracy at low
//! frequencies.
//!
//! ## Why use HHT over plain Newmark?
//!
//! Standard Newmark with `γ = 0.5` is non-dissipative: all frequency content
//! (including spurious high-frequency noise from spatial discretisation) is
//! preserved indefinitely. In seismic analysis this is often undesirable —
//! high-frequency parasitic oscillations can mask the physical low-frequency
//! response and require extremely small time steps to resolve.
//!
//! HHT-α filters these high-frequency components while maintaining the
//! physical low-frequency response. The dissipation is entirely numerical
//! (algorithmic), not physical — it does not represent real material damping.
//!
//! ## Modified equilibrium
//!
//! HHT modifies the equilibrium equation at time `t_{n+1}` by evaluating
//! the internal forces at an intermediate configuration:
//!
//! ```text
//! M ü_{n+1} + (1+α) [C u̇_{n+1} + K_T u_{n+1}]
//!           − α     [C u̇_n     + K_T u_n    ]  = (1+α) F_ext(t_{n+1}) − α F_ext(t_n)
//! ```
//!
//! ## Parameter relationships
//!
//! For unconditional stability and second-order accuracy:
//!
//! ```text
//! γ = 0.5 − α
//! β = (1 − α)² / 4
//! α ∈ [−1/3, 0]
//! ```
//!
//! Setting `α = 0` recovers the standard Newmark average acceleration method.
//! Setting `α = −0.1` adds modest damping (typical for seismic analysis).

use assembly::Model;
use sparse::{SparseMatrix, SymCsrMatrix};
use sparse::MatvecWorkspace;

use crate::error::{AnalysisError, Result};
use crate::integrators::Integrator;
use crate::system::GlobalSystem;

// -----------------------------------------------------------------
// HHT
// -----------------------------------------------------------------

/// Hilber-Hughes-Taylor-α time integrator with algorithmic numerical damping.
///
/// # Example
///
/// ```rust,ignore
/// use analysis::integrators::transient::hht::HHT;
///
/// // α = -0.1 gives modest numerical damping, typical for seismic analysis
/// let integrator = HHT::new(-0.1, 0.02, mass, Some(damping));
/// ```
pub struct HHT {
    /// HHT-α dissipation parameter. Must be in `[−1/3, 0]`.
    ///
    /// `α = 0`    → standard Newmark (no dissipation).
    /// `α = −1/3` → maximum dissipation (first-order accurate only).
    pub alpha: f64,

    /// Newmark β, derived from α: `β = (1 − α)² / 4`.
    pub beta: f64,

    /// Newmark γ, derived from α: `γ = 0.5 − α`.
    pub gamma: f64,

    /// Time step size Δt (seconds).
    pub dt: f64,

    /// Global mass matrix.
    pub mass: SymCsrMatrix<f64>,

    /// Global damping matrix, or `None` for undamped analysis.
    pub damping: Option<SymCsrMatrix<f64>>,

    // ---- State -----------------------------------------------------------
    velocity:          Vec<f64>,
    acceleration:      Vec<f64>,
    current_time:      f64,
    prev_velocity:     Vec<f64>,
    prev_acceleration: Vec<f64>,
    /// F_ext evaluated at the *previous* committed step (for HHT blending).
    f_ext_prev:        Vec<f64>,

    // ---- Persistent Workspaces ----
    f_new:       Vec<f64>,
    pred_buffer: Vec<f64>,
    matvec_ws:   MatvecWorkspace<f64>,
}

impl HHT {
    /// Create an HHT integrator with the given `α` parameter.
    ///
    /// β and γ are set automatically from `α` to guarantee unconditional
    /// stability and second-order accuracy.
    ///
    /// # Arguments
    /// * `alpha`   — dissipation parameter in `[−1/3, 0]`.
    /// * `dt`      — time step in seconds.
    /// * `mass`    — global mass matrix.
    /// * `damping` — global damping matrix, or `None`.
    ///
    /// # Panics
    /// Panics if `alpha ∉ [−1/3, 0]` or `dt ≤ 0`.
    pub fn new(
        alpha:   f64,
        dt:      f64,
        mass:    SymCsrMatrix<f64>,
        damping: Option<SymCsrMatrix<f64>>,
    ) -> Self {
        assert!(
            (-1.0 / 3.0..=0.0).contains(&alpha),
            "HHT: alpha must be in [-1/3, 0], got {alpha}"
        );
        assert!(dt > 0.0, "HHT: dt must be positive");

        let beta  = (1.0 - alpha).powi(2) / 4.0;
        let gamma = 0.5 - alpha;
        let n     = mass.ncols();

        Self {
            alpha,
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
            f_ext_prev:        vec![0.0; n],
            f_new:             vec![0.0; n],
            pred_buffer:       vec![0.0; n],
            matvec_ws:         MatvecWorkspace::new(n),
        }
    }
}

impl Integrator for HHT {
    /// Advance to `t + Δt` and form the HHT effective load vector.
    ///
    /// The effective load blends `F_ext(t + Δt)` and `F_ext(t)`:
    ///
    /// ```text
    /// F_eff = (1+α) F_ext(t+Δt) − α F_ext(t)
    ///         + M [a0 u_n + a2 u̇_n + a3 ü_n]
    ///         + (1+α) C [a1 u_n + a4 u̇_n + a5 ü_n]  (damping term)
    /// ```
    ///
    /// # Errors
    /// - [`AnalysisError::AssemblyError`] if external load assembly or matrix-vector multiplication fails.
    fn new_step(&mut self, system: &mut GlobalSystem, model: &mut Model) -> Result<()> {
        self.current_time += self.dt;
        let n = model.n_dof();

        let a0 = 1.0 / (self.beta * self.dt * self.dt);
        let a2 = 1.0 / (self.beta * self.dt);
        let a3 = 1.0 / (2.0 * self.beta) - 1.0;
        let a1 = self.gamma / (self.beta * self.dt);
        let a4 = self.gamma / self.beta - 1.0;
        let a5 = self.dt * (self.gamma / self.beta - 2.0) / 2.0;

        // 1. Evaluate external load straight into the persistent buffer
        self.f_new.fill(0.0);
        assembly::assemble_load_vector(model, self.current_time, &mut self.f_new)?;

        // HHT blend
        let alpha = self.alpha;
        for i in 0..n {
            system.f_ext[i] = (1.0 + alpha) * self.f_new[i] - alpha * self.f_ext_prev[i];
        }

        // 2. Inertia predictor
        for i in 0..n {
            self.pred_buffer[i] = a0 * model.u_global[i] 
                                + a2 * self.velocity[i] 
                                + a3 * self.acceleration[i];
        }
        
        self.mass.matvec_into(&self.pred_buffer, &mut self.matvec_ws)
            .map_err(|e| AnalysisError::from(assembly::error::AssemblyError::from(e)))?;
            
        for i in 0..n { 
            system.f_ext[i] += self.matvec_ws.as_slice()[i]; 
        }

        // 3. Damping predictor
        if let Some(ref c) = self.damping {
            for i in 0..n {
                self.pred_buffer[i] = a1 * model.u_global[i] 
                                    + a4 * self.velocity[i] 
                                    + a5 * self.acceleration[i];
            }
            
            c.matvec_into(&self.pred_buffer, &mut self.matvec_ws)
                .map_err(|e| AnalysisError::from(assembly::error::AssemblyError::from(e)))?;
                
            for i in 0..n { 
                system.f_ext[i] += (1.0 + alpha) * self.matvec_ws.as_slice()[i]; 
            }
        }

        // Save f_new to f_ext_prev using copy_from_slice!
        self.f_ext_prev.copy_from_slice(&self.f_new);
        Ok(())
    }

    /// # Errors
    /// - [`AnalysisError::AssemblyError`] if adding mass or damping to stiffness fails.
    fn form_tangent(&self, system: &mut GlobalSystem) -> Result<()> {
        // HHT effective stiffness: K_eff = a0*M + (1+α)*a1*C + (1+α)*K_T
        // K_T was just assembled by assemble_stiffness; scale it by (1+α).
        system.k_t.scale(1.0 + self.alpha);

        let a0 = 1.0 / (self.beta * self.dt * self.dt);
        let a1 = self.gamma / (self.beta * self.dt);

        // Add a0 * M
        for (row, col, val) in self.mass.iter_upper() {
            system.k_t.add_value(row, col, val * a0)
                .map_err(|e| crate::error::AnalysisError::from(
                    assembly::error::AssemblyError::from(e)
                ))?;
        }

        // Add (1+α) * a1 * C
        if let Some(ref c) = self.damping {
            let scale = (1.0 + self.alpha) * a1;
            for (row, col, val) in c.iter_upper() {
                system.k_t.add_value(row, col, val * scale)
                    .map_err(|e| crate::error::AnalysisError::from(
                        assembly::error::AssemblyError::from(e)
                    ))?;
            }
        }

        Ok(())
    }

    fn commit(&mut self) {
        self.prev_velocity.copy_from_slice(&self.velocity);
        self.prev_acceleration.copy_from_slice(&self.acceleration);
    }

    fn revert(&mut self) {
        self.current_time  -= self.dt;
        self.velocity.copy_from_slice(&self.prev_velocity);
        self.acceleration.copy_from_slice(&self.prev_acceleration);
    }

    fn name(&self) -> &'static str {
        "HHT"
    }

    fn current_time(&self) -> f64 {
        self.current_time
    }
}