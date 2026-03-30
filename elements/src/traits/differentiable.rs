//! [`DifferentiableElement`] — Engine B extension for autodiff elements.
//!
//! Elements that implement this trait expose a scalar **strain energy**
//! function `energy<T>` that is generic over any dual-number type `T`.
//! From this single function, all of the following are derived automatically:
//!
//! - **Residual** (internal force) — `gradient(energy, u)`
//! - **Stiffness matrix** — `hessian(energy, u)`
//! - **Geometric sensitivity** — `∂energy/∂L`, `∂energy/∂θ`  via AD through
//!   the coordinate transformation
//!
//! ## Default implementations
//!
//! `DifferentiableElement` provides default implementations of `ke_flat` and
//! `f_int` that call `hessian` and `gradient` on `energy`.  Performance-
//! critical elements may override these with closed-form expressions.
//!
//! ## When to implement this trait
//!
//! Implement `DifferentiableElement` for any element whose strain energy can
//! be written as a smooth scalar function of displacements:
//! - Linear elastic elements (truss, beam, shell)
//! - Nonlinear elastic elements (large-displacement beams, hyperelastic solids)
//!
//! Do **not** implement it for elements containing history-dependent
//! (path-dependent) materials.  Those elements implement only `Element`,
//! and sensitivity is handled by `AdjointSensitive` at the material level.

/// Extension of `Element` for elements that can define a scalar strain energy.
///
/// # Type parameter `T`
///
/// `T` is the numeric scalar type.  For Engine A (forward analysis) `T = f64`.
/// For Engine B (sensitivity) `T` is a dual number type from `num-dual`:
/// - `Dual64` — first-order forward-mode AD (gradients)
/// - `HyperDual64` — second-order AD (Hessians, for flexibility sensitivity)
///
/// The `DualNum` bound from `num-dual` provides the arithmetic operations
/// needed.  When the `autodiff` feature is disabled, this trait is still
/// usable with `T = f64`.
///
/// # Default `ke_flat` and `f_int`
///
/// The default implementations use finite differences over `energy` to
/// produce stiffness and residual.  This is correct but slower than a
/// closed-form implementation.  Concrete elements override these where
/// performance matters.
pub trait DifferentiableElement: crate::traits::Element {
    /// Scalar strain energy as a function of nodal displacements.
    ///
    /// This function must be:
    /// 1. **Generic over `T`** — callable with `T = f64` and `T = Dual64`.
    /// 2. **Pure** — no side effects, no mutation.
    /// 3. **Smooth** — continuously differentiable in `u`.
    ///
    /// # Arguments
    /// * `u` — slice of length `n_dof()` as generic type `T`
    ///
    /// # Example (2D truss)
    /// ```rust,ignore
    /// fn energy<T: DualNum<f64> + Copy>(&self, u: &[T]) -> T {
    ///     let u1 = u[0] * T::from(self.cos) + u[1] * T::from(self.sin);
    ///     let u2 = u[2] * T::from(self.cos) + u[3] * T::from(self.sin);
    ///     let delta = u2 - u1;
    ///     T::from(0.5 * self.ea_over_l) * delta * delta
    /// }
    /// ```
    fn energy_f64(&self, u: &[f64]) -> f64;

    /// Stiffness matrix via finite-difference approximation of `energy_f64`.
    ///
    /// Default implementation: second-order central differences.
    /// Override with a closed-form expression for performance-critical elements.
    fn ke_flat_from_energy(&self, u: &[f64]) -> Vec<f64> {
        let n = self.n_dof();
        let h = 1e-7;
        let mut ke = vec![0.0_f64; n * n];

        // Hessian via central finite differences:
        //   ∂²W/∂uᵢ∂uⱼ ≈ [W(u+hᵢ+hⱼ) - W(u+hᵢ-hⱼ) - W(u-hᵢ+hⱼ) + W(u-hᵢ-hⱼ)] / (4h²)
        let mut u_pp = u.to_vec();
        let mut u_pm = u.to_vec();
        let mut u_mp = u.to_vec();
        let mut u_mm = u.to_vec();

        for i in 0..n {
            for j in i..n {
                u_pp[i] = u[i] + h; u_pp[j] = u[j] + h;
                u_pm[i] = u[i] + h; u_pm[j] = u[j] - h;
                u_mp[i] = u[i] - h; u_mp[j] = u[j] + h;
                u_mm[i] = u[i] - h; u_mm[j] = u[j] - h;

                let kij = (self.energy_f64(&u_pp)
                    - self.energy_f64(&u_pm)
                    - self.energy_f64(&u_mp)
                    + self.energy_f64(&u_mm))
                    / (4.0 * h * h);

                ke[i * n + j] = kij;
                ke[j * n + i] = kij; // symmetric

                // Restore
                u_pp[i] = u[i]; u_pp[j] = u[j];
                u_pm[i] = u[i]; u_pm[j] = u[j];
                u_mp[i] = u[i]; u_mp[j] = u[j];
                u_mm[i] = u[i]; u_mm[j] = u[j];
            }
        }
        ke
    }

    /// Internal force via finite-difference gradient of `energy_f64`.
    ///
    /// Default implementation: central differences.
    /// Override with a closed-form expression for performance.
    fn f_int_from_energy(&self, u: &[f64]) -> Vec<f64> {
        let n = self.n_dof();
        let h = 1e-7;
        let mut f = vec![0.0_f64; n];
        let mut u_p = u.to_vec();
        let mut u_m = u.to_vec();

        for i in 0..n {
            u_p[i] = u[i] + h;
            u_m[i] = u[i] - h;
            f[i] = (self.energy_f64(&u_p) - self.energy_f64(&u_m)) / (2.0 * h);
            u_p[i] = u[i];
            u_m[i] = u[i];
        }
        f
    }
}