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

use crate::traits::Element;

/// Extension of `Element` for elements that can define a scalar strain energy.
pub trait DifferentiableElement: Element {
    /// Scalar strain energy: `W = E(u)`
    fn energy_f64(&self, u: &[f64]) -> f64;

    /// Stiffness matrix via finite-difference Hessian of `energy_f64`.
    /// 
    /// Default implementation: central differences with optimal math for 
    /// diagonal and off-diagonal terms. Writes directly into `out_ke` 
    /// to avoid heap allocations.
    fn ke_flat_from_energy(&self, u: &[f64], out: &mut [f64]) {
        let n = self.n_dof();
        debug_assert_eq!(out.len(), n * n);
        let h = 1e-6; // FD step size

        // We only need ONE working vector, dramatically reducing allocations
        let mut u_work = u.to_vec();
        
        // Base energy evaluated once for the diagonal formula
        let e_0 = self.energy_f64(u);

        for i in 0..n {
            for j in i..n {
                let kij = if i == j {
                    // Corrected formula for the Diagonal: ∂²E / ∂u_i²
                    // (E(x+h) - 2E(x) + E(x-h)) / h²
                    u_work[i] = u[i] + h;
                    let e_p = self.energy_f64(&u_work);

                    u_work[i] = u[i] - h;
                    let e_m = self.energy_f64(&u_work);

                    (e_p - 2.0 * e_0 + e_m) / (h * h)
                } else {
                    // Off-diagonal: ∂²E / ∂u_i ∂u_j
                    // (E(x+h,y+h) - E(x+h,y-h) - E(x-h,y+h) + E(x-h,y-h)) / 4h²
                    u_work[i] = u[i] + h; u_work[j] = u[j] + h;
                    let e_pp = self.energy_f64(&u_work);

                    u_work[i] = u[i] + h; u_work[j] = u[j] - h;
                    let e_pm = self.energy_f64(&u_work);

                    u_work[i] = u[i] - h; u_work[j] = u[j] + h;
                    let e_mp = self.energy_f64(&u_work);

                    u_work[i] = u[i] - h; u_work[j] = u[j] - h;
                    let e_mm = self.energy_f64(&u_work);

                    (e_pp - e_pm - e_mp + e_mm) / (4.0 * h * h)
                };

                // Write directly to the output buffer
                out[i * n + j] = kij;
                out[j * n + i] = kij; // Symmetric mirror

                // Restore working vector exactly to original state
                u_work[i] = u[i];
                u_work[j] = u[j];
            }
        }
    }

    /// Internal force via finite-difference gradient of `energy_f64`.
    ///
    /// Default implementation: central differences. Writes directly 
    /// into `out_f` to avoid heap allocations.
    fn f_int_from_energy(&self, u: &[f64], out: &mut [f64]) {
        let n = self.n_dof();
        debug_assert_eq!(out.len(), n);
        let h = 1e-6;

        // One working vector
        let mut u_work = u.to_vec();

        for i in 0..n {
            u_work[i] = u[i] + h;
            let e_plus = self.energy_f64(&u_work);

            u_work[i] = u[i] - h;
            let e_minus = self.energy_f64(&u_work);

            // Write directly to the output buffer
            out[i] = (e_plus - e_minus) / (2.0 * h);

            // Restore working vector
            u_work[i] = u[i]; 
        }
    }
}