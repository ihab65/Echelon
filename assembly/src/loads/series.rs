//! Time series — the temporal scaling half of a load pattern.
//!
//! A [`TimeSeries`] has one job: given the current `pseudo_time` (a scalar
//! driven by the `ControlScheme` in the `analysis` crate), return a scalar
//! factor. The load pattern multiplies its reference loads by this factor
//! before scattering them into the global force vector.
//!
//! ## Strict separation of concerns
//!
//! ```text
//! LoadPattern  =  spatial distribution  ×  TimeSeries (temporal scaling)
//! ```
//!
//! Keeping these separate means you can reuse the same spatial pattern
//! (e.g., a lateral nodal load) with different time histories (constant for
//! static, linear for load-control pushover, tabulated for earthquake).
//!
//! ## `pseudo_time` convention
//!
//! For a **static** analysis:
//! - Load control: `pseudo_time` ∈ [0, 1], incremented by the load step size.
//! - `LinearSeries` returns `pseudo_time` directly → the load grows linearly
//!   from 0 to its reference value over the analysis.
//!
//! For a **time-history** analysis:
//! - `pseudo_time` is the actual simulation time in seconds.
//! - `PathSeries` interpolates a tabulated ground motion record.

// -----------------------------------------------------------------
// TimeSeries trait
// -----------------------------------------------------------------

/// Temporal scaling for a load pattern.
///
/// Implementors must be `Send + Sync` so that models can be evaluated
/// concurrently in population-parallel analyses.
pub trait TimeSeries: Send + Sync {
    /// Return the load scale factor at the given `pseudo_time`.
    fn factor_at(&self, pseudo_time: f64) -> f64;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn TimeSeries>;
}

// -----------------------------------------------------------------
// ConstantSeries
// -----------------------------------------------------------------

/// Always returns `1.0` — the load is applied at its full reference value
/// for the entire analysis.
///
/// # Use case
/// Gravity loads that are constant across a pushover analysis after the
/// gravity phase is locked via `Model::lock_loads`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantSeries;

impl TimeSeries for ConstantSeries {
    #[inline]
    fn factor_at(&self, _pseudo_time: f64) -> f64 {
        1.0
    }
    fn clone_box(&self) -> Box<dyn TimeSeries> { Box::new(*self) }
}

// -----------------------------------------------------------------
// LinearSeries
// -----------------------------------------------------------------

/// Returns `pseudo_time` — the load grows linearly with the load factor.
///
/// For a load-controlled pushover with 10 equal steps from 0 to 1:
/// - Step 1: `pseudo_time = 0.1` → 10 % of reference load
/// - Step 10: `pseudo_time = 1.0` → 100 % of reference load
///
/// # Use case
/// Load-controlled static pushover analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearSeries;

impl TimeSeries for LinearSeries {
    #[inline]
    fn factor_at(&self, pseudo_time: f64) -> f64 {
        pseudo_time
    }
    fn clone_box(&self) -> Box<dyn TimeSeries> { Box::new(*self) }
}

// -----------------------------------------------------------------
// PathSeries
// -----------------------------------------------------------------

/// Tabulated time–factor pairs with linear interpolation between points.
///
/// The `times` and `values` slices must be the same length and `times`
/// must be strictly ascending. Outside the defined range the series clamps
/// to the nearest endpoint value (no extrapolation).
///
/// # Use case
/// - Ground motion acceleration records for time-history analysis.
/// - Non-proportional multi-phase loading.
/// - Any load whose variation cannot be expressed as a simple formula.
///
/// # Example
///
/// ```rust
/// use assembly::loads::series::PathSeries;
/// use assembly::loads::series::TimeSeries;
///
/// // Ramp from 0 to 1 over 1 s, then constant
/// let s = PathSeries::new(
///     vec![0.0, 0.5, 1.0, 2.0],
///     vec![0.0, 0.5, 1.0, 1.0],
/// ).unwrap();
///
/// assert!((s.factor_at(0.25) - 0.25).abs() < 1e-12);
/// assert!((s.factor_at(1.5)  - 1.0).abs()  < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct PathSeries {
    times:  Vec<f64>,
    values: Vec<f64>,
}

impl PathSeries {
    /// Construct from parallel time and value arrays.
    ///
    /// # Errors
    /// Returns an error string if:
    /// - `times` and `values` have different lengths.
    /// - `times` is not strictly ascending.
    /// - The series has fewer than 2 points.
    pub fn new(times: Vec<f64>, values: Vec<f64>) -> Result<Self, String> {
        if times.len() != values.len() {
            return Err(format!(
                "PathSeries: times ({}) and values ({}) must have the same length",
                times.len(), values.len()
            ));
        }
        if times.len() < 2 {
            return Err("PathSeries requires at least 2 data points".into());
        }
        for w in times.windows(2) {
            if w[0] >= w[1] {
                return Err(format!(
                    "PathSeries: times must be strictly ascending, \
                     found {} >= {} at consecutive entries",
                    w[0], w[1]
                ));
            }
        }
        Ok(Self { times, values })
    }
}

impl TimeSeries for PathSeries {
    fn factor_at(&self, pseudo_time: f64) -> f64 {
        // Clamp below lower bound
        if pseudo_time <= self.times[0] {
            return self.values[0];
        }
        // Clamp above upper bound
        let last = self.times.len() - 1;
        if pseudo_time >= self.times[last] {
            return self.values[last];
        }
        // Binary search for the interval [t_i, t_{i+1}] containing pseudo_time
        let idx = self.times
            .partition_point(|&t| t <= pseudo_time)
            .saturating_sub(1);
        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let v0 = self.values[idx];
        let v1 = self.values[idx + 1];
        // Linear interpolation
        let alpha = (pseudo_time - t0) / (t1 - t0);
        v0 + alpha * (v1 - v0)
    }

    fn clone_box(&self) -> Box<dyn TimeSeries> { Box::new(self.clone()) }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ConstantSeries ----

    #[test]
    fn constant_always_one() {
        let s = ConstantSeries;
        assert_eq!(s.factor_at(0.0),  1.0);
        assert_eq!(s.factor_at(0.5),  1.0);
        assert_eq!(s.factor_at(1.0),  1.0);
        assert_eq!(s.factor_at(100.0), 1.0);
    }

    // ---- LinearSeries ----

    #[test]
    fn linear_returns_pseudo_time() {
        let s = LinearSeries;
        assert_eq!(s.factor_at(0.0),  0.0);
        assert_eq!(s.factor_at(0.1),  0.1);
        assert_eq!(s.factor_at(0.5),  0.5);
        assert_eq!(s.factor_at(1.0),  1.0);
    }

    // ---- PathSeries ----

    #[test]
    fn path_construction_ok() {
        let s = PathSeries::new(vec![0.0, 1.0], vec![0.0, 1.0]);
        assert!(s.is_ok());
    }

    #[test]
    fn path_err_different_lengths() {
        let s = PathSeries::new(vec![0.0, 1.0], vec![0.0]);
        assert!(s.is_err());
    }

    #[test]
    fn path_err_not_ascending() {
        let s = PathSeries::new(vec![0.0, 1.0, 0.5], vec![0.0, 1.0, 0.5]);
        assert!(s.is_err());
    }

    #[test]
    fn path_err_too_few_points() {
        let s = PathSeries::new(vec![0.0], vec![1.0]);
        assert!(s.is_err());
    }

    #[test]
    fn path_midpoint_interpolation() {
        let s = PathSeries::new(
            vec![0.0, 1.0],
            vec![0.0, 2.0],
        ).unwrap();
        assert!((s.factor_at(0.5) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn path_clamp_below() {
        let s = PathSeries::new(vec![1.0, 2.0], vec![5.0, 10.0]).unwrap();
        assert_eq!(s.factor_at(0.0), 5.0); // below start → first value
    }

    #[test]
    fn path_clamp_above() {
        let s = PathSeries::new(vec![0.0, 1.0], vec![0.0, 3.0]).unwrap();
        assert_eq!(s.factor_at(5.0), 3.0); // above end → last value
    }

    #[test]
    fn path_multi_segment_ramp_hold() {
        let s = PathSeries::new(
            vec![0.0, 0.5, 1.0, 2.0],
            vec![0.0, 0.5, 1.0, 1.0],
        ).unwrap();
        assert!((s.factor_at(0.25) - 0.25).abs() < 1e-12);
        assert!((s.factor_at(0.75) - 0.75).abs() < 1e-12);
        assert!((s.factor_at(1.5)  - 1.0).abs()  < 1e-12);
    }

    #[test]
    fn path_exact_endpoint_values() {
        let s = PathSeries::new(
            vec![0.0, 1.0, 2.0],
            vec![10.0, 20.0, 30.0],
        ).unwrap();
        assert!((s.factor_at(0.0) - 10.0).abs() < 1e-14);
        assert!((s.factor_at(1.0) - 20.0).abs() < 1e-14);
        assert!((s.factor_at(2.0) - 30.0).abs() < 1e-14);
    }
}