//! Convenience macros for composing load patterns.

/// Build a [`LoadCombo`](crate::loads::combo::LoadCombo) from a list of
/// `factor * load_pattern` terms.
///
/// Each term is wrapped in a `LoadCombo` with the given scale factor and
/// added as a child of the top-level combo (scale = 1.0).
///
/// # Example
///
/// ```rust,ignore
/// use assembly::{load_combo, loads::gravity::GravityLoad};
///
/// let uls = load_combo!(1.35 * dead_load, 1.50 * live_load);
/// model.add_load_typed(uls);
/// ```
#[macro_export]
macro_rules! load_combo {
    ($($factor:expr => $load:expr),+ $(,)?) => {{
        let mut _combo = $crate::loads::combo::LoadCombo::new(1.0);
        $(
            _combo.add(Box::new($crate::loads::combo::LoadCombo::scaled(
                $factor,
                $load,
            )));
        )+
        _combo
    }};
}

/// Build a `Uniform` [`ElementLoadParams`](elements::ElementLoadParams)
/// with only a transverse (Y) component.
///
/// ```rust,ignore
/// let load = uniform_load!(-20e3); // 20 kN/m downward
/// ```
#[macro_export]
macro_rules! uniform_load {
    ($wy:expr) => {
        elements::ElementLoadParams::Uniform { wx: 0.0, wy: $wy }
    };
    (x: $wx:expr, y: $wy:expr) => {
        elements::ElementLoadParams::Uniform { wx: $wx, wy: $wy }
    };
}

/// Build a `Point` [`ElementLoadParams`](elements::ElementLoadParams).
///
/// ```rust,ignore
/// let load = point_load!(-50e3, at: 0.5); // 50 kN downward at mid-span
/// ```
#[macro_export]
macro_rules! point_load {
    ($py:expr, at: $xi:expr) => {
        elements::ElementLoadParams::Point { px: 0.0, py: $py, xi: $xi }
    };
}