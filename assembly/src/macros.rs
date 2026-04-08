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

/// Build a [`Model`](crate::model::Model) from a declarative description.
///
/// # Syntax
///
/// ```rust,ignore
/// use assembly::echelon_model;
///
/// let model = echelon_model! {
///     dim: Frame2d,               // Frame2d (default) or Truss2d
///     nodes: [
///         { id: 0, x: 0.0, y: 0.0, z: 0.0 },
///         { id: 1, x: 0.0, y: 3.0, z: 0.0 },
///     ],
///     materials: [
///         { id: steel, E: 200e9 },
///         { id: concrete, E: 30e9, rho: 2400.0 },
///     ],
///     elements: [
///         { type: Beam2d, nodes: [0, 1], mat: steel, A: 0.01, Iz: 1e-4 },
///     ],
/// };
/// ```
///
/// ## Notes
///
/// - `dim` defaults to `Frame2d` if omitted.
/// - Material `id`s are Rust identifiers (not strings).
/// - `rho` in materials is optional; omit for no density.
/// - Element `type` must be `Beam2d` or `Truss2d`.
/// - `Iz` is only required for `Beam2d` elements.
/// - Call `model.build_state()` separately if you plan to add constraints
///   or loads before the first analysis.
#[macro_export]
macro_rules! echelon_model {
    // Entry point — parse all sections
    (
        $(dim: $dim:ident,)?
        nodes: [ $({ id: $nid:expr, x: $nx:expr, y: $ny:expr, z: $nz:expr }),* $(,)? ],
        materials: [ $({ id: $mid:ident $(, E: $me:expr)? $(, rho: $mrho:expr)? }),* $(,)? ],
        elements: [
            $({ type: $etype:ident, nodes: [$en0:expr, $en1:expr],
                mat: $emat:ident $(, A: $ea:expr)? $(, Iz: $eiz:expr)? }),*
            $(,)?
        ]
        $(,)?
    ) => {{
        use $crate::model::{Model, Node};
        use fem_core::{ModelDim, NodeId};
        use materials::ElasticUniaxial;
        use elements::{ElasticBeam2d, Truss2d};
        use std::collections::HashMap;

        // Resolve dimensionality
        #[allow(unused_mut)]
        let mut _model = {
            let dim = $crate::_echelon_model_dim!($($dim)?);
            Model::new(dim)
        };

        // Add nodes
        $(
            _model.add_node(Node::new(
                NodeId($nid), $nx as f64, $ny as f64, $nz as f64
            )).unwrap();
        )*

        // Build a material map: identifier → ElasticUniaxial
        let mut _mats: HashMap<&str, ElasticUniaxial> = HashMap::new();
        $(
            {
                let _e: f64 = $crate::_echelon_material_e!($($me)?);
                let _rho: Option<f64> = $crate::_echelon_material_rho!($($mrho)?);
                _mats.insert(stringify!($mid), ElasticUniaxial::new(_e, _rho).unwrap());
            }
        )*

        // Add elements
        $(
            {
                let _mat = _mats[stringify!($emat)].clone();
                $crate::_echelon_add_element!(
                    _model, $etype,
                    $en0, $en1, _mat
                    $(, A: $ea)? $(, Iz: $eiz)?
                );
            }
        )*

        _model.build_state();
        _model
    }};
}

// ---- internal helpers (not public API) ----

#[doc(hidden)]
#[macro_export]
macro_rules! _echelon_model_dim {
    (Frame2d) => { ModelDim::frame_2d() };
    (Truss2d) => { ModelDim::truss_2d() };
    ()        => { ModelDim::frame_2d() }; // default
}

#[doc(hidden)]
#[macro_export]
macro_rules! _echelon_material_e {
    ($e:expr) => { $e as f64 };
    ()        => { 200e9_f64 }; // default steel
}

#[doc(hidden)]
#[macro_export]
macro_rules! _echelon_material_rho {
    ($rho:expr) => { Some($rho as f64) };
    ()          => { None };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _echelon_add_element {
    // Beam2d with A and Iz
    ($model:ident, Beam2d, $n0:expr, $n1:expr, $mat:expr, A: $a:expr, Iz: $iz:expr) => {{
        let x0 = $model.nodes[$n0].x;
        let y0 = $model.nodes[$n0].y;
        let x1 = $model.nodes[$n1].x;
        let y1 = $model.nodes[$n1].y;
        $model.add_element_typed(
            ElasticBeam2d::new(
                NodeId($n0), NodeId($n1),
                x0, y0, x1, y1,
                $mat, $a as f64, $iz as f64,
            ).unwrap()
        );
    }};
    // Beam2d with only A (Iz must be provided — this arm is a compile error catch)
    ($model:ident, Beam2d, $n0:expr, $n1:expr, $mat:expr, A: $a:expr) => {
        compile_error!("Beam2d elements require both A and Iz");
    };
    // Truss2d with A
    ($model:ident, Truss2d, $n0:expr, $n1:expr, $mat:expr, A: $a:expr) => {{
        let x0 = $model.nodes[$n0].x;
        let y0 = $model.nodes[$n0].y;
        let x1 = $model.nodes[$n1].x;
        let y1 = $model.nodes[$n1].y;
        $model.add_element_typed(
            Truss2d::new(
                NodeId($n0), NodeId($n1),
                x0, y0, x1, y1,
                $mat, $a as f64,
            ).unwrap()
        );
    }};
    // Truss2d with A and spurious Iz — silently ignore Iz
    ($model:ident, Truss2d, $n0:expr, $n1:expr, $mat:expr, A: $a:expr, Iz: $_iz:expr) => {{
        $crate::_echelon_add_element!($model, Truss2d, $n0, $n1, $mat, A: $a);
    }};
}