//! External (applied) load vector assembly.
//!
//! `assemble_load_vector` constructs the right-hand side of the linear system
//! by combining:
//!
//! 1. **`model.p_base`** — the "baked" gravity / pre-load vector set by
//!    `Model::lock_loads`. Present only after the first analysis phase.
//! 2. **Active load patterns** — each pattern evaluates its `TimeSeries` at
//!    `pseudo_time` and scatters scaled reference loads into `f_ext`.
//!
//! The result is the total external force vector at the current load level.
//! It is passed (after subtracting `f_int`) to the solver as the residual.
//!
//! ## Newton-Raphson usage
//!
//! In a load-controlled Newton-Raphson loop, `pseudo_time` is fixed for all
//! inner iterations of a given load step. Only between load steps does
//! `pseudo_time` advance:
//!
//! ```text
//! for pseudo_time in load_steps:
//!     assemble_load_vector(&model, pseudo_time, &mut f_ext)?;  ← once per step
//!     loop (Newton):
//!         assemble_stiffness(&model, &mut k)?;
//!         assemble_internal_force(&model, &mut f_int)?;
//!         r = f_ext - f_int;
//!         apply_dirichlet_bcs(&model.constraints, &mut k, &mut r)?;
//!         solve → delta_u
//!         u_global += delta_u
//!         if converged { commit_state; break }
//! ```

use crate::error::Result;
use crate::model::Model;

// -----------------------------------------------------------------
// assemble_load_vector
// -----------------------------------------------------------------

/// Assemble the global external load vector at `pseudo_time`.
///
/// 1. Zeros `f_ext`.
/// 2. If `model.p_base` is set, copies it into `f_ext` (seeds with baked loads).
/// 3. For each active load pattern, calls `pattern.apply_to_global_vector`.
///
/// The function does **not** apply boundary conditions — that is the
/// responsibility of `constraints::apply_dirichlet_bcs`, called after
/// assembly is complete.
///
/// # Arguments
/// * `model`       — read-only model (loads + p_base)
/// * `pseudo_time` — current load parameter or simulation time (seconds)
/// * `f_ext`       — mutable global external force vector, length `model.n_dof()`
///
/// # Errors
/// None in current implementation (returns `Result` for future extensibility
/// when distributed or ground-motion loads may fail during evaluation).
pub fn assemble_load_vector(
    model:       &Model,
    pseudo_time: f64,
    f_ext:       &mut [f64],
) -> Result<()> {
    // 1. Zero the output vector
    f_ext.fill(0.0);

    // 2. Seed from the baked base load if present
    if let Some(ref p_base) = model.p_base {
        f_ext.iter_mut()
            .zip(p_base.iter())
            .for_each(|(fi, &bi)| *fi = bi);
    }

    // 3. Apply each active load pattern additively
    for load in &model.loads {
        load.apply_to_global_vector(pseudo_time, model, f_ext, 1.0);
    }

    Ok(())
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::{ModelDim, NodeId};

    use crate::loads::pattern::NodalLoad;
    use crate::loads::series::{ConstantSeries, LinearSeries};
    use crate::model::{Model, Node};

    fn frame_two_nodes() -> Model {
        let mut m = Model::new(ModelDim::frame_2d());
        m.add_node(Node::new(NodeId(0), 0.0, 0.0, 0.0)).unwrap();
        m.add_node(Node::new(NodeId(1), 3.0, 0.0, 0.0)).unwrap();
        m.build_state();
        m
    }

    // ---- No loads → zero vector ----

    #[test]
    fn no_loads_produces_zero_vector() {
        let m = frame_two_nodes();
        let mut f = vec![1.0_f64; 6]; // pre-fill to ensure zeroing works
        assemble_load_vector(&m, 1.0, &mut f).unwrap();
        assert!(f.iter().all(|&v| v == 0.0));
    }

    // ---- Single constant nodal load ----

    #[test]
    fn constant_nodal_load_at_pseudo_time_1() {
        let mut m = frame_two_nodes();
        m.add_load(NodalLoad {
            node_id:         NodeId(1),
            reference_loads: vec![0.0, -50e3, 0.0],
            series:          Box::new(ConstantSeries),
        });

        let mut f = vec![0.0_f64; 6];
        assemble_load_vector(&m, 1.0, &mut f).unwrap();

        // Node 1: DOFs 3, 4, 5
        assert_eq!(f[3], 0.0);
        assert!((f[4] + 50e3).abs() < 1e-6);
        assert_eq!(f[5], 0.0);
    }

    // ---- Linear series at fractional pseudo_time ----

    #[test]
    fn linear_load_scales_with_pseudo_time() {
        let mut m = frame_two_nodes();
        m.add_load(NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![100e3, 0.0, 0.0],
            series:          Box::new(LinearSeries),
        });

        let mut f = vec![0.0_f64; 6];
        assemble_load_vector(&m, 0.4, &mut f).unwrap();

        assert!((f[0] - 40e3).abs() < 1e-6);
    }

    // ---- p_base seeds then load adds on top ----

    #[test]
    fn p_base_plus_active_load_sums_correctly() {
        let mut m = frame_two_nodes();
        m.p_base = Some(vec![0.0, -10e3, 0.0, 0.0, -10e3, 0.0]);

        m.add_load(NodalLoad {
            node_id:         NodeId(1),
            reference_loads: vec![0.0, -50e3, 0.0],
            series:          Box::new(ConstantSeries),
        });

        let mut f = vec![0.0_f64; 6];
        assemble_load_vector(&m, 1.0, &mut f).unwrap();

        // Node 0 UY: only from p_base = -10 kN
        assert!((f[1] + 10e3).abs() < 1e-6);
        // Node 1 UY: p_base (-10) + active (-50) = -60 kN
        assert!((f[4] + 60e3).abs() < 1e-6);
    }

    // ---- Multiple patterns accumulate ----

    #[test]
    fn two_patterns_accumulate_at_same_node() {
        let mut m = frame_two_nodes();
        m.add_load(NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![30e3, 0.0, 0.0],
            series:          Box::new(ConstantSeries),
        });
        m.add_load(NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![20e3, 0.0, 0.0],
            series:          Box::new(ConstantSeries),
        });

        let mut f = vec![0.0_f64; 6];
        assemble_load_vector(&m, 1.0, &mut f).unwrap();

        assert!((f[0] - 50e3).abs() < 1e-6);
    }

    // ---- Re-zeroing on each call ----

    #[test]
    fn second_call_does_not_accumulate() {
        let mut m = frame_two_nodes();
        m.add_load(NodalLoad {
            node_id:         NodeId(0),
            reference_loads: vec![100e3, 0.0, 0.0],
            series:          Box::new(ConstantSeries),
        });

        let mut f = vec![0.0_f64; 6];
        assemble_load_vector(&m, 1.0, &mut f).unwrap();
        let v1 = f[0];
        assemble_load_vector(&m, 1.0, &mut f).unwrap();
        let v2 = f[0];

        assert!((v1 - v2).abs() < 1e-10,
            "second call gave different result: {v1} vs {v2}");
    }
}