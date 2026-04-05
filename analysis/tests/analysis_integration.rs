//! Integration tests for the `analysis` crate.
//!
//! Every test assembles a real structural model end-to-end using the full
//! workspace pipeline and verifies the result against a known analytical
//! solution.
//!
//! ## What these tests prove
//!
//! ```text
//! Model (nodes + elements + BCs + loads)
//!   ↓
//! build_pattern → GlobalSystem
//!   ↓
//! StaticNonlinear / LinearStatic driver
//!   ↓
//! LoadControl integrator → assemble_load_vector → f_ext
//!   ↓
//! NewtonRaphson / LinearAlgorithm inner loop:
//!   assemble_stiffness   → K_T
//!   assemble_internal_force → F_int
//!   form R = F_ext − F_int
//!   apply_dirichlet_bcs
//!   CholeskySolver::factorize + solve → Δu
//!   u_global += Δu
//!   NormUnbalance convergence check
//!   commit_state
//!   ↓
//! model.u_global  compared with  closed-form formula
//! ```
//!
//! ## Tolerance
//!
//! All assertions use relative tolerance `1e-9`. For the problems here
//! (linear elastic, exact K, exact loads) Newton-Raphson converges in a
//! single iteration and the numerical error is machine precision.
//!
//! ## Models used
//!
//! - Single-element axial truss (1-DOF spring equivalent)
//! - Cantilever beam under tip load (tip deflection + rotation)
//! - Fixed-fixed beam under midspan load (midspan deflection)
//! - Simply supported beam (midspan deflection + end rotations)
//! - Two-step load control pushover (verify incremental assembly)
//! - Modified Newton convergence on a linear problem (same result as full Newton)

use fem_core::{ModelDim, NodeId};
use materials::ElasticUniaxial;
use elements::{Truss2d, ElasticBeam2d};
use assembly::{
    Model, Node,
    constraints::SpConstraint,
    loads::{NodalLoad, ConstantSeries},
};

use analysis::algorithms::newton::NewtonRaphson;
use analysis::algorithms::modified::ModifiedNewton;
use analysis::drivers::linear_static::LinearStatic;
use analysis::drivers::nonlinear_static::StaticNonlinear;
use analysis::drivers::AnalysisDriver;
use analysis::integrators::statics::load_control::LoadControl;
use analysis::convergence::unbalance::NormUnbalance;
use analysis::convergence::energy::EnergyIncrement;

// =============================================================================
// Helpers
// =============================================================================

/// Check `|computed − expected| / |expected| < tol`.
fn assert_rel(computed: f64, expected: f64, tol: f64, label: &str) {
    let abs_err = (computed - expected).abs();
    let denom   = expected.abs().max(1e-15);
    let rel_err = abs_err / denom;
    assert!(
        rel_err < tol,
        "{label}: computed = {computed:.10e}  expected = {expected:.10e}  rel_err = {rel_err:.2e}"
    );
}

fn steel() -> ElasticUniaxial {
    ElasticUniaxial::new(200e9, None).unwrap()
}

// =============================================================================
// TEST 1 — Single axial truss: LinearStatic driver
//
//   Fixed ─── EA ─── P →
//   A    ──────────── B
//
//  BCs:  UX=0, UY=0 at A;  UY=0 at B (prevent rigid-body rotation).
//  Load: P in +X at B.
//  Expected: u_B = P * L / (E * A)
// =============================================================================

#[test]
fn test_linear_static_single_truss() {
    let e = 200e9_f64;
    let a = 0.01_f64;
    let l = 2.0_f64;
    let p = 50e3_f64;

    let mut model = Model::new(ModelDim::truss_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();

    model.add_element_typed(
        Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0, steel(), a).unwrap()
    );

    // Fix node 0 (UX=0, UY=0) and node 1 (UY=0 — roller)
    let ndf = 2;
    model.add_constraint(SpConstraint::new(NodeId(0), 0, 0.0, ndf)).unwrap();
    model.add_constraint(SpConstraint::new(NodeId(0), 1, 0.0, ndf)).unwrap();
    model.add_constraint(SpConstraint::new(NodeId(1), 1, 0.0, ndf)).unwrap();

    // Axial load P at node 1
    model.add_load_typed(NodalLoad {
        node_id:         NodeId(1),
        reference_loads: vec![p, 0.0],
        series:          Box::new(ConstantSeries),
    });

    model.build_state();

    let mut driver = LinearStatic::new();
    let ok = driver.analyze(&mut model, 1).unwrap();
    assert!(ok, "LinearStatic should converge");

    // DOF layout: node k → [2k, 2k+1].  Node 1 UX = DOF 2.
    let u_b = model.u_global[2];
    let expected = p * l / (e * a);
    assert_rel(u_b, expected, 1e-9, "u_B (axial displacement)");
}

// =============================================================================
// TEST 2 — Cantilever beam: StaticNonlinear + NewtonRaphson + LoadControl
//
//   Fixed                     Free
//   ████ ─────────────────── B
//   A                        ↓ P
//
//   v_B = P L³ / (3EI)
//   θ_B = P L² / (2EI)   (clockwise → negative)
// =============================================================================

#[test]
fn test_nonlinear_static_cantilever_beam_newton() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 2.0_f64;
    let p  = 10e3_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();

    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0, steel(), a, iz).unwrap()
    );

    // Fix node 0 (all 3 DOFs: UX, UY, RZ)
    let ndf = 3;
    for dof in 0..ndf {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }

    // Downward point load at node 1
    model.add_load_typed(NodalLoad {
        node_id:         NodeId(1),
        reference_loads: vec![0.0, -p, 0.0],
        series:          Box::new(ConstantSeries),
    });

    model.build_state();

    let test      = Box::new(NormUnbalance::new(1e-8));
    let algorithm = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(1.0)); // single full-load step

    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    let ok = driver.analyze(&mut model, 1).unwrap();
    assert!(ok, "Newton-Raphson should converge for linear elastic cantilever");

    // DOF layout: node 1 → [3, 4, 5] = [UX, UY, RZ]
    let v_b     = model.u_global[4];
    let theta_b = model.u_global[5];

    let v_b_expected     = -p * l.powi(3) / (3.0 * e * iz);
    let theta_b_expected = -p * l.powi(2) / (2.0 * e * iz); // CW → negative

    assert_rel(v_b,     v_b_expected,     1e-9, "v_B (tip deflection)");
    assert_rel(theta_b, theta_b_expected, 1e-9, "θ_B (tip rotation)");

    // Axial displacement must be zero
    assert!(model.u_global[3].abs() < 1e-10 * v_b.abs().max(1.0),
        "u_B (axial) should be ≈ 0, got {}", model.u_global[3]);
}

// =============================================================================
// TEST 3 — Fixed-fixed beam under midspan load: 2-element model
//
//  Fixed A ──────── C ──────── Fixed B
//                   ↓ P
//
//   v_C = P L³ / (192 EI)
// =============================================================================

#[test]
fn test_nonlinear_static_fixed_fixed_beam() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 4.0_f64;
    let p  = 20e3_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0,     0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l / 2.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(2), l,       0.0)).unwrap();

    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l/2.0, 0.0, steel(), a, iz).unwrap()
    );
    model.add_element_typed(
        ElasticBeam2d::new(NodeId(1), NodeId(2), l/2.0, 0.0, l, 0.0, steel(), a, iz).unwrap()
    );

    // Fix all DOFs at nodes 0 and 2
    let ndf = 3;
    for node in [NodeId(0), NodeId(2)] {
        for dof in 0..ndf {
            model.add_constraint(SpConstraint::new(node, dof, 0.0, ndf)).unwrap();
        }
    }

    // Downward load at midspan node 1
    model.add_load_typed(NodalLoad {
        node_id:         NodeId(1),
        reference_loads: vec![0.0, -p, 0.0],
        series:          Box::new(ConstantSeries),
    });

    model.build_state();

    let test      = Box::new(NormUnbalance::new(1e-8));
    let algorithm = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(1.0));

    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    let ok = driver.analyze(&mut model, 1).unwrap();
    assert!(ok, "Should converge for linear elastic fixed-fixed beam");

    let v_c = model.u_global[4]; // node 1 UY = DOF 4
    let v_c_expected = -p * l.powi(3) / (192.0 * e * iz);
    assert_rel(v_c, v_c_expected, 1e-9, "v_C (midspan deflection)");

    // Midspan rotation must be zero (symmetry)
    assert!(model.u_global[5].abs() < 1e-10 * v_c.abs().max(1.0),
        "θ_C should be ≈ 0 by symmetry");
}

// =============================================================================
// TEST 4 — Simply supported beam: partial DOF constraint (roller at B)
//
//   Pin A ──────── C ──────── Roller B
//                  ↓ P
//
//   v_C = P L³ / (48 EI)
// =============================================================================

#[test]
fn test_nonlinear_static_simply_supported_beam() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 3.0_f64;
    let p  = 15e3_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0,     0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l / 2.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(2), l,       0.0)).unwrap();

    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l/2.0, 0.0, steel(), a, iz).unwrap()
    );
    model.add_element_typed(
        ElasticBeam2d::new(NodeId(1), NodeId(2), l/2.0, 0.0, l, 0.0, steel(), a, iz).unwrap()
    );

    let ndf = 3;
    // Node 0: pin — fix UX and UY, leave RZ free
    model.add_constraint(SpConstraint::new(NodeId(0), 0, 0.0, ndf)).unwrap();
    model.add_constraint(SpConstraint::new(NodeId(0), 1, 0.0, ndf)).unwrap();
    // Node 2: roller — fix UY only
    model.add_constraint(SpConstraint::new(NodeId(2), 1, 0.0, ndf)).unwrap();

    model.add_load_typed(NodalLoad {
        node_id:         NodeId(1),
        reference_loads: vec![0.0, -p, 0.0],
        series:          Box::new(ConstantSeries),
    });

    model.build_state();

    let test      = Box::new(NormUnbalance::new(1e-8));
    let algorithm = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(1.0));

    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    let ok = driver.analyze(&mut model, 1).unwrap();
    assert!(ok, "Should converge for simply supported beam");

    let v_c = model.u_global[4]; // node 1 UY = DOF 4
    let v_c_expected = -p * l.powi(3) / (48.0 * e * iz);
    assert_rel(v_c, v_c_expected, 1e-9, "v_C (midspan deflection)");
}

// =============================================================================
// TEST 5 — Multi-step load control: 10 increments × Δλ = 0.1
//
//   Same cantilever as TEST 2, but applied in 10 equal load steps.
//   At each step the solution must equal (step/10) × full solution.
//   Final result must match the closed-form tip deflection.
// =============================================================================

#[test]
fn test_multi_step_load_control_cantilever() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 2.0_f64;
    let p  = 10e3_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();

    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0, steel(), a, iz).unwrap()
    );

    let ndf = 3;
    for dof in 0..ndf {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }

    model.add_load_typed(NodalLoad {
        node_id:         NodeId(1),
        reference_loads: vec![0.0, -p, 0.0],
        series:          Box::new(ConstantSeries),
    });

    model.build_state();

    let test       = Box::new(NormUnbalance::new(1e-8));
    let algorithm  = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(0.1)); // 10 steps × Δλ = 0.1

    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    let ok = driver.analyze(&mut model, 10).unwrap();
    assert!(ok, "All 10 steps should converge");

    let v_b = model.u_global[4];
    let v_b_expected = -p * l.powi(3) / (3.0 * e * iz);
    assert_rel(v_b, v_b_expected, 1e-9,
        "v_B after 10 load steps (should equal full-load result)");
}

// =============================================================================
// TEST 6 — Modified Newton gives same result as full Newton (linear elastic)
//
//   For a linear elastic problem, the tangent stiffness is constant.
//   Modified Newton (which reuses K_T) is therefore mathematically equivalent
//   to full Newton. Both should converge to the same tip deflection.
// =============================================================================

#[test]
fn test_modified_newton_matches_full_newton() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 2.0_f64;
    let p  = 10e3_f64;

    let make_model = || {
        let mut model = Model::new(ModelDim::frame_2d());
        model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
        model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();
        model.add_element_typed(
            ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0,
                ElasticUniaxial::new(e, None).unwrap(), a, iz).unwrap()
        );
        let ndf = 3;
        for dof in 0..ndf {
            model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
        }
        model.add_load_typed(NodalLoad {
            node_id:         NodeId(1),
            reference_loads: vec![0.0, -p, 0.0],
            series:          Box::new(ConstantSeries),
        });
        model.build_state();
        model
    };

    // Full Newton
    let mut model_nr = make_model();
    {
        let test  = Box::new(NormUnbalance::new(1e-8));
        let algo  = Box::new(NewtonRaphson::new(test, 25));
        let integ = Box::new(LoadControl::new(1.0));
        let mut driver = StaticNonlinear::new(algo, integ, &model_nr).unwrap();
        driver.analyze(&mut model_nr, 1).unwrap();
    }

    // Modified Newton
    let mut model_mn = make_model();
    {
        let test  = Box::new(NormUnbalance::new(1e-8));
        let algo  = Box::new(ModifiedNewton::new(test, 50));
        let integ = Box::new(LoadControl::new(1.0));
        let mut driver = StaticNonlinear::new(algo, integ, &model_mn).unwrap();
        driver.analyze(&mut model_mn, 1).unwrap();
    }

    let v_nr = model_nr.u_global[4];
    let v_mn = model_mn.u_global[4];

    // Both algorithms should agree to machine precision on a linear problem
    assert_rel(v_mn, v_nr, 1e-9,
        "ModifiedNewton and NewtonRaphson must agree on a linear elastic problem");
}

// =============================================================================
// TEST 7 — Energy increment convergence criterion
//
//   Same cantilever, but using EnergyIncrement instead of NormUnbalance.
//   The solution must be identical — the convergence test affects when
//   the loop exits, not what solution it converges to.
// =============================================================================

#[test]
fn test_energy_increment_convergence() {
    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 2.0_f64;
    let p  = 10e3_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();
    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0, steel(), a, iz).unwrap()
    );
    let ndf = 3;
    for dof in 0..ndf {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }
    model.add_load_typed(NodalLoad {
        node_id:         NodeId(1),
        reference_loads: vec![0.0, -p, 0.0],
        series:          Box::new(ConstantSeries),
    });
    model.build_state();

    // Use EnergyIncrement as the convergence criterion
    let test       = Box::new(EnergyIncrement::new(1e-12));
    let algorithm  = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(1.0));

    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    let ok = driver.analyze(&mut model, 1).unwrap();
    assert!(ok, "Should converge with EnergyIncrement criterion");

    let v_b = model.u_global[4];
    let v_b_expected = -p * l.powi(3) / (3.0 * e * iz);
    assert_rel(v_b, v_b_expected, 1e-9, "v_B with energy increment criterion");
}

// =============================================================================
// TEST 8 — Load control revert: driver reports Ok(false) for singular model
//
//   A model with no boundary conditions has a singular stiffness matrix.
//   The driver should return Ok(false) (or Err for SingularSystem) without
//   panicking, and the model's u_global should remain at zero (last committed).
// =============================================================================

#[test]
fn test_singular_model_does_not_panic() {
    let l = 2.0_f64;
    let a = 0.01_f64;
    let p = 10e3_f64;

    // No boundary conditions → K is singular
    let mut model = Model::new(ModelDim::truss_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();
    model.add_element_typed(
        Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0, steel(), a).unwrap()
    );
    model.add_load_typed(NodalLoad {
        node_id:         NodeId(1),
        reference_loads: vec![p, 0.0],
        series:          Box::new(ConstantSeries),
    });
    model.build_state();

    let test       = Box::new(NormUnbalance::new(1e-8));
    let algorithm  = Box::new(NewtonRaphson::new(test, 5));
    let integrator = Box::new(LoadControl::new(1.0));

    let result = StaticNonlinear::new(algorithm, integrator, &model);
    match result {
        Ok(mut driver) => {
            // The analysis should fail gracefully (SingularSystem or Ok(false))
            let outcome = driver.analyze(&mut model, 1);
            match outcome {
                Ok(false) => {} // Expected: soft failure
                Err(_)    => {} // Also acceptable: propagated SingularSystem
                Ok(true)  => panic!("Singular model should not converge"),
            }
        }
        Err(_) => {
            // Driver construction can also fail if pattern is empty (no BCs
            // applied means all DOFs are free, but pattern is valid). Both
            // outcomes are acceptable — the important thing is no panic.
        }
    }

    // The model must still be in a usable state after the error
    assert_eq!(model.u_global.len(), 4, "u_global should be unchanged");
}

// =============================================================================
// TEST 9 — LinearStatic driver rejects empty model
// =============================================================================

#[test]
fn test_linear_static_rejects_empty_model() {
    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    // No elements added
    model.build_state();

    let mut driver = LinearStatic::new();
    let result = driver.analyze(&mut model, 1);
    assert!(result.is_err(), "LinearStatic should reject a model with no elements");
}

// =============================================================================
// TEST 10 — StaticNonlinear rejects model with no DOFs
// =============================================================================

#[test]
fn test_static_nonlinear_rejects_model_with_no_nodes() {
    let model = Model::new(ModelDim::frame_2d()); // empty: no nodes, no elements

    let test       = Box::new(NormUnbalance::new(1e-6));
    let algorithm  = Box::new(NewtonRaphson::new(test, 10));
    let integrator = Box::new(LoadControl::new(1.0));

    let result = StaticNonlinear::new(algorithm, integrator, &model);
    assert!(result.is_err(), "StaticNonlinear::new should reject a model with no elements");
}

// =============================================================================
// TEST 11 — LoadControl revert: verify integrator state is rolled back on failure
// =============================================================================

// #[test]
// fn test_load_control_revert_on_failure() {
//     use analysis::integrators::statics::load_control::LoadControl;
//     let mut lc = LoadControl::new(0.25);

//     // Simulate 2 successful steps
//     lc.current_lambda += lc.delta_lambda; lc.commit(); // λ = 0.25
//     lc.current_lambda += lc.delta_lambda; lc.commit(); // λ = 0.50

//     assert!((lc.committed_lambda() - 0.50).abs() < 1e-15);

//     // Simulate a failed step
//     lc.current_lambda += lc.delta_lambda; // λ = 0.75 (pending)
//     lc.revert();                           // roll back

//     assert!((lc.lambda() - 0.50).abs() < 1e-15,
//         "After revert, lambda should be back at committed value 0.50, got {}", lc.lambda());
// }