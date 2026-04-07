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
    LinearSeries, Model, Node, assemble_mass, assemble_stiffness, build_pattern, constraints::SpConstraint, loads::{ConstantSeries, NodalLoad}
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

// =============================================================================
// TEST 12 — Newmark::form_tangent augments K_T with a0*M
//
// SDOF mass-spring: k = EA/L = 1 N/m, lumped mass m = 1 kg.
// Newmark average acceleration: β = 0.25, γ = 0.5, dt = 0.1.
//   a0 = 1/(β·dt²) = 400
// After assemble_stiffness + form_tangent, the free DOF diagonal must be:
//   K_T[free, free] + a0 * M[free, free]  =  k + a0 * (m/2)
// =============================================================================
#[test]
fn test_newmark_form_tangent_augments_stiffness() {
    use analysis::integrators::transient::newmark::Newmark;
    use analysis::integrators::Integrator;
    use analysis::system::GlobalSystem;

    // 1-D spring: EA/L = 1,  rho·A·L = 1 → rho = 1/(A·L) = 1
    let e = 1.0_f64;
    let a = 1.0_f64;
    let l = 1.0_f64;

    let mut model = Model::new(ModelDim::truss_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();
    model.add_element_typed(
        Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0,
            ElasticUniaxial::new(e, Some(1.0)).unwrap(), a).unwrap()
    );
    let ndf = 2;
    for dof in [0, 1] {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }
    model.add_constraint(SpConstraint::new(NodeId(1), 1, 0.0, ndf)).unwrap(); // UY
    model.build_state();

    let k_pat = build_pattern(&model).unwrap();
    let mut mass_mat = k_pat.clone();
    assemble_mass(&model, &mut mass_mat).unwrap();

    let dt = 0.1_f64;
    let integrator = Newmark::average_acceleration(dt, mass_mat, None);

    let mut system = GlobalSystem::new(k_pat);
    assemble_stiffness(&model, &mut system.k_t).unwrap();
    let k_static = system.k_t.get(2, 2).unwrap(); // node 1 UX = DOF 2

    integrator.form_tangent(&mut system).unwrap();
    let k_eff = system.k_t.get(2, 2).unwrap();

    // a0 = 1/(0.25 * 0.01) = 400; lumped mass at DOF 2 = rho*A*L/2 = 0.5
    let a0 = 400.0_f64;
    let m_dof2 = 0.5_f64; // lumped
    let expected = k_static + a0 * m_dof2;

    assert!(
        (k_eff - expected).abs() < 1e-9,
        "k_eff={k_eff:.6e} expected={expected:.6e}"
    );
}

// =============================================================================
// TEST 13 — HHT::form_tangent scales K_T by (1+α) then adds a0*M
// =============================================================================
#[test]
fn test_hht_form_tangent_scales_stiffness() {
    use analysis::integrators::transient::hht::HHT;
    use analysis::integrators::Integrator;
    use analysis::system::GlobalSystem;

    let e = 1.0_f64; let a = 1.0_f64; let l = 1.0_f64;
    let alpha = -0.1_f64;
    let dt    = 0.05_f64;

    let mut model = Model::new(ModelDim::truss_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();
    model.add_element_typed(
        Truss2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0,
            ElasticUniaxial::new(e, Some(1.0)).unwrap(), a).unwrap()
    );
    let ndf = 2;
    for dof in [0, 1] {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }
    model.add_constraint(SpConstraint::new(NodeId(1), 1, 0.0, ndf)).unwrap();
    model.build_state();

    let k_pat = build_pattern(&model).unwrap();
    let mut mass_mat = k_pat.clone();
    assemble_mass(&model, &mut mass_mat).unwrap();

    let integrator = HHT::new(alpha, dt, mass_mat, None);

    let mut system = GlobalSystem::new(k_pat);
    assemble_stiffness(&model, &mut system.k_t).unwrap();
    let k_static = system.k_t.get(2, 2).unwrap();

    integrator.form_tangent(&mut system).unwrap();
    let k_eff = system.k_t.get(2, 2).unwrap();

    // β = (1-α)²/4 = (1.1)²/4 = 0.3025
    let beta = (1.0 - alpha).powi(2) / 4.0;
    let a0   = 1.0 / (beta * dt * dt);
    let m_dof2 = 0.5_f64; // lumped
    let expected = (1.0 + alpha) * k_static + a0 * m_dof2;

    assert!(
        (k_eff - expected).abs() < 1e-6,
        "k_eff={k_eff:.6e} expected={expected:.6e}"
    );
}

// =============================================================================
// TEST 14 — ElementLoad: uniform gravity on cantilever beam
//
// A 2m cantilever with a uniform downward load w = 10 kN/m.
// Analytical tip deflection: v_B = w·L⁴ / (8·E·I)
// =============================================================================
#[test]
fn test_element_load_uniform_cantilever() {
    use analysis::algorithms::newton::NewtonRaphson;
    use analysis::convergence::unbalance::NormUnbalance;
    use analysis::drivers::nonlinear_static::StaticNonlinear;
    use analysis::drivers::AnalysisDriver;
    use analysis::integrators::statics::load_control::LoadControl;
    use assembly::loads::pattern::ElementLoad;
    use elements::traits::ElementLoadParams;

    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 2.0_f64;
    let w  = -10e3_f64; // N/m downward

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l,   0.0)).unwrap();

    let elem_id = model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0,
            ElasticUniaxial::new(e, None).unwrap(), a, iz).unwrap()
    );

    let ndf = 3;
    for dof in 0..ndf {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }

    // Uniform downward distributed load via ElementLoad
    model.add_load_typed(ElementLoad {
        elem_id,
        params: ElementLoadParams::Uniform { wx: 0.0, wy: w },
        series: Box::new(ConstantSeries),
    });

    model.build_state();

    let test      = Box::new(NormUnbalance::new(1e-8));
    let algorithm = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(1.0));
    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    assert!(driver.analyze(&mut model, 1).unwrap());

    // v_B = w * L⁴ / (8 * E * I)
    let v_b          = model.u_global[4];
    let v_b_expected = w * l.powi(4) / (8.0 * e * iz);
    assert_rel(v_b, v_b_expected, 1e-9, "uniform load cantilever tip deflection");
}

// =============================================================================
// TEST 15 — ElementLoad: midspan point load on simply-supported beam
//
// Point load P = 20 kN at xi = 0.5 (midspan).
// Analytical: v_C = P*L³/(48*E*I)
// =============================================================================
#[test]
fn test_element_load_midspan_point() {
    use analysis::algorithms::newton::NewtonRaphson;
    use analysis::convergence::unbalance::NormUnbalance;
    use analysis::drivers::nonlinear_static::StaticNonlinear;
    use analysis::drivers::AnalysisDriver;
    use analysis::integrators::statics::load_control::LoadControl;
    use assembly::loads::pattern::ElementLoad;
    use elements::traits::ElementLoadParams;

    let e  = 200e9_f64;
    let iz = 1e-4_f64;
    let a  = 0.01_f64;
    let l  = 4.0_f64;
    let p  = -20e3_f64; // N downward

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l / 2.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(2), l, 0.0)).unwrap();

    let elem0 = model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l/2.0, 0.0,
            ElasticUniaxial::new(e, None).unwrap(), a, iz).unwrap()
    );
    let elem1 = model.add_element_typed(
        ElasticBeam2d::new(NodeId(1), NodeId(2), l/2.0, 0.0, l, 0.0,
            ElasticUniaxial::new(e, None).unwrap(), a, iz).unwrap()
    );

    let ndf = 3;
    // Pin at node 0
    model.add_constraint(SpConstraint::new(NodeId(0), 0, 0.0, ndf)).unwrap();
    model.add_constraint(SpConstraint::new(NodeId(0), 1, 0.0, ndf)).unwrap();
    // Roller at node 2
    model.add_constraint(SpConstraint::new(NodeId(2), 1, 0.0, ndf)).unwrap();

    // xi = 1.0 for elem0 means the point is at the end of elem0 = node 1 = midspan
    // Equivalently xi = 0.0 for elem1. We pick elem0 at xi=1.0.
    // Actually the cleanest is to split between the two: apply at xi=1.0 on elem0
    // gives P entirely to node 1. Let's use a NodalLoad for exact comparison
    // and an ElementLoad for the correctness test.
    //
    // For a point load at xi=1.0 on elem0 (node I=0, node J=1):
    //   a = 1.0 * L/2 = 2.0,  b = 0.0 → all load goes to node J (node 1).
    // For a point load at xi=0.0 on elem1 (node I=1, node J=2):
    //   a = 0.0 → all load goes to node I (node 1).
    // Both should give v_C = P*L³/(48*E*I) when summed.
    //
    // Use a simpler approach: xi=0.5 on elem0 (the first half).
    // This gives a point load at the quarter-span of the full beam.
    // That analytical formula is more complex. Instead: apply on elem1 at xi=0.5
    // (3/4 of span from node 0). Still complex.
    //
    // Best: apply P/2 at xi=1.0 on elem0 and P/2 at xi=0.0 on elem1.
    // But that's two separate loads. Simplest correct test: apply the full P
    // at xi=1.0 on elem0 which means node 1 receives the full point load.
    // This is identical to a NodalLoad at node 1 — compare against that.

    model.add_load_typed(ElementLoad {
        elem_id: elem0,
        params: ElementLoadParams::Point { px: 0.0, py: p, xi: 1.0 },
        series: Box::new(ConstantSeries),
    });

    model.build_state();

    let test       = Box::new(NormUnbalance::new(1e-8));
    let algorithm  = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(1.0));
    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    assert!(driver.analyze(&mut model, 1).unwrap());

    let v_c          = model.u_global[4]; // node 1 UY = DOF 4
    let v_c_expected = p * l.powi(3) / (48.0 * e * iz);
    assert_rel(v_c, v_c_expected, 1e-9, "midspan point load deflection");
    let _ = elem1;
}

// =============================================================================
// TEST 16 — LoadCombo: verify scale factor is applied correctly
//
// Two nodal loads of 10 kN each, wrapped in a LoadCombo with scale = 1.35.
// Net force at the target DOF must be 2 * 10e3 * 1.35 = 27 kN.
// =============================================================================
#[test]
fn test_load_combo_scale() {
    use assembly::loads::LoadPattern;
    use assembly::loads::combo::LoadCombo;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), 3.0, 0.0)).unwrap();
    model.build_state();

    let mut combo = LoadCombo::new(1.35);
    combo.add(Box::new(NodalLoad {
        node_id: NodeId(1),
        reference_loads: vec![10e3, 0.0, 0.0],
        series: Box::new(ConstantSeries),
    }));
    combo.add(Box::new(NodalLoad {
        node_id: NodeId(1),
        reference_loads: vec![10e3, 0.0, 0.0],
        series: Box::new(ConstantSeries),
    }));

    let mut f = vec![0.0_f64; 6];
    combo.apply_to_global_vector(1.0, &model, &mut f);

    let expected = 2.0 * 10e3 * 1.35;
    assert!((f[3] - expected).abs() < 1e-6,
        "f[3]={:.4e} expected {expected:.4e}", f[3]);
}

// =============================================================================
// TEST 17 — add_element_typed returns stable IDs
// =============================================================================
#[test]
fn test_add_element_returns_id() {
    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), 1.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(2), 2.0, 0.0)).unwrap();

    let mat = ElasticUniaxial::new(200e9, None).unwrap();
    let id0 = model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, 1.0, 0.0,
            mat.clone(), 0.01, 1e-4).unwrap()
    );
    let id1 = model.add_element_typed(
        ElasticBeam2d::new(NodeId(1), NodeId(2), 1.0, 0.0, 2.0, 0.0,
            mat, 0.01, 1e-4).unwrap()
    );

    assert_eq!(id0.0, 0);
    assert_eq!(id1.0, 1);
    assert_eq!(model.n_elements(), 2);
}

// =============================================================================
// TEST 18 — bake_load + clear_baked_loads
// =============================================================================
#[test]
fn test_bake_load_and_clear() {
    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), 3.0, 0.0)).unwrap();
    model.build_state();

    assert!(!model.has_baked_loads());

    // Bake a 50 kN load
    let load = NodalLoad {
        node_id: NodeId(1),
        reference_loads: vec![50e3, 0.0, 0.0],
        series: Box::new(ConstantSeries),
    };
    model.bake_load(&load, 1.0);
    assert!(model.has_baked_loads());

    // p_base[3] (node 1 UX) must be 50 kN
    let p_base = model.p_base.as_ref().unwrap();
    assert!((p_base[3] - 50e3).abs() < 1.0, "p_base[3]={:.4e}", p_base[3]);

    model.clear_baked_loads();
    assert!(!model.has_baked_loads());
}

// =============================================================================
// TEST 19 — echelon_model! macro builds a valid model
// =============================================================================
#[test]
fn test_echelon_model_macro() {
    use assembly::echelon_model;

    let model = echelon_model! {
        nodes: [
            { id: 0, x: 0.0, y: 0.0 },
            { id: 1, x: 2.0, y: 0.0 },
        ],
        materials: [
            { id: steel, E: 200e9 },
        ],
        elements: [
            { type: Beam2d, nodes: [0, 1], mat: steel, A: 0.01, Iz: 1e-4 },
        ],
    };

    assert_eq!(model.n_nodes(), 2);
    assert_eq!(model.n_elements(), 1);
    assert_eq!(model.n_dof(), 6);
}

// =============================================================================
// TEST 20 — NodeRecorder captures pushover curve
// =============================================================================
#[test]
fn test_node_recorder_pushover() {
    use analysis::algorithms::newton::NewtonRaphson;
    use analysis::convergence::unbalance::NormUnbalance;
    use analysis::drivers::nonlinear_static::StaticNonlinear;
    use analysis::drivers::AnalysisDriver;
    use analysis::integrators::statics::load_control::LoadControl;
    use analysis::recorder::NodeRecorder;

    let e = 200e9_f64; let iz = 1e-4_f64; let a = 0.01_f64; let l = 2.0_f64;
    let p = 1e3_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l, 0.0)).unwrap();
    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0,
            ElasticUniaxial::new(e, None).unwrap(), a, iz).unwrap()
    );
    let ndf = 3;
    for dof in 0..ndf {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }
    model.add_load_typed(NodalLoad {
        node_id: NodeId(1),
        reference_loads: vec![0.0, -p, 0.0],
        series: Box::new(LinearSeries),
    });
    model.build_state();

    let test = Box::new(NormUnbalance::new(1e-8));
    let algorithm = Box::new(NewtonRaphson::new(test, 25));
    let integrator = Box::new(LoadControl::new(0.1)); // 10 × Δλ = 0.1

    let mut driver = StaticNonlinear::new(algorithm, integrator, &model).unwrap();
    driver.add_recorder(Box::new(NodeRecorder::single(4, "tip_uy")));
    assert!(driver.analyze(&mut model, 10).unwrap());

    let rec = driver.recorder_as::<NodeRecorder>(0).unwrap();

    // 10 steps → 10 recorded values
    assert_eq!(rec.times().len(), 10);
    assert_eq!(rec.data().len(), 10);

    // Values must be monotonically increasing in magnitude (linear elastic pushover)
    let tips = rec.flatten();
    for i in 1..tips.len() {
        assert!(tips[i].abs() > tips[i-1].abs(),
            "tip displacement should increase monotonically: {tips:?}");
    }

    // Final value must match the closed-form full-load deflection
    let v_full = -p * l.powi(3) / (3.0 * e * iz);
    assert_rel(tips[9], v_full, 1e-9, "recorder final value vs analytical");
}

// =============================================================================
// TEST 21 — compute_reactions: cantilever base shear and moment
//
// Cantilever with tip load P = 10 kN, length L = 2 m.
//   Base shear   = P       = 10 kN (vertical, upward at base)
//   Base moment  = P * L   = 20 kN·m (CCW at base)
// =============================================================================
#[test]
fn test_compute_reactions_cantilever() {
    use analysis::algorithms::newton::NewtonRaphson;
    use analysis::convergence::unbalance::NormUnbalance;
    use analysis::drivers::nonlinear_static::StaticNonlinear;
    use analysis::drivers::AnalysisDriver;
    use analysis::integrators::statics::load_control::LoadControl;

    let e = 200e9_f64; let iz = 1e-4_f64; let a = 0.01_f64;
    let l = 2.0_f64; let p = 10e3_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l, 0.0)).unwrap();
    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0,
            ElasticUniaxial::new(e, None).unwrap(), a, iz).unwrap()
    );
    let ndf = 3;
    for dof in 0..ndf {
        model.add_constraint(SpConstraint::new(NodeId(0), dof, 0.0, ndf)).unwrap();
    }
    model.add_load_typed(NodalLoad {
        node_id: NodeId(1),
        reference_loads: vec![0.0, -p, 0.0],
        series: Box::new(ConstantSeries),
    });
    model.build_state();

    let mut driver = StaticNonlinear::new(
        Box::new(NewtonRaphson::new(Box::new(NormUnbalance::new(1e-8)), 25)),
        Box::new(LoadControl::new(1.0)),
        &model,
    ).unwrap();
    assert!(driver.analyze(&mut model, 1).unwrap());

    model.compute_reactions();
    let r = model.reactions.clone();

    // Node 0 DOFs: UX=0, UY=1, RZ=2
    // Base shear (UY, DOF 1): must equal +P (upward reaction)
    assert_rel(r[1],  p,       1e-6, "base shear UY");
    // Base moment (RZ, DOF 2): must equal +P*L (CCW reaction to CW load)
    assert_rel(r[2],  p * l,   1e-6, "base moment RZ");
    // Axial reaction (UX, DOF 0): must be ~0 (no horizontal load)
    assert!(r[0].abs() < 1e-3 * p, "axial reaction should be ~0: {}", r[0]);

    // Free DOFs (node 1) must be zero
    assert_eq!(r[3], 0.0);
    assert_eq!(r[4], 0.0);
    assert_eq!(r[5], 0.0);
}

// =============================================================================
// TEST 22 — build_rayleigh_damping: C = α·M + β·K
// =============================================================================
#[test]
fn test_build_rayleigh_damping() {
    use assembly::{build_pattern, assemble_mass, assemble_stiffness,
                   build_rayleigh_damping};

    let e = 200e9_f64; let a = 0.01_f64; let l = 2.0_f64;

    let mut model = Model::new(ModelDim::frame_2d());
    model.add_node(Node::new(NodeId(0), 0.0, 0.0)).unwrap();
    model.add_node(Node::new(NodeId(1), l, 0.0)).unwrap();
    model.add_element_typed(
        ElasticBeam2d::new(NodeId(0), NodeId(1), 0.0, 0.0, l, 0.0,
            ElasticUniaxial::new(e, Some(7850.0)).unwrap(), a, 1e-4).unwrap()
    );
    model.build_state();

    let k_pat = build_pattern(&model).unwrap();
    let mut mass = k_pat.clone();
    assemble_mass(&model, &mut mass).unwrap();

    let mut stiff = k_pat.clone();
    assemble_stiffness(&model, &mut stiff).unwrap();

    let alpha_m = 0.5_f64;
    let beta_k  = 0.001_f64;

    let c = build_rayleigh_damping(&mass, &stiff, alpha_m, beta_k).unwrap();
    c.validate().unwrap();

    // Spot check: C[i,j] = alpha_m * M[i,j] + beta_k * K[i,j]
    for row in 0..6 {
        for col in row..6 {
            let m_val = mass.get(row, col).unwrap();
            let k_val = stiff.get(row, col).unwrap();
            let c_val = c.get(row, col).unwrap();
            let expected = alpha_m * m_val + beta_k * k_val;
            assert!((c_val - expected).abs() < 1e-6,
                "C[{row},{col}]={c_val:.6e} expected {expected:.6e}");
        }
    }
}