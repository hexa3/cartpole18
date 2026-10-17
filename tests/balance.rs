use cartpole18::{lqr, model::{self, Params}, sim::Sim};
use nalgebra::DVector;

#[test]
fn analytic_linearization_matches_finite_differences() {
    let p = Params::tapered(5, 0.05, 0.1, 1.2, 1.0);
    let (a, b) = lqr::linearize(&p);
    let ns = p.state_dim();
    let z = DVector::<f64>::zeros(ns);
    let eps = 1e-6;
    for k in 0..ns {
        let mut sp = z.clone();
        let mut sm = z.clone();
        sp[k] += eps;
        sm[k] -= eps;
        let col = (model::deriv(&p, &sp, 0.0, None) - model::deriv(&p, &sm, 0.0, None)) / (2.0 * eps);
        for i in 0..ns {
            assert!((col[i] - a[(i, k)].to_f64()).abs() < 1e-6 * (1.0 + col[i].abs()), "A[{i},{k}]");
        }
    }
    let colb = (model::deriv(&p, &z, eps, None) - model::deriv(&p, &z, -eps, None)) / (2.0 * eps);
    for i in 0..ns {
        assert!((colb[i] - b[(i, 0)].to_f64()).abs() < 1e-6 * (1.0 + colb[i].abs()), "B[{i}]");
    }
}

#[test]
fn eighteen_link_chain_recovers_from_lean_and_pushes() {
    let mut sim = Sim::new();
    assert_eq!(sim.p.n_links, 18);
    assert!(sim.ctrl.closed_loop_radius < 1.0);
    sim.reset(1f64.to_radians());
    let pushes = [(3.0, 17, 1e-6), (7.0, 9, -1e-6), (11.0, 0, 1e-6)];
    let mut next = 0;
    while sim.t < 16.0 {
        if next < pushes.len() && sim.t >= pushes[next].0 {
            sim.apply_push(pushes[next].1, pushes[next].2);
            next += 1;
        }
        sim.step();
        assert!(!sim.fallen(), "fell at t={}", sim.t);
    }
    assert!(sim.max_abs_angle() < 0.1f64.to_radians());
}
