//! Headless check of the default 18-link machine: start leaning 1°, then take
//! pushes on the top, middle, bottom and upper-middle links.
use cartpole18::sim::{Sim, CTRL_DT};

fn main() {
    let push: f64 = std::env::args().nth(1).map_or(1e-6, |a| a.parse().expect("push force in N"));
    let mut sim = Sim::new();
    let n = sim.p.n_links;
    println!(
        "{n} links, closed-loop spectral radius {:.6}, motor limit {} N, push {push:.2e} N",
        sim.ctrl.closed_loop_radius, sim.max_force
    );
    sim.reset(1f64.to_radians());

    let pushes = [(5.0, n - 1, 1.0), (10.0, n / 2, -1.0), (15.0, 0, 1.0), (20.0, 2 * n / 3, -1.0), (25.0, n - 1, -1.0)];
    let mut next = 0;
    let (mut peak_u, mut peak_angle, mut peak_x) = (0.0f64, 0.0f64, 0.0f64);
    let total = 32.0;
    for k in 0..(total / CTRL_DT) as usize {
        if next < pushes.len() && sim.t >= pushes[next].0 {
            let (_, link, dir) = pushes[next];
            sim.apply_push(link, dir * push);
            println!("t={:5.2}s  push link {:2} {}", sim.t, link + 1, if dir > 0.0 { "→" } else { "←" });
            next += 1;
        }
        sim.step();
        peak_u = peak_u.max(sim.u.abs());
        peak_angle = peak_angle.max(sim.max_abs_angle());
        peak_x = peak_x.max(sim.s[0].abs());
        if k % 2500 == 0 {
            println!("t={:5.2}s  x={:+.4} m  max|θ|={:.4}°  F={:+7.2} N", sim.t, sim.s[0], sim.max_abs_angle().to_degrees(), sim.u);
        }
        if sim.fallen() {
            println!("FELL at t={:.3}s", sim.t);
            std::process::exit(1);
        }
    }
    println!(
        "BALANCED for {total}s: peak |θ|={:.2}°, peak motor force={:.1} N, peak |x|={:.3} m, final max|θ|={:.4}°",
        peak_angle.to_degrees(), peak_u, peak_x, sim.max_abs_angle().to_degrees()
    );
}
