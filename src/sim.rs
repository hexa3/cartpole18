//! Closed-loop simulation: plant + sampled LQR + one force-limited linear motor.

use crate::lqr::Lqr;
use crate::model::{self, Params, Push};
use nalgebra::DVector;

pub const N_LINKS: usize = 18;
/// controller sample period (1 kHz)
pub const CTRL_DT: f64 = 1e-3;
/// physics sub-steps per control period
pub const SUBSTEPS: usize = 5;
/// duration of a single push pulse
pub const PUSH_DURATION: f64 = 0.05;

pub struct Sim {
    pub p: Params,
    pub ctrl: Lqr,
    pub s: DVector<f64>,
    pub t: f64,
    /// motor force currently commanded (after saturation)
    pub u: f64,
    /// linear motor force limit (N)
    pub max_force: f64,
    /// active external push and the time it ends
    pub push: Option<(Push, f64)>,
}

impl Sim {
    /// The default 18-link machine: links from 3 cm / 0.1 kg at the cart up to
    /// 67 cm / 2.2 kg at the top (3.83 m tall, 12.8 kg chain), 1 kg cart.
    pub fn new() -> Self {
        Self::with_params(Params::tapered(N_LINKS, 0.03, 0.1, 1.2, 1.0), 1e-4)
    }

    pub fn with_params(p: Params, r: f64) -> Self {
        let d = p.dof();
        let mut q = vec![1.0; 2 * d]; // cart position and all velocities
        for w in &mut q[1..d] {
            *w = 100.0; // link angles
        }
        let ctrl = Lqr::design(&p, &q, r, CTRL_DT);
        let s = DVector::zeros(p.state_dim());
        Self { p, ctrl, s, t: 0.0, u: 0.0, max_force: 150.0, push: None }
    }

    /// Reset with every link leaning by the same angle (rad).
    pub fn reset(&mut self, lean: f64) {
        self.s.fill(0.0);
        for i in 0..self.p.n_links {
            self.s[1 + i] = lean;
        }
        self.t = 0.0;
        self.u = 0.0;
        self.push = None;
    }

    /// Apply a horizontal push of `force` newtons on the end of `link` for one pulse.
    pub fn apply_push(&mut self, link: usize, force: f64) {
        self.push = Some((Push { link, force }, self.t + PUSH_DURATION));
    }

    /// Advance one control period.
    pub fn step(&mut self) {
        self.u = self.ctrl.control(&self.s).clamp(-self.max_force, self.max_force);
        let h = CTRL_DT / SUBSTEPS as f64;
        for _ in 0..SUBSTEPS {
            let push = match self.push {
                Some((p, end)) if self.t < end => Some(p),
                _ => {
                    self.push = None;
                    None
                }
            };
            self.s = model::rk4_step(&self.p, &self.s, self.u, push, h);
            self.t += h;
        }
    }

    pub fn max_abs_angle(&self) -> f64 {
        (1..=self.p.n_links).map(|i| self.s[i].abs()).fold(0.0, f64::max)
    }

    pub fn fallen(&self) -> bool {
        !self.s.iter().all(|v| v.is_finite()) || self.max_abs_angle() > 0.5
    }
}

impl Default for Sim {
    fn default() -> Self {
        Self::new()
    }
}
