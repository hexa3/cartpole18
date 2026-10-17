//! Nonlinear dynamics of a cart carrying an N-link inverted pendulum chain.
//!
//! Generalized coordinates: q = [x, θ1, …, θN]
//!   x  – cart position (m), positive to the right
//!   θi – absolute angle of link i measured from the upward vertical (rad),
//!        positive clockwise (tip moves toward +x)
//! Each link is massless with a point mass at its outer end.
//! The only actuator is a horizontal force on the cart (the linear motor).

use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct Params {
    pub n_links: usize,
    pub cart_mass: f64,
    pub link_mass: Vec<f64>,
    pub link_len: Vec<f64>,
    pub g: f64,
    /// viscous friction on the cart rail (N·s/m)
    pub cart_damping: f64,
    /// viscous friction in every joint (N·m·s/rad)
    pub joint_damping: f64,
}

impl Params {
    pub fn uniform(n: usize, link_len: f64, link_mass: f64, cart_mass: f64) -> Self {
        Self {
            n_links: n,
            cart_mass,
            link_mass: vec![link_mass; n],
            link_len: vec![link_len; n],
            g: 9.81,
            cart_damping: 0.05,
            joint_damping: 1e-4,
        }
    }

    /// Chain whose links grow geometrically from the cart upward: link i has
    /// length `l0·ratioⁱ` and mass `m0·ratioⁱ` (constant mass per unit length).
    /// A uniform 18-link chain is so poorly controllable from the cart that its
    /// LQR gains exceed 1e14 and it is knocked over by sub-nano-newton pushes;
    /// the taper keeps the problem well posed while staying a true 18-link chain.
    pub fn tapered(n: usize, l0: f64, m0: f64, ratio: f64, cart_mass: f64) -> Self {
        let mut p = Self::uniform(n, l0, m0, cart_mass);
        for i in 0..n {
            p.link_len[i] = l0 * ratio.powi(i as i32);
            p.link_mass[i] = m0 * ratio.powi(i as i32);
        }
        p
    }

    pub fn dof(&self) -> usize {
        self.n_links + 1
    }

    pub fn state_dim(&self) -> usize {
        2 * self.dof()
    }

    /// μ_i = sum of the masses from link i outward (0-based link index).
    fn tail_masses(&self) -> Vec<f64> {
        let mut mu = vec![0.0; self.n_links];
        let mut acc = 0.0;
        for i in (0..self.n_links).rev() {
            acc += self.link_mass[i];
            mu[i] = acc;
        }
        mu
    }

    /// Cartesian positions of the cart pivot followed by each link's end point.
    pub fn joint_positions(&self, s: &DVector<f64>) -> Vec<(f64, f64)> {
        let mut pts = Vec::with_capacity(self.n_links + 1);
        let (mut px, mut py) = (s[0], 0.0);
        pts.push((px, py));
        for i in 0..self.n_links {
            let th = s[1 + i];
            px += self.link_len[i] * th.sin();
            py += self.link_len[i] * th.cos();
            pts.push((px, py));
        }
        pts
    }
}

/// A horizontal external push applied to the point mass at the end of `link`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Push {
    pub link: usize,
    pub force: f64,
}

/// State derivative ṡ = f(s, u, push), with s = [q; q̇].
pub fn deriv(p: &Params, s: &DVector<f64>, u: f64, push: Option<Push>) -> DVector<f64> {
    let n = p.n_links;
    let d = p.dof();
    let mu = p.tail_masses();
    let l = &p.link_len;
    let th = |i: usize| s[1 + i];
    let om = |i: usize| s[d + 1 + i];

    let mut m = DMatrix::<f64>::zeros(d, d);
    let mut rhs = DVector::<f64>::zeros(d);

    m[(0, 0)] = p.cart_mass + mu[0];
    rhs[0] = u - p.cart_damping * s[d];
    for j in 0..n {
        let c = mu[j] * l[j];
        m[(0, 1 + j)] = c * th(j).cos();
        m[(1 + j, 0)] = m[(0, 1 + j)];
        rhs[0] += c * th(j).sin() * om(j) * om(j);
    }

    for i in 0..n {
        for j in 0..n {
            let c = mu[i.max(j)] * l[i] * l[j];
            let dth = th(i) - th(j);
            m[(1 + i, 1 + j)] = c * dth.cos();
            rhs[1 + i] -= c * dth.sin() * om(j) * om(j);
        }
        rhs[1 + i] += mu[i] * p.g * l[i] * th(i).sin();

        // joint friction acts on relative joint rates
        let below = if i == 0 { 0.0 } else { om(i - 1) };
        rhs[1 + i] -= p.joint_damping * (om(i) - below);
        if i + 1 < n {
            rhs[1 + i] += p.joint_damping * (om(i + 1) - om(i));
        }
    }

    if let Some(push) = push {
        // virtual work of a horizontal force at mass `link`
        rhs[0] += push.force;
        for i in 0..=push.link.min(n - 1) {
            rhs[1 + i] += push.force * l[i] * th(i).cos();
        }
    }

    // M is symmetric positive definite for any finite state; a failed
    // factorization means the state has already blown up.
    let acc = match m.cholesky() {
        Some(ch) => ch.solve(&rhs),
        None => DVector::from_element(d, f64::NAN),
    };

    let mut ds = DVector::<f64>::zeros(2 * d);
    ds.rows_mut(0, d).copy_from(&s.rows(d, d));
    ds.rows_mut(d, d).copy_from(&acc);
    ds
}

pub fn rk4_step(p: &Params, s: &DVector<f64>, u: f64, push: Option<Push>, dt: f64) -> DVector<f64> {
    let k1 = deriv(p, s, u, push);
    let k2 = deriv(p, &(s + &k1 * (dt / 2.0)), u, push);
    let k3 = deriv(p, &(s + &k2 * (dt / 2.0)), u, push);
    let k4 = deriv(p, &(s + &k3 * dt), u, push);
    s + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
}
