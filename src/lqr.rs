//! Discrete-time LQR for the cart + N-link chain.
//!
//! The upright linearization is formed analytically, discretized with a
//! zero-order hold, and the discrete Riccati equation is solved with the
//! structured doubling algorithm — all in double-double precision, because
//! the gains of an 18-link chain span many orders of magnitude and plain f64
//! loses the stabilizing solution.

use crate::dd::{Mat, DD};
use crate::model::Params;
use nalgebra::DVector;

/// Exact linearization about the upright equilibrium: ṡ = A s + B u.
pub fn linearize(p: &Params) -> (Mat, Mat) {
    let n = p.n_links;
    let d = p.dof();
    let dd = DD::new;
    let mut mu = vec![DD::ZERO; n];
    let mut acc = DD::ZERO;
    for i in (0..n).rev() {
        acc = acc + dd(p.link_mass[i]);
        mu[i] = acc;
    }
    let l: Vec<DD> = p.link_len.iter().map(|&v| dd(v)).collect();

    let mut m0 = Mat::zeros(d, d);
    m0[(0, 0)] = dd(p.cart_mass) + mu[0];
    for j in 0..n {
        m0[(0, 1 + j)] = mu[j] * l[j];
        m0[(1 + j, 0)] = mu[j] * l[j];
        for i in 0..n {
            m0[(1 + i, 1 + j)] = mu[i.max(j)] * l[i] * l[j];
        }
    }

    // generalized "stiffness" (gravity, destabilizing) and damping
    let mut g = Mat::zeros(d, d);
    let mut c = Mat::zeros(d, d);
    c[(0, 0)] = dd(-p.cart_damping);
    let cj = dd(p.joint_damping);
    for i in 0..n {
        g[(1 + i, 1 + i)] = mu[i] * dd(p.g) * l[i];
        c[(1 + i, 1 + i)] = c[(1 + i, 1 + i)] - cj;
        if i > 0 {
            c[(1 + i, i)] = c[(1 + i, i)] + cj;
        }
        if i + 1 < n {
            c[(1 + i, 1 + i)] = c[(1 + i, 1 + i)] - cj;
            c[(1 + i, 2 + i)] = c[(1 + i, 2 + i)] + cj;
        }
    }
    let mut e0 = Mat::zeros(d, 1);
    e0[(0, 0)] = DD::ONE;

    let minv_g = m0.solve(&g);
    let minv_c = m0.solve(&c);
    let minv_e = m0.solve(&e0);

    let ns = 2 * d;
    let mut a = Mat::zeros(ns, ns);
    let mut b = Mat::zeros(ns, 1);
    for i in 0..d {
        a[(i, d + i)] = DD::ONE;
        for j in 0..d {
            a[(d + i, j)] = minv_g[(i, j)];
            a[(d + i, d + j)] = minv_c[(i, j)];
        }
        b[(d + i, 0)] = minv_e[(i, 0)];
    }
    (a, b)
}

/// Matrix exponential via scaling and squaring of a Taylor series.
pub fn expm(m: &Mat) -> Mat {
    let n = m.rows;
    let norm = m.norm_inf();
    let squarings = if norm > 0.25 { (norm / 0.25).log2().ceil() as i32 } else { 0 };
    let scaled = m.scale(DD::ONE / DD::new(2f64.powi(squarings)));
    let mut result = Mat::identity(n);
    let mut term = Mat::identity(n);
    for k in 1..=30 {
        term = term.mul(&scaled).scale(DD::ONE / DD::new(k as f64));
        result = result.add(&term);
    }
    for _ in 0..squarings {
        result = result.mul(&result);
    }
    result
}

/// Zero-order-hold discretization.
pub fn c2d(a: &Mat, b: &Mat, dt: f64) -> (Mat, Mat) {
    let n = a.rows;
    let m = b.cols;
    let mut aug = Mat::zeros(n + m, n + m);
    let h = DD::new(dt);
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)] * h;
        }
        for j in 0..m {
            aug[(i, n + j)] = b[(i, j)] * h;
        }
    }
    let e = expm(&aug);
    let mut ad = Mat::zeros(n, n);
    let mut bd = Mat::zeros(n, m);
    for i in 0..n {
        for j in 0..n {
            ad[(i, j)] = e[(i, j)];
        }
        for j in 0..m {
            bd[(i, j)] = e[(i, n + j)];
        }
    }
    (ad, bd)
}

/// Stabilizing solution of the discrete algebraic Riccati equation for a
/// single input with weight r, via the structured doubling algorithm.
pub fn dare(a: &Mat, b: &Mat, q: &Mat, r: f64) -> Mat {
    let id = Mat::identity(a.rows);
    let mut ak = a.clone();
    let mut gk = b.mul(&b.transpose()).scale(DD::ONE / DD::new(r));
    let mut hk = q.clone();
    for _ in 0..200 {
        let w = id.add(&gk.mul(&hk));
        let w_ak = w.solve(&ak);
        let w_gk = w.solve(&gk);
        let a_next = ak.mul(&w_ak);
        let g_next = gk.add(&ak.mul(&w_gk).mul(&ak.transpose()));
        let h_next = hk.add(&ak.transpose().mul(&hk).mul(&w_ak));
        let delta = h_next.sub(&hk).norm_inf() / h_next.norm_inf();
        ak = a_next;
        gk = g_next.symmetrize();
        hk = h_next.symmetrize();
        if delta < 1e-28 {
            break;
        }
    }
    hk
}

/// Spectral radius estimate via repeated squaring: ρ = lim ‖Aᵏ‖^(1/k).
pub fn spectral_radius(a: &Mat) -> f64 {
    let mut m = a.clone();
    let mut log_scale = 0.0;
    let squarings = 40;
    for _ in 0..squarings {
        m = m.mul(&m);
        let s = m.norm_inf();
        m = m.scale(DD::ONE / DD::new(s));
        log_scale = 2.0 * log_scale + s.ln();
    }
    (log_scale / 2f64.powi(squarings)).exp()
}

pub struct Lqr {
    pub k: DVector<f64>,
    pub dt: f64,
    /// spectral radius of the designed discrete closed loop (must be < 1)
    pub closed_loop_radius: f64,
}

impl Lqr {
    /// `q_diag` holds the state weights for [x, θ1..θN, ẋ, θ̇1..θ̇N].
    pub fn design(p: &Params, q_diag: &[f64], r: f64, dt: f64) -> Self {
        let (a, b) = linearize(p);
        let (ad, bd) = c2d(&a, &b, dt);
        let mut q = Mat::zeros(a.rows, a.rows);
        for (i, w) in q_diag.iter().enumerate() {
            q[(i, i)] = DD::new(*w);
        }
        let pm = dare(&ad, &bd, &q, r);
        let btp = bd.transpose().mul(&pm);
        let denom = DD::new(r) + btp.mul(&bd)[(0, 0)];
        let k_row = btp.mul(&ad).scale(DD::ONE / denom);
        let acl = ad.sub(&bd.mul(&k_row));
        let closed_loop_radius = spectral_radius(&acl);
        let k = DVector::from_iterator(k_row.cols, k_row.d.iter().map(|v| v.to_f64()));
        Self { k, dt, closed_loop_radius }
    }

    /// u = -K s
    pub fn control(&self, s: &DVector<f64>) -> f64 {
        -self.k.dot(s)
    }
}
