//! Double-double arithmetic (~32 significant digits) and a minimal dense
//! matrix type. Used only for the offline LQR design, where the 18-link chain
//! is far too ill-conditioned for plain f64.

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DD {
    pub hi: f64,
    pub lo: f64,
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

impl DD {
    pub const ZERO: DD = DD { hi: 0.0, lo: 0.0 };
    pub const ONE: DD = DD { hi: 1.0, lo: 0.0 };

    pub fn new(x: f64) -> Self {
        DD { hi: x, lo: 0.0 }
    }

    pub fn abs(self) -> Self {
        if self.hi < 0.0 { -self } else { self }
    }

    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

impl Add for DD {
    type Output = DD;
    fn add(self, y: DD) -> DD {
        let (s, e) = two_sum(self.hi, y.hi);
        let (t, f) = two_sum(self.lo, y.lo);
        let (s, e) = quick_two_sum(s, e + t);
        let (hi, lo) = quick_two_sum(s, e + f);
        DD { hi, lo }
    }
}

impl Neg for DD {
    type Output = DD;
    fn neg(self) -> DD {
        DD { hi: -self.hi, lo: -self.lo }
    }
}

impl Sub for DD {
    type Output = DD;
    fn sub(self, y: DD) -> DD {
        self + (-y)
    }
}

impl Mul for DD {
    type Output = DD;
    fn mul(self, y: DD) -> DD {
        let p = self.hi * y.hi;
        let e = self.hi.mul_add(y.hi, -p) + (self.hi * y.lo + self.lo * y.hi);
        let (hi, lo) = quick_two_sum(p, e);
        DD { hi, lo }
    }
}

impl Div for DD {
    type Output = DD;
    fn div(self, y: DD) -> DD {
        let q1 = self.hi / y.hi;
        let r = self - y * DD::new(q1);
        let q2 = r.hi / y.hi;
        let r = r - y * DD::new(q2);
        let q3 = r.hi / y.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        DD { hi, lo } + DD::new(q3)
    }
}

/// Row-major dense matrix of double-doubles.
#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub d: Vec<DD>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, d: vec![DD::ZERO; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = DD::ONE;
        }
        m
    }

    pub fn transpose(&self) -> Mat {
        let mut t = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t[(j, i)] = self[(i, j)];
            }
        }
        t
    }

    pub fn mul(&self, o: &Mat) -> Mat {
        assert_eq!(self.cols, o.rows);
        let mut r = Mat::zeros(self.rows, o.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                if a.hi == 0.0 {
                    continue;
                }
                for j in 0..o.cols {
                    r[(i, j)] = r[(i, j)] + a * o[(k, j)];
                }
            }
        }
        r
    }

    pub fn add(&self, o: &Mat) -> Mat {
        let d = self.d.iter().zip(&o.d).map(|(a, b)| *a + *b).collect();
        Mat { rows: self.rows, cols: self.cols, d }
    }

    pub fn sub(&self, o: &Mat) -> Mat {
        let d = self.d.iter().zip(&o.d).map(|(a, b)| *a - *b).collect();
        Mat { rows: self.rows, cols: self.cols, d }
    }

    pub fn scale(&self, s: DD) -> Mat {
        let d = self.d.iter().map(|a| *a * s).collect();
        Mat { rows: self.rows, cols: self.cols, d }
    }

    pub fn symmetrize(&self) -> Mat {
        self.add(&self.transpose()).scale(DD::new(0.5))
    }

    /// Infinity norm (max absolute row sum), as f64.
    pub fn norm_inf(&self) -> f64 {
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self[(i, j)].hi.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    /// Solve self * X = b with partially pivoted Gaussian elimination.
    pub fn solve(&self, b: &Mat) -> Mat {
        let n = self.rows;
        assert_eq!(n, self.cols);
        let mut a = self.clone();
        let mut x = b.clone();
        for col in 0..n {
            let piv = (col..n)
                .max_by(|&i, &j| a[(i, col)].hi.abs().total_cmp(&a[(j, col)].hi.abs()))
                .unwrap();
            assert!(a[(piv, col)].hi != 0.0, "singular matrix");
            if piv != col {
                for j in 0..n {
                    a.d.swap(piv * n + j, col * n + j);
                }
                for j in 0..x.cols {
                    x.d.swap(piv * x.cols + j, col * x.cols + j);
                }
            }
            let inv = DD::ONE / a[(col, col)];
            for i in col + 1..n {
                let f = a[(i, col)] * inv;
                if f.hi == 0.0 {
                    continue;
                }
                for j in col..n {
                    a[(i, j)] = a[(i, j)] - f * a[(col, j)];
                }
                for j in 0..x.cols {
                    x[(i, j)] = x[(i, j)] - f * x[(col, j)];
                }
            }
        }
        for col in (0..n).rev() {
            for j in 0..x.cols {
                let mut s = x[(col, j)];
                for k in col + 1..n {
                    s = s - a[(col, k)] * x[(k, j)];
                }
                x[(col, j)] = s / a[(col, col)];
            }
        }
        x
    }
}

impl std::ops::Index<(usize, usize)> for Mat {
    type Output = DD;
    fn index(&self, (i, j): (usize, usize)) -> &DD {
        &self.d[i * self.cols + j]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut DD {
        &mut self.d[i * self.cols + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dd_division_is_extra_precise() {
        let third = DD::ONE / DD::new(3.0);
        let back = third * DD::new(3.0) - DD::ONE;
        assert!(back.to_f64().abs() < 1e-30);
    }

    #[test]
    fn solve_recovers_solution() {
        let mut a = Mat::zeros(3, 3);
        let vals = [4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        for (k, v) in vals.iter().enumerate() {
            a.d[k] = DD::new(*v);
        }
        let mut b = Mat::zeros(3, 1);
        b.d = vec![DD::ONE, DD::new(2.0), DD::new(3.0)];
        let x = a.solve(&b);
        let r = a.mul(&x).sub(&b);
        assert!(r.norm_inf() < 1e-28);
    }
}
