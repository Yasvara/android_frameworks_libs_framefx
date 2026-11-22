// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Complex FFT for the matcher's phase correlation: radix-2 for powers of two, Bluestein
//! otherwise. Unnormalised in both directions.

use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug, Default)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    pub fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    pub fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn cis(a: f64) -> C {
        C::new(a.cos(), a.sin())
    }
}

impl Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl Div<f64> for C {
    type Output = C;
    fn div(self, s: f64) -> C {
        C::new(self.re / s, self.im / s)
    }
}

fn radix2(a: &mut [C], inverse: bool) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let sgn = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let wl = C::cis(sgn * 2.0 * PI / len as f64);
        for s in (0..n).step_by(len) {
            let mut w = C::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = a[s + k];
                let v = a[s + k + len / 2] * w;
                a[s + k] = u + v;
                a[s + k + len / 2] = u - v;
                w = w * wl;
            }
        }
        len <<= 1;
    }
}

pub fn fft(a: &mut [C], inverse: bool) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2(a, inverse);
        return;
    }
    let sgn = if inverse { 1.0 } else { -1.0 };
    let m = (2 * n - 1).next_power_of_two();
    let chirp: Vec<C> = (0..n).map(|k| C::cis(sgn * PI * ((k * k) % (2 * n)) as f64 / n as f64)).collect();
    let mut x = vec![C::default(); m];
    let mut y = vec![C::default(); m];
    for k in 0..n {
        x[k] = a[k] * chirp[k];
    }
    y[0] = chirp[0].conj();
    for k in 1..n {
        y[k] = chirp[k].conj();
        y[m - k] = chirp[k].conj();
    }
    radix2(&mut x, false);
    radix2(&mut y, false);
    for k in 0..m {
        x[k] = x[k] * y[k];
    }
    radix2(&mut x, true);
    for k in 0..n {
        a[k] = x[k] / m as f64 * chirp[k];
    }
}

pub fn fft2(a: &mut [C], w: usize, h: usize, inverse: bool) {
    let mut row = vec![C::default(); w];
    for y in 0..h {
        row.copy_from_slice(&a[y * w..(y + 1) * w]);
        fft(&mut row, inverse);
        a[y * w..(y + 1) * w].copy_from_slice(&row);
    }
    let mut col = vec![C::default(); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = a[y * w + x];
        }
        fft(&mut col, inverse);
        for y in 0..h {
            a[y * w + x] = col[y];
        }
    }
}
