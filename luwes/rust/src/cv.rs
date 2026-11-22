// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The handful of image operators the reference took from OpenCV, with OpenCV's border rules,
//! rounding and label order.

use crate::grid::{Grid, Mask, Plane, Rgb};

fn reflect101(i: i64, n: usize) -> usize {
    let n = n as i64;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - 2 - i;
        } else {
            return i as usize;
        }
    }
}

fn replicate(i: i64, n: usize) -> usize {
    i.clamp(0, n as i64 - 1) as usize
}

#[derive(Clone, Copy)]
pub enum Border {
    Reflect101,
    Constant,
}

fn bidx(i: i64, n: usize, b: Border) -> Option<usize> {
    match b {
        Border::Reflect101 => Some(reflect101(i, n)),
        Border::Constant => (i >= 0 && i < n as i64).then_some(i as usize),
    }
}

/// Unnormalised (sum) or normalised box filter of size n x n, anchor at the centre, float64 sums.
pub fn box_filter(a: &Plane, n: usize, normalize: bool, border: Border) -> Plane {
    let (w, h) = (a.w, a.h);
    let lo = (n / 2) as i64;
    let mut rows = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0f64;
            for k in 0..n as i64 {
                if let Some(xx) = bidx(x as i64 + k - lo, w, border) {
                    s += a.at(xx, y) as f64;
                }
            }
            rows[y * w + x] = s;
        }
    }
    let scale = if normalize { 1.0 / (n * n) as f64 } else { 1.0 };
    Grid::from_fn(w, h, |x, y| {
        let mut s = 0.0f64;
        for k in 0..n as i64 {
            if let Some(yy) = bidx(y as i64 + k - lo, h, border) {
                s += rows[yy * w + x];
            }
        }
        (s * scale) as f32
    })
}

/// cv2.blur (normalised box, BORDER_REFLECT_101).
pub fn blur(a: &Plane, n: usize) -> Plane {
    box_filter(a, n, true, Border::Reflect101)
}

/// cv2.resize INTER_AREA for an exact 2x downscale.
pub fn resize_area2(a: &Plane) -> Plane {
    assert!(a.w % 2 == 0 && a.h % 2 == 0, "area downscale needs even dims");
    Grid::from_fn(a.w / 2, a.h / 2, |x, y| {
        let (x2, y2) = (2 * x, 2 * y);
        let r0 = a.at(x2, y2) + a.at(x2 + 1, y2);
        let r1 = a.at(x2, y2 + 1) + a.at(x2 + 1, y2 + 1);
        (r0 + r1) * 0.25
    })
}

pub fn resize_area2_rgb(a: &Grid<Rgb>) -> Grid<Rgb> {
    assert!(a.w % 2 == 0 && a.h % 2 == 0, "area downscale needs even dims");
    Grid::from_fn(a.w / 2, a.h / 2, |x, y| {
        let (x2, y2) = (2 * x, 2 * y);
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            let s = ((a.at(x2, y2)[c] + a.at(x2 + 1, y2)[c]) + a.at(x2, y2 + 1)[c]) + a.at(x2 + 1, y2 + 1)[c];
            o[c] = s * 0.25;
        }
        o
    })
}

/// cv2.remap INTER_LINEAR, BORDER_REPLICATE with float maps: exact float coordinates, the
/// lerps fused (t0 = fma(fx, b - a, a), ...), out-of-range taps clamped.
pub fn remap_linear<T: Copy + RemapPix>(src: &Grid<T>, mx: &Plane, my: &Plane) -> Grid<T> {
    Grid::from_fn(mx.w, mx.h, |x, y| {
        let (u, v) = (mx.at(x, y), my.at(x, y));
        let (sx, sy) = (u.floor(), v.floor());
        let (fx, fy) = (u - sx, v - sy);
        let (sx, sy) = (sx as i64, sy as i64);
        let x0 = replicate(sx, src.w);
        let x1 = replicate(sx + 1, src.w);
        let y0 = replicate(sy, src.h);
        let y1 = replicate(sy + 1, src.h);
        T::lerp2(src.at(x0, y0), src.at(x1, y0), src.at(x0, y1), src.at(x1, y1), fx, fy)
    })
}

pub trait RemapPix: Sized {
    fn lerp2(a: Self, b: Self, c: Self, d: Self, fx: f32, fy: f32) -> Self;
}

#[inline]
fn lerp2f(a: f32, b: f32, c: f32, d: f32, fx: f32, fy: f32) -> f32 {
    let t0 = fx.mul_add(b - a, a);
    let t1 = fx.mul_add(d - c, c);
    fy.mul_add(t1 - t0, t0)
}

impl RemapPix for f32 {
    fn lerp2(a: f32, b: f32, c: f32, d: f32, fx: f32, fy: f32) -> f32 {
        lerp2f(a, b, c, d, fx, fy)
    }
}

impl RemapPix for Rgb {
    fn lerp2(a: Rgb, b: Rgb, c: Rgb, d: Rgb, fx: f32, fy: f32) -> Rgb {
        [0, 1, 2].map(|k| lerp2f(a[k], b[k], c[k], d[k], fx, fy))
    }
}

impl RemapPix for [f32; 2] {
    fn lerp2(a: Self, b: Self, c: Self, d: Self, fx: f32, fy: f32) -> Self {
        [0, 1].map(|k| lerp2f(a[k], b[k], c[k], d[k], fx, fy))
    }
}

/// cv2.GaussianBlur(a, (0, 0), sigma) on float32 (separable, symmetric pairs summed first,
/// BORDER_REFLECT_101).
pub fn gaussian_blur(a: &Plane, sigma: f64) -> Plane {
    let n = (((sigma * 8.0 + 1.0).round_ties_even() as i64) | 1) as usize;
    let r = n / 2;
    let sc2 = -0.5 / (sigma * sigma);
    let mut k: Vec<f64> = (0..n).map(|i| {
        let x = i as f64 - r as f64;
        (sc2 * x * x).exp()
    }).collect();
    let s: f64 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    let kf: Vec<f32> = k.iter().map(|&v| v as f32).collect();
    let (w, h) = (a.w, a.h);
    let filt = |get: &dyn Fn(i64) -> f32| {
        let mut s = get(0) * kf[r];
        for j in 1..=r {
            s = (get(-(j as i64)) + get(j as i64)).mul_add(kf[r + j], s);
        }
        s
    };
    let rows = Grid::from_fn(w, h, |x, y| filt(&|d| a.at(reflect101(x as i64 + d, w), y)));
    Grid::from_fn(w, h, |x, y| filt(&|d| rows.at(x, reflect101(y as i64 + d, h))))
}

/// np.gradient along x and y (central inside, one-sided at the edges).
pub fn np_gradient(a: &Plane) -> (Plane, Plane) {
    let (w, h) = (a.w, a.h);
    let gx = Grid::from_fn(w, h, |x, y| {
        if w < 2 {
            0.0
        } else if x == 0 {
            a.at(1, y) - a.at(0, y)
        } else if x == w - 1 {
            a.at(w - 1, y) - a.at(w - 2, y)
        } else {
            (a.at(x + 1, y) - a.at(x - 1, y)) / 2.0
        }
    });
    let gy = Grid::from_fn(w, h, |x, y| {
        if h < 2 {
            0.0
        } else if y == 0 {
            a.at(x, 1) - a.at(x, 0)
        } else if y == h - 1 {
            a.at(x, h - 1) - a.at(x, h - 2)
        } else {
            (a.at(x, y + 1) - a.at(x, y - 1)) / 2.0
        }
    });
    (gx, gy)
}

/// Rectangular structuring element.
pub fn rect(w: usize, h: usize) -> Mask {
    Grid::new(w, h, true)
}

/// getStructuringElement(MORPH_ELLIPSE, (n, n)).
pub fn ellipse(n: usize) -> Mask {
    let r = (n / 2) as i64;
    let c = (n / 2) as i64;
    let inv_r2 = if r > 0 { 1.0 / ((r * r) as f64) } else { 0.0 };
    let mut m = Grid::new(n, n, false);
    for i in 0..n as i64 {
        let dy = i - r;
        if dy.abs() <= r {
            let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round_ties_even() as i64;
            let j1 = (c - dx).max(0);
            let j2 = (c + dx + 1).min(n as i64);
            for j in j1..j2 {
                m.set(j as usize, i as usize, true);
            }
        }
    }
    m
}

fn morph(a: &Mask, k: &Mask, dilate: bool) -> Mask {
    let (ax, ay) = ((k.w / 2) as i64, (k.h / 2) as i64);
    Grid::from_fn(a.w, a.h, |x, y| {
        for ky in 0..k.h {
            for kx in 0..k.w {
                if !k.at(kx, ky) {
                    continue;
                }
                let (xx, yy) = (x as i64 + kx as i64 - ax, y as i64 + ky as i64 - ay);
                if xx < 0 || yy < 0 || xx >= a.w as i64 || yy >= a.h as i64 {
                    continue;
                }
                if a.at(xx as usize, yy as usize) == dilate {
                    return dilate;
                }
            }
        }
        !dilate
    })
}

pub fn dilate(a: &Mask, k: &Mask) -> Mask {
    morph(a, k, true)
}

pub fn erode(a: &Mask, k: &Mask) -> Mask {
    morph(a, k, false)
}

pub fn dilate3(a: &Mask) -> Mask {
    dilate(a, &rect(3, 3))
}

pub fn close3(a: &Mask) -> Mask {
    erode(&dilate(a, &rect(3, 3)), &rect(3, 3))
}

/// cv2.connectedComponents label order: 4-connectivity numbers components by their first pixel in
/// raster order; 8-connectivity (block-based scan) by their first 2x2 block in raster order.
/// Returns (count including background, labels).
pub fn connected_components(a: &Mask, conn8: bool) -> (usize, Grid<i32>) {
    let (w, h) = (a.w, a.h);
    let mut lab = Grid::new(w, h, 0i32);
    let mut comps: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut seen = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !a.d[i] || seen[i] {
                continue;
            }
            let mut stack = vec![i];
            seen[i] = true;
            let mut px = Vec::new();
            while let Some(j) = stack.pop() {
                px.push(j);
                let (jx, jy) = ((j % w) as i64, (j / w) as i64);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if (dx == 0 && dy == 0) || (!conn8 && dx != 0 && dy != 0) {
                            continue;
                        }
                        let (nx, ny) = (jx + dx, jy + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let k = ny as usize * w + nx as usize;
                        if a.d[k] && !seen[k] {
                            seen[k] = true;
                            stack.push(k);
                        }
                    }
                }
            }
            let key = if conn8 {
                px.iter().map(|&j| ((j / w) / 2) * w.div_ceil(2) + (j % w) / 2).min().unwrap()
            } else {
                *px.iter().min().unwrap()
            };
            comps.push((key, px));
        }
    }
    comps.sort_by_key(|c| c.0);
    for (n, (_, px)) in comps.iter().enumerate() {
        for &j in px {
            lab.d[j] = n as i32 + 1;
        }
    }
    (comps.len() + 1, lab)
}

/// 3x3 sum with zero outside (cv2.filter2D / boxFilter BORDER_CONSTANT on a 0/1 map).
pub fn sum3_const(a: &Plane) -> Plane {
    box_filter(a, 3, false, Border::Constant)
}
