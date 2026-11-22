// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Residual-layer field: per 8 px tile, the motion of what the block field leaves unexplained
//! (HUD text, damage numbers, tickers). Evidence-weighted robust cost, coarse exhaustive search at
//! half res, full-res refine, explained-variance and aperture gates.

use crate::cv;
use crate::grid::{field_at, np_mean, np_std, Field, Grid, Image, Plane, FIELD_SCALE, FIELD_TEXEL_I};

const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

#[derive(Clone, Debug)]
/// Layer search and gate settings.
pub struct LayerParams {
    pub(crate) r: i32,
    pub(crate) win: usize,
    pub(crate) tu0: f64,
    pub(crate) tu1: f64,
    pub(crate) lam: f64,
    pub(crate) rho: f64,
    pub(crate) c_max: f64,
    pub(crate) m_min: f64,
    pub(crate) margin: i32,
    pub(crate) lam_min: f64,
    pub(crate) color: bool,
    pub(crate) ap_ratio: f64,
    pub(crate) ap_sup: i32,
    pub(crate) c_max_iso: f64,
}

impl LayerParams {
    /// Luma layer field (the dense path's FL).
    pub fn luma() -> Self {
        LayerParams { r: 32, win: 16, tu0: 0.04, tu1: 0.12, lam: 0.0004, rho: 0.5, c_max: 0.06, m_min: 8.0, margin: 1, lam_min: 0.005, color: false, ap_ratio: 0.0, ap_sup: 0, c_max_iso: 0.0 }
    }
    /// Colour cost, scale-free bracketed aperture, 5x5 group support, colour-scale cap for novel
    /// vectors (the block path's FL).
    pub fn colour() -> Self {
        LayerParams { color: true, ap_ratio: 0.4, ap_sup: 3, c_max_iso: 0.09, ..Self::luma() }
    }
}

/// Tile layer vectors (frame px, NaN = none) and costs (inf = none).
#[derive(Clone, Debug)]
pub struct LayerField {
    /// Tile vectors.
    pub v: Grid<[f32; 2]>,
    /// Tile costs.
    pub z: Plane,
    /// Tile size, frame px.
    pub tile: usize,
    /// Splat margin, frame px.
    pub margin: i32,
}

/// Multi-channel float image (1 = luma, 3 = RGB) with the reference's channel-mean residual.
#[derive(Clone)]
struct Mc {
    w: usize,
    h: usize,
    cn: usize,
    d: Vec<f32>,
}

impl Mc {
    fn from_luma(p: &Plane) -> Mc {
        Mc { w: p.w, h: p.h, cn: 1, d: p.d.clone() }
    }
    fn from_rgb(p: &Image) -> Mc {
        Mc { w: p.w, h: p.h, cn: 3, d: p.d.iter().flat_map(|v| *v).collect() }
    }
    #[inline]
    fn px(&self, x: usize, y: usize) -> &[f32] {
        let i = (y * self.w + x) * self.cn;
        &self.d[i..i + self.cn]
    }
    #[inline]
    fn pxc(&self, x: i64, y: i64) -> &[f32] {
        self.px(x.clamp(0, self.w as i64 - 1) as usize, y.clamp(0, self.h as i64 - 1) as usize)
    }
    fn resize_area2(&self) -> Mc {
        if self.cn == 1 {
            Mc::from_luma(&cv::resize_area2(&Grid::from_vec(self.w, self.h, self.d.clone())))
        } else {
            let g = Grid::from_fn(self.w, self.h, |x, y| {
                let p = self.px(x, y);
                [p[0], p[1], p[2]]
            });
            Mc::from_rgb(&cv::resize_area2_rgb(&g))
        }
    }
}

/// Mean over channels of |a - b|.
#[inline]
fn ad(a: &[f32], b: &[f32]) -> f32 {
    if a.len() == 1 {
        (a[0] - b[0]).abs()
    } else {
        ((0.0 + (a[0] - b[0]).abs()) + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()) / 3.0
    }
}

#[inline]
fn sstep(a: f32, ba: f32, x: f32) -> f32 {
    let u = ((x - a) / ba).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

fn luma(img: &Image) -> Plane {
    img.map(|c| (c[0] * LUMA[0] + c[1] * LUMA[1]) + c[2] * LUMA[2])
}

/// Unnormalised n x n replicate-border box sum of `f(x, y)` at one point (OpenCV anchor n / 2).
fn box_at(w: usize, h: usize, cx: usize, cy: usize, n: usize, f: &dyn Fn(usize, usize) -> f32) -> f32 {
    let lo = (n / 2) as i64;
    let mut s = 0.0f64;
    for ky in 0..n as i64 {
        let y = (cy as i64 + ky - lo).clamp(0, h as i64 - 1) as usize;
        let mut r = 0.0f64;
        for kx in 0..n as i64 {
            let x = (cx as i64 + kx - lo).clamp(0, w as i64 - 1) as usize;
            r += f(x, y) as f64;
        }
        s += r;
    }
    s as f32
}

/// No layers.
pub fn empty(w: usize, h: usize, tile: usize, margin: i32) -> LayerField {
    let (th, tw) = (h.div_ceil(tile), w.div_ceil(tile));
    LayerField { v: Grid::new(tw, th, [f32::NAN; 2]), z: Grid::new(tw, th, f32::INFINITY), tile, margin }
}

/// Layer field of cur against prev given the block field.
pub fn layer_field(cur: &Image, prev: &Image, f: &Field, cut_r: f32, p: &LayerParams) -> LayerField {
    let (w, h) = (cur.w, cur.h);
    let tile = 8usize;
    if cut_r > 1.3 {
        return empty(w, h, tile, p.margin);
    }
    let yc = luma(cur);
    let mut yp = luma(prev);
    let mut prev = prev.clone();
    {
        let g = (np_std(&yc.d) / np_std(&yp.d).max(1e-4)) as f64;
        let o = (np_mean(&yc.d) - (g as f32) * np_mean(&yp.d)) as f64;
        let (gf, of) = (g as f32, o as f32);
        let res: Vec<f32> = yc.d.iter().zip(&yp.d).map(|(&a, &b)| (a - (gf * b + of)).abs()).collect();
        let raw: Vec<f32> = yc.d.iter().zip(&yp.d).map(|(&a, &b)| (a - b).abs()).collect();
        if np_mean(&res) <= 0.7 * np_mean(&raw) {
            yp = yp.map(|v| gf * v + of);
            prev = prev.map(|c| c.map(|v| (gf * v + of).clamp(0.0, 1.0)));
        }
    }
    let (ycm, ypm) = if p.color { (Mc::from_rgb(cur), Mc::from_rgb(&prev)) } else { (Mc::from_luma(&yc), Mc::from_luma(&yp)) };
    let fb = Grid::from_fn(w, h, |x, y| {
        let v = field_at(f, x as f32, y as f32);
        [v[0], v[1]]
    });
    let mx = Grid::from_fn(w, h, |x, y| x as f32 + fb.at(x, y)[0]);
    let my = Grid::from_fn(w, h, |x, y| y as f32 + fb.at(x, y)[1]);
    let pw = if p.color {
        let g = Grid::from_fn(w, h, |x, y| {
            let v = ypm.px(x, y);
            [v[0], v[1], v[2]]
        });
        Mc::from_rgb(&cv::remap_linear(&g, &mx, &my))
    } else {
        Mc::from_luma(&cv::remap_linear(&yp, &mx, &my))
    };
    let u = Grid::from_fn(w, h, |x, y| ad(ycm.px(x, y), pw.px(x, y)));
    let wgt = u.map(|v| sstep(p.tu0 as f32, (p.tu1 - p.tu0) as f32, v));
    let win = p.win;
    let (th, tw) = (h.div_ceil(tile), w.div_ceil(tile));
    let cyv: Vec<usize> = (0..th).map(|i| (i * tile + tile / 2).min(h - 1)).collect();
    let cxv: Vec<usize> = (0..tw).map(|i| (i * tile + tile / 2).min(w - 1)).collect();
    let wu = wgt.zip(&u, |a, b| a * b);
    let mass = Grid::from_fn(tw, th, |tx, ty| box_at(w, h, cxv[tx], cyv[ty], win, &|x, y| wgt.at(x, y)));
    let cp = Grid::from_fn(tw, th, |tx, ty| box_at(w, h, cxv[tx], cyv[ty], win, &|x, y| wu.at(x, y)) / mass.at(tx, ty).max(1e-6));
    let b_t = Grid::from_fn(tw, th, |tx, ty| fb.at(cxv[tx], cyv[ty]));

    // coarse exhaustive search at half resolution (absolute vectors)
    let s = 2usize;
    let (hs, ws) = (h.div_ceil(s), w.div_ceil(s));
    let ycs = ycm.resize_area2();
    let yps = ypm.resize_area2();
    let wgs = cv::resize_area2(&wgt);
    let rs = p.r / s as i32;
    let ws_ = (win / s).max(3);
    let gys: Vec<usize> = cyv.iter().map(|&c| (c / s).min(hs - 1)).collect();
    let gxs: Vec<usize> = cxv.iter().map(|&c| (c / s).min(ws - 1)).collect();
    let msk_t = Grid::from_fn(tw, th, |tx, ty| box_at(ws, hs, gxs[tx], gys[ty], ws_, &|x, y| wgs.at(x, y)).max(1e-6));
    let lam = p.lam as f32;
    let lam_s = (p.lam * s as f64) as f32;
    let mut best = Grid::new(tw, th, f32::INFINITY);
    let mut bv = Grid::new(tw, th, [0.0f32; 2]);
    for dy in -rs..=rs {
        for dx in -rs..=rs {
            for ty in 0..th {
                for tx in 0..tw {
                    let d = |x: usize, y: usize| wgs.at(x, y) * ad(ycs.px(x, y), yps.pxc(x as i64 + dx as i64, y as i64 + dy as i64));
                    let c = box_at(ws, hs, gxs[tx], gys[ty], ws_, &d) / msk_t.at(tx, ty);
                    let bt = b_t.at(tx, ty);
                    let (bt0, bt1) = (bt[0] / s as f32, bt[1] / s as f32);
                    let c = c + lam_s * ((dx as f32 - bt0).abs() + (dy as f32 - bt1).abs()) * s as f32;
                    if c < best.at(tx, ty) {
                        best.set(tx, ty, c);
                        bv.set(tx, ty, [(dx * s as i32) as f32, (dy * s as i32) as f32]);
                    }
                }
            }
        }
    }
    // full-res +-s refine around the coarse vector
    let okc = Grid::from_fn(tw, th, |x, y| {
        mass.at(x, y) >= p.m_min as f32 && best.at(x, y) < (1.5 * p.rho) as f32 * cp.at(x, y) && best.at(x, y) < (1.5 * p.c_max) as f32
    });
    let mut vv = bv.clone();
    let mut costf = Grid::new(tw, th, f32::INFINITY);
    for ty in 0..th {
        for tx in 0..tw {
            if !okc.at(tx, ty) {
                continue;
            }
            let b = bv.at(tx, ty);
            let bt = b_t.at(tx, ty);
            let mf = mass.at(tx, ty).max(1e-6);
            for oy in -(s as i32)..=s as i32 {
                for ox in -(s as i32)..=s as i32 {
                    let (vx, vy) = (b[0] + ox as f32, b[1] + oy as f32);
                    let (ix, iy) = (vx as i64, vy as i64);
                    let d = |x: usize, y: usize| wgt.at(x, y) * ad(ycm.px(x, y), ypm.pxc(x as i64 + ix, y as i64 + iy));
                    let c = box_at(w, h, cxv[tx], cyv[ty], win, &d) / mf;
                    let c = c + lam * ((vx - bt[0]).abs() + (vy - bt[1]).abs());
                    if c < costf.at(tx, ty) {
                        costf.set(tx, ty, c);
                        vv.set(tx, ty, [vx, vy]);
                    }
                }
            }
        }
    }
    let best = costf;
    // aperture: smallest eigenvalue of the evidence-weighted structure tensor of cur luma
    let (gxl, gyl) = cv::np_gradient(&cv::gaussian_blur(&yc, 0.8));
    let jt = |a: &Plane| Grid::from_fn(tw, th, |tx, ty| box_at(w, h, cxv[tx], cyv[ty], win, &|x, y| wgt.at(x, y) * a.at(x, y)) / mass.at(tx, ty).max(1e-6));
    let jxx = jt(&gxl.zip(&gxl, |a, b| a * b));
    let jxy = jt(&gxl.zip(&gyl, |a, b| a * b));
    let jyy = jt(&gyl.zip(&gyl, |a, b| a * b));
    let lmin = Grid::from_fn(tw, th, |x, y| {
        let (a, b, c) = (jxx.at(x, y), jxy.at(x, y), jyy.at(x, y));
        0.5 * (a + c - ((a - c) * (a - c) + 4.0 * (b * b)).sqrt())
    });
    let lmax = Grid::from_fn(tw, th, |x, y| {
        let (a, b, c) = (jxx.at(x, y), jxy.at(x, y), jyy.at(x, y));
        0.5 * (a + c + ((a - c) * (a - c) + 4.0 * (b * b)).sqrt())
    });
    let ap_ok = lmin.map(|l| l >= p.lam_min as f32);
    let (m_min, rho, c_max) = (p.m_min as f32, p.rho as f32, p.c_max as f32);
    let base_ok = |x: usize, y: usize| mass.at(x, y) >= m_min && best.at(x, y) < rho * cp.at(x, y) && best.at(x, y) < c_max;
    let mut iso = Grid::new(tw, th, false);
    if p.ap_ratio > 0.0 {
        let lim = (rs * s as i32) as f32;
        let c_iso = (if p.c_max_iso > 0.0 { p.c_max_iso } else { p.c_max }) as f32;
        let ap_ratio = p.ap_ratio as f32;
        iso = Grid::from_fn(tw, th, |x, y| {
            let b = bv.at(x, y);
            lmin.at(x, y) >= ap_ratio * lmax.at(x, y)
                && b[0].abs() < lim
                && b[1].abs() < lim
                && !ap_ok.at(x, y)
                && mass.at(x, y) >= m_min
                && best.at(x, y) < rho * cp.at(x, y)
                && best.at(x, y) < c_iso
        });
        if c_iso > c_max {
            let (fh, fw) = (f.h as i64, f.w as i64);
            iso = Grid::from_fn(tw, th, |x, y| {
                if !iso.at(x, y) {
                    return false;
                }
                let v = vv.at(x, y);
                let (fy, fx) = ((cyv[y] / FIELD_TEXEL_I) as i64, (cxv[x] / FIELD_TEXEL_I) as i64);
                let (fy, fx) = (fy.min(fh - 1), fx.min(fw - 1));
                let mut dup = false;
                for ddy in -1..=1 {
                    for ddx in -1..=1 {
                        let t = f.atc(fx + ddx, fy + ddy);
                        let (a, b) = (t[0] * FIELD_SCALE, t[1] * FIELD_SCALE);
                        if (a - v[0]).abs().max((b - v[1]).abs()) <= 3.0 {
                            dup = true;
                        }
                    }
                }
                best.at(x, y) < c_max || !dup
            });
        }
        if p.ap_sup > 0 {
            let pre = Grid::from_fn(tw, th, |x, y| (base_ok(x, y) && ap_ok.at(x, y)) || iso.at(x, y));
            iso = Grid::from_fn(tw, th, |x, y| {
                if !iso.at(x, y) {
                    return false;
                }
                let v = vv.at(x, y);
                let mut sup = 0;
                for dy in -2i64..=2 {
                    for dx in -2i64..=2 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        if nx < 0 || ny < 0 || nx >= tw as i64 || ny >= th as i64 {
                            continue;
                        }
                        let (nx, ny) = (nx as usize, ny as usize);
                        if !pre.at(nx, ny) {
                            continue;
                        }
                        let u = vv.at(nx, ny);
                        if (u[0] - v[0]).abs().max((u[1] - v[1]).abs()) <= 1.0 {
                            sup += 1;
                        }
                    }
                }
                sup >= p.ap_sup
            });
        }
    }
    let ok0 = Grid::from_fn(tw, th, |x, y| (base_ok(x, y) && ap_ok.at(x, y)) || iso.at(x, y));
    // isolated valid tiles are aliasing: need a valid 8-neighbour (wrapping, as np.roll) within 2.5 px
    let ok = Grid::from_fn(tw, th, |x, y| {
        if !ok0.at(x, y) {
            return false;
        }
        let v = vv.at(x, y);
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let sx = (x as i64 - dx).rem_euclid(tw as i64) as usize;
                let sy = (y as i64 - dy).rem_euclid(th as i64) as usize;
                if !ok0.at(sx, sy) {
                    continue;
                }
                let u = vv.at(sx, sy);
                if (u[0] - v[0]).abs().max((u[1] - v[1]).abs()) <= 2.5 {
                    return true;
                }
            }
        }
        false
    });
    LayerField {
        v: Grid::from_fn(tw, th, |x, y| if ok.at(x, y) { vv.at(x, y) } else { [f32::NAN; 2] }),
        z: Grid::from_fn(tw, th, |x, y| if ok.at(x, y) { best.at(x, y) } else { f32::INFINITY }),
        tile,
        margin: p.margin,
    }
}
