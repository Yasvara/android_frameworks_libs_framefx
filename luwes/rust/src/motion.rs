// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Hierarchical block matcher: 4-level luma pyramid, exhaustive search at the top, candidate /
//! propagation / local / wide passes down to level 1, wide-range rescue as a second vector,
//! level-0 parabola + Lucas-Kanade, cost-checked median and smoothing. Output field: one texel per
//! 32 frame px of the 128-aligned frame (level-1 8x8 blocks).

use crate::fft;
use crate::grid::{np_mean, np_sum, Field, Grid, Plane, Rgb, V2};

pub(crate) const LEVELS: usize = 4;
const ALIGN: usize = 128;
const BS: usize = 8;
const MARGIN: usize = 192;
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

#[derive(Clone, Debug)]
/// Matcher thresholds and search radii.
pub struct MatchParams {
    pub(crate) lam: f32,
    pub(crate) radius: i32,
    pub(crate) local_r: i32,
    pub(crate) local_r1: i32,
    pub(crate) kappa: f32,
    pub(crate) gain_gate: f32,
    pub(crate) kappa_s: f32,
    pub(crate) tau_s: f32,
    pub(crate) lk_iters: usize,
    pub(crate) lk_lambda: f32,
    pub(crate) lk_maxd: f32,
    pub(crate) rescue_r: i32,
    pub(crate) rescue_thr: f32,
    pub(crate) rescue_ratio: f32,
    pub(crate) rescue_abs: f32,
    pub(crate) rescue_dq: f32,
    pub(crate) pc_min: f64,
    pub(crate) wide_r1: i32,
    pub(crate) cut_lo: f32,
}

impl Default for MatchParams {
    fn default() -> Self {
        MatchParams {
            lam: 0.004,
            radius: 12,
            local_r: 4,
            local_r1: 8,
            kappa: 0.01,
            gain_gate: 0.7,
            kappa_s: 0.1,
            tau_s: 0.01,
            lk_iters: 2,
            lk_lambda: 0.01,
            lk_maxd: 0.25,
            rescue_r: 24,
            rescue_thr: 0.02,
            rescue_ratio: 0.5,
            rescue_abs: 0.02,
            rescue_dq: 0.5,
            pc_min: 10.0,
            wide_r1: 12,
            cut_lo: 1.3,
        }
    }
}

#[derive(Clone, Debug)]
/// Per-pair side outputs of the matcher.
pub struct MatchMeta {
    /// Median unexplained ratio z / texture on level-1 blocks (scene-cut statistic).
    pub cut_r: f32,
    /// Accepted prev -> cur luma gain (g, o); (1, 0) when the gate rejected it.
    pub gain: (f64, f64),
    /// Blocks that took a rescue vector.
    pub n_rescue: usize,
}

impl MatchMeta {
    /// The gate let a real gain through.
    pub fn gain_accepted(&self) -> bool {
        (self.gain.0 - 1.0).abs() > 1e-3 || self.gain.1.abs() > 1e-3
    }
}

/// Matcher output.
pub struct Matched {
    /// Block field (level-1 8x8 blocks, one texel per 32 frame px).
    pub f: Field,
    /// Rescue vectors (z = inf where none).
    pub rescue: Field,
    /// Side outputs.
    pub meta: MatchMeta,
}

#[inline]
fn q8_32(a: f32) -> f32 {
    (a.clamp(0.0, 1.0) * 255.0).round_ties_even() / 255.0
}

/// Luma levels 0..3 of the 128-aligned frame (level 0 = 2x2 taps of the frame, clamp-to-edge).
pub(crate) fn pyramid(rgb: &Grid<Rgb>) -> Vec<Plane> {
    let (w, h) = (rgb.w, rgb.h);
    let ha = h.div_ceil(ALIGN) * ALIGN;
    let wa = w.div_ceil(ALIGN) * ALIGN;
    let l0 = Grid::from_fn(wa / 2, ha / 2, |x, y| {
        let (x0, x1) = ((2 * x).min(w - 1), (2 * x + 1).min(w - 1));
        let (y0, y1) = ((2 * y).min(h - 1), (2 * y + 1).min(h - 1));
        let (a, b, c, d) = (rgb.at(x0, y0), rgb.at(x1, y0), rgb.at(x0, y1), rgb.at(x1, y1));
        let mut m = [0.0f32; 3];
        for k in 0..3 {
            m[k] = 0.25 * (((a[k] + b[k]) + c[k]) + d[k]);
        }
        q8_32((m[0] * LUMA[0] + m[1] * LUMA[1]) + m[2] * LUMA[2])
    });
    let mut lv = vec![l0];
    for _ in 1..LEVELS {
        let a = lv.last().unwrap();
        let n = Grid::from_fn(a.w / 2, a.h / 2, |x, y| {
            let (x2, y2) = (2 * x, 2 * y);
            q8_32(0.25 * (((a.at(x2, y2) + a.at(x2, y2 + 1)) + a.at(x2 + 1, y2)) + a.at(x2 + 1, y2 + 1)))
        });
        lv.push(n);
    }
    lv
}

/// One pyramid level: cur blocks against edge-padded prev, masked to the real frame.
pub(crate) struct Level {
    pub(crate) bs: usize,
    pub(crate) nbx: usize,
    pub(crate) nby: usize,
    cur: Plane,
    prev: Vec<f32>,
    pw: usize,
    rw: usize,
    rh: usize,
    /// 1 / (real pixels of the block), 0 for a block outside the frame.
    inv: Vec<f32>,
}

impl Level {
    pub(crate) fn new(cur: &Plane, prev: &Plane, bs: usize, real_px: (usize, usize)) -> Level {
        let (w, h) = (cur.w, cur.h);
        let (nbx, nby) = (w / bs, h / bs);
        let pw = w + 2 * MARGIN;
        let ph = h + 2 * MARGIN;
        let mut pad = vec![0.0f32; pw * ph];
        for y in 0..ph {
            let sy = (y as i64 - MARGIN as i64).clamp(0, h as i64 - 1) as usize;
            for x in 0..pw {
                let sx = (x as i64 - MARGIN as i64).clamp(0, w as i64 - 1) as usize;
                pad[y * pw + x] = prev.at(sx, sy);
            }
        }
        let (rh, rw) = real_px;
        let mut inv = vec![0.0f32; nbx * nby];
        for by in 0..nby {
            for bx in 0..nbx {
                let cy = (rh as i64 - (by * bs) as i64).clamp(0, bs as i64) as usize;
                let cx = (rw as i64 - (bx * bs) as i64).clamp(0, bs as i64) as usize;
                let cnt = (cx * cy) as f32;
                inv[by * nbx + bx] = 1.0 / cnt.max(1e-6);
            }
        }
        Level { bs, nbx, nby, cur: cur.clone(), prev: pad, pw, rw, rh, inv }
    }

    #[inline]
    fn wm(&self, bx: usize, by: usize, x: usize, y: usize) -> f32 {
        let (gx, gy) = (bx * self.bs + x, by * self.bs + y);
        if gx < self.rw && gy < self.rh {
            self.inv[by * self.nbx + bx]
        } else {
            0.0
        }
    }

    #[inline]
    fn setup(&self, v: V2) -> (usize, usize, f32, f32) {
        let m = (MARGIN - self.bs - 2) as f32;
        let vx = v[0].clamp(-m, m);
        let vy = v[1].clamp(-m, m);
        let (fx, fy) = (vx.floor(), vy.floor());
        (
            (MARGIN as i64 + fx as i64) as usize,
            (MARGIN as i64 + fy as i64) as usize,
            vx - fx,
            vy - fy,
        )
    }

    #[inline]
    fn tap(&self, ox: usize, oy: usize, ax: f32, ay: f32, x: usize, y: usize) -> f32 {
        let i = (oy + y) * self.pw + ox + x;
        let p = &self.prev;
        (p[i] * (1.0 - ax) + p[i + 1] * ax) * (1.0 - ay) + (p[i + self.pw] * (1.0 - ax) + p[i + self.pw + 1] * ax) * ay
    }

    /// Masked mean |cur - prev(x + v)| of block (bx, by), bilinear prev.
    pub(crate) fn sad1(&self, bx: usize, by: usize, v: V2) -> f32 {
        let (ox, oy, ax, ay) = self.setup(v);
        let (ox, oy) = (ox + bx * self.bs, oy + by * self.bs);
        let n = self.bs * self.bs;
        let mut d = [0.0f32; 256];
        for y in 0..self.bs {
            for x in 0..self.bs {
                let t = self.tap(ox, oy, ax, ay, x, y);
                let c = self.cur.at(bx * self.bs + x, by * self.bs + y);
                d[y * self.bs + x] = (t - c).abs() * self.wm(bx, by, x, y);
            }
        }
        np_sum(&d[..n])
    }

    pub(crate) fn sad(&self, v: &Grid<V2>) -> Plane {
        Grid::from_fn(self.nbx, self.nby, |bx, by| self.sad1(bx, by, v.at(bx, by)))
    }

    pub(crate) fn sad_const(&self, v: V2) -> Plane {
        Grid::from_fn(self.nbx, self.nby, |bx, by| self.sad1(bx, by, v))
    }

    fn sample1(&self, bx: usize, by: usize, v: V2, out: &mut [f32]) {
        let (ox, oy, ax, ay) = self.setup(v);
        let (ox, oy) = (ox + bx * self.bs, oy + by * self.bs);
        for y in 0..self.bs {
            for x in 0..self.bs {
                out[y * self.bs + x] = self.tap(ox, oy, ax, ay, x, y);
            }
        }
    }

    fn origin(&self, bx: usize, by: usize) -> (f32, f32) {
        ((bx * self.bs) as f32, (by * self.bs) as f32)
    }
}

fn nb(i: usize) -> (i64, i64) {
    ((i % 3) as i64 - 1, (i / 3) as i64 - 1)
}

fn around(i: usize) -> (i64, i64) {
    nb(i + if i >= 4 { 1 } else { 0 })
}

pub(crate) fn shifted(f: &Grid<V2>, dx: i64, dy: i64, real: (usize, usize)) -> Grid<V2> {
    let (rh, rw) = real;
    Grid::from_fn(f.w, f.h, |x, y| {
        let xi = (x as i64 + dx).clamp(0, rw as i64 - 1) as usize;
        let yi = (y as i64 + dy).clamp(0, rh as i64 - 1) as usize;
        f.at(xi, yi)
    })
}

fn parent_at(parent: &Grid<V2>, w: usize, h: usize, dx: i64, dy: i64) -> Grid<V2> {
    Grid::from_fn(w, h, |x, y| {
        let xi = ((x as i64 + 2 * dx) / 2).clamp(0, parent.w as i64 - 1) as usize;
        let yi = ((y as i64 + 2 * dy) / 2).clamp(0, parent.h as i64 - 1) as usize;
        let v = parent.at(xi, yi);
        [v[0] * 2.0, v[1] * 2.0]
    })
}

fn vmed_lanes(n: &[V2; 9]) -> V2 {
    let mut best = 0;
    let mut bc = f32::INFINITY;
    for a in 0..9 {
        let mut c = 0.0f32;
        for b in 0..9 {
            c += (n[a][0] - n[b][0]).abs();
            c += (n[a][1] - n[b][1]).abs();
        }
        if c < bc {
            bc = c;
            best = a;
        }
    }
    n[best]
}

fn vmed9(lanes: &[Grid<V2>]) -> Grid<V2> {
    let (w, h) = (lanes[0].w, lanes[0].h);
    Grid::from_fn(w, h, |x, y| {
        let n: [V2; 9] = std::array::from_fn(|l| lanes[l].at(x, y));
        vmed_lanes(&n)
    })
}

fn own9(f: &Grid<V2>, real: (usize, usize)) -> Vec<Grid<V2>> {
    (0..9).map(|l| {
        let (dx, dy) = nb(l);
        shifted(f, dx, dy, real)
    }).collect()
}

fn parent9(parent: &Grid<V2>, w: usize, h: usize) -> Vec<Grid<V2>> {
    (0..9).map(|l| {
        let (dx, dy) = nb(l);
        parent_at(parent, w, h, dx, dy)
    }).collect()
}

fn round2(v: V2) -> V2 {
    [v[0].round_ties_even(), v[1].round_ties_even()]
}

pub(crate) fn copy_pad<T: Copy>(f: &Grid<T>, real: (usize, usize)) -> Grid<T> {
    let (rh, rw) = real;
    Grid::from_fn(f.w, f.h, |x, y| f.at(x.min(rw - 1), y.min(rh - 1)))
}

/// Lowest cost wins, the earliest candidate on a tie; cost = SAD + lam |v - pred|_1.
fn consider_all(l: &Level, cands: &[Grid<V2>], pred: &Grid<V2>, lam: f32) -> Grid<V2> {
    Grid::from_fn(l.nbx, l.nby, |bx, by| {
        let p = pred.at(bx, by);
        let mut best = f32::INFINITY;
        let mut bv = [0.0f32; 2];
        for c in cands {
            let v = c.at(bx, by);
            let e = l.sad1(bx, by, v);
            let cst = e + lam * ((v[0] - p[0]).abs() + (v[1] - p[1]).abs());
            if cst < best {
                best = cst;
                bv = v;
            }
        }
        bv
    })
}

fn consider_offsets(l: &Level, base: &Grid<V2>, offs: &[V2], pred: &Grid<V2>, lam: f32, first: Option<&Grid<V2>>) -> Grid<V2> {
    Grid::from_fn(l.nbx, l.nby, |bx, by| {
        let p = pred.at(bx, by);
        let b = base.at(bx, by);
        let mut best = f32::INFINITY;
        let mut bv = [0.0f32; 2];
        let mut try_v = |v: V2| {
            let e = l.sad1(bx, by, v);
            let cst = e + lam * ((v[0] - p[0]).abs() + (v[1] - p[1]).abs());
            if cst < best {
                best = cst;
                bv = v;
            }
        };
        if let Some(f) = first {
            try_v(f.at(bx, by));
        }
        for o in offs {
            try_v([b[0] + o[0], b[1] + o[1]]);
        }
        bv
    })
}

fn dominant(top: &Grid<V2>, radius: i32) -> V2 {
    let v: Vec<(i64, i64)> = top.d.iter().map(|a| (a[0].round_ties_even() as i64, a[1].round_ties_even() as i64)).collect();
    let n = v.len();
    let mut best: i64 = 0;
    for i in 0..n {
        if (v[i].0 == 0 && v[i].1 == 0) || v[i].0.abs() > radius as i64 || v[i].1.abs() > radius as i64 {
            continue;
        }
        let votes = v.iter().filter(|&&u| u == v[i]).count() as i64;
        best = best.max(votes << 16 | i as i64);
    }
    if (best >> 16) * 10 >= n as i64 && best != 0 {
        let u = v[(best & 0xffff) as usize];
        [u.0 as f32, u.1 as f32]
    } else {
        [0.0, 0.0]
    }
}

/// Whole-frame translation by phase correlation (Hann window, whitened spectrum): (back vector,
/// peak / rms of the correlation surface).
fn phase_corr(a: &Plane, b: &Plane) -> (V2, f64) {
    let (w, h) = (a.w, a.h);
    let han = |n: usize| -> Vec<f64> {
        if n == 1 {
            return vec![1.0];
        }
        (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()).collect()
    };
    let (hy, hx) = (han(h), han(w));
    let ma = np_mean(&a.d);
    let mb = np_mean(&b.d);
    let mut fa: Vec<fft::C> = Vec::with_capacity(w * h);
    let mut fb: Vec<fft::C> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let wv = hy[y] * hx[x];
            fa.push(fft::C::new(((a.at(x, y) - ma) as f64) * wv, 0.0));
            fb.push(fft::C::new(((b.at(x, y) - mb) as f64) * wv, 0.0));
        }
    }
    fft::fft2(&mut fa, w, h, false);
    fft::fft2(&mut fb, w, h, false);
    let mut r: Vec<fft::C> = fa.iter().zip(&fb).map(|(&pa, &pb)| {
        let c = pb * pa.conj();
        c / (c.norm() + 1e-9)
    }).collect();
    fft::fft2(&mut r, w, h, true);
    let re: Vec<f64> = r.iter().map(|c| c.re / (w * h) as f64).collect();
    let mut bi = 0;
    for i in 1..re.len() {
        if re[i] > re[bi] {
            bi = i;
        }
    }
    let (iy, ix) = (bi / w, bi % w);
    let dy = if iy <= h / 2 { iy as i64 } else { iy as i64 - h as i64 };
    let dx = if ix <= w / 2 { ix as i64 } else { ix as i64 - w as i64 };
    let rms = (re.iter().map(|v| v * v).sum::<f64>() / re.len() as f64).sqrt();
    ([dx as f32, dy as f32], re[bi] / (rms + 1e-9))
}

/// Median over level-1 blocks of z / max(texture, 2/255), texture = mean |L - box5(L)| of cur.
pub(crate) fn unexplained_z(pc1: &Plane, z: &Plane, h_frame: usize, w_frame: usize) -> f32 {
    let bl = crate::cv::blur(pc1, 5);
    let (hh, ww) = (h_frame.div_ceil(2), w_frame.div_ceil(2));
    let (rh, rw) = ((hh / 8).min(z.h), (ww / 8).min(z.w));
    let mut v = Vec::with_capacity(rh * rw);
    for by in 0..rh {
        for bx in 0..rw {
            let mut hp = [0.0f32; 64];
            for y in 0..8 {
                for x in 0..8 {
                    let (gx, gy) = (bx * 8 + x, by * 8 + y);
                    hp[y * 8 + x] = (pc1.at(gx, gy) - bl.at(gx, gy)).abs();
                }
            }
            let t = rows_seq_mean(&hp, 8);
            v.push(z.at(bx, by) / t.max(2.0 / 255.0));
        }
    }
    crate::grid::median(&mut v)
}

/// Mean of a block reduced row by row (numpy over a strided view): sum of per-row pairwise sums.
fn rows_seq_mean(a: &[f32], bs: usize) -> f32 {
    let mut s = 0.0f32;
    for r in a.chunks(bs) {
        s += np_sum(r);
    }
    s / a.len() as f32
}

fn rows_seq_sum(a: &[f32], bs: usize) -> f32 {
    let mut s = 0.0f32;
    for r in a.chunks(bs) {
        s += np_sum(r);
    }
    s
}

fn subpel_level0(c0: &Plane, p0: &Plane, f: &Grid<V2>, real0: (usize, usize)) -> Grid<V2> {
    let l0 = Level::new(c0, p0, 2 * BS, real0);
    Grid::from_fn(f.w, f.h, |bx, by| {
        let v = f.at(bx, by);
        let base = [(2.0 * v[0]).round_ties_even(), (2.0 * v[1]).round_ties_even()];
        let mut best = f32::INFINITY;
        let mut bv = base;
        for i in 0..9 {
            let (dx, dy) = nb(i);
            let c = [base[0] + dx as f32, base[1] + dy as f32];
            let e = l0.sad1(bx, by, c);
            if e < best {
                best = e;
                bv = c;
            }
        }
        let e0 = l0.sad1(bx, by, bv);
        let mut off = [0.0f32; 2];
        for ax in 0..2 {
            let mut m = bv;
            let mut p = bv;
            m[ax] -= 1.0;
            p[ax] += 1.0;
            let (e_minus, e_plus) = (l0.sad1(bx, by, m), l0.sad1(bx, by, p));
            let den = 2.0 * (e_minus - 2.0 * e0 + e_plus);
            let o = if den > 1e-6 { (e_minus - e_plus) / den.max(1e-6) } else { 0.0 };
            off[ax] = o.clamp(-0.5, 0.5);
        }
        [(bv[0] + off[0]) / 2.0, (bv[1] + off[1]) / 2.0]
    })
}

fn lk_level0(c0: &Plane, p0: &Plane, f: &Grid<V2>, real0: (usize, usize), iters: usize, lam: f32, max_d: f32) -> Grid<V2> {
    let l0 = Level::new(c0, p0, 2 * BS, real0);
    let (gx, gy) = crate::cv::np_gradient(c0);
    let bs = l0.bs;
    let n = bs * bs;
    Grid::from_fn(f.w, f.h, |bx, by| {
        let mut wg = vec![0.0f32; n];
        let mut gxb = vec![0.0f32; n];
        let mut gyb = vec![0.0f32; n];
        for y in 0..bs {
            for x in 0..bs {
                let i = y * bs + x;
                wg[i] = l0.wm(bx, by, x, y);
                gxb[i] = gx.at(bx * bs + x, by * bs + y);
                gyb[i] = gy.at(bx * bs + x, by * bs + y);
            }
        }
        let prod = |a: &[f32], b: &[f32]| -> Vec<f32> { (0..n).map(|i| wg[i] * a[i] * b[i]).collect() };
        let hxx = rows_seq_sum(&prod(&gxb, &gxb), bs);
        let hyy = rows_seq_sum(&prod(&gyb, &gyb), bs);
        let hxy = rows_seq_sum(&prod(&gxb, &gyb), bs);
        let damp = lam * 0.5 * (hxx + hyy);
        let (a, c, b) = (hxx + damp, hyy + damp, hxy);
        let det = a * c - b * b;
        let ok_h = det > 1e-12;
        let fv = f.at(bx, by);
        let u0 = [2.0 * fv[0], 2.0 * fv[1]];
        let mut u = u0;
        let mut s_old = l0.sad1(bx, by, u);
        let mut smp = vec![0.0f32; n];
        for _ in 0..iters {
            l0.sample1(bx, by, u, &mut smp);
            let mut rxv = vec![0.0f32; n];
            let mut ryv = vec![0.0f32; n];
            for y in 0..bs {
                for x in 0..bs {
                    let i = y * bs + x;
                    let r = l0.cur.at(bx * bs + x, by * bs + y) - smp[i];
                    rxv[i] = wg[i] * gxb[i] * r;
                    ryv[i] = wg[i] * gyb[i] * r;
                }
            }
            let rx = np_sum(&rxv);
            let ry = np_sum(&ryv);
            let (dx, dy) = if ok_h { ((c * rx - b * ry) / det, (a * ry - b * rx) / det) } else { (0.0, 0.0) };
            let un = [u[0] + dx, u[1] + dy];
            let dev = [un[0] - u0[0], un[1] - u0[1]];
            let nrm = crate::grid::hypot(dev[0], dev[1]);
            let s = 1.0f32.min(max_d / nrm.max(1e-9));
            let un = [u0[0] + dev[0] * s, u0[1] + dev[1] * s];
            let s_new = l0.sad1(bx, by, un);
            if s_new <= s_old {
                u = un;
                s_old = s_new;
            }
        }
        [u[0] / 2.0, u[1] / 2.0]
    })
}

pub(crate) fn distinct(l: &Level, f: &Grid<V2>, e: &Plane, kappa_s: f32, tau_s: f32) -> Plane {
    Grid::from_fn(f.w, f.h, |bx, by| {
        let v = f.at(bx, by);
        let ev = e.at(bx, by);
        let mut kap = f32::INFINITY;
        for d in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            kap = kap.min(l.sad1(bx, by, [v[0] + d.0, v[1] + d.1]) - ev);
        }
        kap.max(0.0) / (kappa_s * (ev + tau_s))
    })
}

/// Level-3 slice statistics as numpy reduces them: rows of a strided slice are summed one by one,
/// a fresh array pairwise.
fn slice_mean(a: &Plane, rw: usize, rh: usize) -> f32 {
    if rw == a.w {
        np_mean(&a.d[..rw * rh])
    } else {
        let mut s = 0.0f32;
        for y in 0..rh {
            s += np_sum(&a.d[y * a.w..y * a.w + rw]);
        }
        s / (rw * rh) as f32
    }
}

fn slice_std(a: &Plane, rw: usize, rh: usize) -> f32 {
    let m = slice_mean(a, rw, rh);
    let mut sq = Vec::with_capacity(rw * rh);
    for y in 0..rh {
        for x in 0..rw {
            let d = a.at(x, y) - m;
            sq.push(d * d);
        }
    }
    (np_sum(&sq) / (rw * rh) as f32).sqrt()
}

/// Block field of cur against prev (back vectors).
pub fn estimate(cur: &Grid<Rgb>, prev: &Grid<Rgb>, p: &MatchParams) -> Matched {
    let (w_f, h_f) = (cur.w, cur.h);
    let pc = pyramid(cur);
    let mut pp = pyramid(prev);
    let top = LEVELS - 1;
    let lam = p.lam;

    let (rh3, rw3) = (h_f.div_ceil(16), w_f.div_ceil(16));
    let (a3, b3) = (&pc[top], &pp[top]);
    let sa = slice_std(a3, rw3, rh3);
    let sb = slice_std(b3, rw3, rh3);
    let mut g = (sa / sb.max(1e-4)) as f64;
    let mut o = (slice_mean(a3, rw3, rh3) - (g as f32) * slice_mean(b3, rw3, rh3)) as f64;
    {
        let (gf, of) = (g as f32, o as f32);
        let mut res = Vec::with_capacity(rw3 * rh3);
        let mut raw = Vec::with_capacity(rw3 * rh3);
        for y in 0..rh3 {
            for x in 0..rw3 {
                res.push((a3.at(x, y) - (gf * b3.at(x, y) + of)).abs());
                raw.push((a3.at(x, y) - b3.at(x, y)).abs());
            }
        }
        if np_mean(&res) as f64 > p.gain_gate as f64 * np_mean(&raw) as f64 {
            g = 1.0;
            o = 0.0;
        }
    }
    let gain = (g, o);
    if g != 1.0 || o != 0.0 {
        let (gf, of) = (g as f32, o as f32);
        for l in pp.iter_mut() {
            for v in l.d.iter_mut() {
                *v = gf * *v + of;
            }
        }
    }
    let pcv = {
        let (v, conf) = phase_corr(&pc[0], &pp[0]);
        if conf < p.pc_min { None } else { Some(v) }
    };

    let real = |k: usize| (h_f.div_ceil(BS << (k + 1)), w_f.div_ceil(BS << (k + 1)));
    let rpx = |k: usize| (h_f.div_ceil(2 << k), w_f.div_ceil(2 << k));

    let l = Level::new(&pc[top], &pp[top], BS, rpx(top));
    let side = 2 * p.radius + 1;
    let mut f = Grid::from_fn(l.nbx, l.nby, |bx, by| {
        let mut best = f32::INFINITY;
        let mut bv = [0.0f32; 2];
        for i in 0..side * side {
            let v = [(i % side - p.radius) as f32, (i / side - p.radius) as f32];
            let c = l.sad1(bx, by, v) + lam * 0.25 * (v[0].abs() + v[1].abs());
            if c < best {
                best = c;
                bv = v;
            }
        }
        bv
    });
    let rtop = real(top);
    f = vmed9(&own9(&f, rtop));
    f = copy_pad(&f, rtop);
    let glob = dominant(&f, p.radius);
    let mut parent = f.clone();
    let mut cut_r = None;
    let mut resc: Option<(Grid<bool>, Grid<V2>, Plane)> = None;
    let mut last_level = l;
    for k in (1..top).rev() {
        let l = Level::new(&pc[k], &pp[k], BS, rpx(k));
        let (w, h) = (l.nbx, l.nby);
        let ro = real(k);
        let scale = (1u32 << (top - k)) as f32;
        let gv = [(glob[0] * scale).round_ties_even(), (glob[1] * scale).round_ties_even()];
        let mut base = vec![Grid::new(w, h, [0.0f32; 2]), Grid::new(w, h, gv), parent_at(&parent, w, h, 0, 0).map(round2)];
        for i in 0..8 {
            let (dx, dy) = around(i);
            base.push(parent_at(&parent, w, h, dx, dy).map(round2));
        }
        if let Some(v) = pcv {
            let s = 2.0f32.powi(-(k as i32));
            base.push(Grid::new(w, h, [(v[0] * s).round_ties_even(), (v[1] * s).round_ties_even()]));
        }
        let pred = vmed9(&parent9(&parent, w, h));
        f = consider_all(&l, &base, &pred, lam);
        let pred = vmed9(&own9(&f, ro));
        let mut cands = base.clone();
        for i in 0..8 {
            let (dx, dy) = around(i);
            cands.push(shifted(&f, dx, dy, ro).map(round2));
        }
        f = consider_all(&l, &cands, &pred, lam);
        // local exhaustive window, then the wider ring at level 1 unless the frame looks like a cut
        let pred = vmed9(&own9(&f, ro));
        let r = if k == 1 { p.local_r1 } else { p.local_r };
        let offs: Vec<V2> = (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| [dx as f32, dy as f32])).collect();
        let f0 = f.clone();
        f = consider_offsets(&l, &f, &offs, &pred, lam, None);
        let mut cutflag = false;
        if k == 1 {
            let cr = unexplained_z(&pc[1], &l.sad(&f), h_f, w_f);
            cutflag = cr > p.cut_lo;
            cut_r = Some(cr);
            if p.wide_r1 > r && !cutflag {
                let ring: Vec<V2> = (-p.wide_r1..=p.wide_r1)
                    .flat_map(|dy| (-p.wide_r1..=p.wide_r1).map(move |dx| (dx, dy)))
                    .filter(|&(dx, dy)| dx.abs().max(dy.abs()) > r)
                    .map(|(dx, dy)| [dx as f32, dy as f32])
                    .collect();
                f = consider_offsets(&l, &f0, &ring, &pred, lam, Some(&f));
            }
        }
        if p.rescue_r > 0 && k == 1 && !cutflag && gain == (1.0, 0.0) {
            let e_cur = l.sad(&f);
            let flag = e_cur.map(|e| e > p.rescue_thr);
            if flag.any() {
                let pred = vmed9(&own9(&f, ro));
                let (rh_, rw_) = rpx(k);
                let rr = p.rescue_r;
                let offs: Vec<V2> = (-rr..=rr).flat_map(|dy| (-rr..=rr).map(move |dx| [dx as f32, dy as f32])).collect();
                let mut ok = Grid::new(w, h, false);
                let mut vr = Grid::new(w, h, [0.0f32; 2]);
                let mut sbm = Grid::new(w, h, f32::INFINITY);
                for by in 0..h {
                    for bx in 0..w {
                        if !flag.at(bx, by) {
                            continue;
                        }
                        let fv = f.at(bx, by);
                        let pv = pred.at(bx, by);
                        let (oxb, oyb) = l.origin(bx, by);
                        let s: Vec<f32> = offs.iter().map(|o| {
                            let v = [fv[0] + o[0], fv[1] + o[1]];
                            let (x0, y0) = (oxb + v[0], oyb + v[1]);
                            if x0 >= 0.0 && y0 >= 0.0 && x0 + BS as f32 <= rw_ as f32 && y0 + BS as f32 <= rh_ as f32 {
                                l.sad1(bx, by, v)
                            } else {
                                f32::INFINITY
                            }
                        }).collect();
                        let cmin_pen = |o: &V2| 0.0 * ((fv[0] + o[0] - pv[0]).abs() + (fv[1] + o[1] - pv[1]).abs());
                        let c: Vec<f32> = s.iter().zip(&offs).map(|(&sv, o)| sv + cmin_pen(o)).collect();
                        let mut bi = 0;
                        for i in 1..c.len() {
                            if c[i] < c[bi] {
                                bi = i;
                            }
                        }
                        let sb = s[bi];
                        let ob = offs[bi];
                        let mut sec2 = f32::INFINITY;
                        for (i, o) in offs.iter().enumerate() {
                            if (o[0] - ob[0]).abs().max((o[1] - ob[1]).abs()) > 2.0 {
                                sec2 = sec2.min(s[i]);
                            }
                        }
                        let ec = e_cur.at(bx, by);
                        let pen = lam * ((fv[0] - pv[0]).abs() + (fv[1] - pv[1]).abs());
                        let good = sb < p.rescue_ratio * ec && sb < p.rescue_abs && sb < p.rescue_dq * sec2 && c[bi] < ec + pen;
                        ok.set(bx, by, good);
                        vr.set(bx, by, [fv[0] + ob[0], fv[1] + ob[1]]);
                        sbm.set(bx, by, sb);
                    }
                }
                resc = Some((ok, vr, sbm));
            }
        }
        for _ in 0..2 {
            let pred = vmed9(&own9(&f, ro));
            let offs: Vec<V2> = (0..9).map(|i| {
                let (dx, dy) = nb(i);
                [dx as f32, dy as f32]
            }).collect();
            f = consider_offsets(&l, &f, &offs, &pred, lam, None);
        }
        if k == 1 {
            f = subpel_level0(&pc[0], &pp[0], &f, rpx(0));
            f = lk_level0(&pc[0], &pp[0], &f, rpx(0), p.lk_iters, p.lk_lambda, p.lk_maxd);
        }
        let n9 = own9(&f, ro);
        let mut med = vmed9(&n9);
        let keep = Grid::from_fn(w, h, |bx, by| {
            let (fv, mv) = (f.at(bx, by), med.at(bx, by));
            l.sad1(bx, by, fv) + lam * ((fv[0] - mv[0]).abs() + (fv[1] - mv[1]).abs()) + p.kappa < l.sad1(bx, by, mv)
        });
        med = Grid::from_fn(w, h, |x, y| if keep.at(x, y) { f.at(x, y) } else { med.at(x, y) });
        if k == 1 {
            med = Grid::from_fn(w, h, |x, y| {
                let m = med.at(x, y);
                let mut wsum = 0.0f32;
                let mut acc = [0.0f32; 2];
                for lane in &n9 {
                    let v = lane.at(x, y);
                    let (dx, dy) = (v[0] - m[0], v[1] - m[1]);
                    let wgt = (-(dx * dx + dy * dy) / 1.0).exp();
                    acc[0] += wgt * v[0];
                    acc[1] += wgt * v[1];
                    wsum += wgt;
                }
                [acc[0] / wsum, acc[1] / wsum]
            });
        }
        f = copy_pad(&med, ro);
        parent = f.clone();
        last_level = l;
    }
    let l = last_level;
    let e = l.sad(&f);
    let still = l.sad_const([0.0, 0.0]);
    let dc = distinct(&l, &f, &e, p.kappa_s, p.tau_s);
    let conf = Grid::from_fn(f.w, f.h, |x, y| {
        let (s, ev) = (still.at(x, y), e.at(x, y));
        dc.at(x, y).max((2.0 * (s - ev) / (s + 0.001)).clamp(0.0, 1.0))
    });
    let field = Grid::from_fn(f.w, f.h, |x, y| {
        let v = f.at(x, y);
        [v[0], v[1], e.at(x, y), conf.at(x, y)]
    });
    let mut n_rescue = 0;
    let f2 = match resc {
        Some((ok, vr, sb)) => {
            n_rescue = ok.count();
            let f2 = Grid::from_fn(f.w, f.h, |x, y| {
                if ok.at(x, y) {
                    let v = vr.at(x, y);
                    [v[0], v[1], sb.at(x, y), conf.at(x, y)]
                } else {
                    [0.0, 0.0, f32::INFINITY, conf.at(x, y)]
                }
            });
            copy_pad(&f2, real(1))
        }
        None => Grid::from_fn(f.w, f.h, |x, y| [0.0, 0.0, f32::INFINITY, conf.at(x, y)]),
    };
    Matched { f: field, rescue: f2, meta: MatchMeta { cut_r: cut_r.unwrap_or(0.0), gain, n_rescue } }
}
