// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Layer analysis of the block field and the per-pixel layer assignment that runs before the
//! photometric choice: code 1 covered (prev sample), 2 uncovered (cur sample), 3 object-at-t
//! (layer sample) or the never-seen hole behind an extrapolated mover.

use super::{d, f, tex, Mode, Params};
use crate::cv;
use crate::grid::{hypot, median, Grid, Image, Mask, Plane, Rgb, V2, FIELD_SCALE, FIELD_TEXEL_I};
use crate::layer::LayerField;

#[derive(Clone, Debug)]
pub struct Layer {
    /// Generous region and strict core on the field grid.
    pub m: Mask,
    pub s: Mask,
    /// Level px.
    pub v: V2,
    pub cov: bool,
    pub unc: bool,
    pub n: usize,
    pub fl_seed: bool,
    pub fl_dim: bool,
    pub prefit: bool,
    pub expl: Option<f64>,
    pub thin: Option<f32>,
}

pub struct Analysis {
    pub bdom: V2,
    pub layers: Vec<Layer>,
}

/// Python's round(x, 4).
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap()
}

fn med_stack(a: &Plane, r: i64) -> Plane {
    Grid::from_fn(a.w, a.h, |x, y| {
        let mut v = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
        for dy in -r..=r {
            for dx in -r..=r {
                v.push(a.atc(x as i64 + dx, y as i64 + dy));
            }
        }
        median(&mut v)
    })
}

fn kron_mask(m: &Mask, s: usize, w: usize, h: usize) -> Mask {
    Grid::from_fn(w, h, |x, y| {
        let (cx, cy) = (x / s, y / s);
        cx < m.w && cy < m.h && m.at(cx, cy)
    })
}

fn bbox(m: &Mask) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..m.h {
        for x in 0..m.w {
            if m.at(x, y) {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    (x0 != usize::MAX).then_some((x0, y0, x1, y1))
}

pub fn analyse_layers(fld: &crate::grid::Field, f2: Option<&crate::grid::Field>, p: &Params, seed_all: bool, fl: Option<&LayerField>, frame: (usize, usize)) -> Analysis {
    let (w, h) = (fld.w, fld.h);
    let (fw, fh) = frame;
    let mut b = fld.map(|t| [t[0], t[1]]);
    let mut z = fld.map(|t| t[2]);
    for ty in 0..h {
        for tx in 0..w {
            let cy = fh.saturating_sub(ty * FIELD_TEXEL_I).min(FIELD_TEXEL_I);
            let cx = fw.saturating_sub(tx * FIELD_TEXEL_I).min(FIELD_TEXEL_I);
            if cx * cy < 256 {
                z.set(tx, ty, f32::INFINITY);
            }
        }
    }
    if let Some(f2) = f2 {
        for i in 0..b.d.len() {
            let t = f2.d[i];
            if t[2].is_finite() && t[2] < z.d[i] {
                b.d[i] = [t[0], t[1]];
                z.d[i] = t[2];
            }
        }
    }
    let bdom = [median(&mut b.d.iter().map(|v| v[0]).collect::<Vec<_>>()), median(&mut b.d.iter().map(|v| v[1]).collect::<Vec<_>>())];
    let bx = b.map(|v| v[0]);
    let by = b.map(|v| v[1]);
    let (mx, my) = (med_stack(&bx, 1), med_stack(&by, 1));
    let mut res: Vec<f32> = bx.d.iter().zip(&mx.d).map(|(a, m)| (a - m).abs()).collect();
    res.extend(by.d.iter().zip(&my.d).map(|(a, m)| (a - m).abs()));
    let sig = (1.4826 * median(&mut res) as f64).max(p.divergence_sigma_floor);
    let thr = p.divergence_k * 8f64.sqrt() * sig;
    let dv = Grid::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i64, y as i64);
        (bx.atc(xi + 1, yi) - bx.atc(xi - 1, yi)) + (by.atc(xi, yi + 1) - by.atc(xi, yi - 1))
    });
    let dm = med_stack(&dv, 2);
    let dp = dv.zip(&dm, |a, m| a - m);
    let rel = b.map(|v| hypot(v[0] - bdom[0], v[1] - bdom[1]));
    let fin = z.map(|v| v.is_finite());
    let (ro, zg, zo, zu) = (f(p.layer_min_rel), f(p.layer_cost), f(p.layer_strict_cost), f(p.layer_unexplained_cost));
    let objg = Grid::from_fn(w, h, |x, y| rel.at(x, y) > ro && z.at(x, y) <= zg && fin.at(x, y));
    let strict = Grid::from_fn(w, h, |x, y| rel.at(x, y) > ro && z.at(x, y) <= zo && fin.at(x, y));
    let objg = cv::close3(&objg);
    let (n, lab) = cv::connected_components(&objg, false);
    let thr32 = f(thr);
    let fp = cv::dilate3(&dp.map(|v| v > thr32));
    let fm = cv::dilate3(&dp.map(|v| v < -thr32));
    let mut layers = Vec::new();
    for c in 1..n as i32 {
        let m0 = lab.map(|l| l == c);
        let s = m0.zip(&strict, |a, b| a && b);
        let dil = cv::dilate3(&m0);
        let m = Grid::from_fn(w, h, |x, y| m0.at(x, y) || (dil.at(x, y) && z.at(x, y) > zu && fin.at(x, y)));
        if !s.any() {
            continue;
        }
        let pick = |g: &Plane| median(&mut g.d.iter().zip(&s.d).filter(|(_, &k)| k).map(|(&v, _)| v).collect::<Vec<_>>());
        let v = [pick(&bx), pick(&by)];
        let cnt = m.count();
        if cnt < p.layer_min_texels {
            continue;
        }
        let cov = m.d.iter().zip(&fp.d).any(|(&a, &b)| a && b);
        let unc = m.d.iter().zip(&fm.d).any(|(&a, &b)| a && b);
        layers.push(Layer { m, s, v, cov, unc, n: cnt, fl_seed: false, fl_dim: false, prefit: false, expl: None, thin: None });
    }
    if let Some(fl) = fl {
        let (vf, zf) = (&fl.v, &fl.z);
        let tile = fl.tile;
        let ro_px = 2.0f32;
        let bpx = [bdom[0] * FIELD_SCALE, bdom[1] * FIELD_SCALE];
        let mut seed = Grid::from_fn(vf.w, vf.h, |x, y| {
            let v = vf.at(x, y);
            let zv = zf.at(x, y);
            let zok = if seed_all { zv.is_finite() } else { zv.is_finite() && zv <= zo };
            zok && hypot(v[0], v[1]) >= ro_px && hypot(v[0] - bpx[0], v[1] - bpx[1]) >= ro_px
        });
        for l in &layers {
            let lv = [l.v[0] * FIELD_SCALE, l.v[1] * FIELD_SCALE];
            seed = Grid::from_fn(vf.w, vf.h, |x, y| {
                let v = vf.at(x, y);
                seed.at(x, y) && hypot(v[0] - lv[0], v[1] - lv[1]) >= ro_px
            });
        }
        if seed.any() {
            let (ns, labs) = cv::connected_components(&seed, true);
            for cs in 1..ns as i32 {
                let comp0 = labs.map(|l| l == cs);
                if comp0.count() < 2 {
                    continue;
                }
                let mut queue = std::collections::VecDeque::from([comp0]);
                while let Some(comp) = queue.pop_front() {
                    if comp.count() < 2 {
                        continue;
                    }
                    let pts: Vec<V2> = vf.d.iter().zip(&comp.d).filter(|(_, &k)| k).map(|(&v, _)| v).collect();
                    let med_v = [median(&mut pts.iter().map(|v| v[0]).collect::<Vec<_>>()), median(&mut pts.iter().map(|v| v[1]).collect::<Vec<_>>())];
                    let dev: Vec<f32> = pts.iter().map(|v| hypot(v[0] - med_v[0], v[1] - med_v[1])).collect();
                    let dmax = dev.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    if dmax >= ro_px {
                        let i1 = dev.iter().position(|&v| v == dmax).unwrap();
                        let p1 = pts[i1];
                        let dd: Vec<f32> = pts.iter().map(|v| hypot(v[0] - p1[0], v[1] - p1[1])).collect();
                        let dmx = dd.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let p2 = pts[dd.iter().position(|&v| v == dmx).unwrap()];
                        let c1 = Grid::from_fn(vf.w, vf.h, |x, y| {
                            let v = vf.at(x, y);
                            comp.at(x, y) && hypot(v[0] - p1[0], v[1] - p1[1]) <= hypot(v[0] - p2[0], v[1] - p2[1])
                        });
                        let c2 = Grid::from_fn(vf.w, vf.h, |x, y| {
                            let v = vf.at(x, y);
                            comp.at(x, y) && hypot(v[0] - p1[0], v[1] - p1[1]) > hypot(v[0] - p2[0], v[1] - p2[1])
                        });
                        if c1.any() && c2.any() {
                            queue.push_back(c1);
                            queue.push_back(c2);
                            continue;
                        }
                    }
                    let v_seed = [med_v[0] / FIELD_SCALE, med_v[1] / FIELD_SCALE];
                    let mut ms = Grid::new(w, h, false);
                    for ty in 0..comp.h {
                        for tx in 0..comp.w {
                            if comp.at(tx, ty) {
                                ms.set(((tx * tile) / FIELD_TEXEL_I).min(w - 1), ((ty * tile) / FIELD_TEXEL_I).min(h - 1), true);
                            }
                        }
                    }
                    if ms.any() {
                        let zs = median(&mut zf.d.iter().zip(&comp.d).filter(|(_, &k)| k).map(|(&v, _)| v).collect::<Vec<_>>());
                        let n = ms.count();
                        layers.push(Layer { s: ms.clone(), m: ms, v: v_seed, cov: true, unc: true, n, fl_seed: true, fl_dim: zs > zo, prefit: false, expl: None, thin: None });
                    }
                }
            }
        }
    }
    layers.sort_by_key(|l| std::cmp::Reverse(l.n));
    layers.truncate(8);
    Analysis { bdom, layers }
}

/// |grad I| in D units per px (central differences).
pub fn grad(cur: &Image) -> Plane {
    let (w, h) = (cur.w, cur.h);
    let half = |a: Rgb, b: Rgb| d([(a[0] - b[0]) / 2.0, (a[1] - b[1]) / 2.0, (a[2] - b[2]) / 2.0], [0.0; 3]);
    Grid::from_fn(w, h, |x, y| {
        let gx = if x >= 1 && x + 1 < w { half(cur.at(x + 1, y), cur.at(x - 1, y)) } else { 0.0 };
        let gy = if y >= 1 && y + 1 < h { half(cur.at(x, y + 1), cur.at(x, y - 1)) } else { 0.0 };
        hypot(gx, gy)
    })
}

/// tau_m(x) = sqrt(tau_0^2 + (k tg |grad I|)^2).
fn tau_g(p: &Params, g: f32) -> f32 {
    let dg = f(p.grad_tolerance_k * p.grad_tolerance) * g;
    (f(p.layer_agree_tau * p.layer_agree_tau) + dg * dg).sqrt()
}

#[inline]
fn samp(m: &Plane, x: f32, y: f32) -> f32 {
    m.atc(x.round_ties_even() as i64, y.round_ties_even() as i64)
}

pub struct Membership {
    pub ms: Vec<Plane>,
    pub cell: Grid<i32>,
}

pub fn membership(prev: &Image, cur: &Image, a: &mut Analysis, p: &Params, pixmem: bool) -> Membership {
    let (w, h) = (cur.w, cur.h);
    let k = a.layers.len();
    let s_ = p.membership_cell;
    let (ch, cw) = (h.div_ceil(s_), w.div_ceil(s_));
    let tau = f(p.layer_agree_tau);
    let mut vecs: Vec<V2> = a.layers.iter().map(|l| [l.v[0] * FIELD_SCALE, l.v[1] * FIELD_SCALE]).collect();
    vecs.push([a.bdom[0] * FIELD_SCALE, a.bdom[1] * FIELD_SCALE]);
    let eraw: Vec<Mask> = vecs.iter().map(|u| Grid::from_fn(w, h, |x, y| d(tex(prev, x as f32 + u[0], y as f32 + u[1]), cur.at(x, y)) < tau)).collect();
    let up = |m: &Grid<bool>| kron_mask(m, s_, w, h);
    let tpc = FIELD_TEXEL_I / s_;
    let mut regs: Vec<Mask> = a.layers.iter().map(|l| kron_mask(&cv::dilate3(&l.m), FIELD_TEXEL_I, w, h)).collect();
    let kg = cv::rect(2 * tpc + 1, 2 * tpc + 1);
    let mut cell = Grid::new(cw, ch, -1i32);
    let mut eb: Vec<Mask> = Vec::new();
    for it in 0..=p.membership_grow {
        let uni = Grid::from_fn(w, h, |x, y| regs.iter().any(|r| r.at(x, y)));
        eb = (0..=k).map(|i| {
            let rg = if i < k { &regs[i] } else { &uni };
            eraw[i].zip(rg, |a, b| a && b)
        }).collect();
        let rho: Vec<Plane> = eb.iter().map(|e| Grid::from_fn(cw, ch, |cx, cy| {
            let mut n = 0u32;
            for y in cy * s_..((cy + 1) * s_).min(h) {
                for x in cx * s_..((cx + 1) * s_).min(w) {
                    n += e.at(x, y) as u32;
                }
            }
            n as f32 / (s_ * s_) as f32
        })).collect();
        let (fr, fd) = (f(p.cell_explained), f(p.cell_margin));
        let mut lab = Grid::new(cw, ch, 0i32);
        let mut rmax = Grid::new(cw, ch, 0.0f32);
        let mut tie = Grid::new(cw, ch, false);
        for cy in 0..ch {
            for cx in 0..cw {
                let mut v: Vec<f32> = rho.iter().map(|r| r.at(cx, cy)).collect();
                let mut li = 0;
                for i in 1..v.len() {
                    if v[i] > v[li] {
                        li = i;
                    }
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let r1 = v[v.len() - 1];
                let r2 = if k >= 1 { v[v.len() - 2] } else { 0.0 };
                let expl = r1 >= fr;
                let clear = expl && r1 - r2 >= fd;
                lab.set(cx, cy, li as i32);
                rmax.set(cx, cy, r1);
                tie.set(cx, cy, expl && !clear);
                cell.set(cx, cy, if clear { li as i32 } else { -1 });
            }
        }
        if tie.any() {
            let kap = ((1.0 - 0.4) / p.membership_eps).ln() + ((1.0 - p.membership_eps) / 0.4).ln();
            let ck = f((s_ * s_) as f64 * kap);
            let cost: Vec<Plane> = rho.iter().map(|r| Grid::from_fn(cw, ch, |x, y| if r.at(x, y) >= fr { ck * (rmax.at(x, y) - r.at(x, y)) } else { f32::INFINITY })).collect();
            let mut cur_l = Grid::from_fn(cw, ch, |x, y| if tie.at(x, y) { lab.at(x, y) } else { cell.at(x, y) });
            for _ in 0..2 {
                let prevl = cur_l.clone();
                let nbv = |x: usize, y: usize, dx: i64, dy: i64| {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= cw as i64 || ny >= ch as i64 { -2 } else { prevl.at(nx as usize, ny as usize) }
                };
                cur_l = Grid::from_fn(cw, ch, |x, y| {
                    if !tie.at(x, y) {
                        return prevl.at(x, y);
                    }
                    let nb = [nbv(x, y, 0, -1), nbv(x, y, 0, 1), nbv(x, y, -1, 0), nbv(x, y, 1, 0)];
                    let mut bi = 0;
                    let mut bv = f64::INFINITY;
                    for u in 0..=k {
                        let cnt = nb.iter().filter(|&&n| n != u as i32).count() as f64;
                        let t = cost[u].at(x, y) as f64 + p.cell_potts_weight * cnt;
                        if t < bv {
                            bv = t;
                            bi = u;
                        }
                    }
                    bi as i32
                });
            }
            cell = cur_l;
        }
        if it < p.membership_grow {
            let mut grew = false;
            for i in 0..k {
                let ext = kron_mask(&cv::dilate(&cell.map(|c| c == i as i32), &kg), s_, w, h);
                let new = regs[i].zip(&ext, |a, b| a || b);
                grew |= new != regs[i];
                regs[i] = new;
            }
            if !grew {
                break;
            }
        }
    }
    if p.entering_cells && k > 0 {
        for i in 0..k {
            let v = vecs[i];
            let vn = hypot(v[0], v[1]);
            if vn < 1.0 {
                continue;
            }
            let dx = (v[0] / vn).round_ties_even() as i64;
            let dy = (v[1] / vn).round_ties_even() as i64;
            let ent = Grid::from_fn(cw, ch, |x, y| {
                let (ccx, ccy) = ((x * s_) as f64 + 0.5 * s_ as f64, (y * s_) as f64 + 0.5 * s_ as f64);
                let (ex, ey) = (ccx + v[0] as f64, ccy + v[1] as f64);
                cell.at(x, y) == -1 && (ex < 0.0 || ex > (w - 1) as f64 || ey < 0.0 || ey > (h - 1) as f64)
            });
            for _ in 0..((vn as f64 / s_ as f64).ceil() as usize + 1) {
                let isl = cell.map(|c| c == i as i32);
                let new = Grid::from_fn(cw, ch, |x, y| {
                    let (sx, sy) = (x as i64 - dx, y as i64 - dy);
                    let nb = sx >= 0 && sy >= 0 && sx < cw as i64 && sy < ch as i64 && isl.at(sx as usize, sy as usize);
                    ent.at(x, y) && nb && cell.at(x, y) == -1
                });
                if !new.any() {
                    break;
                }
                cell = Grid::from_fn(cw, ch, |x, y| if new.at(x, y) { i as i32 } else { cell.at(x, y) });
            }
        }
    }
    let eps = f(p.membership_eps);
    let mut ms = Vec::with_capacity(k);
    let mut e0: Option<Mask> = None;
    for i in 0..k {
        let isl = cell.map(|c| c == i as i32);
        let dil = cv::dilate3(&isl);
        let adj = Grid::from_fn(cw, ch, |x, y| dil.at(x, y) && cell.at(x, y) == -1);
        let cnt = cv::sum3_const(&eb[i].map(|b| b as u8 as f32));
        let maj = cnt.map(|c| c >= f(5.0 - 1e-6));
        let (upl, upa) = (up(&isl), up(&adj));
        let mut mem = Grid::from_fn(w, h, |x, y| upl.at(x, y) || (upa.at(x, y) && maj.at(x, y)));
        let vl = vecs[i];
        let sep = hypot(vl[0], vl[1]).min(hypot(vl[0] - vecs[k][0], vl[1] - vecs[k][1]));
        if pixmem && sep >= f(p.pixel_membership_sep) {
            let e0m = e0.get_or_insert_with(|| Grid::from_fn(w, h, |x, y| d(prev.at(x, y), cur.at(x, y)) < tau));
            let pix = Grid::from_fn(w, h, |x, y| eb[i].at(x, y) && !eraw[k].at(x, y) && !e0m.at(x, y));
            let npx = pix.count();
            let both = pix.d.iter().zip(&mem.d).filter(|(&a, &b)| a && b).count();
            let thin = 1.0 - (both as f64 / npx.max(1) as f64);
            a.layers[i].thin = Some(thin as f32);
            let usep = if p.thin_layer > 0.0 {
                thin >= p.thin_layer && a.layers[i].expl.unwrap_or(1.0) >= p.seed_min_explained
            } else {
                a.layers[i].fl_dim || !p.pixel_membership_dim_only
            };
            if usep {
                mem = mem.zip(&pix, |a, b| a || b);
            }
        }
        if p.boundary_membership {
            let e0m = e0.get_or_insert_with(|| Grid::from_fn(w, h, |x, y| d(prev.at(x, y), cur.at(x, y)) < tau)).clone();
            let near = up(&dil);
            let mov = hypot(vl[0], vl[1]) >= 0.5;
            mem = Grid::from_fn(w, h, |x, y| {
                let pe = eraw[i].at(x, y) && !eraw[k].at(x, y) && near.at(x, y) && !(mov && e0m.at(x, y));
                mem.at(x, y) || pe
            });
        }
        ms.push(mem.map(|b| if b { 1.0 - eps } else { eps }));
    }
    Membership { ms, cell }
}

/// Count of pixels of the crop whose prev sample at offset u matches cur, minus a pull to v0.
struct Scorer<'a> {
    prev: &'a Image,
    cur: &'a Image,
    rr: Mask,
    x0: usize,
    y0: usize,
    tau: f32,
    v0: [f64; 2],
    cache: std::collections::HashMap<(u64, u64, usize), f64>,
}

impl<'a> Scorer<'a> {
    fn score(&mut self, u: [f64; 2], sb: usize) -> f64 {
        let k = (round4(u[0]), round4(u[1]));
        let key = (k.0.to_bits(), k.1.to_bits(), sb);
        if let Some(&v) = self.cache.get(&key) {
            return v;
        }
        let (k0, k1) = (k.0 as f32, k.1 as f32);
        let mut n = 0usize;
        let mut y = 0;
        while y < self.rr.h {
            let mut x = 0;
            while x < self.rr.w {
                if self.rr.at(x, y) {
                    let (gx, gy) = (self.x0 + x, self.y0 + y);
                    if d(tex(self.prev, gx as f32 + k0, gy as f32 + k1), self.cur.at(gx, gy)) < self.tau {
                        n += 1;
                    }
                }
                x += sb;
            }
            y += sb;
        }
        let v = n as f64 - 1e-3 * (k.0 - self.v0[0]).hypot(k.1 - self.v0[1]);
        self.cache.insert(key, v);
        v
    }
}

fn argmax3(g: &[[f64; 3]; 3]) -> (usize, usize) {
    let mut best = (0, 0);
    for iy in 0..3 {
        for ix in 0..3 {
            if g[iy][ix] > g[best.0][best.1] {
                best = (iy, ix);
            }
        }
    }
    best
}

/// Sub-pixel layer vector by explanation count over the layer's dilated texels (residual set of
/// the dominant motion), coarse-to-fine bracketing on the 1/8 px lattice.
pub fn refine_layer_vectors(prev: &Image, cur: &Image, a: &mut Analysis, p: &Params) {
    let (w, h) = (cur.w, cur.h);
    let tau = f(p.layer_agree_tau);
    let bb = [a.bdom[0] * FIELD_SCALE, a.bdom[1] * FIELD_SCALE];
    for l in a.layers.iter_mut() {
        if l.prefit {
            continue;
        }
        let rg = kron_mask(&cv::dilate3(&l.m), FIELD_TEXEL_I, w, h);
        let Some((x0, y0, x1, y1)) = bbox(&rg) else { continue };
        let rr = Grid::from_fn(x1 - x0, y1 - y0, |x, y| {
            let (gx, gy) = (x0 + x, y0 + y);
            rg.at(gx, gy) && !(d(tex(prev, gx as f32 + bb[0], gy as f32 + bb[1]), cur.at(gx, gy)) < tau)
        });
        let v0 = [((l.v[0] * FIELD_SCALE) as f64 * 8.0 + 0.5).floor() / 8.0, ((l.v[1] * FIELD_SCALE) as f64 * 8.0 + 0.5).floor() / 8.0];
        let mut sc = Scorer { prev, cur, rr, x0, y0, tau, v0, cache: Default::default() };
        let mut c0 = v0;
        let steps: &[f64] = &[2.0, 0.5, 0.125];
        for &st in steps {
            let sb = if st >= 2.0 { p.refine_coarse_step } else { 1 };
            let mut ix0 = 1;
            let mut iy0 = 1;
            for esc in 0..=2 {
                let g = if p.refine_compass && st < 2.0 {
                    let mut g = [[f64::NEG_INFINITY; 3]; 3];
                    g[1][1] = sc.score(c0, sb);
                    for dd in [-1i32, 1] {
                        g[1][(1 + dd) as usize] = sc.score([c0[0] + dd as f64 * st, c0[1]], sb);
                        g[(1 + dd) as usize][1] = sc.score([c0[0], c0[1] + dd as f64 * st], sb);
                    }
                    let jx = if g[1][0] > g[1][1].max(g[1][2]) { 0 } else if g[1][2] > g[1][1] { 2 } else { 1 };
                    let jy = if g[0][1] > g[1][1].max(g[2][1]) { 0 } else if g[2][1] > g[1][1] { 2 } else { 1 };
                    if jx != 1 && jy != 1 {
                        g[jy][jx] = sc.score([c0[0] + (jx as f64 - 1.0) * st, c0[1] + (jy as f64 - 1.0) * st], sb);
                    }
                    g
                } else {
                    let mut g = [[0.0; 3]; 3];
                    for (iy, dy) in [-st, 0.0, st].iter().enumerate() {
                        for (ix, dx) in [-st, 0.0, st].iter().enumerate() {
                            g[iy][ix] = sc.score([c0[0] + dx, c0[1] + dy], sb);
                        }
                    }
                    g
                };
                let (iy, ix) = argmax3(&g);
                if g[iy][ix] <= g[1][1] {
                    iy0 = 1;
                    ix0 = 1;
                    break;
                }
                iy0 = iy;
                ix0 = ix;
                if (ix != 1 || iy != 1) && esc < 2 {
                    c0 = [c0[0] + (ix as f64 - 1.0) * st, c0[1] + (iy as f64 - 1.0) * st];
                } else {
                    break;
                }
            }
            c0 = [c0[0] + (ix0 as f64 - 1.0) * st, c0[1] + (iy0 as f64 - 1.0) * st];
        }
        let best = c0;
        let nres = sc.rr.count() as f64;
        let expl = (sc.score(best, 1) + 1e-3 * (best[0] - v0[0]).hypot(best[1] - v0[1])) / nres.max(1.0);
        l.expl = Some(expl);
        l.v = [(best[0] / FIELD_SCALE as f64) as f32, (best[1] / FIELD_SCALE as f64) as f32];
    }
}

/// Alternating step: v_L on the cells labelled L (parabola on a 1/8 grid, 1/16 lattice).
fn refine_on_cells(prev: &Image, cur: &Image, a: &mut Analysis, p: &Params, cell: &Grid<i32>) {
    let (w, h) = (cur.w, cur.h);
    let tau = f(p.layer_agree_tau);
    for (i, l) in a.layers.iter_mut().enumerate() {
        let isl = kron_mask(&cell.map(|c| c == i as i32), p.membership_cell, w, h);
        let Some((x0, y0, x1, y1)) = bbox(&isl) else { continue };
        let rr = isl.crop(x0, y0, x1 - x0, y1 - y0);
        let v0 = [(l.v[0] * FIELD_SCALE) as f64, (l.v[1] * FIELD_SCALE) as f64];
        let mut sc = Scorer { prev, cur, rr, x0, y0, tau, v0, cache: Default::default() };
        let mut g = [[0.0; 3]; 3];
        let q = 0.125;
        for (iy, dy) in [-q, 0.0, q].iter().enumerate() {
            for (ix, dx) in [-q, 0.0, q].iter().enumerate() {
                g[iy][ix] = sc.score([v0[0] + dx, v0[1] + dy], 1);
            }
        }
        let (iy0, ix0) = argmax3(&g);
        let mut bx = v0[0] + (ix0 as f64 - 1.0) * q;
        let mut by = v0[1] + (iy0 as f64 - 1.0) * q;
        let denx = g[iy0][0] - 2.0 * g[iy0][1] + g[iy0][2];
        if ix0 == 1 && denx < -1e-6 {
            bx += (0.0625 * (g[iy0][0] - g[iy0][2]) / denx).clamp(-0.0625, 0.0625);
        }
        let deny = g[0][ix0] - 2.0 * g[1][ix0] + g[2][ix0];
        if iy0 == 1 && deny < -1e-6 {
            by += (0.0625 * (g[0][ix0] - g[2][ix0]) / deny).clamp(-0.0625, 0.0625);
        }
        let cand = [(bx * 16.0).round_ties_even() / 16.0, (by * 16.0).round_ties_even() / 16.0];
        let best = if sc.score(cand, 1) >= g[iy0][ix0] { cand } else { [v0[0] + (ix0 as f64 - 1.0) * q, v0[1] + (iy0 as f64 - 1.0) * q] };
        l.v = [(best[0] / FIELD_SCALE as f64) as f32, (best[1] / FIELD_SCALE as f64) as f32];
    }
}

/// Static-hypothesis residual with the fringe unmix (quad partner's temporal difference).
fn static_residual(prev: &Image, cur: &Image) -> Plane {
    let (w, h) = (cur.w, cur.h);
    let dp = |x: usize, y: usize| {
        let (c, p) = (cur.at(x, y), prev.at(x, y));
        [c[0] - p[0], c[1] - p[1], c[2] - p[2]]
    };
    Grid::from_fn(w, h, |x, y| {
        let v = dp(x, y);
        let qa = dp((x ^ 1).min(w - 1), y);
        let qb = dp(x, (y ^ 1).min(h - 1));
        let (ma, mb) = (d(qa, [0.0; 3]), d(qb, [0.0; 3]));
        let db = if ma >= mb { qa } else { qb };
        let num = ((0.0 + v[0] * db[0]) + v[1] * db[1]) + v[2] * db[2];
        let den = ((0.0 + db[0] * db[0]) + db[1] * db[1]) + db[2] * db[2];
        let beta = (num / (den + 1e-8)).clamp(0.0, 1.0);
        d([v[0] - beta * db[0], v[1] - beta * db[1], v[2] - beta * db[2]], [0.0; 3])
    })
}

/// 1-D Catmull-Rom along an axis at mid phase when the other axis is near-integer; bilinear else.
fn cubic_tap(img: &Image, x: f32, y: f32) -> (Rgb, bool) {
    let (w, h) = (img.w, img.h);
    let out = tex(img, x, y);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let ix = fx.min(1.0 - fx) <= 0.125;
    let iy = fy.min(1.0 - fy) <= 0.125;
    let ux = (fx - 0.5).abs() <= 0.25 && iy;
    let uy = (fy - 0.5).abs() <= 0.25 && ix;
    let cr = |fr: f32, tap: &dyn Fn(i32) -> Rgb| {
        let f3 = fr.powf(3.0);
        let wts = [
            -0.5 * fr + fr * fr - 0.5 * f3,
            1.0 - 2.5 * (fr * fr) + 1.5 * f3,
            0.5 * fr + 2.0 * (fr * fr) - 1.5 * f3,
            -0.5 * (fr * fr) + 0.5 * f3,
        ];
        let mut acc = [0.0f32; 3];
        for (j, wk) in wts.iter().enumerate() {
            let s = tap(j as i32 - 1);
            for c in 0..3 {
                acc[c] += wk * s[c];
            }
        }
        acc.map(|v| v.clamp(0.0, 1.0))
    };
    if ux {
        let yr = y.round_ties_even();
        return (cr(fx, &|k| img.bilinear((x.floor() + k as f32).clamp(0.0, (w - 1) as f32), yr)), true);
    }
    if uy {
        let xr = x.round_ties_even();
        return (cr(fy, &|k| img.bilinear(xr, (y.floor() + k as f32).clamp(0.0, (h - 1) as f32))), true);
    }
    (out, false)
}

pub struct LayerAssignment {
    /// 0 none, 1 covered, 2 uncovered, 3 object-at-t / hole.
    pub code: Grid<i8>,
    pub col: Image,
    /// Frame px.
    pub bdom: V2,
    pub lv: Vec<V2>,
    pub has_layers: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn assign_layers(prev: &Image, cur: &Image, fld: &crate::grid::Field, f2: Option<&crate::grid::Field>, t: f32, mode: Mode, p: &Params, pixmem: bool, fl: Option<&LayerField>, guard: bool, prefit: &[V2]) -> LayerAssignment {
    let (w, h) = (cur.w, cur.h);
    let interp = mode == Mode::Interp;
    let (omt, opt) = (f(1.0 - t as f64), f(1.0 + t as f64));
    let ext_lay = !interp && !guard;
    let mut a = analyse_layers(fld, f2, p, p.seed_all_tiles, if interp { fl } else { None }, (w, h));
    for l in a.layers.iter_mut() {
        l.prefit = !prefit.is_empty() && prefit.iter().any(|pv| (l.v[0] - pv[0]).abs().max((l.v[1] - pv[1]).abs()) < 1e-5);
    }
    if !a.layers.is_empty() && (interp || ext_lay) {
        refine_layer_vectors(prev, cur, &mut a, p);
        if p.seed_min_explained > 0.0 {
            a.layers.retain(|l| !l.fl_seed || l.expl.unwrap_or(1.0) >= p.seed_min_explained);
        }
    }
    let mut code = Grid::new(w, h, 0i8);
    let mut col = Grid::new(w, h, [0.0f32; 3]);
    let mut strct = Grid::new(w, h, false);
    let bdom = [a.bdom[0] * FIELD_SCALE, a.bdom[1] * FIELD_SCALE];
    let mk = |code, col, _strct, a: &Analysis| LayerAssignment {
        code,
        col,
        bdom,
        lv: a.layers.iter().map(|l| [l.v[0] * FIELD_SCALE, l.v[1] * FIELD_SCALE]).collect(),
        has_layers: !a.layers.is_empty(),
    };
    if (!interp && !ext_lay) || a.layers.is_empty() {
        return mk(code, col, strct, &a);
    }
    let gc = grad(cur);
    let te = f(p.agree_tau);
    let bb = bdom;
    if ext_lay {
        let mut mem = membership(prev, cur, &mut a, p, false);
        for _ in 0..p.joint_rounds {
            refine_on_cells(prev, cur, &mut a, p, &mem.cell);
            mem = membership(prev, cur, &mut a, p, false);
        }
        let ms = &mem.ms;
        let in_frame = |x: f32, y: f32| x >= 0.0 && x <= (w - 1) as f32 && y >= 0.0 && y <= (h - 1) as f32;
        let n = w * h;
        let mut obj = vec![false; n];
        let mut wake2 = vec![false; n];
        let mut wake1 = vec![false; n];
        let mut wake_h = vec![false; n];
        let mut dsg = vec![0.0f32; n];
        let mut o_ext = vec![[0.0f32; 3]; n];
        let mut o_h = vec![[0.0f32; 3]; n];
        let s_ = p.membership_cell;
        let a_h = p.hole_exponent;
        let td = t as f64;
        let w_h64 = (0.5 * (1.0 - ((1.0 + td).powf(a_h) - td.powf(a_h)))).clamp(0.0, 1.0);
        let (w_h, w_h1) = (f(w_h64), f(1.0 - w_h64));
        let c1b: Vec<Rgb> = (0..n).map(|i| tex(cur, (i % w) as f32 + t * bb[0], (i / w) as f32 + t * bb[1])).collect();
        let c0b: Vec<Rgb> = (0..n).map(|i| tex(prev, (i % w) as f32 + opt * bb[0], (i / w) as f32 + opt * bb[1])).collect();
        for (li, l) in a.layers.iter().enumerate() {
            if mem.cell.d.iter().filter(|&&c| c == li as i32).count() < 16 {
                continue;
            }
            let v = [l.v[0] * FIELD_SCALE, l.v[1] * FIELD_SCALE];
            let isl_d = cv::dilate3(&mem.cell.map(|c| c == li as i32));
            let foot = kron_mask(&isl_d, s_, w, h);
            let r_b = (hypot((v[0] - bb[0]) * t, (v[1] - bb[1]) * t) as f64 / 12.0).max(1.5) as f32;
            for i in 0..n {
                let (px, py) = ((i % w) as f32, (i / w) as f32);
                let (x1x, x1y) = (px + t * bb[0], py + t * bb[1]);
                let (x0x, x0y) = (px + opt * bb[0], py + opt * bb[1]);
                let (in_x1, in_x0) = (in_frame(x1x, x1y), in_frame(x0x, x0y));
                let (qx, qy) = (px + t * v[0], py + t * v[1]);
                let (rx, ry) = (qx + v[0], qy + v[1]);
                let (in_q, in_r) = (in_frame(qx, qy), in_frame(rx, ry));
                let mq = samp(&ms[li], qx, qy);
                let mut o = mq > 0.5 && in_q;
                let m1b = samp(&ms[li], x1x, x1y) > 0.5;
                let m0b = samp(&ms[li], x0x - v[0], x0y - v[1]) > 0.5;
                let mut w2 = !o && m1b && !m0b && in_x0;
                let mut w1 = !o && !w2 && foot.at(i % w, i / w) && !m1b && in_x1;
                let mut hh = m1b && m0b && in_q;
                if !(o || w2 || w1 || hh) {
                    continue;
                }
                strct.d[i] = true;
                let l1 = tex(cur, qx, qy);
                let mut l0 = tex(prev, rx, ry);
                if o && in_q && !in_r {
                    l0 = l1;
                }
                let eo = d(l1, l0);
                let teo_x = tau_g(p, samp(&gc, qx, qy));
                o &= eo < teo_x;
                let mut okx = eo >= teo_x;
                if p.wake_coincidence && (w2 || w1) && !okx {
                    let tc = f(0.5 * p.agree_tau);
                    okx = d(l1, tex(prev, qx + bb[0], qy + bb[1])) < tc;
                }
                let eb = d(c1b[i], c0b[i]);
                w2 &= okx && eb >= te;
                w1 &= okx && (eb < te || m0b || !in_x0);
                hh &= !o && !w2 && !w1;
                o &= !obj[i];
                let mut sg2 = f(2.0 * p.object_sigma * p.object_sigma);
                let gq = samp(&gc, qx, qy) * f(p.grad_tolerance);
                sg2 += 2.0 * (gq * gq);
                let lm = (mq / (1.0 - mq)).ln().clamp(-4.0, 4.0);
                let eb_eff = if in_x0 { eb.max(0.04) } else { te };
                let data_t = if in_r { (eb_eff * eb_eff - eo * eo) / sg2 } else { 0.0 };
                if o {
                    dsg[i] = data_t + lm;
                    obj[i] = true;
                    o_ext[i] = if in_r && eo < f(0.5 * p.layer_agree_tau) { [0, 1, 2].map(|c| 0.85 * l1[c] + 0.15 * l0[c]) } else { l1 };
                }
                if hh {
                    let s4 = |sx: f32, sy: f32| tex(cur, qx + sx * r_b, qy + sy * r_b);
                    let (aa, bq, cq, dq) = (s4(-1.0, -1.0), s4(-1.0, 1.0), s4(1.0, -1.0), s4(1.0, 1.0));
                    let l1h = [0, 1, 2].map(|c| 0.25 * (((0.0 + aa[c]) + bq[c]) + cq[c] + dq[c]));
                    o_h[i] = if in_r && mq <= 0.5 { [0, 1, 2].map(|c| w_h1 * l1h[c] + w_h * l0[c]) } else { l1h };
                    wake_h[i] = true;
                }
                wake2[i] |= w2;
                wake1[i] |= w1;
            }
        }
        if obj.iter().any(|&b| b) {
            let lam = f(p.object_potts_weight);
            let o1 = Grid::from_fn(w, h, |x, y| (obj[y * w + x] && dsg[y * w + x] > 0.0) as u8 as f32);
            let no = cv::sum3_const(&o1).zip(&o1, |s, v| s - v);
            let nin = cv::sum3_const(&Grid::new(w, h, 1.0f32)).map(|s| s - 1.0);
            for i in 0..n {
                obj[i] = obj[i] && dsg[i] > lam * (nin.d[i] - 2.0 * no.d[i]);
            }
        }
        for i in 0..n {
            wake2[i] &= !obj[i];
            wake1[i] &= !obj[i] && !wake2[i];
            wake_h[i] &= !obj[i] && !wake2[i] && !wake1[i];
            let c = if obj[i] { 3 } else if wake_h[i] || wake1[i] { 2 } else if wake2[i] { 1 } else { 0 };
            code.d[i] = c;
            col.d[i] = if obj[i] {
                o_ext[i]
            } else if c == 1 {
                c0b[i]
            } else if wake1[i] {
                c1b[i]
            } else if wake_h[i] {
                o_h[i]
            } else {
                [0.0; 3]
            };
        }
        return mk(code, col, strct, &a);
    }

    // interpolation
    let x1 = |px: f32, py: f32| (px - omt * bb[0], py - omt * bb[1]);
    let x0 = |px: f32, py: f32| (px + t * bb[0], py + t * bb[1]);
    let mut mem = membership(prev, cur, &mut a, p, pixmem);
    for _ in 0..p.joint_rounds {
        refine_on_cells(prev, cur, &mut a, p, &mem.cell);
        mem = membership(prev, cur, &mut a, p, pixmem);
    }
    let ms = &mem.ms;
    let es = static_residual(prev, cur);
    let n = w * h;
    let mut obj = vec![false; n];
    let mut cov = vec![false; n];
    let mut unc = vec![false; n];
    let mut dsg = vec![0.0f32; n];
    let mut o0 = vec![[0.0f32; 3]; n];
    let mut o1 = vec![[0.0f32; 3]; n];
    let in_frame = |x: f32, y: f32| x >= 0.0 && x <= (w - 1) as f32 && y >= 0.0 && y <= (h - 1) as f32;
    let tc = f(0.5 * p.agree_tau);
    for (li, l) in a.layers.iter().enumerate() {
        let v = [l.v[0] * FIELD_SCALE, l.v[1] * FIELD_SCALE];
        let sc = l.cov;
        let su = l.unc;
        let c3g = p.object_ground_veto > 0 && hypot(v[0] - bb[0], v[1] - bb[1]) >= 1.0;
        let inner = if c3g && p.object_ground_rim > 0 {
            let r = p.object_ground_rim;
            let e = cv::erode(&ms[li].map(|m| m > 0.5), &cv::rect(2 * r + 1, 2 * r + 1));
            Some(e.map(|b| b as u8 as f32))
        } else {
            None
        };
        let flags = |i: usize| {
            let (px, py) = ((i % w) as f32, (i / w) as f32);
            let (qx, qy) = (px - omt * v[0], py - omt * v[1]);
            let mq = samp(&ms[li], qx, qy);
            let prot = mq > 0.5;
            let (x1x, x1y) = x1(px, py);
            let (x0x, x0y) = x0(px, py);
            let m1 = samp(&ms[li], x1x, x1y) > 0.5;
            let m0 = samp(&ms[li], x0x - v[0], x0y - v[1]) > 0.5;
            (qx, qy, mq, prot, sc && m1 && !prot && !m0, su && m0 && !prot && !m1)
        };
        if !(0..n).any(|i| {
            let (_, _, _, o, c, u) = flags(i);
            o || c || u
        }) {
            continue;
        }
        for i in 0..n {
            let (px, py) = ((i % w) as f32, (i / w) as f32);
            let (qx, qy, mut mq, prot, mut c_, mut u_) = flags(i);
            let mut o = prot;
            strct.d[i] |= c_ || u_ || o;
            let (rx, ry) = (px + t * v[0], py + t * v[1]);
            let (in_q, in_r) = (in_frame(qx, qy), in_frame(rx, ry));
            if !(o || c_ || u_) && !(in_r && !in_q) {
                continue;
            }
            let es_r = d(cur.at(i % w, i / w), prev.at(i % w, i / w));
            let ent = o && in_q && !in_r && es_r > 0.015;
            let mut ext = false;
            if in_r && !in_q && es_r > 0.015 {
                let ax = if v[0] > 1e-4 { rx / v[0] } else if v[0] < -1e-4 { (rx - (w - 1) as f32) / v[0] } else { 1.0 };
                let ay = if v[1] > 1e-4 { ry / v[1] } else if v[1] < -1e-4 { (ry - (h - 1) as f32) / v[1] } else { 1.0 };
                let al = ax.min(ay).clamp(0.0, 1.0);
                let (qxc, qyc) = (rx - al * v[0], ry - al * v[1]);
                let mq_c = samp(&ms[li], qxc, qyc);
                if mq_c > 0.5 {
                    ext = true;
                    mq = mq_c;
                }
            }
            o |= ext;
            if !(o || c_ || u_) {
                continue;
            }
            let (mut l0, mut l1) = if p.layer_cubic { (cubic_tap(prev, rx, ry).0, cubic_tap(cur, qx, qy).0) } else { (tex(prev, rx, ry), tex(cur, qx, qy)) };
            if ent {
                l0 = l1;
            }
            if ext {
                l1 = l0;
            }
            let eo = d(l0, l1);
            let teo_x = tau_g(p, samp(&gc, qx, qy));
            o &= eo < teo_x;
            if c3g && o && !obj[i] {
                let mut vt = eo >= f(p.object_ground_min_cost);
                if let Some(inn) = &inner {
                    vt &= samp(inn, qx, qy) < 0.5;
                }
                if vt {
                    let d1 = d(l1, tex(prev, qx + bb[0], qy + bb[1]));
                    let d0 = d(l0, tex(cur, rx - bb[0], ry - bb[1]));
                    let tcv = tc.min(f(p.object_ground_rel) * eo);
                    let (g1, g0) = (d1 < tcv, d0 < tcv);
                    o = !(if p.object_ground_veto == 2 { g1 || g0 } else { g1 && g0 });
                }
            }
            let mut pass = eo >= teo_x;
            if p.band_coincidence && (c_ || u_) && !pass {
                let g1 = d(l1, tex(prev, qx + bb[0], qy + bb[1])) < tc;
                let g0 = d(l0, tex(cur, rx - bb[0], ry - bb[1])) < tc;
                let ok = if p.band_coincidence_one_side { (c_ && !u_ && g1) || (u_ && !c_ && g0) || (c_ && u_ && g1 && g0) } else { g1 && g0 };
                pass |= ok;
            }
            c_ &= pass;
            u_ &= pass;
            o &= !obj[i];
            if o {
                let mut sg2 = f(2.0 * p.object_sigma * p.object_sigma);
                let gq = samp(&gc, qx, qy) * f(p.grad_tolerance);
                sg2 += 2.0 * (gq * gq);
                dsg[i] = (es.d[i] * es.d[i] - eo * eo) / sg2 + (mq / (1.0 - mq)).ln();
                obj[i] = true;
                o0[i] = l0;
                o1[i] = l1;
            }
            cov[i] |= c_;
            unc[i] |= u_;
        }
    }
    let mut c0 = vec![[0.0f32; 3]; n];
    let mut c1 = vec![[0.0f32; 3]; n];
    for i in 0..n {
        let (px, py) = ((i % w) as f32, (i / w) as f32);
        let (x0x, x0y) = x0(px, py);
        let (x1x, x1y) = x1(px, py);
        c0[i] = tex(prev, x0x, x0y);
        c1[i] = tex(cur, x1x, x1y);
        let eb = d(c0[i], c1[i]);
        cov[i] &= eb >= te;
        unc[i] &= eb >= te;
    }
    if obj.iter().any(|&b| b) {
        let lam = f(p.object_potts_weight);
        let o1m = Grid::from_fn(w, h, |x, y| (obj[y * w + x] && dsg[y * w + x] > 0.0) as u8 as f32);
        let no = cv::sum3_const(&o1m).zip(&o1m, |s, v| s - v);
        for i in 0..n {
            obj[i] = obj[i] && dsg[i] > lam * (8.0 - 2.0 * no.d[i]);
        }
    }
    for i in 0..n {
        let both = cov[i] && unc[i];
        let cv_ = cov[i] && !both && !obj[i];
        let uv_ = unc[i] && !both && !obj[i];
        let c = if obj[i] { 3 } else if uv_ { 2 } else if cv_ { 1 } else { 0 };
        code.d[i] = c;
        col.d[i] = match c {
            1 => c0[i],
            2 => c1[i],
            3 => [0, 1, 2].map(|k| o0[i][k] + (o1[i][k] - o0[i][k]) * t),
            _ => [0.0; 3],
        };
    }
    mk(code, col, strct, &a)
}
