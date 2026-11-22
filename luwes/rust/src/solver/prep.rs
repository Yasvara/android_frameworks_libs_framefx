// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Candidate construction and the threshold-free taps of every candidate, for the pixels that pass
//! the 2/255 early-out (or may receive moving content).

use super::layers::{assign_layers, LayerAssignment};
use super::fieldops::{apply_gain, channel_gain, dominant_vector, extend_entering, two_motion_pan, median_repair, swept_mover};
use super::{d, f, lvl, sub, tex, vmed9, Gain, Mode, Params};
use crate::cv;
use crate::grid::{field_at, hypot, median, Field, Grid, Image, Rgb, V2, FIELD_SCALE, FIELD_TEXEL};
use crate::layer::LayerField;

/// Everything one pair contributes to a solve.
pub struct PairInputs<'a> {
    /// Previous frame.
    pub prev: &'a Image,
    /// Current frame.
    pub cur: &'a Image,
    /// Block field.
    pub f: &'a Field,
    /// Rescue (second) vectors; z = inf where none.
    pub rescue: Option<&'a Field>,
    /// Scene-cut statistic of the matcher.
    pub cut_r: f32,
    /// Matcher's gain.
    pub gain: Gain,
    /// Residual-layer field.
    pub fl: Option<&'a LayerField>,
    /// Dense flow pair (cur -> prev, prev -> cur), frame px; the dense path only.
    pub dense: Option<(&'a Grid<V2>, &'a Grid<V2>)>,
}

pub(crate) struct ActiveLayers {
    pub code: Vec<i8>,
    pub col: Vec<Rgb>,
    pub bdom: V2,
    pub lv: Vec<V2>,
    pub has_layers: bool,
    /// Full-frame object-at-t pixels outside the active list and their colour.
    pub c3: Option<(Vec<bool>, Vec<Rgb>)>,
}

/// Candidates and taps of the active pixels; consumed by `decide`.
pub struct Prepared {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) t: f32,
    pub(crate) interp: bool,
    pub(crate) k: usize,
    pub(crate) lay_k: Option<usize>,
    pub(crate) dense: bool,
    pub(crate) idx: Vec<usize>,
    pub(crate) cur: Image,
    /// prev as the decision and the static blend read it.
    pub(crate) prev: Image,
    pub(crate) now: Vec<Rgb>,
    pub(crate) before: Vec<Rgb>,
    pub(crate) dp: Vec<f32>,
    pub(crate) qdp: Vec<f32>,
    pub(crate) fp: Vec<[f32; 4]>,
    /// Per candidate k, per active pixel j at k * n + j.
    pub(crate) b: Vec<V2>,
    pub(crate) zw: Vec<[f32; 2]>,
    pub(crate) kept: Vec<bool>,
    pub(crate) c1: Vec<Rgb>,
    pub(crate) c0: Vec<Rgb>,
    pub(crate) e: Vec<f32>,
    pub(crate) s1: Vec<f32>,
    pub(crate) s0: Vec<f32>,
    pub(crate) r1: Vec<f32>,
    pub(crate) lv: Vec<f32>,
    pub(crate) lb: Vec<f32>,
    pub(crate) key: Vec<f32>,
    pub(crate) sup: Vec<u8>,
    pub(crate) var: Vec<f32>,
    pub(crate) lay: Vec<bool>,
    pub(crate) moving: Vec<bool>,
    pub(crate) overlay_res: Vec<f32>,
    pub(crate) arrival_res: Vec<f32>,
    pub(crate) arrival_test: Vec<bool>,
    pub(crate) gain_phase: Option<GainPhase>,
    pub(crate) edge: Option<(Vec<V2>, Vec<Rgb>, Vec<Rgb>)>,
    pub(crate) horizon: Option<(Vec<bool>, Vec<Rgb>)>,
    pub(crate) layer_assign: Option<ActiveLayers>,
    pub(crate) gain: (f64, f64),
    pub(crate) cut_guard: bool,
    pub(crate) cut_pi: f64,
}

pub(crate) struct GainPhase {
    pub gch: [f32; 3],
    pub och: [f32; 3],
    pub c0eb: Option<Vec<Rgb>>,
    pub c1eb: Option<Vec<Rgb>>,
    pub eb_same: Option<Vec<bool>>,
}

impl Prepared {
    /// Active pixel count.
    pub fn n(&self) -> usize {
        self.idx.len()
    }
}

fn quad_min(a: &Grid<f32>) -> Grid<f32> {
    Grid::from_fn(a.w, a.h, |x, y| {
        let (x0, y0) = (x & !1, y & !1);
        let mut m = a.at(x0, y0);
        for (dx, dy) in [(1, 0), (0, 1), (1, 1)] {
            let (xx, yy) = (x0 + dx, y0 + dy);
            if xx < a.w && yy < a.h {
                m = m.min(a.at(xx, yy));
            }
        }
        m
    })
}

/// Fields and frames -> candidates and their taps for one (t, mode).
pub fn prepare(inp: &PairInputs, t: f32, mode: Mode, p: &Params) -> Prepared {
    let (cur, prev_in) = (inp.cur, inp.prev);
    let (w, h) = (cur.w, cur.h);
    let interp = mode == Mode::Interp;
    let (omt, opt) = (f(1.0 - t as f64), f(1.0 + t as f64));
    let gain_m = Gain { scale: inp.gain.scale, offset: inp.gain.offset };
    let gain = if gain_m.accepted() { gain_m.clone() } else { Gain { scale: 1.0, offset: 0.0 } };
    let cut_guard = inp.cut_r > f(p.cut_lo);
    let guard = cut_guard;

    let mut fld = median_repair(prev_in, cur, inp.f, &inp.gain, inp.cut_r, p.cut_lo);
    let mut f2: Option<Field> = inp.rescue.cloned();
    let fl = inp.fl;
    let nonzero = |fl: &Field| fl.d.iter().any(|t| t[0] != 0.0 || t[1] != 0.0);
    let mut prefit: Vec<V2> = Vec::new();
    if (!interp || gain.accepted()) && !guard && nonzero(&fld) {
        let r = two_motion_pan(prev_in, cur, &fld, f2.as_ref(), p, &gain);
        fld = r.f;
        f2 = r.f2;
        prefit = r.prefit;
    }
    if interp && !guard && !gain.accepted() && nonzero(&fld) {
        if let Some(fc) = swept_mover(prev_in, cur, &fld, f2.as_ref(), p, fl) {
            fld = fc;
            f2 = None;
        }
    }
    let n_px = w * h;
    let pxy = |i: usize| ((i % w) as f32, (i / w) as f32);
    let mut fp: Vec<[f32; 4]> = (0..n_px).map(|i| {
        let (x, y) = pxy(i);
        field_at(&fld, x, y)
    }).collect();
    let mut b_bil: Vec<V2> = fp.iter().map(|v| [v[0], v[1]]).collect();
    let ends = |b: V2, x: f32, y: f32| -> (f32, f32, f32, f32) {
        if interp {
            (x - b[0] * omt, y - b[1] * omt, x + b[0] * t, y + b[1] * t)
        } else {
            (x + b[0] * t, y + b[1] * t, x + b[0] * opt, y + b[1] * opt)
        }
    };
    let dense = p.dense && inp.dense.is_some();
    let fat = |x: f32, y: f32| -> V2 {
        let fd = inp.dense.unwrap().0;
        fd.bilinear(x, y)
    };
    if dense {
        let (fd, gd) = inp.dense.unwrap();
        let mut bd: Vec<V2> = (0..n_px).map(|i| if interp { [0.5 * (fd.d[i][0] - gd.d[i][0]), 0.5 * (fd.d[i][1] - gd.d[i][1])] } else { fd.d[i] }).collect();
        if !interp {
            for _ in 0..p.extrap_fixed_point {
                bd = (0..n_px).map(|i| {
                    let (x, y) = pxy(i);
                    fat(x + t * bd[i][0], y + t * bd[i][1])
                }).collect();
            }
        }
        b_bil = bd;
        for i in 0..n_px {
            fp[i][0] = b_bil[i][0];
            fp[i][1] = b_bil[i][1];
        }
    }
    let it: Vec<[f32; 4]> = (0..n_px).map(|i| {
        let (x, y) = pxy(i);
        let (ax, ay, _, _) = ends(b_bil[i], x, y);
        let mut v = field_at(&fld, ax, ay);
        if dense {
            let u = fat(ax, ay);
            v[0] = u[0];
            v[1] = u[1];
        }
        v
    }).collect();
    let (fw, fh) = (fld.w, fld.h);
    let texel = |i: usize| -> (i64, i64) {
        let (x, y) = pxy(i);
        (((x / FIELD_TEXEL).floor() as i64).clamp(0, fw as i64 - 1), ((y / FIELD_TEXEL).floor() as i64).clamp(0, fh as i64 - 1))
    };
    let nbr: Vec<[V2; 9]> = (0..n_px).map(|i| {
        let (tx, ty) = texel(i);
        std::array::from_fn(|l| {
            let tv = fld.atc(tx + (l % 3) as i64 - 1, ty + (l / 3) as i64 - 1);
            [tv[0] * FIELD_SCALE, tv[1] * FIELD_SCALE]
        })
    }).collect();
    let mut vecs: Vec<Vec<V2>> = vec![b_bil.clone(), it.iter().map(|v| [v[0], v[1]]).collect()];
    let mut zws: Vec<Vec<[f32; 2]>> = vec![fp.iter().map(|v| [v[2], v[3]]).collect(), it.iter().map(|v| [v[2], v[3]]).collect()];
    for l in 0..9 {
        if l == 4 {
            continue;
        }
        let (dx, dy) = ((l % 3) as i64 - 1, (l / 3) as i64 - 1);
        let mut vv = Vec::with_capacity(n_px);
        let mut zz = Vec::with_capacity(n_px);
        for i in 0..n_px {
            let (tx, ty) = texel(i);
            let tv = fld.atc(tx + dx, ty + dy);
            vv.push([tv[0] * FIELD_SCALE, tv[1] * FIELD_SCALE]);
            zz.push([tv[2], tv[3]]);
        }
        vecs.push(vv);
        zws.push(zz);
    }
    let n_main = vecs.len();
    if p.projected_candidates > 0 {
        let fmax = fld.d.iter().fold(0.0f32, |m, t| m.max(t[0].abs()).max(t[1].abs()));
        let span = f(if interp { (1.0 - t as f64).max(0.25) } else { (t as f64).max(0.25) });
        let rw = ((fmax * FIELD_SCALE * span / FIELD_TEXEL).ceil().clamp(1.0, 5.0)) as i64;
        let sg = if interp { 1.0 - t } else { -t };
        let mut picks: Vec<Vec<V2>> = vec![Vec::with_capacity(n_px); p.projected_candidates];
        let mut pickz: Vec<Vec<[f32; 2]>> = vec![Vec::with_capacity(n_px); p.projected_candidates];
        let ded = f(p.dedupe_dist);
        let lim = 16.0 + p.projection_margin;
        for i in 0..n_px {
            let (px, py) = pxy(i);
            let (tx, ty) = texel(i);
            let mut cand: Vec<(f64, V2, [f32; 2])> = Vec::new();
            for dy in -rw..=rw {
                for dx in -rw..=rw {
                    let tv = fld.atc(tx + dx, ty + dy);
                    let b = [tv[0] * FIELD_SCALE, tv[1] * FIELD_SCALE];
                    let cx = ((tx + dx) * 32) as f64 + 16.0 - 0.5;
                    let cy = ((ty + dy) * 32) as f64 + 16.0 - 0.5;
                    let lx = cx + (b[0] * sg) as f64;
                    let ly = cy + (b[1] * sg) as f64;
                    let outside = tx + dx < 0 || tx + dx >= fw as i64 || ty + dy < 0 || ty + dy >= fh as i64;
                    let dist = if outside { f64::INFINITY } else { (px as f64 - lx).abs().max((py as f64 - ly).abs()) };
                    cand.push((dist, b, [tv[2], tv[3]]));
                }
            }
            let mut ok: Vec<bool> = cand.iter().map(|c| c.0 <= lim).collect();
            for pk in 0..p.projected_candidates {
                let mut j = 0;
                let mut bd = f64::INFINITY;
                for (q, c) in cand.iter().enumerate() {
                    let dd = if ok[q] { c.0 } else { f64::INFINITY };
                    if dd < bd {
                        bd = dd;
                        j = q;
                    }
                }
                let found = bd.is_finite();
                let (bj, zj) = (cand[j].1, cand[j].2);
                picks[pk].push(if found { bj } else { b_bil[i] });
                pickz[pk].push(if found { zj } else { [fp[i][2], fp[i][3]] });
                for (q, c) in cand.iter().enumerate() {
                    ok[q] = ok[q] && hypot(c.1[0] - bj[0], c.1[1] - bj[1]) / FIELD_SCALE > ded;
                }
            }
        }
        for pk in 0..p.projected_candidates {
            vecs.push(std::mem::take(&mut picks[pk]));
            zws.push(std::mem::take(&mut pickz[pk]));
        }
    }
    let mut lay_k: Option<(usize, Vec<bool>)> = None;
    if let Some(fl) = fl {
        let (vm, zm) = splat(fl, w, h, t, interp);
        let ok: Vec<bool> = zm.iter().map(|z| z.is_finite()).collect();
        let v0: Vec<V2> = (0..n_px).map(|i| if ok[i] { vm[i] } else { b_bil[i] }).collect();
        let z0: Vec<[f32; 2]> = (0..n_px).map(|i| [if ok[i] { zm[i] } else { fp[i][2] }, fp[i][3]]).collect();
        lay_k = Some((vecs.len(), ok));
        vecs.push(v0);
        zws.push(z0);
    }
    if p.dominant_candidate && nonzero(&fld) && !guard {
        let bd = dominant_vector(prev_in, cur, &fld, p);
        if hypot(bd[0], bd[1]) >= 0.5 {
            vecs.push(vec![bd; n_px]);
            zws.push(fp.iter().map(|v| [v[2], v[3]]).collect());
        }
    }
    let n_cur = vecs.len();
    if let Some(f2) = &f2 {
        for l in 0..9 {
            let (dx, dy) = ((l % 3) as i64 - 1, (l / 3) as i64 - 1);
            let mut vv = Vec::with_capacity(n_px);
            let mut zz = Vec::with_capacity(n_px);
            for i in 0..n_px {
                let (tx, ty) = texel(i);
                let tv = f2.atc(tx + dx, ty + dy);
                vv.push([tv[0] * FIELD_SCALE, tv[1] * FIELD_SCALE]);
                zz.push([tv[2], tv[3]]);
            }
            vecs.push(vv);
            zws.push(zz);
        }
    }
    let k = vecs.len();

    // photometric reference: prev under the per-channel gain when the matcher accepted one
    let gain_on = gain.accepted();
    let (prev, gch_och) = if gain_on {
        let (g, o, _) = channel_gain(cur, prev_in, &gain, false);
        (apply_gain(prev_in, &g, &o, false), Some(([g[0] as f32, g[1] as f32, g[2] as f32], o)))
    } else {
        (prev_in.clone(), None)
    };
    let g_scale = match &gch_och {
        Some((g, _)) => {
            let m = (g[0] + g[1] + g[2]) / 3.0;
            (((1.0 + (m as f64) * (m as f64)) / 2.0).sqrt()).max(1.0) as f32
        }
        None => 1.0,
    };
    let dp_full: Grid<f32> = Grid::from_fn(w, h, |x, y| d(cur.at(x, y), prev.at(x, y)));
    let mut act: Vec<bool> = (0..n_px).map(|i| {
        let (c, pv) = (cur.d[i], prev.d[i]);
        !((c[0] - pv[0]).abs().max((c[1] - pv[1]).abs()).max((c[2] - pv[2]).abs()) < 2.0 / 255.0)
    }).collect();
    let mut mov = vec![false; n_px];
    let mut sf = vec![0.0f32; n_px];
    let mut sa = vec![0.0f32; n_px];
    let mut test = vec![false; n_px];
    for i in 0..n_px {
        let (x, y) = pxy(i);
        let b = b_bil[i];
        let (x1x, x1y, _, _) = ends(b, x, y);
        mov[i] = hypot(x1x - x, x1y - y) >= 0.5;
        sf[i] = d(cur.d[i], tex(cur, x - b[0], y - b[1]));
        sa[i] = d(cur.d[i], tex(cur, x1x, x1y));
        test[i] = !act[i] && mov[i];
        act[i] |= test[i];
    }
    let idx: Vec<usize> = (0..n_px).filter(|&i| act[i]).collect();
    let n = idx.len();

    // dedupe (layer vectors only against b_bil), second vectors need a finite z
    let mut kept = vec![true; k * n];
    let ded = f(p.dedupe_dist);
    for (j, &i) in idx.iter().enumerate() {
        for kk in 1..k {
            let mut kp = true;
            for jj in 0..kk {
                if kept[jj * n + j] && lvl(sub(vecs[kk][i], vecs[jj][i])) <= ded {
                    kp = false;
                }
            }
            kept[kk * n + j] = kp;
        }
        if f2.is_some() {
            for kk in n_cur..k {
                kept[kk * n + j] &= zws[kk][i][0].is_finite();
            }
        }
        if let Some((lk, ok)) = &lay_k {
            kept[lk * n + j] = ok[i] && lvl(sub(vecs[*lk][i], b_bil[i])) > 0.02;
        }
    }

    let mut b = vec![[0.0f32; 2]; k * n];
    let mut zw = vec![[0.0f32; 2]; k * n];
    let mut c1 = vec![[0.0f32; 3]; k * n];
    let mut c0 = vec![[0.0f32; 3]; k * n];
    let mut e = vec![0.0f32; k * n];
    let mut s1 = vec![0.0f32; k * n];
    let mut s0 = vec![0.0f32; k * n];
    let mut r1 = vec![0.0f32; k * n];
    let mut lvv = vec![0.0f32; k * n];
    let mut lb = vec![0.0f32; k * n];
    let mut key = vec![0.0f32; k * n];
    let mut sup = vec![0u8; k * n];
    let mut var = vec![0.0f32; n];
    let cons = f(p.support_radius);
    let box_e = dense && p.dense_cost_box > 0;
    let mut e_full: Vec<Grid<f32>> = Vec::new();
    for kk in 0..k {
        if box_e {
            let ef = Grid::from_fn(w, h, |x, y| {
                let i = y * w + x;
                let bv = vecs[kk][i];
                let (ax, ay, cx, cy) = ends(bv, x as f32, y as f32);
                d(tex(cur, ax, ay), tex(&prev, cx, cy)) / g_scale
            });
            e_full.push(cv::blur(&ef, 2 * p.dense_cost_box + 1));
        }
        for (j, &i) in idx.iter().enumerate() {
            let (x, y) = pxy(i);
            let bv = vecs[kk][i];
            let (ax, ay, cx, cy) = ends(bv, x, y);
            let cc1 = tex(cur, ax, ay);
            let cc0 = tex(&prev, cx, cy);
            let o = kk * n + j;
            b[o] = bv;
            zw[o] = zws[kk][i];
            c1[o] = cc1;
            c0[o] = cc0;
            e[o] = if box_e { e_full[kk].d[i] } else { d(cc1, cc0) / g_scale };
            s1[o] = d(cc1, tex(&prev, ax, ay)) / g_scale;
            s0[o] = d(tex(cur, cx, cy), cc0) / g_scale;
            let fk = if dense { fat(ax, ay) } else { let v = field_at(&fld, ax, ay); [v[0], v[1]] };
            r1[o] = lvl(sub(fk, bv));
            lvv[o] = lvl(sub(bv, b_bil[i]));
            lb[o] = lvl(bv);
        }
    }
    for (j, &i) in idx.iter().enumerate() {
        let med = vmed9(&nbr[i]);
        for kk in 0..k {
            let o = kk * n + j;
            key[o] = lvl(sub(b[o], med));
            let mut s = 0u8;
            for l in 0..9 {
                s += (lvl(sub(nbr[i][l], b[o])) <= cons) as u8;
            }
            sup[o] = s;
        }
        let mut vmax = f32::NEG_INFINITY;
        for kk in 2..n_main {
            vmax = vmax.max(lvv[kk * n + j]);
        }
        var[j] = vmax;
    }
    let lay: Vec<bool> = (0..n).map(|j| lay_k.as_ref().map(|(lk, _)| kept[lk * n + j]).unwrap_or(false)).collect();
    let qd = quad_min(&dp_full);
    let gather = |v: &Vec<f32>| -> Vec<f32> { idx.iter().map(|&i| v[i]).collect() };
    let mut fpz: Vec<[f32; 4]> = idx.iter().map(|&i| fp[i]).collect();

    let mut gain_phase = None;
    if let Some((gch, och)) = gch_och {
        let mut z = GainPhase { gch, och, c0eb: None, c1eb: None, eb_same: None };
        if interp {
            let fe = extend_entering(&fld, h, w, p.edge_extend_overlap, false);
            let mut c0eb = Vec::with_capacity(n_px);
            let mut c1eb = Vec::with_capacity(n_px);
            let mut same = Vec::with_capacity(n_px);
            for i in 0..n_px {
                let (x, y) = pxy(i);
                let be = field_at(&fe, x, y);
                c0eb.push(tex(prev_in, x + t * be[0], y + t * be[1]));
                c1eb.push(tex(cur, x - omt * be[0], y - omt * be[1]));
                same.push(hypot(be[0] - b_bil[i][0], be[1] - b_bil[i][1]) < 1e-3);
            }
            z.c0eb = Some(c0eb);
            z.c1eb = Some(c1eb);
            z.eb_same = Some(same);
        }
        gain_phase = Some(z);
    }
    let mut edge = None;
    if interp {
        let fe = extend_entering(&fld, h, w, p.edge_extend_overlap, false);
        let mut be = Vec::with_capacity(n);
        let mut c1e = Vec::with_capacity(n);
        let mut c0e = Vec::with_capacity(n);
        for &i in &idx {
            let (x, y) = pxy(i);
            let v = field_at(&fe, x, y);
            let bv = [v[0], v[1]];
            let (ax, ay, cx, cy) = ends(bv, x, y);
            be.push(bv);
            c1e.push(tex(cur, ax, ay));
            c0e.push(tex(&prev, cx, cy));
        }
        edge = Some((be, c1e, c0e));
    }
    let mut horizon = None;
    let no_horizon = interp || guard || (dense && p.dense_no_horizon);
    if !no_horizon && nonzero(&fld) {
        let fe = extend_entering(&fld, h, w, 0.5, true);
        let (rh, rw) = (h.div_ceil(32), w.div_ceil(32));
        let pick = |c: usize| {
            let mut v = Vec::new();
            for y in 0..rh.min(fe.h) {
                for x in 0..rw.min(fe.w) {
                    v.push(fe.at(x, y)[c]);
                }
            }
            median(&mut v) * FIELD_SCALE
        };
        let bdom = [pick(0), pick(1)];
        let mut hor0 = vec![false; n];
        let mut c1v = vec![[0.0f32; 3]; n];
        for (j, &i) in idx.iter().enumerate() {
            let (x, y) = pxy(i);
            let fpe = field_at(&fe, x, y);
            let be = [fpe[0], fpe[1]];
            let diff = hypot(b[j][0] - be[0], b[j][1] - be[1]) > 0.5;
            b[j] = be;
            if diff {
                fpz[j] = fpe;
            }
            let (x1f, y1f) = (x + t * be[0], y + t * be[1]);
            let (x0f, y0f) = (x + opt * be[0], y + opt * be[1]);
            let (ix, iy) = ((i % w) as f64, (i / w) as f64);
            let (x1d, y1d) = (ix + (t * be[0]) as f64, iy + (t * be[1]) as f64);
            let (x0d, y0d) = (ix + (opt * be[0]) as f64, iy + (opt * be[1]) as f64);
            let c1e = tex(cur, x1f, y1f);
            let c0e = tex(&prev, x0f, y0f);
            let s1e = d(tex(cur, x1f, y1f), tex(&prev, x1f, y1f));
            let in1 = x1d >= 0.0 && x1d <= (w - 1) as f64 && y1d >= 0.0 && y1d <= (h - 1) as f64;
            let in0 = x0d >= 0.0 && x0d <= (w - 1) as f64 && y0d >= 0.0 && y0d <= (h - 1) as f64;
            if diff {
                c1[j] = c1e;
                c0[j] = c0e;
                e[j] = d(c1e, c0e);
                s1[j] = s1e;
            }
            let dom_ok = hypot(be[0] - bdom[0], be[1] - bdom[1]) / FIELD_SCALE <= 1.25;
            let h0 = (in1 || p.horizon_both_out) && !in0 && dom_ok;
            if h0 {
                e[j] = f(0.25 * p.agree_tau);
                s0[j] = s1[j];
            }
            hor0[j] = h0;
            c1v[j] = c1e;
        }
        horizon = Some((hor0, c1v));
    }
    let mut layers_z = None;
    if p.layer_bands {
        let pixmem = p.pixel_membership && !(p.pixel_membership_needs_constancy && gain.accepted());
        let a: LayerAssignment = assign_layers(&prev, cur, &fld, f2.as_ref(), t, mode, p, pixmem, fl, guard, &prefit);
        let c3 = if p.static_object_pass {
            let m: Vec<bool> = (0..n_px).map(|i| a.code.d[i] == 3 && !act[i]).collect();
            let col: Vec<Rgb> = (0..n_px).map(|i| if m[i] { a.col.d[i] } else { [0.0; 3] }).collect();
            Some((m, col))
        } else {
            None
        };
        layers_z = Some(ActiveLayers {
            code: idx.iter().map(|&i| a.code.d[i]).collect(),
            col: idx.iter().map(|&i| a.col.d[i]).collect(),
            bdom: a.bdom,
            lv: a.lv.clone(),
            has_layers: a.has_layers,
            c3,
        });
    }
    let prev_dec = if gain_on && !cut_guard { prev.clone() } else { prev_in.clone() };
    let cut_pi = {
        let u = ((inp.cut_r as f64 - p.cut_lo) / (p.cut_hi - p.cut_lo)).clamp(0.0, 1.0);
        u * u * (3.0 - 2.0 * u)
    };
    Prepared {
        w,
        h,
        t,
        interp,
        k,
        lay_k: lay_k.map(|l| l.0),
        dense,
        cur: cur.clone(),
        prev: prev_dec,
        now: idx.iter().map(|&i| cur.d[i]).collect(),
        before: idx.iter().map(|&i| prev.d[i]).collect(),
        dp: gather(&dp_full.d),
        qdp: gather(&qd.d),
        fp: fpz,
        b,
        zw,
        kept,
        c1,
        c0,
        e,
        s1,
        s0,
        r1,
        lv: lvv,
        lb,
        key,
        sup,
        var,
        lay,
        moving: idx.iter().map(|&i| mov[i]).collect(),
        overlay_res: gather(&sf),
        arrival_res: gather(&sa),
        arrival_test: idx.iter().map(|&i| test[i]).collect(),
        gain_phase,
        edge,
        horizon,
        layer_assign: layers_z,
        gain: (gain.scale, gain.offset),
        cut_guard,
        cut_pi,
        idx,
    }
}

/// Forward-project the layer tiles to phase t (interp: content at x in cur sits at x + v in prev);
/// lowest-cost tiles are written last and win overlaps. Per pixel (vector, cost; inf = none).
pub(crate) fn splat(fl: &LayerField, w: usize, h: usize, t: f32, interp: bool) -> (Vec<V2>, Vec<f32>) {
    let tsz = fl.tile as f64;
    let m = fl.margin as f64;
    let mut vm = vec![[0.0f32; 2]; w * h];
    let mut zm = vec![f32::INFINITY; w * h];
    let mut tiles: Vec<(usize, usize, f32)> = Vec::new();
    for a in 0..fl.z.h {
        for bx in 0..fl.z.w {
            let z = fl.z.at(bx, a);
            if z.is_finite() {
                tiles.push((a, bx, z));
            }
        }
    }
    tiles.sort_by(|p, q| q.2.total_cmp(&p.2));
    for (a, bx, z) in tiles {
        let v = fl.v.at(bx, a);
        let omt = super::f(1.0 - t as f64);
        let sh = if interp { [-v[0] * omt, -v[1] * omt] } else { [v[0] * t, v[1] * t] };
        let x0 = ((bx as f64 * tsz + sh[0] as f64 - m).floor() as i64).max(0);
        let x1 = (((bx + 1) as f64 * tsz + sh[0] as f64 + m).ceil() as i64).min(w as i64);
        let y0 = ((a as f64 * tsz + sh[1] as f64 - m).floor() as i64).max(0);
        let y1 = (((a + 1) as f64 * tsz + sh[1] as f64 + m).ceil() as i64).min(h as i64);
        if x1 > x0 && y1 > y0 {
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y as usize * w + x as usize;
                    vm[i] = v;
                    zm[i] = z;
                }
            }
        }
    }
    (vm, zm)
}
