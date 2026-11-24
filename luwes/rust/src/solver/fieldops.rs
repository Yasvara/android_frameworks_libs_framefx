// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Field-level rewrites before the per-pixel solve: gated median repair, entering-strip
//! extension, the two-motion model for 2-D pans (extrapolation / gained pairs), swept-mover
//! decontamination for interpolation, and the global dominant vector.

use super::layers::{analyse_layers, refine_layer_vectors, round4, Analysis, Layer};
use super::{d, f, tex, Gain, Params};
use crate::cv;
use crate::grid::{hypot, median, Field, Grid, Image, Mask, V2, FIELD_SCALE, FIELD_TEXEL_I};
use crate::layer::LayerField;
use crate::motion::{copy_pad, distinct, pyramid, Level};

fn vmed_n(n: &[V2]) -> V2 {
    let mut best = 0;
    let mut bc = f32::INFINITY;
    for a in 0..n.len() {
        let mut c = 0.0f32;
        for b in n {
            c += (n[a][0] - b[0]).abs();
            c += (n[a][1] - b[1]).abs();
        }
        if c < bc {
            bc = c;
            best = a;
        }
    }
    n[best]
}

/// 5x5 vector-median repair of level-1 texels whose block the median explains clearly better;
/// skipped under an accepted gain, a likely cut, or an all-zero field.
pub fn median_repair(prev: &Image, cur: &Image, fld: &Field, gain: &Gain, cut_r: f32, cut_lo: f64) -> Field {
    let zero = fld.d.iter().all(|t| t[0] == 0.0 && t[1] == 0.0);
    if (gain.scale - 1.0).abs() > 1e-6 || gain.offset.abs() > 1e-6 || cut_r > f(cut_lo) || zero {
        return fld.clone();
    }
    let (h, w) = (cur.h, cur.w);
    let (pc, pp) = (pyramid(cur), pyramid(prev));
    let rpx1 = (h.div_ceil(4), w.div_ceil(4));
    let real = (h.div_ceil(32), w.div_ceil(32));
    let l = Level::new(&pc[1], &pp[1], 8, rpx1);
    let f0 = copy_pad(&fld.map(|t| [t[0], t[1]]), real);
    let (rh, rw) = real;
    let med = Grid::from_fn(f0.w, f0.h, |x, y| {
        let mut win = [[0.0f32; 2]; 25];
        let mut idx = 0;
        for dy in -2i64..=2 {
            let yi = (y as i64 + dy).clamp(0, rh as i64 - 1) as usize;
            for dx in -2i64..=2 {
                let xi = (x as i64 + dx).clamp(0, rw as i64 - 1) as usize;
                win[idx] = f0.at(xi, yi);
                idx += 1;
            }
        }
        vmed_n(&win)
    });
    let e0 = l.sad(&f0);
    let e_med = l.sad(&med);
    let kg = 0.01f32;
    let use_ = Grid::from_fn(f0.w, f0.h, |x, y| e_med.at(x, y) + kg < e0.at(x, y));
    let ff = copy_pad(&Grid::from_fn(f0.w, f0.h, |x, y| if use_.at(x, y) { med.at(x, y) } else { f0.at(x, y) }), real);
    let e = copy_pad(&Grid::from_fn(f0.w, f0.h, |x, y| if use_.at(x, y) { e_med.at(x, y) } else { fld.at(x, y)[2] }), real);
    let mut wv = fld.map(|t| t[3]);
    if use_.any() {
        let still = l.sad_const([0.0, 0.0]);
        let dc = distinct(&l, &ff, &e, 0.1, 0.01);
        let dc = Grid::from_fn(f0.w, f0.h, |x, y| {
            let (s, ev) = (still.at(x, y), e.at(x, y));
            dc.at(x, y).max((2.0 * (s - ev) / (s + 0.001)).clamp(0.0, 1.0))
        });
        wv = copy_pad(&Grid::from_fn(f0.w, f0.h, |x, y| if use_.at(x, y) { dc.at(x, y) } else { wv.at(x, y) }), real);
    }
    Grid::from_fn(f0.w, f0.h, |x, y| {
        let v = ff.at(x, y);
        [v[0], v[1], e.at(x, y), wv.at(x, y)]
    })
}

/// Texels whose displaced prev window leaves the frame (entering strip) take the mean vector of
/// their valid 8-neighbours, propagated until filled.
pub fn extend_entering(fld: &Field, h: usize, w: usize, ov: f64, dom_ext: bool) -> Field {
    let (fh, fw) = (fld.h, fld.w);
    let t = 32.0f64;
    let m_ = ov * t - t / 2.0;
    let cxy = |tx: usize, ty: usize| (tx as f64 * t + t / 2.0, ty as f64 * t + t / 2.0);
    let real = Grid::from_fn(fw, fh, |x, y| {
        let (cx, cy) = cxy(x, y);
        cx < w as f64 && cy < h as f64
    });
    let inside = |px: f64, py: f64| px >= m_ && px <= w as f64 - m_ && py >= m_ && py <= h as f64 - m_;
    let mut valid = Grid::from_fn(fw, fh, |x, y| {
        let (cx, cy) = cxy(x, y);
        let v = fld.at(x, y);
        real.at(x, y) && inside(cx + (v[0] * FIELD_SCALE) as f64, cy + (v[1] * FIELD_SCALE) as f64)
    });
    if dom_ext && real.any() {
        let pick = |c: usize| median(&mut fld.d.iter().zip(&real.d).filter(|(_, &r)| r).map(|(v, _)| v[c] * FIELD_SCALE).collect::<Vec<_>>());
        let bdom = [pick(0), pick(1)];
        valid = Grid::from_fn(fw, fh, |x, y| {
            let (cx, cy) = cxy(x, y);
            valid.at(x, y) && (inside(cx + bdom[0] as f64, cy + bdom[1] as f64) || fld.at(x, y)[2] <= 0.025)
        });
    }
    if !valid.any() {
        return fld.clone();
    }
    let mut v = fld.map(|t| [t[0], t[1]]);
    let mut wc = fld.map(|t| t[3]);
    let mut cv_ = valid.clone();
    let mut frontier: Vec<(usize, usize)> = Vec::new();
    let mut queued = Grid::new(fw, fh, false);
    for y in 0..fh {
        for x in 0..fw {
            if cv_.at(x, y) {
                continue;
            }
            let mut touches = false;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < fw && (ny as usize) < fh && cv_.at(nx as usize, ny as usize) {
                        touches = true;
                    }
                }
            }
            if touches {
                queued.set(x, y, true);
                frontier.push((x, y));
            }
        }
    }
    let mut updates: Vec<(usize, usize, V2, f32)> = Vec::new();
    while !frontier.is_empty() {
        if (0..fw * fh).all(|i| !real.d[i] || cv_.d[i]) {
            break;
        }
        updates.clear();
        for &(x, y) in &frontier {
            let (mut sv, mut sw, mut sm) = ([0.0f32; 2], 0.0f32, 0.0f32);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < fw && (ny as usize) < fh {
                        let (nx, ny) = (nx as usize, ny as usize);
                        if cv_.at(nx, ny) {
                            let vv = v.at(nx, ny);
                            sv[0] += vv[0];
                            sv[1] += vv[1];
                            sw += wc.at(nx, ny);
                            sm += 1.0;
                        }
                    }
                }
            }
            if sm > 0.0 {
                updates.push((x, y, [sv[0] / sm, sv[1] / sm], sw / sm));
            }
        }
        frontier.clear();
        for &(x, y, nv, nw) in &updates {
            v.set(x, y, nv);
            wc.set(x, y, nw);
            cv_.set(x, y, true);
        }
        for &(x, y, _, _) in &updates {
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < fw && (ny as usize) < fh {
                        let (nx, ny) = (nx as usize, ny as usize);
                        if !cv_.at(nx, ny) && !queued.at(nx, ny) {
                            queued.set(nx, ny, true);
                            frontier.push((nx, ny));
                        }
                    }
                }
            }
        }
    }
    Grid::from_fn(fw, fh, |x, y| {
        let o = fld.at(x, y);
        if valid.at(x, y) || !real.at(x, y) {
            o
        } else {
            let vv = v.at(x, y);
            [vv[0], vv[1], o[2], if dom_ext { wc.at(x, y) } else { o[3] }]
        }
    })
}

/// Per-channel gain from unsaturated pixels (medians), else the matcher's scalar gain.
pub fn channel_gain(cur: &Image, prev: &Image, gain: &Gain, f64_gain: bool) -> ([f64; 3], [f32; 3], bool) {
    let g_c = gain.scale.clamp(0.5, 2.0);
    let mx = |c: [f32; 3]| c[0].max(c[1]).max(c[2]);
    let idx: Vec<usize> = (0..cur.d.len()).filter(|&i| mx(cur.d[i]) < 0.96 && mx(prev.d[i]) > 0.05 && mx(prev.d[i]) < 0.72).collect();
    if idx.len() > 1000 {
        let mut g = [0.0f64; 3];
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            let mut r: Vec<f32> = idx.iter().map(|&i| cur.d[i][c] / prev.d[i][c].max(1e-3)).collect();
            let m = median(&mut r) as f64;
            g[c] = if f64_gain { m.clamp(0.5, 2.0) } else { (m as f32).clamp(0.5, 2.0) as f64 };
        }
        for c in 0..3 {
            let v = if f64_gain {
                let mut r: Vec<f64> = idx.iter().map(|&i| cur.d[i][c] as f64 - g[c] * prev.d[i][c] as f64).collect();
                crate::grid::median64(&mut r) as f32
            } else {
                let gc = g[c] as f32;
                let mut r: Vec<f32> = idx.iter().map(|&i| cur.d[i][c] - gc * prev.d[i][c]).collect();
                median(&mut r)
            };
            o[c] = if v.abs() < 0.015 { 0.0 } else { v };
        }
        (g, o, f64_gain)
    } else {
        let gc = g_c as f32 as f64;
        ([gc; 3], [gain.offset as f32; 3], false)
    }
}

pub fn apply_gain(prev: &Image, g: &[f64; 3], o: &[f32; 3], f64_gain: bool) -> Image {
    prev.map(|c| {
        [0, 1, 2].map(|k| {
            if f64_gain {
                (c[k] as f64 * g[k] + o[k] as f64).clamp(0.0, 1.0) as f32
            } else {
                (c[k] * g[k] as f32 + o[k]).clamp(0.0, 1.0)
            }
        })
    })
}

pub struct RegOut {
    pub f: Field,
    pub f2: Option<Field>,
    /// Layer vectors refined here (level px); the layer assignment does not refine them again.
    pub prefit: Vec<V2>,
}

/// Explanation count of a crop under offset u, pulled towards c.
fn crop_score(prev: &Image, cur: &Image, rr: &Mask, x0: usize, y0: usize, tau: f32, u: [f64; 2], c: [f64; 2]) -> f64 {
    let k = (round4(u[0]), round4(u[1]));
    let (k0, k1) = (k.0 as f32, k.1 as f32);
    let mut n = 0usize;
    for y in 0..rr.h {
        for x in 0..rr.w {
            if rr.at(x, y) {
                let (gx, gy) = (x0 + x, y0 + y);
                if d(tex(prev, gx as f32 + k0, gy as f32 + k1), cur.at(gx, gy)) < tau {
                    n += 1;
                }
            }
        }
    }
    n as f64 - 1e-3 * (k.0 - c[0]).hypot(k.1 - c[1])
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

/// Two-motion model of a 2-D pan: the background vector by explanation count around the
/// frame centre, one refined vector per connected compact mover; every texel takes one of them.
pub fn two_motion_pan(prev: &Image, cur: &Image, fld: &Field, f2: Option<&Field>, p: &Params, gain: &Gain) -> RegOut {
    let (h, w) = (cur.h, cur.w);
    let (th, tw) = (h.div_ceil(FIELD_TEXEL_I), w.div_ceil(FIELD_TEXEL_I));
    let prev_use = if gain.accepted() {
        let (g, o, wide) = channel_gain(cur, prev, gain, true);
        apply_gain(prev, &g, &o, wide)
    } else {
        prev.clone()
    };
    let mut b = Grid::from_fn(tw, th, |x, y| {
        let t = fld.at(x, y);
        [t[0], t[1]]
    });
    let mut z = Grid::from_fn(tw, th, |x, y| fld.at(x, y)[2]);
    if let Some(f2) = f2 {
        for y in 0..th {
            for x in 0..tw {
                let t = f2.at(x, y);
                if t[2].is_finite() && t[2] < z.at(x, y) {
                    b.set(x, y, [t[0], t[1]]);
                    z.set(x, y, t[2]);
                }
            }
        }
    }
    let bdom = [median(&mut b.d.iter().map(|v| v[0]).collect::<Vec<_>>()), median(&mut b.d.iter().map(|v| v[1]).collect::<Vec<_>>())];
    let unchanged = || RegOut { f: fld.clone(), f2: f2.cloned(), prefit: Vec::new() };
    if !(bdom[0].abs() >= 1.0 && bdom[1].abs() >= 1.0) {
        return unchanged();
    }
    let rel = b.map(|v| hypot(v[0] - bdom[0], v[1] - bdom[1]));
    let z_cmax = if gain.scale > 1.05 { 0.065f32 } else { 0.035 };
    let mut core = Grid::from_fn(tw, th, |x, y| {
        let v = b.at(x, y);
        let border = x == 0 || y == 0 || x == tw - 1 || y == th - 1;
        let axis = (v[0] - bdom[0]).abs() <= 0.45 || (v[1] - bdom[1]).abs() <= 0.45;
        rel.at(x, y) > 2.2 && z.at(x, y) <= z_cmax && !border && !axis
    });
    if core.count() == 1 {
        let i = core.d.iter().position(|&c| c).unwrap();
        let (cx, cy) = ((i % tw) as i64, (i / tw) as i64);
        let mut best: Option<(usize, usize)> = None;
        let mut br = -1.0f64;
        for (dy, dx) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (cx + dx, cy + dy);
            if nx >= 0 && ny >= 0 && nx < tw as i64 && ny < th as i64 {
                let r = rel.at(nx as usize, ny as usize);
                if r > 2.2 && r as f64 > br {
                    br = r as f64;
                    best = Some((nx as usize, ny as usize));
                }
            }
        }
        if let Some((x, y)) = best {
            core.set(x, y, true);
        }
    }
    let rt = if core.any() { cv::dilate3(&core) } else { Grid::new(tw, th, false) };
    let pick = |c: usize| median(&mut b.d.iter().zip(&rt.d).filter(|(_, &r)| !r).map(|(v, _)| v[c]).collect::<Vec<_>>());
    let b0 = [(pick(0) * FIELD_SCALE) as f64, (pick(1) * FIELD_SCALE) as f64];
    let (cy, cx) = (h / 2, w / 2);
    let (y0, y1, x0, x1) = (cy.saturating_sub(48), (cy + 48).min(h), cx.saturating_sub(64), (cx + 64).min(w));
    let rr = Grid::from_fn(x1 - x0, y1 - y0, |x, y| !rt.at((x0 + x) / FIELD_TEXEL_I, (y0 + y) / FIELD_TEXEL_I));
    let mut cache = std::collections::HashMap::new();
    let mut score_bg = |u: [f64; 2]| -> f64 {
        let key = (round4(u[0]).to_bits(), round4(u[1]).to_bits());
        *cache.entry(key).or_insert_with(|| crop_score(&prev_use, cur, &rr, x0, y0, 0.03, u, b0))
    };
    let mut c_esc = b0;
    let mut g = [[0.0f64; 3]; 3];
    let (mut iy0, mut ix0) = (1usize, 1usize);
    for esc in 0..3 {
        for (iy, dy) in [-0.5, 0.0, 0.5].iter().enumerate() {
            for (ix, dx) in [-0.5, 0.0, 0.5].iter().enumerate() {
                g[iy][ix] = score_bg([c_esc[0] + dx, c_esc[1] + dy]);
            }
        }
        let (iy, ix) = argmax3(&g);
        if g[iy][ix] <= g[1][1] {
            iy0 = 1;
            ix0 = 1;
            break;
        }
        iy0 = iy;
        ix0 = ix;
        if (ix != 1 || iy != 1) && esc < 2 {
            c_esc = [c_esc[0] + (ix as f64 - 1.0) * 0.5, c_esc[1] + (iy as f64 - 1.0) * 0.5];
        } else {
            break;
        }
    }
    let c_star = [c_esc[0] + (ix0 as f64 - 1.0) * 0.5, c_esc[1] + (iy0 as f64 - 1.0) * 0.5];
    let den_x = g[1][1] - g[1][0].min(g[1][2]);
    let dx_eq = if den_x > 1e-6 { (0.25 * (g[1][2] - g[1][0]) / den_x).clamp(-0.25, 0.25) } else { 0.0 };
    let den_y = g[1][1] - g[0][1].min(g[2][1]);
    let dy_eq = if den_y > 1e-6 { (0.25 * (g[2][1] - g[0][1]) / den_y).clamp(-0.25, 0.25) } else { 0.0 };
    let b_cand = [(8.0 * (c_star[0] + dx_eq) + 0.5).floor() / 8.0, (8.0 * (c_star[1] + dy_eq) + 0.5).floor() / 8.0];
    let b_ref_bg = if score_bg(b_cand) > g[1][1] { b_cand } else { [(8.0 * c_star[0] + 0.5).floor() / 8.0, (8.0 * c_star[1] + 0.5).floor() / 8.0] };
    let mut prefit = Vec::new();
    let mut comps: Vec<(Mask, [f64; 2])> = Vec::new();
    let mut v_ref_obj = b_ref_bg;
    if core.any() {
        let cmasks: Vec<Mask> = if p.per_mover_regularise {
            let (n, lab) = cv::connected_components(&core, true);
            (1..n as i32).map(|i| lab.map(|l| l == i)).collect()
        } else {
            vec![core.clone()]
        };
        let to_field = |m: &Mask| Grid::from_fn(fld.w, fld.h, |x, y| x < tw && y < th && m.at(x, y));
        let layers: Vec<Layer> = cmasks.iter().map(|mk| {
            let pk = |c: usize| median(&mut b.d.iter().zip(&mk.d).filter(|(_, &k)| k).map(|(v, _)| v[c]).collect::<Vec<_>>());
            let m = to_field(mk);
            Layer { s: m.clone(), m, v: [pk(0), pk(1)], cov: false, unc: false, n: 0, fl_seed: false, fl_dim: false, prefit: false, expl: None, thin: None }
        }).collect();
        let zf = Grid::from_fn(fld.w, fld.h, |x, y| if x < tw && y < th { z.at(x, y) } else { f32::INFINITY });
        let _ = zf;
        let mut a = Analysis { bdom: [(b_ref_bg[0] / FIELD_SCALE as f64) as f32, (b_ref_bg[1] / FIELD_SCALE as f64) as f32], layers };
        refine_layer_vectors(&prev_use, cur, &mut a, p);
        for (mk, l) in cmasks.into_iter().zip(&a.layers) {
            comps.push((mk, [(l.v[0] * FIELD_SCALE) as f64, (l.v[1] * FIELD_SCALE) as f64]));
            prefit.push(l.v);
        }
        v_ref_obj = comps[0].1;
    }
    let mut vobj = Grid::new(tw, th, v_ref_obj);
    for (mk, vc) in &comps {
        for i in 0..mk.d.len() {
            if mk.d[i] {
                vobj.d[i] = *vc;
            }
        }
    }
    let s4 = FIELD_SCALE as f64;
    let out = Grid::from_fn(fld.w, fld.h, |x, y| {
        let inr = y < th && x < tw;
        if inr && core.at(x, y) {
            let v = vobj.at(x, y);
            [(v[0] / s4) as f32, (v[1] / s4) as f32, 0.010, 1.0]
        } else {
            let zz = if inr && rt.at(x, y) && z.at(x, y) > 0.020 { 0.025 } else { 0.005 };
            [(b_ref_bg[0] / s4) as f32, (b_ref_bg[1] / s4) as f32, zz, 1.0]
        }
    });
    let f2o = f2.map(|f2| f2.map(|t| [t[0], t[1], f32::INFINITY, t[3]]));
    RegOut { f: out, f2: f2o, prefit }
}

/// One compact fast mover in an interpolated pair: its refined vector on its footprint, the
/// background vector elsewhere (no second hypotheses).
pub fn swept_mover(prev: &Image, cur: &Image, fld: &Field, f2: Option<&Field>, p: &Params, fl: Option<&LayerField>) -> Option<Field> {
    let (h, w) = (cur.h, cur.w);
    let (th, tw) = (h.div_ceil(FIELD_TEXEL_I), w.div_ceil(FIELD_TEXEL_I));
    let mut a = analyse_layers(fld, f2, p, false, fl, (w, h));
    if a.layers.len() != 1 {
        return None;
    }
    {
        let l0 = &a.layers[0];
        if hypot(l0.v[0] - a.bdom[0], l0.v[1] - a.bdom[1]) < 5.0 || l0.s.count() > 8 {
            return None;
        }
    }
    refine_layer_vectors(prev, cur, &mut a, p);
    let l0 = &a.layers[0];
    let v_ref = l0.v;
    let bdom = a.bdom;
    let fast = hypot(v_ref[0], v_ref[1]) > hypot(bdom[0], bdom[1]);
    let use_m = if fast { &l0.m } else { &l0.s };
    Some(Grid::from_fn(fld.w, fld.h, |x, y| {
        let inr = y < th && x < tw;
        if inr && use_m.at(x, y) {
            [v_ref[0], v_ref[1], 0.005, 1.0]
        } else {
            let zz = if !fast && inr && l0.m.at(x, y) { 0.025 } else { 0.005 };
            [bdom[0], bdom[1], zz, 1.0]
        }
    }))
}

/// Global dominant back vector: field median refined by explanation count on a 4x4 lattice,
/// 3x3 then compass at 1, 1/2, 1/4, 1/8 px.
pub fn dominant_vector(prev: &Image, cur: &Image, fld: &Field, p: &Params) -> V2 {
    let (h, w) = (cur.h, cur.w);
    let (rh, rw) = (h.div_ceil(FIELD_TEXEL_I), w.div_ceil(FIELD_TEXEL_I));
    let mut vx = Vec::new();
    let mut vy = Vec::new();
    for y in 0..rh.min(fld.h) {
        for x in 0..rw.min(fld.w) {
            vx.push(fld.at(x, y)[0]);
            vy.push(fld.at(x, y)[1]);
        }
    }
    let v0 = [(median(&mut vx) as f64) * FIELD_SCALE as f64, (median(&mut vy) as f64) * FIELD_SCALE as f64];
    let v0 = [(8.0 * v0[0] + 0.5).floor() / 8.0, (8.0 * v0[1] + 0.5).floor() / 8.0];
    let sub = 4usize;
    let tau = f(p.layer_agree_tau);
    let mut cache = std::collections::HashMap::new();
    let mut score = |u: [f64; 2]| -> f64 {
        let k = (round4(u[0]), round4(u[1]));
        *cache.entry((k.0.to_bits(), k.1.to_bits())).or_insert_with(|| {
            let (k0, k1) = (k.0 as f32, k.1 as f32);
            let mut n = 0usize;
            let mut y = sub / 2;
            while y < h {
                let mut x = sub / 2;
                while x < w {
                    if d(tex(prev, x as f32 + k0, y as f32 + k1), cur.at(x, y)) < tau {
                        n += 1;
                    }
                    x += sub;
                }
                y += sub;
            }
            n as f64 - 1e-3 * (k.0 - v0[0]).hypot(k.1 - v0[1])
        })
    };
    let mut c = v0;
    for st in [1.0, 0.5, 0.25, 0.125] {
        for _ in 0..4 {
            let mut best = (0i32, 0i32);
            let mut bv = f64::NEG_INFINITY;
            for dx in -1i32..=1 {
                for dy in -1i32..=1 {
                    if !(dx == 0 || dy == 0 || st >= 1.0) {
                        continue;
                    }
                    let s = score([c[0] + dx as f64 * st, c[1] + dy as f64 * st]);
                    if s > bv {
                        bv = s;
                        best = (dx, dy);
                    }
                }
            }
            if best == (0, 0) {
                break;
            }
            c = [c[0] + best.0 as f64 * st, c[1] + best.1 as f64 * st];
        }
    }
    [c[0] as f32, c[1] as f32]
}
