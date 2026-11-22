// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Per-pixel decision on the prepared candidates: admission, two-sided choice with a Potts pass,
//! occluder / mover / collision / shield rules, static lock, layer bands, edge and horizon
//! overrides, gain phase and the scene-cut blend.

use super::layers::grad;
use super::prep::Prepared;
use super::{d, f, sat01, tex, Params};
use crate::cv;
use crate::grid::{catmull_rom, hypot, q8, Grid, Image, Rgb, FIELD_SCALE};

type Rgb64 = [f64; 3];

/// The reference's smoothstep with Python-float edges: x - e0 in float32, the rest in float64.
#[inline]
fn ss(e0: f64, e1: f64, x: f32) -> f64 {
    let tt = (((x - e0 as f32) as f64) / (e1 - e0).max(1e-9)).clamp(0.0, 1.0);
    tt * tt * (3.0 - 2.0 * tt)
}

#[inline]
fn sb(x: f32, tau: f64, soft: f64) -> f64 {
    1.0 - ss(tau * (1.0 - soft), tau * (1.0 + soft), x)
}

#[inline]
fn up(a: Rgb) -> Rgb64 {
    a.map(|v| v as f64)
}

/// a + (b - a) w with b - a in float32 and the weight in float64.
#[inline]
fn mix64(a: Rgb, b: Rgb, w: f64) -> Rgb64 {
    [0, 1, 2].map(|c| a[c] as f64 + (b[c] - a[c]) as f64 * w)
}

#[inline]
fn mix6(a: Rgb64, b: Rgb64, w: f64) -> Rgb64 {
    [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * w)
}

/// A solved frame.
pub struct Solved {
    /// Generated frame (8-bit values).
    pub out: Image,
    /// Winner's two-sided cost (0 on locked / static pixels): the fusion cue.
    pub warp_error: Grid<f32>,
}

#[inline]
fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * t)
}

fn shielded(p: &Params, now: Rgb, before: Rgb, fb: Rgb, c0: Rgb, c1: Rgb, e: f32, s1: f32, s0: f32, wnom: f32) -> Rgb {
    let nominal = mix(c0, c1, wnom);
    let den = f(p.shield_rho) * e + f(p.noise_tau);
    let st1 = (1.0 - s1 / den).clamp(0.0, 1.0);
    let st0 = (1.0 - s0 / den).clamp(0.0, 1.0);
    let o1 = st1 * (1.0 - st0);
    let o0 = st0 * (1.0 - st1);
    let bb = st1 * st0;
    let nn = (1.0 - st1) * (1.0 - st0);
    let unmix = |t: Rgb, wc: Rgb| -> Rgb {
        let dd = [wc[0] - t[0], wc[1] - t[1], wc[2] - t[2]];
        let s3 = |a: [f32; 3]| ((0.0 + a[0]) + a[1]) + a[2];
        let na = s3([0, 1, 2].map(|c| (now[c] - t[c]) * dd[c]));
        let nb = s3([0, 1, 2].map(|c| (before[c] - t[c]) * dd[c]));
        let den = s3([0, 1, 2].map(|c| dd[c] * dd[c])) + 1e-4;
        let beta = (0.5 * (na + nb) / den).clamp(0.0, 1.0);
        [0, 1, 2].map(|c| (1.0 - beta) * t[c] + beta * wc[c])
    };
    let u1 = unmix(c1, c0);
    let u0 = unmix(c0, c1);
    [0, 1, 2].map(|c| o1 * u1[c] + o0 * u0[c] + bb * fb[c] + nn * nominal[c])
}

/// The decision rules on prepared candidates.
pub fn decide(z: &Prepared, p: &Params) -> Solved {
    let (w, h, t, interp, k) = (z.w, z.h, z.t, z.interp, z.k);
    let n = z.n();
    let cur = &z.cur;
    if (interp && t >= 1.0) || (!interp && t <= 0.0) {
        return Solved { out: cur.clone(), warp_error: Grid::new(w, h, 0.0) };
    }
    let omt = f(1.0 - t as f64);
    let at = |kk: usize, j: usize| kk * n + j;
    let fb: Vec<Rgb> = (0..n).map(|j| if interp { mix(z.before[j], z.now[j], t) } else { z.now[j] }).collect();
    let lay = z.layer_assign.as_ref();
    let band_on = lay.is_some() && !z.cut_guard;
    let te = f(p.agree_tau);
    let mut ws = vec![0.0f32; n];
    let mut early = vec![false; n];
    let mut eo = vec![false; n];
    let (fta, fto) = (f(p.arrival_tau), f(p.overlay_tau));
    for j in 0..n {
        eo[j] = z.arrival_test[j] && (z.arrival_res[j] < fta || z.overlay_res[j] > fto);
        let mut wsj = 1.0 - ss(2.0 / 255.0, p.lock_tau, z.qdp[j]);
        let ov = ss(p.overlay_tau / 2.0, p.overlay_tau, z.overlay_res[j]) * (1.0 - ss(p.overlay_own_tau / 2.0, p.overlay_own_tau, z.dp[j]));
        let fv = (1.0 - ss(p.arrival_tau, 2.0 * p.arrival_tau, z.arrival_res[j])).max(ov);
        if z.moving[j] {
            wsj *= fv;
        }
        ws[j] = wsj as f32;
        early[j] = ws[j] >= 1.0 || eo[j];
    }
    if p.dense_unlock_tau > 0.0 && z.dense {
        for j in 0..n {
            let b0 = z.b[at(0, j)];
            if z.e[at(0, j)] < f(p.dense_unlock_tau) && hypot(b0[0], b0[1]) >= 0.5 {
                ws[j] = 0.0;
                early[j] = false;
            }
        }
    }
    let e = &z.e;
    let (tv, z_ok, tau_a) = (f(p.spread_tau), f(p.trusted_cost), f(p.distinct_tau));
    let mut phase2 = vec![false; n];
    let mut valid = z.kept.clone();
    let mut score = vec![0.0f32; k * n];
    for j in 0..n {
        let mut vt = z.var[j] > tv;
        if interp {
            vt &= !(e[at(0, j)] < f(0.5 * p.agree_tau) && z.sup[at(0, j)] >= 7);
        }
        phase2[j] = e[at(0, j)] > te || vt || z.lay[j];
        let lp = z.lay[j] && !(e[at(0, j)] > te || z.var[j] > tv);
        for kk in 0..k {
            let o = at(kk, j);
            if kk >= 1 {
                valid[o] &= phase2[j];
            }
            if kk >= 2 {
                valid[o] &= z.sup[o] as i32 >= p.min_support || z.zw[o][0] <= z_ok;
                valid[o] &= z.zw[o][1] > tau_a || z.zw[o][0] <= z_ok;
            }
            if lp && kk != 0 && Some(kk) != z.lay_k {
                valid[o] = false;
            }
            score[o] = if valid[o] { e[o] + f(p.smooth_weight) * z.lv[o] } else { f32::INFINITY };
        }
    }
    let argmin = |j: usize, s: &[f32]| {
        let mut bi = 0;
        for kk in 1..k {
            if s[at(kk, j)] < s[at(bi, j)] {
                bi = kk;
            }
        }
        bi
    };
    let mut kmin: Vec<usize> = (0..n).map(|j| argmin(j, &score)).collect();
    let kmax: Vec<usize> = (0..n).map(|j| {
        let mut bi = 0;
        let mut bv = f32::NEG_INFINITY;
        for kk in 0..k {
            let v = if valid[at(kk, j)] { e[at(kk, j)] } else { f32::NEG_INFINITY };
            if v > bv {
                bv = v;
                bi = kk;
            }
        }
        bi
    }).collect();
    if p.potts_weight > 0.0 {
        let mut bimg = vec![[f32::NAN; 2]; w * h];
        for (j, &i) in z.idx.iter().enumerate() {
            bimg[i] = if eo[j] { [f32::NAN; 2] } else { z.b[at(kmin[j], j)] };
        }
        let (pr, pl) = (f(p.potts_radius), f(p.potts_weight));
        for (j, &i) in z.idx.iter().enumerate() {
            let (ix, iy) = ((i % w) as i64, (i / w) as i64);
            for kk in 0..k {
                let bk = z.b[at(kk, j)];
                let mut nk = 0.0f32;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (ix + dx, iy + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let q = bimg[ny as usize * w + nx as usize];
                        if hypot(bk[0] - q[0], bk[1] - q[1]) / FIELD_SCALE < pr {
                            nk += 1.0;
                        }
                    }
                }
                score[at(kk, j)] -= pl * nk;
            }
        }
        kmin = (0..n).map(|j| argmin(j, &score)).collect();
    }
    let g13 = z.gain;
    let no_gain = (g13.0 - 1.0).abs() < 1e-3 && g13.1.abs() < 1e-3;
    let mut winner_err: Vec<f32> = (0..n).map(|j| e[at(kmin[j], j)]).collect();
    let mut bm: Vec<[f32; 2]> = (0..n).map(|j| z.b[at(kmin[j], j)]).collect();
    if p.zero_motion_margin > 0.0 && interp && no_gain {
        let gq = grad(cur);
        for j in 0..n {
            let es = d(z.now[j], z.before[j]);
            let mut zs = es <= winner_err[j] + f(p.zero_motion_margin) && es < f(p.zero_motion_tau) && hypot(bm[j][0], bm[j][1]) > 0.5;
            let g = gq.d[z.idx[j]];
            zs &= g >= f(p.zero_motion_grad) && es < f(p.zero_motion_ratio) * g;
            zs &= z.sup[at(kmin[j], j)] < p.zero_motion_max_support;
            ws[j] = ws[j].max(zs as u8 as f32);
        }
    }
    if p.extrap_unsupported_to_base && !interp {
        let tc = f(0.5 * p.agree_tau);
        for j in 0..n {
            let km = kmin[j];
            let b0 = z.b[at(0, j)];
            let cand = km != 0 && z.sup[at(km, j)] <= p.extrap_unsupported_support && z.sup[at(0, j)] >= 7 && valid[at(0, j)] && !early[j] && hypot(bm[j][0] - b0[0], bm[j][1] - b0[1]) > FIELD_SCALE;
            if !cand {
                continue;
            }
            let i = z.idx[j];
            let (xq, yq) = ((i % w) as f64, (i / w) as f64);
            let b = bm[j];
            let g1 = d(z.c1[at(km, j)], tex(&z.prev, (xq + (t * b[0]) as f64 + b0[0] as f64) as f32, (yq + (t * b[1]) as f64 + b0[1] as f64) as f32)) < tc;
            let g0 = d(z.c1[at(0, j)], tex(&z.prev, (xq + (t * b0[0]) as f64 + b[0] as f64) as f32, (yq + (t * b0[1]) as f64 + b[1] as f64) as f32)) >= tc;
            if g1 && g0 {
                kmin[j] = 0;
                winner_err[j] = e[at(0, j)];
                bm[j] = b0;
            }
        }
    }
    let c1m: Vec<Rgb> = (0..n).map(|j| z.c1[at(kmin[j], j)]).collect();
    let c0m: Vec<Rgb> = (0..n).map(|j| z.c0[at(kmin[j], j)]).collect();
    let mut cr_d: Option<Vec<Rgb>> = None;
    if p.cubic_output && interp {
        cr_d = Some((0..n).map(|j| {
            let i = z.idx[j];
            let (ix, iy) = ((i % w) as f64, (i / w) as f64);
            let b = bm[j];
            let c1r = catmull_rom(cur, (ix - (omt * b[0]) as f64) as f32, (iy - (omt * b[1]) as f64) as f32);
            let c0r = catmull_rom(&z.prev, (ix + (t * b[0]) as f64) as f32, (iy + (t * b[1]) as f64) as f32);
            [0, 1, 2].map(|c| {
                let a = c0r[c] - c0m[j][c];
                a + ((c1r[c] - c1m[j][c]) - a) * t
            })
        }).collect());
    }
    if p.silhouette_unlock && interp && band_on {
        let fz = lay.unwrap();
        if fz.has_layers && p.ground_coincidence {
            let mut c3 = Grid::new(w, h, false);
            for (j, &i) in z.idx.iter().enumerate() {
                c3.d[i] = fz.code[j] == 3;
            }
            if let Some((m, _)) = &fz.c3 {
                for i in 0..w * h {
                    c3.d[i] |= m[i];
                }
            }
            let near = cv::dilate(&c3, &cv::ellipse(2 * p.silhouette_radius + 1));
            let bgv = fz.bdom;
            for j in 0..n {
                let mut onl = false;
                for v in &fz.lv {
                    if hypot(v[0] - bgv[0], v[1] - bgv[1]) >= 1.0 {
                        onl |= hypot(bm[j][0] - v[0], bm[j][1] - v[1]) <= 1.0;
                    }
                }
                if near.d[z.idx[j]] && fz.code[j] <= 0 && onl && winner_err[j] < te {
                    ws[j] = 0.0;
                    early[j] = false;
                }
            }
        }
    }
    let mut agree: Vec<f64> = winner_err.iter().map(|&v| sb(v, p.agree_tau, p.soft_width)).collect();
    let mut out1: Vec<Rgb> = (0..n).map(|j| if interp { mix(c0m[j], c1m[j], t) } else { c1m[j] }).collect();
    if p.fuse_temperature > 0.0 {
        let (ft, fr) = (f(p.fuse_temperature), f(p.fuse_radius));
        for j in 0..n {
            let smin = score[at(kmin[j], j)];
            let mut acc = [0.0f32; 3];
            let mut ws_ = 0.0f32;
            for kk in 0..k {
                let o = at(kk, j);
                let mut wk = if valid[o] { (-(score[o] - smin) / ft).exp() } else { 0.0 };
                let bv = z.b[o];
                if !(hypot(bv[0] - bm[j][0], bv[1] - bm[j][1]) / FIELD_SCALE <= fr) {
                    wk *= 0.0;
                }
                let wkc = if interp { mix(z.c0[o], z.c1[o], t) } else { z.c1[o] };
                for c in 0..3 {
                    acc[c] += wk * wkc[c];
                }
                ws_ += wk;
            }
            out1[j] = acc.map(|a| a / ws_);
        }
    }
    let mut best_q = vec![0.0f64; n];
    let mut best_c: Option<Vec<Rgb>> = None;
    if !interp {
        let mut best_k = vec![0usize; n];
        let mut best_key = vec![-1.0f32; n];
        for j in 0..n {
            let rf = z.key[at(kmin[j], j)] + 0.25;
            for kk in 0..k {
                let o = at(kk, j);
                let mut q = sb(e[o], p.agree_tau, p.soft_width) * (1.0 - sb(d(z.c1[o], c1m[j]), p.agree_tau, p.soft_width)) * ((z.key[o] > rf) as u8 as f64) * (valid[o] as u8 as f64);
                let mut of = 1.0 - ss(0.5, 1.5, z.r1[o]);
                if p.mover_pixel_ownership && q > 0.5 && of < 0.5 {
                    let i = z.idx[j];
                    let (xq, yq) = ((i % w) as f64, (i / w) as f64);
                    let (bk, bw) = (z.b[o], bm[j]);
                    let x1x = xq + (bk[0] * t) as f64;
                    let x1y = yq + (bk[1] * t) as f64;
                    let cp = tex(&z.prev, (x1x + bw[0] as f64) as f32, (x1y + bw[1] as f64) as f32);
                    of = of.max(ss(0.5 * p.agree_tau, p.agree_tau, d(z.c1[o], cp)));
                }
                q *= of;
                if q > 0.5 && z.key[o] > best_key[j] {
                    best_key[j] = z.key[o];
                    best_q[j] = q;
                    best_k[j] = kk;
                }
            }
        }
        best_c = Some((0..n).map(|j| z.c1[at(best_k[j], j)]).collect());
    }
    let mut coll = vec![0.0f64; n];
    let kc = p.collision_contrast;
    for j in 0..n {
        let other = d(c1m[j], z.now[j]).min(d(c1m[j], z.before[j]));
        let gate = ss(0.5 * kc, 1.5 * kc, other / (z.dp[j] + 1e-4));
        let den = f(p.collision_rho) * z.dp[j] + f(p.noise_tau);
        let s1m = z.s1[at(kmin[j], j)];
        let s0m = z.s0[at(kmin[j], j)];
        let still = (1.0 - s1m / den).clamp(0.0, 1.0) * (1.0 - s0m / den).clamp(0.0, 1.0);
        coll[j] = ss(p.collision_tau * 0.5, (p.collision_tau * 1.5).min(1.0), still) * gate;
    }
    let tnom = if interp { t } else { 1.0 };
    let trust_supported = p.extrap_trust_supported && !interp;
    let mut res = vec![[0.0f64; 3]; n];
    let mut err_map = Grid::new(w, h, 0.0f32);
    for j in 0..n {
        let (km, kx) = (kmin[j], kmax[j]);
        let (om, ox) = (at(km, j), at(kx, j));
        let outm = shielded(p, z.now[j], z.before[j], fb[j], c0m[j], c1m[j], e[om], z.s1[om], z.s0[om], tnom);
        let o1: Rgb64 = match &best_c {
            Some(bc) => mix64(out1[j], bc[j], best_q[j]),
            None => up(out1[j]),
        };
        let (c1x, c0x) = (z.c1[ox], z.c0[ox]);
        let v1 = d(c1x, c1m[j]) + d(c1x, c0m[j]);
        let v0 = d(c0x, c1m[j]) + d(c0x, c0m[j]);
        let w1 = (0.5 + f(p.occluder_gain) * (v0 - v1) / (v0 + v1 + 1e-4)).clamp(0.0, 1.0);
        let occ = ss(1.5, 3.0, e[ox] / (winner_err[j] + 1.0 / 255.0)) * ((kx != km) as u8 as f64);
        let out_mx = shielded(p, z.now[j], z.before[j], fb[j], c0x, c1x, e[ox], z.s1[ox], z.s0[ox], w1);
        let out3 = mix64(outm, out_mx, occ);
        let conf = sat01(z.fp[j][3]) as f64;
        let out3 = [0, 1, 2].map(|c| fb[j][c] as f64 + (out3[c] - fb[j][c] as f64) * conf);
        let (ag, ac) = if trust_supported {
            let sure = z.sup[om] >= p.extrap_trusted_support && z.r1[om] <= f(p.extrap_trusted_own);
            let a32 = if sure { 1.0 } else { agree[j] as f32 };
            let c32 = if sure { 0.0 } else { coll[j] as f32 };
            agree[j] = a32 as f64;
            (a32 as f64, (a32 * c32) as f64)
        } else {
            (agree[j], agree[j] * coll[j])
        };
        let mut r = mix6(out3, o1, ag);
        if let Some(cr) = &cr_d {
            r = [0, 1, 2].map(|c| r[c] + cr[j][c] as f64 * ag);
        }
        r = mix6(r, up(fb[j]), ac);
        r = mix6(r, up(fb[j]), ws[j] as f64);
        res[j] = r;
    }
    let mut band_px = vec![false; n];
    if band_on {
        let fz = lay.unwrap();
        for j in 0..n {
            band_px[j] = fz.code[j] > 0 && (!early[j] || fz.code[j] == 3);
            if p.dense_band_tau > 0.0 && z.dense && (fz.code[j] == 1 || fz.code[j] == 2) && e[at(0, j)] < f(p.dense_band_tau) {
                band_px[j] = false;
            }
        }
    }
    for j in 0..n {
        if early[j] {
            res[j] = up(if interp { fb[j] } else { z.now[j] });
        }
        err_map.d[z.idx[j]] = if early[j] { 0.0 } else { winner_err[j] };
    }
    if !interp && !z.cut_guard {
        if let Some((hor0, c1e)) = &z.horizon {
            for j in 0..n {
                if hor0[j] && !early[j] && z.sup[at(0, j)] >= 5 {
                    res[j] = up(c1e[j]);
                }
            }
        }
    }
    if band_on {
        let fz = lay.unwrap();
        for j in 0..n {
            if band_px[j] {
                res[j] = up(fz.col[j]);
            }
        }
        if p.ground_coincidence && interp && fz.has_layers && no_gain {
            let bg = fz.bdom;
            let (tec, teo) = (f(p.ground_coincidence_tau), f(p.ground_occupancy_tau));
            for j in 0..n {
                if early[j] || band_px[j] || !(hypot(bm[j][0] - bg[0], bm[j][1] - bg[1]) > 1.0) || !(winner_err[j] > f(p.ground_coincidence_min_cost)) {
                    continue;
                }
                if p.ground_coincidence_layer_only {
                    let onl = fz.lv.iter().any(|v| hypot(v[0] - bg[0], v[1] - bg[1]) >= 1.0 && hypot(bm[j][0] - v[0], bm[j][1] - v[1]) <= 1.0);
                    if !onl {
                        continue;
                    }
                }
                let i = z.idx[j];
                let (xq, yq) = ((i % w) as f64, (i / w) as f64);
                let b = bm[j];
                let (x1x, x1y) = (xq - (omt * b[0]) as f64, yq - (omt * b[1]) as f64);
                let (x0x, x0y) = (xq + (t * b[0]) as f64, yq + (t * b[1]) as f64);
                let g1 = d(c1m[j], tex(&z.prev, (x1x + bg[0] as f64) as f32, (x1y + bg[1] as f64) as f32)) < tec;
                let g0 = d(c0m[j], tex(cur, (x0x - bg[0] as f64) as f32, (x0y - bg[1] as f64) as f32)) < tec;
                if !(if p.ground_coincidence_one_end { g1 || g0 } else { g1 && g0 }) {
                    continue;
                }
                let (xb1, yb1) = ((xq - (omt * bg[0]) as f64) as f32, (yq - (omt * bg[1]) as f64) as f32);
                let (xb0, yb0) = ((xq + (t * bg[0]) as f64) as f32, (yq + (t * bg[1]) as f64) as f32);
                let cb1 = tex(cur, xb1, yb1);
                let cb0 = tex(&z.prev, xb0, yb0);
                let (mut occ1, mut occ0) = (false, false);
                for v in &fz.lv {
                    if hypot(v[0] - bg[0], v[1] - bg[1]) < 1.0 {
                        continue;
                    }
                    occ1 |= d(cb1, tex(&z.prev, xb1 + v[0], yb1 + v[1])) < teo;
                    occ0 |= d(cb0, tex(cur, xb0 - v[0], yb0 - v[1])) < teo;
                }
                let mut use0 = occ1 && !occ0;
                let mut use1 = occ0 && !occ1;
                if p.paired_rescue == 1 {
                    use0 &= g1;
                    use1 &= g0;
                }
                if use0 {
                    res[j] = up(cb0);
                }
                if use1 {
                    res[j] = up(cb1);
                }
            }
        }
    }
    if interp {
        if let Some((be, c1e, c0e)) = &z.edge {
            let m_ = p.edge_margin;
            let (wf, hf) = ((w - 1) as f64, (h - 1) as f64);
            for j in 0..n {
                let i = z.idx[j];
                let (ix, iy) = ((i % w) as f64, (i / w) as f64);
                let bb = be[j];
                let (x1, y1) = (ix - (bb[0] * omt) as f64, iy - (bb[1] * omt) as f64);
                let (x0, y0) = (ix + (bb[0] * t) as f64, iy + (bb[1] * t) as f64);
                let o1 = x1 < -m_ || x1 > wf + m_ || y1 < -m_ || y1 > hf + m_;
                let o0 = x0 < -m_ || x0 > wf + m_ || y0 < -m_ || y0 > hf + m_;
                let okv = z.sup[at(0, j)] >= p.edge_min_support && !band_px[j];
                if o1 && !o0 && !early[j] && okv {
                    res[j] = up(c0e[j]);
                } else if o0 && !o1 && !early[j] && okv {
                    res[j] = up(c1e[j]);
                }
            }
        }
    }
    if !interp {
        for j in 0..n {
            if (0..k).all(|kk| z.lb[at(kk, j)] == 0.0) {
                res[j] = up(z.now[j]);
            }
        }
    }
    let gain_out = z.gain_phase.as_ref().filter(|_| !z.cut_guard);
    let mut out: Vec<Rgb> = if interp {
        let prev_sb: Image = match gain_out {
            Some(g) => z.prev.map(|c| [0, 1, 2].map(|k| (c[k] * g.gch[k] + g.och[k]).clamp(0.0, 1.0))),
            None => z.prev.clone(),
        };
        (0..w * h).map(|i| mix(prev_sb.d[i], cur.d[i], t)).collect()
    } else {
        cur.d.clone()
    };
    for (j, &i) in z.idx.iter().enumerate() {
        out[i] = res[j].map(|v| v as f32);
    }
    if band_on {
        if let Some((m, col)) = &lay.unwrap().c3 {
            for i in 0..w * h {
                if m[i] {
                    out[i] = col[i];
                }
            }
        }
    }
    if let Some(g) = gain_out {
        let mean = ((0.0 + g.gch[0]) + g.gch[1] + g.gch[2]) / 3.0;
        let tph = if interp { t as f64 } else if mean < 0.95 { 1.0 + t as f64 } else { 1.0 };
        let tp = f(tph);
        let s_t = g.gch.map(|v| 1.0 + tp * (v - 1.0));
        for i in 0..w * h {
            out[i] = [0, 1, 2].map(|c| ((s_t[c] / g.gch[c].max(1e-4)) * (out[i][c] - g.och[c]) + tp * g.och[c]).clamp(0.0, 1.0));
        }
        if interp {
            if let (Some(c0eb), Some(c1eb)) = (&g.c0eb, &g.c1eb) {
                for i in 0..w * h {
                    let mx = c1eb[i][0].max(c1eb[i][1]).max(c1eb[i][2]);
                    let sat1 = ss(0.96, 0.995, mx);
                    if sat1 > 0.0 {
                        let phi = [0, 1, 2].map(|c| (s_t[c] * c0eb[i][c] + tp * g.och[c]).clamp(0.0, 1.0));
                        out[i] = [0, 1, 2].map(|c| ((1.0 - sat1) * out[i][c] as f64 + sat1 * phi[c] as f64) as f32);
                    }
                }
            }
        }
    }
    let mut out: Vec<Rgb> = out.iter().map(|c| c.map(q8)).collect();
    if z.cut_pi > 0.0 {
        let (pa, pb) = (f(1.0 - z.cut_pi), f(z.cut_pi));
        for i in 0..w * h {
            let y = if interp { mix(z.prev.d[i], cur.d[i], t) } else { cur.d[i] };
            out[i] = [0, 1, 2].map(|c| q8(pa * out[i][c] + pb * y[c]));
        }
    }
    Solved { out: Grid::from_vec(w, h, out), warp_error: err_map }
}
