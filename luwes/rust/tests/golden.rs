// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Golden tests against the Python reference (lab/luwes): matcher, layer fields,
//! both solvers on the reference's own fields, the whole pipeline, and the fusion fallback.
//! Fixtures: tests/golden/luwes (small frames, lab/luwes/framefx_golden/dump.py --small); point
//! LUWES_GOLDEN at the lab's full-size set for the long run.

use luwes_fg::fusion::{self, FusionModel, Tensor};
use luwes_fg::grid::{Field, Grid, Image, V2};
use luwes_fg::layer::{layer_field, LayerField, LayerParams};
use luwes_fg::motion::{estimate, MatchParams};
use luwes_fg::npy::Npy;
use luwes_fg::pipeline::{DenseFlow, Pair};
use luwes_fg::solver::decide::decide;
use luwes_fg::solver::prep::{prepare, PairInputs};
use luwes_fg::solver::{Gain, Mode, Params};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    if let Ok(p) = std::env::var("LUWES_GOLDEN") {
        return PathBuf::from(p);
    }
    // soong installs the data next to the test binary; the lab's cargo harness sits beside the tree
    let mut c = vec![PathBuf::from("tests/golden/luwes")];
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
        c.push(d.join("tests/golden/luwes"));
    }
    if let Some(m) = option_env!("CARGO_MANIFEST_DIR") {
        c.push(PathBuf::from(m).join("../../../../src/frameworks/libs/framefx/tests/golden/luwes"));
    }
    c.into_iter().find(|p| p.exists()).expect("golden fixtures not found (set LUWES_GOLDEN)")
}

fn pairs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root()).expect("golden dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("pair.json").exists()).collect();
    v.sort();
    v
}

fn npy(p: &Path, name: &str) -> Npy {
    Npy::load(&p.join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("{}: {e}", p.join(name).display()))
}

fn image(p: &Path, name: &str) -> Image {
    let n = npy(p, name);
    let (h, w) = (n.shape[0], n.shape[1]);
    let v = n.u8();
    Grid::from_fn(w, h, |x, y| {
        let i = (y * w + x) * 3;
        [v[i] as f32 / 255.0, v[i + 1] as f32 / 255.0, v[i + 2] as f32 / 255.0]
    })
}

fn field(p: &Path, name: &str) -> Field {
    let n = npy(p, name);
    let (h, w) = (n.shape[0], n.shape[1]);
    let v = n.f32();
    Grid::from_fn(w, h, |x, y| {
        let i = (y * w + x) * 4;
        [v[i], v[i + 1], v[i + 2], v[i + 3]]
    })
}

fn json_num(p: &Path, key: &str) -> f64 {
    let s = std::fs::read_to_string(p.join("pair.json")).unwrap();
    let t = s.split(&format!("\"{key}\":")).nth(1).unwrap();
    let t = t.trim_start().trim_start_matches('[').trim_start();
    t.split([',', '\n', ']']).next().unwrap().trim().parse().unwrap()
}

/// The matcher's accepted gain (g, o) recorded with the pair.
fn json_gain(p: &Path) -> (f64, f64) {
    let s = std::fs::read_to_string(p.join("pair.json")).unwrap();
    let t = s.split("\"gain\":").nth(1).unwrap();
    let t = &t[t.find('[').unwrap() + 1..t.find(']').unwrap()];
    let v: Vec<f64> = t.split(',').map(|x| x.trim().parse().unwrap()).collect();
    (v[0], v[1])
}

/// (t, interp) cases of a pair.
fn cases(p: &Path) -> Vec<(f32, bool, PathBuf)> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(p).unwrap().flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if let Some(rest) = n.strip_prefix("interp_") {
            v.push((rest.parse().unwrap(), true, e.path()));
        } else if let Some(rest) = n.strip_prefix("extrap_") {
            v.push((rest.parse().unwrap(), false, e.path()));
        }
    }
    v.sort_by(|a, b| a.2.cmp(&b.2));
    v
}

fn psnr_u8(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| {
        let d = x as f64 - y as f64;
        d * d
    }).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

fn to_u8(img: &Image) -> Vec<u8> {
    img.d.iter().flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)).collect()
}

#[test]
fn matcher_matches_reference() {
    for p in pairs() {
        let prev = image(&p, "prev");
        let cur = image(&p, "cur");
        let m = estimate(&cur, &prev, &MatchParams::default());
        let fr = field(&p, "field");
        let f2r = field(&p, "rescue_field");
        let cut = json_num(&p, "cut_R") as f32;
        let mut nvec = 0;
        let mut maxd = [0.0f32; 4];
        for (a, b) in m.f.d.iter().zip(&fr.d) {
            for k in 0..4 {
                maxd[k] = maxd[k].max((a[k] - b[k]).abs());
            }
            if (a[0] - b[0]).abs() > 1e-6 || (a[1] - b[1]).abs() > 1e-6 {
                nvec += 1;
            }
        }
        let n2 = m.rescue.d.iter().zip(&f2r.d).filter(|(a, b)| a[2].is_finite() != b[2].is_finite() || (a[2].is_finite() && ((a[0] - b[0]).abs() > 1e-6 || (a[1] - b[1]).abs() > 1e-6))).count();
        eprintln!("{:<14} field {}x{} vec mismatch {}/{} maxdiff {:?} | rescue mismatch {} | cut {} vs {}", p.file_name().unwrap().to_string_lossy(), m.f.w, m.f.h, nvec, fr.d.len(), maxd, n2, m.meta.cut_r, cut);
    }
}

#[test]
fn layer_field_matches_reference() {
    for p in pairs() {
        let prev = image(&p, "prev");
        let cur = image(&p, "cur");
        let f = field(&p, "field");
        let cut = json_num(&p, "cut_R") as f32;
        for (name, lp) in [("luma", LayerParams::luma()), ("colour", LayerParams::colour())] {
            let fl = layer_field(&cur, &prev, &f, cut, &lp);
            let v = npy(&p, &format!("layers_{name}_v")).f32();
            let z = npy(&p, &format!("layers_{name}_z")).f32();
            let (mut nv, mut nz, mut ok_r, mut ok_m) = (0, 0.0f32, 0, 0);
            for i in 0..fl.z.d.len() {
                let (a, b) = (fl.z.d[i], z[i]);
                ok_r += b.is_finite() as usize;
                ok_m += (a.is_finite() != b.is_finite()) as usize;
                if a.is_finite() && b.is_finite() {
                    nz = nz.max((a - b).abs());
                    let u = fl.v.d[i];
                    if (u[0] - v[2 * i]).abs() > 1e-6 || (u[1] - v[2 * i + 1]).abs() > 1e-6 {
                        nv += 1;
                    }
                }
            }
            eprintln!("{:<14} {name:<7} tiles {} valid ref {} | validity mismatch {} vec mismatch {} max dz {:e}", p.file_name().unwrap().to_string_lossy(), fl.z.d.len(), ok_r, ok_m, nv, nz);
        }
    }
}

fn layer(p: &Path, name: &str) -> LayerField {
    let v = npy(p, &format!("layers_{name}_v"));
    let z = npy(p, &format!("layers_{name}_z"));
    let (th, tw) = (z.shape[0], z.shape[1]);
    let vv = v.f32();
    LayerField { v: Grid::from_fn(tw, th, |x, y| [vv[(y * tw + x) * 2], vv[(y * tw + x) * 2 + 1]]), z: Grid::from_vec(tw, th, z.f32()), tile: 8, margin: 1 }
}

fn flow(p: &Path, name: &str) -> Grid<V2> {
    let n = npy(p, name);
    let (h, w) = (n.shape[0], n.shape[1]);
    let v = n.f32();
    Grid::from_fn(w, h, |x, y| [v[(y * w + x) * 2], v[(y * w + x) * 2 + 1]])
}

/// The solver on the reference's own fields: isolates the solver port from matcher drift.
#[test]
fn solver_matches_reference_on_reference_fields() {
    let mut worst = f64::INFINITY;
    for p in pairs() {
        let prev = image(&p, "prev");
        let cur = image(&p, "cur");
        let f = field(&p, "field");
        let f2 = field(&p, "rescue_field");
        let cut = json_num(&p, "cut_R") as f32;
        let (g, o) = json_gain(&p);
        let flc = layer(&p, "colour");
        let fll = layer(&p, "luma");
        let (fd, gd) = (flow(&p, "flow_cur_prev"), flow(&p, "flow_prev_cur"));
        for (t, interp, c) in cases(&p) {
            for (v, prm) in [("block", Params::block_warp()), ("dense", Params::dense_flow())] {
                let inp = PairInputs {
                    prev: &prev,
                    cur: &cur,
                    f: &f,
                    rescue: Some(&f2),
                    cut_r: cut,
                    gain: Gain { scale: g, offset: o },
                    fl: Some(if prm.colour_layer { &flc } else { &fll }),
                    dense: if prm.dense { Some((&fd, &gd)) } else { None },
                };
                let z = prepare(&inp, t, if interp { Mode::Interp } else { Mode::Extrap }, &prm);
                let dcd = decide(&z, &prm);
                let got = to_u8(&dcd.out);
                let want = npy(&c, &format!("out_{v}")).u8();
                let ps = psnr_u8(&got, &want);
                let ndiff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                let maxd = got.iter().zip(&want).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
                eprintln!("{:<12} {:<12} {v:<5} psnr {ps:7.2} dB  differing {ndiff}/{} max {maxd}  nact {}", p.file_name().unwrap().to_string_lossy(), c.file_name().unwrap().to_string_lossy(), got.len(), z.n());
                worst = worst.min(ps);
            }
        }
    }
    eprintln!("worst {worst:.2} dB");
    assert!(worst >= 60.0, "solver drifted from the reference: {worst:.2} dB");
}

/// The reference's DIS flow, read back (the GPU matcher is a separate port).
struct Fixed(Grid<V2>, Grid<V2>);
impl DenseFlow for Fixed {
    fn flow(&mut self, _: &Image, _: &Image, _: &Grid<V2>) -> (Grid<V2>, Grid<V2>) {
        (self.0.clone(), self.1.clone())
    }
}


fn stats(got: &[u8], want: &[u8]) -> (f64, usize, i32) {
    let nd = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let mx = got.iter().zip(want).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap_or(0);
    (psnr_u8(got, want), nd, mx)
}

/// Rust matcher + layer fields + both solvers + fusion record vs the reference end to end.
#[test]
fn pipeline_matches_reference() {
    let mut worst = f64::INFINITY;
    let mut worst_rec = f64::INFINITY;
    for p in pairs() {
        let mut fx = Fixed(flow(&p, "flow_cur_prev"), flow(&p, "flow_prev_cur"));
        let pair = Pair::analyse(image(&p, "prev"), image(&p, "cur"), Some(&mut fx));
        for (t, interp, c) in cases(&p) {
            let mode = if interp { Mode::Interp } else { Mode::Extrap };
            let (ob, eb) = pair.solve(t, mode, &Params::block_warp()).unwrap();
            let (od, ed) = pair.solve(t, mode, &Params::dense_flow()).unwrap();
            let rec = pair.fusion_features(&ob, &od, &eb, &ed).unwrap();
            let n = rec.w * rec.h;
            let cues = npy(&c, "cues").u8();
            let mine = |k: usize| -> Vec<u8> { (0..n).map(|i| rec.d[i][k]).collect() };
            let cat = |a: &[usize]| -> Vec<u8> { a.iter().flat_map(|&k| mine(k)).collect() };
            let planar = |v: &[u8], c: usize| -> Vec<u8> { (0..c).flat_map(|k| (0..n).map(move |i| v[i * c + k])).collect() };
            let (pb, nb, mb) = stats(&cat(&[6, 7, 8]), &planar(&npy(&c, "out_block").u8(), 3));
            let (pd, nd, md) = stats(&cat(&[9, 10, 11]), &planar(&npy(&c, "out_dense").u8(), 3));
            let (pe, ne, me) = stats(&cat(&[12, 13, 14]), &planar(&cues, 3));
            eprintln!(
                "{:<12} {:<12} block {pb:7.2} dB ({nb} px-ch, max {mb}) | dense {pd:7.2} dB ({nd}, max {md}) | cues (warp errors, gap) {pe:7.2} dB ({ne}, max {me})",
                p.file_name().unwrap().to_string_lossy(),
                c.file_name().unwrap().to_string_lossy()
            );
            worst = worst.min(pb).min(pd);
            worst_rec = worst_rec.min(pe);
        }
    }
    eprintln!("worst output {worst:.2} dB, worst cue {worst_rec:.2} dB");
    assert!(worst >= 60.0, "output drifted from the reference: {worst:.2} dB");
    assert!(worst_rec >= 40.0, "fusion cues drifted from the reference: {worst_rec:.2} dB");
}

struct Const(f32, f32);
impl FusionModel for Const {
    fn infer(&mut self, x: &Tensor) -> Option<Tensor> {
        let mut y = Tensor::zeros(fusion::COUT, x.h, x.w);
        for i in 0..x.h * x.w {
            y.d[i] = self.0;
            for c in 1..4 {
                y.d[c * x.h * x.w + i] = self.1;
            }
        }
        Some(y)
    }
}

struct Absent;
impl FusionModel for Absent {
    fn infer(&mut self, _: &Tensor) -> Option<Tensor> {
        None
    }
}

/// No model, a failed model, w = 0, w = 1: block, block, block, dense.
#[test]
fn fusion_fallback_and_blend_endpoints() {
    let p = pairs().into_iter().find(|p| p.ends_with("bench_t2obj")).unwrap_or_else(|| pairs()[0].clone());
    let mut fx = Fixed(flow(&p, "flow_cur_prev"), flow(&p, "flow_prev_cur"));
    let pair = Pair::analyse(image(&p, "prev"), image(&p, "cur"), Some(&mut fx));
    let (t, interp, _) = cases(&p)[0].clone();
    let mode = if interp { Mode::Interp } else { Mode::Extrap };
    let (ob, eb) = pair.solve(t, mode, &Params::block_warp()).unwrap();
    let (od, ed) = pair.solve(t, mode, &Params::dense_flow()).unwrap();
    let rec = pair.fusion_features(&ob, &od, &eb, &ed).unwrap();
    assert_eq!(to_u8(&pair.generate(t, mode, None)), to_u8(&ob));
    assert_eq!(to_u8(&pair.generate(t, mode, Some(&mut Absent))), to_u8(&ob));
    assert_eq!(to_u8(&fusion::fused_frame(&ob, Some(&rec), t, !interp, Some(&mut Const(0.0, 0.0)))), to_u8(&ob));
    assert_eq!(to_u8(&fusion::fused_frame(&ob, Some(&rec), t, !interp, Some(&mut Const(1.0, 0.0)))), to_u8(&od));
    let x = rec.net_input(t, !interp);
    assert_eq!((x.c, x.h * 2, x.w * 2), (fusion::CIN, rec.h, rec.w));
}