// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The labeler, planner and cross-check over synthetic frames shaped like the survey's games. The
//! survey's own frame graphs run through the C ABI in tests/host/lapis_hud_test.cpp.

use super::plan::{lookup, slot, Planner, MAX_PROBE, TABLE_SIZE};
use super::*;
use crate::model::{Image, Span, Step, Target};
use crate::signals::matrix::test_util::{perspective, transpose, view};
use crate::signals::matrix::{mul, Mat4};

const SWAPCHAIN: u32 = 1;
const DEPTH: u32 = 2;
const HDR: u32 = 3;
const RGBA8: u32 = 37;
const RGBA16F: u32 = 97;
const D24S8: u32 = 129;
const SCENE_PIPE: u64 = 0x1000;
const UI_PIPE: u64 = 0x2000;
const TEXT_PIPE: u64 = 0x3000;
const POST_PIPE: u64 = 0x4000;
const PRESENT_PASS: u64 = 0xabc;
const SCENE_PASS: u64 = 0xdef;

fn scene_draw(index: u32) -> HudDraw {
    HudDraw {
        pipeline: SCENE_PIPE,
        index,
        elements: 2400,
        instances: 1,
        shader: 0x51,
        state: DRAW_DEPTH_TEST | DRAW_DEPTH_COMPARE | DRAW_DEPTH_WRITE | DRAW_INDEXED,
        textures: TEX_UPLOADED,
        scissor: 255,
        vertex: 5,
        pad: [0; 3],
    }
}

fn ui_draw(pipeline: u64, index: u32) -> HudDraw {
    HudDraw {
        pipeline,
        index,
        elements: 6 * 40,
        instances: 1,
        shader: if pipeline == TEXT_PIPE { 0x77 } else { 0x76 },
        state: DRAW_BLEND | DRAW_INDEXED,
        textures: if pipeline == TEXT_PIPE { TEX_SINGLE_CHANNEL } else { TEX_UPLOADED },
        scissor: 255,
        vertex: 3 | VERTEX_COLOUR8 | VERTEX_2D,
        pad: [0; 3],
    }
}

fn post_draw(index: u32) -> HudDraw {
    HudDraw {
        pipeline: POST_PIPE,
        index,
        elements: 3,
        instances: 1,
        shader: 0x90,
        state: 0,
        textures: TEX_RENDER_TARGET | TEX_UNTONED,
        scissor: 255,
        vertex: VERTEX_NONE,
        pad: [0; 3],
    }
}

/// A frame: a scene pass into HDR with depth, then the presenting pass, whose draws are given.
struct Fx {
    steps: Vec<Step>,
    targets: Vec<Target>,
    reads: Vec<u32>,
    images: Vec<Image>,
    passes: Vec<HudPass>,
    draws: Vec<HudDraw>,
    blocks: Vec<HudBlock>,
    floats: Vec<f32>,
    roles: Roles,
}

fn target(image: u32, format: u32, load: u32) -> Target {
    Target { image, format, load, store: 0, samples: 1 }
}

fn step(colours: Span, reads: Span, draws: u32, depth: Option<Target>) -> Step {
    Step {
        is_pass: true,
        has_depth: depth.is_some(),
        op_kind: 0,
        flips: 0,
        depth: depth.unwrap_or(target(0, 0, 0)),
        clear_depth: f32::NAN,
        draws,
        depth_writes: if depth.is_some() { draws } else { 0 },
        depth_clears: 0,
        last_clear_draw: 0,
        colors: colours,
        reads,
        src: 0,
        dst: 0,
    }
}

impl Fx {
    fn new(scene: &[HudDraw], present: &[HudDraw]) -> Fx {
        let mut fx = Fx {
            steps: Vec::new(),
            targets: vec![target(HDR, RGBA16F, 1), target(SWAPCHAIN, RGBA8, 1)],
            reads: vec![HDR],
            images: vec![
                Image { id: SWAPCHAIN, format: RGBA8, width: 2400, height: 1080 },
                Image { id: DEPTH, format: D24S8, width: 2400, height: 1080 },
                Image { id: HDR, format: RGBA16F, width: 2400, height: 1080 },
            ],
            passes: Vec::new(),
            draws: Vec::new(),
            blocks: Vec::new(),
            floats: Vec::new(),
            roles: Roles { depth: DEPTH, scene: HDR, target: SWAPCHAIN, geometry: 0, ui: 1, ..Roles::default() },
        };
        fx.steps.push(step(Span { start: 0, len: 1 }, Span { start: 0, len: 0 }, scene.len() as u32, Some(target(DEPTH, D24S8, 1))));
        fx.passes.push(HudPass { key: SCENE_PASS, draw_start: 0, draw_count: scene.len() as u32 });
        fx.draws.extend_from_slice(scene);
        let mut s = step(Span { start: 1, len: 1 }, Span { start: 0, len: 1 }, present.len() as u32, None);
        s.depth_writes = 0;
        fx.steps.push(s);
        fx.passes.push(HudPass { key: PRESENT_PASS, draw_start: scene.len() as u32, draw_count: present.len() as u32 });
        fx.draws.extend_from_slice(present);
        fx
    }

    /// The uniform block draw `draw` of step `step` read.
    fn block(&mut self, step: u32, draw: u32, m: &Mat4) {
        self.blocks.push(HudBlock { step, draw, start: self.floats.len() as u32, len: 16 });
        self.floats.extend_from_slice(&transpose(m));
    }

    fn frame(&self) -> Frame {
        Frame {
            steps: self.steps.as_ptr(),
            step_count: self.steps.len() as u32,
            targets: self.targets.as_ptr(),
            target_count: self.targets.len() as u32,
            reads: self.reads.as_ptr(),
            read_count: self.reads.len() as u32,
            images: self.images.as_ptr(),
            image_count: self.images.len() as u32,
            presented: SWAPCHAIN,
            width: 2400,
            height: 1080,
        }
    }

    fn run(&self, state: &mut HudState, hide: bool, skipped: u32, grid: Option<Grid>) -> HudOutput {
        let frame = self.frame();
        // SAFETY: `frame` points into self, alive for the call.
        let view = unsafe { View::new(&frame) };
        let f = HudFrame {
            view,
            passes: &self.passes,
            draws: &self.draws,
            blocks: &self.blocks,
            floats: &self.floats,
            roles: self.roles,
            hide_hud: hide,
            skipped,
            grid,
        };
        state.update(&f)
    }
}

fn ortho() -> Mat4 {
    [2.0 / 2400.0, 0.0, 0.0, -1.0, 0.0, -2.0 / 1080.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0]
}

fn camera() -> Mat4 {
    mul(&perspective(55.0, 2400.0 / 1080.0, 0.2), &view([1.0, 2.0, 3.0], 10.0))
}

/// The presenting pass of a game that interleaves its HUD with in-pass scene draws: tonemap,
/// then HUD, a late scene draw, more HUD and text.
fn interleaved() -> Fx {
    let scene: Vec<HudDraw> = (0..20).map(scene_draw).collect();
    let present = [
        post_draw(0),
        ui_draw(UI_PIPE, 1),
        ui_draw(TEXT_PIPE, 2),
        scene_draw(3),
        ui_draw(UI_PIPE, 4),
        ui_draw(TEXT_PIPE, 5),
    ];
    let mut fx = Fx::new(&scene, &present);
    fx.block(0, 0, &camera());
    fx.block(1, 3, &camera());
    fx.block(1, 1, &ortho());
    fx.block(1, 2, &ortho());
    fx
}

#[test]
fn draws_are_labelled_by_what_they_are() {
    let fx = interleaved();
    let mut s = HudState::default();
    fx.run(&mut s, false, 0, None);
    let labels = s.labels();
    assert!(labels[..20].iter().all(|&l| l == LABEL_SCENE), "{labels:?}");
    assert_eq!(&labels[20..], &[LABEL_POST, LABEL_HUD, LABEL_HUD, LABEL_SCENE, LABEL_HUD, LABEL_HUD]);
    assert!(s.confidences()[21] > 0.9, "{}", s.confidences()[21]);
    assert_eq!(&s.weak_labels()[20..23], &[LABEL_POST, LABEL_HUD, LABEL_HUD]);
    assert_eq!(s.feature_rows().len(), fx.draws.len() * features::FEATURES);
}

#[test]
fn interleaved_hud_is_skipped_by_pipeline_and_the_scene_draw_between_kept() {
    let fx = interleaved();
    let mut s = HudState::default();
    let mut out = HudOutput::default();
    for frame in 0..6 {
        out = fx.run(&mut s, true, 0, None);
        assert_eq!(out.skipping, frame >= 3, "frame {frame}");
    }
    assert!(out.coverage >= 0.99);
    let t = s.table();
    assert_eq!(lookup(t, PRESENT_PASS, UI_PIPE, 1), (true, true));
    assert_eq!(lookup(t, PRESENT_PASS, TEXT_PIPE, 5), (true, true));
    assert_eq!(lookup(t, PRESENT_PASS, SCENE_PIPE, 3), (false, true));
    assert!(!lookup(t, PRESENT_PASS, POST_PIPE, 0).0);
    assert_eq!(lookup(t, SCENE_PASS, SCENE_PIPE, 7), (false, true));
    assert_eq!(lookup(t, SCENE_PASS, UI_PIPE, 7), (false, false), "another pass's pair is its own");
    assert!(out.changed || out.entries > 0);
    let again = fx.run(&mut s, true, 4, None);
    assert!(again.applied && !again.changed);
}

#[test]
fn nothing_is_skipped_while_hide_hud_is_off_but_cameras_stay_known() {
    let fx = interleaved();
    let mut s = HudState::default();
    for _ in 0..6 {
        let out = fx.run(&mut s, false, 0, None);
        assert!(!out.skipping);
    }
    assert_eq!(lookup(s.table(), PRESENT_PASS, UI_PIPE, 1), (false, true));
    let unknown = HudState::default();
    assert_eq!(lookup(unknown.table(), PRESENT_PASS, UI_PIPE, 1), (false, false));
}

fn labels_for(fx: &Fx, hud: &[usize]) -> (Vec<u8>, Vec<f32>) {
    let labels = (0..fx.draws.len()).map(|i| if hud.contains(&i) { LABEL_HUD } else { LABEL_SCENE }).collect();
    (labels, vec![0.95; fx.draws.len()])
}

fn plan(planner: &mut Planner, fx: &Fx, hud: &[usize]) -> plan::Planned {
    let frame = fx.frame();
    // SAFETY: `frame` points into fx, alive for the call.
    let view = unsafe { View::new(&frame) };
    let f = HudFrame {
        view,
        passes: &fx.passes,
        draws: &fx.draws,
        blocks: &[],
        floats: &[],
        roles: fx.roles,
        hide_hud: true,
        skipped: 0,
        grid: None,
    };
    let (labels, confidence) = labels_for(fx, hud);
    planner.update(&f, &labels, &confidence, |_| false, true)
}

#[test]
fn a_pipeline_shared_by_scene_then_hud_skips_from_past_its_last_scene_draw() {
    // One pipeline: three scene draws, then HUD from index 3 on, the count varying by frame.
    let mut p = Planner::default();
    for frame in 0..6 {
        let n = 6 + frame % 3;
        let present: Vec<HudDraw> = (0..n).map(|i| ui_draw(UI_PIPE, i)).collect();
        let fx = Fx::new(&[], &present);
        plan(&mut p, &fx, &(3..n as usize).collect::<Vec<_>>());
    }
    assert!(!lookup(p.table(), PRESENT_PASS, UI_PIPE, 2).0);
    assert!(lookup(p.table(), PRESENT_PASS, UI_PIPE, 3).0);
    assert!(lookup(p.table(), PRESENT_PASS, UI_PIPE, 40).0);
}

#[test]
fn a_pipeline_that_interleaves_scene_and_hud_skips_nothing_and_holds_the_table_back() {
    let mut p = Planner::default();
    let present: Vec<HudDraw> = (0..6).map(|i| ui_draw(UI_PIPE, i)).collect();
    let fx = Fx::new(&[], &present);
    let mut last = None;
    for _ in 0..6 {
        last = Some(plan(&mut p, &fx, &[1, 3, 5]));
    }
    let last = last.unwrap();
    assert!(!last.skipping && last.coverage == 0.0);
    assert!(!lookup(p.table(), PRESENT_PASS, UI_PIPE, 5).0);
}

#[test]
fn the_table_lets_go_once_it_stops_covering_the_hud() {
    let mut p = Planner::default();
    let present: Vec<HudDraw> = (0..4).map(|i| ui_draw(UI_PIPE + u64::from(i), i)).collect();
    let fx = Fx::new(&[], &present);
    for _ in 0..5 {
        plan(&mut p, &fx, &[0, 1, 2, 3]);
    }
    assert!(p.skipping());
    // A new HUD pipeline appears while two of the old ones turn out to be scene.
    let mut other = Fx::new(&[], &present);
    other.draws.push(ui_draw(0x9999, 4));
    other.passes[1].draw_count += 1;
    for frame in 0..3 {
        let out = plan(&mut p, &other, &[1, 4]);
        assert_eq!(out.skipping, frame < 2, "frame {frame}");
    }
}

#[test]
fn keys_are_never_zero_and_probe_within_reach() {
    let mut table = [HudEntry::default(); TABLE_SIZE];
    for pass in 0..64u64 {
        for pipeline in 0..32u64 {
            let key = table_key(pass * 0x1000, pipeline * 0x40 + 0x7f00_0000_0000);
            assert_ne!(key, 0);
            let mut placed = false;
            for probe in 0..MAX_PROBE {
                let e = &mut table[(slot(key) + probe) & (TABLE_SIZE - 1)];
                if e.key == 0 {
                    *e = HudEntry { key, value: VALUE_KNOWN };
                    placed = true;
                    break;
                }
            }
            assert!(placed, "half-full table probes too far");
        }
    }
}

/// A grid of a world texture panned by `pan` cells, with a HUD block that holds still.
fn grid(width: usize, height: usize, pan: usize, hud: bool) -> Vec<u8> {
    let mut g = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let wx = x + pan;
            g[y * width + x] = (((wx * 37 + y * 11) ^ (wx * 5)) % 200) as u8 + 20;
            if hud && (2..8).contains(&x) && (2..6).contains(&y) {
                g[y * width + x] = if (x + y) % 2 == 0 { 250 } else { 30 };
            }
        }
    }
    g
}

fn turning_camera(frame: u32) -> Mat4 {
    mul(&perspective(55.0, 2.0, 0.2), &view([0.0; 3], frame as f32 * 4.0))
}

#[test]
fn the_screen_confirms_a_hud_and_catches_skipping_that_leaves_it() {
    let fx = interleaved();
    let (w, h) = (48u32, 24u32);
    let mut s = HudState::default();
    let mut out = HudOutput::default();
    for frame in 0..40u32 {
        let g = grid(w as usize, h as usize, frame as usize * 3, true);
        let grid = Grid { luma: &g, width: w, height: h, seq: frame, camera: Some(turning_camera(frame)) };
        out = fx.run(&mut s, false, 0, Some(grid));
    }
    assert_eq!(out.cross, CROSS_AGREES);
    assert!(out.hud_share > 0.01 && out.hud_share < 0.05, "{}", out.hud_share);

    // Skipping that leaves the HUD on screen is caught, and the table stops skipping for a while.
    let mut residue = HudState::default();
    let (mut skipped_once, mut caught) = (false, false);
    for frame in 0..120u32 {
        let g = grid(w as usize, h as usize, frame as usize * 3, true);
        let grid = Grid { luma: &g, width: w, height: h, seq: frame, camera: Some(turning_camera(frame)) };
        out = fx.run(&mut residue, frame >= 40, 4, Some(grid));
        skipped_once |= out.skipping;
        caught |= out.cross == CROSS_RESIDUE;
        if caught {
            assert!(!out.skipping, "frame {frame}");
        }
    }
    assert!(skipped_once && caught, "{skipped_once} {caught} {out:?}");

    // Skipping that clears it agrees.
    let mut clean = HudState::default();
    for frame in 0..80u32 {
        let skipping = clean.planner.skipping();
        let g = grid(w as usize, h as usize, frame as usize * 3, !skipping);
        let grid = Grid { luma: &g, width: w, height: h, seq: frame, camera: Some(turning_camera(frame)) };
        out = fx.run(&mut clean, frame >= 40, 4, Some(grid));
    }
    assert!(out.skipping);
    assert_eq!(out.cross, CROSS_AGREES);
    assert!(out.residue < 0.005, "{}", out.residue);
}

#[test]
fn hud_labels_with_no_hud_on_screen_are_not_trusted() {
    let fx = interleaved();
    let (w, h) = (48u32, 24u32);
    let mut s = HudState::default();
    let mut out = HudOutput::default();
    for frame in 0..60u32 {
        let g = grid(w as usize, h as usize, frame as usize * 3, false);
        let grid = Grid { luma: &g, width: w, height: h, seq: frame, camera: Some(turning_camera(frame)) };
        out = fx.run(&mut s, frame >= 40, 0, Some(grid));
        if frame >= 40 {
            assert_eq!(out.cross, CROSS_NO_HUD, "frame {frame}");
            assert!(!out.skipping);
        }
    }
    assert_eq!(out.cross, CROSS_NO_HUD);
}

#[test]
fn a_still_camera_says_nothing() {
    let fx = interleaved();
    let mut s = HudState::default();
    let g = grid(48, 24, 0, true);
    let mut out = HudOutput::default();
    for frame in 0..40u32 {
        let grid = Grid { luma: &g, width: 48, height: 24, seq: frame, camera: Some(turning_camera(0)) };
        out = fx.run(&mut s, false, 0, Some(grid));
    }
    assert_eq!(out.cross, CROSS_NONE);
    assert_eq!(out.hud_share, 0.0);
}

#[test]
fn learned_shaders_carry_into_the_next_run() {
    let fx = interleaved();
    let mut s = HudState::default();
    for _ in 0..12 {
        fx.run(&mut s, false, 0, None);
    }
    let line = s.profile();
    assert!(line.contains("00000076:"), "{line}");
    let mut next = HudState::default();
    assert!(next.warm(&line));
    assert!(next.priors.prior(0x76) > 0.5 && next.priors.prior(0x51) < -0.5);
    assert!(!next.set_weights("hudweights 1 4 3 0 0 0"));
}

#[test]
fn a_ui_draw_in_the_scene_pass_with_the_scene_camera_is_world_ui() {
    // A name plate: blended, depth-tested, glyph atlas, the scene's camera, after the opaque draws.
    let mut plate = ui_draw(TEXT_PIPE, 20);
    plate.state = DRAW_BLEND | DRAW_DEPTH_TEST | DRAW_DEPTH_COMPARE | DRAW_INDEXED;
    plate.vertex = 3 | VERTEX_COLOUR8;
    plate.shader = 0;
    let mut scene: Vec<HudDraw> = (0..20).map(scene_draw).collect();
    scene.push(plate);
    let mut fx = Fx::new(&scene, &[post_draw(0)]);
    fx.steps[0].draws = 21;
    fx.block(0, 20, &camera());
    let mut s = HudState::default();
    fx.run(&mut s, false, 0, None);
    assert_eq!(s.labels()[20], LABEL_WORLD_UI, "{:?}", &s.feature_rows()[20 * features::FEATURES..21 * features::FEATURES]);
    assert_eq!(s.labels()[21], LABEL_POST);
}
