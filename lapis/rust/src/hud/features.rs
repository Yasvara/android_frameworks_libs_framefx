// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The features of one draw, in [0, 1] (the shader prior in [-1, 1]): its own state first, then
//! what its pass is in the frame. Unknowns read 0. The order is the weights' and the survey's.

use super::projection::Camera;
use super::*;
use crate::model::{is_untoned, Step};

pub const FEATURE_NAMES: [&str; 33] = [
    "bias",
    "depth_write",
    "depth_compare",
    "blend",
    "stencil_write",
    "perspective",
    "screen_ortho",
    "quad",
    "log_elements",
    "fullscreen",
    "instanced",
    "tex_single_channel",
    "tex_render_target",
    "tex_depth",
    "tex_uploaded_only",
    "vertex_colour8",
    "vertex_2d",
    "vertex_few",
    "scissor_partial",
    "shader_prior",
    "pass_depth",
    "pass_presents",
    "pass_untoned_target",
    "before_scene_end",
    "pass_geometry",
    "pass_ui",
    "after_depth_clear",
    "after_last_depth_write",
    "after_post_draw",
    "position_in_pass",
    "pass_offscreen_small",
    "pass_reads_scene",
    "pass_few_draws",
];
pub const FEATURES: usize = FEATURE_NAMES.len();

/// What one step is in the frame.
#[derive(Clone, Copy, Default)]
struct PassFacts {
    is_pass: bool,
    has_depth: bool,
    draws: u32,
    depth_clears: u32,
    last_clear_draw: u32,
    presents: bool,
    loads_target: bool,
    untoned_target: bool,
    geometry: bool,
    ui: bool,
    clear_split: bool,
    before_geometry: bool,
    after_geometry: bool,
    offscreen_small: bool,
    reads_scene: bool,
    /// Draw indices in the pass, from its captured draws.
    last_depth_write: Option<u32>,
    post_draw: Option<u32>,
}

pub struct Context {
    facts: Vec<PassFacts>,
}

fn fullscreen(d: &HudDraw) -> bool {
    let triangle = d.elements == 3 && d.state & DRAW_INDEXED == 0;
    let few = triangle || matches!(d.elements, 4 | 6) || (d.vertex & VERTEX_NONE != 0 && d.elements <= 6);
    few && d.instances <= 1 && d.scissor >= 250 && d.state & DRAW_DEPTH_WRITE == 0
}

/// The draw that turns the scene into the screen's colours: full-screen, sampling what a pass drew.
fn post_draw(draws: &[HudDraw], reads_scene: bool) -> Option<u32> {
    let textured = draws.iter().any(|d| d.textures != 0);
    if textured {
        return draws
            .iter()
            .find(|d| fullscreen(d) && d.textures & (TEX_RENDER_TARGET | TEX_UNTONED) != 0)
            .map(|d| d.index);
    }
    // Without texture bindings (survey logs before they were recorded): a pass that samples the
    // scene and opens on a draw that neither writes nor tests depth starts with its post draw.
    let first = draws.first()?;
    (reads_scene && first.state & (DRAW_DEPTH_WRITE | DRAW_DEPTH_COMPARE) == 0).then_some(first.index)
}

impl Context {
    pub fn new(f: &HudFrame) -> Self {
        let v = &f.view;
        let r = &f.roles;
        let screen = u64::from(v.width) * u64::from(v.height);
        let mut facts = Vec::with_capacity(v.steps.len());
        for (i, s) in v.steps.iter().enumerate() {
            let i = i as i32;
            let mut p = PassFacts { is_pass: s.is_pass, ..PassFacts::default() };
            if s.is_pass {
                let colours = v.colors(s);
                let reads = v.reads(s);
                let drawn = |id: u32| id != 0 && colours.iter().any(|t| t.image == id);
                p.has_depth = s.has_depth;
                p.draws = s.draws;
                p.depth_clears = s.depth_clears;
                p.last_clear_draw = s.last_clear_draw;
                p.presents = drawn(v.presented) || drawn(r.target);
                p.loads_target = colours
                    .iter()
                    .any(|t| (t.image == v.presented || t.image == r.target) && t.load == crate::model::LOAD_OP_LOAD);
                p.untoned_target = colours.first().is_some_and(|t| is_untoned(t.format));
                p.geometry = i == r.geometry;
                p.ui = i == r.ui || i == r.split;
                p.clear_split = r.clear_split && i == r.split;
                p.before_geometry = r.geometry >= 0 && i < r.geometry;
                p.after_geometry = r.geometry >= 0 && i > r.geometry;
                let size = pass_size(v, s);
                p.offscreen_small = screen > 0 && size.area() * 4 < screen;
                p.reads_scene = reads.iter().any(|&id| id != 0 && (id == r.scene || id == r.untoned));
                if let Some(pass) = f.passes.get(i as usize) {
                    let draws = &f.draws[draw_range(pass, f.draws.len())];
                    p.last_depth_write =
                        draws.iter().filter(|d| d.state & DRAW_DEPTH_WRITE != 0).map(|d| d.index).max();
                    p.post_draw = post_draw(draws, p.reads_scene);
                }
            }
            facts.push(p);
        }
        Context { facts }
    }

    /// Writes the features of draw `d` of step `step` into `row` (FEATURES long).
    pub fn fill(&self, step: usize, d: &HudDraw, camera: Camera, prior: f32, row: &mut [f32]) {
        let p = self.facts.get(step).copied().unwrap_or_default();
        let bit = |b: bool| if b { 1.0 } else { 0.0 };
        let attributes = d.vertex & VERTEX_ATTRIBUTES;
        let after = |at: Option<u32>| at.is_some_and(|a| d.index > a);
        let values: [f32; FEATURES] = [
            1.0,
            bit(d.state & DRAW_DEPTH_WRITE != 0),
            bit(d.state & DRAW_DEPTH_COMPARE != 0),
            bit(d.state & DRAW_BLEND != 0),
            bit(d.state & DRAW_STENCIL_WRITE != 0),
            bit(camera.perspective()),
            bit(camera.flat()),
            bit((1..=6).contains(&d.elements) && d.instances <= 1 && d.state & DRAW_INDIRECT == 0),
            ((d.elements as f32 + 1.0).log2() / 20.0).min(1.0),
            bit(fullscreen(d)),
            bit(d.instances > 1),
            bit(d.textures & TEX_SINGLE_CHANNEL != 0),
            bit(d.textures & TEX_RENDER_TARGET != 0),
            bit(d.textures & TEX_DEPTH != 0),
            bit(d.textures & TEX_UPLOADED != 0 && d.textures & (TEX_RENDER_TARGET | TEX_DEPTH | TEX_UNTONED) == 0),
            bit(d.vertex & VERTEX_COLOUR8 != 0),
            bit(d.vertex & VERTEX_2D != 0),
            bit((1..=3).contains(&attributes)),
            bit(d.scissor < 230),
            prior,
            bit(p.has_depth),
            bit(p.presents),
            bit(p.untoned_target),
            bit(p.before_geometry || (p.geometry && !after(p.last_depth_write))),
            bit(p.geometry),
            bit(p.ui),
            bit(p.depth_clears > 0 && d.index >= p.last_clear_draw),
            bit(after(p.last_depth_write)),
            bit(after(p.post_draw)),
            if p.draws > 0 { (d.index as f32 / p.draws as f32).min(1.0) } else { 0.0 },
            bit(p.offscreen_small),
            bit(p.reads_scene),
            bit(p.draws <= 2),
        ];
        row[..FEATURES].copy_from_slice(&values);
    }

    /// Whether `d` is its pass's post draw: what carries the scene to the screen is never HUD,
    /// whatever its features say.
    pub fn is_post_draw(&self, step: usize, d: &HudDraw) -> bool {
        self.facts.get(step).is_some_and(|p| p.post_draw == Some(d.index))
    }

    /// The label the frame's structure alone gives a draw, as Lapis's pass roles see it: what the
    /// training tool starts from, and what host tests hold the labeler to.
    pub fn weak_label(&self, step: usize, d: &HudDraw) -> u8 {
        let Some(p) = self.facts.get(step) else { return LABEL_SCENE };
        if !p.is_pass {
            return LABEL_SCENE;
        }
        if p.clear_split {
            return if d.index >= p.last_clear_draw { LABEL_HUD } else { LABEL_SCENE };
        }
        if p.ui || (p.presents && p.after_geometry) {
            return match p.post_draw {
                Some(t) if d.index == t => LABEL_POST,
                Some(t) if d.index > t => LABEL_HUD,
                Some(_) => LABEL_SCENE,
                None if p.loads_target || !p.has_depth || p.depth_clears > 0 => LABEL_HUD,
                None if d.state & DRAW_DEPTH_WRITE != 0 => LABEL_SCENE,
                None => LABEL_HUD,
            };
        }
        if p.before_geometry || p.geometry || d.state & DRAW_DEPTH_WRITE != 0 {
            return LABEL_SCENE;
        }
        if p.after_geometry && d.state & DRAW_DEPTH_TEST != 0 && !fullscreen(d) {
            return LABEL_SCENE;
        }
        if p.after_geometry && (fullscreen(d) || p.offscreen_small || !p.has_depth) {
            return LABEL_POST;
        }
        LABEL_SCENE
    }
}

fn pass_size(v: &View, s: &Step) -> crate::model::Size {
    let colours = v.colors(s);
    match colours.first() {
        Some(t) => v.size_of(t.image),
        None if s.has_depth => v.size_of(s.depth.image),
        None => crate::model::Size::default(),
    }
}
