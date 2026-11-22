// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Hide HUD by not drawing it. Every draw the recorder captured gets a label (scene, post, HUD,
//! in-world UI) from what is readable about it: its camera, depth and blend state, mesh, textures,
//! vertex layout, fragment shader and where it sits in the frame. Labels over the last frames
//! become a skip table the recorder's draw hooks look up lock-free, keyed by the pass's shape and
//! the pipeline, with a draw index range. A low-resolution copy of the presented frame confirms it:
//! under camera rotation, what holds still on screen is HUD, and it should be gone while skipping.
//! lapis/hud.cpp flattens one frame into `HudInput`, once per present.

mod crosscheck;
mod features;
pub mod ffi;
mod plan;
mod profile;
mod projection;
mod weights;

use core::mem::size_of;
use std::collections::HashMap;

use crate::model::{Frame, Roles, View};
use crosscheck::{CrossCheck, Verdict};
use features::{Context, FEATURES};
use plan::{Planner, TABLE_SIZE};
use profile::ShaderPriors;
use projection::{classify_block, Camera};
use weights::Weights;

pub const LABEL_SCENE: u8 = 0;
pub const LABEL_POST: u8 = 1;
pub const LABEL_HUD: u8 = 2;
pub const LABEL_WORLD_UI: u8 = 3;
pub const LABELS: usize = 4;

// HudDraw::state.
pub const DRAW_BLEND: u16 = 1 << 0;
pub const DRAW_DEPTH_TEST: u16 = 1 << 1;
pub const DRAW_DEPTH_WRITE: u16 = 1 << 2;
pub const DRAW_STENCIL_WRITE: u16 = 1 << 3;
/// Tests depth with a compare other than ALWAYS.
pub const DRAW_DEPTH_COMPARE: u16 = 1 << 4;
pub const DRAW_INDEXED: u16 = 1 << 5;
pub const DRAW_INDIRECT: u16 = 1 << 6;

// HudDraw::textures: what the bound descriptor sets sample.
pub const TEX_SINGLE_CHANNEL: u8 = 1 << 0;
pub const TEX_RENDER_TARGET: u8 = 1 << 1;
pub const TEX_DEPTH: u8 = 1 << 2;
pub const TEX_UPLOADED: u8 = 1 << 3;
pub const TEX_UNTONED: u8 = 1 << 4;

// HudDraw::vertex: the pipeline's vertex input. Low nibble: attribute count.
pub const VERTEX_ATTRIBUTES: u8 = 0x0f;
pub const VERTEX_COLOUR8: u8 = 1 << 4;
pub const VERTEX_2D: u8 = 1 << 5;
pub const VERTEX_NONE: u8 = 1 << 6;

// HudInput::flags.
pub const INPUT_HIDE_HUD: u32 = 1 << 0;
pub const INPUT_GRID: u32 = 1 << 1;
pub const INPUT_GRID_CAMERA: u32 = 1 << 2;

/// One draw as recorded, in its pass's order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HudDraw {
    pub pipeline: u64,
    /// In its pass, counting every draw the pass recorded.
    pub index: u32,
    /// Vertices or indices; 0 when indirect.
    pub elements: u32,
    pub instances: u32,
    /// The fragment shader's SPIR-V hash; 0 unknown.
    pub shader: u32,
    pub state: u16,
    pub textures: u8,
    /// Scissor area over the pass's, 0..255; 255 whole or unknown.
    pub scissor: u8,
    pub vertex: u8,
    pub pad: [u8; 3],
}

/// The captured draws of one step of the frame; `draw_count` 0 for transfers and passes that
/// captured none.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HudPass {
    /// The pass's shape, as the recorder keys its lookups.
    pub key: u64,
    pub draw_start: u32,
    pub draw_count: u32,
}

/// The uniform bytes one sampled draw read; its floats are `HudInput::floats[start..start + len]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HudBlock {
    pub step: u32,
    /// The draw's index in its pass.
    pub draw: u32,
    pub start: u32,
    pub len: u32,
}

#[repr(C)]
pub struct HudInput {
    pub frame: Frame,
    /// One per step of `frame`.
    pub passes: *const HudPass,
    pub pass_count: u32,
    pub draws: *const HudDraw,
    pub draw_count: u32,
    pub blocks: *const HudBlock,
    pub block_count: u32,
    pub floats: *const f32,
    pub float_count: u32,
    /// The presented frame of `grid_seq`, luma, `grid_width` x `grid_height`.
    pub grid: *const u8,
    pub grid_width: u32,
    pub grid_height: u32,
    pub grid_seq: u32,
    pub roles: Roles,
    pub flags: u32,
    /// Draws the table skipped while this frame was recorded.
    pub skipped: u32,
    /// The grid frame's camera: row-major, clip = M * world.
    pub grid_view_projection: [f32; 16],
}

const _: () = assert!(size_of::<HudDraw>() == 32 && size_of::<HudPass>() == 16 && size_of::<HudBlock>() == 16);
const _: () = assert!(size_of::<HudOutput>() == 48 && size_of::<HudEntry>() == 16);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    size_of::<HudInput>() == 296
        && core::mem::offset_of!(HudInput, roles) == 156
        && core::mem::offset_of!(HudInput, grid_view_projection) == 228
);

pub const CROSS_NONE: u8 = 0;
pub const CROSS_AGREES: u8 = 1;
pub const CROSS_RESIDUE: u8 = 2;
pub const CROSS_NO_HUD: u8 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HudOutput {
    /// Draws per label this frame.
    pub labelled: [u32; LABELS],
    /// Share of the HUD draws of this frame the table skips.
    pub coverage: f32,
    /// Mean confidence of the HUD labels.
    pub confidence: f32,
    /// Cross-check: share of the screen that holds still under rotation, unskipped and skipped.
    pub hud_share: f32,
    pub residue: f32,
    pub entries: u32,
    /// (pass, pipeline) pairs whose camera is not known yet.
    pub unsampled: u32,
    /// The table skips HUD draws.
    pub skipping: bool,
    /// The recorder skipped draws by it while this frame was recorded.
    pub applied: bool,
    /// The table differs from the last one handed out.
    pub changed: bool,
    pub cross: u8,
    /// Something learned for the profile since it was last written.
    pub dirty: bool,
}

/// A table slot: `key` 0 is empty.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HudEntry {
    pub key: u64,
    pub value: u64,
}

// HudEntry::value.
pub const VALUE_FROM_SHIFT: u32 = 0;
pub const VALUE_TO_SHIFT: u32 = 24;
pub const VALUE_INDEX_MASK: u64 = 0xff_ffff;
/// `to` when the range runs to the pass's end.
pub const VALUE_OPEN: u64 = VALUE_INDEX_MASK;
pub const VALUE_SKIP: u64 = 1 << 48;
/// The pair's camera is known: the recorder stops sampling its constants.
pub const VALUE_KNOWN: u64 = 1 << 49;

const fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The table key of a pipeline in a pass: never 0. lapis/hud.h HudKey computes the same.
pub const fn table_key(pass: u64, pipeline: u64) -> u64 {
    mix(pass ^ mix(pipeline ^ 0x9e37_79b9_7f4a_7c15)) | 1
}

/// The labeler for one swapchain's frames.
#[derive(Default)]
pub struct HudState {
    weights: Weights,
    priors: ShaderPriors,
    planner: Planner,
    cross: CrossCheck,
    /// Per (pass, pipeline) key: the camera its constants showed, and the frame it was seen.
    cameras: HashMap<u64, (Camera, u32)>,
    frame: u32,
    /// Skipping missed the HUD: none until this frame.
    blocked_until: u32,
    labels: Vec<u8>,
    confidence: Vec<f32>,
    weak: Vec<u8>,
    features: Vec<f32>,
    last: HudOutput,
}


/// A frame as the labeler reads it.
pub struct HudFrame<'a> {
    pub view: View<'a>,
    pub passes: &'a [HudPass],
    pub draws: &'a [HudDraw],
    pub blocks: &'a [HudBlock],
    pub floats: &'a [f32],
    pub roles: Roles,
    pub hide_hud: bool,
    pub skipped: u32,
    pub grid: Option<Grid<'a>>,
}

pub struct Grid<'a> {
    pub luma: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub seq: u32,
    pub camera: Option<[f32; 16]>,
}

/// How long a pair's camera stays known without being seen again: its pipeline may be gone and
/// the handle reused.
const CAMERA_FORGET_FRAMES: u32 = 1800;
/// Frames without skipping after the screen showed it missed the HUD.
const RESIDUE_COOLDOWN_FRAMES: u32 = 900;
/// The profile is offered for writing at most this often.
const PROFILE_PERIOD: u32 = 600;

impl HudState {
    pub fn reset(&mut self) {
        let weights = core::mem::take(&mut self.weights);
        let priors = core::mem::take(&mut self.priors);
        *self = HudState { weights, priors, ..HudState::default() };
    }

    pub fn labels(&self) -> &[u8] {
        &self.labels
    }

    pub fn confidences(&self) -> &[f32] {
        &self.confidence
    }

    pub fn weak_labels(&self) -> &[u8] {
        &self.weak
    }

    pub fn feature_rows(&self) -> &[f32] {
        &self.features
    }

    pub fn table(&self) -> &[HudEntry; TABLE_SIZE] {
        self.planner.table()
    }

    /// Labels `f`, plans the skips the next frame's draws look up, and checks them against the
    /// screen.
    pub fn update(&mut self, f: &HudFrame) -> HudOutput {
        self.frame = self.frame.wrapping_add(1);
        self.learn_cameras(f);
        let n = f.draws.len();
        self.labels.clear();
        self.labels.resize(n, LABEL_SCENE);
        self.confidence.clear();
        self.confidence.resize(n, 0.0);
        self.weak.clear();
        self.weak.resize(n, LABEL_SCENE);
        self.features.clear();
        self.features.resize(n * FEATURES, 0.0);
        let mut out = HudOutput::default();
        let ctx = Context::new(f);
        let mut hud_confidence = 0.0f32;
        for (step, pass) in f.passes.iter().enumerate() {
            let range = draw_range(pass, n);
            for i in range {
                let d = &f.draws[i];
                let key = table_key(pass.key, d.pipeline);
                let camera = self.cameras.get(&key).map_or(Camera::Unknown, |c| c.0);
                let row = &mut self.features[i * FEATURES..(i + 1) * FEATURES];
                ctx.fill(step, d, camera, self.priors.prior(d.shader), row);
                let (mut label, p) = self.weights.label(row);
                if ctx.is_post_draw(step, d) {
                    label = LABEL_POST;
                }
                self.labels[i] = label;
                self.confidence[i] = p;
                self.weak[i] = ctx.weak_label(step, d);
                out.labelled[label as usize] += 1;
                if label == LABEL_HUD || label == LABEL_WORLD_UI {
                    hud_confidence += p;
                }
            }
        }
        let huds = out.labelled[LABEL_HUD as usize] + out.labelled[LABEL_WORLD_UI as usize];
        out.confidence = if huds > 0 { hud_confidence / huds as f32 } else { 0.0 };

        let verdict = self.cross.update(f, self.planner.skipping(), huds > 0);
        out.hud_share = self.cross.hud_share();
        out.residue = self.cross.residue();
        out.cross = match verdict {
            Verdict::Unknown => CROSS_NONE,
            Verdict::Agrees => CROSS_AGREES,
            Verdict::Residue => CROSS_RESIDUE,
            Verdict::NoHud => CROSS_NO_HUD,
        };
        if verdict == Verdict::Residue {
            self.blocked_until = self.frame.wrapping_add(RESIDUE_COOLDOWN_FRAMES);
        }
        let trusted = verdict != Verdict::NoHud && self.frame >= self.blocked_until;
        if trusted {
            self.priors.learn(f.draws, &self.labels, &self.confidence, &self.weak);
        }
        let known = |key: u64| self.cameras.contains_key(&key);
        let planned = self.planner.update(f, &self.labels, &self.confidence, known, f.hide_hud && trusted);
        out.coverage = planned.coverage;
        out.entries = planned.entries;
        out.unsampled = planned.unsampled;
        out.skipping = planned.skipping;
        out.applied = self.last.skipping && f.skipped > 0;
        out.changed = planned.changed;
        if planned.skipping != self.last.skipping {
            self.cross.plan_changed();
        }
        out.dirty = self.priors.dirty() && self.frame.is_multiple_of(PROFILE_PERIOD);
        self.last = out;
        out
    }

    fn learn_cameras(&mut self, f: &HudFrame) {
        for b in f.blocks {
            let Some(pass) = f.passes.get(b.step as usize) else { continue };
            let draws = &f.draws[draw_range(pass, f.draws.len())];
            let Some(d) = draws.iter().find(|d| d.index == b.draw) else { continue };
            let start = b.start as usize;
            let Some(floats) = f.floats.get(start..start.saturating_add(b.len as usize)) else { continue };
            let key = table_key(pass.key, d.pipeline);
            let seen = classify_block(floats);
            let entry = self.cameras.entry(key).or_insert((Camera::None, self.frame));
            entry.0 = entry.0.merge(seen);
            entry.1 = self.frame;
        }
        let frame = self.frame;
        if frame.is_multiple_of(64) {
            self.cameras.retain(|_, (_, seen)| frame.wrapping_sub(*seen) < CAMERA_FORGET_FRAMES);
        }
        // A pair drawn this frame stays known.
        for pass in f.passes {
            for d in &f.draws[draw_range(pass, f.draws.len())] {
                if let Some(c) = self.cameras.get_mut(&table_key(pass.key, d.pipeline)) {
                    c.1 = frame;
                }
            }
        }
    }

    pub fn profile(&self) -> String {
        self.priors.to_line()
    }

    pub fn mark_written(&mut self) {
        self.priors.mark_clean();
    }

    pub fn warm(&mut self, line: &str) -> bool {
        self.priors.warm(line)
    }

    pub fn set_weights(&mut self, line: &str) -> bool {
        match Weights::from_line(line) {
            Some(w) => {
                self.weights = w;
                true
            }
            None => false,
        }
    }
}

fn draw_range(pass: &HudPass, n: usize) -> core::ops::Range<usize> {
    let start = (pass.draw_start as usize).min(n);
    start..start.saturating_add(pass.draw_count as usize).min(n)
}

#[cfg(test)]
mod tests;
