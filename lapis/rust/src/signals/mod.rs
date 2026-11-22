// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Engine signals for frame generation, found in each run from the game's own graphics calls: the
//! camera (current, previous, next frame's, jitter), scene cuts, the post-processing chain and the
//! colour before it, transparency, render scale, per-object stencil, the simulation clock, object
//! motion and exposure. Each comes with a confidence; frame N's statistics decide what frame N + 1
//! trusts. lapis/signals.cpp flattens the recorder's frame into `SignalInput`.

pub mod camera;
pub mod clock;
pub mod cut;
pub mod exposure;
pub mod jitter;
pub mod matrix;
pub mod next_camera;
pub mod objects;
pub mod passes;
pub mod velocity;

use core::mem::size_of;
use core::slice;

use crate::model::{Frame, Roles, View};
use camera::{Block, BlockSlot, CameraAddress, CameraTracker, MatrixSlot};
use clock::SimulationClock;
use cut::{CutDetector, CutInput};
use exposure::ExposureTracker;
use jitter::JitterTracker;
use next_camera::NextCameraTracker;
use objects::{DrawConstants, ObjectTracker};
use passes::PassExtra;

/// A uniform block a draw read; its floats are `SignalInput::floats[start..start + len]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SignalBlock {
    pub memory: u64,
    pub offset: u64,
    pub set: u32,
    pub binding: u32,
    pub start: u32,
    pub len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SignalDraw {
    pub key: u64,
    pub start: u32,
    pub len: u32,
}

/// 16 floats read from a watched address at `time_ns`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SignalSample {
    pub memory: u64,
    pub offset: u64,
    pub time_ns: i64,
    pub row_major: u32,
    pub floats: [f32; 16],
}

/// An address the recorder reads at each submit until the next present.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignalWatch {
    pub memory: u64,
    pub offset: u64,
    pub row_major: u32,
    pub _pad: u32,
}

#[repr(C)]
pub struct SignalInput {
    pub frame: Frame,
    /// One per step of `frame`.
    pub extras: *const PassExtra,
    pub extra_count: u32,
    pub roles: Roles,
    /// The scene pass's first draw: its uniform blocks (camera, clock).
    pub camera_blocks: *const SignalBlock,
    pub camera_block_count: u32,
    /// The tonemap pass's first draw.
    pub tonemap_blocks: *const SignalBlock,
    pub tonemap_block_count: u32,
    pub floats: *const f32,
    pub float_count: u32,
    pub draws: *const SignalDraw,
    pub draw_count: u32,
    pub samples: *const SignalSample,
    pub sample_count: u32,
    pub present_ns: i64,
    pub scene_width: u32,
    pub scene_height: u32,
}

pub const CAMERA_FOUND: u32 = 1 << 0;
pub const CAMERA_WORLD: u32 = 1 << 1;
pub const CAMERA_JITTER_EXACT: u32 = 1 << 2;
pub const CAMERA_PREVIOUS: u32 = 1 << 3;
pub const CAMERA_PREVIOUS_FROM_GAME: u32 = 1 << 4;
pub const CLOCK_FOUND: u32 = 1 << 0;
pub const CLOCK_PAUSED: u32 = 1 << 1;
pub const CLOCK_DELTA: u32 = 1 << 2;
pub const PRE_POST_UNTONED: u32 = 1 << 0;
pub const PRE_POST_OVERWRITTEN: u32 = 1 << 1;
pub const MAX_OBJECTS: usize = 16;

/// What frame generation may use of this frame, each with a confidence in 0..1. Matrices are
/// row-major, clip = M * world, without jitter. framefx::lapis::EngineSignals, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EngineSignals {
    pub structure: u64,
    pub sim_seconds: f64,
    pub next_seen_before_present_ns: i64,
    pub next_lead_median_ns: i64,
    pub next_own_lead_median_ns: i64,

    pub camera_confidence: f32,
    pub camera_flags: u32,
    pub view_projection: [f32; 16],
    pub previous_view_projection: [f32; 16],
    pub turn_degrees: f32,
    pub travel: f32,
    pub zoom: f32,

    pub next_confidence: f32,
    pub next_valid: u32,
    pub next_hit_rate: f32,
    pub next_early_rate: f32,
    pub next_view_projection: [f32; 16],

    pub jitter_confidence: f32,
    pub jitter_pattern: u32,
    pub jitter_period: u32,
    pub jitter_next_valid: u32,
    pub jitter_ndc: [f32; 2],
    pub jitter_pixels: [f32; 2],
    pub next_jitter_pixels: [f32; 2],

    pub cut: u32,
    pub cut_reasons: u32,
    pub scene_state: u32,
    pub frames_since_cut: u32,
    pub cut_confidence: f32,

    pub post_effects: u32,
    pub pre_post_image: u32,
    pub pre_post_step: i32,
    pub pre_post_flags: u32,
    pub tonemap_step: i32,
    pub scene_draws: u32,
    pub transparent_draws: u32,
    pub transparent_step: i32,
    pub transparent_from: u32,
    pub render_scale: [f32; 2],
    pub drawn: [f32; 2],
    pub stencil_ids: u32,
    pub stencil_writes: u32,

    pub clock_confidence: f32,
    pub clock_flags: u32,
    pub sim_step: f32,
    pub present_interval: f32,

    pub object_confidence: f32,
    pub objects_transformed: u32,
    pub objects_matched: u32,
    pub objects_moving: u32,
    /// NDC x, y and motion dx, dy of the first moving objects.
    pub objects: [[f32; 4]; MAX_OBJECTS],

    pub exposure_confidence: f32,
    pub exposure_gain: f32,
}

impl Default for EngineSignals {
    fn default() -> Self {
        // SAFETY: plain numbers; all-zero is a valid value of every field.
        let mut s: EngineSignals = unsafe { core::mem::zeroed() };
        s.pre_post_step = -1;
        s.tonemap_step = -1;
        s.transparent_step = -1;
        s.exposure_gain = 1.0;
        s
    }
}

const _: () = assert!(size_of::<PassExtra>() == 148);
// Layouts as lapis/signals.cpp asserts them on 64-bit; 32-bit hosts pack 8-byte fields at 4.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<SignalInput>() == 248 && core::mem::offset_of!(SignalInput, roles) == 84);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::offset_of!(SignalInput, present_ns) == 232);
const _: () = assert!(size_of::<SignalBlock>() == 32 && size_of::<SignalDraw>() == 16);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<SignalSample>() == 96 && size_of::<SignalWatch>() == 24);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::offset_of!(EngineSignals, camera_confidence) == 40);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::offset_of!(EngineSignals, next_view_projection) == 204);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::offset_of!(EngineSignals, objects) == 420);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<EngineSignals>() == 688);

/// Per-package memory of where the signals were: a prior the trackers start from, never a rule.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SignalProfile {
    pub signature: u32,
    pub camera: Option<MatrixSlot>,
    pub clock: Option<u32>,
}

impl SignalProfile {
    pub fn to_line(self) -> String {
        let cam = self.camera.map_or("-".to_string(), |s| {
            format!("{}.{}.{}.{}", s.block.set, s.block.binding, s.float, s.row_major as u8)
        });
        let clock = self.clock.map_or("-".to_string(), |c| c.to_string());
        format!("signals1 {:08x} {} {}", self.signature, cam, clock)
    }

    pub fn from_line(line: &str) -> Option<SignalProfile> {
        let mut w = line.split_whitespace();
        if w.next()? != "signals1" {
            return None;
        }
        let signature = u32::from_str_radix(w.next()?, 16).ok()?;
        let cam = w.next()?;
        let camera = if cam == "-" {
            None
        } else {
            let p: Vec<u32> = cam.split('.').map(|x| x.parse().ok()).collect::<Option<_>>()?;
            if p.len() != 4 {
                return None;
            }
            Some(MatrixSlot { block: BlockSlot { set: p[0], binding: p[1] }, float: p[2], row_major: p[3] == 1 })
        };
        let clock = match w.next()? {
            "-" => None,
            c => Some(c.parse().ok()?),
        };
        Some(SignalProfile { signature, camera, clock })
    }
}

#[derive(Default)]
pub struct SignalState {
    camera: CameraTracker,
    next: NextCameraTracker,
    jitter: JitterTracker,
    cut: CutDetector,
    clock: SimulationClock,
    objects: ObjectTracker,
    exposure: ExposureTracker,
    signature: u32,
    profile: SignalProfile,
}

fn blocks(floats: &[f32], list: &[SignalBlock]) -> Vec<Block> {
    let mut out: Vec<Block> = list
        .iter()
        .filter_map(|b| {
            let range = b.start as usize..(b.start as usize).saturating_add(b.len as usize);
            floats.get(range).map(|f| Block {
                slot: BlockSlot { set: b.set, binding: b.binding },
                memory: b.memory,
                offset: b.offset,
                floats: f.to_vec(),
            })
        })
        .collect();
    out.sort_by_key(|b| (b.slot.set, b.slot.binding));
    out
}

fn concat(blocks: &[Block], cap: usize) -> Vec<f32> {
    let mut v = Vec::new();
    for b in blocks {
        let room = cap.saturating_sub(v.len());
        v.extend_from_slice(&b.floats[..b.floats.len().min(room)]);
    }
    v
}

impl SignalState {
    pub fn reset(&mut self) {
        let profile = self.profile;
        *self = SignalState { profile, ..Default::default() };
    }

    pub fn warm(&mut self, p: SignalProfile) {
        self.profile = p;
        self.camera.prefer(p.camera);
        self.clock.prefer(p.clock);
    }

    /// What has been learned, for the package's profile.
    pub fn learned(&self) -> SignalProfile {
        SignalProfile { signature: self.signature, camera: self.camera.chosen(), clock: self.clock.chosen() }
    }

    /// Addresses to read at each submit before the next present.
    pub fn watch(&self) -> Vec<SignalWatch> {
        self.next
            .watch()
            .into_iter()
            .map(|a| SignalWatch { memory: a.memory, offset: a.offset, row_major: a.row_major as u32, _pad: 0 })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        view: &View,
        extras: &[PassExtra],
        roles: &Roles,
        camera_blocks: &[Block],
        tonemap_blocks: &[Block],
        draws: &[DrawConstants],
        samples: &[SignalSample],
        present_ns: i64,
        scene: (u32, u32),
    ) -> EngineSignals {
        let mut s = EngineSignals::default();
        if roles.signature != 0 && roles.signature != self.signature {
            // Another screen with other images: what was learned of the last one does not carry.
            if self.signature != 0 {
                self.camera.reset();
                self.next.reset();
                self.jitter.reset();
                self.objects.reset();
            }
            self.signature = roles.signature;
            if self.profile.signature == roles.signature {
                self.camera.prefer(self.profile.camera);
                self.clock.prefer(self.profile.clock);
            }
        }
        let g = passes::analyse(view, extras, roles);
        s.structure = g.structure;
        s.post_effects = g.post;
        s.pre_post_image = g.pre_post;
        s.pre_post_step = g.pre_post_step;
        s.pre_post_flags = (g.pre_post_untoned as u32 * PRE_POST_UNTONED) | (g.pre_post_overwritten as u32 * PRE_POST_OVERWRITTEN);
        s.tonemap_step = g.tonemap;
        s.scene_draws = g.scene_draws;
        s.transparent_draws = g.transparent_draws;
        s.transparent_step = g.transparent_step;
        s.transparent_from = g.transparent_from;
        s.render_scale = g.render_scale;
        s.drawn = g.drawn;
        s.stencil_ids = g.stencil_ids as u32;
        s.stencil_writes = g.stencil_writes;

        let (w, h) = scene;
        let aspect = if h > 0 { w as f32 / h as f32 } else { 0.0 };
        let cam = self.camera.update(camera_blocks, aspect);
        if cam.found {
            s.camera_confidence = cam.confidence;
            s.camera_flags = CAMERA_FOUND
                | (cam.world as u32 * CAMERA_WORLD)
                | (cam.jitter_exact as u32 * CAMERA_JITTER_EXACT)
                | (cam.previous_valid as u32 * CAMERA_PREVIOUS)
                | (cam.previous_from_game as u32 * CAMERA_PREVIOUS_FROM_GAME);
            s.view_projection = cam.unjittered;
            s.previous_view_projection = cam.previous;
            s.turn_degrees = cam.turn;
            s.travel = cam.travel;
            s.zoom = cam.zoom;

            let j = self.jitter.update(cam.jitter, w, h);
            s.jitter_confidence = j.confidence;
            s.jitter_pattern = u32::from(j.pattern);
            s.jitter_period = j.period;
            s.jitter_next_valid = j.next_valid as u32;
            s.jitter_ndc = cam.jitter;
            s.jitter_pixels = j.pixels;
            s.next_jitter_pixels = j.next;

            for p in samples {
                let a = CameraAddress { memory: p.memory, offset: p.offset, row_major: p.row_major != 0 };
                self.next.observe(a, &p.floats, p.time_ns);
            }
            let n = self.next.present(&cam.jittered, cam.address, aspect, present_ns);
            s.next_valid = n.valid as u32;
            s.next_hit_rate = n.hit_rate;
            s.next_early_rate = n.early_rate;
            s.next_seen_before_present_ns = n.seen_before_present_ns;
            s.next_lead_median_ns = n.lead_median_ns;
            s.next_own_lead_median_ns = n.own_lead_median_ns;
            s.next_confidence = n.confidence;
            if n.valid {
                // Its own offset out, keeping any fixed off-centre shift the current one keeps.
                let shift = matrix::fit_view_projection(&cam.jittered, true, 0.0)
                    .map_or([0.0, 0.0], |f| [f.jx - cam.jitter[0], f.jy - cam.jitter[1]]);
                let next = matrix::fit_view_projection(&n.matrix, true, 0.0);
                s.next_view_projection = match next {
                    Some(f) => matrix::without_offset(&n.matrix, f.jx - shift[0], f.jy - shift[1]),
                    None => n.matrix,
                };
            }
        } else {
            self.jitter.reset();
        }

        let c = self.cut.update(&CutInput {
            scene_depth_writes: g.scene_depth_writes,
            structure: g.structure,
            signature: roles.signature,
            camera_found: cam.found,
            turn: cam.turn,
            travel: cam.travel,
            zoom: cam.zoom,
        });
        s.cut = c.cut as u32;
        s.cut_reasons = c.reasons;
        s.scene_state = u32::from(c.state);
        s.frames_since_cut = c.frames_since_cut;
        s.cut_confidence = c.confidence;

        let k = self.clock.update(&concat(camera_blocks, clock::MAX_FLOATS), present_ns);
        if k.found {
            s.clock_confidence = k.confidence;
            s.clock_flags = CLOCK_FOUND | (k.paused as u32 * CLOCK_PAUSED) | ((k.delta_index >= 0) as u32 * CLOCK_DELTA);
            s.sim_seconds = k.seconds;
            s.sim_step = k.step;
        }
        s.present_interval = k.interval;

        if cam.found && !draws.is_empty() {
            let o = self.objects.update(draws, &cam.unjittered);
            s.object_confidence = o.confidence;
            s.objects_transformed = o.transformed;
            s.objects_matched = o.matched;
            s.objects_moving = o.moving.len() as u32;
            for (slot, m) in s.objects.iter_mut().zip(&o.moving) {
                *slot = [m.ndc[0], m.ndc[1], m.motion[0], m.motion[1]];
            }
        }
        if !tonemap_blocks.is_empty() {
            let e = self.exposure.update(&concat(tonemap_blocks, 256));
            s.exposure_confidence = e.confidence;
            s.exposure_gain = e.gain;
        }
        s
    }
}

/// # Safety
/// Every pointer in `i` holds its count of elements.
pub unsafe fn update_from(state: &mut SignalState, i: &SignalInput) -> EngineSignals {
    unsafe fn parts<'a, T>(p: *const T, n: u32) -> &'a [T] {
        if p.is_null() || n == 0 {
            &[]
        } else {
            // SAFETY: the caller's contract.
            unsafe { slice::from_raw_parts(p, n as usize) }
        }
    }
    // SAFETY: per this function's contract.
    unsafe {
        let view = View::new(&i.frame);
        let floats = parts(i.floats, i.float_count);
        let draws: Vec<DrawConstants> = parts(i.draws, i.draw_count)
            .iter()
            .filter_map(|d| {
                let r = d.start as usize..(d.start as usize).saturating_add(d.len as usize);
                floats.get(r).map(|f| DrawConstants { key: d.key, floats: f.to_vec() })
            })
            .collect();
        state.update(
            &view,
            parts(i.extras, i.extra_count),
            &i.roles,
            &blocks(floats, parts(i.camera_blocks, i.camera_block_count)),
            &blocks(floats, parts(i.tonemap_blocks, i.tonemap_block_count)),
            &draws,
            parts(i.samples, i.sample_count),
            i.present_ns,
            (i.scene_width, i.scene_height),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_round_trip_and_junk_is_refused() {
        let p = SignalProfile {
            signature: 0xbeef,
            camera: Some(MatrixSlot { block: BlockSlot { set: 1, binding: 2 }, float: 16, row_major: false }),
            clock: Some(9),
        };
        assert_eq!(SignalProfile::from_line(&p.to_line()), Some(p));
        let none = SignalProfile { signature: 1, camera: None, clock: None };
        assert_eq!(SignalProfile::from_line(&none.to_line()), Some(none));
        assert!(SignalProfile::from_line("signals1 zz - -").is_none());
        assert!(SignalProfile::from_line("gbuffer1 1 2 3 0").is_none());
    }
}

