// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! What of the game's G-buffer frame generation can trust this frame: a confidence per channel
//! (depth, velocity, camera, HUD-less scene), the camera reprojection from this frame's clip space
//! to the previous frame's, and a per-package profile that lets a known game start trusted.
//! Decided from the frames before; the engine applies the result to the next.

use core::mem::{offset_of, size_of};

use crate::model::is_velocity_format;

/// The frame's taps, flattened by luwes/gbuffer.cpp.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GBufferInput {
    pub signature: u32,
    pub scene_width: u32,
    pub scene_height: u32,
    pub depth_width: u32,
    pub depth_height: u32,
    pub depth_format: u32,
    pub velocity_width: u32,
    pub velocity_height: u32,
    pub velocity_format: u32,
    /// Depth clear value of the scene's pass; NaN when the pass loads or none was seen.
    pub clear_depth: f32,
    pub has_scene: bool,
    pub has_depth: bool,
    pub depth_sampled: bool,
    pub reversed: bool,
    pub has_velocity: bool,
    /// The game resolves a jittered scene: velocity is what its TAA reads.
    pub temporal: bool,
    pub camera_known: bool,
    pub camera_matrices: bool,
    pub held: bool,
    pub game_generates: bool,
    /// `vp` holds the camera's position; a camera-relative one only rotates.
    pub camera_world: bool,
    pub _pad: u8,
    /// Focal scales and depth terms of P (view distance = b / (d - c)).
    pub sx: f32,
    pub sy: f32,
    pub c: f32,
    pub b: f32,
    /// Row-major view-projection, jitter included, and that jitter in NDC.
    pub vp: [f32; 16],
    pub jitter: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct GBufferOutput {
    pub depth: f32,
    pub velocity: f32,
    pub camera: f32,
    pub scene: f32,
    /// Row-major: previous clip = reproject * current clip, both without jitter.
    pub reproject: [f32; 16],
    pub jitter: [f32; 2],
    pub previous_jitter: [f32; 2],
    pub reproject_valid: bool,
    /// The camera jumped: nothing from the frame before lines up with this one.
    pub cut: bool,
    /// The stored depth range: 0 reversed, 1 forward, 2 forward GL clip.
    pub range: u8,
    pub _pad: u8,
}

/// framefx::luwes::GBufferState, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GBufferTracker {
    signature: u32,
    streak: u32,
    depth_streak: u32,
    velocity_streak: u32,
    camera_streak: u32,
    /// Frames of trust a matching profile grants before any are seen.
    warm: u32,
    previous_vp: [f32; 16],
    previous_jitter: [f32; 2],
    previous_sx: f32,
    previous_sy: f32,
    have_previous: bool,
    _pad: [u8; 3],
}

const _: () = assert!(size_of::<GBufferInput>() == 140 && offset_of!(GBufferInput, vp) == 68);
const _: () = assert!(size_of::<GBufferOutput>() == 100 && offset_of!(GBufferOutput, range) == 98);
const _: () = assert!(size_of::<GBufferTracker>() == 108);

impl Default for GBufferTracker {
    fn default() -> Self {
        GBufferTracker {
            signature: 0,
            streak: 0,
            depth_streak: 0,
            velocity_streak: 0,
            camera_streak: 0,
            warm: 0,
            previous_vp: [0.0; 16],
            previous_jitter: [0.0; 2],
            previous_sx: 0.0,
            previous_sy: 0.0,
            have_previous: false,
            _pad: [0; 3],
        }
    }
}

/// Frames a channel must hold before it is fully trusted.
const SETTLE: u32 = 30;
/// Largest relative focal change still the same lens (zoom aims are cuts).
const LENS_SLACK: f32 = 0.01;
/// Most a camera-relative view-projection is trusted: it carries the turn, not the move.
const ROTATION_ONLY: f32 = 0.3;
/// A reprojection moving the screen centre further than this (NDC) is a cut.
const CUT_NDC: f32 = 0.6;

fn ramp(streak: u32) -> f32 {
    (streak.min(SETTLE) as f32) / SETTLE as f32
}

fn aspect(w: u32, h: u32) -> f32 {
    if w == 0 || h == 0 {
        0.0
    } else {
        w.max(h) as f32 / w.min(h) as f32
    }
}

fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            let mut s = 0.0f64;
            for k in 0..4 {
                s += f64::from(a[r * 4 + k]) * f64::from(b[k * 4 + c]);
            }
            m[r * 4 + c] = s as f32;
        }
    }
    m
}

/// General 4x4 inverse in f64; None when singular.
pub fn invert(m: &[f32; 16]) -> Option<[f32; 16]> {
    let mut a = [[0.0f64; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = f64::from(m[r * 4 + c]);
        }
        a[r][4 + r] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let p = a[col][col];
        for v in a[col].iter_mut() {
            *v /= p;
        }
        let pivot_row = a[col];
        for (r, row) in a.iter_mut().enumerate() {
            let f = row[col];
            if r != col && f != 0.0 {
                for (v, p) in row.iter_mut().zip(pivot_row) {
                    *v -= f * p;
                }
            }
        }
    }
    let mut out = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = a[r][4 + c] as f32;
        }
    }
    Some(out)
}

/// The view-projection without its jitter: the jitter adds `j * w` to clip x and y.
pub fn unjittered(vp: &[f32; 16], jitter: [f32; 2]) -> [f32; 16] {
    let mut m = *vp;
    for c in 0..4 {
        m[c] -= jitter[0] * vp[12 + c];
        m[4 + c] -= jitter[1] * vp[12 + c];
    }
    m
}

/// Where the reprojection sends the screen centre at a mid depth, in NDC; infinite when it lands
/// behind the previous camera.
fn centre_shift(r: &[f32; 16]) -> f32 {
    // Clip point (0, 0, 0.5, 1): NDC centre at mid depth.
    let p = [0.0f32, 0.0, 0.5, 1.0];
    let mut q = [0.0f32; 4];
    for (row, out) in q.iter_mut().enumerate() {
        *out = (0..4).map(|k| r[row * 4 + k] * p[k]).sum();
    }
    if q[3] < 1e-6 {
        return f32::INFINITY;
    }
    (q[0] / q[3]).hypot(q[1] / q[3])
}

/// Range of the stored depth: the projection's say, else the clear value's.
fn range_of(i: &GBufferInput, gl: bool) -> u8 {
    if i.reversed {
        0
    } else if gl {
        2
    } else {
        1
    }
}

/// Whether the scene pass's clear agrees with the range: reversed clears to 0, forward to 1.
fn clear_agrees(i: &GBufferInput) -> bool {
    if i.clear_depth.is_nan() {
        return true;
    }
    if i.reversed {
        i.clear_depth <= 0.01
    } else {
        i.clear_depth >= 0.99
    }
}

impl GBufferTracker {
    pub fn reset(&mut self) {
        *self = GBufferTracker::default();
    }

    /// Trust a profile learned on an earlier run, until the game shows otherwise.
    pub fn warm_from(&mut self, p: &Profile) {
        self.signature = p.signature;
        self.warm = SETTLE / 2;
    }

    pub fn update(&mut self, i: &GBufferInput, gl: bool) -> GBufferOutput {
        let mut out = GBufferOutput { range: range_of(i, gl), ..Default::default() };
        if i.held || i.game_generates || !i.has_scene {
            // Nothing new this frame; the camera history stays as it was.
            return out;
        }
        if i.signature != self.signature {
            let warm = if self.signature == 0 { self.warm } else { 0 };
            *self = GBufferTracker { signature: i.signature, warm, ..Default::default() };
        }
        self.streak += 1;
        let settle = |s: u32, w: u32| ramp(s + w);

        out.scene = settle(self.streak, self.warm);

        let depth_shape = i.has_depth
            && i.depth_width > 0
            && (aspect(i.depth_width, i.depth_height) - aspect(i.scene_width, i.scene_height)).abs()
                < 0.02 * aspect(i.scene_width, i.scene_height).max(1.0);
        let depth_ok = depth_shape && clear_agrees(i);
        self.depth_streak = if depth_ok { self.depth_streak + 1 } else { 0 };
        if depth_ok {
            let readable = if i.depth_sampled { 1.0 } else { 0.7 };
            out.depth = readable * (0.5 + 0.5 * settle(self.depth_streak, self.warm));
        }

        let velocity_ok = i.has_velocity
            && is_velocity_format(i.velocity_format)
            && i.velocity_width.abs_diff(i.depth_width) <= 2
            && i.velocity_height.abs_diff(i.depth_height) <= 2;
        self.velocity_streak = if velocity_ok { self.velocity_streak + 1 } else { 0 };
        if velocity_ok {
            let temporal = if i.temporal { 1.0 } else { 0.6 };
            out.velocity = temporal * (0.4 + 0.6 * settle(self.velocity_streak, self.warm));
        }

        out.jitter = i.jitter;
        let lens_steady = self.previous_sx > 0.0
            && ((i.sx - self.previous_sx) / self.previous_sx).abs() < LENS_SLACK
            && ((i.sy - self.previous_sy) / self.previous_sy).abs() < LENS_SLACK;
        if i.camera_matrices {
            let current = unjittered(&i.vp, i.jitter);
            if self.have_previous && lens_steady {
                if let Some(inv) = invert(&current) {
                    let r = mul(&self.previous_vp, &inv);
                    let shift = centre_shift(&r);
                    if r.iter().all(|v| v.is_finite()) {
                        out.cut = shift.is_nan() || shift > CUT_NDC;
                        if !out.cut {
                            out.reproject = r;
                            out.previous_jitter = self.previous_jitter;
                            out.reproject_valid = true;
                        }
                    }
                }
            }
            self.camera_streak = if out.reproject_valid { self.camera_streak + 1 } else { 0 };
            if out.reproject_valid {
                // Without the camera's position the reprojection misses its translation: an orbit
                // camera (third person) moves the scene more than it turns it.
                let trust = if i.camera_world { 1.0 } else { ROTATION_ONLY };
                out.camera = trust * (0.5 + 0.5 * settle(self.camera_streak, self.warm));
            }
            self.previous_vp = current;
            self.previous_jitter = i.jitter;
            self.have_previous = true;
        } else {
            self.camera_streak = 0;
            self.have_previous = false;
            if i.camera_known && lens_steady {
                // P alone: depth linearises, but motion needs the view too.
                out.camera = 0.25;
            }
        }
        if i.camera_known {
            self.previous_sx = i.sx;
            self.previous_sy = i.sy;
        }
        if out.cut {
            out.depth *= 0.5;
            out.velocity *= 0.5;
        }
        out
    }
}

/// What a package's G-buffer looked like when it was last trusted; one text line on disk.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Profile {
    pub signature: u32,
    pub depth_format: u32,
    pub velocity_format: u32,
    pub reversed: bool,
}

impl Profile {
    pub fn of(i: &GBufferInput) -> Profile {
        Profile {
            signature: i.signature,
            depth_format: if i.has_depth { i.depth_format } else { 0 },
            velocity_format: if i.has_velocity { i.velocity_format } else { 0 },
            reversed: i.reversed,
        }
    }

    pub fn to_line(self) -> String {
        format!(
            "gbuffer1 {:08x} {} {} {}",
            self.signature, self.depth_format, self.velocity_format, self.reversed as u8
        )
    }

    pub fn from_line(line: &str) -> Option<Profile> {
        let mut w = line.split_whitespace();
        if w.next()? != "gbuffer1" {
            return None;
        }
        let p = Profile {
            signature: u32::from_str_radix(w.next()?, 16).ok()?,
            depth_format: w.next()?.parse().ok()?,
            velocity_format: w.next()?.parse().ok()?,
            reversed: w.next()? == "1",
        };
        (p.signature != 0).then_some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perspective(fov_y: f32, aspect: f32, near: f32) -> [f32; 16] {
        // Reversed, infinite: clip z = near, w = -z_view (looking down -z).
        let sy = 1.0 / (fov_y.to_radians() / 2.0).tan();
        let sx = sy / aspect;
        [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 0.0, near, 0.0, 0.0, -1.0, 0.0]
    }

    fn translate(x: f32, y: f32, z: f32) -> [f32; 16] {
        [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0]
    }

    fn yaw(deg: f32) -> [f32; 16] {
        let (s, c) = deg.to_radians().sin_cos();
        [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0]
    }

    fn input(vp: [f32; 16], jitter: [f32; 2]) -> GBufferInput {
        let p = perspective(45.0, 1662.0 / 768.0, 0.25);
        GBufferInput {
            signature: 0x1234,
            scene_width: 1662,
            scene_height: 768,
            depth_width: 1662,
            depth_height: 768,
            depth_format: 129,
            velocity_width: 1662,
            velocity_height: 768,
            velocity_format: 58,
            clear_depth: 0.0,
            has_scene: true,
            has_depth: true,
            depth_sampled: true,
            reversed: true,
            has_velocity: true,
            temporal: true,
            camera_known: true,
            camera_matrices: true,
            camera_world: true,
            sx: p[0],
            sy: p[5],
            c: 0.0,
            b: 0.25,
            vp,
            jitter,
            ..Default::default()
        }
    }

    fn apply(m: &[f32; 16], p: [f32; 4]) -> [f32; 4] {
        let mut q = [0.0; 4];
        for r in 0..4 {
            q[r] = (0..4).map(|k| m[r * 4 + k] * p[k]).sum();
        }
        q
    }

    fn jittered(vp: &[f32; 16], j: [f32; 2]) -> [f32; 16] {
        let mut m = *vp;
        for c in 0..4 {
            m[c] += j[0] * vp[12 + c];
            m[4 + c] += j[1] * vp[12 + c];
        }
        m
    }

    #[test]
    fn inverse_round_trips() {
        let m = mul(&perspective(60.0, 2.0, 0.1), &mul(&yaw(17.0), &translate(1.0, -2.0, 3.0)));
        let i = invert(&m).unwrap();
        let id = mul(&m, &i);
        for r in 0..4 {
            for c in 0..4 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((id[r * 4 + c] - want).abs() < 1e-4, "{r},{c} = {}", id[r * 4 + c]);
            }
        }
        assert!(invert(&[0.0; 16]).is_none());
    }

    #[test]
    fn reprojection_maps_a_world_point_to_where_it_was() {
        let p = perspective(45.0, 1662.0 / 768.0, 0.25);
        let before = mul(&p, &mul(&yaw(0.0), &translate(0.0, 0.0, 0.0)));
        let now = mul(&p, &mul(&yaw(2.0), &translate(0.1, 0.0, 0.05)));
        let (ja, jb) = ([0.0007, -0.0003], [-0.0004, 0.0006]);
        let mut t = GBufferTracker::default();
        t.update(&input(jittered(&before, ja), ja), false);
        let out = t.update(&input(jittered(&now, jb), jb), false);
        assert!(out.reproject_valid && !out.cut);
        assert_eq!(out.previous_jitter, ja);
        // A world point seen by both cameras, without jitter.
        let world = [0.3, -0.2, -5.0, 1.0];
        let cur = apply(&now, world);
        let prev = apply(&before, world);
        let got = apply(&out.reproject, cur);
        for k in 0..2 {
            let (a, b) = (got[k] / got[3], prev[k] / prev[3]);
            assert!((a - b).abs() < 1e-4, "axis {k}: {a} vs {b}");
        }
        let (a, b) = (got[2] / got[3], prev[2] / prev[3]);
        assert!((a - b).abs() < 1e-5, "depth {a} vs {b}");
    }

    #[test]
    fn trust_grows_with_a_steady_frame_and_resets_on_a_new_structure() {
        let vp = perspective(45.0, 1662.0 / 768.0, 0.25);
        let mut t = GBufferTracker::default();
        let first = t.update(&input(vp, [0.0; 2]), false);
        assert!(first.depth > 0.5 && first.depth < 0.6);
        assert_eq!(first.camera, 0.0); // no frame before yet
        let mut last = first;
        for _ in 0..40 {
            last = t.update(&input(vp, [0.0; 2]), false);
        }
        assert!((last.depth - 1.0).abs() < 1e-6 && (last.velocity - 1.0).abs() < 1e-6);
        assert!((last.camera - 1.0).abs() < 1e-6 && (last.scene - 1.0).abs() < 1e-6);
        let mut other = input(vp, [0.0; 2]);
        other.signature = 0x9999;
        let changed = t.update(&other, false);
        assert!(changed.depth < 0.6 && changed.camera == 0.0);
    }

    #[test]
    fn a_profile_starts_a_known_game_half_trusted() {
        let vp = perspective(45.0, 1662.0 / 768.0, 0.25);
        let i = input(vp, [0.0; 2]);
        let p = Profile::from_line(&Profile::of(&i).to_line()).unwrap();
        assert_eq!(p, Profile::of(&i));
        let mut t = GBufferTracker::default();
        t.warm_from(&p);
        let out = t.update(&i, false);
        assert!(out.depth > 0.75);
        assert!(Profile::from_line("gbuffer1 0 1 2 0").is_none());
        assert!(Profile::from_line("junk").is_none());
    }

    #[test]
    fn a_jump_is_a_cut_and_lowers_trust() {
        let p = perspective(45.0, 1662.0 / 768.0, 0.25);
        let mut t = GBufferTracker::default();
        for _ in 0..5 {
            t.update(&input(p, [0.0; 2]), false);
        }
        let out = t.update(&input(mul(&p, &yaw(90.0)), [0.0; 2]), false);
        assert!(out.cut && !out.reproject_valid && out.camera == 0.0);
    }

    #[test]
    fn a_zoom_is_not_reprojected() {
        let mut t = GBufferTracker::default();
        let wide = input(perspective(45.0, 1662.0 / 768.0, 0.25), [0.0; 2]);
        let mut zoomed = input(perspective(30.0, 1662.0 / 768.0, 0.25), [0.0; 2]);
        t.update(&wide, false);
        zoomed.sx = zoomed.vp[0];
        zoomed.sy = zoomed.vp[5];
        let out = t.update(&zoomed, false);
        assert!(!out.reproject_valid);
    }

    #[test]
    fn clear_value_against_the_range_vetoes_depth() {
        let mut i = input(perspective(45.0, 2.0, 0.25), [0.0; 2]);
        i.clear_depth = 1.0; // a forward clear on a reversed claim
        let out = GBufferTracker::default().update(&i, false);
        assert_eq!(out.depth, 0.0);
        i.reversed = false;
        let out = GBufferTracker::default().update(&i, true);
        assert!(out.depth > 0.0 && out.range == 2);
    }

    #[test]
    fn velocity_needs_the_depth_size_and_a_velocity_format() {
        let mut i = input(perspective(45.0, 2.0, 0.25), [0.0; 2]);
        i.velocity_width = 800;
        assert_eq!(GBufferTracker::default().update(&i, false).velocity, 0.0);
        let mut j = input(perspective(45.0, 2.0, 0.25), [0.0; 2]);
        j.velocity_format = 37;
        assert_eq!(GBufferTracker::default().update(&j, false).velocity, 0.0);
        let mut k = input(perspective(45.0, 2.0, 0.25), [0.0; 2]);
        k.temporal = false;
        let v = GBufferTracker::default().update(&k, false).velocity;
        assert!(v > 0.0 && v < 0.5);
    }

    #[test]
    fn held_or_game_generated_frames_change_nothing() {
        let p = perspective(45.0, 2.0, 0.25);
        let mut t = GBufferTracker::default();
        t.update(&input(p, [0.0; 2]), false);
        let mut held = input(p, [0.0; 2]);
        held.held = true;
        assert_eq!(t.update(&held, false).depth, 0.0);
        assert!(t.update(&input(p, [0.0; 2]), false).reproject_valid);
    }
}
