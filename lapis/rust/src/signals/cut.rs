// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Whether this frame continues the last one: camera jumps, the frame's pass structure changing,
//! and the 3D scene appearing or going (menus, loading screens, videos). A cut means nothing of the
//! frame before lines up with this one: frame generation shows the real frame instead.

pub const CUT_CAMERA_TURN: u32 = 1 << 0;
pub const CUT_CAMERA_TRAVEL: u32 = 1 << 1;
pub const CUT_LENS: u32 = 1 << 2;
pub const CUT_STRUCTURE: u32 = 1 << 3;
pub const CUT_SCENE_GONE: u32 = 1 << 4;
pub const CUT_SCENE_BACK: u32 = 1 << 5;
pub const CUT_CAMERA_LOST: u32 = 1 << 6;

pub const STATE_UNKNOWN: u8 = 0;
/// A 3D scene drawn with a moving or still camera.
pub const STATE_SCENE: u8 = 1;
/// No depth-writing scene: a menu, loading screen or video.
pub const STATE_FLAT: u8 = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct CutInput {
    /// Draws that wrote depth at the scene's size this frame.
    pub scene_depth_writes: u32,
    /// The pass graph's shape, hashed (`passes::structure_hash`).
    pub structure: u64,
    /// Lapis's S/D shape signature, 0 without S.
    pub signature: u32,
    pub camera_found: bool,
    pub turn: f32,
    pub travel: f32,
    pub zoom: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CutFrame {
    pub cut: bool,
    pub reasons: u32,
    pub state: u8,
    pub frames_since_cut: u32,
    pub confidence: f32,
}

const MOTION_WINDOW: usize = 30;
/// Turns beyond this (degrees per frame) are always a cut: no game pans that fast.
const TURN_ALWAYS: f32 = 30.0;
const TURN_FACTOR: f32 = 8.0;
const TURN_FLOOR: f32 = 4.0;
const TRAVEL_FACTOR: f32 = 12.0;
const LENS_JUMP: f32 = 0.15;
/// Depth writes below this are not a 3D scene (a HUD's few depth-tested quads).
const SCENE_DRAWS: u32 = 4;

#[derive(Default)]
pub struct CutDetector {
    turns: Vec<f32>,
    travels: Vec<f32>,
    structure: u64,
    signature: u32,
    state: u8,
    camera_seen: bool,
    since: u32,
    frames: u32,
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    s[s.len() / 2]
}

fn push(v: &mut Vec<f32>, x: f32) {
    if v.len() == MOTION_WINDOW {
        v.remove(0);
    }
    v.push(x);
}

impl CutDetector {
    pub fn update(&mut self, i: &CutInput) -> CutFrame {
        self.frames += 1;
        let mut reasons = 0u32;
        let state = if i.scene_depth_writes >= SCENE_DRAWS { STATE_SCENE } else { STATE_FLAT };
        if self.state != STATE_UNKNOWN && state != self.state {
            reasons |= if state == STATE_FLAT { CUT_SCENE_GONE } else { CUT_SCENE_BACK };
        }
        if self.state == STATE_SCENE && state == STATE_SCENE {
            if self.structure != 0 && i.structure != self.structure && i.signature != self.signature {
                reasons |= CUT_STRUCTURE;
            }
            if i.camera_found {
                let typical_turn = median(&self.turns);
                if i.turn > TURN_ALWAYS || (i.turn > TURN_FLOOR && i.turn > TURN_FACTOR * typical_turn) {
                    reasons |= CUT_CAMERA_TURN;
                }
                // World units differ per game: travel is judged against the game's own.
                let typical_travel = median(&self.travels);
                let peak_travel = self.travels.iter().copied().fold(0.0, f32::max);
                if self.travels.len() >= 8
                    && typical_travel > 0.0
                    && i.travel > (TRAVEL_FACTOR * typical_travel).max(2.0 * peak_travel)
                {
                    reasons |= CUT_CAMERA_TRAVEL;
                }
                if i.zoom > LENS_JUMP {
                    reasons |= CUT_LENS;
                }
            } else if self.camera_seen {
                reasons |= CUT_CAMERA_LOST;
            }
        }
        if reasons & (CUT_CAMERA_TURN | CUT_CAMERA_TRAVEL | CUT_LENS) == 0 && i.camera_found {
            push(&mut self.turns, i.turn);
            push(&mut self.travels, i.travel);
        }
        if reasons != 0 {
            self.since = 0;
            if reasons & (CUT_SCENE_GONE | CUT_SCENE_BACK | CUT_STRUCTURE) != 0 {
                self.turns.clear();
                self.travels.clear();
            }
        } else {
            self.since += 1;
        }
        self.camera_seen = i.camera_found && state == STATE_SCENE;
        self.state = state;
        self.structure = i.structure;
        self.signature = i.signature;
        CutFrame {
            cut: reasons != 0,
            reasons,
            state,
            frames_since_cut: self.since,
            confidence: (self.frames.min(8) as f32) / 8.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(turn: f32, travel: f32) -> CutInput {
        CutInput {
            scene_depth_writes: 400,
            structure: 0xabc,
            signature: 7,
            camera_found: true,
            turn,
            travel,
            zoom: 0.0,
        }
    }

    fn settle(d: &mut CutDetector) {
        for k in 0..20 {
            assert!(!d.update(&scene(0.5 + 0.1 * (k % 3) as f32, 0.05)).cut);
        }
    }

    #[test]
    fn steady_play_is_never_a_cut_and_a_fast_pan_is_not_either() {
        let mut d = CutDetector::default();
        settle(&mut d);
        // A flick in a shooter: 3 degrees a frame is fast but continuous.
        let out = d.update(&scene(3.0, 0.05));
        assert!(!out.cut, "{:#x}", out.reasons);
    }

    #[test]
    fn a_camera_teleport_or_switch_is_a_cut() {
        let mut d = CutDetector::default();
        settle(&mut d);
        assert_eq!(d.update(&scene(0.4, 5.0)).reasons, CUT_CAMERA_TRAVEL);
        settle(&mut d);
        assert_eq!(d.update(&scene(45.0, 0.0)).reasons & CUT_CAMERA_TURN, CUT_CAMERA_TURN);
        settle(&mut d);
        let mut zoom = scene(0.4, 0.05);
        zoom.zoom = 0.4;
        assert_eq!(d.update(&zoom).reasons, CUT_LENS);
        let after = d.update(&scene(0.4, 0.05));
        assert!(!after.cut && after.frames_since_cut == 1);
    }

    #[test]
    fn a_menu_over_a_frozen_scene_and_back_are_cuts() {
        let mut d = CutDetector::default();
        settle(&mut d);
        let menu = CutInput { scene_depth_writes: 0, structure: 0x1, signature: 0, ..Default::default() };
        let out = d.update(&menu);
        assert!(out.cut && out.reasons == CUT_SCENE_GONE && out.state == STATE_FLAT);
        assert!(!d.update(&menu).cut, "a still menu continues itself");
        let back = d.update(&scene(0.3, 0.05));
        assert_eq!(back.reasons, CUT_SCENE_BACK);
    }

    #[test]
    fn a_new_pass_structure_with_new_images_is_a_cut() {
        let mut d = CutDetector::default();
        settle(&mut d);
        let mut other = scene(0.4, 0.05);
        other.structure = 0xdef;
        other.signature = 9;
        assert_eq!(d.update(&other).reasons, CUT_STRUCTURE);
        // Same images, another structure (a particle pass coming and going): not a cut.
        let mut flicker = scene(0.4, 0.05);
        flicker.structure = 0x123;
        flicker.signature = 9;
        assert!(!d.update(&flicker).cut);
    }
}
