// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The labels held against the screen. While the camera turns, everything in the world moves
//! across the presented image by about the same amount, whatever its depth; what holds still
//! there is drawn in screen space: the HUD (or a first-person weapon). On a low-resolution luma
//! copy of the presented frame, a frame or two late, the share of textured cells that hold still
//! under rotation measures the HUD. While skipping, most of it should be gone.

use super::*;

/// Grid cells the camera must sweep the screen by for a frame pair to count, and at most (more is
/// a cut, not a turn).
const MIN_SHIFT_CELLS: f32 = 1.5;
const MAX_SHIFT_SHARE: f32 = 0.33;
/// Luma steps: a cell's 3x3 neighbourhood spans at least this to be textured; it changes by at
/// most this to hold still.
const MIN_CONTRAST: u8 = 12;
const STILL_DIFF: u8 = 4;
const EMA: f32 = 0.1;
/// A cell is HUD once it held still this often while textured.
const STILL_SHARE: f32 = 0.6;
const TEXTURED_SHARE: f32 = 0.3;
/// Turning frames before a verdict, unskipped and skipped.
const VERDICT_FRAMES: u32 = 30;
const SKIPPED_VERDICT_FRAMES: u32 = 20;
/// Below this share of the screen there is no HUD to speak of.
const MIN_HUD_SHARE: f32 = 0.005;
const MIN_COMPARED_SHARE: f32 = 0.01;
/// Skipping that leaves more than this share of the HUD on screen missed it; without a measure of
/// the HUD from before skipping, this share of the screen.
const MAX_RESIDUE: f32 = 0.7;
const MAX_UNCOMPARED_RESIDUE: f32 = 0.04;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Unknown,
    /// A HUD is on screen while the labels find one, and is gone while skipping.
    Agrees,
    /// Skipping left the HUD on screen.
    Residue,
    /// The labels find HUD draws, the screen shows none.
    NoHud,
}

#[derive(Default)]
struct StillMap {
    still: Vec<f32>,
    turns: u32,
}

impl StillMap {
    fn reset(&mut self, cells: usize) {
        self.still.clear();
        self.still.resize(cells, 0.0);
        self.turns = 0;
    }
}

#[derive(Default)]
pub struct CrossCheck {
    previous: Vec<u8>,
    width: u32,
    height: u32,
    seq: u32,
    camera: Option<[f32; 16]>,
    textured: Vec<f32>,
    unskipped: StillMap,
    skipped: StillMap,
    hud_share: f32,
    residue: f32,
}

/// How many grid cells the world moved across the screen between two cameras, from rotation
/// alone: where the earlier view axis lands under the later camera, against where its own does.
pub fn rotation_shift(before: &[f32; 16], after: &[f32; 16], width: u32, height: u32) -> Option<f32> {
    let axis = |m: &[f32; 16]| {
        let a = [m[12], m[13], m[14]];
        let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        (len > 1e-6).then(|| [a[0] / len, a[1] / len, a[2] / len])
    };
    let project = |m: &[f32; 16], d: [f32; 3]| {
        let row = |r: usize| m[r * 4] * d[0] + m[r * 4 + 1] * d[1] + m[r * 4 + 2] * d[2];
        let w = row(3);
        (w > 1e-3).then(|| (row(0) / w, row(1) / w))
    };
    let (from, to) = (axis(before)?, axis(after)?);
    let (x0, y0) = project(after, from)?;
    let (x1, y1) = project(after, to)?;
    let dx = (x0 - x1) * width as f32 / 2.0;
    let dy = (y0 - y1) * height as f32 / 2.0;
    Some((dx * dx + dy * dy).sqrt())
}

fn contrast(luma: &[u8], width: usize, height: usize, x: usize, y: usize) -> u8 {
    let (mut lo, mut hi) = (u8::MAX, 0u8);
    for yy in y.saturating_sub(1)..(y + 2).min(height) {
        for xx in x.saturating_sub(1)..(x + 2).min(width) {
            let v = luma[yy * width + xx];
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    hi - lo
}

fn share(map: &StillMap, textured: &[f32]) -> f32 {
    if map.still.is_empty() {
        return 0.0;
    }
    let hud = map.still.iter().zip(textured).filter(|(&s, &t)| t >= TEXTURED_SHARE && s >= STILL_SHARE).count();
    hud as f32 / map.still.len() as f32
}

impl CrossCheck {
    pub fn hud_share(&self) -> f32 {
        self.hud_share
    }

    pub fn residue(&self) -> f32 {
        self.residue
    }

    /// The table started or stopped skipping: what skipping leaves is measured afresh.
    pub fn plan_changed(&mut self) {
        let cells = self.textured.len();
        self.skipped.reset(cells);
    }

    pub fn update(&mut self, f: &HudFrame, skipping: bool, labelled_hud: bool) -> Verdict {
        if let Some(g) = &f.grid {
            self.take(g, skipping);
        }
        self.hud_share = share(&self.unskipped, &self.textured);
        self.residue = share(&self.skipped, &self.textured);
        if skipping && self.skipped.turns >= SKIPPED_VERDICT_FRAMES {
            let baseline = self.unskipped.turns >= VERDICT_FRAMES && self.hud_share >= MIN_COMPARED_SHARE;
            return match baseline {
                true if self.residue > MAX_RESIDUE * self.hud_share => Verdict::Residue,
                true => Verdict::Agrees,
                false if self.residue >= MAX_UNCOMPARED_RESIDUE => Verdict::Residue,
                false => Verdict::Unknown,
            };
        }
        if !skipping && self.unskipped.turns >= VERDICT_FRAMES && labelled_hud {
            return if self.hud_share < MIN_HUD_SHARE { Verdict::NoHud } else { Verdict::Agrees };
        }
        Verdict::Unknown
    }

    fn take(&mut self, g: &Grid, skipping: bool) {
        let (w, h) = (g.width as usize, g.height as usize);
        if w < 3 || h < 3 || g.luma.len() != w * h {
            return;
        }
        if g.width != self.width || g.height != self.height {
            self.width = g.width;
            self.height = g.height;
            self.textured.clear();
            self.textured.resize(w * h, 0.0);
            self.unskipped.reset(w * h);
            self.skipped.reset(w * h);
            self.previous.clear();
        }
        let consecutive = !self.previous.is_empty() && g.seq == self.seq.wrapping_add(1);
        let shift = match (consecutive, self.camera, g.camera) {
            (true, Some(a), Some(b)) => rotation_shift(&a, &b, g.width, g.height),
            _ => None,
        };
        let turning =
            shift.is_some_and(|s| (MIN_SHIFT_CELLS..=MAX_SHIFT_SHARE * g.width.max(g.height) as f32).contains(&s));
        if turning {
            let map = if skipping { &mut self.skipped } else { &mut self.unskipped };
            map.turns += 1;
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let textured = contrast(g.luma, w, h, x, y) >= MIN_CONTRAST;
                    self.textured[i] += EMA * (f32::from(u8::from(textured)) - self.textured[i]);
                    if textured {
                        let still = g.luma[i].abs_diff(self.previous[i]) <= STILL_DIFF;
                        map.still[i] += EMA * (f32::from(u8::from(still)) - map.still[i]);
                    }
                }
            }
        }
        self.previous.clear();
        self.previous.extend_from_slice(g.luma);
        self.seq = g.seq;
        self.camera = g.camera;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::matrix::mul;
    use crate::signals::matrix::test_util::{perspective, view};

    #[test]
    fn rotation_moves_the_world_and_translation_alone_does_not() {
        let p = perspective(60.0, 2.0, 0.1);
        let a = mul(&p, &view([0.0, 0.0, 0.0], 0.0));
        let turned = mul(&p, &view([0.0, 0.0, 0.0], 5.0));
        let moved = mul(&p, &view([3.0, 0.0, -2.0], 0.0));
        let s = rotation_shift(&a, &turned, 64, 32).unwrap();
        assert!(s > 2.0, "{s}");
        assert!(rotation_shift(&a, &moved, 64, 32).unwrap() < 1e-3);
        assert!(rotation_shift(&a, &mul(&p, &view([0.0; 3], 120.0)), 64, 32).is_none(), "behind: a cut");
    }
}
