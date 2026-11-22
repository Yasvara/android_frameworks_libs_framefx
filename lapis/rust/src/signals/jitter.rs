// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The game's TAA jitter as a sequence of sub-pixel offsets: its period, whether it is a scaled
//! Halton (2, 3) walk, what it will be next frame, and how sure that is. Offsets come in pixels of
//! the scene, from the camera tracker.

pub const PATTERN_NONE: u8 = 0;
/// Repeats with a period but matches no known walk.
pub const PATTERN_PERIODIC: u8 = 1;
/// A scaled, shifted Halton (2, 3) walk.
pub const PATTERN_HALTON: u8 = 2;
/// Moves every frame without repeating within the window.
pub const PATTERN_RANDOM: u8 = 3;

const WINDOW: usize = 64;
const MAX_PERIOD: usize = 32;
/// Below this (pixels) on both axes over the window, there is no jitter.
const STILL_PX: f32 = 0.01;
const MATCH_PX: f32 = 0.002;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JitterFrame {
    /// This frame's offset in scene pixels, x right, y as the NDC y of the projection.
    pub pixels: [f32; 2],
    /// The offset of the next frame, when the sequence repeats.
    pub next: [f32; 2],
    pub next_valid: bool,
    pub period: u32,
    pub pattern: u8,
    /// A constant off-centre shift the projection has besides jitter (pixels).
    pub bias: [f32; 2],
    pub confidence: f32,
}

#[derive(Default)]
pub struct JitterTracker {
    history: Vec<[f32; 2]>,
    period: usize,
    hits: u32,
    misses: u32,
    predicted: Option<[f32; 2]>,
}

pub fn halton(mut i: usize, base: usize) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

fn near(a: [f32; 2], b: [f32; 2], tol: f32) -> bool {
    (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol
}

/// The smallest period the last frames repeat with, needing two full cycles.
fn find_period(h: &[[f32; 2]]) -> usize {
    for p in 1..=MAX_PERIOD {
        if h.len() < 2 * p + 1 {
            break;
        }
        let span = (h.len() - p).min(3 * p);
        let start = h.len() - span;
        if (start..h.len()).all(|i| near(h[i], h[i - p], MATCH_PX)) {
            return p;
        }
    }
    0
}

/// Whether one cycle is Halton (2, 3) at some phase and rotation, each axis at its own scale,
/// sign and offset.
fn is_halton(cycle: &[[f32; 2]]) -> bool {
    let n = cycle.len();
    if n < 4 {
        return false;
    }
    let axis_fits = |axis: usize, rot: usize, phase: usize| -> bool {
        let base = if axis == 0 { 2 } else { 3 };
        let pts: Vec<(f32, f32)> =
            (0..n).map(|k| (halton(phase + k + 1, base), cycle[(rot + k) % n][axis])).collect();
        let (mw, mv) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
        let (mw, mv) = (mw / n as f32, mv / n as f32);
        let (mut sww, mut swv) = (0.0f32, 0.0f32);
        for (w, v) in &pts {
            sww += (w - mw) * (w - mw);
            swv += (w - mw) * (v - mv);
        }
        if sww < 1e-6 {
            return false;
        }
        let s = swv / sww;
        if s.abs() < 1e-3 {
            return false;
        }
        let tol = MATCH_PX.max(0.01 * s.abs());
        pts.iter().all(|(w, v)| (v - mv - s * (w - mw)).abs() <= tol)
    };
    (0..n).any(|rot| (0..=32).any(|phase| axis_fits(0, rot, phase) && axis_fits(1, rot, phase)))
}

impl JitterTracker {
    pub fn reset(&mut self) {
        *self = JitterTracker::default();
    }

    /// One frame's NDC offset of the projection over a `width` x `height` scene.
    pub fn update(&mut self, ndc: [f32; 2], width: u32, height: u32) -> JitterFrame {
        let px = [ndc[0] * width as f32 * 0.5, ndc[1] * height as f32 * 0.5];
        if let Some(p) = self.predicted.take() {
            if near(p, px, MATCH_PX * 4.0) {
                self.hits += 1;
            } else {
                self.misses += 1;
                self.hits = 0;
            }
        }
        if self.history.len() == WINDOW {
            self.history.remove(0);
        }
        self.history.push(px);
        let h = &self.history;
        let mean = |axis: usize| h.iter().map(|v| v[axis]).sum::<f32>() / h.len() as f32;
        let bias = [mean(0), mean(1)];
        let spread = h.iter().fold(0.0f32, |m, v| m.max((v[0] - bias[0]).abs()).max((v[1] - bias[1]).abs()));

        let mut out = JitterFrame { pixels: px, bias, ..Default::default() };
        if h.len() < 3 || spread < STILL_PX {
            out.pattern = PATTERN_NONE;
            out.confidence = if h.len() >= 8 { 1.0 } else { 0.0 };
            self.period = 0;
            return out;
        }
        self.period = find_period(h);
        out.period = self.period as u32;
        if self.period == 0 {
            let moving = h.windows(2).filter(|w| !near(w[0], w[1], MATCH_PX)).count();
            out.pattern = if moving + 1 >= h.len() { PATTERN_RANDOM } else { PATTERN_NONE };
            out.confidence = 0.0;
            return out;
        }
        let cycle = &h[h.len() - self.period..];
        let mean_c = |axis: usize| cycle.iter().map(|v| v[axis]).sum::<f32>() / cycle.len() as f32;
        out.bias = [mean_c(0), mean_c(1)];
        out.pattern = if self.period > 1 && is_halton(cycle) { PATTERN_HALTON } else { PATTERN_PERIODIC };
        out.next = h[h.len() - self.period];
        out.next_valid = true;
        self.predicted = Some(out.next);
        let cycles = (self.history.len() / self.period).min(4) as f32 / 4.0;
        let run = (self.hits.min(16) as f32) / 16.0;
        out.confidence = (0.5 * cycles + 0.5 * run).min(1.0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(t: &mut JitterTracker, n: usize, f: impl Fn(usize) -> [f32; 2]) -> JitterFrame {
        let mut out = JitterFrame::default();
        for i in 0..n {
            out = t.update(f(i), 1600, 720);
        }
        out
    }

    #[test]
    fn unreal_style_halton_8_is_found_and_predicted() {
        // UE: Halton(i, 2), Halton(i, 3) minus 0.5, scaled to NDC by 2 / size, y negated.
        let ndc = |i: usize| {
            let k = i % 8 + 1;
            [(halton(k, 2) - 0.5) * 2.0 / 1600.0, -(halton(k, 3) - 0.5) * 2.0 / 720.0]
        };
        let mut t = JitterTracker::default();
        let out = feed(&mut t, 40, ndc);
        assert_eq!(out.period, 8);
        assert_eq!(out.pattern, PATTERN_HALTON);
        let next = ndc(40);
        assert!((out.next[0] - next[0] * 800.0).abs() < 1e-4 && (out.next[1] - next[1] * 360.0).abs() < 1e-4);
        assert!(out.confidence > 0.9, "{}", out.confidence);
    }

    #[test]
    fn a_shifted_scaled_halton_16_with_an_off_centre_bias_is_still_halton() {
        let ndc = |i: usize| {
            let k = i % 16 + 5;
            [0.3 * (halton(k, 2) - 0.5) * 2.0 / 1600.0 + 0.002, (halton(k, 3) - 0.5) * 2.0 / 720.0]
        };
        let out = feed(&mut JitterTracker::default(), 70, ndc);
        assert_eq!((out.period, out.pattern), (16, PATTERN_HALTON));
        assert!((out.bias[0] - 0.002 * 800.0).abs() < 0.01);
    }

    #[test]
    fn a_four_tap_grid_is_periodic_not_halton() {
        let taps = [[0.25, 0.25], [-0.25, 0.25], [-0.25, -0.25], [0.25, -0.25]];
        let ndc = |i: usize| [taps[i % 4][0] / 800.0, taps[i % 4][1] / 360.0];
        let out = feed(&mut JitterTracker::default(), 20, ndc);
        assert_eq!(out.period, 4);
        assert_ne!(out.pattern, PATTERN_NONE);
        assert!(out.next_valid);
    }

    #[test]
    fn no_jitter_and_random_jitter_are_told_apart() {
        let still = feed(&mut JitterTracker::default(), 20, |_| [0.0, 0.0]);
        assert_eq!(still.pattern, PATTERN_NONE);
        assert!(!still.next_valid);
        let mut seed = 12345u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let values: Vec<[f32; 2]> = (0..64).map(|_| [rnd() / 800.0, rnd() / 360.0]).collect();
        let out = feed(&mut JitterTracker::default(), 64, |i| values[i]);
        assert_eq!(out.pattern, PATTERN_RANDOM);
        assert!(!out.next_valid && out.confidence == 0.0);
    }

    #[test]
    fn a_broken_prediction_drops_confidence() {
        let ndc = |i: usize| {
            let k = i % 8 + 1;
            [(halton(k, 2) - 0.5) * 2.0 / 1600.0, (halton(k, 3) - 0.5) * 2.0 / 720.0]
        };
        let mut t = JitterTracker::default();
        let settled = feed(&mut t, 48, ndc);
        let broken = t.update([0.3 / 800.0, -0.4 / 360.0], 1600, 720);
        assert!(broken.confidence < settled.confidence);
    }
}
