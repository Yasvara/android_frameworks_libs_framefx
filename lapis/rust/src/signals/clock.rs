// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The game's simulation clock, from the constants its scene draws read: a float that grows with
//! the presents (time), and one that equals its step (delta). Frame generation places generated
//! frames on this clock rather than on present times, which carry the game's frame-time jitter.

const WINDOW: usize = 32;
/// Floats watched per frame: the clock lives with the camera, near the start of the block.
pub const MAX_FLOATS: usize = 1024;
/// Plausible time units against seconds: seconds, Unity's t / 20, 2t, 3t, milliseconds.
const SCALES: [f32; 5] = [1.0, 0.05, 2.0, 3.0, 1000.0];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClockFrame {
    pub found: bool,
    /// The float that is time, its unit against seconds, and its value in seconds.
    pub time_index: u32,
    pub scale: f32,
    pub seconds: f64,
    /// Simulation step to this frame (seconds), and the present interval it came in.
    pub step: f32,
    pub interval: f32,
    /// The game's own delta float, when it keeps one.
    pub delta_index: i32,
    /// The clock stood still (pause, menu) while frames came.
    pub paused: bool,
    pub confidence: f32,
}

/// Frames between searches for a clock while none is found: a search fits every float.
const SEARCH_EVERY: u32 = 16;

#[derive(Default)]
pub struct SimulationClock {
    /// The last WINDOW frames' floats, frame-major in a ring: `slots` frames of `width` floats.
    values: Vec<f32>,
    presents: Vec<i64>,
    width: usize,
    head: usize,
    filled: usize,
    frames: u32,
    chosen: Option<(usize, f32)>,
    prior: Option<usize>,
}

impl SimulationClock {
    pub fn prefer(&mut self, index: Option<u32>) {
        self.prior = index.map(|i| i as usize);
    }

    pub fn chosen(&self) -> Option<u32> {
        self.chosen.map(|(k, _)| k as u32)
    }

    /// Float `k` over the frames on file, oldest first, with their present times.
    fn history<'a>(&self, k: usize, h: &'a mut [f32; WINDOW], p: &mut [i64; WINDOW]) -> &'a [f32] {
        for j in 0..self.filled {
            let slot = (self.head + WINDOW - self.filled + j) % WINDOW;
            h[j] = self.values[slot * self.width + k];
            p[j] = self.presents[slot];
        }
        &h[..self.filled]
    }

    /// Of float `k`'s history, the unit against seconds its steps follow the presents with, and
    /// how many steps agree; None when it is not a clock.
    fn fit(&self, k: usize) -> Option<(f32, usize)> {
        let n = self.filled;
        if n < 8 {
            return None;
        }
        let (mut hb, mut p) = ([0.0f32; WINDOW], [0i64; WINDOW]);
        let h = self.history(k, &mut hb, &mut p);
        let mut ratios = [0.0f32; WINDOW];
        let mut m = 0;
        for k in 1..n {
            let d = h[k] - h[k - 1];
            let dt = (p[k] - p[k - 1]) as f32 * 1e-9;
            // A wrap or reset is not counted either way; a still step neither.
            if d > 0.0 && d.is_finite() && dt > 0.0 {
                ratios[m] = d / dt;
                m += 1;
            }
        }
        if m < n / 2 {
            return None;
        }
        let ratios = &mut ratios[..m];
        ratios.sort_by(f32::total_cmp);
        let r = ratios[m / 2];
        // Frame pacing scatters the ratio; the unit is whichever plausible one it sits near.
        let scale = *SCALES.iter().min_by(|a, b| ((r / *a).ln().abs()).total_cmp(&(r / *b).ln().abs()))?;
        if (r / scale).ln().abs() > 0.35 {
            return None;
        }
        let agree = ratios.iter().filter(|q| ((*q / scale).ln()).abs() < 0.7).count();
        (agree * 4 >= m * 3).then_some((scale, agree))
    }

    /// One frame's constants and its present time.
    pub fn update(&mut self, floats: &[f32], present_ns: i64) -> ClockFrame {
        let n = floats.len().min(MAX_FLOATS);
        if self.width != n || self.presents.is_empty() {
            self.width = n;
            self.values = vec![0.0; n * WINDOW];
            self.presents = vec![0; WINDOW];
            self.head = 0;
            self.filled = 0;
            self.chosen = None;
        }
        self.values[self.head * n..(self.head + 1) * n].copy_from_slice(&floats[..n]);
        self.presents[self.head] = present_ns;
        self.head = (self.head + 1) % WINDOW;
        self.filled = (self.filled + 1).min(WINDOW);
        self.frames += 1;

        // Keep a clock once found while it still fits; else look again now and then, seconds first.
        let keep = self.chosen.filter(|&(k, _)| self.fit(k).is_some());
        let prior = self.prior.filter(|&k| k < n).and_then(|k| self.fit(k).map(|(s, _)| (k, s)));
        let search = keep.is_none() && prior.is_none() && self.frames.is_multiple_of(SEARCH_EVERY);
        self.chosen = keep.or(prior).or_else(|| {
            if !search {
                return None;
            }
            let mut best: Option<(usize, f32, usize)> = None;
            for k in 0..n {
                if let Some((scale, agree)) = self.fit(k) {
                    let rank = |s: f32| SCALES.iter().position(|x| *x == s).unwrap_or(9);
                    let better = best.is_none_or(|(_, bs, ba)| agree > ba || (agree == ba && rank(scale) < rank(bs)));
                    if better {
                        best = Some((k, scale, agree));
                    }
                }
            }
            best.map(|(k, s, _)| (k, s))
        });
        let mut out = ClockFrame { delta_index: -1, ..Default::default() };
        let Some((k, scale)) = self.chosen else {
            return out;
        };
        let (mut hb, mut pb) = ([0.0f32; WINDOW], [0i64; WINDOW]);
        let h = self.history(k, &mut hb, &mut pb);
        let m = h.len();
        out.found = true;
        out.time_index = k as u32;
        out.scale = scale;
        out.seconds = f64::from(h[m - 1]) / f64::from(scale);
        if m >= 2 {
            out.step = (h[m - 1] - h[m - 2]) / scale;
            out.interval = (pb[m - 1] - pb[m - 2]) as f32 * 1e-9;
            out.paused = out.step == 0.0;
            // The game's delta: a float equal to the step this frame and the ones before.
            if out.step > 0.0 {
                let depth = (m - 1).min(4);
                let steps: [f32; 4] = core::array::from_fn(|j| if j < depth { (h[m - 1 - j] - h[m - 2 - j]) / scale } else { 0.0 });
                let slot = |back: usize| (self.head + WINDOW - 1 - back) % WINDOW;
                out.delta_index = (0..n)
                    .find(|&d| {
                        (0..depth).all(|j| {
                            let v = self.values[slot(j) * n + d];
                            (v - steps[j]).abs() <= 0.02 * steps[j] + 1e-6
                        })
                    })
                    .map_or(-1, |d| d as i32);
            }
        }
        let filled = (m as f32) / WINDOW as f32;
        out.confidence = filled * if out.delta_index >= 0 { 1.0 } else { 0.8 };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block with clutter, Unity's _Time (t/20, t, 2t, 3t) at 8 and unity_DeltaTime at 12.
    fn unity_block(t: f32, dt: f32, i: usize) -> Vec<f32> {
        let mut b = vec![0.0f32; 64];
        for (k, v) in b.iter_mut().enumerate() {
            *v = ((k * 7 + i) % 13) as f32 * 0.1;
        }
        b[8..12].copy_from_slice(&[t / 20.0, t, 2.0 * t, 3.0 * t]);
        b[12..16].copy_from_slice(&[dt, 1.0 / dt, dt, 1.0 / dt]);
        b[20] = 0.75; // a constant
        b
    }

    #[test]
    fn finds_unity_time_in_seconds_and_its_delta_under_uneven_pacing() {
        let mut c = SimulationClock::default();
        let (mut t, mut present) = (100.0f32, 0i64);
        let mut out = ClockFrame::default();
        for i in 0..40 {
            // Uneven frames: 14-20 ms, with the game stepping its clock by the frame it just had.
            let dt = 0.014 + 0.002 * (i % 4) as f32;
            t += dt;
            present += (dt * 1e9) as i64;
            out = c.update(&unity_block(t, dt, i), present);
        }
        assert!(out.found);
        assert_eq!((out.time_index, out.scale), (9, 1.0));
        assert_eq!(out.delta_index, 12);
        assert!((out.seconds - f64::from(t)).abs() < 1e-3);
        assert!(out.confidence > 0.95);
    }

    #[test]
    fn a_pause_keeps_the_clock_and_says_so() {
        let mut c = SimulationClock::default();
        let (mut t, mut present) = (5.0f32, 0i64);
        for i in 0..20 {
            t += 0.016;
            present += 16_000_000;
            c.update(&unity_block(t, 0.016, i), present);
        }
        present += 16_000_000;
        let out = c.update(&unity_block(t, 0.016, 20), present);
        assert!(out.found && out.paused);
    }

    #[test]
    fn no_constants_is_no_clock() {
        let mut c = SimulationClock::default();
        for i in 0..40 {
            assert!(!c.update(&[], i * 16_000_000).found);
        }
    }

    #[test]
    fn milliseconds_are_a_unit_and_noise_is_no_clock() {
        let mut c = SimulationClock::default();
        let mut out = ClockFrame::default();
        for i in 0..20 {
            let ms = 1000.0 + 16.6 * i as f32;
            out = c.update(&[0.5, ms, (i % 3) as f32], i as i64 * 16_600_000);
        }
        assert!(out.found && out.time_index == 1 && out.scale == 1000.0);
        let mut c = SimulationClock::default();
        for i in 0..20 {
            out = c.update(&[(i % 5) as f32, 1.0], i as i64 * 16_600_000);
        }
        assert!(!out.found);
    }
}
