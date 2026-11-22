// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Exposure from the tonemap's constants: a positive scalar that drifts smoothly while the rest of
//! the block holds still, as eye adaptation does. Semantics are guessed from behaviour alone, so
//! trust stays capped; frame generation should confirm a gain step against the colour it sees.

const WINDOW: usize = 32;
const MAX_FLOATS: usize = 256;
/// Largest per-frame step (log) adaptation takes; fades and cuts step further.
const SMOOTH_LOG: f32 = 0.15;
const TRUST_CAP: f32 = 0.5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExposureFrame {
    pub found: bool,
    pub index: u32,
    /// This frame's value over the last frame's: the brightness step the tonemap applies.
    pub gain: f32,
    pub confidence: f32,
}

#[derive(Default)]
pub struct ExposureTracker {
    values: Vec<Vec<f32>>,
}

impl ExposureTracker {
    pub fn update(&mut self, floats: &[f32]) -> ExposureFrame {
        let n = floats.len().min(MAX_FLOATS);
        if self.values.len() != n {
            self.values = vec![Vec::new(); n];
        }
        for (h, v) in self.values.iter_mut().zip(floats) {
            if h.len() == WINDOW {
                h.remove(0);
            }
            h.push(*v);
        }
        let mut best: Option<(usize, u32)> = None;
        for (k, h) in self.values.iter().enumerate() {
            if h.len() < 8 || h.iter().any(|v| !(v.is_finite() && *v > 1e-6 && *v < 1e6)) {
                continue;
            }
            let (mut changes, mut smooth, mut up, mut down) = (0u32, true, true, true);
            for w in h.windows(2) {
                let step = (w[1] / w[0]).ln();
                changes += (step.abs() > 1e-5) as u32;
                smooth &= step.abs() < SMOOTH_LOG;
                up &= step >= 0.0;
                down &= step <= 0.0;
            }
            // A clock grows every frame in one direction; adaptation turns and rests.
            let one_way = up || down;
            let steps = h.len() - 1;
            if changes >= 2 && smooth && !(one_way && changes as usize == steps) && best.is_none_or(|(_, c)| changes > c) {
                best = Some((k, changes));
            }
        }
        let Some((k, changes)) = best else {
            return ExposureFrame { gain: 1.0, ..Default::default() };
        };
        let h = &self.values[k];
        ExposureFrame {
            found: true,
            index: k as u32,
            gain: h[h.len() - 1] / h[h.len() - 2],
            confidence: TRUST_CAP * (changes.min(8) as f32 / 8.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_adaptation_is_found_and_constants_and_clocks_are_not() {
        let mut t = ExposureTracker::default();
        let mut out = ExposureFrame::default();
        for i in 0..30 {
            let exposure = 1.0 + 0.3 * (i as f32 * 0.2).sin();
            out = t.update(&[0.5, 2.2, 0.016 * i as f32 + 1.0, exposure, 4.0]);
        }
        assert!(out.found && out.index == 3);
        let want = (1.0 + 0.3 * (29.0f32 * 0.2).sin()) / (1.0 + 0.3 * (28.0f32 * 0.2).sin());
        assert!((out.gain - want).abs() < 1e-5);
        assert!(out.confidence <= TRUST_CAP);
    }

    #[test]
    fn nothing_moving_is_no_exposure() {
        let mut t = ExposureTracker::default();
        let mut out = ExposureFrame::default();
        for _ in 0..20 {
            out = t.update(&[0.5, 2.2, 4.0]);
        }
        assert!(!out.found && out.gain == 1.0);
    }
}
