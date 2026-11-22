// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The next frame's camera, read from the game's memory before the game draws with it. Games write
//! per-frame constants into a ring of host-visible slots; once the camera slot of the last frames is
//! known, the slots the ring will use next are watched at every submit and present, a new
//! view-projection there is the next frame's, and the next frame proves or refutes it. How early it
//! arrives against the presents is measured, not assumed.

use super::camera::CameraAddress;
use super::matrix::{self, Mat4};

const HISTORY: usize = 4;
const FRESH: usize = 8;
const LEADS: usize = 64;
const SLACK: f32 = 1e-5;

#[derive(Clone, Copy, Debug)]
struct Fresh {
    m: Mat4,
    first_seen: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NextCameraFrame {
    /// The next frame's view-projection as stored (row-major, its jitter included), seen by this
    /// present.
    pub matrix: Mat4,
    pub valid: bool,
    /// When it was first seen, against this present (positive: before it).
    pub seen_before_present_ns: i64,
    /// Over the frames checked: the share the offer matched what the next frame drew with, and the
    /// share whose camera was in memory before the present before it.
    pub hit_rate: f32,
    pub early_rate: f32,
    /// Median of present(N) - first seen(camera of N + 1), and of present(N + 1) - first seen.
    pub lead_median_ns: i64,
    pub own_lead_median_ns: i64,
    pub confidence: f32,
}

#[derive(Default)]
pub struct NextCameraTracker {
    used: Vec<(CameraAddress, Mat4)>,
    fresh: Vec<Fresh>,
    offered: Option<Mat4>,
    last_present: i64,
    checked: u32,
    hits: u32,
    early: u32,
    leads: Vec<i64>,
    own_leads: Vec<i64>,
    aspect: f32,
}

fn close(a: &Mat4, b: &Mat4) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= SLACK * (1.0 + x.abs().max(y.abs())))
}

fn median(v: &[i64]) -> i64 {
    if v.is_empty() {
        return 0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

fn push_capped<T>(v: &mut Vec<T>, x: T, cap: usize) {
    if v.len() == cap {
        v.remove(0);
    }
    v.push(x);
}

impl NextCameraTracker {
    pub fn reset(&mut self) {
        *self = NextCameraTracker::default();
    }

    /// Where the next camera may land: the ring slots of the last frames, and one stride on.
    pub fn watch(&self) -> Vec<CameraAddress> {
        let mut out: Vec<CameraAddress> = Vec::new();
        let mut add = |a: CameraAddress| {
            if !out.contains(&a) {
                out.push(a);
            }
        };
        if let [.., before, last] = self.used.as_slice() {
            let (b, l) = (before.0, last.0);
            if b.memory == l.memory && l.offset > b.offset {
                add(CameraAddress { offset: 2 * l.offset - b.offset, ..l });
            }
        }
        for (a, _) in self.used.iter().rev() {
            add(*a);
        }
        out
    }

    /// The 16 floats at `address` at `now`.
    pub fn observe(&mut self, address: CameraAddress, floats: &[f32], now: i64) {
        if floats.len() < 16 || self.used.is_empty() {
            return;
        }
        let Some(fit) = matrix::fit_view_projection(floats, address.row_major, self.aspect) else {
            return;
        };
        if self.used.iter().any(|(_, m)| close(m, &fit.m)) || self.fresh.iter().any(|f| close(&f.m, &fit.m)) {
            return;
        }
        push_capped(&mut self.fresh, Fresh { m: fit.m, first_seen: now }, FRESH);
    }

    /// The frame presented at `now` drew with `used` (as stored) from `address`.
    pub fn present(&mut self, used: &Mat4, address: CameraAddress, aspect: f32, now: i64) -> NextCameraFrame {
        self.aspect = aspect;
        // The frame the last present offered for proves or refutes the offer, and dates its camera.
        if self.last_present != 0 {
            self.checked += 1;
            if self.offered.is_some_and(|o| close(&o, used)) {
                self.hits += 1;
            }
            if let Some(f) = self.fresh.iter().find(|f| close(&f.m, used)) {
                push_capped(&mut self.leads, self.last_present - f.first_seen, LEADS);
                push_capped(&mut self.own_leads, now - f.first_seen, LEADS);
                self.early += (f.first_seen <= self.last_present) as u32;
            }
        }
        self.fresh.retain(|f| !close(&f.m, used) && now - f.first_seen < 1_000_000_000);
        push_capped(&mut self.used, (address, *used), HISTORY);

        let mut out = NextCameraFrame::default();
        // The newest fresh camera still on file is the next frame's.
        if let Some(f) = self.fresh.iter().max_by_key(|f| f.first_seen) {
            out.matrix = f.m;
            out.valid = true;
            out.seen_before_present_ns = now - f.first_seen;
            self.offered = Some(f.m);
        } else {
            self.offered = None;
        }
        self.last_present = now;
        if self.checked > 0 {
            out.hit_rate = self.hits as f32 / self.checked as f32;
            out.early_rate = self.early as f32 / self.checked as f32;
        }
        out.lead_median_ns = median(&self.leads);
        out.own_lead_median_ns = median(&self.own_leads);
        let settled = (self.checked.min(30) as f32) / 30.0;
        out.confidence = if out.valid { out.hit_rate * settled } else { 0.0 };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::matrix::test_util::*;
    use super::super::matrix::mul;
    use super::*;

    const ASPECT: f32 = 2.0;

    fn camera(i: usize) -> Mat4 {
        mul(&perspective(50.0, ASPECT, 0.1), &view([0.2 * i as f32, 1.0, 0.0], 0.5 * i as f32))
    }

    fn slot(k: usize) -> CameraAddress {
        CameraAddress { memory: 3, offset: 4096 * (k % 3) as u64, row_major: true }
    }

    /// A pipelined engine: frame i + 1's constants land in the ring `early_ns` before frame i's
    /// present (negative: after it).
    fn run(early_ns: i64, frames: usize) -> NextCameraFrame {
        let mut t = NextCameraTracker::default();
        let mut mem = [[0.0f32; 16]; 3];
        let period = 16_666_667i64;
        let mut out = NextCameraFrame::default();
        mem[0] = camera(0);
        for i in 0..frames {
            let present = 1_000_000_000 + i as i64 * period;
            let write_next = present - early_ns;
            // Submits every 4 ms; the next camera is written at write_next.
            let mut t_sample = present - period + 1;
            let mut written = false;
            while t_sample <= present {
                if !written && t_sample >= write_next {
                    mem[(i + 1) % 3] = camera(i + 1);
                    written = true;
                }
                for a in t.watch() {
                    let k = (a.offset / 4096) as usize;
                    if k < 3 && a.memory == 3 {
                        t.observe(a, &mem[k], t_sample);
                    }
                }
                t_sample += 4_000_000;
            }
            out = t.present(&camera(i), slot(i), ASPECT, present);
            if !written {
                mem[(i + 1) % 3] = camera(i + 1);
            }
        }
        out
    }

    #[test]
    fn a_camera_written_before_the_present_is_offered_and_proven() {
        let out = run(6_000_000, 60);
        assert!(out.valid);
        assert!(out.hit_rate > 0.95, "{}", out.hit_rate);
        assert!(out.early_rate > 0.95);
        // Sampled every 4 ms: seen within a sample of when it was written.
        assert!(out.lead_median_ns > 2_000_000 && out.lead_median_ns <= 6_000_000, "{}", out.lead_median_ns);
        assert!(out.confidence > 0.9);
    }

    #[test]
    fn a_camera_written_only_after_the_present_is_never_offered_early() {
        let out = run(-5_000_000, 60);
        assert!(!out.valid);
        assert_eq!(out.early_rate, 0.0);
        assert_eq!(out.confidence, 0.0);
    }

    #[test]
    fn the_ring_and_its_stride_are_watched() {
        let mut t = NextCameraTracker::default();
        let m = camera(0);
        t.present(&m, CameraAddress { memory: 1, offset: 256, row_major: false }, ASPECT, 1);
        t.present(&camera(1), CameraAddress { memory: 1, offset: 512, row_major: false }, ASPECT, 2);
        let w = t.watch();
        assert_eq!(w[0].offset, 768);
        assert!(w.iter().any(|a| a.offset == 256) && w.iter().any(|a| a.offset == 512));
    }
}
