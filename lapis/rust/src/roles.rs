// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Which of the game's G-buffer targets is what, found from how each agrees with the frame
//! (luwes/shaders/rays.comp measures, luwes reads back): an albedo's hue follows the scene's under
//! any light, so their log chromas correlate across the frame; a normal target, decoded the way
//! the game encoded it, agrees with the triangles' own normals. Nothing is assumed of the game.

use std::collections::HashMap;

/// The encodings a normal target is tried in (rays.comp `decodeNormal`): xyz in world space, xyz
/// in view space, octahedral world, octahedral view, view xy with z rebuilt.
pub const DECODINGS: usize = 5;

/// One frame's measures of one target.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub image: u32,
    pub pad: u32,
    /// Per log-chroma channel (red over green, blue over green): n, Σx, Σy, Σxy, Σx², Σy²; x the
    /// scene's, y the target's.
    pub chroma: [[f64; 6]; 2],
    /// Sum of the cosine with the triangle's normal, per decoding, and how many.
    pub normal: [f64; DECODINGS],
    pub normal_n: f64,
}

/// Past measures fade by this each frame: a target the game repurposes is found again.
const DECAY: f64 = 0.995;
/// Frames a target not seen is kept.
const FORGET: u64 = 600;
/// Measures a pick needs, and how well it must agree.
const MIN_SAMPLES: f64 = 500.0;
const MIN_HUE: f64 = 0.5;
const MIN_NORMAL: f64 = 0.7;
/// A new pick must beat the current one by this.
const MARGIN: f64 = 0.1;

#[derive(Clone, Default)]
struct Seen {
    chroma: [[f64; 6]; 2],
    normal: [f64; DECODINGS],
    normal_n: f64,
    frame: u64,
}

impl Seen {
    fn hue(&self) -> Option<f64> {
        let mut r = 0.0;
        for c in &self.chroma {
            let [n, x, y, xy, xx, yy] = *c;
            if n < MIN_SAMPLES {
                return None;
            }
            let var = (n * xx - x * x) * (n * yy - y * y);
            if var <= 0.0 {
                return None;
            }
            r += (n * xy - x * y) / var.sqrt();
        }
        Some(r / 2.0)
    }

    fn normal(&self) -> Option<(f64, u8)> {
        if self.normal_n < MIN_SAMPLES {
            return None;
        }
        (0..DECODINGS)
            .map(|d| (self.normal[d] / self.normal_n, d as u8))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    }
}

#[derive(Default)]
pub struct Roles {
    seen: HashMap<u32, Seen>,
    frame: u64,
    albedo: (u32, f64),
    normal: (u32, u8, f64),
}

impl Roles {
    pub fn add(&mut self, s: &Sample) {
        if s.image == 0 {
            return;
        }
        let e = self.seen.entry(s.image).or_default();
        for (into, from) in e.chroma.iter_mut().zip(&s.chroma) {
            for (a, b) in into.iter_mut().zip(from) {
                *a += b;
            }
        }
        for (a, b) in e.normal.iter_mut().zip(&s.normal) {
            *a += b;
        }
        e.normal_n += s.normal_n;
        e.frame = self.frame;
    }

    /// Ends a frame: measures fade, targets gone a while are forgotten, the picks move only for
    /// a clearly better one.
    pub fn settle(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.seen.retain(|_, e| frame - e.frame <= FORGET);
        for e in self.seen.values_mut() {
            e.chroma.iter_mut().flatten().for_each(|v| *v *= DECAY);
            e.normal.iter_mut().for_each(|v| *v *= DECAY);
            e.normal_n *= DECAY;
        }

        let current = self.seen.get(&self.albedo.0).and_then(Seen::hue).unwrap_or(f64::MIN);
        let best = self
            .seen
            .iter()
            .filter_map(|(&id, e)| e.hue().map(|r| (id, r)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        self.albedo = match best {
            Some((id, r)) if r >= MIN_HUE && (id == self.albedo.0 || r > current + MARGIN) => (id, r),
            _ if current >= MIN_HUE => (self.albedo.0, current),
            _ => (0, 0.0),
        };

        let current = self
            .seen
            .get(&self.normal.0)
            .and_then(Seen::normal)
            .filter(|&(_, d)| d == self.normal.1)
            .map_or(f64::MIN, |(c, _)| c);
        let best = self
            .seen
            .iter()
            .filter_map(|(&id, e)| e.normal().map(|(c, d)| (id, d, c)))
            .max_by(|a, b| a.2.total_cmp(&b.2));
        self.normal = match best {
            Some((id, d, c))
                if c >= MIN_NORMAL
                    && ((id, d) == (self.normal.0, self.normal.1) || c > current + MARGIN) =>
            {
                (id, d, c)
            }
            _ if current >= MIN_NORMAL => (self.normal.0, self.normal.1, current),
            _ => (0, 0, 0.0),
        };
    }

    /// The albedo target and how well it agrees, (0, 0) none yet.
    pub fn albedo(&self) -> (u32, f64) {
        self.albedo
    }

    /// The normal target, its decoding and how well it agrees, (0, 0, 0) none yet.
    pub fn normal(&self) -> (u32, u8, f64) {
        self.normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic noise in [-1, 1].
    fn noise(i: u32) -> f64 {
        let x = (i.wrapping_mul(2654435761) >> 8) as f64 / (1u32 << 24) as f64;
        2.0 * x - 1.0
    }

    // Pearson sums of n pairs (x, f(x)).
    fn sums(n: u32, salt: u32, f: impl Fn(f64, u32) -> f64) -> [f64; 6] {
        let mut s = [0.0; 6];
        for i in 0..n {
            let x = noise(i * 7 + salt);
            let y = f(x, i);
            s[0] += 1.0;
            s[1] += x;
            s[2] += y;
            s[3] += x * y;
            s[4] += x * x;
            s[5] += y * y;
        }
        s
    }

    fn albedo_like(image: u32) -> Sample {
        // The scene's log chroma is the albedo's plus the light's, nearly constant.
        let c = sums(400, 1, |x, i| x - 0.3 + 0.05 * noise(i * 13 + 5));
        Sample { image, chroma: [c, c], ..Sample::default() }
    }

    fn normal_like(image: u32, decoding: usize) -> Sample {
        let c = sums(400, 2, |_, i| noise(i * 31 + 9));
        let mut s = Sample { image, chroma: [c, c], normal_n: 400.0, ..Sample::default() };
        for d in 0..DECODINGS {
            s.normal[d] = if d == decoding { 0.93 } else { 0.1 } * 400.0;
        }
        s
    }

    #[test]
    fn finds_albedo_and_normals_by_agreement() {
        let mut r = Roles::default();
        for _ in 0..3 {
            r.add(&albedo_like(11));
            r.add(&normal_like(12, 3));
            r.settle();
        }
        assert_eq!(r.albedo().0, 11);
        assert!(r.albedo().1 > 0.9);
        let (id, d, c) = r.normal();
        assert_eq!((id, d), (12, 3));
        assert!(c > 0.9);
    }

    #[test]
    fn needs_enough_measures() {
        let mut r = Roles::default();
        r.add(&albedo_like(11));
        r.settle();
        assert_eq!(r.albedo().0, 0, "400 samples are not enough");
    }

    #[test]
    fn keeps_its_pick_against_a_near_equal() {
        let mut r = Roles::default();
        for _ in 0..3 {
            r.add(&albedo_like(11));
            r.settle();
        }
        for _ in 0..3 {
            r.add(&albedo_like(11));
            r.add(&albedo_like(13));
            r.settle();
        }
        assert_eq!(r.albedo().0, 11);
    }

    #[test]
    fn forgets_a_target_gone() {
        let mut r = Roles::default();
        for _ in 0..3 {
            r.add(&albedo_like(11));
            r.settle();
        }
        for _ in 0..=FORGET {
            r.settle();
        }
        assert_eq!(r.albedo().0, 0);
    }
}
