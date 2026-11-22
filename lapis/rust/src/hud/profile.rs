// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! What a game's fragment shaders drew: per shader hash, how often its draws were confidently HUD
//! and how often something else. Kept in the game's cache across runs, so a UI shader is known
//! from the first frames, wherever the game interleaves it with the scene.

use std::collections::HashMap;

use super::*;

/// A label this sure, agreeing with the frame's structure, teaches the shader's prior.
const TEACHING_CONFIDENCE: f32 = 0.85;
/// Counts halve past this: a shader's recent use outweighs its history.
const MAX_COUNT: f32 = 512.0;
/// Pseudo-count: a shader seen a few times says little.
const PRIOR_WEIGHT: f32 = 4.0;
const MAX_SHADERS: usize = 512;
const TAG: &str = "hudpriors";
const VERSION: u32 = 1;

#[derive(Default)]
pub struct ShaderPriors {
    /// hash -> (HUD, other)
    counts: HashMap<u32, (f32, f32)>,
    dirty: bool,
}

impl ShaderPriors {
    /// In [-1, 1]: positive for a shader that draws HUD.
    pub fn prior(&self, shader: u32) -> f32 {
        match self.counts.get(&shader) {
            Some(&(hud, other)) if shader != 0 => (hud - other) / (hud + other + PRIOR_WEIGHT),
            _ => 0.0,
        }
    }

    pub fn learn(&mut self, draws: &[HudDraw], labels: &[u8], confidence: &[f32], weak: &[u8]) {
        for (((d, &label), &p), &structural) in draws.iter().zip(labels).zip(confidence).zip(weak) {
            if d.shader == 0 || p < TEACHING_CONFIDENCE {
                continue;
            }
            let hud = matches!(label, LABEL_HUD | LABEL_WORLD_UI);
            if hud != matches!(structural, LABEL_HUD | LABEL_WORLD_UI) {
                continue;
            }
            if !self.counts.contains_key(&d.shader) && self.counts.len() >= MAX_SHADERS * 2 {
                continue;
            }
            let fresh = !self.counts.contains_key(&d.shader);
            let c = self.counts.entry(d.shader).or_insert((0.0, 0.0));
            let before = c.0 > c.1;
            if hud {
                c.0 += 1.0;
            } else {
                c.1 += 1.0;
            }
            if c.0 + c.1 > MAX_COUNT {
                c.0 *= 0.5;
                c.1 *= 0.5;
            }
            self.dirty |= fresh || before != (c.0 > c.1);
        }
    }

    pub fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// `hudpriors 1 hash:hud:other ...`, the most used shaders first.
    pub fn to_line(&self) -> String {
        let mut all: Vec<(&u32, &(f32, f32))> = self.counts.iter().collect();
        all.sort_by(|a, b| (b.1 .0 + b.1 .1).total_cmp(&(a.1 .0 + a.1 .1)).then(a.0.cmp(b.0)));
        let mut line = format!("{TAG} {VERSION}");
        for (hash, (hud, other)) in all.into_iter().take(MAX_SHADERS) {
            line.push_str(&format!(" {hash:08x}:{}:{}", hud.round() as u32, other.round() as u32));
        }
        line
    }

    pub fn warm(&mut self, line: &str) -> bool {
        let mut words = line.split_whitespace();
        if words.next() != Some(TAG) || words.next().and_then(|v| v.parse::<u32>().ok()) != Some(VERSION) {
            return false;
        }
        let mut counts = HashMap::new();
        for word in words.take(MAX_SHADERS) {
            match parse_entry(word) {
                Some((hash, c)) if hash != 0 => {
                    counts.insert(hash, c);
                }
                _ => return false,
            }
        }
        self.counts = counts;
        self.dirty = false;
        true
    }
}

fn parse_entry(word: &str) -> Option<(u32, (f32, f32))> {
    let mut parts = word.split(':');
    let hash = u32::from_str_radix(parts.next()?, 16).ok()?;
    let hud = parts.next()?.parse::<u32>().ok()?;
    let other = parts.next()?.parse::<u32>().ok()?;
    let cap = MAX_COUNT as u32;
    (parts.next().is_none()).then_some((hash, (hud.min(cap) as f32, other.min(cap) as f32)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(shader: u32) -> HudDraw {
        HudDraw { shader, ..HudDraw::default() }
    }

    #[test]
    fn confident_agreeing_labels_teach_and_round_trip() {
        let mut p = ShaderPriors::default();
        let draws = [draw(0xaa), draw(0xbb), draw(0xbb), draw(0)];
        for _ in 0..10 {
            p.learn(&draws, &[LABEL_HUD, LABEL_SCENE, LABEL_SCENE, LABEL_HUD], &[0.95; 4], &[
                LABEL_HUD,
                LABEL_SCENE,
                LABEL_SCENE,
                LABEL_HUD,
            ]);
        }
        assert!(p.prior(0xaa) > 0.6, "{}", p.prior(0xaa));
        assert!(p.prior(0xbb) < -0.6);
        assert_eq!(p.prior(0), 0.0);
        assert!(p.dirty());
        let mut q = ShaderPriors::default();
        assert!(q.warm(&p.to_line()));
        assert!((q.prior(0xaa) - p.prior(0xaa)).abs() < 1e-6);
        assert!(!q.dirty());
        assert!(!q.warm("hudpriors 1 zz:1:2"));
        assert!(!q.warm("hudweights 1"));
    }

    #[test]
    fn unsure_or_disagreeing_labels_teach_nothing() {
        let mut p = ShaderPriors::default();
        p.learn(&[draw(0xaa)], &[LABEL_HUD], &[0.5], &[LABEL_HUD]);
        p.learn(&[draw(0xaa)], &[LABEL_HUD], &[0.99], &[LABEL_SCENE]);
        assert_eq!(p.prior(0xaa), 0.0);
        assert!(!p.dirty());
    }
}
