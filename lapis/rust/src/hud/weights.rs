// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! A linear score per label over the draw's features, softmaxed: the label is the best score, its
//! confidence that label's probability. The shipped weights are set by hand; lab/hud-detect's
//! trainer fits a game's own from survey captures, written as a `hudweights` line to the game's
//! cache, which replaces them for that game.

use super::features::{FEATURES, FEATURE_NAMES};
use super::*;

/// The starting weights: (label, feature, weight); everything else 0. Depth writes, a perspective
/// camera and the passes up to the geometry pass say scene; one full-screen draw sampling what a
/// pass drew says post; a screen projection, blending without depth, 2D vertices, a glyph atlas
/// and what follows the post draw or a depth clear in the presenting pass say HUD. In-world UI
/// is HUD material under the scene's camera, and needs most of that evidence at once.
const HAND_SET: &[(u8, &str, f32)] = &[
    (LABEL_SCENE, "bias", 0.5),
    (LABEL_SCENE, "depth_write", 3.0),
    (LABEL_SCENE, "depth_compare", 1.0),
    (LABEL_SCENE, "perspective", 2.0),
    (LABEL_SCENE, "screen_ortho", -2.0),
    (LABEL_SCENE, "fullscreen", -2.0),
    (LABEL_SCENE, "vertex_2d", -2.0),
    (LABEL_SCENE, "shader_prior", -2.0),
    (LABEL_SCENE, "pass_depth", 1.0),
    (LABEL_SCENE, "pass_presents", -1.0),
    (LABEL_SCENE, "before_scene_end", 2.0),
    (LABEL_SCENE, "pass_geometry", 1.5),
    (LABEL_SCENE, "pass_ui", -1.0),
    (LABEL_SCENE, "after_depth_clear", -2.0),
    (LABEL_SCENE, "after_post_draw", -3.0),
    (LABEL_SCENE, "pass_untoned_target", 1.0),
    (LABEL_SCENE, "pass_offscreen_small", 1.0),
    (LABEL_SCENE, "blend", -1.0),
    (LABEL_SCENE, "after_last_depth_write", -1.0),
    (LABEL_SCENE, "tex_single_channel", -1.0),
    (LABEL_POST, "bias", -1.0),
    (LABEL_POST, "fullscreen", 3.0),
    (LABEL_POST, "tex_render_target", 1.5),
    (LABEL_POST, "tex_depth", 0.5),
    (LABEL_POST, "depth_write", -3.0),
    (LABEL_POST, "depth_compare", -1.0),
    (LABEL_POST, "perspective", -1.0),
    (LABEL_POST, "log_elements", -3.0),
    (LABEL_POST, "pass_reads_scene", 1.0),
    (LABEL_POST, "pass_offscreen_small", 1.0),
    (LABEL_POST, "before_scene_end", -2.0),
    (LABEL_POST, "after_post_draw", -2.5),
    (LABEL_POST, "after_depth_clear", -1.5),
    (LABEL_POST, "pass_ui", 0.5),
    (LABEL_POST, "pass_depth", -0.5),
    (LABEL_POST, "pass_few_draws", 2.0),
    (LABEL_SCENE, "pass_few_draws", -0.5),
    (LABEL_HUD, "pass_few_draws", -1.0),
    (LABEL_HUD, "bias", -1.5),
    (LABEL_HUD, "screen_ortho", 3.0),
    (LABEL_HUD, "perspective", -2.0),
    (LABEL_HUD, "blend", 1.0),
    (LABEL_HUD, "depth_write", -3.0),
    (LABEL_HUD, "depth_compare", -0.5),
    (LABEL_HUD, "vertex_2d", 2.0),
    (LABEL_HUD, "vertex_colour8", 1.0),
    (LABEL_HUD, "tex_single_channel", 1.0),
    (LABEL_HUD, "tex_uploaded_only", 0.5),
    (LABEL_HUD, "scissor_partial", 1.0),
    (LABEL_HUD, "shader_prior", 3.0),
    (LABEL_HUD, "pass_presents", 1.5),
    (LABEL_HUD, "pass_ui", 1.5),
    (LABEL_HUD, "after_depth_clear", 3.0),
    (LABEL_HUD, "after_last_depth_write", 1.0),
    (LABEL_HUD, "after_post_draw", 4.0),
    (LABEL_HUD, "pass_untoned_target", -2.5),
    (LABEL_HUD, "pass_geometry", -1.0),
    (LABEL_HUD, "before_scene_end", -3.0),
    (LABEL_HUD, "fullscreen", -1.5),
    (LABEL_HUD, "pass_offscreen_small", -2.0),
    (LABEL_HUD, "position_in_pass", 0.5),
    (LABEL_WORLD_UI, "bias", -5.0),
    (LABEL_WORLD_UI, "perspective", 1.5),
    (LABEL_WORLD_UI, "blend", 1.5),
    (LABEL_WORLD_UI, "depth_compare", 1.0),
    (LABEL_WORLD_UI, "depth_write", -3.0),
    (LABEL_WORLD_UI, "vertex_colour8", 1.0),
    (LABEL_WORLD_UI, "tex_single_channel", 3.0),
    (LABEL_WORLD_UI, "shader_prior", 2.0),
    (LABEL_WORLD_UI, "after_last_depth_write", 1.5),
    (LABEL_WORLD_UI, "pass_geometry", 1.5),
    (LABEL_WORLD_UI, "screen_ortho", -1.5),
    (LABEL_WORLD_UI, "vertex_2d", -1.0),
    (LABEL_WORLD_UI, "after_post_draw", -1.0),
    (LABEL_WORLD_UI, "pass_untoned_target", -0.5),
    (LABEL_WORLD_UI, "before_scene_end", -1.0),
];

fn feature(name: &str) -> Option<usize> {
    FEATURE_NAMES.iter().position(|n| *n == name)
}

#[derive(Clone, PartialEq, Debug)]
pub struct Weights {
    w: [[f32; FEATURES]; LABELS],
}

impl Default for Weights {
    fn default() -> Self {
        let mut w = [[0.0; FEATURES]; LABELS];
        for &(label, name, value) in HAND_SET {
            if let Some(f) = feature(name) {
                w[label as usize][f] = value;
            }
        }
        Weights { w }
    }
}

const TAG: &str = "hudweights";
const VERSION: u32 = 1;

impl Weights {
    /// The best label for `row` and its probability.
    pub fn label(&self, row: &[f32]) -> (u8, f32) {
        let mut scores = [0.0f32; LABELS];
        for (s, w) in scores.iter_mut().zip(&self.w) {
            *s = w.iter().zip(row).map(|(a, b)| a * b).sum();
        }
        let (best, top) = scores
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bs), (i, &s)| if s > bs { (i, s) } else { (bi, bs) });
        let total: f32 = scores.iter().map(|s| (s - top).exp()).sum();
        (best as u8, 1.0 / total)
    }

    /// `hudweights 1 <labels> <features> w...`, label by label; None unless it fits this build.
    pub fn from_line(line: &str) -> Option<Weights> {
        let mut words = line.split_whitespace();
        if words.next()? != TAG || words.next()?.parse::<u32>().ok()? != VERSION {
            return None;
        }
        if words.next()?.parse::<usize>().ok()? != LABELS || words.next()?.parse::<usize>().ok()? != FEATURES {
            return None;
        }
        let mut w = [[0.0; FEATURES]; LABELS];
        for row in w.iter_mut() {
            for v in row.iter_mut() {
                *v = words.next()?.parse::<f32>().ok().filter(|v| v.is_finite())?;
            }
        }
        words.next().is_none().then_some(Weights { w })
    }

    #[cfg(test)]
    pub fn to_line(&self) -> String {
        let mut s = format!("{TAG} {VERSION} {LABELS} {FEATURES}");
        for row in &self.w {
            for v in row {
                s += &format!(" {v}");
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_hand_set_weight_names_a_feature() {
        for &(_, name, _) in HAND_SET {
            assert!(feature(name).is_some(), "{name}");
        }
    }

    #[test]
    fn a_line_round_trips_and_a_wrong_shape_is_refused() {
        let w = Weights::default();
        assert_eq!(Weights::from_line(&w.to_line()), Some(w.clone()));
        let short = format!("{TAG} {VERSION} {LABELS} {} 1 2 3", FEATURES - 1);
        assert!(Weights::from_line(&short).is_none());
        assert!(Weights::from_line(&(w.to_line() + " 7")).is_none());
        assert!(Weights::from_line("hudweights 2").is_none());
    }

    #[test]
    fn probabilities_are_a_softmax() {
        let w = Weights::default();
        let mut row = [0.0f32; FEATURES];
        row[0] = 1.0;
        row[feature("depth_write").unwrap()] = 1.0;
        row[feature("perspective").unwrap()] = 1.0;
        let (label, p) = w.label(&row);
        assert_eq!(label, LABEL_SCENE);
        assert!(p > 0.9 && p <= 1.0, "{p}");
    }
}
