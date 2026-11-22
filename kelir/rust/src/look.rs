// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The scene band's look pass and ray-traced kinds: their constants, and the tonemap curves the
//! look pass applies (luwes/shaders/scene.frag mode 2 holds the same curves in GLSL).
//!
//! The game's frame is already tonemapped, so the look pass first expands its highlights back
//! toward scene light, then applies the chosen curve with a gain that keeps mid grey where it was:
//! the curve reshapes highlights and shadows, never the overall brightness.

use crate::params::*;
use crate::plan::PlanCore;

pub const LOOK_VEC4S: usize = 6;
pub const RAYS_VEC4S: usize = 4;
const MID_GREY: f32 = 0.18;

/// Narkowicz's fit of the ACES filmic curve, on linear light.
fn filmic(x: f32) -> f32 {
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
}

/// The minimal AgX curve (Wrensch) on a grey value: the inset and outset matrices leave greys
/// grey, so the polynomial over log2 exposure is all that remains.
fn agx(x: f32) -> f32 {
    const MIN_EV: f32 = -12.47393;
    const MAX_EV: f32 = 4.026069;
    let v = ((x.max(1e-10).log2()).clamp(MIN_EV, MAX_EV) - MIN_EV) / (MAX_EV - MIN_EV);
    let v2 = v * v;
    let v4 = v2 * v2;
    let y = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
    y.max(0.0).powf(2.2)
}

/// Khronos PBR Neutral on a grey value.
fn neutral(x: f32) -> f32 {
    const START: f32 = 0.8 - 0.04;
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let c = x - offset;
    if c < START {
        return c;
    }
    let d = 1.0 - START;
    1.0 - d * d / (c + d - START)
}

pub fn curve(kind: i32, x: f32) -> f32 {
    match kind {
        CURVE_AGX => agx(x),
        CURVE_NEUTRAL => neutral(x),
        _ => filmic(x),
    }
}

/// The input gain that maps mid grey through `kind` onto mid grey itself.
pub fn grey_gain(kind: i32) -> f32 {
    let (mut lo, mut hi) = (0.01f32, 100.0f32);
    for _ in 0..60 {
        let mid = (lo * hi).sqrt();
        if curve(kind, MID_GREY * mid) < MID_GREY {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo * hi).sqrt()
}

/// The look pass's constants, LOOK_VEC4S vec4s; a kind not in the plan has intensity 0:
///   0: tonemap         x curve, y input gain (exposure and the grey gain), z highlight expansion,
///                      w contrast
///   1: sky             x brightness (stops), y saturation, z hue (radians), w warmth
///   2: sun shafts      x length, y threshold, z warmth, w intensity
///   3: outline         x thickness px, y darkness, z detail, w intensity
///   4: intensities     x tonemap, y sky, z edge smoothing, w edge threshold
///   5: edge smoothing  x 1 for the best quality
pub fn pack_scene_look(plan: &PlanCore, out: &mut [f32; LOOK_VEC4S * 4]) {
    out.fill(0.0);
    if let Some(t) = plan.scene_step(TONEMAP) {
        let kind = t.v[p::CURVE] as i32;
        out[0] = kind as f32;
        out[1] = t.v[p::TONE_EXPOSURE].exp2() * grey_gain(kind);
        out[2] = 0.9 * t.v[p::TONE_HIGHLIGHTS];
        out[3] = t.v[p::TONE_CONTRAST];
        out[16] = t.intensity;
    }
    if let Some(s) = plan.scene_step(SKY) {
        out[4] = s.v[p::SKY_BRIGHTNESS];
        out[5] = s.v[p::SKY_SATURATION];
        out[6] = s.v[p::SKY_HUE].to_radians();
        out[7] = s.v[p::SKY_WARMTH];
        out[17] = s.intensity;
    }
    if let Some(s) = plan.scene_step(SUN_SHAFTS) {
        out[8] = s.v[p::SHAFT_LENGTH];
        out[9] = s.v[p::SHAFT_THRESHOLD];
        out[10] = s.v[p::SHAFT_WARMTH];
        out[11] = s.intensity;
    }
    if let Some(s) = plan.scene_step(OUTLINE) {
        out[12] = s.v[p::LINE_THICKNESS];
        out[13] = s.v[p::LINE_DARKNESS];
        out[14] = s.v[p::LINE_DETAIL];
        out[15] = s.intensity;
    }
    if let Some(s) = plan.scene_step(EDGE_SMOOTHING) {
        out[18] = s.intensity;
        out[19] = s.v[p::SMOOTH_THRESHOLD];
        out[20] = s.v[p::SMOOTH_QUALITY];
    }
}

/// Whether the plan asks for the look pass.
pub fn wants_look(plan: &PlanCore) -> bool {
    [TONEMAP, SKY, SUN_SHAFTS, OUTLINE, EDGE_SMOOTHING].iter().any(|&k| plan.scene_step(k).is_some())
}

/// The ray-traced kinds' constants, RAYS_VEC4S vec4s; a kind not in the plan has intensity 0:
///   0: shadows      x softness (cone, radians), y distance m, z intensity
///   1: reflections  x gloss, y distance m, z intensity
///   2: path traced  x rays per pixel, y reach m, z bounce, w intensity
///   3: reserved
pub fn pack_scene_rays(plan: &PlanCore, out: &mut [f32; RAYS_VEC4S * 4]) {
    out.fill(0.0);
    if let Some(s) = plan.scene_step(RT_SHADOWS) {
        out[0] = 0.1 * s.v[p::RAY_SOFTNESS];
        out[1] = s.v[p::RAY_DISTANCE];
        out[2] = s.intensity;
    }
    if let Some(s) = plan.scene_step(RT_REFLECTIONS) {
        out[4] = s.v[p::RAY_GLOSS];
        out[5] = s.v[p::RAY_DISTANCE];
        out[6] = s.intensity;
    }
    if let Some(s) = plan.scene_step(PATH_TRACED_LIGHT) {
        out[8] = s.v[p::PATH_RAYS] + 1.0;
        out[9] = s.v[p::PATH_REACH];
        out[10] = s.v[p::PATH_BOUNCE];
        out[11] = s.intensity;
    }
}

/// Whether the plan asks for ray-traced light.
pub fn wants_rays(plan: &PlanCore) -> bool {
    [RT_SHADOWS, RT_REFLECTIONS, PATH_TRACED_LIGHT].iter().any(|&k| plan.scene_step(k).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_curve_keeps_mid_grey() {
        for kind in [CURVE_FILMIC, CURVE_AGX, CURVE_NEUTRAL] {
            let g = grey_gain(kind);
            assert!((curve(kind, MID_GREY * g) - MID_GREY).abs() < 1e-4, "curve {kind} gain {g}");
        }
        assert!((grey_gain(CURVE_FILMIC) - 0.723).abs() < 0.01);
    }

    #[test]
    fn curves_rise_and_stay_under_white() {
        for kind in [CURVE_FILMIC, CURVE_AGX, CURVE_NEUTRAL] {
            let mut last = -1.0;
            for i in 1..200 {
                let y = curve(kind, i as f32 * 0.1);
                assert!(y >= last - 1e-6 && y <= 1.05, "curve {kind} at {i}: {y}");
                last = y;
            }
        }
    }
}
