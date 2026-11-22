// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The Kelir kinds and their parameters: the one numeric table the codec, the plan, the bake and
//! the shaders read. Ordinals and parameter order are the wire format; kinds are append-only.
//! Mirrored by tests/golden/kelir_params.json, which Pentas pins too.

use core::ffi::c_char;

pub const MAX_PARAMS: usize = 8;
pub const Q_MAX: i16 = 32767;

pub const BAND_DETAIL: u8 = 0;
pub const BAND_COLOUR: u8 = 1;
pub const BAND_SCREEN: u8 = 2;
pub const BAND_SCENE: u8 = 3;

pub const BRIGHTNESS_CONTRAST: u8 = 0;
pub const COLOR: u8 = 1;
pub const LEVELS: u8 = 2;
pub const BLACK_WHITE: u8 = 3;
pub const SEPIA: u8 = 4;
pub const NIGHT_MODE: u8 = 5;
pub const COLORBLIND: u8 = 6;
pub const SHARPEN: u8 = 7;
pub const DETAILS: u8 = 8;
pub const TILT_SHIFT: u8 = 9;
pub const PAINTERLY: u8 = 10;
pub const WATERCOLOR: u8 = 11;
pub const VIGNETTE: u8 = 12;
// 13 Film Grain, 14 Letterbox, 15 Old Film: the table below names them; nothing here reads them.
pub const RETRO: u8 = 16;
// 17 Halftone.
pub const AMBIENT_OCCLUSION: u8 = 18;
pub const DEPTH_FOG: u8 = 19;
pub const DEPTH_OF_FIELD: u8 = 20;
pub const GLOBAL_ILLUMINATION: u8 = 21;
pub const TONEMAP: u8 = 22;
pub const SKY: u8 = 23;
pub const SUN_SHAFTS: u8 = 24;
pub const OUTLINE: u8 = 25;
pub const EDGE_SMOOTHING: u8 = 26;
pub const RT_SHADOWS: u8 = 27;
pub const RT_REFLECTIONS: u8 = 28;
pub const PATH_TRACED_LIGHT: u8 = 29;
pub const KIND_COUNT: usize = 30;

/// Parameter indices, named where the bake and the passes read them.
pub mod p {
    pub const EXPOSURE: usize = 0;
    pub const CONTRAST: usize = 1;
    pub const HIGHLIGHTS: usize = 2;
    pub const SHADOWS: usize = 3;
    pub const GAMMA: usize = 4;
    pub const TEMPERATURE: usize = 0;
    pub const TINT: usize = 1;
    pub const HUE: usize = 2;
    pub const SATURATION: usize = 3;
    pub const VIBRANCE: usize = 4;
    pub const TINT_COLOUR: usize = 5;
    pub const TINT_INTENSITY: usize = 6;
    pub const BLACK: usize = 0;
    pub const WHITE: usize = 1;
    pub const MIDTONES: usize = 2;
    pub const BRIGHTNESS: usize = 3;
    pub const TYPE: usize = 0;
    pub const QUALITY: usize = 1;
    pub const CLARITY: usize = 0;
    pub const HDR_TONING: usize = 1;
    pub const BLOOM: usize = 2;
    pub const BLUR: usize = 5;
    pub const RADIUS: usize = 0;
    pub const FOG_START: usize = 0;
    pub const DENSITY: usize = 1;
    pub const FOG_HUE: usize = 2;
    pub const FOG_SATURATION: usize = 3;
    pub const FOG_BRIGHTNESS: usize = 4;
    pub const FOCUS: usize = 0;
    pub const DISTANCE: usize = 1;
    pub const DOF_BLUR: usize = 2;
    pub const RANGE: usize = 3;
    pub const REACH: usize = 0;
    pub const BOUNCE: usize = 1;
    pub const CURVE: usize = 0;
    pub const TONE_EXPOSURE: usize = 1;
    pub const TONE_HIGHLIGHTS: usize = 2;
    pub const TONE_CONTRAST: usize = 3;
    pub const SKY_BRIGHTNESS: usize = 0;
    pub const SKY_SATURATION: usize = 1;
    pub const SKY_HUE: usize = 2;
    pub const SKY_WARMTH: usize = 3;
    pub const SHAFT_LENGTH: usize = 0;
    pub const SHAFT_THRESHOLD: usize = 1;
    pub const SHAFT_WARMTH: usize = 2;
    pub const LINE_THICKNESS: usize = 0;
    pub const LINE_DARKNESS: usize = 1;
    pub const LINE_DETAIL: usize = 2;
    pub const SMOOTH_THRESHOLD: usize = 0;
    pub const SMOOTH_QUALITY: usize = 1;
    pub const RAY_SOFTNESS: usize = 0;
    pub const RAY_DISTANCE: usize = 1;
    pub const RAY_GLOSS: usize = 0;
    pub const PATH_RAYS: usize = 0;
    pub const PATH_REACH: usize = 1;
    pub const PATH_BOUNCE: usize = 2;
}

/// Tonemap curves (TONEMAP's Curve); 0 is filmic.
#[cfg(test)]
pub const CURVE_FILMIC: i32 = 0;
pub const CURVE_AGX: i32 = 1;
pub const CURVE_NEUTRAL: i32 = 2;

pub const TRITAN: i32 = 2;
pub const SHARPEN_PLUS: i32 = 1;
pub const FIXED_FOCUS: i32 = 1;

/// A static C string for the C++ side's `const char*` names.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Name(pub *const c_char);
// SAFETY: every Name points at a 'static, immutable, NUL-terminated literal.
unsafe impl Sync for Name {}

/// Mirrors framefx::kelir::Param.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Param {
    pub name: Name,
    /// Discrete: the number of choices, coded 0..choices-1. Continuous: 0, and lo <= def <= hi.
    pub choices: u8,
    pub lo: f32,
    pub def: f32,
    pub hi: f32,
    /// Index of a parameter that must be away from its default for this one to matter, or -1.
    pub needs: i8,
}

/// Mirrors framefx::kelir::KindInfo.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KindInfo {
    pub name: Name,
    pub band: u8,
    /// The heavy detail kinds, at most one per stack.
    pub style: bool,
    pub param_count: u8,
    pub params: [Param; MAX_PARAMS],
}

macro_rules! cstr {
    ($s:literal) => {
        Name(concat!($s, "\0").as_ptr() as *const c_char)
    };
}

const fn c(name: Name, lo: f32, def: f32, hi: f32, needs: i8) -> Param {
    Param { name, choices: 0, lo, def, hi, needs }
}

const fn d(name: Name, choices: u8) -> Param {
    Param { name, choices, lo: 0.0, def: 0.0, hi: 0.0, needs: -1 }
}

const NONE: Param = Param { name: Name(core::ptr::null()), choices: 0, lo: 0.0, def: 0.0, hi: 0.0, needs: 0 };

const fn kind<const N: usize>(name: Name, band: u8, style: bool, params: [Param; N]) -> KindInfo {
    let mut all = [NONE; MAX_PARAMS];
    let mut i = 0;
    while i < N {
        all[i] = params[i];
        i += 1;
    }
    KindInfo { name, band, style, param_count: N as u8, params: all }
}

pub static KINDS: [KindInfo; KIND_COUNT] = [
    kind(cstr!("Brightness/Contrast"), BAND_COLOUR, false, [
        c(cstr!("Exposure"), -2.0, 0.0, 2.0, -1),
        c(cstr!("Contrast"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Highlights"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Shadows"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Gamma"), 0.5, 1.0, 2.0, -1),
    ]),
    kind(cstr!("Color"), BAND_COLOUR, false, [
        c(cstr!("Temperature"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Tint"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Hue"), -180.0, 0.0, 180.0, -1),
        c(cstr!("Saturation"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Vibrance"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Tint colour"), 0.0, 0.0, 360.0, p::TINT_INTENSITY as i8),
        c(cstr!("Tint intensity"), 0.0, 0.0, 1.0, -1),
    ]),
    kind(cstr!("Levels"), BAND_COLOUR, false, [
        c(cstr!("Black"), 0.0, 0.0, 0.5, -1),
        c(cstr!("White"), 0.5, 1.0, 1.0, -1),
        c(cstr!("Midtones"), 0.5, 1.0, 2.0, -1),
        c(cstr!("Brightness"), -0.5, 0.0, 0.5, -1),
    ]),
    kind(cstr!("Black & White"), BAND_COLOUR, false, []),
    kind(cstr!("Sepia"), BAND_COLOUR, false, []),
    kind(cstr!("Night Mode"), BAND_COLOUR, false, []),
    kind(cstr!("Colorblind"), BAND_COLOUR, false, [d(cstr!("Type"), 3)]),
    kind(cstr!("Sharpen"), BAND_DETAIL, false, [
        c(cstr!("Ignore film grain"), 0.0, 0.0, 1.0, -1),
        d(cstr!("Quality"), 2),
    ]),
    kind(cstr!("Details"), BAND_DETAIL, false, [
        c(cstr!("Clarity"), -1.0, 0.0, 1.0, -1),
        c(cstr!("HDR toning"), 0.0, 0.0, 1.0, -1),
        c(cstr!("Bloom"), 0.0, 0.0, 1.0, -1),
        c(cstr!("Bloom threshold"), 0.2, 0.75, 1.0, p::BLOOM as i8),
    ]),
    kind(cstr!("Tilt-Shift"), BAND_DETAIL, false, [
        d(cstr!("Shape"), 2),
        c(cstr!("Position"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Size"), 0.05, 0.3, 1.0, -1),
        c(cstr!("Softness"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Angle"), -90.0, 0.0, 90.0, -1),
        c(cstr!("Blur"), 0.0, 0.5, 1.0, -1),
    ]),
    kind(cstr!("Painterly"), BAND_DETAIL, true, [
        c(cstr!("Brush"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Detail"), 0.0, 0.5, 1.0, -1),
    ]),
    kind(cstr!("Watercolor"), BAND_DETAIL, true, [
        c(cstr!("Blurriness"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Fuzziness"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Pencil detail"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Pencil intensity"), 0.0, 0.5, 1.0, -1),
    ]),
    kind(cstr!("Vignette"), BAND_SCREEN, false, [
        c(cstr!("Size"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Roundness"), 0.0, 0.5, 1.0, -1),
    ]),
    kind(cstr!("Film Grain"), BAND_SCREEN, false, [
        c(cstr!("Size"), 0.5, 1.0, 3.0, -1),
        c(cstr!("Colour"), 0.0, 0.0, 1.0, -1),
    ]),
    kind(cstr!("Letterbox"), BAND_SCREEN, false, [c(cstr!("Aspect"), 1.33, 2.39, 3.0, -1)]),
    kind(cstr!("Old Film"), BAND_SCREEN, false, [
        c(cstr!("Dust"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Scratches"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Flicker"), 0.0, 0.3, 1.0, -1),
    ]),
    kind(cstr!("Retro"), BAND_SCREEN, false, [
        c(cstr!("Pixel size"), 1.0, 4.0, 16.0, -1),
        c(cstr!("Colour levels"), 2.0, 8.0, 32.0, -1),
        c(cstr!("Scanlines"), 0.0, 0.5, 1.0, -1),
    ]),
    kind(cstr!("Halftone"), BAND_SCREEN, false, [
        c(cstr!("Dot size"), 2.0, 6.0, 16.0, -1),
        d(cstr!("Ink"), 2),
    ]),
    kind(cstr!("Ambient occlusion"), BAND_SCENE, false, [c(cstr!("Radius"), 0.1, 0.6, 3.0, -1)]),
    kind(cstr!("Depth fog"), BAND_SCENE, false, [
        c(cstr!("Start"), 0.0, 30.0, 500.0, -1),
        c(cstr!("Density"), 0.0, 0.3, 1.0, -1),
        c(cstr!("Hue"), 0.0, 210.0, 360.0, -1),
        c(cstr!("Saturation"), 0.0, 0.15, 1.0, -1),
        c(cstr!("Brightness"), 0.0, 0.85, 1.0, -1),
    ]),
    kind(cstr!("Depth of field"), BAND_SCENE, false, [
        d(cstr!("Focus"), 2),
        c(cstr!("Distance"), 0.3, 10.0, 300.0, p::FOCUS as i8),
        c(cstr!("Blur"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Range"), 0.0, 0.3, 1.0, -1),
    ]),
    kind(cstr!("Global illumination"), BAND_SCENE, false, [
        c(cstr!("Reach"), 0.5, 4.0, 20.0, -1),
        c(cstr!("Bounce"), 0.0, 1.0, 3.0, -1),
    ]),
    kind(cstr!("Tonemap"), BAND_SCENE, false, [
        d(cstr!("Curve"), 3),
        c(cstr!("Exposure"), -2.0, 0.0, 2.0, -1),
        c(cstr!("Highlights"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Contrast"), -1.0, 0.0, 1.0, -1),
    ]),
    kind(cstr!("Sky"), BAND_SCENE, false, [
        c(cstr!("Brightness"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Saturation"), -1.0, 0.0, 1.0, -1),
        c(cstr!("Hue"), -180.0, 0.0, 180.0, -1),
        c(cstr!("Warmth"), -1.0, 0.0, 1.0, -1),
    ]),
    kind(cstr!("Sun shafts"), BAND_SCENE, false, [
        c(cstr!("Length"), 0.1, 0.5, 1.0, -1),
        c(cstr!("Threshold"), 0.3, 0.8, 1.0, -1),
        c(cstr!("Warmth"), 0.0, 0.3, 1.0, -1),
    ]),
    kind(cstr!("Outline"), BAND_SCENE, false, [
        c(cstr!("Thickness"), 0.5, 1.0, 3.0, -1),
        c(cstr!("Darkness"), 0.0, 0.8, 1.0, -1),
        c(cstr!("Detail"), 0.0, 0.5, 1.0, -1),
    ]),
    kind(cstr!("Edge smoothing"), BAND_SCENE, false, [
        c(cstr!("Edge threshold"), 0.03, 0.1, 0.3, -1),
        d(cstr!("Quality"), 2),
    ]),
    kind(cstr!("Ray-traced shadows"), BAND_SCENE, false, [
        c(cstr!("Softness"), 0.0, 0.3, 1.0, -1),
        c(cstr!("Distance"), 5.0, 40.0, 200.0, -1),
    ]),
    kind(cstr!("Ray-traced reflections"), BAND_SCENE, false, [
        c(cstr!("Gloss"), 0.0, 0.5, 1.0, -1),
        c(cstr!("Distance"), 5.0, 40.0, 200.0, -1),
    ]),
    kind(cstr!("Path-traced light"), BAND_SCENE, false, [
        d(cstr!("Rays"), 2),
        c(cstr!("Reach"), 1.0, 8.0, 30.0, -1),
        c(cstr!("Bounce"), 0.0, 1.0, 3.0, -1),
    ]),
];

pub fn info(kind: u8) -> &'static KindInfo {
    &KINDS[kind as usize]
}

pub fn known(kind: i32) -> bool {
    kind >= 0 && (kind as usize) < KIND_COUNT
}

/// Codes are s16, piecewise around the default: q >= 0 spans def..hi, q < 0 spans lo..def, so the
/// default is exactly 0 and every code decodes inside [lo, hi]. -32768 is never valid.
pub fn code_valid(p: &Param, q: i16) -> bool {
    if p.choices > 0 {
        return q >= 0 && q < p.choices as i16;
    }
    if q == i16::MIN || (q > 0 && p.hi == p.def) || (q < 0 && p.lo == p.def) {
        return false;
    }
    true
}

pub fn decode(p: &Param, q: i16) -> f32 {
    if p.choices > 0 {
        return q as f32;
    }
    // The ends exactly: rounding must not step outside the range.
    if q == Q_MAX {
        return p.hi;
    }
    if q == -Q_MAX {
        return p.lo;
    }
    let t = q as f32 / Q_MAX as f32;
    let v = if q >= 0 { p.def + t * (p.hi - p.def) } else { p.def + t * (p.def - p.lo) };
    v.max(p.lo).min(p.hi)
}

/// Nearest code for a value, clamped into range. Exact for the default and the ends.
pub fn encode(p: &Param, value: f32) -> i16 {
    if p.choices > 0 {
        let v = value.round() as i64;
        return if v < 0 { 0 } else if v >= p.choices as i64 { p.choices as i16 - 1 } else { v as i16 };
    }
    if value >= p.def {
        if p.hi == p.def {
            return 0;
        }
        let t = ((value - p.def) / (p.hi - p.def)).min(1.0);
        return (t * Q_MAX as f32).round() as i16;
    }
    if p.lo == p.def {
        return 0;
    }
    let t = ((p.def - value) / (p.def - p.lo)).min(1.0);
    -((t * Q_MAX as f32).round() as i16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_the_default_and_the_ends() {
        for k in KINDS.iter() {
            for p in &k.params[..k.param_count as usize] {
                if p.choices > 0 {
                    continue;
                }
                assert_eq!(decode(p, 0), p.def);
                assert_eq!(encode(p, p.def), 0);
                if p.hi != p.def {
                    assert_eq!(decode(p, Q_MAX), p.hi);
                    assert_eq!(encode(p, p.hi), Q_MAX);
                }
                if p.lo != p.def {
                    assert_eq!(decode(p, -Q_MAX), p.lo);
                    assert_eq!(encode(p, p.lo), -Q_MAX);
                }
                assert!(!code_valid(p, i16::MIN));
            }
        }
    }
}
