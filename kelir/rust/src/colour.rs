// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The colour band evaluated exactly on the CPU: what the LUT bake samples, and the reference the
//! tests hold the LUT and the passes to.
//!
//! Values are the target's encoded RGB (sRGB transfer, sRGB or Display P3 primaries) and are never
//! clamped between steps; linear-light steps decode and re-encode exactly.

use std::sync::OnceLock;

use crate::params::{info, p, BLACK_WHITE, BRIGHTNESS_CONTRAST, COLOR, COLORBLIND, LEVELS, MAX_PARAMS, NIGHT_MODE, SEPIA, TRITAN};

/// Mirrors framefx::kelir::Rgb.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[cfg(test)]
pub const PRIMARIES_SRGB: u8 = 0;
pub const PRIMARIES_P3: u8 = 1;

/// One node of a plan, decoded. Mirrors framefx::kelir::Step.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Step {
    pub kind: u8,
    pub intensity: f32,
    pub v: [f32; MAX_PARAMS],
}

const LUMA_SRGB: [f32; 3] = [0.2126, 0.7152, 0.0722];
const LUMA_P3: [f32; 3] = [0.2289746, 0.6917385, 0.0792869];

// Linear Display P3 <-> linear sRGB (both D65).
const P3_TO_SRGB: [f32; 9] = [1.2249401, -0.2249404, 0.0, -0.0420569, 1.0420571, 0.0, -0.0196376, -0.0786361, 1.0982735];
const SRGB_TO_P3: [f32; 9] = [0.8224621, 0.177538, 0.0, 0.0331941, 0.9668058, 0.0, 0.0170827, 0.0723974, 0.9105199];

// Machado, Oliveira and Fernandes 2009, severity 1.0, linear sRGB.
const SIMULATE: [[f32; 9]; 3] = [
    [0.152286, 1.052583, -0.204868, 0.114503, 0.786281, 0.099216, -0.003882, -0.048116, 1.051998],
    [0.367322, 0.860646, -0.227968, 0.280085, 0.672501, 0.047413, -0.011820, 0.042940, 0.968881],
    [1.255528, -0.076749, -0.178779, -0.078411, 0.930809, 0.147602, 0.004733, 0.691367, 0.303900],
];

// Where the colours a deficiency loses go back in (Fidaner, Lin and Ozguven): red-green losses
// into green and blue, blue-yellow losses into red and green.
const SHIFT_RED_GREEN: [f32; 9] = [0.0, 0.0, 0.0, 0.7, 1.0, 0.0, 0.7, 0.0, 1.0];
const SHIFT_BLUE_YELLOW: [f32; 9] = [1.0, 0.0, 0.7, 0.0, 1.0, 0.7, 0.0, 0.0, 0.0];

// A warm white near 2850 K against D65, linear: blue light drops to 42 %.
const NIGHT_WHITE: Rgb = Rgb { r: 1.0, g: 0.70, b: 0.42 };

// Sepia duotone: brown in the shadows, cream at white.
const SEPIA_SHADOW: Rgb = Rgb { r: 1.0, g: 0.78, b: 0.55 };
const SEPIA_LIGHT: Rgb = Rgb { r: 1.0, g: 0.95, b: 0.85 };

impl core::ops::Add for Rgb {
    type Output = Rgb;
    fn add(self, b: Rgb) -> Rgb {
        Rgb { r: self.r + b.r, g: self.g + b.g, b: self.b + b.b }
    }
}
impl core::ops::Sub for Rgb {
    type Output = Rgb;
    fn sub(self, b: Rgb) -> Rgb {
        Rgb { r: self.r - b.r, g: self.g - b.g, b: self.b - b.b }
    }
}
impl core::ops::Mul<f32> for Rgb {
    type Output = Rgb;
    fn mul(self, s: f32) -> Rgb {
        Rgb { r: self.r * s, g: self.g * s, b: self.b * s }
    }
}
impl core::ops::Mul for Rgb {
    type Output = Rgb;
    fn mul(self, b: Rgb) -> Rgb {
        Rgb { r: self.r * b.r, g: self.g * b.g, b: self.b * b.b }
    }
}

fn grey(v: f32) -> Rgb {
    Rgb { r: v, g: v, b: v }
}

pub fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    a + (b - a) * t
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn dot(a: Rgb, w: &[f32; 3]) -> f32 {
    a.r * w[0] + a.g * w[1] + a.b * w[2]
}

fn mul(m: &[f32; 9], v: Rgb) -> Rgb {
    Rgb {
        r: m[0] * v.r + m[1] * v.g + m[2] * v.b,
        g: m[3] * v.r + m[4] * v.g + m[5] * v.b,
        b: m[6] * v.r + m[7] * v.g + m[8] * v.b,
    }
}

fn spow(v: f32, e: f32) -> f32 {
    v.abs().powf(e).copysign(v)
}

fn decode_rgb(x: Rgb) -> Rgb {
    Rgb { r: decode(x.r), g: decode(x.g), b: decode(x.b) }
}

fn encode_rgb(x: Rgb) -> Rgb {
    Rgb { r: encode(x.r), g: encode(x.g), b: encode(x.b) }
}

fn to_srgb(lin: Rgb, p: u8) -> Rgb {
    if p == PRIMARIES_P3 { mul(&P3_TO_SRGB, lin) } else { lin }
}

fn from_srgb(lin: Rgb, p: u8) -> Rgb {
    if p == PRIMARIES_P3 { mul(&SRGB_TO_P3, lin) } else { lin }
}

/// Luma weights of the primaries, for encoded values.
pub fn luma_weights(p: u8) -> &'static [f32; 3] {
    if p == PRIMARIES_P3 { &LUMA_P3 } else { &LUMA_SRGB }
}

/// Linear luminance of an encoded colour, re-encoded.
fn encoded_luminance(x: Rgb, p: u8) -> f32 {
    encode(dot(decode_rgb(x), luma_weights(p)))
}

fn brightness_contrast(s: &Step, mut x: Rgb, p: u8) -> Rgb {
    let exposure = s.v[p::EXPOSURE];
    if exposure != 0.0 {
        x = encode_rgb(decode_rgb(x) * exposure.exp2());
    }
    let gamma = s.v[p::GAMMA];
    if gamma != 1.0 {
        x = Rgb { r: spow(x.r, 1.0 / gamma), g: spow(x.g, 1.0 / gamma), b: spow(x.b, 1.0 / gamma) };
    }
    let contrast = s.v[p::CONTRAST].exp2();
    x = grey(0.5) + (x - grey(0.5)) * contrast;
    // Tone curve on luma, 0 and 1 fixed: shadows peak at a third, highlights at two thirds.
    // Outside [0, 1] each weight continues with its slope at the end, so the curve has no kink for
    // the LUT to miss.
    let l = dot(x, luma_weights(p));
    let shadows = if l < 0.0 { l } else if l > 1.0 { 0.0 } else { l * (1.0 - l) * (1.0 - l) };
    let highlights = if l < 0.0 { 0.0 } else if l > 1.0 { 1.0 - l } else { l * l * (1.0 - l) };
    x + grey(0.8 * (s.v[p::SHADOWS] * shadows + s.v[p::HIGHLIGHTS] * highlights))
}

/// Hue in turns of a colour's own HSV hexagon, for the skin test.
fn hue(c: Rgb) -> f32 {
    let mx = c.r.max(c.g).max(c.b);
    let mn = c.r.min(c.g).min(c.b);
    let d = mx - mn;
    if d <= 0.0 {
        return 0.0;
    }
    let mut h = if mx == c.r {
        (c.g - c.b) / d
    } else if mx == c.g {
        2.0 + (c.b - c.r) / d
    } else {
        4.0 + (c.r - c.g) / d
    };
    h /= 6.0;
    if h < 0.0 { h + 1.0 } else { h }
}

fn from_hue(degrees: f32) -> Rgb {
    let h = degrees / 60.0;
    let channel = |n: f32| {
        let k = (n + h) % 6.0;
        clamp01(1.0 - 0.0f32.max(k.min(4.0 - k).min(1.0)))
    };
    Rgb { r: channel(5.0), g: channel(3.0), b: channel(1.0) }
}

fn color(s: &Step, mut x: Rgb, p: u8) -> Rgb {
    let w = luma_weights(p);
    // White balance in linear light, luminance kept: warmer raises red over blue, a positive tint
    // takes green out (magenta).
    let t = s.v[p::TEMPERATURE];
    let tint = s.v[p::TINT];
    if t != 0.0 || tint != 0.0 {
        let mut gain = Rgb { r: (0.35 * t).exp2(), g: (-0.25 * tint).exp2(), b: (-0.35 * t).exp2() };
        gain = gain * (1.0 / dot(gain, w));
        x = encode_rgb(decode_rgb(x) * gain);
    }
    let h = s.v[p::HUE];
    if h != 0.0 {
        // Rotation about the grey axis.
        let a = h * core::f32::consts::PI / 180.0;
        let c = a.cos();
        let k = 1.0 / 3.0f32.sqrt();
        let sn = a.sin() * k;
        let along = (x.r + x.g + x.b) * k * k * (1.0 - c);
        x = Rgb {
            r: x.r * c + (x.b - x.g) * sn + along,
            g: x.g * c + (x.r - x.b) * sn + along,
            b: x.b * c + (x.g - x.r) * sn + along,
        };
    }
    let vibrance = s.v[p::VIBRANCE];
    if vibrance != 0.0 {
        // Weighted by 4 s (1 - s) of the saturation s: nothing for greys, most for muted colours,
        // nothing for colours already saturated, so they never crush. Skin tones move a third as
        // much. Every weight is smooth over the cube, so the LUT holds it.
        let c = Rgb { r: clamp01(x.r), g: clamp01(x.g), b: clamp01(x.b) };
        let sat = c.r.max(c.g).max(c.b) - c.r.min(c.g).min(c.b);
        let d = (hue(c) - 0.075).abs();
        let skin = (-(d.min(1.0 - d) / 0.07).powf(2.0)).exp();
        let k = 1.0 + vibrance * 4.0 * sat * (1.0 - sat) * (1.0 - 0.7 * skin);
        let l = dot(x, w);
        x = grey(l) + (x - grey(l)) * k;
    }
    let saturation = s.v[p::SATURATION];
    if saturation != 0.0 {
        let l = dot(x, w);
        x = grey(l) + (x - grey(l)) * (1.0 + saturation);
    }
    let tint_intensity = s.v[p::TINT_INTENSITY];
    if tint_intensity != 0.0 {
        // Colourise toward the hue at the pixel's own luma, as far as black and white allow.
        let tone = from_hue(s.v[p::TINT_COLOUR]);
        let chroma = tone - grey(dot(tone, w));
        let l = dot(x, w);
        let target = grey(l) + chroma * 0.0f32.max(l.min(1.0 - l));
        x = mix(x, target, tint_intensity);
    }
    x
}

fn levels(s: &Step, x: Rgb) -> Rgb {
    let white = s.v[p::WHITE];
    let black = s.v[p::BLACK].min(white - 1.0 / 255.0);
    let range = 1.0 / (white - black);
    let e = 1.0 / s.v[p::MIDTONES];
    let f = |c: f32| spow((c - black) * range, e) + s.v[p::BRIGHTNESS];
    Rgb { r: f(x.r), g: f(x.g), b: f(x.b) }
}

fn sepia(x: Rgb, p: u8) -> Rgb {
    let y = encoded_luminance(x, p);
    mix(SEPIA_SHADOW, SEPIA_LIGHT, clamp01(y)) * y
}

fn colorblind(s: &Step, x: Rgb, p: u8) -> Rgb {
    let kind = (s.v[p::TYPE] as i32).clamp(0, 2);
    let lin = to_srgb(decode_rgb(x), p);
    let error = lin - mul(&SIMULATE[kind as usize], lin);
    let shifted = mul(if kind == TRITAN { &SHIFT_BLUE_YELLOW } else { &SHIFT_RED_GREEN }, error);
    encode_rgb(from_srgb(lin + shifted, p))
}

fn decode_exact(a: f32) -> f32 {
    if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) }
}

fn encode_exact(a: f32) -> f32 {
    if a <= 0.0031308 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 }
}

// The bake evaluates the transfer pair millions of times. Inside [0, 1] tables interpolate it to
// within 1e-7, far under half-float precision: decode over the encoded value, encode over the
// square root of the linear one, where it is smooth.
const TABLE_SIZE: usize = 4096;

struct Tables {
    decode: Vec<f32>,
    encode: Vec<f32>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables { decode: vec![0.0; TABLE_SIZE + 1], encode: vec![0.0; TABLE_SIZE + 1] };
        for i in 0..=TABLE_SIZE {
            let x = i as f32 / TABLE_SIZE as f32;
            t.decode[i] = decode_exact(x);
            t.encode[i] = encode_exact(x * x);
        }
        t
    })
}

fn lookup(table: &[f32], t: f32) -> f32 {
    let x = t * TABLE_SIZE as f32;
    let i = (x as i32).min(TABLE_SIZE as i32 - 1).max(0) as usize;
    table[i] + (table[i + 1] - table[i]) * (x - i as f32)
}

/// Exact piecewise sRGB transfer, odd-extended below 0 and continued above 1.
pub fn decode(e: f32) -> f32 {
    let a = e.abs();
    (if a <= 1.0 { lookup(&tables().decode, a) } else { decode_exact(a) }).copysign(e)
}

pub fn encode(l: f32) -> f32 {
    let a = l.abs();
    (if a <= 1.0 { lookup(&tables().encode, a.sqrt()) } else { encode_exact(a) }).copysign(l)
}

/// The step's function of one colour, without its intensity. Colour-band kinds only.
pub fn apply_colour(s: &Step, x: Rgb, p: u8) -> Rgb {
    match s.kind {
        BRIGHTNESS_CONTRAST => brightness_contrast(s, x, p),
        COLOR => color(s, x, p),
        LEVELS => levels(s, x),
        BLACK_WHITE => grey(encoded_luminance(x, p)),
        SEPIA => sepia(x, p),
        NIGHT_MODE => encode_rgb(from_srgb(to_srgb(decode_rgb(x), p) * NIGHT_WHITE, p)),
        COLORBLIND => colorblind(s, x, p),
        _ => x,
    }
}

/// The step with its intensity: mixed with its input in encoded space.
pub fn blend(s: &Step, x: Rgb, p: u8) -> Rgb {
    mix(x, apply_colour(s, x, p), s.intensity)
}

/// True when the step leaves every colour unchanged.
pub fn is_identity_colour(s: &Step) -> bool {
    let params = &info(s.kind).params;
    let at_default = |i: usize| s.v[i] == params[i].def;
    match s.kind {
        BRIGHTNESS_CONTRAST => [p::EXPOSURE, p::CONTRAST, p::HIGHLIGHTS, p::SHADOWS, p::GAMMA].iter().all(|&i| at_default(i)),
        COLOR => [p::TEMPERATURE, p::TINT, p::HUE, p::SATURATION, p::VIBRANCE, p::TINT_INTENSITY].iter().all(|&i| at_default(i)),
        LEVELS => [p::BLACK, p::WHITE, p::MIDTONES, p::BRIGHTNESS].iter().all(|&i| at_default(i)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transfer_pair_round_trips() {
        for i in 0..=1000 {
            let v = i as f32 / 1000.0;
            assert!((encode(decode(v)) - v).abs() < 1e-5, "{v}");
        }
        assert_eq!(decode(-0.5), -decode(0.5));
    }

    #[test]
    fn identity_steps_change_nothing() {
        let s = Step { kind: BRIGHTNESS_CONTRAST, intensity: 1.0, v: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] };
        assert!(is_identity_colour(&s));
        let x = Rgb { r: 0.3, g: 0.6, b: 0.9 };
        let y = apply_colour(&s, x, PRIMARIES_SRGB);
        assert!((y.r - x.r).abs() < 1e-6 && (y.b - x.b).abs() < 1e-6);
    }
}
