// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The colour band as one 3D LUT: 33^3 RGBA16F, red fastest, holding the result unclamped.
//!
//! The grid is denser toward black: it is indexed by each encoded channel to the power
//! LUT_SHAPER, which keeps curves that rise steeply out of black within an LSB. The result is
//! stored encoded, or linear when the band ends in a linear-light matrix (Colorblind, Night Mode):
//! those cross zero light for wide colours, where the encoding is steepest, and are exact in
//! linear storage. The pass encodes after the tap.

use crate::colour::{blend, decode, encode, luma_weights, Rgb, Step};
use crate::params::{COLORBLIND, NIGHT_MODE};

pub const LUT_SIZE: usize = 33;
pub const LUT_TEXELS: usize = LUT_SIZE * LUT_SIZE * LUT_SIZE;
pub const LUT_SHAPER: f32 = 2.0 / 3.0;
pub const SWATCH_COLOURS: usize = 8;
pub const CURVE_POINTS: usize = 17;

/// Black, 25/50/75 % grey, white, then skin, sky and foliage as sRGB-encoded references.
const REFERENCES: [Rgb; SWATCH_COLOURS] = [
    Rgb { r: 0.0, g: 0.0, b: 0.0 },
    Rgb { r: 0.25, g: 0.25, b: 0.25 },
    Rgb { r: 0.5, g: 0.5, b: 0.5 },
    Rgb { r: 0.75, g: 0.75, b: 0.75 },
    Rgb { r: 1.0, g: 1.0, b: 1.0 },
    Rgb { r: 0.84, g: 0.62, b: 0.50 },
    Rgb { r: 0.42, g: 0.62, b: 0.85 },
    Rgb { r: 0.33, g: 0.47, b: 0.22 },
];

fn unorm8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn to_half(v: f32) -> u16 {
    let f = v.to_bits();
    let sign = (f >> 16) & 0x8000;
    let exponent = ((f >> 23) & 0xff) as i32 - 127 + 15;
    let mut mantissa = f & 0x7fffff;
    if (f >> 23) & 0xff == 0xff {
        return (sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 }) as u16;
    }
    if exponent >= 31 {
        return (sign | 0x7bff) as u16; // largest finite
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign as u16;
        }
        mantissa |= 0x800000;
        let shift = (14 - exponent) as u32;
        let mut half = mantissa >> shift;
        let rest = mantissa & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rest > halfway || (rest == halfway && half & 1 != 0) {
            half += 1;
        }
        return (sign | half) as u16;
    }
    let mut half = (exponent as u32) << 10 | mantissa >> 13;
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 != 0) {
        half += 1; // nearest even
    }
    (sign | half.min(0x7bff)) as u16
}

pub fn from_half(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exponent = ((h >> 10) & 0x1f) as u32;
    let mut mantissa = (h & 0x3ff) as u32;
    let f = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut e = -1i32;
            loop {
                e += 1;
                mantissa <<= 1;
                if mantissa & 0x400 != 0 {
                    break;
                }
            }
            sign | ((127 - 15 - e) as u32) << 23 | (mantissa & 0x3ff) << 13
        }
    } else if exponent == 31 {
        sign | 0x7f800000 | mantissa << 13
    } else {
        sign | (exponent + 127 - 15) << 23 | mantissa << 13
    };
    f32::from_bits(f)
}

/// The exact result of the steps for one colour, as the LUT approximates it.
pub fn evaluate_colour(steps: &[Step], p: u8, mut x: Rgb) -> Rgb {
    for s in steps {
        x = blend(s, x, p);
    }
    x
}

/// Whether the LUT of these steps holds linear light.
pub fn ends_linear(steps: &[Step]) -> bool {
    matches!(steps.last(), Some(s) if s.kind == COLORBLIND || s.kind == NIGHT_MODE)
}

/// Bakes the steps in order for targets with primaries `p` into LUT_TEXELS x RGBA half floats.
/// Returns whether the texels hold linear light.
pub fn bake(steps: &[Step], p: u8, out: &mut [u16]) -> bool {
    let linear = ends_linear(steps);
    let mut grid = [0.0f32; LUT_SIZE];
    for (i, g) in grid.iter_mut().enumerate() {
        *g = (i as f32 / (LUT_SIZE - 1) as f32).powf(1.0 / LUT_SHAPER);
    }
    let one = to_half(1.0);
    let mut t = 0;
    for b in 0..LUT_SIZE {
        for g in 0..LUT_SIZE {
            for r in 0..LUT_SIZE {
                let mut y = evaluate_colour(steps, p, Rgb { r: grid[r], g: grid[g], b: grid[b] });
                if linear {
                    y = Rgb { r: decode(y.r), g: decode(y.g), b: decode(y.b) };
                }
                out[t] = to_half(y.r);
                out[t + 1] = to_half(y.g);
                out[t + 2] = to_half(y.b);
                out[t + 3] = one;
                t += 4;
            }
        }
    }
    linear
}

fn texel(texels: &[u16], r: usize, g: usize, b: usize) -> Rgb {
    let t = &texels[((b * LUT_SIZE + g) * LUT_SIZE + r) * 4..];
    Rgb { r: from_half(t[0]), g: from_half(t[1]), b: from_half(t[2]) }
}

fn lerp(a: Rgb, b: Rgb, t: f32) -> Rgb {
    Rgb { r: a.r + (b.r - a.r) * t, g: a.g + (b.g - a.g) * t, b: a.b + (b.b - a.b) * t }
}

/// What the pass reads from the LUT for x: the shaper, the hardware's trilinear filter and, for a
/// linear LUT, the encoding after it.
pub fn sample(texels: &[u16], linear: bool, x: Rgb) -> Rgb {
    let axis = |v: f32| {
        let s = v.clamp(0.0, 1.0).powf(LUT_SHAPER) * (LUT_SIZE - 1) as f32;
        let i = (s as i32).min(LUT_SIZE as i32 - 2) as usize;
        (i, s - i as f32)
    };
    let (r, fr) = axis(x.r);
    let (g, fg) = axis(x.g);
    let (b, fb) = axis(x.b);
    let c00 = lerp(texel(texels, r, g, b), texel(texels, r + 1, g, b), fr);
    let c10 = lerp(texel(texels, r, g + 1, b), texel(texels, r + 1, g + 1, b), fr);
    let c01 = lerp(texel(texels, r, g, b + 1), texel(texels, r + 1, g, b + 1), fr);
    let c11 = lerp(texel(texels, r, g + 1, b + 1), texel(texels, r + 1, g + 1, b + 1), fr);
    let y = lerp(lerp(c00, c10, fg), lerp(c01, c11, fg), fb);
    if linear { Rgb { r: encode(y.r), g: encode(y.g), b: encode(y.b) } } else { y }
}

/// The swatch of reference colours after the whole band, while the editor is open.
pub fn swatch(steps: &[Step], p: u8, out: &mut [u8; SWATCH_COLOURS * 3]) {
    for (i, reference) in REFERENCES.iter().enumerate() {
        let y = evaluate_colour(steps, p, *reference);
        out[3 * i] = unorm8(y.r);
        out[3 * i + 1] = unorm8(y.g);
        out[3 * i + 2] = unorm8(y.b);
    }
}

/// One colour step's own luma curve over a grey ramp.
pub fn curve(step: &Step, p: u8, out: &mut [u8; CURVE_POINTS]) {
    let w = luma_weights(p);
    for (i, o) in out.iter_mut().enumerate() {
        let v = i as f32 / (CURVE_POINTS - 1) as f32;
        let y = blend(step, Rgb { r: v, g: v, b: v }, p);
        *o = unorm8(y.r * w[0] + y.g * w[1] + y.b * w[2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::PRIMARIES_SRGB;

    #[test]
    fn halves_round_trip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, 1e-5, 1.234] {
            let h = to_half(v);
            assert_eq!(to_half(from_half(h)), h);
        }
        assert_eq!(to_half(1.0), 0x3c00);
        assert_eq!(to_half(1e9), 0x7bff);
    }

    #[test]
    fn an_empty_band_bakes_the_identity() {
        let mut t = vec![0u16; LUT_TEXELS * 4];
        assert!(!bake(&[], PRIMARIES_SRGB, &mut t));
        let x = Rgb { r: 0.2, g: 0.5, b: 0.8 };
        let y = sample(&t, false, x);
        assert!((y.r - x.r).abs() < 2e-3 && (y.g - x.g).abs() < 2e-3 && (y.b - x.b).abs() < 2e-3);
    }
}
