// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! How a game encodes its velocity buffer, found by fitting texels of it against the motion the
//! camera alone predicts at the same texels (depth + reprojection). Static world texels agree with
//! the camera; moving objects and sky are outliers the fit trims. The result maps a raw texel to
//! NDC motion from this frame to the previous one.

pub const ENCODING_UNKNOWN: u8 = 0;
/// motion = scale * (raw - bias)
pub const ENCODING_LINEAR: u8 = 1;
/// s = 2 (raw - bias); motion = scale * sign(s) s^2 (more precision for slow motion in 8-10 bits).
pub const ENCODING_SQUARED: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VelocitySample {
    /// The texel as the shader reads it (UNORM in 0..1, SNORM in -1..1, floats as stored).
    pub raw: [f32; 2],
    /// NDC motion to the previous frame the camera predicts there.
    pub camera: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VelocityEncoding {
    pub bias: f32,
    pub scale: [f32; 2],
    /// RMS residual over the kept texels, in NDC, and the share kept.
    pub residual: f32,
    pub inliers: f32,
    pub confidence: f32,
    pub kind: u8,
    pub _pad: [u8; 3],
}

const BIASES: [f32; 3] = [0.0, 0.5, 127.0 / 255.0];

fn encode(kind: u8, bias: f32, raw: f32) -> f32 {
    let v = raw - bias;
    match kind {
        ENCODING_SQUARED => {
            let s = 2.0 * v;
            s.signum() * s * s
        }
        _ => v,
    }
}

/// Least squares scale of `pred ~ scale * e` on the samples kept, with its RMS residual.
fn fit_axis(samples: &[VelocitySample], keep: &[bool], axis: usize, kind: u8, bias: f32) -> (f32, f32) {
    let (mut ee, mut ep, mut n) = (0.0f64, 0.0f64, 0usize);
    for (s, k) in samples.iter().zip(keep) {
        if *k {
            let e = f64::from(encode(kind, bias, s.raw[axis]));
            ee += e * e;
            ep += e * f64::from(s.camera[axis]);
            n += 1;
        }
    }
    if ee < 1e-12 || n == 0 {
        return (0.0, f32::INFINITY);
    }
    let scale = (ep / ee) as f32;
    let mut sq = 0.0f64;
    for (s, k) in samples.iter().zip(keep) {
        if *k {
            let r = f64::from(scale * encode(kind, bias, s.raw[axis]) - s.camera[axis]);
            sq += r * r;
        }
    }
    (scale, (sq / n as f64).sqrt() as f32)
}

/// The encoding that best explains `samples`; needs camera motion in them to tell anything.
pub fn fit(samples: &[VelocitySample]) -> VelocityEncoding {
    let moving = samples.iter().filter(|s| s.camera[0].hypot(s.camera[1]) > 1e-4).count();
    if samples.len() < 16 || moving * 2 < samples.len() {
        return VelocityEncoding { kind: ENCODING_UNKNOWN, ..Default::default() };
    }
    let spread: f32 = {
        let mut m: Vec<f32> = samples.iter().map(|s| s.camera[0].hypot(s.camera[1])).collect();
        m.sort_by(f32::total_cmp);
        m[m.len() / 2]
    };
    let mut best = VelocityEncoding { residual: f32::INFINITY, ..Default::default() };
    for kind in [ENCODING_LINEAR, ENCODING_SQUARED] {
        for bias in BIASES {
            let mut keep = vec![true; samples.len()];
            let mut scale = [0.0f32; 2];
            let mut res = [0.0f32; 2];
            // Fit, keep texels within 3x the median error (moving objects and sky go), refit.
            for round in 0..4 {
                for axis in 0..2 {
                    (scale[axis], res[axis]) = fit_axis(samples, &keep, axis, kind, bias);
                }
                if round == 3 {
                    break;
                }
                let err: Vec<f32> = samples
                    .iter()
                    .map(|s| {
                        let dx = scale[0] * encode(kind, bias, s.raw[0]) - s.camera[0];
                        let dy = scale[1] * encode(kind, bias, s.raw[1]) - s.camera[1];
                        dx.hypot(dy)
                    })
                    .collect();
                let mut sorted = err.clone();
                sorted.sort_by(f32::total_cmp);
                let limit = 3.0 * sorted[sorted.len() / 2] + 1e-7;
                for (k, e) in keep.iter_mut().zip(&err) {
                    *k = *e <= limit;
                }
                if keep.iter().filter(|k| **k).count() < samples.len() / 2 {
                    break;
                }
            }
            let r = res[0].hypot(res[1]);
            if r < best.residual && scale[0] != 0.0 && scale[1] != 0.0 {
                let kept = keep.iter().filter(|k| **k).count() as f32 / samples.len() as f32;
                best = VelocityEncoding { kind, bias, scale, residual: r, inliers: kept, ..Default::default() };
            }
        }
    }
    if best.residual.is_finite() {
        // Residual against the typical motion: 10 % error leaves half the trust.
        best.confidence = (1.0 / (1.0 + (best.residual / spread.max(1e-6) / 0.1).powi(2))).clamp(0.0, 1.0);
    }
    best
}

/// A raw texel as NDC motion to the previous frame (the shader does the same).
#[cfg(test)]
pub fn decode(e: &VelocityEncoding, raw: [f32; 2]) -> [f32; 2] {
    [e.scale[0] * encode(e.kind, e.bias, raw[0]), e.scale[1] * encode(e.kind, e.bias, raw[1])]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(f: impl Fn([f32; 2]) -> [f32; 2]) -> Vec<VelocitySample> {
        (0..200)
            .map(|i| {
                let a = i as f32 * 0.37;
                let camera = [0.02 * a.sin() + 0.01, 0.015 * (1.3 * a).cos()];
                let mut raw = f(camera);
                if i % 9 == 0 {
                    // A moving object: its own motion, not the camera's.
                    raw = f([camera[0] + 0.05, camera[1] - 0.03]);
                }
                VelocitySample { raw, camera }
            })
            .collect()
    }

    #[test]
    fn unreal_rg16f_in_uv_units_is_linear_half_scale() {
        // UE stores (current - previous) in UV: NDC / 2 with y down, so previous - current NDC is
        // (-2, +2) times it.
        let e = fit(&samples(|c| [-c[0] / 2.0, c[1] / 2.0]));
        assert_eq!((e.kind, e.bias), (ENCODING_LINEAR, 0.0));
        assert!((e.scale[0] + 2.0).abs() < 1e-3 && (e.scale[1] - 2.0).abs() < 1e-3, "{e:?}");
        assert!(e.confidence > 0.9 && e.inliers > 0.75);
    }

    #[test]
    fn a_squared_unorm_encoding_like_genshins_is_told_from_a_linear_one() {
        let enc = |c: [f32; 2]| {
            let s = |m: f32| m.signum() * m.abs().sqrt();
            [s(c[0]) / 2.0 + 127.0 / 255.0, s(c[1]) / 2.0 + 127.0 / 255.0]
        };
        let e = fit(&samples(enc));
        assert_eq!(e.kind, ENCODING_SQUARED);
        assert!((e.bias - 127.0 / 255.0).abs() < 1e-6);
        let d = decode(&e, enc([0.01, -0.02]));
        assert!((d[0] - 0.01).abs() < 1e-4 && (d[1] + 0.02).abs() < 1e-4, "{d:?}");
    }

    #[test]
    fn a_still_camera_tells_nothing() {
        let still: Vec<VelocitySample> = (0..64).map(|_| VelocitySample { raw: [0.5, 0.5], camera: [0.0, 0.0] }).collect();
        assert_eq!(fit(&still).kind, ENCODING_UNKNOWN);
    }
}
