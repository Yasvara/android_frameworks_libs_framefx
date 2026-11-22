// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Object motion from per-draw transforms, for games that keep no velocity buffer. A draw's
//! constants hold its object-to-world matrix (or the full model-view-projection, from which the
//! camera's is divided out); a draw is the same object next frame when its pipeline and geometry
//! are. Gives sparse motion: one vector per moving object at its origin. Skinned meshes move
//! inside their bounds and are only seen as a whole.

use super::matrix::{self, Mat4};

/// One draw of the scene pass: `key` hashes what makes it the same object across frames (pipeline,
/// buffers, index range), `floats` the constants only it reads.
#[derive(Clone, Debug, Default)]
pub struct DrawConstants {
    pub key: u64,
    pub floats: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObjectMotion {
    pub key: u64,
    /// Origin in NDC this frame, and where the object's own motion moved it from (NDC), camera
    /// motion excluded.
    pub ndc: [f32; 2],
    pub motion: [f32; 2],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjectsFrame {
    pub draws: u32,
    /// Draws whose transform was found, and of those the ones matched to last frame.
    pub transformed: u32,
    pub matched: u32,
    pub moving: Vec<ObjectMotion>,
    pub confidence: f32,
}

/// Moves under this (NDC) are still.
const STILL_NDC: f32 = 1e-4;
pub const MAX_MOVING: usize = 64;

#[derive(Default)]
pub struct ObjectTracker {
    last: Vec<(u64, u32, Mat4)>,
    streak: u32,
}

/// The object-to-world transform among a draw's constants; `vp` divides a full M V P.
fn transform(f: &[f32], vp: Option<&Mat4>, vp_inv: Option<&Mat4>) -> Option<Mat4> {
    let mut i = 0;
    while i + 16 <= f.len() {
        let s = &f[i..i + 16];
        if let Some(m) = matrix::fit_rigid(s, false).or_else(|| matrix::fit_rigid(s, true)) {
            if m != matrix::IDENTITY || vp.is_none() {
                return Some(m);
            }
        }
        if let (Some(vp), Some(inv)) = (vp, vp_inv) {
            for row in [false, true] {
                let mvp = matrix::read(s, row);
                // M V P with a different w row than V P alone is an object's.
                if mvp[12..16] != vp[12..16] && matrix::fit_view_projection(s, row, 0.0).is_some() {
                    let m = matrix::mul(inv, &mvp);
                    if matrix::fit_rigid(&m, true).is_some() {
                        return Some(m);
                    }
                }
            }
        }
        i += 4;
    }
    None
}

fn ndc(vp: &Mat4, m: &Mat4) -> Option<[f32; 2]> {
    let o = matrix::apply(vp, [m[3], m[7], m[11], 1.0]);
    (o[3] > 1e-6).then(|| [o[0] / o[3], o[1] / o[3]])
}

impl ObjectTracker {
    pub fn reset(&mut self) {
        *self = ObjectTracker::default();
    }

    /// This frame's draws under the unjittered camera `vp`.
    pub fn update(&mut self, draws: &[DrawConstants], vp: &Mat4) -> ObjectsFrame {
        let inv = matrix::invert(vp);
        let mut out = ObjectsFrame { draws: draws.len() as u32, ..Default::default() };
        let mut now: Vec<(u64, u32, Mat4)> = Vec::with_capacity(draws.len());
        for d in draws {
            let Some(m) = transform(&d.floats, Some(vp), inv.as_ref()) else { continue };
            // The n-th draw of a key this frame is the n-th of that key last frame.
            let nth = now.iter().filter(|(k, _, _)| *k == d.key).count() as u32;
            now.push((d.key, nth, m));
            out.transformed += 1;
            let Some((_, _, before)) = self.last.iter().find(|(k, n, _)| *k == d.key && *n == nth) else {
                continue;
            };
            out.matched += 1;
            if let (Some(a), Some(b)) = (ndc(vp, before), ndc(vp, &m)) {
                let motion = [b[0] - a[0], b[1] - a[1]];
                if motion[0].hypot(motion[1]) > STILL_NDC && out.moving.len() < MAX_MOVING {
                    out.moving.push(ObjectMotion { key: d.key, ndc: b, motion });
                }
            }
        }
        self.streak = if out.matched > 0 { self.streak + 1 } else { 0 };
        out.confidence = if out.transformed == 0 {
            0.0
        } else {
            (out.matched as f32 / out.transformed as f32) * (self.streak.min(10) as f32 / 10.0)
        };
        self.last = now;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::matrix::test_util::*;
    use super::super::matrix::mul;
    use super::*;

    fn cam() -> Mat4 {
        mul(&perspective(50.0, 2.0, 0.1), &view([0.0, 1.0, 5.0], 0.0))
    }

    fn draw(key: u64, m: &Mat4, as_mvp: bool) -> DrawConstants {
        let mut floats = vec![0.5f32; 8];
        let mat = if as_mvp { mul(&cam(), m) } else { *m };
        floats.extend_from_slice(&transpose(&mat));
        floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        DrawConstants { key, floats }
    }

    #[test]
    fn a_moving_object_gives_its_own_motion_and_static_ones_none() {
        let mut t = ObjectTracker::default();
        let mut out = ObjectsFrame::default();
        for i in 0..12 {
            let car = mul(&translate(0.1 * i as f32, 0.0, 0.0), &yaw(3.0 * i as f32));
            let house = translate(-2.0, 0.0, -3.0);
            out = t.update(&[draw(1, &house, false), draw(2, &car, false), draw(3, &house, true)], &cam());
        }
        assert_eq!((out.transformed, out.matched), (3, 3));
        assert_eq!(out.moving.len(), 1);
        let m = out.moving[0];
        assert_eq!(m.key, 2);
        assert!(m.motion[0] > 0.0 && m.motion[1].abs() < 1e-4);
        assert!(out.confidence > 0.99);
    }

    #[test]
    fn instances_of_one_mesh_are_matched_in_draw_order() {
        let mut t = ObjectTracker::default();
        let a = translate(1.0, 0.0, 0.0);
        let b = translate(-1.0, 0.0, 0.0);
        t.update(&[draw(9, &a, false), draw(9, &b, false)], &cam());
        let b2 = translate(-0.9, 0.0, 0.0);
        let out = t.update(&[draw(9, &a, false), draw(9, &b2, false)], &cam());
        assert_eq!(out.moving.len(), 1);
        assert!(out.moving[0].motion[0] > 0.0);
    }
}
