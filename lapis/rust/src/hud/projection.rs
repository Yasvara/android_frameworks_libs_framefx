// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! What camera a draw's own uniform bytes carry: a perspective view-projection (the 3D scene and
//! anything placed in it), or a screen orthographic projection (a HUD drawn in pixels).

use crate::signals::matrix::{fit_view_projection, read};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Camera {
    /// Never sampled.
    Unknown,
    /// Sampled, and neither kind found.
    None,
    Perspective,
    Flat,
    /// Both: a UI camera's block next to the scene's.
    Both,
}

impl Camera {
    fn of(perspective: bool, flat: bool) -> Camera {
        match (perspective, flat) {
            (true, true) => Camera::Both,
            (true, false) => Camera::Perspective,
            (false, true) => Camera::Flat,
            (false, false) => Camera::None,
        }
    }

    pub fn perspective(self) -> bool {
        matches!(self, Camera::Perspective | Camera::Both)
    }

    pub fn flat(self) -> bool {
        matches!(self, Camera::Flat | Camera::Both)
    }

    /// What two samples of one pair showed together.
    pub fn merge(self, other: Camera) -> Camera {
        match (self, other) {
            (Camera::Unknown, o) | (o, Camera::Unknown) => o,
            (a, b) => Camera::of(a.perspective() || b.perspective(), a.flat() || b.flat()),
        }
    }
}

// A screen projection maps pixels or canvas units: 2 / extent between these.
const FLAT_SCALE: core::ops::RangeInclusive<f32> = 1.0e-4..=0.04;
const ZERO: f32 = 1.0e-6;

/// Whether `f` (16 floats) is an orthographic projection onto the screen: no rotation, no w, x and
/// y scaled by two over a screen-sized extent, and centred or anchored at a corner.
pub fn fit_screen_ortho(f: &[f32], row_major: bool) -> bool {
    if f.len() < 16 || f[..16].iter().any(|v| !v.is_finite()) {
        return false;
    }
    let m = read(f, row_major);
    let affine = m[12].abs() < ZERO && m[13].abs() < ZERO && m[14].abs() < ZERO && (m[15] - 1.0).abs() < 1e-4;
    let unrotated = [m[1], m[2], m[4], m[6], m[8], m[9]].iter().all(|v| v.abs() < ZERO);
    let anchored = |t: f32| t.abs() < 0.01 || (t.abs() - 1.0).abs() < 0.1;
    affine
        && unrotated
        && FLAT_SCALE.contains(&m[0].abs())
        && FLAT_SCALE.contains(&m[5].abs())
        && anchored(m[3])
        && anchored(m[7])
}

/// The cameras in one uniform block, at every vec4 boundary in both layouts.
pub fn classify_block(floats: &[f32]) -> Camera {
    let (mut perspective, mut flat) = (false, false);
    let mut at = 0;
    while at + 16 <= floats.len() && !(perspective && flat) {
        let m = &floats[at..at + 16];
        for row_major in [true, false] {
            perspective = perspective || fit_view_projection(m, row_major, 0.0).is_some();
            flat = flat || fit_screen_ortho(m, row_major);
        }
        at += 4;
    }
    Camera::of(perspective, flat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::matrix::mul;
    use crate::signals::matrix::test_util::{perspective, transpose, view};

    fn ortho(width: f32, height: f32) -> [f32; 16] {
        [2.0 / width, 0.0, 0.0, -1.0, 0.0, -2.0 / height, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn a_scene_block_shows_its_perspective() {
        let vp = mul(&perspective(50.0, 2.1, 0.3), &view([4.0, 2.0, 1.0], 20.0));
        let mut block = vec![0.25f32; 8];
        block.extend_from_slice(&transpose(&vp));
        block.extend_from_slice(&[1.0; 12]);
        assert_eq!(classify_block(&block), Camera::Perspective);
    }

    #[test]
    fn a_ui_block_shows_a_screen_projection_in_either_layout() {
        let mut block = vec![0.0f32; 4];
        block.extend_from_slice(&ortho(2400.0, 1080.0));
        assert_eq!(classify_block(&block), Camera::Flat);
        assert_eq!(classify_block(&transpose(&ortho(1920.0, 1080.0))), Camera::Flat);
    }

    #[test]
    fn rigid_transforms_and_noise_are_neither() {
        let model = view([1.0, 2.0, 3.0], 35.0);
        assert_eq!(classify_block(&model), Camera::None);
        let noise: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        assert_eq!(classify_block(&noise), Camera::None);
    }

    #[test]
    fn samples_merge() {
        assert_eq!(Camera::Unknown.merge(Camera::Flat), Camera::Flat);
        assert_eq!(Camera::Flat.merge(Camera::Perspective), Camera::Both);
        assert_eq!(Camera::None.merge(Camera::None), Camera::None);
        assert_eq!(Camera::Perspective.merge(Camera::Unknown), Camera::Perspective);
    }
}
