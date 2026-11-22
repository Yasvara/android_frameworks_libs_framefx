// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! 4x4 matrices as games upload them: shape tests that tell a perspective projection, a
//! view-projection and a rigid object transform apart in raw constant memory, in either layout.

pub type Mat4 = [f32; 16];

pub const IDENTITY: Mat4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// Row-major product.
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            let s: f64 = (0..4).map(|k| f64::from(a[r * 4 + k]) * f64::from(b[k * 4 + c])).sum();
            m[r * 4 + c] = s as f32;
        }
    }
    m
}

pub fn apply(m: &Mat4, p: [f32; 4]) -> [f32; 4] {
    let mut q = [0.0f32; 4];
    for (r, out) in q.iter_mut().enumerate() {
        *out = (0..4).map(|k| m[r * 4 + k] * p[k]).sum();
    }
    q
}

pub use crate::gbuffer::invert;

fn dot3(a: &[f32], b: &[f32]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn len3(a: &[f32]) -> f32 {
    dot3(a, a).sqrt()
}

/// Reads 16 floats as a row-major matrix.
pub fn read(f: &[f32], row_major: bool) -> Mat4 {
    let mut m = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            m[r * 4 + c] = if row_major { f[r * 4 + c] } else { f[c * 4 + r] };
        }
    }
    m
}

/// A view-projection P V with V rigid, decomposed: what tells two of them apart.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraFit {
    /// Row-major, as uploaded (jitter included).
    pub m: Mat4,
    /// Focal scales.
    pub sx: f32,
    pub sy: f32,
    /// The NDC offset the projection adds: jitter, plus any fixed off-centre shift.
    pub jx: f32,
    pub jy: f32,
    /// The view axis in world space (unit), sign as the w row has it.
    pub axis: [f32; 3],
    /// The translation column is non-zero: world space, not camera-relative.
    pub world: bool,
    pub row_major: bool,
}

/// Whether `f` (16 floats) is P V: the w row a unit direction (the view axis), the x and y rows,
/// jitter taken out, perpendicular to it and to each other. `aspect` (x over y of the view, either
/// way up) is checked when non-zero.
pub fn fit_view_projection(f: &[f32], row_major: bool, aspect: f32) -> Option<CameraFit> {
    if f.len() < 16 || f[..16].iter().any(|v| !v.is_finite()) {
        return None;
    }
    let m = read(f, row_major);
    let (x, y, w) = (&m[0..4], &m[4..8], &m[12..16]);
    let ww = dot3(w, w);
    if (ww - 1.0).abs() > 0.02 {
        return None;
    }
    let jx = dot3(x, w) / ww;
    let jy = dot3(y, w) / ww;
    let xs = [x[0] - jx * w[0], x[1] - jx * w[1], x[2] - jx * w[2]];
    let ys = [y[0] - jy * w[0], y[1] - jy * w[1], y[2] - jy * w[2]];
    let (sx, sy) = (len3(&xs), len3(&ys));
    if !(0.05..=50.0).contains(&sx) || !(0.05..=50.0).contains(&sy) || jx.abs() > 0.05 || jy.abs() > 0.05
    {
        return None;
    }
    if dot3(&xs, &ys).abs() > 0.01 * sx * sy {
        return None;
    }
    if aspect > 0.0 {
        let a = sy / sx;
        let want = aspect;
        if (a / want - 1.0).abs() > 0.02 && (a * want - 1.0).abs() > 0.02 {
            return None;
        }
    }
    let world = w[3].abs() > 1e-3 || x[3].abs() > 1e-3 || y[3].abs() > 1e-3;
    Some(CameraFit {
        m,
        sx,
        sy,
        jx,
        jy,
        axis: [w[0] / ww.sqrt(), w[1] / ww.sqrt(), w[2] / ww.sqrt()],
        world,
        row_major,
    })
}

/// The view-projection without its NDC offset (jitter adds `j * w` to clip x and y).
pub fn without_offset(m: &Mat4, jx: f32, jy: f32) -> Mat4 {
    crate::gbuffer::unjittered(m, [jx, jy])
}

/// Whether `f` is a rigid object-to-world transform (rotation, uniform-ish scale, translation,
/// last row 0 0 0 1), in either layout. The identity counts: static geometry.
pub fn fit_rigid(f: &[f32], row_major: bool) -> Option<Mat4> {
    if f.len() < 16 || f[..16].iter().any(|v| !v.is_finite()) {
        return None;
    }
    let m = read(f, row_major);
    if m[12].abs() > 1e-5 || m[13].abs() > 1e-5 || m[14].abs() > 1e-5 || (m[15] - 1.0).abs() > 1e-5 {
        return None;
    }
    let cols: [[f32; 3]; 3] = core::array::from_fn(|c| [m[c], m[4 + c], m[8 + c]]);
    let lens: [f32; 3] = core::array::from_fn(|c| len3(&cols[c]));
    if lens.iter().any(|l| !(1e-3..=1e4).contains(l)) {
        return None;
    }
    for a in 0..3 {
        for b in a + 1..3 {
            if dot3(&cols[a], &cols[b]).abs() > 0.02 * lens[a] * lens[b] {
                return None;
            }
        }
    }
    Some(m)
}

/// Angle between two unit view axes, in degrees.
pub fn axis_angle(a: [f32; 3], b: [f32; 3]) -> f32 {
    dot3(&a, &b).clamp(-1.0, 1.0).acos().to_degrees()
}

/// The eye of a world-space P V: the one point its x, y and w rows all send to 0.
pub fn eye(m: &Mat4) -> Option<[f32; 3]> {
    // Rows x, y, w: r . (p, 1) = 0 for the eye p.
    let rows = [&m[0..4], &m[4..8], &m[12..16]];
    let a = [
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ];
    let b = [-rows[0][3], -rows[1][3], -rows[2][3]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-9 {
        return None;
    }
    let solve = |col: usize| {
        let mut t = a;
        for r in 0..3 {
            t[r][col] = b[r];
        }
        (t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0])
            + t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0]))
            / det
    };
    Some([solve(0), solve(1), solve(2)])
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    pub fn transpose(m: &Mat4) -> Mat4 {
        let mut t = [0.0f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                t[c * 4 + r] = m[r * 4 + c];
            }
        }
        t
    }

    /// Reversed infinite perspective looking down -z.
    pub fn perspective(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
        let sy = 1.0 / (fov_y.to_radians() / 2.0).tan();
        [sy / aspect, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 0.0, near, 0.0, 0.0, -1.0, 0.0]
    }

    pub fn translate(x: f32, y: f32, z: f32) -> Mat4 {
        [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0]
    }

    pub fn yaw(deg: f32) -> Mat4 {
        let (s, c) = deg.to_radians().sin_cos();
        [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0]
    }

    /// View of a camera at `pos` turned `deg` about y.
    pub fn view(pos: [f32; 3], deg: f32) -> Mat4 {
        mul(&yaw(-deg), &translate(-pos[0], -pos[1], -pos[2]))
    }

    pub fn jittered(vp: &Mat4, jx: f32, jy: f32) -> Mat4 {
        let mut m = *vp;
        for c in 0..4 {
            m[c] += jx * vp[12 + c];
            m[4 + c] += jy * vp[12 + c];
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::*;
    use super::*;

    #[test]
    fn a_view_projection_fits_in_both_layouts_and_gives_its_jitter() {
        let vp = jittered(&mul(&perspective(45.0, 2.16, 0.25), &view([3.0, 1.0, -2.0], 30.0)), 0.0004, -0.0007);
        let row = fit_view_projection(&vp, true, 2.16).unwrap();
        assert!((row.jx - 0.0004).abs() < 1e-6 && (row.jy + 0.0007).abs() < 1e-6);
        assert!(row.world);
        let col = fit_view_projection(&transpose(&vp), false, 2.16).unwrap();
        assert_eq!(col.m, row.m);
        assert!(fit_view_projection(&vp, true, 1.0).is_none(), "aspect checked");
        assert!(fit_view_projection(&IDENTITY, true, 0.0).is_none(), "w row of the identity is not a view axis");
    }

    #[test]
    fn the_eye_is_recovered() {
        let vp = mul(&perspective(60.0, 1.5, 0.1), &view([3.0, 1.0, -2.0], 30.0));
        let e = eye(&vp).unwrap();
        for (a, b) in e.iter().zip([3.0, 1.0, -2.0]) {
            assert!((a - b).abs() < 1e-3, "{e:?}");
        }
    }

    #[test]
    fn rigid_transforms_fit_and_projections_do_not() {
        let m = mul(&translate(1.0, 2.0, 3.0), &yaw(40.0));
        assert!(fit_rigid(&m, true).is_some());
        assert!(fit_rigid(&transpose(&m), false).is_some());
        assert!(fit_rigid(&IDENTITY, true).is_some());
        assert!(fit_rigid(&perspective(45.0, 2.0, 0.1), true).is_none());
        let mut skew = m;
        skew[1] = 0.7;
        assert!(fit_rigid(&skew, true).is_none());
    }
}
