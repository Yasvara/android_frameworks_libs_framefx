// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Row-major 2-D grids and the sampling primitives every stage shares. Arithmetic follows the
//! reference's float32 order so decisions on thresholds land on the same side.

/// Linear RGB in [0, 1].
pub type Rgb = [f32; 3];
/// 2-D vector (x, y).
pub type V2 = [f32; 2];

#[derive(Clone, Debug, PartialEq)]
/// Row-major w x h grid.
pub struct Grid<T> {
    /// Width.
    pub w: usize,
    /// Height.
    pub h: usize,
    /// Row-major cells.
    pub d: Vec<T>,
}

impl<T: Copy> Grid<T> {
    /// Filled with v.
    pub fn new(w: usize, h: usize, v: T) -> Self {
        Grid { w, h, d: vec![v; w * h] }
    }

    /// Wraps a row-major vector.
    pub fn from_vec(w: usize, h: usize, d: Vec<T>) -> Self {
        assert_eq!(d.len(), w * h);
        Grid { w, h, d }
    }

    /// Cell (x, y) from f(x, y).
    pub fn from_fn(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut d = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                d.push(f(x, y));
            }
        }
        Grid { w, h, d }
    }

    /// Cell (x, y).
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> T {
        self.d[y * self.w + x]
    }

    /// Store at (x, y).
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        self.d[y * self.w + x] = v;
    }

    /// Clamped integer fetch.
    #[inline]
    pub fn atc(&self, x: i64, y: i64) -> T {
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        self.d[y * self.w + x]
    }

    /// Element-wise map.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Grid<U> {
        Grid { w: self.w, h: self.h, d: self.d.iter().map(|&v| f(v)).collect() }
    }

    /// Element-wise map of two grids of one size.
    pub fn zip<U: Copy, R: Copy>(&self, o: &Grid<U>, f: impl Fn(T, U) -> R) -> Grid<R> {
        assert!(self.w == o.w && self.h == o.h);
        Grid { w: self.w, h: self.h, d: self.d.iter().zip(&o.d).map(|(&a, &b)| f(a, b)).collect() }
    }

    /// Sub-grid.
    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Grid<T> {
        Grid::from_fn(w, h, |x, y| self.at(x0 + x, y0 + y))
    }
}

impl Grid<bool> {
    /// Any cell set.
    pub fn any(&self) -> bool {
        self.d.iter().any(|&b| b)
    }
    /// Set cells.
    pub fn count(&self) -> usize {
        self.d.iter().filter(|&&b| b).count()
    }
}

/// Scalar image.
pub type Plane = Grid<f32>;
/// Colour image.
pub type Image = Grid<Rgb>;
/// Boolean image.
pub type Mask = Grid<bool>;

/// D(a, b) = 0.5 |dr| + |dg| + 0.5 |db| (the solver's colour distance).
#[inline]
pub fn dist(a: Rgb, b: Rgb) -> f32 {
    0.5 * (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + 0.5 * (a[2] - b[2]).abs()
}

/// float32 hypot (correctly rounded through float64).
#[inline]
pub fn hypot(x: f32, y: f32) -> f32 {
    ((x as f64) * (x as f64) + (y as f64) * (y as f64)).sqrt() as f32
}

/// Hermite step on float32 edges.
#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// ~1 for x << tau, ~0 for x >> tau.
#[inline]
pub fn soft_below(x: f32, tau: f32, soft: f32) -> f32 {
    1.0 - smoothstep(tau * (1.0 - soft), tau * (1.0 + soft), x)
}

/// RGBA8 store (round half to even).
#[inline]
pub fn q8(v: f32) -> f32 {
    (((v.clamp(0.0, 1.0) as f64) * 255.0).round_ties_even() / 255.0) as f32
}

/// Bilinear with clamp-to-edge; the base texel is clamped to w-2 so the right / bottom edge reads
/// the last pair with weight 1.
#[inline]
fn bil_setup(n: usize, v: f32) -> (usize, f32) {
    let v = v.clamp(0.0, (n - 1) as f32);
    let i = (v.floor() as i32).min((n as i32 - 2).max(0)) as usize;
    (i, v - i as f32)
}

#[inline]
fn bil_setup64(n: usize, v: f64) -> (usize, f32) {
    let v = v.clamp(0.0, (n - 1) as f64);
    let i = (v.floor() as i32).min((n as i32 - 2).max(0)) as usize;
    (i, (v - i as f64) as f32)
}

/// Bilinear blend of four samples.
pub trait Lerp: Copy {
    /// (a + (b - a) fx)(1 - fy) + (c + (d - c) fx) fy.
    fn bil(a: Self, b: Self, c: Self, d: Self, fx: f32, fy: f32) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn bil(a: f32, b: f32, c: f32, d: f32, fx: f32, fy: f32) -> f32 {
        (a + (b - a) * fx) * (1.0 - fy) + (c + (d - c) * fx) * fy
    }
}

impl<const N: usize> Lerp for [f32; N] {
    #[inline]
    fn bil(a: Self, b: Self, c: Self, d: Self, fx: f32, fy: f32) -> Self {
        let mut o = [0.0; N];
        for k in 0..N {
            o[k] = f32::bil(a[k], b[k], c[k], d[k], fx, fy);
        }
        o
    }
}

impl<T: Lerp> Grid<T> {
    #[inline]
    fn bil_at(&self, x0: usize, fx: f32, y0: usize, fy: f32) -> T {
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        T::bil(self.at(x0, y0), self.at(x1, y0), self.at(x0, y1), self.at(x1, y1), fx, fy)
    }

    /// Bilinear, clamp-to-edge.
    #[inline]
    pub fn bilinear(&self, x: f32, y: f32) -> T {
        let (x0, fx) = bil_setup(self.w, x);
        let (y0, fy) = bil_setup(self.h, y);
        self.bil_at(x0, fx, y0, fy)
    }

    /// Coordinates the reference computed in float64.
    #[inline]
    pub fn bilinear64(&self, x: f64, y: f64) -> T {
        let (x0, fx) = bil_setup64(self.w, x);
        let (y0, fy) = bil_setup64(self.h, y);
        self.bil_at(x0, fx, y0, fy)
    }
}

/// Field texels are 32 frame px; vectors are level-1 px (x4 = frame px).
/// Frame px per field texel.
pub const FIELD_TEXEL: f32 = 32.0;
/// Frame px per field texel.
pub const FIELD_TEXEL_I: usize = 32;
/// Level-1 px to frame px.
pub const FIELD_SCALE: f32 = 4.0;

/// Block field: xy back vector (level px), z match error, w confidence.
pub type Field = Grid<[f32; 4]>;

/// texture(field, (q + 0.5) / size): bilinear over texel centres, xy in frame px.
#[inline]
pub fn field_at(f: &Field, x: f32, y: f32) -> [f32; 4] {
    let mut v = f.bilinear((x + 0.5) / FIELD_TEXEL - 0.5, (y + 0.5) / FIELD_TEXEL - 0.5);
    v[0] *= FIELD_SCALE;
    v[1] *= FIELD_SCALE;
    v
}

/// 5-bilinear-tap Catmull-Rom (corner taps dropped, weights renormalised), clipped to [0, 1].
pub fn catmull_rom(img: &Image, x: f32, y: f32) -> Rgb {
    let (fx0, fy0) = (x.floor(), y.floor());
    let (fx, fy) = (x - fx0, y - fy0);
    let w = |f: f32| {
        let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        let w3 = f * f * (-0.5 + 0.5 * f);
        (w0, w1 + w2, w2 / (w1 + w2), w3)
    };
    let (w0x, w12x, o12x, w3x) = w(fx);
    let (w0y, w12y, o12y, w3y) = w(fy);
    let (x0, x12, x3) = (fx0 - 1.0, fx0 + o12x, fx0 + 2.0);
    let (y0, y12, y3) = (fy0 - 1.0, fy0 + o12y, fy0 + 2.0);
    let taps = [
        (x12, y0, w12x * w0y),
        (x0, y12, w0x * w12y),
        (x12, y12, w12x * w12y),
        (x3, y12, w3x * w12y),
        (x12, y3, w12x * w3y),
    ];
    let mut acc = [0.0f32; 3];
    let mut tot = 0.0f32;
    for (tx, ty, wt) in taps {
        let s = img.bilinear(tx, ty);
        for k in 0..3 {
            acc[k] += s[k] * wt;
        }
        tot += wt;
    }
    acc.map(|a| (a / tot).clamp(0.0, 1.0))
}

/// ceil(n / d).
pub fn grid_dims(n: usize, d: usize) -> usize {
    n.div_ceil(d)
}

/// numpy's float32 pairwise sum over a contiguous run (8 accumulators up to 128, halving above).
pub fn np_sum(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f32;
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// numpy's float64 pairwise sum.
pub fn np_sum64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0, |r, &v| r + v)
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum64(&a[..n2]) + np_sum64(&a[n2..])
    }
}

/// numpy float32 mean of a contiguous run.
pub fn np_mean(a: &[f32]) -> f32 {
    np_sum(a) / a.len() as f32
}

/// np.std (population) of a float32 run.
pub fn np_std(a: &[f32]) -> f32 {
    let m = np_mean(a);
    let sq: Vec<f32> = a.iter().map(|&v| (v - m) * (v - m)).collect();
    (np_sum(&sq) / a.len() as f32).sqrt()
}

/// np.median of float32 values (mean of the two middle ones for even counts, in float32).
pub fn median(v: &mut [f32]) -> f32 {
    let n = v.len();
    assert!(n > 0);
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// np.median of float64 values.
pub fn median64(v: &mut [f64]) -> f64 {
    let n = v.len();
    assert!(n > 0);
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
