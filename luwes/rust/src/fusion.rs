// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Fusion of the block-warp and dense-flow outputs. A small U-Net on half-res cues predicts a
//! per-pixel weight w and a residual r; the blend runs at full res:
//!     out = o_block + clip(w, 0, 1) (o_dense - o_block) + 0.25 r
//! Without a loaded model the block output passes through unchanged.

use crate::grid::{Grid, Image, Plane, V2};

/// Model input channels: prev, cur, block warp, dense warp (RGB), both warp errors, |dense - block|, t, extrap.
pub const CIN: usize = 17;
/// Net output channels: w, r (RGB).
pub const COUT: usize = 4;
/// Residual scale of the net's r output.
pub const RSCALE: f32 = 0.25;

/// Planar float tensor [c][h][w].
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Channels.
    pub c: usize,
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// Planar data.
    pub d: Vec<f32>,
}

impl Tensor {
    /// All zero.
    pub fn zeros(c: usize, h: usize, w: usize) -> Tensor {
        Tensor { c, h, w, d: vec![0.0; c * h * w] }
    }
    /// Element (c, y, x).
    #[inline]
    pub fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.d[(c * self.h + y) * self.w + x]
    }
    /// Store at (c, y, x).
    #[inline]
    pub fn set(&mut self, c: usize, y: usize, x: usize, v: f32) {
        self.d[(c * self.h + y) * self.w + x] = v;
    }
}

/// Fusion model backend (HTP V73 through QNN on device). `None` = no model or a failed run; the caller
/// falls back to the block output.
pub trait FusionModel {
    /// [CIN, h/2, w/2] -> [COUT, h/2, w/2].
    fn infer(&mut self, x: &Tensor) -> Option<Tensor>;
}

#[inline]
fn u8q(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Fusion features at full res, 8-bit as the net was trained: frames and both warps as rendered,
/// warp errors x4, dense-vs-block vector gap x16 px.
pub struct FusionFeatures {
    /// Width.
    pub w: usize,
    /// Height.
    pub h: usize,
    /// [h][w][15]: prev, cur, block warp, dense warp (RGB), block / dense warp error, |dv|.
    pub d: Vec<[u8; 15]>,
}

impl FusionFeatures {
    /// The record of one solved pair: both warps and their two-sided warp-error maps.
    pub fn new(prev: &Image, cur: &Image, o_block: &Image, o_dense: &Image, err_block: &Plane, err_dense: &Plane, dense_v: &Grid<V2>, block_v: &Grid<V2>) -> FusionFeatures {
        let (w, h) = (cur.w, cur.h);
        let mut d = Vec::with_capacity(w * h);
        for i in 0..w * h {
            let mut r = [0u8; 15];
            for c in 0..3 {
                r[c] = u8q(prev.d[i][c]);
                r[3 + c] = u8q(cur.d[i][c]);
                r[6 + c] = u8q(o_block.d[i][c]);
                r[9 + c] = u8q(o_dense.d[i][c]);
            }
            r[12] = u8q(err_block.d[i] * 4.0);
            r[13] = u8q(err_dense.d[i] * 4.0);
            let (a, b) = (dense_v.d[i], block_v.d[i]);
            let dv = (((a[0] - b[0]) as f64).hypot((a[1] - b[1]) as f64)) as f32;
            r[14] = (dv * 16.0 + 0.5).clamp(0.0, 255.0) as u8;
            d.push(r);
        }
        FusionFeatures { w, h, d }
    }

    /// Half-res net input: 2x2 box of the record channels (/255) plus the t and extrap planes.
    pub fn net_input(&self, t: f32, extrap: bool) -> Tensor {
        let rec = self;
        let (h2, w2) = (rec.h / 2, rec.w / 2);
        let mut x = Tensor::zeros(CIN, h2, w2);
        let ex = if extrap { 1.0 } else { 0.0 };
        for y in 0..h2 {
            for xx in 0..w2 {
                let at = |dx: usize, dy: usize| &rec.d[(2 * y + dy) * rec.w + 2 * xx + dx];
                let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
                for ch in 0..15 {
                    let s = ((a[ch] as f32 / 255.0 + b[ch] as f32 / 255.0) + c[ch] as f32 / 255.0) + d[ch] as f32 / 255.0;
                    x.set(ch, y, xx, s / 4.0);
                }
                x.set(15, y, xx, t);
                x.set(16, y, xx, ex);
            }
        }
        x
    }
}

/// Bilinear x2 upsample, half-pixel centres (align_corners = false).
fn up2(y: &Tensor, c: usize, oy: usize, ox: usize) -> f32 {
    let src = |d: usize, n: usize| {
        let s = (0.5 * (d as f32 + 0.5) - 0.5).max(0.0);
        let i0 = s as usize;
        let i1 = if i0 < n - 1 { i0 + 1 } else { i0 };
        (i0, i1, s - i0 as f32)
    };
    let (y0, y1, ly) = src(oy, y.h);
    let (x0, x1, lx) = src(ox, y.w);
    (1.0 - ly) * ((1.0 - lx) * y.at(c, y0, x0) + lx * y.at(c, y0, x1)) + ly * ((1.0 - lx) * y.at(c, y1, x0) + lx * y.at(c, y1, x1))
}

/// Full-res blend of the 8-bit block / dense outputs under the net's half-res output.
pub fn fuse_warps(rec: &FusionFeatures, y: &Tensor) -> Image {
    assert!(y.c == COUT && y.h * 2 == rec.h && y.w * 2 == rec.w);
    Grid::from_fn(rec.w, rec.h, |x, yy| {
        let r = &rec.d[yy * rec.w + x];
        let wv = up2(y, 0, yy, x).clamp(0.0, 1.0);
        [0, 1, 2].map(|c| {
            let ob = r[6 + c] as f32 / 255.0;
            let od = r[9 + c] as f32 / 255.0;
            let o = ob + wv * (od - ob) + RSCALE * up2(y, 1 + c, yy, x);
            (o.clamp(0.0, 1.0) * 255.0).round_ties_even() / 255.0
        })
    })
}

/// The frame to present: the net's blend when a model ran, the block output otherwise.
pub fn fused_frame(o_block: &Image, rec: Option<&FusionFeatures>, t: f32, extrap: bool, net: Option<&mut dyn FusionModel>) -> Image {
    let (Some(rec), Some(net)) = (rec, net) else { return o_block.clone() };
    if rec.w % 2 != 0 || rec.h % 2 != 0 {
        return o_block.clone();
    }
    let x = rec.net_input(t, extrap);
    match net.infer(&x) {
        Some(y) if y.c == COUT && y.h == x.h && y.w == x.w => fuse_warps(rec, &y),
        _ => o_block.clone(),
    }
}
