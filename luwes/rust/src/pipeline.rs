// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! One frame pair end to end: matcher, layer fields, the block and dense solves, the fusion record.

use crate::fusion::{self, FusionModel, FusionFeatures};
use crate::grid::{field_at, Grid, Image, Plane, V2};
use crate::layer::{layer_field, LayerField, LayerParams};
use crate::motion::{estimate, MatchParams, Matched};
use crate::solver::decide::decide;
use crate::solver::prep::{prepare, PairInputs};
use crate::solver::{Gain, Mode, Params};

/// Per-pixel flow pair seeded by the block field: (cur -> prev, prev -> cur), frame px.
pub trait DenseFlow {
    /// Flow pair for cur against prev, seeded per pixel.
    fn flow(&mut self, prev: &Image, cur: &Image, seed: &Grid<V2>) -> (Grid<V2>, Grid<V2>);
}

/// A frame pair and everything measured on it.
pub struct Pair {
    /// Previous frame.
    pub prev: Image,
    /// Current frame.
    pub cur: Image,
    /// Block matcher output.
    pub m: Matched,
    /// Colour layer field (block path).
    pub fl_colour: LayerField,
    /// Luma layer field (dense path).
    pub fl_luma: LayerField,
    /// Dense flow pair when a `DenseFlow` ran.
    pub dense: Option<(Grid<V2>, Grid<V2>)>,
}

/// Block field at every pixel (frame px): the dense matcher's seed and the fusion cue's reference.
pub fn block_vectors(p: &Pair) -> Grid<V2> {
    Grid::from_fn(p.cur.w, p.cur.h, |x, y| {
        let v = field_at(&p.m.f, x as f32, y as f32);
        [v[0], v[1]]
    })
}

impl Pair {
    /// Matcher, both layer fields and the optional dense flow.
    pub fn analyse(prev: Image, cur: Image, flow: Option<&mut dyn DenseFlow>) -> Pair {
        let m = estimate(&cur, &prev, &MatchParams::default());
        let fl_colour = layer_field(&cur, &prev, &m.f, m.meta.cut_r, &LayerParams::colour());
        let fl_luma = layer_field(&cur, &prev, &m.f, m.meta.cut_r, &LayerParams::luma());
        let mut p = Pair { prev, cur, m, fl_colour, fl_luma, dense: None };
        if let Some(fw) = flow {
            let seed = block_vectors(&p);
            p.dense = Some(fw.flow(&p.prev, &p.cur, &seed));
        }
        p
    }

    fn inputs<'a>(&'a self, prm: &Params) -> PairInputs<'a> {
        PairInputs {
            prev: &self.prev,
            cur: &self.cur,
            f: &self.m.f,
            rescue: Some(&self.m.rescue),
            cut_r: self.m.meta.cut_r,
            gain: Gain { scale: self.m.meta.gain.0, offset: self.m.meta.gain.1 },
            fl: Some(if prm.colour_layer { &self.fl_colour } else { &self.fl_luma }),
            dense: if prm.dense { self.dense.as_ref().map(|(a, b)| (a, b)) } else { None },
        }
    }

    /// One solve; `None` for the dense path without a flow.
    pub fn solve(&self, t: f32, mode: Mode, prm: &Params) -> Option<(Image, Plane)> {
        if prm.dense && self.dense.is_none() {
            return None;
        }
        let z = prepare(&self.inputs(prm), t, mode, prm);
        let d = decide(&z, prm);
        Some((d.out, d.warp_error))
    }

    /// The generated frame: block warp, then the net's blend with the dense-flow warp when both a flow
    /// and a model exist.
    pub fn generate(&self, t: f32, mode: Mode, net: Option<&mut dyn FusionModel>) -> Image {
        let (ob, eb) = self.solve(t, mode, &Params::block_warp()).unwrap();
        let Some(net) = net else { return ob };
        let Some((od, ed)) = self.solve(t, mode, &Params::dense_flow()) else { return ob };
        let rec = self.fusion_features(&ob, &od, &eb, &ed);
        fusion::fused_frame(&ob, rec.as_ref(), t, mode == Mode::Extrap, Some(net))
    }

    /// Fusion record of the two solves (needs the dense flow).
    pub fn fusion_features(&self, ob: &Image, od: &Image, eb: &Plane, ed: &Plane) -> Option<FusionFeatures> {
        let (fd, _) = self.dense.as_ref()?;
        Some(fusion::FusionFeatures::new(&self.prev, &self.cur, ob, od, eb, ed, fd, &block_vectors(self)))
    }
}
