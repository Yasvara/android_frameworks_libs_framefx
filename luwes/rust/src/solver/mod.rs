// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Generated-frame solver: candidate trajectories from the block field (own / neighbour / forward
//! projected / layer / dominant / rescue texels), two-sided photometric choice, occlusion and
//! static-lock rules, layer assignment (covered / uncovered bands, object-at-t), edge and
//! horizon handling. `Params::block_warp` is the block path, `Params::dense_flow` the dense path.

pub mod decide;
pub(crate) mod layers;
pub(crate) mod fieldops;
pub mod prep;

use crate::grid::{dist, hypot, Grid, Image, Rgb, V2, FIELD_SCALE};

/// Where the generated frame sits: between prev and cur (t in (0, 1)) or past cur (t > 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Interpolation.
    Interp,
    /// Extrapolation.
    Extrap,
}

/// Every switch and threshold either path reads. Floats are f64 as the reference's Python
/// scalars; they meet float32 data the way numpy does (scalar cast to f32 first).
#[derive(Clone, Debug)]
pub struct Params {
    pub(crate) agree_tau: f64,
    pub(crate) spread_tau: f64,
    pub(crate) smooth_weight: f64,
    pub(crate) shield_rho: f64,
    pub(crate) collision_rho: f64,
    pub(crate) noise_tau: f64,
    pub(crate) collision_tau: f64,
    pub(crate) occluder_gain: f64,
    pub(crate) soft_width: f64,
    pub(crate) dedupe_dist: f64,
    pub(crate) lock_tau: f64,
    pub(crate) collision_contrast: f64,
    pub(crate) trusted_cost: f64,
    pub(crate) min_support: i32,
    pub(crate) distinct_tau: f64,
    /// Soft multi-hypothesis fusion temperature (0 = hard winner).
    pub fuse_temperature: f64,
    pub(crate) fuse_radius: f64,
    pub(crate) support_radius: f64,
    pub(crate) projected_candidates: usize,
    pub(crate) projection_margin: f64,
    pub(crate) potts_weight: f64,
    pub(crate) potts_radius: f64,
    pub(crate) arrival_tau: f64,
    pub(crate) overlay_tau: f64,
    pub(crate) overlay_own_tau: f64,
    pub(crate) cut_lo: f64,
    pub(crate) cut_hi: f64,

    pub(crate) layer_bands: bool,
    pub(crate) layer_min_rel: f64,
    pub(crate) layer_strict_cost: f64,
    pub(crate) layer_cost: f64,
    pub(crate) layer_agree_tau: f64,
    pub(crate) layer_min_texels: usize,
    pub(crate) layer_unexplained_cost: f64,
    pub(crate) divergence_k: f64,
    pub(crate) divergence_sigma_floor: f64,
    pub(crate) object_potts_weight: f64,
    pub(crate) grad_tolerance: f64,
    pub(crate) grad_tolerance_k: f64,
    pub(crate) membership_grow: usize,
    pub(crate) joint_rounds: usize,
    pub(crate) membership_cell: usize,
    pub(crate) cell_explained: f64,
    pub(crate) cell_margin: f64,
    pub(crate) membership_eps: f64,
    pub(crate) object_sigma: f64,
    pub(crate) cell_potts_weight: f64,
    pub(crate) edge_margin: f64,
    pub(crate) edge_min_support: u8,
    pub(crate) edge_extend_overlap: f64,
    pub(crate) hole_exponent: f64,

    pub(crate) static_object_pass: bool,
    pub(crate) dominant_candidate: bool,
    pub(crate) entering_cells: bool,
    pub(crate) horizon_both_out: bool,
    pub(crate) refine_compass: bool,
    pub(crate) refine_coarse_step: usize,
    pub(crate) seed_all_tiles: bool,
    pub(crate) pixel_membership: bool,
    pub(crate) pixel_membership_needs_constancy: bool,
    pub(crate) pixel_membership_sep: f64,
    pub(crate) thin_layer: f64,
    pub(crate) seed_min_explained: f64,
    pub(crate) pixel_membership_dim_only: bool,
    pub(crate) zero_motion_margin: f64,
    pub(crate) zero_motion_tau: f64,
    pub(crate) zero_motion_grad: f64,
    pub(crate) zero_motion_ratio: f64,
    pub(crate) zero_motion_max_support: u8,
    pub(crate) layer_cubic: bool,
    pub(crate) mover_pixel_ownership: bool,
    pub(crate) ground_coincidence: bool,
    pub(crate) ground_coincidence_tau: f64,
    pub(crate) ground_occupancy_tau: f64,
    pub(crate) ground_coincidence_min_cost: f64,
    pub(crate) ground_coincidence_layer_only: bool,
    pub(crate) ground_coincidence_one_end: bool,
    pub(crate) extrap_unsupported_to_base: bool,
    pub(crate) extrap_unsupported_support: u8,
    pub(crate) extrap_trust_supported: bool,
    pub(crate) extrap_trusted_support: u8,
    pub(crate) extrap_trusted_own: f64,
    pub(crate) silhouette_unlock: bool,
    pub(crate) silhouette_radius: usize,
    pub(crate) boundary_membership: bool,
    pub(crate) band_coincidence: bool,
    pub(crate) band_coincidence_one_side: bool,
    pub(crate) paired_rescue: i32,
    pub(crate) per_mover_regularise: bool,
    pub(crate) wake_coincidence: bool,
    pub(crate) object_ground_veto: i32,
    pub(crate) object_ground_rel: f64,
    pub(crate) object_ground_min_cost: f64,
    pub(crate) object_ground_rim: usize,

    /// Dense path: the per-pixel flow is candidate 0 (needs a `DenseFlow`).
    pub dense: bool,
    pub(crate) dense_cost_box: usize,
    pub(crate) cubic_output: bool,
    pub(crate) extrap_fixed_point: usize,
    pub(crate) dense_unlock_tau: f64,
    pub(crate) dense_band_tau: f64,
    pub(crate) dense_no_horizon: bool,
    /// Which layer field feeds the candidates: the colour build (block path) or the luma build.
    pub colour_layer: bool,
}

impl Params {
    /// Block path: field candidates, layer bands, object-at-t.
    pub fn block_warp() -> Params {
        Params {
            agree_tau: 0.15,
            spread_tau: 1.0,
            smooth_weight: 0.005,
            shield_rho: 0.05,
            collision_rho: 0.5,
            noise_tau: 8.0 / 255.0,
            collision_tau: 0.5,
            occluder_gain: 1.0,
            soft_width: 0.95,
            dedupe_dist: 0.25,
            lock_tau: 4.0 / 255.0,
            collision_contrast: 2.0,
            trusted_cost: 0.08,
            min_support: 2,
            distinct_tau: 0.35,
            fuse_temperature: 0.3,
            fuse_radius: 1.0,
            support_radius: 0.75,
            projected_candidates: 2,
            projection_margin: 24.0,
            potts_weight: 0.02,
            potts_radius: 0.25,
            arrival_tau: 6.0 / 255.0,
            overlay_tau: 24.0 / 255.0,
            overlay_own_tau: 16.0 / 255.0,
            cut_lo: 1.3,
            cut_hi: 1.8,
            layer_bands: true,
            layer_min_rel: 2.0,
            layer_strict_cost: 0.03,
            layer_cost: 0.03,
            layer_agree_tau: 0.04,
            layer_min_texels: 2,
            layer_unexplained_cost: 0.01,
            divergence_k: 3.0,
            divergence_sigma_floor: 0.05,
            object_potts_weight: (9.0f64).ln() / 4.0,
            grad_tolerance: 0.15,
            grad_tolerance_k: 2.0,
            membership_grow: 2,
            joint_rounds: 1,
            membership_cell: 8,
            cell_explained: 0.7,
            cell_margin: 0.1,
            membership_eps: 0.1,
            object_sigma: 0.01,
            cell_potts_weight: 1.0,
            edge_margin: 0.5,
            edge_min_support: 5,
            edge_extend_overlap: 0.5,
            hole_exponent: 0.7,
            static_object_pass: true,
            dominant_candidate: true,
            entering_cells: true,
            horizon_both_out: true,
            refine_compass: true,
            refine_coarse_step: 2,
            seed_all_tiles: true,
            pixel_membership: true,
            pixel_membership_needs_constancy: true,
            pixel_membership_sep: 4.0,
            thin_layer: 0.5,
            seed_min_explained: 0.15,
            pixel_membership_dim_only: true,
            zero_motion_margin: 0.02,
            zero_motion_tau: 0.05,
            zero_motion_grad: 0.1,
            zero_motion_ratio: 0.25,
            zero_motion_max_support: 6,
            layer_cubic: true,
            mover_pixel_ownership: true,
            ground_coincidence: true,
            ground_coincidence_tau: 0.01,
            ground_occupancy_tau: 0.01,
            ground_coincidence_min_cost: 0.02,
            ground_coincidence_layer_only: true,
            ground_coincidence_one_end: true,
            extrap_unsupported_to_base: true,
            extrap_unsupported_support: 2,
            extrap_trust_supported: true,
            extrap_trusted_support: 7,
            extrap_trusted_own: 0.5,
            silhouette_unlock: true,
            silhouette_radius: 4,
            boundary_membership: true,
            band_coincidence: true,
            band_coincidence_one_side: true,
            paired_rescue: 1,
            per_mover_regularise: true,
            wake_coincidence: true,
            object_ground_veto: 2,
            object_ground_rel: 0.25,
            object_ground_min_cost: 0.02,
            object_ground_rim: 2,
            dense: false,
            dense_cost_box: 0,
            cubic_output: false,
            extrap_fixed_point: 0,
            dense_unlock_tau: 0.0,
            dense_band_tau: 0.0,
            dense_no_horizon: false,
            colour_layer: true,
        }
    }

    /// Dense path: dense candidate 0, Catmull-Rom output taps, no layer bands.
    pub fn dense_flow() -> Params {
        Params {
            layer_bands: false,
            static_object_pass: false,
            dominant_candidate: false,
            entering_cells: false,
            horizon_both_out: false,
            refine_compass: false,
            refine_coarse_step: 1,
            seed_all_tiles: false,
            pixel_membership: false,
            pixel_membership_needs_constancy: false,
            pixel_membership_sep: 0.0,
            thin_layer: 0.0,
            seed_min_explained: 0.0,
            pixel_membership_dim_only: false,
            zero_motion_margin: 0.0,
            zero_motion_grad: 0.0,
            zero_motion_max_support: 0,
            layer_cubic: false,
            mover_pixel_ownership: false,
            ground_coincidence: false,
            ground_coincidence_tau: 0.0,
            ground_occupancy_tau: 0.0,
            ground_coincidence_layer_only: false,
            ground_coincidence_one_end: false,
            extrap_unsupported_to_base: false,
            extrap_trust_supported: false,
            silhouette_unlock: false,
            silhouette_radius: 2,
            boundary_membership: false,
            band_coincidence: false,
            band_coincidence_one_side: false,
            paired_rescue: 0,
            per_mover_regularise: false,
            wake_coincidence: false,
            object_ground_veto: 0,
            object_ground_rel: 1e9,
            object_ground_min_cost: 0.0,
            object_ground_rim: 0,
            dense: true,
            dense_cost_box: 2,
            cubic_output: true,
            extrap_fixed_point: 3,
            dense_unlock_tau: 0.08,
            dense_band_tau: 0.08,
            dense_no_horizon: true,
            colour_layer: false,
            ..Params::block_warp()
        }
    }
}

/// Python float meeting float32 data: the scalar is cast first.
#[inline]
pub fn f(v: f64) -> f32 {
    v as f32
}

/// Vector length in level-1 px.
#[inline]
pub fn lvl(v: V2) -> f32 {
    hypot(v[0], v[1]) / FIELD_SCALE
}

/// a - b.
#[inline]
pub fn sub(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}

/// Bilinear colour tap.
#[inline]
pub fn tex(img: &Image, x: f32, y: f32) -> Rgb {
    img.bilinear(x, y)
}

/// Solver colour distance.
#[inline]
pub fn d(a: Rgb, b: Rgb) -> f32 {
    dist(a, b)
}

/// L1 vector median of 9 lanes (lowest lane on a tie).
pub fn vmed9(n: &[V2; 9]) -> V2 {
    let mut best = 0;
    let mut bc = f32::INFINITY;
    for a in 0..9 {
        let mut c = 0.0f32;
        for b in 0..9 {
            c += (n[a][0] - n[b][0]).abs();
            c += (n[a][1] - n[b][1]).abs();
        }
        if c < bc {
            bc = c;
            best = a;
        }
    }
    n[best]
}

/// Clamp to [0, 1].
pub fn sat01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Scalar inputs of one solve beyond the frames and fields.
#[derive(Clone, Debug)]
pub struct Gain {
    /// Matcher's accepted prev -> cur luma gain (1, 0 when rejected).
    pub scale: f64,
    /// Offset of the gain.
    pub offset: f64,
}

impl Gain {
    /// The gate let a real gain through.
    pub fn accepted(&self) -> bool {
        (self.scale - 1.0).abs() > 1e-3 || self.offset.abs() > 1e-3
    }
}

/// 0 / 1 plane of a mask.
pub fn mask_to_plane(m: &Grid<bool>) -> Grid<f32> {
    m.map(|b| b as u8 as f32)
}
