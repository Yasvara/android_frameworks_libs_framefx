// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Luwes colour-only frame generation: block matcher, residual-layer field, the block-warp solver
//! and the dense-flow solver, and the fusion stage that blends them (net slot, block-warp fallback).
//!
//! Arithmetic follows the Python reference in lab/luwes (float32 where numpy keeps float32, float64
//! where numpy promotes), so the golden tests hold it bit-exact apart from numpy's own SIMD expf.

// Index loops mirror the reference's array maths; negated float comparisons are NaN-aware.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord, clippy::too_many_arguments, clippy::manual_is_multiple_of)]

pub(crate) mod cv;
pub(crate) mod fft;
pub mod fusion;
pub mod grid;
pub mod layer;
pub mod motion;
pub mod npy;
pub mod pipeline;
pub mod solver;
