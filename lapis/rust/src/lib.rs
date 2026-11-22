// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Lapis's frame classifier and tracker, behind the C ABI in `ffi`. The engine flattens each
//! frame it recorded into the arrays of `model`; nothing here allocates per call beyond the
//! classifier's own scratch, and nothing panics on malformed input.

mod alpha;
mod classify;
mod ffi;
mod gbuffer;
mod glass;
mod hud;
mod model;
mod orient;
mod roles;
mod scene;
mod shading;
mod tracker;
mod xfb;
mod signals;
