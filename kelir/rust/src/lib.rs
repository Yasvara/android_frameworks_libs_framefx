// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Kelir's logic: the kinds table, the section codec, the colour band and its LUT bake, plan
//! compilation and packing, and the controller, behind the C ABI in `ffi`. The C++ side records
//! passes and owns GPU objects; it calls in here once per change or per frame, never per draw.

mod colour;
mod controller;
mod ffi;
mod look;
mod lut;
mod params;
mod plan;
mod rays;
mod section;
