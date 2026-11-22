// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The C ABI kelir/*.cpp call. Every struct here is #[repr(C)] and mirrored, with its size checked,
//! in the C++ headers. Pointers come from C++ and are valid for the call.

use core::slice;

use crate::colour::{self, Rgb, Step};
use crate::controller::{self, Controller, Update};
use crate::look;
use crate::lut;
use crate::params::{self, KindInfo, Param};
use crate::plan::{self, FrameInfo, PlanCore, Program, Stages, Target};
use crate::rays::{RayBudget, RayMesh, RayPlan, RayScene};
use crate::section::{self, Stack};

/// # Safety
/// `ptr` holds `len` elements, or `len` is 0.
unsafe fn parts<'a, T>(ptr: *const T, len: i32) -> &'a [T] {
    if ptr.is_null() || len <= 0 {
        &[]
    } else {
        // SAFETY: per this function's contract.
        unsafe { slice::from_raw_parts(ptr, len as usize) }
    }
}

#[no_mangle]
pub extern "C" fn framefx_kelir_info(kind: i32) -> *const KindInfo {
    if params::known(kind) { &params::KINDS[kind as usize] } else { core::ptr::null() }
}

/// # Safety
/// `p` points at a Param.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_code_valid(p: *const Param, q: i16) -> bool {
    // SAFETY: per this function's contract.
    params::code_valid(unsafe { &*p }, q)
}

/// # Safety
/// `p` points at a Param.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_param_decode(p: *const Param, q: i16) -> f32 {
    // SAFETY: per this function's contract.
    params::decode(unsafe { &*p }, q)
}

/// # Safety
/// `p` points at a Param.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_param_encode(p: *const Param, value: f32) -> i16 {
    // SAFETY: per this function's contract.
    params::encode(unsafe { &*p }, value)
}

/// The reject reason; `out` is written only when it is 0.
///
/// # Safety
/// `data` holds `len` bytes (or is null); `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_decode_section(data: *const u8, len: usize, out: *mut Stack) -> u8 {
    if data.is_null() {
        return section::Reject::Size as u8;
    }
    // SAFETY: per this function's contract.
    let d = unsafe { slice::from_raw_parts(data, len) };
    match section::decode_section(d) {
        Ok(s) => {
            // SAFETY: per this function's contract.
            unsafe { out.write(s) };
            0
        }
        Err(r) => r as u8,
    }
}

/// # Safety
/// `stack` points at a Stack; `out` has room for 16 + 20 * count bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_encode_section(stack: *const Stack, out: *mut u8) -> usize {
    // SAFETY: per this function's contract.
    let out = unsafe { slice::from_raw_parts_mut(out, section::MAX_SECTION_SIZE) };
    // SAFETY: per this function's contract.
    let stack = unsafe { &*stack };
    section::encode_section(stack, out)
}

#[no_mangle]
pub extern "C" fn framefx_kelir_luma_weights(primaries: u8) -> *const f32 {
    colour::luma_weights(primaries).as_ptr()
}

#[no_mangle]
pub extern "C" fn framefx_kelir_srgb_decode(encoded: f32) -> f32 {
    colour::decode(encoded)
}

#[no_mangle]
pub extern "C" fn framefx_kelir_srgb_encode(linear: f32) -> f32 {
    colour::encode(linear)
}

/// # Safety
/// `step` points at a Step.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_apply_colour(step: *const Step, x: Rgb, primaries: u8) -> Rgb {
    // SAFETY: per this function's contract.
    colour::apply_colour(unsafe { &*step }, x, primaries)
}

/// # Safety
/// `step` points at a Step.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_blend(step: *const Step, x: Rgb, primaries: u8) -> Rgb {
    // SAFETY: per this function's contract.
    colour::blend(unsafe { &*step }, x, primaries)
}

/// # Safety
/// `step` points at a Step.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_is_identity_colour(step: *const Step) -> bool {
    // SAFETY: per this function's contract.
    colour::is_identity_colour(unsafe { &*step })
}

/// # Safety
/// `steps` holds `count` Steps.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_evaluate(steps: *const Step, count: i32, primaries: u8, x: Rgb) -> Rgb {
    // SAFETY: per this function's contract.
    lut::evaluate_colour(unsafe { parts(steps, count) }, primaries, x)
}

/// Bakes into LUT_TEXELS x 4 halves at `out`; true when they hold linear light.
///
/// # Safety
/// `steps` holds `count` Steps; `out` holds 33^3 * 4 u16.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_bake_lut(steps: *const Step, count: i32, primaries: u8, out: *mut u16) -> bool {
    // SAFETY: per this function's contract.
    let out = unsafe { slice::from_raw_parts_mut(out, lut::LUT_TEXELS * 4) };
    // SAFETY: per this function's contract.
    let steps = unsafe { parts(steps, count) };
    lut::bake(steps, primaries, out)
}

/// # Safety
/// `texels` holds 33^3 * 4 u16.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_sample_lut(texels: *const u16, linear: bool, x: Rgb) -> Rgb {
    // SAFETY: per this function's contract.
    lut::sample(unsafe { slice::from_raw_parts(texels, lut::LUT_TEXELS * 4) }, linear, x)
}

#[no_mangle]
pub extern "C" fn framefx_kelir_to_half(v: f32) -> u16 {
    lut::to_half(v)
}

#[no_mangle]
pub extern "C" fn framefx_kelir_from_half(h: u16) -> f32 {
    lut::from_half(h)
}

/// # Safety
/// `steps` holds `count` Steps; `out` holds 24 bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_swatch(steps: *const Step, count: i32, primaries: u8, out: *mut u8) {
    // SAFETY: per this function's contract.
    let out = unsafe { &mut *(out as *mut [u8; lut::SWATCH_COLOURS * 3]) };
    // SAFETY: per this function's contract.
    let steps = unsafe { parts(steps, count) };
    lut::swatch(steps, primaries, out);
}

/// # Safety
/// `step` points at a Step; `out` holds 17 bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_curve(step: *const Step, primaries: u8, out: *mut u8) {
    // SAFETY: per this function's contract.
    let out = unsafe { &mut *(out as *mut [u8; lut::CURVE_POINTS]) };
    // SAFETY: per this function's contract.
    let step = unsafe { &*step };
    lut::curve(step, primaries, out);
}

/// # Safety
/// `stack` points at a validated Stack; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_compile(stack: *const Stack, out: *mut PlanCore) {
    // SAFETY: per this function's contract.
    unsafe { out.write(plan::compile(&*stack)) };
}

/// # Safety
/// `plan` points at a PlanCore.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_plan_has(plan: *const PlanCore, kind: u8) -> bool {
    // SAFETY: per this function's contract.
    unsafe { &*plan }.has(kind)
}

/// The scene step's index for `kind`, or -1.
///
/// # Safety
/// `plan` points at a PlanCore.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_scene_step(plan: *const PlanCore, kind: u8) -> i32 {
    // SAFETY: per this function's contract.
    let plan = unsafe { &*plan };
    plan.scene().iter().position(|s| s.kind == kind).map_or(-1, |i| i as i32)
}

/// # Safety
/// `plan` points at a PlanCore.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_needs_pyramid(plan: *const PlanCore) -> bool {
    // SAFETY: per this function's contract.
    unsafe { &*plan }.needs_pyramid()
}

/// # Safety
/// `plan` points at a PlanCore.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_needs_source(plan: *const PlanCore) -> bool {
    // SAFETY: per this function's contract.
    unsafe { &*plan }.needs_source()
}

/// # Safety
/// Both point at PlanCores.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_same_colour_band(a: *const PlanCore, b: *const PlanCore) -> bool {
    // SAFETY: per this function's contract.
    let (a, b) = unsafe { (&*a, &*b) };
    a.same_colour_band(b)
}

/// # Safety
/// `plan` points at a PlanCore, `target` at a Target.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_signature(plan: *const PlanCore, target: *const Target) -> u64 {
    // SAFETY: per this function's contract.
    let (plan, target) = unsafe { (&*plan, &*target) };
    plan::signature(plan, target)
}

/// # Safety
/// `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_unpack(signature: u64, out: *mut Program) {
    // SAFETY: per this function's contract.
    unsafe { out.write(plan::unpack(signature)) };
}

/// # Safety
/// `plan` points at a PlanCore; `out` holds 64 floats.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_pack_plan(plan: *const PlanCore, out: *mut f32) {
    // SAFETY: per this function's contract.
    let (plan, out) = unsafe { (&*plan, &mut *(out as *mut [f32; plan::PLAN_VEC4S * 4])) };
    plan::pack_plan(plan, out);
}

/// # Safety
/// `plan` points at a PlanCore; `out` holds 24 floats.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_pack_scene(plan: *const PlanCore, out: *mut f32) {
    // SAFETY: per this function's contract.
    let (plan, out) = unsafe { (&*plan, &mut *(out as *mut [f32; plan::SCENE_VEC4S * 4])) };
    plan::pack_scene(plan, out);
}

/// # Safety
/// `plan` points at a PlanCore; `out` holds 24 floats.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_pack_scene_look(plan: *const PlanCore, out: *mut f32) -> bool {
    // SAFETY: per this function's contract.
    let (plan, out) = unsafe { (&*plan, &mut *(out as *mut [f32; look::LOOK_VEC4S * 4])) };
    look::pack_scene_look(plan, out);
    look::wants_look(plan)
}

/// # Safety
/// `plan` points at a PlanCore; `out` holds 16 floats.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_pack_scene_rays(plan: *const PlanCore, out: *mut f32) -> bool {
    // SAFETY: per this function's contract.
    let (plan, out) = unsafe { (&*plan, &mut *(out as *mut [f32; look::RAYS_VEC4S * 4])) };
    look::pack_scene_rays(plan, out);
    look::wants_rays(plan)
}

/// # Safety
/// `plan` points at a PlanCore, `frame` at a FrameInfo; `out` holds 16 floats.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_pack_frame(plan: *const PlanCore, frame: *const FrameInfo, out: *mut f32) {
    // SAFETY: per this function's contract.
    let (plan, frame, out) = unsafe { (&*plan, &*frame, &mut *(out as *mut [f32; plan::FRAME_VEC4S * 4])) };
    plan::pack_frame(plan, frame, out);
}

/// # Safety
/// `plan` points at a PlanCore or is null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_schedule(plan: *const PlanCore, generation: bool, real: bool, out: *mut Stages) {
    // SAFETY: per this function's contract.
    unsafe { out.write(plan::schedule(plan.as_ref(), generation, real)) };
}

/// # Safety
/// `c` points at a Controller; `data` holds `len` bytes, or is null for no section.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_controller_update(c: *mut Controller, data: *const u8, len: usize, plan_empty: bool) -> Update {
    let section = if data.is_null() {
        None
    } else {
        // SAFETY: per this function's contract.
        Some(unsafe { slice::from_raw_parts(data, len) })
    };
    // SAFETY: per this function's contract.
    let c = unsafe { &mut *c };
    controller::update(c, section, plan_empty)
}

/// # Safety
/// `out` holds 8 slots.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_sizes(out: *mut usize) {
    // SAFETY: per this function's contract.
    let out = unsafe { slice::from_raw_parts_mut(out, 8) };
    out[0] = core::mem::size_of::<KindInfo>();
    out[1] = core::mem::size_of::<Stack>();
    out[2] = core::mem::size_of::<Step>();
    out[3] = core::mem::size_of::<PlanCore>();
    out[4] = core::mem::size_of::<Program>();
    out[5] = core::mem::size_of::<FrameInfo>();
    out[6] = core::mem::size_of::<Controller>();
    out[7] = core::mem::size_of::<Target>();
}

#[no_mangle]
pub extern "C" fn framefx_kelir_ray_scene_new() -> *mut RayScene {
    Box::into_raw(Box::default())
}

/// # Safety
/// `scene` came from framefx_kelir_ray_scene_new and is not used after.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_ray_scene_free(scene: *mut RayScene) {
    if !scene.is_null() {
        // SAFETY: per this function's contract.
        drop(unsafe { Box::from_raw(scene) });
    }
}

/// One frame's builds, refits, instances and evictions for the meshes drawn.
///
/// # Safety
/// `scene` is live; `meshes` holds `count`; `budget` and `out` are valid.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_ray_plan(
    scene: *mut RayScene,
    meshes: *const RayMesh,
    count: i32,
    budget: *const RayBudget,
    out: *mut RayPlan,
) {
    // SAFETY: per this function's contract.
    let (scene, meshes, budget) = unsafe { (&mut *scene, parts(meshes, count), &*budget) };
    let plan = scene.plan(meshes, budget);
    // SAFETY: per this function's contract.
    unsafe { out.write(plan) };
}

/// # Safety
/// `scene` is live.
#[no_mangle]
pub unsafe extern "C" fn framefx_kelir_ray_failed(scene: *mut RayScene, id: u64) {
    // SAFETY: per this function's contract.
    unsafe { &mut *scene }.failed(id);
}
