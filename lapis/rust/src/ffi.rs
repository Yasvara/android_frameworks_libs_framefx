// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The C ABI lapis/classifier.cpp calls.

use crate::classify::classify;
use crate::gbuffer::{GBufferInput, GBufferOutput, GBufferTracker, Profile};
use crate::glass::{GlassCut, GlassPass, GlassTracker};
use crate::model::*;
use crate::orient::{orientation, Orientation};
use crate::shading::ShadingGovernor;
use crate::signals::{EngineSignals, SignalInput, SignalProfile, SignalState, SignalWatch};
use crate::tracker::Tracker;

/// # Safety
/// `frame` points at a valid Frame whose arrays hold their counts; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_classify(frame: *const Frame, out: *mut Roles) {
    if frame.is_null() || out.is_null() {
        return;
    }
    // SAFETY: per this function's contract.
    unsafe {
        let view = View::new(&*frame);
        out.write(classify(&view));
    }
}

/// # Safety
/// `tracker` points at a framefx::lapis::TrackerState, `roles` at a Roles.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_tracker_update(
    tracker: *mut Tracker,
    roles: *const Roles,
    width: u32,
    height: u32,
    engage: i32,
    grace: i32,
) -> bool {
    // SAFETY: per this function's contract.
    unsafe {
        match (tracker.as_mut(), roles.as_ref()) {
            (Some(t), Some(r)) => t.update(r, width, height, engage, grace),
            _ => false,
        }
    }
}

/// # Safety
/// `tracker` points at a framefx::lapis::TrackerState.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_tracker_reset(tracker: *mut Tracker) {
    // SAFETY: per this function's contract.
    if let Some(t) = unsafe { tracker.as_mut() } {
        t.reset();
    }
}

#[no_mangle]
pub extern "C" fn framefx_lapis_is_depth_format(format: u32) -> bool {
    is_depth_format(format)
}

#[no_mangle]
pub extern "C" fn framefx_lapis_is_depth_like_format(format: u32) -> bool {
    is_depth_like_format(format)
}

#[no_mangle]
pub extern "C" fn framefx_lapis_is_untoned(format: u32) -> bool {
    is_untoned(format)
}

/// # Safety
/// `tracker` points at a framefx::luwes::GBufferState, `input` at a GBufferInput, `out` is
/// writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_gbuffer_update(
    tracker: *mut GBufferTracker,
    input: *const GBufferInput,
    gl: bool,
    out: *mut GBufferOutput,
) {
    // SAFETY: per this function's contract.
    unsafe {
        if let (Some(t), Some(i), false) = (tracker.as_mut(), input.as_ref(), out.is_null()) {
            out.write(t.update(i, gl));
        }
    }
}

/// # Safety
/// `tracker` points at a framefx::luwes::GBufferState.
#[no_mangle]
pub unsafe extern "C" fn framefx_gbuffer_reset(tracker: *mut GBufferTracker) {
    // SAFETY: per this function's contract.
    if let Some(t) = unsafe { tracker.as_mut() } {
        t.reset();
    }
}

/// The profile line for `input` into `out` (NUL-terminated); its length, 0 when it did not fit.
///
/// # Safety
/// `input` points at a GBufferInput; `out` holds `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_gbuffer_profile(
    input: *const GBufferInput,
    out: *mut u8,
    capacity: usize,
) -> usize {
    // SAFETY: per this function's contract.
    let Some(i) = (unsafe { input.as_ref() }) else { return 0 };
    let line = Profile::of(i).to_line();
    if out.is_null() || line.len() + 1 > capacity {
        return 0;
    }
    // SAFETY: `out` holds `capacity` bytes, more than the line and its NUL.
    unsafe {
        core::ptr::copy_nonoverlapping(line.as_ptr(), out, line.len());
        out.add(line.len()).write(0);
    }
    line.len()
}

/// Warms `tracker` from a profile line of an earlier run; false when the line is not one.
///
/// # Safety
/// `tracker` points at a framefx::luwes::GBufferState; `line` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_gbuffer_warm(
    tracker: *mut GBufferTracker,
    line: *const u8,
    len: usize,
) -> bool {
    // SAFETY: per this function's contract.
    unsafe {
        let Some(t) = tracker.as_mut() else { return false };
        if line.is_null() {
            return false;
        }
        let bytes = core::slice::from_raw_parts(line, len);
        match core::str::from_utf8(bytes).ok().and_then(Profile::from_line) {
            Some(p) => {
                t.warm_from(&p);
                true
            }
            None => false,
        }
    }
}

/// # Safety
/// `m` holds 16 floats, `out` room for 16.
#[no_mangle]
pub unsafe extern "C" fn framefx_gbuffer_invert(m: *const [f32; 16], out: *mut [f32; 16]) -> bool {
    // SAFETY: per this function's contract.
    unsafe {
        match (m.as_ref(), out.is_null()) {
            (Some(m), false) => match crate::gbuffer::invert(m) {
                Some(i) => {
                    out.write(i);
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

/// A signal state for one swapchain's frames; free it with framefx_signals_free.
#[no_mangle]
pub extern "C" fn framefx_signals_new() -> *mut SignalState {
    Box::into_raw(Box::default())
}

/// # Safety
/// `state` came from framefx_signals_new, or is null.
#[no_mangle]
pub unsafe extern "C" fn framefx_signals_free(state: *mut SignalState) {
    if !state.is_null() {
        // SAFETY: made by Box::into_raw in framefx_signals_new.
        drop(unsafe { Box::from_raw(state) });
    }
}

/// # Safety
/// `state` came from framefx_signals_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_signals_reset(state: *mut SignalState) {
    // SAFETY: per this function's contract.
    if let Some(s) = unsafe { state.as_mut() } {
        s.reset();
    }
}

/// One presented frame, consumed whole.
///
/// # Safety
/// `state` came from framefx_signals_new; every pointer in `input` holds its count; `out` is
/// writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_signals_update(
    state: *mut SignalState,
    input: *const SignalInput,
    out: *mut EngineSignals,
) {
    // SAFETY: per this function's contract.
    unsafe {
        if let (Some(s), Some(i), false) = (state.as_mut(), input.as_ref(), out.is_null()) {
            out.write(crate::signals::update_from(s, i));
        }
    }
}

/// The addresses to read at each submit until the next present; how many were written.
///
/// # Safety
/// `state` came from framefx_signals_new; `out` holds `capacity` entries.
#[no_mangle]
pub unsafe extern "C" fn framefx_signals_watch(
    state: *const SignalState,
    out: *mut SignalWatch,
    capacity: usize,
) -> usize {
    // SAFETY: per this function's contract.
    unsafe {
        let Some(s) = state.as_ref() else { return 0 };
        if out.is_null() {
            return 0;
        }
        let w = s.watch();
        let n = w.len().min(capacity);
        core::ptr::copy_nonoverlapping(w.as_ptr(), out, n);
        n
    }
}

/// What was learned, as a profile line into `out` (NUL-terminated); its length, 0 when it did not
/// fit or nothing is learned yet.
///
/// # Safety
/// `state` came from framefx_signals_new; `out` holds `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_signals_profile(state: *const SignalState, out: *mut u8, capacity: usize) -> usize {
    // SAFETY: per this function's contract.
    unsafe {
        let Some(s) = state.as_ref() else { return 0 };
        let p = s.learned();
        if p.signature == 0 || p.camera.is_none() {
            return 0;
        }
        let line = p.to_line();
        if out.is_null() || line.len() + 1 > capacity {
            return 0;
        }
        core::ptr::copy_nonoverlapping(line.as_ptr(), out, line.len());
        out.add(line.len()).write(0);
        line.len()
    }
}

/// Starts `state` from a profile line of an earlier run; false when the line is not one.
///
/// # Safety
/// `state` came from framefx_signals_new; `line` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_signals_warm(state: *mut SignalState, line: *const u8, len: usize) -> bool {
    // SAFETY: per this function's contract.
    unsafe {
        let Some(s) = state.as_mut() else { return false };
        if line.is_null() {
            return false;
        }
        match core::str::from_utf8(core::slice::from_raw_parts(line, len)).ok().and_then(SignalProfile::from_line) {
            Some(p) => {
                s.warm(p);
                true
            }
            None => false,
        }
    }
}

/// The velocity encoding that explains `count` texels sampled against the camera's motion there.
///
/// # Safety
/// `samples` holds `count` entries; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_signals_fit_velocity(
    samples: *const crate::signals::velocity::VelocitySample,
    count: usize,
    out: *mut crate::signals::velocity::VelocityEncoding,
) {
    if samples.is_null() || out.is_null() {
        return;
    }
    // SAFETY: per this function's contract.
    unsafe { out.write(crate::signals::velocity::fit(core::slice::from_raw_parts(samples, count))) }
}

/// # Safety
/// `governor` points at a framefx::lapis::ShadingGovernor.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_shading_frame(
    governor: *mut ShadingGovernor,
    now_ns: i64,
    refresh_ns: i64,
    max_level: u8,
) -> u8 {
    // SAFETY: per this function's contract.
    match unsafe { governor.as_mut() } {
        Some(g) => g.frame(now_ns, refresh_ns, max_level),
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn framefx_lapis_shading_governor_size() -> usize {
    core::mem::size_of::<ShadingGovernor>()
}

/// Which way `from`, as it stands after step `after` (negative: from the start), reaches the
/// presented image: the flips | 0x100 when certain, | 0x200 when only hinted, or -1.
///
/// # Safety
/// `frame` points at a valid Frame whose arrays hold their counts.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_orientation(frame: *const Frame, from: u32, after: i32) -> i32 {
    if frame.is_null() {
        return -1;
    }
    // SAFETY: per this function's contract.
    let view = unsafe { View::new(&*frame) };
    let after = usize::try_from(after).ok();
    match orientation(&view, from, after) {
        Orientation::Exact(f) => 0x100 | f as i32,
        Orientation::Hint(f) => 0x200 | f as i32,
        Orientation::Unknown => -1,
    }
}

/// The transparency cut of one frame, through `tracker` across frames.
///
/// # Safety
/// `frame` points at a valid Frame; `passes` holds one GlassPass per step; `tracker` points at a
/// framefx::lapis::GlassTrackerState; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_glass(
    frame: *const Frame,
    passes: *const GlassPass,
    depth: u32,
    tracker: *mut GlassTracker,
    out: *mut GlassCut,
) {
    if frame.is_null() || tracker.is_null() || out.is_null() {
        return;
    }
    // SAFETY: per this function's contract.
    unsafe {
        let view = View::new(&*frame);
        let passes = if passes.is_null() {
            &[][..]
        } else {
            core::slice::from_raw_parts(passes, view.steps.len())
        };
        let cut = crate::glass::find(&view, passes, depth);
        out.write((*tracker).update(cut));
    }
}

#[no_mangle]
pub extern "C" fn framefx_lapis_glass_tracker_size() -> usize {
    core::mem::size_of::<GlassTracker>()
}

/// Whether `code` is a module with one entry point, a vertex shader.
///
/// # Safety
/// `code` points at `words` readable words.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_xfb_is_vertex(code: *const u32, words: usize) -> bool {
    if code.is_null() {
        return false;
    }
    // SAFETY: per this function's contract.
    crate::xfb::is_vertex(unsafe { core::slice::from_raw_parts(code, words) })
}

/// The capturing copy of a vertex module (lapis/rust/src/xfb.rs) into `out`: the words written,
/// or 0 with the refusal in `why` when the module keeps its code, or when `capacity` is short.
/// `varying` the location of an output captured beside the position, `u32::MAX` none.
///
/// # Safety
/// `code` points at `words` readable words, `out` at `capacity` writable ones, `why` at one byte.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_xfb_patch(
    code: *const u32,
    words: usize,
    out: *mut u32,
    capacity: usize,
    varying: u32,
    why: *mut u8,
) -> usize {
    if code.is_null() || out.is_null() || why.is_null() {
        return 0;
    }
    // SAFETY: per this function's contract.
    let input = unsafe { core::slice::from_raw_parts(code, words) };
    let mut patched = Vec::new();
    let varying = (varying != u32::MAX).then_some(varying);
    match crate::xfb::patch_with(input, &mut patched, varying) {
        Ok(()) if patched.len() <= capacity => {
            // SAFETY: `out` holds `capacity` words, at least `patched.len()`.
            unsafe { core::ptr::copy_nonoverlapping(patched.as_ptr(), out, patched.len()) };
            patched.len()
        }
        Ok(()) => 0,
        Err(r) => {
            // SAFETY: per this function's contract.
            unsafe { why.write(r as u8) };
            0
        }
    }
}

/// Whether a fragment module can drop its fragments (lapis/rust/src/xfb.rs).
///
/// # Safety
/// `code` points at `words` readable words.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_xfb_discards(code: *const u32, words: usize) -> bool {
    if code.is_null() {
        return true;
    }
    // SAFETY: per this function's contract.
    crate::xfb::discards(unsafe { core::slice::from_raw_parts(code, words) })
}

/// The ray-traced scene's cache (lapis/rust/src/scene.rs) with arenas of `tree_bytes` and
/// `point_bytes` for the trees and their points; freed with framefx_lapis_scene_free.
#[no_mangle]
pub extern "C" fn framefx_lapis_scene_new(tree_bytes: u64, point_bytes: u64) -> *mut SceneState {
    Box::into_raw(Box::new(SceneState {
        cache: crate::scene::SceneCache::default(),
        trees: crate::scene::Arena::new(tree_bytes),
        points: crate::scene::Arena::new(point_bytes),
        evicted: Vec::new(),
    }))
}

pub struct SceneState {
    cache: crate::scene::SceneCache,
    trees: crate::scene::Arena,
    points: crate::scene::Arena,
    evicted: Vec<u32>,
}

/// # Safety
/// `s` from framefx_lapis_scene_new, or null; not used after.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_free(s: *mut SceneState) {
    if !s.is_null() {
        // SAFETY: per this function's contract.
        drop(unsafe { Box::from_raw(s) });
    }
}

/// # Safety
/// `s` from framefx_lapis_scene_new.
unsafe fn scene<'a>(s: *mut SceneState) -> &'a mut SceneState {
    // SAFETY: per the callers' contracts.
    unsafe { &mut *s }
}

/// A new frame.
///
/// # Safety
/// `s` from framefx_lapis_scene_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_begin(s: *mut SceneState) {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    unsafe { scene(s) }.cache.begin();
}

/// A draw of the frame: its entry into `entry`, and KEEP, BUILD or REFIT.
///
/// # Safety
/// `s` from framefx_lapis_scene_new; `entry` writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_place(
    s: *mut SceneState,
    key: u64,
    vertices: u32,
    entry: *mut u32,
) -> u8 {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    let (e, action) = unsafe { scene(s) }.cache.place(key, vertices);
    // SAFETY: per this function's contract.
    unsafe { *entry = e };
    action
}

/// The GPU saw `entry`'s points move; `far` nonzero past what a refit keeps tight.
///
/// # Safety
/// `s` from framefx_lapis_scene_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_moved(s: *mut SceneState, entry: u32, far: u8) {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    unsafe { scene(s) }.cache.moved(entry, far != 0);
}

/// `entry`'s tree could not be made: it goes from the cache.
///
/// # Safety
/// `s` from framefx_lapis_scene_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_drop(s: *mut SceneState, entry: u32) {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    unsafe { scene(s) }.cache.drop_entry(entry);
}

/// Entries not drawn for more than `age` frames, into `out` (at most `capacity`, the rest on the
/// next call); their count. The caller frees their trees.
///
/// # Safety
/// `s` from framefx_lapis_scene_new; `out` writable for `capacity`.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_evict(
    s: *mut SceneState,
    age: u64,
    out: *mut u32,
    capacity: usize,
) -> usize {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    let st = unsafe { scene(s) };
    if st.evicted.is_empty() {
        st.cache.evict(age, &mut st.evicted);
    }
    let n = st.evicted.len().min(capacity);
    for (i, e) in st.evicted.drain(..n).enumerate() {
        // SAFETY: i < capacity, per this function's contract.
        unsafe { *out.add(i) = e };
    }
    n
}

/// Entries drawn within `age` frames but not this one and still (scene.rs `retained`), into `out`
/// (at most `capacity`); their count.
///
/// # Safety
/// `s` from framefx_lapis_scene_new; `out` writable for `capacity`.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_retained(
    s: *mut SceneState,
    age: u64,
    out: *mut u32,
    capacity: usize,
) -> usize {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    let st = unsafe { scene(s) };
    let mut kept = Vec::new();
    st.cache.retained(age, &mut kept);
    let n = kept.len().min(capacity);
    for (i, e) in kept.into_iter().take(n).enumerate() {
        // SAFETY: i < capacity, per this function's contract.
        unsafe { *out.add(i) = e };
    }
    n
}

/// Draws out of sight left out as moving since the last call, then zero.
///
/// # Safety
/// `s` from framefx_lapis_scene_new, not freed.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_take_moving_out_of_sight(s: *mut SceneState) -> u64 {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    let st = unsafe { scene(s) };
    std::mem::take(&mut st.cache.moving_out_of_sight)
}

/// Space in arena `which` (0 trees, 1 points): its offset, or u64::MAX.
///
/// # Safety
/// `s` from framefx_lapis_scene_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_alloc(s: *mut SceneState, which: u32, bytes: u64) -> u64 {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    let st = unsafe { scene(s) };
    if which == 0 { st.trees.alloc(bytes) } else { st.points.alloc(bytes) }
}

/// # Safety
/// `s` from framefx_lapis_scene_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_release(s: *mut SceneState, which: u32, offset: u64) {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    let st = unsafe { scene(s) };
    if which == 0 { st.trees.release(offset) } else { st.points.release(offset) }
}

/// Bytes in use in arena `which`, for the log.
///
/// # Safety
/// `s` from framefx_lapis_scene_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_scene_in_use(s: *mut SceneState, which: u32) -> u64 {
    // SAFETY: `s` from framefx_lapis_scene_new, per this function's contract.
    let st = unsafe { scene(s) };
    if which == 0 { st.trees.in_use() } else { st.points.in_use() }
}

/// The G-buffer roles' vote (lapis/rust/src/roles.rs); freed with framefx_lapis_roles_free.
#[no_mangle]
pub extern "C" fn framefx_lapis_roles_new() -> *mut crate::roles::Roles {
    Box::into_raw(Box::default())
}

/// # Safety
/// `r` from framefx_lapis_roles_new, or null; not used after.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_roles_free(r: *mut crate::roles::Roles) {
    if !r.is_null() {
        // SAFETY: per this function's contract.
        drop(unsafe { Box::from_raw(r) });
    }
}

/// One frame's measures of `count` targets, then the frame's end.
///
/// # Safety
/// `r` from framefx_lapis_roles_new; `samples` holds `count` of them.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_roles_frame(
    r: *mut crate::roles::Roles,
    samples: *const crate::roles::Sample,
    count: usize,
) {
    // SAFETY: per this function's contract.
    let (r, samples) = unsafe { (&mut *r, core::slice::from_raw_parts(samples, count)) };
    for s in samples {
        r.add(s);
    }
    r.settle();
}

/// The picks: albedo image (0 none) and its agreement; normal image (0 none), its decoding and
/// agreement.
///
/// # Safety
/// `r` from framefx_lapis_roles_new; the outputs writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_roles_picks(
    r: *const crate::roles::Roles,
    albedo: *mut u32,
    albedo_agreement: *mut f64,
    normal: *mut u32,
    decoding: *mut u8,
    normal_agreement: *mut f64,
) {
    // SAFETY: per this function's contract.
    let r = unsafe { &*r };
    let (a, aa) = r.albedo();
    let (n, d, na) = r.normal();
    // SAFETY: per this function's contract.
    unsafe {
        *albedo = a;
        *albedo_agreement = aa;
        *normal = n;
        *decoding = d;
        *normal_agreement = na;
    }
}

/// The alpha test of fragment shader `code` (lapis/rust/src/alpha.rs) into `out`, and its
/// program into `program` (free with framefx_lapis_alpha_free): 0 read, else why not
/// (alpha::Unread).
///
/// # Safety
/// `code` points at `words` readable words, `out` at one AlphaTest, `program` at one pointer.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_alpha_read(
    code: *const u32,
    words: usize,
    out: *mut crate::alpha::AlphaTest,
    program: *mut *mut crate::alpha::Program,
) -> u8 {
    if code.is_null() || out.is_null() || program.is_null() {
        return crate::alpha::Unread::Malformed as u8;
    }
    // SAFETY: per this function's contract.
    let code = unsafe { core::slice::from_raw_parts(code, words) };
    match crate::alpha::read(code) {
        Ok((t, p)) => {
            // SAFETY: per this function's contract.
            unsafe {
                out.write(t);
                program.write(Box::into_raw(Box::new(p)));
            }
            0
        }
        Err(e) => e as u8,
    }
}

/// # Safety
/// `p` null or from framefx_lapis_alpha_read, not used after.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_alpha_free(p: *mut crate::alpha::Program) {
    if !p.is_null() {
        // SAFETY: per this function's contract, made by Box::into_raw.
        drop(unsafe { Box::from_raw(p) });
    }
}

/// The uniform blocks program `p` reads, up to `capacity` into `out`; how many it reads.
///
/// # Safety
/// `p` from framefx_lapis_alpha_read; `out` holds `capacity` Blocks.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_alpha_blocks(
    p: *const crate::alpha::Program,
    out: *mut crate::alpha::Block,
    capacity: usize,
) -> usize {
    if p.is_null() {
        return 0;
    }
    // SAFETY: per this function's contract.
    let blocks = unsafe { &(*p).blocks };
    if !out.is_null() {
        for (i, b) in blocks.iter().take(capacity).enumerate() {
            // SAFETY: `out` holds `capacity` Blocks, i below it.
            unsafe { out.add(i).write(*b) };
        }
    }
    blocks.len()
}

/// Where each of program `p`'s parts samples, up to `capacity` into `out`; how many parts.
///
/// # Safety
/// `p` from framefx_lapis_alpha_read; `out` null or holding `capacity` AlphaTests.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_alpha_parts(
    p: *const crate::alpha::Program,
    out: *mut crate::alpha::AlphaTest,
    capacity: usize,
) -> usize {
    if p.is_null() {
        return 0;
    }
    // SAFETY: per this function's contract.
    let program = unsafe { &*p };
    if !out.is_null() {
        for (i, t) in program.tests().take(capacity).enumerate() {
            // SAFETY: `out` holds `capacity` AlphaTests, i below it.
            unsafe { out.add(i).write(*t) };
        }
    }
    program.tests().count()
}

/// Program `p` over one draw's uniform blocks: `data[i]` (null unknown) of `sizes[i]` bytes for
/// its i-th block. 0 it cuts where `ab[0]·t + ab[1] < 0`, `t` sampled as part `*part` says; 1
/// its surface is whole; 2 unknown.
///
/// # Safety
/// `p` from framefx_lapis_alpha_read; `data` and `sizes` hold `n` entries, each non-null datum
/// that many readable bytes; `ab` two writable floats, `part` one.
#[no_mangle]
pub unsafe extern "C" fn framefx_lapis_alpha_eval(
    p: *const crate::alpha::Program,
    data: *const *const u8,
    sizes: *const usize,
    n: usize,
    ab: *mut f32,
    part: *mut u32,
) -> u8 {
    if p.is_null() || ab.is_null() || part.is_null() || (n > 0 && (data.is_null() || sizes.is_null())) {
        return 2;
    }
    let mut blocks: Vec<Option<&[u8]>> = Vec::with_capacity(n);
    for i in 0..n {
        // SAFETY: per this function's contract.
        let (d, size) = unsafe { (*data.add(i), *sizes.add(i)) };
        // SAFETY: a non-null datum holds `size` bytes, per this function's contract.
        blocks.push((!d.is_null()).then(|| unsafe { core::slice::from_raw_parts(d, size) }));
    }
    // SAFETY: per this function's contract.
    match unsafe { (*p).eval(&blocks) } {
        crate::alpha::Verdict::Cuts { part: which, a, b } => {
            // SAFETY: `ab` holds two floats, `part` one.
            unsafe {
                ab.write(a);
                ab.add(1).write(b);
                part.write(which as u32);
            }
            0
        }
        crate::alpha::Verdict::Whole => 1,
        crate::alpha::Verdict::Unknown => 2,
    }
}
