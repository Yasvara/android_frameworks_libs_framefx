// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The C ABI lapis/hud.cpp calls, once per present.

use core::slice;

use super::features::{FEATURES, FEATURE_NAMES};
use super::plan::TABLE_SIZE;
use super::*;

unsafe fn parts<'a, T>(data: *const T, len: u32) -> &'a [T] {
    if data.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: the caller guarantees `len` valid elements at `data` for 'a.
        unsafe { slice::from_raw_parts(data, len as usize) }
    }
}

unsafe fn text<'a>(line: *const u8, len: usize) -> Option<&'a str> {
    if line.is_null() {
        return None;
    }
    // SAFETY: the caller guarantees `len` bytes at `line`.
    core::str::from_utf8(unsafe { slice::from_raw_parts(line, len) }).ok()
}

fn copy_line(line: &str, out: *mut u8, capacity: usize) -> usize {
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

/// A labeler for one swapchain's frames; free it with framefx_hud_free.
#[no_mangle]
pub extern "C" fn framefx_hud_new() -> *mut HudState {
    Box::into_raw(Box::default())
}

/// # Safety
/// `state` came from framefx_hud_new, or is null.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_free(state: *mut HudState) {
    if !state.is_null() {
        // SAFETY: made by Box::into_raw in framefx_hud_new.
        drop(unsafe { Box::from_raw(state) });
    }
}

/// # Safety
/// `state` came from framefx_hud_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_reset(state: *mut HudState) {
    // SAFETY: per this function's contract.
    if let Some(s) = unsafe { state.as_mut() } {
        s.reset();
    }
}

/// One presented frame, consumed whole.
///
/// # Safety
/// `state` came from framefx_hud_new; every pointer in `input` holds its count; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_update(state: *mut HudState, input: *const HudInput, out: *mut HudOutput) {
    // SAFETY: per this function's contract.
    unsafe {
        let (Some(s), Some(i), false) = (state.as_mut(), input.as_ref(), out.is_null()) else { return };
        let grid = (i.flags & INPUT_GRID != 0).then(|| Grid {
            luma: parts(i.grid, i.grid_width.saturating_mul(i.grid_height)),
            width: i.grid_width,
            height: i.grid_height,
            seq: i.grid_seq,
            camera: (i.flags & INPUT_GRID_CAMERA != 0).then_some(i.grid_view_projection),
        });
        let f = HudFrame {
            view: View::new(&i.frame),
            passes: parts(i.passes, i.pass_count),
            draws: parts(i.draws, i.draw_count),
            blocks: parts(i.blocks, i.block_count),
            floats: parts(i.floats, i.float_count),
            roles: i.roles,
            hide_hud: i.flags & INPUT_HIDE_HUD != 0,
            skipped: i.skipped,
            grid,
        };
        out.write(s.update(&f));
    }
}

/// Copies the skip table into `out`; the entries written (TABLE_SIZE, or 0 when it does not fit).
///
/// # Safety
/// `state` came from framefx_hud_new; `out` holds `capacity` entries.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_table(state: *const HudState, out: *mut HudEntry, capacity: usize) -> usize {
    // SAFETY: per this function's contract.
    unsafe {
        let Some(s) = state.as_ref() else { return 0 };
        if out.is_null() || capacity < TABLE_SIZE {
            return 0;
        }
        core::ptr::copy_nonoverlapping(s.table().as_ptr(), out, TABLE_SIZE);
        TABLE_SIZE
    }
}

/// The last frame's label, confidence and structural label per draw; any output may be null.
/// How many draws were written.
///
/// # Safety
/// `state` came from framefx_hud_new; each non-null output holds `capacity` entries.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_labels(
    state: *const HudState,
    labels: *mut u8,
    confidence: *mut f32,
    weak: *mut u8,
    capacity: usize,
) -> usize {
    // SAFETY: per this function's contract.
    unsafe {
        let Some(s) = state.as_ref() else { return 0 };
        let n = s.labels().len().min(capacity);
        if !labels.is_null() {
            core::ptr::copy_nonoverlapping(s.labels().as_ptr(), labels, n);
        }
        if !confidence.is_null() {
            core::ptr::copy_nonoverlapping(s.confidences().as_ptr(), confidence, n);
        }
        if !weak.is_null() {
            core::ptr::copy_nonoverlapping(s.weak_labels().as_ptr(), weak, n);
        }
        n
    }
}

/// The last frame's feature rows, FEATURES floats per draw; how many floats were written.
///
/// # Safety
/// `state` came from framefx_hud_new; `out` holds `capacity` floats.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_features(state: *const HudState, out: *mut f32, capacity: usize) -> usize {
    // SAFETY: per this function's contract.
    unsafe {
        let Some(s) = state.as_ref() else { return 0 };
        if out.is_null() {
            return 0;
        }
        let rows = s.feature_rows();
        let n = (capacity / FEATURES * FEATURES).min(rows.len());
        core::ptr::copy_nonoverlapping(rows.as_ptr(), out, n);
        n
    }
}

#[no_mangle]
pub extern "C" fn framefx_hud_feature_count() -> usize {
    FEATURES
}

/// Feature `index`'s name into `out` (NUL-terminated); its length, 0 when there is none or it did
/// not fit.
#[no_mangle]
pub extern "C" fn framefx_hud_feature_name(index: usize, out: *mut u8, capacity: usize) -> usize {
    FEATURE_NAMES.get(index).map_or(0, |name| copy_line(name, out, capacity))
}

/// What was learned of the game's shaders, as a profile line into `out` (NUL-terminated); its
/// length, 0 when it did not fit.
///
/// # Safety
/// `state` came from framefx_hud_new; `out` holds `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_profile(state: *const HudState, out: *mut u8, capacity: usize) -> usize {
    // SAFETY: per this function's contract.
    match unsafe { state.as_ref() } {
        Some(s) => copy_line(&s.profile(), out, capacity),
        None => 0,
    }
}

/// The profile was written: nothing new to write until more is learned.
///
/// # Safety
/// `state` came from framefx_hud_new.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_profile_written(state: *mut HudState) {
    // SAFETY: per this function's contract.
    if let Some(s) = unsafe { state.as_mut() } {
        s.mark_written();
    }
}

/// Starts `state` from a profile line of an earlier run; false when the line is not one.
///
/// # Safety
/// `state` came from framefx_hud_new; `line` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_warm(state: *mut HudState, line: *const u8, len: usize) -> bool {
    // SAFETY: per this function's contract.
    unsafe {
        match (state.as_mut(), text(line, len)) {
            (Some(s), Some(t)) => s.warm(t.trim()),
            _ => false,
        }
    }
}

/// Replaces the shipped weights with a game's own `hudweights` line; false when it does not fit
/// this build.
///
/// # Safety
/// `state` came from framefx_hud_new; `line` holds `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn framefx_hud_set_weights(state: *mut HudState, line: *const u8, len: usize) -> bool {
    // SAFETY: per this function's contract.
    unsafe {
        match (state.as_mut(), text(line, len)) {
            (Some(s), Some(t)) => s.set_weights(t.trim()),
            _ => false,
        }
    }
}

/// The table key and first slot of a pipeline in a pass, for the C++ side's own copy to be held to.
#[no_mangle]
pub extern "C" fn framefx_hud_key(pass: u64, pipeline: u64, slot: *mut u32) -> u64 {
    let key = table_key(pass, pipeline);
    if !slot.is_null() {
        // SAFETY: a writable u32 when not null.
        unsafe { slot.write(plan::slot(key) as u32) };
    }
    key
}

#[no_mangle]
pub extern "C" fn framefx_hud_table_size() -> usize {
    TABLE_SIZE
}

#[no_mangle]
pub extern "C" fn framefx_hud_max_probe() -> usize {
    plan::MAX_PROBE
}
