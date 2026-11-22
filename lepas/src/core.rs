// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! lepasd's decisions, apart from syscalls and clocks so the host tests drive them: how an engine
//! stores a cap, when a frame rate is a steady cap, when a probe found the slot, what the hold does
//! with what it reads, maps parsing, the single-pass scan matcher, generic structural ranking,
//! fingerprints, the cache file and the start property.

use std::collections::{HashMap, HashSet};

pub(crate) const CAPS: [f64; 11] = [20.0, 24.0, 25.0, 30.0, 40.0, 45.0, 48.0, 50.0, 60.0, 72.0, 90.0];

// ---------------------------------------------------------------- forms

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Kind {
    I32,
    I64,
    F32,
    F64,
}

/// One way an engine keeps its cap: the value it stores for a given frame rate.
pub(crate) struct Form {
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
    pub(crate) of: fn(f64) -> f64,
}

pub(crate) const FORMS: &[Form] = &[
    Form { name: "fps_i32", kind: Kind::I32, of: |f| f.round() },
    Form { name: "fps_f32", kind: Kind::F32, of: |f| f },
    Form { name: "fps_f64", kind: Kind::F64, of: |f| f },
    Form { name: "s_f64", kind: Kind::F64, of: |f| 1.0 / f },
    Form { name: "s_f32", kind: Kind::F32, of: |f| 1.0 / f },
    Form { name: "ms_f64", kind: Kind::F64, of: |f| 1000.0 / f },
    Form { name: "ms_f32", kind: Kind::F32, of: |f| 1000.0 / f },
    Form { name: "us_i32", kind: Kind::I32, of: |f| (1e6 / f).floor() },
    Form { name: "us_i32r", kind: Kind::I32, of: |f| (1e6 / f).round() },
    Form { name: "us_i64", kind: Kind::I64, of: |f| (1e6 / f).floor() },
    Form { name: "ns_i64", kind: Kind::I64, of: |f| (1e9 / f).floor() },
    Form { name: "ns_i64r", kind: Kind::I64, of: |f| (1e9 / f).round() },
    Form { name: "ns_i32", kind: Kind::I32, of: |f| (1e9 / f).floor() },
];

impl Form {
    pub(crate) fn size(&self) -> usize {
        match self.kind {
            Kind::I32 | Kind::F32 => 4,
            Kind::I64 | Kind::F64 => 8,
        }
    }

    /// The slot's bytes for `fps`, little-endian in the low `size()` bytes.
    pub(crate) fn bits(&self, fps: f64) -> u64 {
        let v = (self.of)(fps);
        match self.kind {
            Kind::I32 => v as i32 as u32 as u64,
            Kind::I64 => v as i64 as u64,
            Kind::F32 => (v as f32).to_bits() as u64,
            Kind::F64 => v.to_bits(),
        }
    }

    pub(crate) fn index(name: &str) -> Option<usize> {
        FORMS.iter().position(|f| f.name == name)
    }
}

pub(crate) fn nearest_cap(f: f64) -> f64 {
    *CAPS.iter().min_by(|a, b| (*a - f).abs().total_cmp(&(*b - f).abs())).unwrap()
}

/// Engine cap behind a built-in 2x frame-generation layer (e.g. 90 presented = 45 real,
/// 60 presented = 30 real), when `cap / 2` is itself a standard cap.
pub(crate) fn half_cap(cap: f64) -> Option<f64> {
    let h = cap / 2.0;
    (cap >= 60.0 && CAPS.contains(&h)).then_some(h)
}

// ---------------------------------------------------------------- timing decisions

/// A steady cap below the target: `need` readings in a row within 3 % of the same cap.
pub(crate) struct Steady {
    run: u32,
    need: u32,
    last: f64,
}

impl Default for Steady {
    fn default() -> Self {
        Self::with_need(3)
    }
}

impl Steady {
    pub(crate) fn with_need(need: u32) -> Self {
        Self { run: 0, need, last: 0.0 }
    }

    pub(crate) fn feed(&mut self, fps: f64, target: f64) -> Option<f64> {
        let c = nearest_cap(fps);
        if fps < 5.0 || (fps - c).abs() > c * 0.03 || c >= target * 0.9 {
            self.run = 0;
            return None;
        }
        self.run = if c == self.last { self.run + 1 } else { 1 };
        self.last = c;
        (self.run >= self.need).then_some(c)
    }
}

/// Lower cap written during a probe. Avoids fractional rates (22.5) and values below common
/// engine minimum-FPS floors (~25 fps) whenever a clean cap like 30 still produces a clear drop.
pub(crate) fn probe_rate(slot_cap: f64) -> f64 {
    if slot_cap >= 55.0 {
        (slot_cap / 2.0).round()
    } else if slot_cap >= 44.0 {
        30.0
    } else {
        (slot_cap / 2.0).round().max(10.0)
    }
}

/// With `probe_val` written into a slot holding `slot_cap` (while the game presents at
/// `measured_cap`), the game is slowed when its rate drops near `measured_cap * (probe_val / slot_cap)`.
pub(crate) fn probe_slowed(fps: f64, measured_cap: f64, slot_cap: f64, probe_val: f64) -> bool {
    let ceiling = (probe_val / slot_cap + 0.12).min(0.82);
    fps > 1.0 && fps < measured_cap * ceiling
}

pub(crate) fn probe_recovered(fps: f64, cap: f64) -> bool {
    fps > cap * 0.75
}

/// With the target written, the game ran `fps`. A slot can slow the game without lifting it: a
/// pace derived from the real cap, which the engine still clamps to. Only a lift proves the slot.
pub(crate) fn lifted(fps: f64, cap: f64) -> bool {
    fps > cap * 1.08
}

/// After a candidate appeared to lift `measured_cap` to `fps_up`, restoring its original value
/// must drop the frame rate back down: rules out a coincident loading or scene transition.
pub(crate) fn reverted(fps_back: f64, fps_up: f64, measured_cap: f64) -> bool {
    fps_back > 1.0 && fps_back < fps_up * 0.94 && fps_back <= measured_cap * 1.10
}

/// The game runs well above the cap being learned: it left the screen that set that cap (a
/// loading screen at 30, the world at 60), so its slots no longer hold it.
pub(crate) fn cap_moved(fps: f64, cap: f64) -> bool {
    fps > cap * 1.25
}

#[derive(Debug, PartialEq)]
pub(crate) enum Hold {
    /// The target is there.
    Keep,
    /// The game wrote one of its caps back: write the target again.
    Rewrite,
    /// Something that is no cap: the object may be gone.
    Strange,
}

pub(crate) fn hold_verdict(value: u64, form: &Form, target: f64) -> Hold {
    if value == form.bits(target) {
        Hold::Keep
    } else if CAPS.iter().any(|&c| c < target && form.bits(c) == value) {
        Hold::Rewrite
    } else {
        Hold::Strange
    }
}

/// A cached library slot is trusted when it holds a cap the game may start with, or the target.
pub(crate) fn plausible(value: u64, form: &Form, target: f64) -> bool {
    hold_verdict(value, form, target) != Hold::Strange
}

// ---------------------------------------------------------------- built-in frame generation

/// Decodes a NUL-terminated printable UTF-8/ASCII or UTF-16LE string from raw bytes.
/// Requires at least 6 characters and at least 4 ASCII letters to reject raw binary data.
pub(crate) fn decode_text(bytes: &[u8]) -> Option<String> {
    let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    if nul >= 6 {
        let valid = match std::str::from_utf8(&bytes[..nul]) {
            Ok(s) => Some(s),
            Err(e) if e.error_len().is_none() && e.valid_up_to() >= 6 => {
                std::str::from_utf8(&bytes[..e.valid_up_to()]).ok()
            }
            Err(_) => None,
        };
        if let Some(s) = valid {
            if s.chars().all(|c| !c.is_control() || matches!(c, '\t' | '\n' | '\r'))
                && s.bytes().filter(|b| b.is_ascii_alphabetic()).count() >= 4
            {
                return Some(s.to_string());
            }
        }
    }
    let mut out = String::new();
    let mut ascii_alpha = 0usize;
    for pair in bytes.chunks_exact(2) {
        let u = u16::from_le_bytes([pair[0], pair[1]]);
        if u == 0 {
            break;
        }
        let ok = (0x20..=0x7e).contains(&u)
            || matches!(u, 0x09 | 0x0a | 0x0d)
            || (0x00a0..=0xd7ff).contains(&u)
            || (0xe000..=0xfffd).contains(&u);
        if !ok {
            return None;
        }
        let ch = char::from_u32(u as u32)?;
        if ch.is_ascii_alphabetic() {
            ascii_alpha += 1;
        }
        out.push(ch);
    }
    (out.chars().count() >= 6 && ascii_alpha >= 4).then_some(out)
}

/// Priority score (0..=3) that a setting's description names a built-in frame-generation /
/// frame-interpolation toggle. Uses only generic graphics terminology (no engine or vendor names).
pub(crate) fn is_framegen_desc(text: &str) -> u8 {
    let s = text.to_ascii_lowercase();
    let direct = [
        "framegen",
        "frame gen",
        "frame_gen",
        "frame-gen",
        "frameinterp",
        "frame interp",
        "frame_interp",
        "frame-interp",
        "frameextrap",
        "frame extrap",
        "frame_extrap",
        "frame-extrap",
        "frameinsert",
        "frame insert",
        "frame_insert",
        "frame-insert",
    ]
    .iter()
    .any(|k| s.contains(k));
    let split = s.contains("frame") && ["interpolat", "extrapolat"].iter().any(|k| s.contains(k));
    if !direct && !split {
        return 0;
    }
    let toggle = ["enable", "disable", "toggle", "switch", "0:", "1:", "on/off", "off/on"]
        .iter()
        .any(|k| s.contains(k));
    match (direct, toggle) {
        (true, true) => 3,
        (true, false) | (false, true) => 2,
        (false, false) => 1,
    }
}

/// Recognizes an enabled boolean/mode toggle stored at a setting slot:
///   - `(v, v)` with `v in 1..=3` (when `paired_slots` is true) -> 8-byte paired `i32` toggle
///   - `lo == 0x0000_0101` -> 4-byte paired `bool` toggle `[true, true]`
///   - `lo in 1..=3` -> 4-byte scalar `bool`/`i32` toggle
pub(crate) fn toggle_on_size(word: u64, paired_slots: bool) -> Option<(usize, u64)> {
    let lo = word as u32;
    let hi = (word >> 32) as u32;
    if paired_slots && (1..=3).contains(&lo) && hi == lo {
        Some((8, word))
    } else if lo == 0x0000_0101 || (1..=3).contains(&lo) {
        Some((4, lo as u64))
    } else {
        None
    }
}

/// Disabling 2x built-in framegen while the engine cap sits at `slot_cap` (~`measured_cap / 2`)
/// drops presented FPS sharply down near `slot_cap`.
pub(crate) fn fg_disabled_drop(fps_off: f64, measured_cap: f64, slot_cap: f64) -> bool {
    fps_off > 1.0 && fps_off < measured_cap * 0.68 && fps_off <= slot_cap * 1.25
}

/// Causal revert check for built-in framegen: restoring the toggle jumps FPS back up near
/// `measured_cap`, proving the drop was not a scene transition or stall.
pub(crate) fn fg_restored_rise(fps_on: f64, fps_off: f64, measured_cap: f64) -> bool {
    fps_on > measured_cap * 0.78 && fps_on > fps_off * 1.35
}

/// What `hold()` does with a built-in framegen toggle being held at `0` (disabled).
pub(crate) fn fg_hold_verdict(value: u64, size: usize, on_bits: u64) -> Hold {
    if value == 0 {
        Hold::Keep
    } else if value == on_bits || is_toggle_on_value(value, size) {
        Hold::Rewrite
    } else {
        Hold::Strange
    }
}

fn is_toggle_on_value(value: u64, size: usize) -> bool {
    if size == 8 {
        let lo = value as u32;
        let hi = (value >> 32) as u32;
        (1..=3).contains(&lo) && (hi == lo || hi == 0)
    } else {
        let lo = value as u32;
        lo == 0x0000_0101 || lo == 0x0000_0100 || (1..=3).contains(&lo)
    }
}

// ---------------------------------------------------------------- maps

#[derive(Debug, PartialEq)]
pub(crate) struct Region {
    pub(crate) lo: u64,
    pub(crate) hi: u64,
    pub(crate) label: String,
    pub(crate) file_off: u64,
    pub(crate) lib: bool,
}

/// The writable private mappings of a /proc/pid/maps text; a .bss is named after the library
/// before it. Dalvik, GPU buffers and stacks are skipped: no engine keeps its cap there.
pub(crate) fn regions(maps: &str) -> Vec<Region> {
    let mut out = Vec::new();
    let mut owner: Option<String> = None;
    for line in maps.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let Some((lo, hi)) = f[0].split_once('-') else { continue };
        let (Ok(lo), Ok(hi)) = (u64::from_str_radix(lo, 16), u64::from_str_radix(hi, 16)) else { continue };
        let name = f.get(5).copied().unwrap_or("");
        let base = name.rsplit('/').next().unwrap_or("").to_string();
        let perms = f[1];
        if perms.starts_with("rw")
            && perms.ends_with('p')
            && !["dalvik", "kgsl", "stack", "dmabuf", "gralloc"].iter().any(|s| name.contains(s))
        {
            let bss = name == "[anon:.bss]";
            let label = match (&owner, bss) {
                (Some(o), true) => format!("{o} .bss"),
                _ => base.clone(),
            };
            let lib = base.ends_with(".so") || (bss && owner.is_some());
            out.push(Region { lo, hi, label, file_off: u64::from_str_radix(f[2], 16).unwrap_or(0), lib });
        }
        owner = if base.ends_with(".so") {
            Some(base)
        } else if name == "[anon:.bss]" {
            owner
        } else {
            None
        };
    }
    out
}

pub(crate) fn region_of(regs: &[Region], a: u64) -> Option<&Region> {
    regs.iter().find(|r| r.lo <= a && a < r.hi)
}

/// Sorted, merged address ranges of the app's own `.so` mappings (falling back to all `.so`
/// mappings when none live under `/data/app/`). Used to recognise vtable/rodata pointers on the
/// heap without naming any engine or library.
pub(crate) fn lib_ranges(maps: &str) -> Vec<(u64, u64)> {
    collect_so_ranges(maps, false)
}

/// Non-executable readable segments (`r--p`, `rw-p`) of the app's `.so` mappings where C++
/// vtables and constant data live (excluding `.text` function pointers).
pub(crate) fn lib_data_ranges(maps: &str) -> Vec<(u64, u64)> {
    collect_so_ranges(maps, true)
}

fn collect_so_ranges(maps: &str, data_only: bool) -> Vec<(u64, u64)> {
    let mut app = Vec::new();
    let mut any = Vec::new();
    for line in maps.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 6 {
            continue;
        }
        let name = f[5];
        if !name.ends_with(".so") || !f[1].starts_with('r') || (data_only && f[1].contains('x')) {
            continue;
        }
        let Some((lo, hi)) = f[0].split_once('-') else { continue };
        let (Ok(lo), Ok(hi)) = (u64::from_str_radix(lo, 16), u64::from_str_radix(hi, 16)) else { continue };
        if name.starts_with("/data/app/") {
            app.push((lo, hi));
        }
        any.push((lo, hi));
    }
    let mut raw = if app.is_empty() { any } else { app };
    raw.sort_unstable_by_key(|r| r.0);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(raw.len());
    for (lo, hi) in raw {
        if let Some(last) = merged.last_mut() {
            if lo <= last.1 {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    merged
}

pub(crate) fn strip_tag(addr: u64) -> u64 {
    addr & 0x00ff_ffff_ffff_ffff
}

pub(crate) fn containing_range(ranges: &[(u64, u64)], addr: u64) -> Option<(u64, u64)> {
    let tag = addr >> 56;
    if tag != 0 && tag != 0xb4 {
        return None;
    }
    let raw = strip_tag(addr);
    let idx = ranges.partition_point(|r| r.0 <= raw);
    (idx > 0 && raw < ranges[idx - 1].1).then(|| ranges[idx - 1])
}

pub(crate) fn in_ranges(ranges: &[(u64, u64)], addr: u64) -> bool {
    containing_range(ranges, addr).is_some()
}

/// The start of the framefx status page in a maps text.
pub(crate) fn status_mapping(maps: &str, name: &str) -> Vec<u64> {
    maps.lines()
        .filter(|l| l.contains(name))
        .filter_map(|l| u64::from_str_radix(l.split('-').next()?, 16).ok())
        .collect()
}

// ---------------------------------------------------------------- fast single-pass scan

#[inline(always)]
fn bloom32(v: u32) -> u64 {
    1u64 << (v.wrapping_mul(0x9e37_79b1) >> 26)
}

#[inline(always)]
fn bloom64(v: u64) -> u64 {
    1u64 << (v.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 58)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TargetSpec {
    pub(crate) id: usize,
    pub(crate) size: usize,
    pub(crate) bits: u64,
}

/// Targets split by width with 64-bit bloom prefilters, for one single-read pass over memory.
pub(crate) struct Targets {
    four: Vec<(usize, u32)>,
    eight: Vec<(usize, u64)>,
    four_bloom: u64,
    eight_bloom: u64,
}

impl Targets {
    pub(crate) fn new(specs: &[TargetSpec]) -> Targets {
        let mut four = Vec::new();
        let mut eight = Vec::new();
        let mut four_bloom = 0u64;
        let mut eight_bloom = 0u64;
        for &s in specs {
            if s.size == 4 {
                let b = s.bits as u32;
                four.push((s.id, b));
                four_bloom |= bloom32(b);
            } else {
                eight.push((s.id, s.bits));
                eight_bloom |= bloom64(s.bits);
            }
        }
        Targets { four, eight, four_bloom, eight_bloom }
    }

    /// Scans `bytes` (read from `base`, 8-byte aligned) in ONE linear pass over `u64` words.
    /// Invokes `on_hit(target_id, addr, byte_offset_in_bytes)` on every match.
    pub(crate) fn scan_with<F: FnMut(usize, u64, usize)>(&self, base: u64, bytes: &[u8], mut on_hit: F) {
        let has4 = !self.four.is_empty();
        let has8 = !self.eight.is_empty();
        let b4 = self.four_bloom;
        let b8 = self.eight_bloom;
        for (k, chunk) in bytes.chunks_exact(8).enumerate() {
            let w = u64::from_le_bytes(chunk.try_into().unwrap());
            // No cap value in FORMS encodes to zero; skipping zero words bypasses most idle RAM.
            if w == 0 {
                continue;
            }
            let off = k * 8;
            let addr = base + off as u64;
            if has8 && (b8 & bloom64(w)) != 0 {
                for &(id, b) in &self.eight {
                    if w == b {
                        on_hit(id, addr, off);
                    }
                }
            }
            if has4 {
                let lo = w as u32;
                if lo != 0 && (b4 & bloom32(lo)) != 0 {
                    for &(id, b) in &self.four {
                        if lo == b {
                            on_hit(id, addr, off);
                        }
                    }
                }
                let hi = (w >> 32) as u32;
                if hi != 0 && (b4 & bloom32(hi)) != 0 {
                    for &(id, b) in &self.four {
                        if hi == b {
                            on_hit(id, addr + 4, off + 4);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- generic structural ranking

/// Compact neighborhood summary captured inline from the scan buffer (zero extra syscalls).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SlotMeta {
    pub(crate) addr: u64,
    /// Closest app-`.so` pointer within `-128..0` bytes packed with its distance: `(ptr << 8) | dist`.
    pub(crate) vptr_key: u64,
    /// 1-byte digest per 8-byte word across `-64..+64` (excluding the slot's word), to count
    /// live mutating neighbors across the stability interval without keeping full buffers.
    pub(crate) ctx_sig: [u8; 16],
    pub(crate) ptr_count: u8,
    pub(crate) live_neighbors: u8,
    pub(crate) paired_first: bool,
    pub(crate) paired_second: bool,
    pub(crate) in_array: bool,
    pub(crate) lib: bool,
}

#[inline]
fn read_slot(bytes: &[u8], off: usize, size: usize) -> u64 {
    if size == 4 {
        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as u64
    } else {
        u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
    }
}

#[inline]
fn word_digest(w: u64) -> u8 {
    let x = w ^ (w >> 32);
    let y = (x ^ (x >> 16)) as u16;
    (y ^ (y >> 8)) as u8
}

/// Computes the 16-word neighborhood digest around `off` in `bytes` (`[-64..+64]` aligned).
pub(crate) fn context_sig(bytes: &[u8], off: usize) -> [u8; 16] {
    let mut sig = [0u8; 16];
    let slot_word = off & !7;
    for (i, d) in sig.iter_mut().enumerate() {
        let delta = (i as isize - 8) * 8;
        if delta == 0 {
            continue;
        }
        let pos = slot_word as isize + delta;
        if pos >= 0 && (pos as usize) + 8 <= bytes.len() {
            let p = pos as usize;
            let w = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
            *d = word_digest(w);
        }
    }
    sig
}

pub(crate) fn diff_sig(a: &[u8; 16], b: &[u8; 16]) -> u8 {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as u8
}

/// Extracts generic structural properties of a candidate slot directly from the read buffer.
pub(crate) fn inspect_slot(
    addr: u64,
    bytes: &[u8],
    off: usize,
    size: usize,
    bits: u64,
    lib: bool,
    so_ranges: &[(u64, u64)],
) -> SlotMeta {
    let prev_eq = off >= size && read_slot(bytes, off - size, size) == bits;
    let next_eq = off + 2 * size <= bytes.len() && read_slot(bytes, off + size, size) == bits;
    let prev2_eq = off >= 2 * size && read_slot(bytes, off - 2 * size, size) == bits;
    let next2_eq = off + 3 * size <= bytes.len() && read_slot(bytes, off + 2 * size, size) == bits;
    let in_array = (prev_eq && (prev2_eq || next_eq)) || (next_eq && next2_eq);
    let paired_first = next_eq && !prev_eq && !next2_eq;
    let paired_second = prev_eq && !next_eq && !prev2_eq;

    let mut vptr_key = 0u64;
    let mut ptr_count = 0u8;
    if !lib {
        let slot_word = off & !7;
        let lo = slot_word.saturating_sub(128);
        let hi = (slot_word + 128).min(bytes.len() & !7);
        let mut p = lo;
        while p + 8 <= hi {
            if p != slot_word {
                let w = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
                if is_pointer(w) {
                    ptr_count = ptr_count.saturating_add(1);
                    if p < slot_word && in_ranges(so_ranges, w) {
                        let dist = (off - p) as u64;
                        // Loop walks forward toward `off`, so the last match is the closest one
                        // before `off` (inside this object, not a preceding pool entry).
                        if dist <= 128 {
                            vptr_key = ((w & 0x00ff_ffff_ffff_ffff) << 8) | dist;
                        }
                    }
                }
            }
            p += 8;
        }
    }

    SlotMeta {
        addr,
        vptr_key,
        ctx_sig: if lib { [0u8; 16] } else { context_sig(bytes, off) },
        ptr_count,
        live_neighbors: 0,
        paired_first,
        paired_second,
        in_array,
        lib,
    }
}

/// Purely structural likelihood that a heap slot belongs to an engine singleton, CVar, or live
/// pacing struct rather than raw numeric arrays, UI coordinates, or repeated scene actors.
pub(crate) fn structural_score(m: &SlotMeta, vptr_freq: u32) -> u32 {
    if m.in_array {
        return 0;
    }
    let mut s = 0u32;
    if m.vptr_key != 0 {
        s += match vptr_freq {
            1..=4 => 60,
            5..=12 => 30,
            _ => 5,
        };
    }
    if m.paired_first {
        s += 45;
    } else if m.paired_second {
        s += 35;
    }
    if (1..=10).contains(&m.live_neighbors) && m.ptr_count >= 2 {
        s += 45;
    }
    if m.ptr_count >= 3 {
        s += 15;
    } else if m.ptr_count >= 1 {
        s += 5;
    }
    s
}

/// One candidate scheduled for single-slot probing: `(form_index, slot_cap, addr)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) form: usize,
    pub(crate) slot_cap: f64,
    pub(crate) addr: u64,
}

/// One scanned bucket `(form_index, slot_cap, surviving_slots)`.
pub(crate) struct Bucket {
    pub(crate) form: usize,
    pub(crate) slot_cap: f64,
    pub(crate) primary: bool,
    pub(crate) slots: Vec<SlotMeta>,
}

const SMALL_LIB: usize = 48;
const TINY_HEAP: usize = 32;
const HIGH_STRUCT_SCORE: u32 = 85;
const MED_STRUCT_SCORE: u32 = 55;
const MAX_RANKED_PER_BUCKET: usize = 28;
const MAX_HALF_LIB: usize = 150;
const MAX_HALF_HEAP: usize = 64;

/// Builds a prioritized, deduplicated single-slot probe order across primary and optional
/// half-cap (2x framegen) buckets without ever batch-writing or naming any game/engine:
///   1. Followers: slots that held the previous screen's cap and now hold this one.
///   2. Small library parts (<= 48) and tiny primary heap parts (<= 32).
///   3. High-confidence structural heap objects (e.g. rare vtable + shadowed pair, or live
///      vtable pacing struct) — even when the raw heap bucket has 30,000+ slots.
///   4. Remaining primary library parts (<= `max`).
///   5. Medium-confidence structural heap objects (e.g. live pacing struct with pointers).
///   6. Remaining primary medium heap parts (<= `max`) and half-cap library/small-heap fallbacks.
pub(crate) fn plan(
    buckets: &[Bucket],
    prev: &[(usize, Vec<u64>)],
    skip: &[(usize, bool)],
    max: usize,
) -> (Vec<Candidate>, usize) {
    let mut out: Vec<Candidate> = Vec::new();
    let mut taken: HashSet<(usize, u64)> = HashSet::new();

    // 1. Slots that followed a cap change (primary first, then half-cap).
    let prev_sets: HashMap<usize, HashSet<u64>> = prev
        .iter()
        .map(|(f, v)| (*f, v.iter().copied().collect()))
        .collect();
    for want_primary in [true, false] {
        for b in buckets.iter().filter(|b| b.primary == want_primary) {
            let Some(before) = prev_sets.get(&b.form) else { continue };
            for s in &b.slots {
                if before.contains(&s.addr) && taken.insert((b.form, s.addr)) {
                    out.push(Candidate { form: b.form, slot_cap: b.slot_cap, addr: s.addr });
                }
            }
        }
    }
    let followed = out.len();

    // Precompute structural scores per bucket for heap slots.
    struct RankedBucket {
        form: usize,
        slot_cap: f64,
        primary: bool,
        lib_addrs: Vec<u64>,
        heap_scored: Vec<(u32, u64)>,
    }

    let mut ranked: Vec<RankedBucket> = Vec::with_capacity(buckets.len());
    for b in buckets {
        let mut lib_addrs = Vec::new();
        let mut vfreq: HashMap<u64, u32> = HashMap::new();
        for s in &b.slots {
            if !s.lib && s.vptr_key != 0 {
                *vfreq.entry(s.vptr_key).or_insert(0) += 1;
            }
        }
        let mut heap_scored: Vec<(u32, u64)> = Vec::new();
        for s in &b.slots {
            if s.lib {
                lib_addrs.push(s.addr);
            } else {
                let freq = if s.vptr_key != 0 { *vfreq.get(&s.vptr_key).unwrap_or(&1) } else { 0 };
                heap_scored.push((structural_score(s, freq), s.addr));
            }
        }
        // Highest structural score first; deterministic address tie-break.
        heap_scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        ranked.push(RankedBucket {
            form: b.form,
            slot_cap: b.slot_cap,
            primary: b.primary,
            lib_addrs,
            heap_scored,
        });
    }

    let mut push = |form: usize, slot_cap: f64, addr: u64| {
        if taken.insert((form, addr)) {
            out.push(Candidate { form, slot_cap, addr });
        }
    };

    // 2. Small library parts (primary <= 48) + tiny primary heap parts (<= 32), smallest first.
    let mut tier2: Vec<(usize, usize, f64, Vec<u64>)> = Vec::new();
    for r in &ranked {
        if !r.primary {
            continue;
        }
        if !r.lib_addrs.is_empty() && r.lib_addrs.len() <= SMALL_LIB.min(max) && !skip.contains(&(r.form, true)) {
            tier2.push((r.lib_addrs.len() / 2, r.form, r.slot_cap, r.lib_addrs.clone()));
        }
        if !r.heap_scored.is_empty() && r.heap_scored.len() <= TINY_HEAP.min(max) && !skip.contains(&(r.form, false)) {
            tier2.push((
                r.heap_scored.len(),
                r.form,
                r.slot_cap,
                r.heap_scored.iter().map(|&(_, a)| a).collect(),
            ));
        }
    }
    tier2.sort_by_key(|t| t.0);
    for (_, form, cap, addrs) in tier2 {
        for a in addrs {
            push(form, cap, a);
        }
    }

    // 3. High-confidence structural heap objects across all forms, sorted globally by score desc
    // (then primary before half-cap, then smaller raw bucket first) so true singletons / paired
    // CVars (score 120) run before weaker score-85 hits in noisy integer buckets.
    let mut tier3: Vec<(u32, bool, usize, usize, f64, u64)> = Vec::new();
    for r in &ranked {
        if skip.contains(&(r.form, false)) {
            continue;
        }
        let raw_len = r.heap_scored.len();
        for &(sc, a) in r
            .heap_scored
            .iter()
            .take_while(|&&(sc, _)| sc >= HIGH_STRUCT_SCORE)
            .take(MAX_RANKED_PER_BUCKET)
        {
            tier3.push((sc, r.primary, raw_len, r.form, r.slot_cap, a));
        }
    }
    tier3.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.5.cmp(&b.5))
    });
    for (_, _, _, form, cap, a) in tier3 {
        push(form, cap, a);
    }

    // 4. Remaining primary library parts (<= max), smallest first (Unity / Genshin .data/.bss).
    let mut tier4: Vec<(usize, usize, f64, Vec<u64>)> = Vec::new();
    for r in ranked.iter().filter(|r| r.primary) {
        if !r.lib_addrs.is_empty()
            && r.lib_addrs.len() > SMALL_LIB
            && r.lib_addrs.len() <= max
            && !skip.contains(&(r.form, true))
        {
            tier4.push((r.lib_addrs.len(), r.form, r.slot_cap, r.lib_addrs.clone()));
        }
    }
    tier4.sort_by_key(|t| t.0);
    for (_, form, cap, addrs) in tier4 {
        for a in addrs {
            push(form, cap, a);
        }
    }

    // 5. Medium-confidence structural heap objects (primary then half-cap, e.g. Endfield fps_f64).
    // Ordered by raw heap size so cleaner forms (f64/f32) go ahead of noisy i32.
    for want_primary in [true, false] {
        let mut groups: Vec<(usize, usize, f64, Vec<u64>)> = Vec::new();
        for r in ranked.iter().filter(|r| r.primary == want_primary) {
            if skip.contains(&(r.form, false)) {
                continue;
            }
            let top: Vec<u64> = r
                .heap_scored
                .iter()
                .filter(|&&(sc, _)| (MED_STRUCT_SCORE..HIGH_STRUCT_SCORE).contains(&sc))
                .take(MAX_RANKED_PER_BUCKET)
                .map(|&(_, a)| a)
                .collect();
            if !top.is_empty() {
                groups.push((r.heap_scored.len(), r.form, r.slot_cap, top));
            }
        }
        groups.sort_by_key(|g| g.0);
        for (_, form, cap, addrs) in groups {
            for a in addrs {
                push(form, cap, a);
            }
        }
    }

    // 6. Half-cap library parts (<= 150) + remaining primary heap parts (<= max) + half-cap small heap (<= 64).
    let mut tier6: Vec<(usize, usize, f64, Vec<u64>)> = Vec::new();
    for r in &ranked {
        if !r.primary {
            if !r.lib_addrs.is_empty() && r.lib_addrs.len() <= MAX_HALF_LIB.min(max) && !skip.contains(&(r.form, true)) {
                tier6.push((r.lib_addrs.len() / 2, r.form, r.slot_cap, r.lib_addrs.clone()));
            }
            if !r.heap_scored.is_empty() && r.heap_scored.len() <= MAX_HALF_HEAP.min(max) && !skip.contains(&(r.form, false)) {
                tier6.push((
                    r.heap_scored.len() + 64,
                    r.form,
                    r.slot_cap,
                    r.heap_scored.iter().map(|&(_, a)| a).collect(),
                ));
            }
        } else if r.heap_scored.len() > TINY_HEAP && r.heap_scored.len() <= max && !skip.contains(&(r.form, false)) {
            tier6.push((
                r.heap_scored.len(),
                r.form,
                r.slot_cap,
                r.heap_scored.iter().map(|&(_, a)| a).collect(),
            ));
        }
    }
    tier6.sort_by_key(|t| t.0);
    for (_, form, cap, addrs) in tier6 {
        for a in addrs {
            push(form, cap, a);
        }
    }

    (out, followed)
}

/// One line of `<pkg>.poison`: a part that killed `version` of the game under test.
pub(crate) fn poison_line(version: &str, form: usize, lib: bool) -> String {
    format!("{version} {} {}\n", FORMS[form].name, if lib { "lib" } else { "heap" })
}

/// The parts `version` of the game died on; lines of other versions or unknown forms are ignored.
pub(crate) fn poisoned(text: &str, version: &str) -> Vec<(usize, bool)> {
    text.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let [v, name, part] = f[..] else { return None };
            let form = FORMS.iter().position(|x| x.name == name)?;
            (v == version && matches!(part, "lib" | "heap")).then_some((form, part == "lib"))
        })
        .collect()
}

// ---------------------------------------------------------------- fingerprint

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Token {
    Any,
    Pointer,
    Exact(u64),
}

/// A user-space pointer, tagged (0xb4, Android's heap tag) or not.
pub(crate) fn is_pointer(v: u64) -> bool {
    let tag = v >> 56;
    let low = v & 0x00ff_ffff_ffff_ffff;
    (tag == 0 || tag == 0xb4) && (0x10_0000_0000..0x80_0000_0000).contains(&low)
}

pub(crate) fn words(bytes: &[u8]) -> Vec<u64> {
    bytes.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect()
}

/// A neighbourhood read twice: words that moved match anything, pointers match any pointer.
pub(crate) fn fingerprint(a: &[u8], b: &[u8]) -> Vec<Token> {
    words(a)
        .into_iter()
        .zip(words(b))
        .map(|(x, y)| if x != y { Token::Any } else if is_pointer(x) { Token::Pointer } else { Token::Exact(x) })
        .collect()
}

/// The share of a fingerprint's fixed words a candidate's neighbourhood matches.
pub(crate) fn score(fp: &[Token], bytes: &[u8]) -> f64 {
    let (mut total, mut same) = (0u32, 0u32);
    for (t, w) in fp.iter().zip(bytes.chunks_exact(8)) {
        let w = u64::from_le_bytes(w.try_into().unwrap());
        match t {
            Token::Any => {}
            Token::Pointer => {
                total += 1;
                same += is_pointer(w) as u32;
            }
            Token::Exact(v) => {
                total += 1;
                same += (*v == w) as u32;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        same as f64 / total as f64
    }
}

/// From scores of every candidate: the one to write at once, the one to confirm by a probe first,
/// or none. A clear winner is needed either way.
#[derive(Debug, PartialEq)]
pub(crate) enum Pick {
    Sure(u64),
    Check(u64),
    None,
}

pub(crate) fn pick(mut scored: Vec<(f64, u64)>) -> Pick {
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let Some(&(best, addr)) = scored.first() else { return Pick::None };
    let second = scored.get(1).map(|s| s.0).unwrap_or(0.0);
    if best >= 0.97 && best - second >= 0.05 {
        Pick::Sure(addr)
    } else if best >= 0.8 && best - second >= 0.1 {
        Pick::Check(addr)
    } else {
        Pick::None
    }
}

// ---------------------------------------------------------------- cache

/// Cached location of a built-in frame-generation toggle disabled alongside the cap slot.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameGenCache {
    pub(crate) size: usize,
    pub(crate) on_bits: u64,
    pub(crate) lib: bool,
    pub(crate) label: String,
    pub(crate) file_off: u64,
    pub(crate) offset: u64,
    pub(crate) fingerprint: Vec<Token>,
    pub(crate) delta: usize,
}

fn encode_fp(fp: &[Token]) -> String {
    fp.iter()
        .map(|t| match t {
            Token::Any => "*".to_string(),
            Token::Pointer => "p".to_string(),
            Token::Exact(v) => format!("{v:x}"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_fp(s: &str) -> Option<Vec<Token>> {
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| match t {
            "*" => Some(Token::Any),
            "p" => Some(Token::Pointer),
            h => u64::from_str_radix(h, 16).ok().map(Token::Exact),
        })
        .collect()
}

/// Where a game's cap was found, for its next start: a library slot by offset, a heap slot by the
/// region it lives in and its neighbourhood.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Found {
    pub(crate) version: String,
    pub(crate) form: usize,
    pub(crate) lib: bool,
    pub(crate) label: String,
    pub(crate) file_off: u64,
    pub(crate) offset: u64,
    pub(crate) fingerprint: Vec<Token>,
    pub(crate) delta: usize,
    /// 1 normally; 2 when the game presents at 2x the stored engine cap (built-in framegen).
    pub(crate) cap_div: u8,
    /// Optional built-in frame-generation toggle disabled so native FPS can run cleanly.
    pub(crate) fg: Option<FrameGenCache>,
}

impl Found {
    pub(crate) fn to_text(&self) -> String {
        let mut out = format!(
            "version={}\nform={}\nlib={}\nregion={}\nfile_off={}\noffset={}\ndelta={}\ncap_div={}\nfingerprint={}\n",
            self.version,
            FORMS[self.form].name,
            self.lib as u8,
            self.label,
            self.file_off,
            self.offset,
            self.delta,
            self.cap_div.max(1),
            encode_fp(&self.fingerprint)
        );
        if let Some(fg) = &self.fg {
            out.push_str(&format!(
                "fg_size={}\nfg_on={:x}\nfg_lib={}\nfg_region={}\nfg_file_off={}\nfg_offset={}\nfg_delta={}\nfg_fingerprint={}\n",
                fg.size,
                fg.on_bits,
                fg.lib as u8,
                fg.label,
                fg.file_off,
                fg.offset,
                fg.delta,
                encode_fp(&fg.fingerprint)
            ));
        }
        out
    }

    pub(crate) fn from_text(text: &str) -> Option<Found> {
        let kv: HashMap<&str, &str> = text.lines().filter_map(|l| l.split_once('=')).collect();
        let fingerprint = decode_fp(kv.get("fingerprint")?)?;
        let cap_div = kv
            .get("cap_div")
            .and_then(|s| s.parse::<u8>().ok())
            .filter(|&d| d == 1 || d == 2)
            .unwrap_or(1);
        let fg = if let Some(sz) = kv.get("fg_size") {
            let size: usize = sz.parse().ok().filter(|&s| s == 4 || s == 8)?;
            let on_bits = u64::from_str_radix(kv.get("fg_on")?, 16).ok().filter(|&b| b != 0)?;
            Some(FrameGenCache {
                size,
                on_bits,
                lib: *kv.get("fg_lib")? == "1",
                label: kv.get("fg_region")?.to_string(),
                file_off: kv.get("fg_file_off")?.parse().ok()?,
                offset: kv.get("fg_offset")?.parse().ok()?,
                fingerprint: decode_fp(kv.get("fg_fingerprint")?)?,
                delta: kv.get("fg_delta")?.parse().ok()?,
            })
        } else {
            None
        };
        Some(Found {
            version: kv.get("version")?.to_string(),
            form: Form::index(kv.get("form")?)?,
            lib: *kv.get("lib")? == "1",
            label: kv.get("region")?.to_string(),
            file_off: kv.get("file_off")?.parse().ok()?,
            offset: kv.get("offset")?.parse().ok()?,
            fingerprint,
            delta: kv.get("delta")?.parse().ok()?,
            cap_div,
            fg,
        })
    }
}

// ---------------------------------------------------------------- start property

#[derive(Debug, PartialEq)]
pub(crate) struct Start {
    pub(crate) pid: i32,
    pub(crate) target: f64,
    pub(crate) version: String,
    pub(crate) package: String,
}

/// The broker's "<pid> <target> <version> <package>".
pub(crate) fn parse_start(value: &str) -> Option<Start> {
    let f: Vec<&str> = value.splitn(4, ' ').collect();
    let [pid, target, version, package] = f[..] else { return None };
    let pid: i32 = pid.parse().ok().filter(|&p| p > 0)?;
    let target: f64 = target.parse().ok().filter(|&t| (30.0..=240.0).contains(&t))?;
    if package.is_empty() || package.contains('/') || package.contains("..") || package.contains(' ') {
        return None;
    }
    Some(Start { pid, target, version: version.to_string(), package: package.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(name: &str) -> &'static Form {
        &FORMS[Form::index(name).unwrap()]
    }

    fn meta(addr: u64, lib: bool) -> SlotMeta {
        SlotMeta {
            addr,
            vptr_key: 0,
            ctx_sig: [0u8; 16],
            ptr_count: 0,
            live_neighbors: 0,
            paired_first: false,
            paired_second: false,
            in_array: false,
            lib,
        }
    }

    #[test]
    fn forms_encode_what_engines_store() {
        assert_eq!(form("fps_i32").bits(60.0), 60);
        assert_eq!(form("fps_f32").bits(60.0), 60.0f32.to_bits() as u64);
        assert_eq!(form("fps_f64").bits(60.0), 60.0f64.to_bits());
        assert_eq!(form("s_f64").bits(60.0), 0x3f91_1111_1111_1111);
        assert_eq!(form("s_f64").bits(120.0), 0x3f81_1111_1111_1111);
        assert_eq!(form("ms_f64").bits(60.0), (1000.0f64 / 60.0).to_bits());
        assert_eq!(form("us_i32").bits(60.0), 16666);
        assert_eq!(form("us_i32r").bits(60.0), 16667);
        assert_eq!(form("ns_i64").bits(60.0), 16_666_666);
        assert_eq!(form("ns_i64r").bits(60.0), 16_666_667);
        assert_eq!(form("ns_i32").bits(30.0), 33_333_333);
        assert_eq!(form("fps_i32").size(), 4);
        assert_eq!(form("s_f64").size(), 8);
        assert_eq!(form("us_i64").size(), 8);
    }

    #[test]
    fn form_names_are_unique_and_found() {
        for (i, f) in FORMS.iter().enumerate() {
            assert_eq!(Form::index(f.name), Some(i));
        }
        assert_eq!(Form::index("nope"), None);
    }

    #[test]
    fn probe_rate_differs_from_cap_in_every_form_and_avoids_low_floors() {
        assert_eq!(probe_rate(90.0), 45.0);
        assert_eq!(probe_rate(60.0), 30.0);
        // 45 fps (e.g. engine cap behind 90 fps 2x framegen) probes at 30 instead of 22.5.
        assert_eq!(probe_rate(45.0), 30.0);
        assert_eq!(probe_rate(30.0), 15.0);
        for f in FORMS {
            for &c in &CAPS {
                let p = probe_rate(c);
                assert!(p < c, "{c} -> {p}");
                assert_ne!(f.bits(c), f.bits(p), "{} {c} -> {p}", f.name);
                assert_ne!(f.bits(c), 0, "{} {c}", f.name);
            }
        }
    }

    #[test]
    fn half_cap_detects_framegen_pairs() {
        assert_eq!(half_cap(90.0), Some(45.0));
        assert_eq!(half_cap(60.0), Some(30.0));
        assert_eq!(half_cap(45.0), None);
        assert_eq!(half_cap(30.0), None);
    }

    #[test]
    fn nearest_cap_snaps() {
        assert_eq!(nearest_cap(59.4), 60.0);
        assert_eq!(nearest_cap(29.8), 30.0);
        assert_eq!(nearest_cap(44.1), 45.0);
        assert_eq!(nearest_cap(88.0), 90.0);
        assert_eq!(nearest_cap(0.0), 20.0);
    }

    #[test]
    fn steady_needs_three_close_readings_below_target() {
        let mut s = Steady::default();
        assert_eq!(s.feed(60.1, 120.0), None);
        assert_eq!(s.feed(59.8, 120.0), None);
        assert_eq!(s.feed(60.0, 120.0), Some(60.0));
    }

    #[test]
    fn steady_ignores_transients_and_menus_in_between() {
        let mut s = Steady::default();
        s.feed(60.0, 120.0);
        s.feed(60.0, 120.0);
        assert_eq!(s.feed(52.5, 120.0), None);
        assert_eq!(s.feed(45.1, 120.0), None);
        assert_eq!(s.feed(45.0, 120.0), None);
        assert_eq!(s.feed(44.9, 120.0), Some(45.0));
    }

    #[test]
    fn steady_never_reports_paused_or_already_fast_games() {
        let mut s = Steady::default();
        for _ in 0..5 {
            assert_eq!(s.feed(0.0, 120.0), None);
        }
        for _ in 0..5 {
            assert_eq!(s.feed(118.0, 120.0), None);
        }
        let mut s = Steady::default();
        s.feed(90.0, 120.0);
        s.feed(90.0, 120.0);
        assert_eq!(s.feed(90.0, 120.0), Some(90.0));
        let mut s = Steady::default();
        for _ in 0..4 {
            assert_eq!(s.feed(90.0, 90.0), None);
        }
    }

    #[test]
    fn probe_thresholds_and_causal_revert() {
        // Direct cap 60 probed at 30.
        assert!(probe_slowed(30.2, 60.0, 60.0, 30.0));
        assert!(!probe_slowed(59.0, 60.0, 60.0, 30.0));
        assert!(!probe_slowed(40.0, 60.0, 60.0, 30.0));
        assert!(!probe_slowed(0.0, 60.0, 60.0, 30.0));
        // 2x framegen: presented 90, engine cap 45 probed at 30 -> presented drops to ~60.
        assert!(probe_slowed(59.5, 90.0, 45.0, 30.0));
        assert!(!probe_slowed(85.0, 90.0, 45.0, 30.0));
        assert!(probe_recovered(58.0, 60.0));
        assert!(!probe_recovered(30.0, 60.0));

        // Causal revert: real lift drops back when undone; scene transition stays high.
        assert!(reverted(88.7, 120.1, 90.0));
        assert!(reverted(60.0, 68.0, 60.0));
        assert!(!reverted(89.9, 89.9, 45.0)); // loading 45 -> lobby 90 false positive rejected
        assert!(!reverted(0.0, 120.0, 60.0));
    }

    #[test]
    fn plan_prioritizes_small_sets_and_high_structural_heap_over_raw_size() {
        let mut cvar = meta(5000, false);
        cvar.vptr_key = (0x7b24_573a_a800 << 8) | 80;
        cvar.paired_first = true;
        cvar.ptr_count = 4;

        let mut big_heap: Vec<SlotMeta> = (0..5000).map(|a| meta(a, false)).collect();
        big_heap.push(cvar);
        big_heap.push(meta(9000, true));
        big_heap.push(meta(9001, true));

        let buckets = vec![
            Bucket { form: 1, slot_cap: 60.0, primary: true, slots: big_heap },
            Bucket { form: 3, slot_cap: 60.0, primary: true, slots: vec![meta(100, false), meta(101, false)] },
        ];
        let (order, followed) = plan(&buckets, &[], &[], 300);
        assert_eq!(followed, 0);
        // Tiny lib (2) and tiny heap (2) first, then the structural CVar out of the 5000-slot heap!
        let addrs: Vec<u64> = order.iter().map(|c| c.addr).collect();
        assert_eq!(addrs, vec![9000, 9001, 100, 101, 5000]);
    }

    #[test]
    fn plan_includes_half_cap_structural_candidates_and_followers_first() {
        let mut half_cvar = meta(777, false);
        half_cvar.vptr_key = (0x7b24_573a_a800 << 8) | 80;
        half_cvar.paired_first = true;
        half_cvar.ptr_count = 4;

        let buckets = vec![
            Bucket { form: 0, slot_cap: 90.0, primary: true, slots: (0..10_000).map(|a| meta(a, false)).collect() },
            Bucket { form: 1, slot_cap: 45.0, primary: false, slots: vec![half_cvar] },
        ];
        let (order, followed) = plan(&buckets, &[(0, vec![42])], &[], 300);
        assert_eq!(followed, 1);
        assert_eq!(order[0], Candidate { form: 0, slot_cap: 90.0, addr: 42 });
        assert_eq!(order[1], Candidate { form: 1, slot_cap: 45.0, addr: 777 });
    }

    #[test]
    fn a_part_that_killed_the_game_is_skipped_but_not_its_followers() {
        let buckets = vec![
            Bucket { form: 0, slot_cap: 60.0, primary: true, slots: vec![meta(1, false), meta(2, false), meta(3, false)] },
            Bucket { form: 1, slot_cap: 60.0, primary: true, slots: vec![meta(9, true)] },
        ];
        let (order, _) = plan(&buckets, &[(0, vec![2])], &[(0, false)], 300);
        assert_eq!(
            order,
            vec![Candidate { form: 0, slot_cap: 60.0, addr: 2 }, Candidate { form: 1, slot_cap: 60.0, addr: 9 }]
        );
    }

    #[test]
    fn poison_round_trips_per_version() {
        let text = poison_line("61", 1, false) + &poison_line("60", 0, true) + "61 nonsense heap\n61 fps_i32\n";
        assert_eq!(poisoned(&text, "61"), vec![(1, false)]);
        assert_eq!(poisoned(&text, "60"), vec![(0, true)]);
    }

    #[test]
    fn cap_moves_and_lift_thresholds() {
        assert!(cap_moved(60.0, 30.0));
        assert!(!cap_moved(31.0, 30.0));
        assert!(lifted(68.0, 60.0));
        assert!(lifted(119.5, 60.0));
        assert!(!lifted(60.4, 60.0));
        assert!(!lifted(63.0, 60.0));
        assert!(!lifted(0.0, 60.0));
    }

    #[test]
    fn hold_rewrites_caps_keeps_target_flags_the_rest() {
        let f = form("fps_f32");
        assert_eq!(hold_verdict(f.bits(120.0), f, 120.0), Hold::Keep);
        assert_eq!(hold_verdict(f.bits(60.0), f, 120.0), Hold::Rewrite);
        assert_eq!(hold_verdict(f.bits(45.0), f, 120.0), Hold::Rewrite);
        assert_eq!(hold_verdict(0, f, 120.0), Hold::Strange);
        assert_eq!(hold_verdict(f.bits(144.0), f, 120.0), Hold::Strange);
        let s = form("s_f64");
        assert_eq!(hold_verdict(s.bits(60.0), s, 120.0), Hold::Rewrite);
        assert!(plausible(s.bits(45.0), s, 120.0));
        assert!(plausible(s.bits(120.0), s, 120.0));
        assert!(!plausible(0, s, 120.0));
    }

    const MAPS: &str = "\
6c00000000-6c00100000 r--p 00000000 fd:00 1   /data/app/x/lib/arm64/libunity.so
6c00100000-6c00200000 r-xp 00100000 fd:00 1   /data/app/x/lib/arm64/libunity.so
6c00200000-6c00210000 rw-p 00200000 fd:00 1   /data/app/x/lib/arm64/libunity.so
6c00210000-6c00220000 rw-p 00000000 00:00 0   [anon:.bss]
6c00220000-6c00230000 rw-p 00000000 00:00 0   [anon:scudo:primary]
6c00230000-6c00240000 rw-p 00000000 00:00 0   [anon:.bss]
6c00240000-6c00250000 rw-p 00000000 00:00 0   [anon:dalvik-main space]
6c00250000-6c00260000 rw-s 00000000 00:01 7   /memfd:framefx-status (deleted)
6c00260000-6c00270000 rw-p 00000000 00:00 0
6c00270000-6c00280000 rw-p 00000000 00:00 0   [stack]
7000000000-7000100000 r-xp 00000000 fd:00 2   /system/lib64/libc.so
";

    #[test]
    fn regions_and_lib_ranges_parse_correctly() {
        let r = regions(MAPS);
        let labels: Vec<&str> = r.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["libunity.so", "libunity.so .bss", "[anon:scudo:primary]", "[anon:.bss]", ""]);
        assert!(r[0].lib && r[1].lib && !r[2].lib && !r[3].lib && !r[4].lib);
        assert_eq!(r[0].file_off, 0x200000);
        assert_eq!(region_of(&r, 0x6c00215000).unwrap().label, "libunity.so .bss");
        assert!(region_of(&r, 0x6c00245000).is_none());

        let lr = lib_ranges(MAPS);
        assert_eq!(lr, vec![(0x6c00000000, 0x6c00210000)]);
        assert!(in_ranges(&lr, 0x6c00201234));
        assert!(in_ranges(&lr, 0xb400_006c_0020_1234));
        assert!(!in_ranges(&lr, 0x7000001000)); // system libc excluded when app .so exists
        assert!(!in_ranges(&lr, 0x6c00220000));

        // Non-executable data ranges exclude the r-xp text segment (0x6c00100000..0x6c00200000).
        let dr = lib_data_ranges(MAPS);
        assert_eq!(dr, vec![(0x6c00000000, 0x6c00100000), (0x6c00200000, 0x6c00210000)]);
        assert_eq!(containing_range(&dr, 0x6c00201234), Some((0x6c00200000, 0x6c00210000)));
        assert_eq!(containing_range(&dr, 0x6c00150000), None);
    }

    #[test]
    fn status_page_is_found_by_name() {
        assert_eq!(status_mapping(MAPS, "framefx-status"), vec![0x6c00250000]);
        assert!(status_mapping(MAPS, "nothing").is_empty());
    }

    #[test]
    fn single_pass_scan_matches_each_width_on_its_alignment() {
        let mut bytes = vec![0u8; 64];
        bytes[4..8].copy_from_slice(&60u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&(1.0f64 / 60.0).to_bits().to_le_bytes());
        // Misaligned double is not reported.
        bytes[36..44].copy_from_slice(&(1.0f64 / 60.0).to_bits().to_le_bytes());
        bytes[48..52].copy_from_slice(&60.0f32.to_bits().to_le_bytes());
        let i32f = Form::index("fps_i32").unwrap();
        let f32f = Form::index("fps_f32").unwrap();
        let sf64 = Form::index("s_f64").unwrap();
        let specs = [
            TargetSpec { id: 0, size: 4, bits: FORMS[i32f].bits(60.0) },
            TargetSpec { id: 1, size: 4, bits: FORMS[f32f].bits(60.0) },
            TargetSpec { id: 2, size: 8, bits: FORMS[sf64].bits(60.0) },
        ];
        let t = Targets::new(&specs);
        let mut hits = [Vec::new(), Vec::new(), Vec::new()];
        t.scan_with(0x1000, &bytes, |id, addr, _| hits[id].push(addr));
        assert_eq!(hits[0], vec![0x1004]);
        assert_eq!(hits[1], vec![0x1030]);
        assert_eq!(hits[2], vec![0x1010]);
    }

    #[test]
    fn inspect_slot_detects_vtable_pair_array_and_live_neighbors() {
        let lr = vec![(0x7b24_0000_00u64, 0x7b25_0000_00u64)];
        let mut buf = vec![0u8; 256];
        // Vtable pointer at offset 48 (80 bytes before slot at 128).
        buf[48..56].copy_from_slice(&0x7b24_573a_a8u64.to_le_bytes());
        buf[56..64].copy_from_slice(&0x7b32_881e_80u64.to_le_bytes());
        buf[120..128].copy_from_slice(&0x7ab8_b544_00u64.to_le_bytes());
        // Paired float 45.0f at offsets 128 and 132.
        let bits = 45.0f32.to_bits();
        buf[128..132].copy_from_slice(&bits.to_le_bytes());
        buf[132..136].copy_from_slice(&bits.to_le_bytes());

        let m0 = inspect_slot(0x7b32_8ac6_f0, &buf, 128, 4, bits as u64, false, &lr);
        let m1 = inspect_slot(0x7b32_8ac6_f4, &buf, 132, 4, bits as u64, false, &lr);
        assert!(m0.paired_first && !m0.in_array);
        assert!(m1.paired_second && !m1.in_array);
        assert_eq!(m0.vptr_key, (0x7b24_573a_a8u64 << 8) | 80);
        assert!(m0.ptr_count >= 3);
        assert!(structural_score(&m0, 2) >= HIGH_STRUCT_SCORE);
        assert!(structural_score(&m0, 2) > structural_score(&m1, 2));

        // 3 identical values in a row -> array, score 0.
        buf[136..140].copy_from_slice(&bits.to_le_bytes());
        let ma = inspect_slot(0x7b32_8ac6_f0, &buf, 128, 4, bits as u64, false, &lr);
        assert!(ma.in_array);
        assert_eq!(structural_score(&ma, 1), 0);

        // Live mutating neighbor detection via context_sig.
        let sig1 = context_sig(&buf, 128);
        buf[112..120].copy_from_slice(&12345u64.to_le_bytes());
        buf[144..152].copy_from_slice(&67890u64.to_le_bytes());
        let sig2 = context_sig(&buf, 128);
        assert_eq!(diff_sig(&sig1, &sig2), 2);
    }

    #[test]
    fn pointers_tagged_or_not() {
        assert!(is_pointer(0x0000_006b_ef78_21a8));
        assert!(is_pointer(0xb400_007e_d4a4_5810));
        assert!(!is_pointer(0x4030_aaaa_aaaa_aaab));
        assert!(!is_pointer(60));
        assert!(!is_pointer(0));
        assert!(!is_pointer(0xff00_007e_d4a4_5810));
    }

    fn neighbourhood(seed: u64, counter: u64) -> Vec<u8> {
        let mut w = vec![
            0x0000_006b_ef78_0000 + seed,
            1,
            0x4030_aaaa_aaaa_aaab,
            counter,
            7,
            0xb400_007e_0000_1000 + seed,
            60.0f32.to_bits() as u64,
            0,
        ];
        w.extend_from_slice(&[3, 5, 8, 13]);
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn fingerprint_keeps_fixed_words_and_survives_new_pointers() {
        let fp = fingerprint(&neighbourhood(0, 1), &neighbourhood(0, 2));
        assert_eq!(fp[0], Token::Pointer);
        assert_eq!(fp[3], Token::Any);
        assert_eq!(fp[5], Token::Pointer);
        assert_eq!(fp[4], Token::Exact(7));
        assert_eq!(score(&fp, &neighbourhood(0x5000, 99)), 1.0);
        let mut other = neighbourhood(0, 1);
        other[8..16].copy_from_slice(&2u64.to_le_bytes());
        other[32..40].copy_from_slice(&9u64.to_le_bytes());
        assert!(score(&fp, &other) < 0.9);
        assert_eq!(score(&[Token::Any, Token::Any], &[0u8; 16]), 0.0);
    }

    #[test]
    fn pick_needs_a_clear_winner() {
        assert_eq!(pick(vec![(1.0, 0xa), (0.5, 0xb)]), Pick::Sure(0xa));
        assert_eq!(pick(vec![(0.9, 0xa), (0.5, 0xb)]), Pick::Check(0xa));
        assert_eq!(pick(vec![(1.0, 0xa), (1.0, 0xb)]), Pick::None);
        assert_eq!(pick(vec![(0.7, 0xa)]), Pick::None);
        assert_eq!(pick(vec![]), Pick::None);
        assert_eq!(pick(vec![(0.5, 0xb), (0.99, 0xc)]), Pick::Sure(0xc));
    }

    #[test]
    fn framegen_text_decoding_and_keyword_matching() {
        // ASCII / UTF-8
        assert_eq!(decode_text(b"Enable Frame Generation\0junk"), Some("Enable Frame Generation".into()));
        assert_eq!(decode_text(b"hi\0"), None);
        assert_eq!(decode_text(&[0xa8, 0x3a, 0x57, 0x24, 0x7b, 0x00, 0x00, 0x00]), None);

        // UTF-16LE (as in wide-string engine settings)
        let utf16: Vec<u8> = "frame interpolation Disable/Enable0: Disable1: Enable\0"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        let decoded = decode_text(&utf16).unwrap();
        assert_eq!(decoded, "frame interpolation Disable/Enable0: Disable1: Enable");
        assert_eq!(is_framegen_desc(&decoded), 3);

        // Mixed CJK + ASCII UTF-16LE
        let mixed: Vec<u8> = "开启 frame interpolation 开关\0"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(is_framegen_desc(&decode_text(&mixed).unwrap()), 2);

        // Priority levels & non-matches
        assert_eq!(is_framegen_desc("Toggle FrameGen (0: off, 1: on)"), 3);
        assert_eq!(is_framegen_desc("Enable optical flow frame insertion"), 3);
        assert_eq!(is_framegen_desc("Frame extrapolation quality"), 2);
        assert_eq!(is_framegen_desc("Interpolate frames between renders"), 1);
        assert_eq!(is_framegen_desc("Caps FPS to the given value."), 0);
        assert_eq!(is_framegen_desc("Enable wireframe debug overlay"), 0);
    }

    #[test]
    fn framegen_toggle_patterns_thresholds_and_hold_verdict() {
        // Bool shadowed pair [true, true] = 0x0101, even with non-zero high 32-bit neighbor.
        assert_eq!(toggle_on_size(0xdead_beef_0000_0101, true), Some((4, 0x101)));
        assert_eq!(toggle_on_size(0xdead_beef_0000_0101, false), Some((4, 0x101)));
        // Paired i32 [1, 1] or [2, 2] uses 8 bytes only when setting slots are paired.
        assert_eq!(toggle_on_size(0x0000_0001_0000_0001, true), Some((8, 0x0000_0001_0000_0001)));
        assert_eq!(toggle_on_size(0x0000_0001_0000_0001, false), Some((4, 1)));
        assert_eq!(toggle_on_size(0x0000_0002_0000_0002, true), Some((8, 0x0000_0002_0000_0002)));
        // Scalar toggle
        assert_eq!(toggle_on_size(1, false), Some((4, 1)));
        assert_eq!(toggle_on_size(0, true), None);
        assert_eq!(toggle_on_size(45.0f32.to_bits() as u64, true), None);

        // Causal drop / restore thresholds
        assert!(fg_disabled_drop(44.8, 90.0, 45.0));
        assert!(fg_disabled_drop(36.0, 90.0, 45.0));
        assert!(!fg_disabled_drop(85.0, 90.0, 45.0));
        assert!(!fg_disabled_drop(0.0, 90.0, 45.0));
        assert!(fg_restored_rise(89.5, 44.8, 90.0));
        assert!(!fg_restored_rise(46.0, 44.8, 90.0));

        // Hold verdicts for disabled framegen slot
        assert_eq!(fg_hold_verdict(0, 4, 0x101), Hold::Keep);
        assert_eq!(fg_hold_verdict(0x101, 4, 0x101), Hold::Rewrite);
        assert_eq!(fg_hold_verdict(1, 4, 0x101), Hold::Rewrite);
        assert_eq!(fg_hold_verdict(0x100, 4, 0x101), Hold::Rewrite);
        assert_eq!(fg_hold_verdict(0x7b32_0000, 4, 0x101), Hold::Strange);
        assert_eq!(fg_hold_verdict(0, 8, 0x0000_0001_0000_0001), Hold::Keep);
        assert_eq!(fg_hold_verdict(0x0000_0001_0000_0001, 8, 0x0000_0001_0000_0001), Hold::Rewrite);
        assert_eq!(fg_hold_verdict(1, 8, 0x0000_0001_0000_0001), Hold::Rewrite);
        assert_eq!(fg_hold_verdict(999, 8, 0x0000_0001_0000_0001), Hold::Strange);
    }

    #[test]
    fn cache_round_trips_and_stays_backwards_compatible() {
        let fg = FrameGenCache {
            size: 4,
            on_bits: 0x101,
            lib: false,
            label: "[anon:libc_malloc]".into(),
            file_off: 0,
            offset: 0x12340,
            fingerprint: vec![Token::Pointer, Token::Any, Token::Exact(0x42)],
            delta: 128,
        };
        let found = Found {
            version: "182544764".into(),
            form: Form::index("s_f64").unwrap(),
            lib: false,
            label: "[anon:mimalloc]".into(),
            file_off: 0,
            offset: 0xa5638,
            fingerprint: vec![Token::Pointer, Token::Any, Token::Exact(0x4030_aaaa_aaaa_aaab), Token::Exact(0)],
            delta: 128,
            cap_div: 2,
            fg: Some(fg),
        };
        let text = found.to_text();
        assert_eq!(Found::from_text(&text), Some(found.clone()));
        // Legacy cache files without cap_div= or fg_* still load cleanly.
        let legacy = text.lines().filter(|l| !l.starts_with("cap_div=") && !l.starts_with("fg_")).collect::<Vec<_>>().join("\n") + "\n";
        let loaded_legacy = Found::from_text(&legacy).unwrap();
        assert_eq!(loaded_legacy.cap_div, 1);
        assert_eq!(loaded_legacy.fg, None);
        assert_eq!(Found::from_text(&text.replace("form=s_f64", "form=bogus")), None);
        assert_eq!(Found::from_text(&text.replace("fingerprint=p,", "fingerprint=zz,")), None);
        assert_eq!(Found::from_text(&text.replace("fg_size=4", "fg_size=3")), None);
        assert_eq!(Found::from_text("version=1\n"), None);

        let lib = Found {
            version: "1".into(),
            form: 0,
            lib: true,
            label: "libunity.so .bss".into(),
            file_off: 0x200000,
            offset: 0x2d1c,
            fingerprint: vec![],
            delta: 0,
            cap_div: 1,
            fg: None,
        };
        let lib_text = lib.to_text();
        assert!(!lib_text.contains("fg_"));
        assert_eq!(Found::from_text(&lib_text), Some(lib));
    }

    #[test]
    fn start_property_is_parsed_strictly() {
        assert_eq!(
            parse_start("16438 120 182544764 com.kurogame.wutheringwaves.global"),
            Some(Start { pid: 16438, target: 120.0, version: "182544764".into(), package: "com.kurogame.wutheringwaves.global".into() })
        );
        assert_eq!(parse_start(""), None);
        assert_eq!(parse_start("0 120 1 a.b"), None);
        assert_eq!(parse_start("12 500 1 a.b"), None);
        assert_eq!(parse_start("12 120 1 ../etc"), None);
        assert_eq!(parse_start("12 120 1 a/b"), None);
        assert_eq!(parse_start("12 120 1"), None);
        assert_eq!(parse_start("x 120 1 a.b"), None);
    }
}
