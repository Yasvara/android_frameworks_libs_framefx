// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! lepasd: holds an armed game's frame rate at the panel's rate, from outside the game.
//!
//! The broker sets `yasvara.lepas.start` to "<pid> <target> <version> <package>" for each process
//! of a game whose Lepas section is on; init starts this daemon on the first one and it then
//! blocks on the property for the next, so it costs nothing between games.
//!
//! Per game process:
//!   1. Times the game by the present count framefx keeps in the game's status page (ignoring
//!      helper subprocesses that never present).
//!   2. Waits for a steady cap (30/45/60/90...).
//!   3. Finds the slot holding it: the cached slot of this version (library data by offset, heap
//!      by inline fingerprint scoring during a parallel scan), else learns it once in a single
//!      parallel pass across all 13 forms (plus `cap / 2` for built-in 2x framegen), page-batched
//!      stability verification, generic structural ranking, and single-slot causal probing.
//!   4. Writes the target and holds it while the game writes any known cap back.
//!
//! Memory access is process_vm_readv/writev on pages the game has made resident: no
//! /proc/pid/mem (games watch it), no ptrace attach, nothing mapped into the game. Every decision
//! lives in core.rs, tested on the host.

mod core;

use crate::core::{
    cap_moved, containing_range, context_sig, decode_text, diff_sig, fg_disabled_drop, fg_hold_verdict,
    fg_restored_rise, fingerprint, half_cap, hold_verdict, inspect_slot, is_framegen_desc, is_pointer,
    lib_data_ranges, lib_ranges, lifted, parse_start, pick, plan, plausible, poison_line, poisoned,
    probe_rate, probe_recovered, probe_slowed, region_of, regions, reverted, score, status_mapping,
    strip_tag, toggle_on_size, Bucket, Form, Found, FrameGenCache, Hold, Pick, Region, SlotMeta, Steady,
    TargetSpec, Targets, Token, FORMS,
};
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::fs::{self, File};
use std::io::Write;
use std::os::raw::{c_char, c_int};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, sleep};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const START_PROPERTY: &str = "yasvara.lepas.start";
const DIR: &str = "/data/misc/lepas";
const STATUS_MAPPING: &str = "framefx-status";
const PRESENT_COUNT_OFFSET: u64 = 352; // framefx/channel.h kPresentCountOffset
const STATUS_MAGIC: u32 = 0x5358_4646; // "FFXS"
const MAX_SET: usize = 300; // larger raw sets still contribute structurally ranked heap objects
const LEARN_BUDGET: Duration = Duration::from_secs(300);
const FINGERPRINT_SPAN: u64 = 128; // bytes on each side of the slot
const GAME_LOG_MAX: u64 = 32 * 1024;
const SCAN_THREADS: usize = 4;
const SCAN_CHUNK_PAGES: usize = 256; // 1 MiB per work slice

// ---------------------------------------------------------------- bionic

#[repr(C)]
struct IoVec {
    base: *mut u8,
    len: usize,
}

#[repr(C)]
struct Timespec {
    sec: i64,
    nsec: i64,
}

enum PropInfo {}

extern "C" {
    fn process_vm_readv(pid: i32, l: *const IoVec, ln: u64, r: *const IoVec, rn: u64, f: u64) -> isize;
    fn process_vm_writev(pid: i32, l: *const IoVec, ln: u64, r: *const IoVec, rn: u64, f: u64) -> isize;
    fn __system_property_find(name: *const c_char) -> *const PropInfo;
    fn __system_property_serial(pi: *const PropInfo) -> u32;
    fn __system_property_wait(pi: *const PropInfo, old: u32, new: *mut u32, timeout: *const Timespec) -> bool;
    fn __system_property_get(name: *const c_char, value: *mut c_char) -> c_int;
    fn __android_log_write(prio: c_int, tag: *const c_char, text: *const c_char) -> c_int;
}

fn log(text: &str) {
    if let Ok(t) = CString::new(text) {
        // SAFETY: both pointers are NUL-terminated strings that outlive the call.
        unsafe { __android_log_write(4, c"lepasd".as_ptr(), t.as_ptr()) };
    }
}

fn property(name: &str) -> String {
    let name = CString::new(name).unwrap();
    let mut buf = [0 as c_char; 92];
    // SAFETY: `buf` is PROP_VALUE_MAX (92) bytes, the most __system_property_get writes, and it
    // NUL-terminates what it writes.
    unsafe {
        __system_property_get(name.as_ptr(), buf.as_mut_ptr());
        CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
    }
}

// ---------------------------------------------------------------- memory

struct Mem {
    pid: i32,
    pagemap: File,
}

impl Mem {
    fn open(pid: i32) -> Option<Mem> {
        Some(Mem { pid, pagemap: File::open(format!("/proc/{pid}/pagemap")).ok()? })
    }

    /// Reads resident flags for `n` pages from `addr` into reusable `pm_buf` (`n * 8` bytes).
    fn resident_into(&self, addr: u64, n: usize, pm_buf: &mut [u8], out: &mut [bool]) {
        let bytes = &mut pm_buf[..n * 8];
        let got = self.pagemap.read_at(bytes, (addr / 4096) * 8).unwrap_or(0);
        for i in 0..n {
            out[i] = i * 8 + 8 <= got
                && (u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()) >> 62) != 0;
        }
    }

    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        if buf.is_empty() {
            return true;
        }
        let first = addr / 4096;
        let last = (addr + buf.len() as u64 - 1) / 4096;
        let n = (last - first + 1) as usize;
        let mut pm = vec![0u8; n * 8];
        let mut res = vec![false; n];
        self.resident_into(first * 4096, n, &mut pm, &mut res);
        !res.contains(&false) && self.read_raw(addr, buf)
    }

    fn read_raw(&self, addr: u64, buf: &mut [u8]) -> bool {
        let l = IoVec { base: buf.as_mut_ptr(), len: buf.len() };
        let r = IoVec { base: addr as *mut u8, len: buf.len() };
        // SAFETY: the local iovec covers `buf`, which the kernel fills; the remote one names the
        // other process's memory, which the kernel checks and never dereferences here.
        unsafe { process_vm_readv(self.pid, &l, 1, &r, 1, 0) == buf.len() as isize }
    }

    fn word(&self, addr: u64, size: usize) -> Option<u64> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b[..size]).then(|| u64::from_le_bytes(b))
    }

    fn put(&self, addr: u64, size: usize, v: u64) -> bool {
        let bytes = v.to_le_bytes();
        let l = IoVec { base: bytes.as_ptr() as *mut u8, len: size };
        let r = IoVec { base: addr as *mut u8, len: size };
        // SAFETY: the local iovec covers `size` (<= 8) bytes of `bytes`; the remote one names the
        // other process's memory, which the kernel checks and never dereferences here.
        unsafe { process_vm_writev(self.pid, &l, 1, &r, 1, 0) == size as isize }
    }

    fn maps_text(&self) -> String {
        fs::read_to_string(format!("/proc/{}/maps", self.pid)).unwrap_or_default()
    }

    fn regions(&self) -> Vec<Region> {
        regions(&self.maps_text())
    }

    /// Parallel work-stealing scan over resident pages of `regs`, invoking `visit` per contiguous
    /// resident run on each worker thread and merging the thread-local accumulators.
    fn scan_parallel<T, Init, Visit>(&self, regs: &[&Region], init: Init, visit: Visit) -> Vec<T>
    where
        T: Send,
        Init: Fn() -> T + Sync,
        Visit: Fn(&mut T, u64, &[u8], bool) + Sync,
    {
        const SLICE_BYTES: u64 = SCAN_CHUNK_PAGES as u64 * 4096;
        let mut slices: Vec<(u64, u64, bool)> = Vec::new();
        for r in regs {
            let mut at = r.lo;
            while at < r.hi {
                let end = (at + SLICE_BYTES).min(r.hi);
                slices.push((at, end, r.lib));
                at = end;
            }
        }
        if slices.is_empty() {
            return Vec::new();
        }
        let cursor = AtomicUsize::new(0);
        let workers = SCAN_THREADS.min(slices.len());
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                handles.push(s.spawn(|| {
                    let mut acc = init();
                    let mut pm_buf = vec![0u8; SCAN_CHUNK_PAGES * 8];
                    let mut res = vec![false; SCAN_CHUNK_PAGES];
                    let mut data = vec![0u8; SCAN_CHUNK_PAGES * 4096];
                    loop {
                        let idx = cursor.fetch_add(1, Ordering::Relaxed);
                        if idx >= slices.len() {
                            break;
                        }
                        let (lo, hi, lib) = slices[idx];
                        let n = ((hi - lo) / 4096) as usize;
                        self.resident_into(lo, n, &mut pm_buf, &mut res);
                        let mut i = 0;
                        while i < n {
                            if !res[i] {
                                i += 1;
                                continue;
                            }
                            let mut j = i;
                            while j < n && res[j] {
                                j += 1;
                            }
                            let base = lo + i as u64 * 4096;
                            let run = &mut data[..(j - i) * 4096];
                            if self.read_raw(base, run) {
                                visit(&mut acc, base, run, lib);
                            }
                            i = j;
                        }
                    }
                    acc
                }));
            }
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        })
    }

    /// One parallel pass over `regs` collecting `SlotMeta` per target spec directly in L1 cache.
    fn scan_meta(&self, regs: &[&Region], specs: &[TargetSpec], so_ranges: &[(u64, u64)]) -> Vec<Vec<SlotMeta>> {
        let targets = Targets::new(specs);
        let parts = self.scan_parallel(
            regs,
            || vec![Vec::<SlotMeta>::new(); specs.len()],
            |hits, base, run, lib| {
                targets.scan_with(base, run, |id, addr, off| {
                    let sp = specs[id];
                    hits[id].push(inspect_slot(addr, run, off, sp.size, sp.bits, lib, so_ranges));
                });
            },
        );
        let mut merged = vec![Vec::<SlotMeta>::new(); specs.len()];
        for part in parts {
            for (dst, mut src) in merged.iter_mut().zip(part) {
                dst.append(&mut src);
            }
        }
        merged
    }

    /// Page-batched stability filter: re-reads only the pages containing hits (in contiguous runs
    /// up to 1 MiB), verifies every slot still holds its exact bits, and records `live_neighbors`.
    fn verify_stable(&self, specs: &[TargetSpec], candidates: Vec<Vec<SlotMeta>>) -> Vec<Vec<SlotMeta>> {
        // Flatten into `(addr, spec_id, meta)` sorted by address so every page is read at most once.
        let total: usize = candidates.iter().map(Vec::len).sum();
        let mut items: Vec<(u64, usize, SlotMeta)> = Vec::with_capacity(total);
        for (id, list) in candidates.into_iter().enumerate() {
            for m in list {
                items.push((m.addr, id, m));
            }
        }
        items.sort_unstable_by_key(|x| x.0);

        let mut kept = vec![Vec::<SlotMeta>::new(); specs.len()];
        let mut pm_buf = vec![0u8; SCAN_CHUNK_PAGES * 8];
        let mut res = vec![false; SCAN_CHUNK_PAGES];
        let mut buf = vec![0u8; SCAN_CHUNK_PAGES * 4096];

        let mut idx = 0;
        while idx < items.len() {
            let start_page = items[idx].0 & !0xfff;
            let mut end_page = start_page + 4096;
            let mut end_idx = idx + 1;
            // Coalesce nearby hit pages (up to 64 KiB gap, 1 MiB max run) into one read.
            while end_idx < items.len() {
                let p = items[end_idx].0 & !0xfff;
                let next_end = p + 4096;
                if next_end - start_page > (SCAN_CHUNK_PAGES as u64) * 4096 || p > end_page + 16 * 4096 {
                    break;
                }
                end_page = end_page.max(next_end);
                end_idx += 1;
            }
            let npages = ((end_page - start_page) / 4096) as usize;
            self.resident_into(start_page, npages, &mut pm_buf, &mut res);

            // Read each contiguous resident sub-run and check all items inside it.
            let mut pi = 0;
            let mut cur = idx;
            while pi < npages && cur < end_idx {
                if !res[pi] {
                    let skip_end = start_page + (pi as u64 + 1) * 4096;
                    while cur < end_idx && items[cur].0 < skip_end {
                        cur += 1;
                    }
                    pi += 1;
                    continue;
                }
                let mut pj = pi;
                while pj < npages && res[pj] {
                    pj += 1;
                }
                let run_lo = start_page + pi as u64 * 4096;
                let run_hi = start_page + pj as u64 * 4096;
                let run = &mut buf[..(pj - pi) * 4096];
                if self.read_raw(run_lo, run) {
                    while cur < end_idx && items[cur].0 < run_hi {
                        let (addr, id, mut m) = items[cur];
                        cur += 1;
                        if addr < run_lo {
                            continue;
                        }
                        let off = (addr - run_lo) as usize;
                        let sp = specs[id];
                        if off + sp.size > run.len() {
                            continue;
                        }
                        let val = if sp.size == 4 {
                            u32::from_le_bytes(run[off..off + 4].try_into().unwrap()) as u64
                        } else {
                            u64::from_le_bytes(run[off..off + 8].try_into().unwrap())
                        };
                        if val == sp.bits {
                            if !m.lib {
                                let sig2 = context_sig(run, off);
                                m.live_neighbors = diff_sig(&m.ctx_sig, &sig2);
                            }
                            kept[id].push(m);
                        }
                    }
                } else {
                    while cur < end_idx && items[cur].0 < run_hi {
                        cur += 1;
                    }
                }
                pi = pj;
            }
            idx = end_idx;
        }
        kept
    }
}

fn alive(pid: i32) -> bool {
    fs::metadata(format!("/proc/{pid}")).is_ok()
}

enum Learned {
    /// `(form, slot_cap, addr)`
    Slot(usize, f64, u64),
    /// The game left the cap being learned; its primary slots at that cap, per form.
    Moved(Vec<(usize, Vec<u64>)>),
    Nothing,
}

// ---------------------------------------------------------------- timing

/// The game's present count, read from its framefx status page.
struct Meter {
    addr: u64,
}

impl Meter {
    fn find(mem: &Mem) -> Option<Meter> {
        let maps = fs::read_to_string(format!("/proc/{}/maps", mem.pid)).ok()?;
        status_mapping(&maps, STATUS_MAPPING).into_iter().find_map(|lo| {
            let mut magic = [0u8; 4];
            (mem.read_raw(lo, &mut magic) && u32::from_le_bytes(magic) == STATUS_MAGIC)
                .then_some(Meter { addr: lo + PRESENT_COUNT_OFFSET })
        })
    }

    fn count(&self, mem: &Mem) -> Option<u32> {
        let mut b = [0u8; 4];
        mem.read_raw(self.addr, &mut b).then(|| u32::from_le_bytes(b))
    }

    /// Presents per second over `ms`.
    fn fps(&self, mem: &Mem, ms: u64) -> f64 {
        let (Some(a), t) = (self.count(mem), Instant::now()) else { return 0.0 };
        sleep(Duration::from_millis(ms));
        let Some(b) = self.count(mem) else { return 0.0 };
        b.wrapping_sub(a) as f64 / t.elapsed().as_secs_f64()
    }
}

// ---------------------------------------------------------------- state on disk

/// Logcat and `<pkg>.log`: the ring buffer rolls over long before anyone reads a game's run.
fn game_log(pkg: &str, text: &str) {
    log(&format!("{pkg}: {text}"));
    let path = format!("{DIR}/{pkg}.log");
    if fs::metadata(&path).map(|m| m.len() > GAME_LOG_MAX).unwrap_or(false) {
        let _ = fs::rename(&path, format!("{path}.old"));
    }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
        let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let _ = writeln!(f, "{t} {text}");
    }
}

fn write_state(pkg: &str, text: &str) {
    let _ = fs::write(format!("{DIR}/{pkg}.state"), text);
}

/// Test writes in flight, `(addr, orig)` each, so a daemon that dies mid-test is undone by its
/// next start.
fn pending_put(mem: &Mem, size: usize, slots: &[(u64, u64)], value: u64) {
    let text: String = slots.iter().map(|(a, o)| format!("{a:x} {size} {o}\n")).collect();
    let _ = fs::write(format!("{DIR}/pending.{}", mem.pid), text);
    for &(a, _) in slots {
        mem.put(a, size, value);
    }
}

fn pending_undo(mem: &Mem, size: usize, slots: &[(u64, u64)]) {
    for &(a, o) in slots {
        mem.put(a, size, o);
    }
    let _ = fs::remove_file(format!("{DIR}/pending.{}", mem.pid));
}

fn pending_keep(mem: &Mem) {
    let _ = fs::remove_file(format!("{DIR}/pending.{}", mem.pid));
}

fn recover() {
    let Ok(dir) = fs::read_dir(DIR) else { return };
    for e in dir.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(pid) = name.strip_prefix("pending.").and_then(|p| p.parse::<i32>().ok()) else { continue };
        let text = fs::read_to_string(e.path()).unwrap_or_default();
        if let Some(mem) = Mem::open(pid) {
            let mut restored = 0;
            for line in text.lines() {
                let f: Vec<&str> = line.split_whitespace().collect();
                if let [a, s, o] = &f[..] {
                    if let (Ok(a), Ok(s), Ok(o)) = (u64::from_str_radix(a, 16), s.parse(), o.parse()) {
                        restored += mem.put(a, s, o) as u32;
                    }
                }
            }
            log(&format!("pid {pid}: restored {restored} slots after an interrupted test"));
        }
        let _ = fs::remove_file(e.path());
    }
}

fn cache_load(pkg: &str) -> Option<Found> {
    Found::from_text(&fs::read_to_string(format!("{DIR}/{pkg}.cache")).ok()?)
}

fn cache_save(pkg: &str, found: &Found) {
    let _ = fs::write(format!("{DIR}/{pkg}.cache"), found.to_text());
}

// ---------------------------------------------------------------- one game

struct Game {
    pkg: String,
    version: String,
    target: f64,
    mem: Mem,
    meter: Meter,
}

impl Game {
    /// Writes a clean lower cap into one slot and watches for a matching drop, then restores it.
    /// Returns `(hit, observed_fps)` so the learn loop reuses `observed_fps` without sleeping again.
    fn probe(&self, form: &Form, addr: u64, measured_cap: f64, slot_cap: f64) -> (bool, f64) {
        let bits = form.bits(slot_cap);
        if self.mem.word(addr, form.size()) != Some(bits) {
            return (false, measured_cap);
        }
        let low = probe_rate(slot_cap);
        let slot = [(addr, bits)];
        pending_put(&self.mem, form.size(), &slot, form.bits(low));
        sleep(Duration::from_millis(90));
        let first = self.meter.fps(&self.mem, 240);
        let hit = probe_slowed(first, measured_cap, slot_cap, low)
            && probe_slowed(self.meter.fps(&self.mem, 400), measured_cap, slot_cap, low);
        pending_undo(&self.mem, form.size(), &slot);
        if !hit {
            return (false, first);
        }
        sleep(Duration::from_millis(220));
        let rec = self.meter.fps(&self.mem, 450);
        (probe_recovered(rec, measured_cap), rec)
    }

    /// Writes the target into a candidate slot, verifies lift AND causal revert (restoring `orig`
    /// must drop fps back down, defeating loading/scene-transition false positives), then re-applies.
    fn lifts(&self, form: &Form, addr: u64, measured_cap: f64, expected_slot_cap: Option<f64>) -> bool {
        let Some(orig) = self.mem.word(addr, form.size()) else { return false };
        if let Some(sc) = expected_slot_cap {
            if orig != form.bits(sc) {
                return false;
            }
        }
        let slot = [(addr, orig)];
        pending_put(&self.mem, form.size(), &slot, form.bits(self.target));
        sleep(Duration::from_millis(350));
        let fps_up = self.meter.fps(&self.mem, 750);
        if !lifted(fps_up, measured_cap) {
            pending_undo(&self.mem, form.size(), &slot);
            game_log(&self.pkg, &format!("{} slot {addr:x}: target {} -> {fps_up:.1} fps, no lift", form.name, self.target));
            return false;
        }
        // Causal check: restore original value and verify fps drops back near `measured_cap`.
        self.mem.put(addr, form.size(), orig);
        sleep(Duration::from_millis(300));
        let fps_back = self.meter.fps(&self.mem, 600);
        if !reverted(fps_back, fps_up, measured_cap) {
            pending_undo(&self.mem, form.size(), &slot);
            game_log(
                &self.pkg,
                &format!(
                    "{} slot {addr:x}: target {} -> {fps_up:.1} fps, undo -> {fps_back:.1} fps (false lift)",
                    form.name, self.target
                ),
            );
            return false;
        }
        self.mem.put(addr, form.size(), form.bits(self.target));
        pending_keep(&self.mem);
        game_log(
            &self.pkg,
            &format!("{} slot {addr:x}: target {} -> {fps_up:.1} fps (undo -> {fps_back:.1})", form.name, self.target),
        );
        true
    }

    fn read_neighbourhood(&self, addr: u64) -> Option<(u64, usize, Vec<u8>)> {
        let start = addr.checked_sub(FINGERPRINT_SPAN)? & !7;
        let delta = (addr - start) as usize;
        let mut buf = vec![0u8; 2 * FINGERPRINT_SPAN as usize + 8];
        self.mem.read(start, &mut buf).then_some((start, delta, buf))
    }

    fn finalize_fp(a: &[u8], b: &[u8], delta: usize, size: usize) -> (Vec<Token>, usize) {
        let mut fp = fingerprint(a, b);
        // The slot's own word holds our written value (`target` or `0`), whereas on the next
        // launch it will hold the game's original value; wildcard it so relocation scores 1.0.
        let first = delta / 8;
        if let Some(slot) = fp.get_mut(first) {
            *slot = Token::Any;
        }
        if (delta % 8) + size > 8 {
            if let Some(next) = fp.get_mut(first + 1) {
                *next = Token::Any;
            }
        }
        (fp, delta)
    }

    /// Captures neighbourhoods of `addr` and an optional second slot in ONE shared 500 ms wait.
    fn fingerprint_pair(
        &self,
        addr: u64,
        size: usize,
        other: Option<(u64, usize)>,
    ) -> (Option<(Vec<Token>, usize)>, Option<(Vec<Token>, usize)>) {
        let a1 = self.read_neighbourhood(addr);
        let a2 = other.and_then(|(oa, _)| self.read_neighbourhood(oa));
        if a1.is_none() && a2.is_none() {
            return (None, None);
        }
        sleep(Duration::from_millis(500));
        let fp1 = a1.and_then(|(start, delta, a)| {
            let mut b = vec![0u8; a.len()];
            self.mem.read(start, &mut b).then(|| Self::finalize_fp(&a, &b, delta, size))
        });
        let fp2 = match (a2, other) {
            (Some((start, delta, a)), Some((_, osize))) => {
                let mut b = vec![0u8; a.len()];
                self.mem.read(start, &mut b).then(|| Self::finalize_fp(&a, &b, delta, osize))
            }
            _ => None,
        };
        (fp1, fp2)
    }

    fn score_fp_at(&self, fp: &[Token], delta: usize, run: &[u8], off: usize, addr: u64) -> f64 {
        let fp_bytes = fp.len() * 8;
        if off >= delta && off - delta + fp_bytes <= run.len() {
            let start = off - delta;
            score(fp, &run[start..start + fp_bytes])
        } else {
            let mut b = vec![0u8; fp_bytes];
            if addr < delta as u64 || !self.mem.read(addr - delta as u64, &mut b) {
                return 0.0;
            }
            score(fp, &b)
        }
    }

    fn read_string_at(&self, ptr: u64) -> Option<String> {
        if !is_pointer(ptr) {
            return None;
        }
        let raw = strip_tag(ptr);
        let page_rem = (4096 - (raw & 0xfff)) as usize;
        let len = if page_rem >= 16 { page_rem.min(160) } else { 160 };
        let mut buf = vec![0u8; len];
        if !self.mem.read(raw, &mut buf) {
            return None;
        }
        decode_text(&buf)
    }

    /// Relocates a cached built-in frame-generation toggle on its own.
    fn relocate_fg(&self, fg: &FrameGenCache, regs: &[Region], sure_only: bool) -> Option<(u64, usize, u64)> {
        if fg.lib {
            let r = regs.iter().find(|r| r.label == fg.label && r.file_off == fg.file_off)?;
            let addr = r.lo + fg.offset;
            return (self.mem.word(addr, fg.size) == Some(fg.on_bits)).then_some((addr, fg.size, fg.on_bits));
        }
        let specs = [TargetSpec { id: 0, size: fg.size, bits: fg.on_bits }];
        let targets = Targets::new(&specs);
        for same in [true, false] {
            let pool: Vec<&Region> = regs.iter().filter(|r| !same || r.label == fg.label).collect();
            let parts = self.mem.scan_parallel(
                &pool,
                Vec::<(f64, u64)>::new,
                |out, base, run, _| {
                    targets.scan_with(base, run, |_, addr, off| {
                        let s = self.score_fp_at(&fg.fingerprint, fg.delta, run, off, addr);
                        if s >= 0.75 {
                            out.push((s, addr));
                        }
                    });
                },
            );
            let scored: Vec<(f64, u64)> = parts.into_iter().flatten().collect();
            match pick(scored) {
                Pick::Sure(addr) => return Some((addr, fg.size, fg.on_bits)),
                Pick::Check(addr) if !sure_only => return Some((addr, fg.size, fg.on_bits)),
                _ => {}
            }
        }
        None
    }

    /// Relocates the cached cap slot, and if a cached built-in framegen toggle shares the same
    /// heap region, relocates both in ONE parallel pass (halving warm-launch scan time).
    fn relocate(&self, found: &Found, cap: f64) -> (Option<(usize, f64, u64)>, Option<(u64, usize, u64)>) {
        let form = &FORMS[found.form];
        let regs = self.mem.regions();
        let expected_cap = if found.cap_div == 2 { half_cap(cap).unwrap_or(cap) } else { cap };
        if found.lib {
            let cap_hit = regs
                .iter()
                .find(|r| r.label == found.label && r.file_off == found.file_off)
                .and_then(|r| {
                    let addr = r.lo + found.offset;
                    plausible(self.mem.word(addr, form.size())?, form, self.target)
                        .then_some((found.form, expected_cap, addr))
                });
            let fg_hit = match (cap_hit, &found.fg) {
                (Some((_, sc, _)), Some(fg)) if (sc * 2.0 - cap).abs() < 1.0 => {
                    self.relocate_fg(fg, &regs, false)
                }
                _ => None,
            };
            return (cap_hit, fg_hit);
        }

        // Scan for `expected_cap`, its alternate (`cap` vs `cap/2`), and co-located `fg` in one pass.
        let mut specs = vec![TargetSpec { id: 0, size: form.size(), bits: form.bits(expected_cap) }];
        let mut cap_for_id = vec![expected_cap];
        let alt = if found.cap_div == 2 { Some(cap) } else { half_cap(cap) };
        if let Some(ac) = alt {
            if form.bits(ac) != specs[0].bits {
                specs.push(TargetSpec { id: 1, size: form.size(), bits: form.bits(ac) });
                cap_for_id.push(ac);
            }
        }
        let fg_inline = found.fg.as_ref().filter(|fg| !fg.lib && fg.label == found.label);
        if let Some(fg) = fg_inline {
            specs.push(TargetSpec { id: 2, size: fg.size, bits: fg.on_bits });
        }
        let targets = Targets::new(&specs);

        let mut cap_hit: Option<(usize, f64, u64)> = None;
        let mut fg_hit: Option<(u64, usize, u64)> = None;

        for same in [true, false] {
            let pool: Vec<&Region> = regs.iter().filter(|r| !same || r.label == found.label).collect();
            let parts = self.mem.scan_parallel(
                &pool,
                || (Vec::<(f64, u64, usize)>::new(), Vec::<(f64, u64)>::new()),
                |(cap_out, fg_out), base, run, _| {
                    targets.scan_with(base, run, |id, addr, off| {
                        if id == 2 {
                            if let Some(fg) = fg_inline {
                                let s = self.score_fp_at(&fg.fingerprint, fg.delta, run, off, addr);
                                if s >= 0.75 {
                                    fg_out.push((s, addr));
                                }
                            }
                        } else {
                            let s = self.score_fp_at(&found.fingerprint, found.delta, run, off, addr);
                            if s >= 0.75 {
                                cap_out.push((s, addr, id));
                            }
                        }
                    });
                },
            );
            let mut scored: Vec<(f64, u64)> = Vec::new();
            let mut cap_of: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
            let mut fg_scored: Vec<(f64, u64)> = Vec::new();
            for (cp, fp) in parts {
                for (s, addr, id) in cp {
                    scored.push((s, addr));
                    cap_of.insert(addr, cap_for_id[id]);
                }
                fg_scored.extend(fp);
            }
            if fg_hit.is_none() {
                if let Some(fg) = fg_inline {
                    match pick(fg_scored) {
                        Pick::Sure(a) | Pick::Check(a) => fg_hit = Some((a, fg.size, fg.on_bits)),
                        Pick::None => {}
                    }
                }
            }
            match pick(scored) {
                Pick::Sure(addr) => {
                    let sc = *cap_of.get(&addr).unwrap_or(&expected_cap);
                    cap_hit = Some((found.form, sc, addr));
                    break;
                }
                Pick::Check(addr) => {
                    let sc = *cap_of.get(&addr).unwrap_or(&expected_cap);
                    if self.probe(form, addr, cap, sc).0 {
                        cap_hit = Some((found.form, sc, addr));
                        break;
                    }
                }
                Pick::None => {}
            }
        }
        if let (Some((_, sc, _)), Some(fg)) = (cap_hit, &found.fg) {
            if (sc * 2.0 - cap).abs() < 1.0 && fg_hit.is_none() {
                fg_hit = self.relocate_fg(fg, &regs, false);
            }
        }
        (cap_hit, fg_hit)
    }

    /// Disables a relocated cached built-in framegen toggle and confirms presented FPS drops near
    /// `slot_cap`; restores it if no drop occurred.
    fn disable_fg_and_verify(&self, fg_addr: u64, fg_size: usize, fg_on: u64, cap: f64, slot_cap: f64) -> bool {
        if self.mem.word(fg_addr, fg_size) != Some(fg_on) {
            return false;
        }
        let slot = [(fg_addr, fg_on)];
        pending_put(&self.mem, fg_size, &slot, 0);
        sleep(Duration::from_millis(400));
        let mut fps_off = self.meter.fps(&self.mem, 550);
        if !fg_disabled_drop(fps_off, cap, slot_cap) {
            fps_off = self.meter.fps(&self.mem, 550);
        }
        if fg_disabled_drop(fps_off, cap, slot_cap) {
            pending_keep(&self.mem);
            game_log(
                &self.pkg,
                &format!("disabled cached built-in framegen slot {fg_addr:x}: {cap:.1} -> {fps_off:.1} fps"),
            );
            true
        } else {
            pending_undo(&self.mem, fg_size, &slot);
            game_log(
                &self.pkg,
                &format!("cached built-in framegen slot {fg_addr:x} did not drop fps ({fps_off:.1}); restored"),
            );
            false
        }
    }

    /// Generically discovers and disables a built-in 2x frame-generation toggle when `cap_div == 2`:
    ///   1. Infers the setting object's `(vptr, val_off, str_offs)` dynamically from `cap_addr`.
    ///   2. Scans writable memory in parallel for sibling setting objects whose value is an active
    ///      toggle and whose description matches generic frame-interpolation / generation terms.
    ///   3. Holds `cap_addr` at `slot_cap` and causally verifies `write 0 -> drop -> restore -> rise`.
    fn find_framegen_toggle(
        &self,
        cap_form: &Form,
        cap_addr: u64,
        measured_cap: f64,
        slot_cap: f64,
    ) -> Option<(u64, usize, u64)> {
        let t0 = Instant::now();
        let maps = self.mem.maps_text();
        let regs = regions(&maps);
        let data_ranges = lib_data_ranges(&maps);

        let win_lo = cap_addr.checked_sub(128)? & !7;
        let cap_off = (cap_addr - win_lo) as usize;
        let mut win = vec![0u8; 256];
        if !self.mem.read(win_lo, &mut win) {
            return None;
        }

        // Discover the enclosing setting object's vtable, value offset, and string field offsets.
        let mut layout: Option<(u64, u64, u64, usize, Vec<usize>)> = None;
        let mut p = cap_off & !7;
        while p >= 8 {
            p -= 8;
            let w = u64::from_le_bytes(win[p..p + 8].try_into().unwrap());
            let Some((seg_lo, seg_hi)) = containing_range(&data_ranges, w) else { continue };
            let val_off = cap_off - p;
            if val_off == 0 || val_off > 128 {
                continue;
            }
            let obj_addr = win_lo + p as u64;
            let mut str_offs = Vec::new();
            for s_off in (8..160usize).step_by(8) {
                if s_off == (val_off & !7) {
                    continue;
                }
                let ptr = if p + s_off + 8 <= win.len() {
                    u64::from_le_bytes(win[p + s_off..p + s_off + 8].try_into().unwrap())
                } else {
                    self.mem.word(obj_addr + s_off as u64, 8).unwrap_or(0)
                };
                if self.read_string_at(ptr).is_some() {
                    str_offs.push(s_off);
                }
            }
            if !str_offs.is_empty() {
                layout = Some((strip_tag(w), seg_lo, seg_hi, val_off, str_offs));
                break;
            }
        }
        let Some((cap_vptr, seg_lo, seg_hi, val_off, str_offs)) = layout else {
            game_log(&self.pkg, &format!("no sibling setting layout around {cap_addr:x}; keeping cap lift only"));
            return None;
        };

        let vptr_lo = cap_vptr.saturating_sub(0x10000).max(seg_lo);
        let vptr_hi = cap_vptr.saturating_add(0x10000).min(seg_hi);

        // Restore `cap_addr` to `slot_cap` before scanning so the ~350 ms parallel scan overlaps
        // the game settling back to its 2x framegen rate (`~measured_cap`).
        self.mem.put(cap_addr, cap_form.size(), cap_form.bits(slot_cap));

        let all: Vec<&Region> = regs.iter().collect();
        let parts = self.mem.scan_parallel(
            &all,
            Vec::<(u8, u64, u64)>::new,
            |out, base, run, _| {
                for (k, chunk) in run.chunks_exact(8).enumerate() {
                    let w = u64::from_le_bytes(chunk.try_into().unwrap());
                    if w == 0 {
                        continue;
                    }
                    let tag = w >> 56;
                    if tag != 0 && tag != 0xb4 {
                        continue;
                    }
                    let raw = strip_tag(w);
                    if raw < vptr_lo || raw > vptr_hi {
                        continue;
                    }
                    let off = k * 8;
                    let obj_addr = base + off as u64;
                    let val_addr = obj_addr + val_off as u64;
                    if val_addr == cap_addr {
                        continue;
                    }
                    let val_word = if off + val_off + 8 <= run.len() {
                        u64::from_le_bytes(run[off + val_off..off + val_off + 8].try_into().unwrap())
                    } else if let Some(v) = self.mem.word(val_addr, 8) {
                        v
                    } else if let Some(v) = self.mem.word(val_addr, 4) {
                        v
                    } else {
                        continue;
                    };
                    if toggle_on_size(val_word, true).is_none() {
                        continue;
                    }
                    let mut best_prio = 0u8;
                    for &s_off in &str_offs {
                        let ptr = if off + s_off + 8 <= run.len() {
                            u64::from_le_bytes(run[off + s_off..off + s_off + 8].try_into().unwrap())
                        } else {
                            self.mem.word(obj_addr + s_off as u64, 8).unwrap_or(0)
                        };
                        if let Some(text) = self.read_string_at(ptr) {
                            best_prio = best_prio.max(is_framegen_desc(&text));
                        }
                    }
                    if best_prio > 0 {
                        out.push((best_prio, val_addr, val_word));
                    }
                }
            },
        );

        let cap_sz = cap_form.size();
        let paired_slots = self.mem.word(cap_addr, cap_sz) == self.mem.word(cap_addr + cap_sz as u64, cap_sz);
        let mut seen = HashSet::new();
        let mut cands: Vec<(u8, u64, usize, u64)> = Vec::new();
        for part in parts {
            for (prio, val_addr, val_word) in part {
                if seen.insert(val_addr) {
                    if let Some((fg_size, fg_on)) = toggle_on_size(val_word, paired_slots) {
                        cands.push((prio, val_addr, fg_size, fg_on));
                    }
                }
            }
        }
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        cands.truncate(16);
        let scan_ms = t0.elapsed().as_millis();
        game_log(
            &self.pkg,
            &format!(
                "framegen search (+{val_off:#x}, paired {paired_slots}, {scan_ms}ms): {} candidates",
                cands.len()
            ),
        );
        if cands.is_empty() {
            return None;
        }

        // Confirm the game settled back into 2x framegen above `slot_cap` before probing.
        let mut fps_base = self.meter.fps(&self.mem, 450);
        if fps_base <= slot_cap * 1.35 {
            sleep(Duration::from_millis(400));
            fps_base = self.meter.fps(&self.mem, 450);
        }
        if fps_base <= slot_cap * 1.35 {
            game_log(
                &self.pkg,
                &format!("framegen search skipped: baseline {fps_base:.1} fps not above 2x of {slot_cap}"),
            );
            return None;
        }
        let ref_cap = fps_base.max(measured_cap * 0.85);

        for (prio, fg_addr, fg_size, fg_on) in cands {
            if !alive(self.mem.pid) {
                return None;
            }
            if self.mem.word(fg_addr, fg_size) != Some(fg_on) {
                continue;
            }
            let slot = [(fg_addr, fg_on)];
            pending_put(&self.mem, fg_size, &slot, 0);
            sleep(Duration::from_millis(450));
            let fps_off = self.meter.fps(&self.mem, 650);
            if !fg_disabled_drop(fps_off, ref_cap, slot_cap) {
                pending_undo(&self.mem, fg_size, &slot);
                continue;
            }
            // Causal revert: re-enabling must jump FPS back up near `ref_cap`. Frame-generation
            // pipelines often need 1-2 s to re-warm motion-vector history buffers, so sample up
            // to 4 windows and return as soon as the rise appears.
            self.mem.put(fg_addr, fg_size, fg_on);
            sleep(Duration::from_millis(400));
            let mut fps_on = 0.0;
            let mut rose = false;
            for _ in 0..4 {
                fps_on = self.meter.fps(&self.mem, 550);
                if fg_restored_rise(fps_on, fps_off, ref_cap) {
                    rose = true;
                    break;
                }
            }
            if !rose {
                pending_undo(&self.mem, fg_size, &slot);
                game_log(
                    &self.pkg,
                    &format!(
                        "framegen candidate {fg_addr:x}: {fps_base:.1} -> {fps_off:.1}, undo -> {fps_on:.1} (false drop)"
                    ),
                );
                continue;
            }
            self.mem.put(fg_addr, fg_size, 0);
            pending_keep(&self.mem);
            game_log(
                &self.pkg,
                &format!(
                    "disabled built-in framegen slot {fg_addr:x} (size {fg_size}, prio {prio}, was {fg_on:x}): {fps_base:.1} -> {fps_off:.1} -> {fps_on:.1} fps"
                ),
            );
            return Some((fg_addr, fg_size, fg_on));
        }
        None
    }

    /// Learns the slot: parallel single-pass scan across all forms (+ `cap/2` for 2x framegen),
    /// page-batched stability check, generic structural ranking, and causal probing.
    fn learn(&self, cap: f64, prev: &[(usize, Vec<u64>)]) -> Learned {
        let t0 = Instant::now();
        let maps = self.mem.maps_text();
        let regs = regions(&maps);
        let so_ranges = lib_ranges(&maps);
        let all: Vec<&Region> = regs.iter().collect();

        let mut specs: Vec<TargetSpec> = Vec::with_capacity(FORMS.len() * 2);
        let mut meta_desc: Vec<(usize, f64, bool)> = Vec::with_capacity(FORMS.len() * 2);
        for (f, form) in FORMS.iter().enumerate() {
            specs.push(TargetSpec { id: specs.len(), size: form.size(), bits: form.bits(cap) });
            meta_desc.push((f, cap, true));
        }
        if let Some(hc) = half_cap(cap) {
            for (f, form) in FORMS.iter().enumerate() {
                specs.push(TargetSpec { id: specs.len(), size: form.size(), bits: form.bits(hc) });
                meta_desc.push((f, hc, false));
            }
        }

        let first = self.mem.scan_meta(&all, &specs, &so_ranges);
        let scan_ms = t0.elapsed().as_millis();
        sleep(Duration::from_millis(500));
        let verified = self.mem.verify_stable(&specs, first);
        let stable_ms = t0.elapsed().as_millis();

        let buckets: Vec<Bucket> = verified
            .into_iter()
            .zip(meta_desc)
            .map(|(slots, (form, slot_cap, primary))| Bucket { form, slot_cap, primary, slots })
            .collect();

        let counts: Vec<String> = buckets
            .iter()
            .filter(|b| b.primary && !b.slots.is_empty())
            .map(|b| {
                format!(
                    "{} {}+{}",
                    FORMS[b.form].name,
                    b.slots.iter().filter(|s| s.lib).count(),
                    b.slots.iter().filter(|s| !s.lib).count()
                )
            })
            .collect();
        let poison = fs::read_to_string(format!("{DIR}/{}.poison", self.pkg)).unwrap_or_default();
        let (order, followed) = plan(&buckets, prev, &poisoned(&poison, &self.version), MAX_SET);
        game_log(
            &self.pkg,
            &format!(
                "learning at cap {cap} (scan {scan_ms}ms, stable {stable_ms}ms), {} planned, {followed} followed (library+heap): {}",
                order.len(),
                counts.join(", ")
            ),
        );

        let primary_slots = || -> Vec<(usize, Vec<u64>)> {
            buckets
                .iter()
                .filter(|b| b.primary)
                .map(|b| (b.form, b.slots.iter().map(|s| s.addr).collect()))
                .collect()
        };

        let mut above = 0u32;
        let mut below = Steady::default();
        let mut part: Option<(usize, bool, f64)> = None;
        let mut last_fps = cap;
        let died = |last: Option<(usize, bool, u64)>| {
            if let Some((form, lib, addr)) = last {
                game_log(
                    &self.pkg,
                    &format!("the game died testing {} {addr:x}; that part is skipped from now on", FORMS[form].name),
                );
                if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(format!("{DIR}/{}.poison", self.pkg)) {
                    let _ = f.write_all(poison_line(&self.version, form, lib).as_bytes());
                }
            }
            Learned::Nothing
        };
        let mut last = None;
        for cand in order {
            let lib = region_of(&regs, cand.addr).map(|r| r.lib).unwrap_or(false);
            if part != Some((cand.form, lib, cand.slot_cap)) {
                part = Some((cand.form, lib, cand.slot_cap));
                game_log(
                    &self.pkg,
                    &format!(
                        "testing {} (cap {}) in {} at {:.1} s",
                        FORMS[cand.form].name,
                        cand.slot_cap,
                        if lib { "library data" } else { "the heap" },
                        t0.elapsed().as_secs_f64()
                    ),
                );
            }
            // Only pause/re-sample when the previous probe did not already see healthy `cap`.
            if !probe_recovered(last_fps, cap) || cap_moved(last_fps, cap) {
                loop {
                    let fps = self.meter.fps(&self.mem, 300);
                    above = if cap_moved(fps, cap) { above + 1 } else { 0 };
                    if above >= 3 {
                        game_log(&self.pkg, &format!("the game left cap {cap} ({fps:.1} fps); learning again"));
                        return Learned::Moved(primary_slots());
                    }
                    if probe_recovered(fps, cap) && !cap_moved(fps, cap) {
                        below = Steady::default();
                        break;
                    }
                    if let Some(lower) = below.feed(fps, self.target).filter(|&c| c < cap) {
                        game_log(&self.pkg, &format!("the game left cap {cap} for {lower}; learning again"));
                        return Learned::Moved(primary_slots());
                    }
                    if !alive(self.mem.pid) {
                        return died(last);
                    }
                    if t0.elapsed() > LEARN_BUDGET {
                        return Learned::Nothing;
                    }
                    sleep(Duration::from_secs(1));
                }
            }
            if t0.elapsed() > LEARN_BUDGET {
                return Learned::Nothing;
            }
            last = Some((cand.form, lib, cand.addr));
            let (hit, observed) = self.probe(&FORMS[cand.form], cand.addr, cap, cand.slot_cap);
            last_fps = observed;
            if cap_moved(observed, cap) {
                above += 1;
            } else {
                above = 0;
            }
            if hit && self.lifts(&FORMS[cand.form], cand.addr, cap, Some(cand.slot_cap)) {
                return Learned::Slot(cand.form, cand.slot_cap, cand.addr);
            }
            if !alive(self.mem.pid) {
                return died(last);
            }
        }
        Learned::Nothing
    }

    /// Keeps the cap slot at `target` and (when present) keeps the built-in framegen toggle at `0`.
    /// Returns true when the cap slot leaves its cap set for 10 s or hits a higher secondary cap
    /// (need to re-find), false when the game exits.
    fn hold(
        &self,
        form: &Form,
        addr: u64,
        learned_cap: f64,
        mut fg: Option<(u64, usize, u64)>,
        fg_cache: Option<&FrameGenCache>,
    ) -> bool {
        let want = form.bits(self.target);
        let mut last = self.meter.count(&self.mem);
        let mut quiet = Instant::now();
        let mut strange: Option<Instant> = None;
        let mut fg_strange: Option<Instant> = None;
        let mut fg_retry_at: Option<Instant> = None;
        let mut rewrites = 0u32;
        let mut fg_rewrites = 0u32;
        let mut sec_mark = Instant::now();
        let mut sec_count = last;
        let mut stuck = Steady::with_need(6);
        loop {
            if !alive(self.mem.pid) {
                return false;
            }
            if let Some(v) = self.mem.word(addr, form.size()) {
                match hold_verdict(v, form, self.target) {
                    Hold::Keep => strange = None,
                    Hold::Rewrite => {
                        strange = None;
                        self.mem.put(addr, form.size(), want);
                        rewrites += 1;
                        if rewrites.is_power_of_two() {
                            game_log(&self.pkg, &format!("game wrote its cap back ({rewrites})"));
                        }
                        // If the user switched into a framegen mode in-game after starting in a
                        // native mode, opportunistically locate and disable the cached toggle.
                        if fg.is_none() {
                            if let Some(fgc) = fg_cache {
                                let due = fg_retry_at.map(|t| t.elapsed() >= Duration::from_secs(30)).unwrap_or(true);
                                if due {
                                    fg_retry_at = Some(Instant::now());
                                    let regs = self.mem.regions();
                                    if let Some((fa, fsz, fon)) = self.relocate_fg(fgc, &regs, true) {
                                        self.mem.put(fa, fsz, 0);
                                        game_log(
                                            &self.pkg,
                                            &format!("disabled built-in framegen slot {fa:x} after mode switch"),
                                        );
                                        fg = Some((fa, fsz, fon));
                                    }
                                }
                            }
                        }
                    }
                    Hold::Strange => {
                        if strange.get_or_insert_with(Instant::now).elapsed() > Duration::from_secs(10) {
                            return true;
                        }
                    }
                }
            }
            if let Some((fg_addr, fg_size, fg_on)) = fg {
                if let Some(fv) = self.mem.word(fg_addr, fg_size) {
                    match fg_hold_verdict(fv, fg_size, fg_on) {
                        Hold::Keep => fg_strange = None,
                        Hold::Rewrite => {
                            fg_strange = None;
                            self.mem.put(fg_addr, fg_size, 0);
                            fg_rewrites += 1;
                            if fg_rewrites.is_power_of_two() {
                                game_log(
                                    &self.pkg,
                                    &format!("game re-enabled built-in framegen ({fg_rewrites}); disabled again"),
                                );
                            }
                        }
                        Hold::Strange => {
                            if fg_strange.get_or_insert_with(Instant::now).elapsed() > Duration::from_secs(10) {
                                game_log(
                                    &self.pkg,
                                    &format!("built-in framegen slot {fg_addr:x} left its values; stopped holding it"),
                                );
                                fg = None;
                                fg_strange = None;
                            }
                        }
                    }
                }
            }
            let now = self.meter.count(&self.mem);
            if now != last {
                last = now;
                quiet = Instant::now();
            }
            // Passive 1 s fps check: if the game settled on a higher secondary cap below target
            // (e.g. lifted 45 -> 90 while target is 120), drop back into discovery.
            if sec_mark.elapsed() >= Duration::from_secs(1) {
                if let (Some(a), Some(b)) = (sec_count, now) {
                    let fps = b.wrapping_sub(a) as f64 / sec_mark.elapsed().as_secs_f64();
                    if let Some(c) = stuck.feed(fps, self.target) {
                        if c > learned_cap * 1.05 {
                            game_log(&self.pkg, &format!("still capped at {c} while holding {addr:x}; finding again"));
                            return true;
                        }
                    }
                }
                sec_mark = Instant::now();
                sec_count = now;
            }
            sleep(Duration::from_millis(if quiet.elapsed() > Duration::from_secs(1) { 1000 } else { 100 }));
        }
    }

    /// A steady cap other than the ones no slot lifted.
    fn steady_cap(&self, failed: &[f64]) -> Option<f64> {
        let mut steady = Steady::default();
        loop {
            if !alive(self.mem.pid) {
                return None;
            }
            match steady.feed(self.meter.fps(&self.mem, 1000), self.target) {
                Some(cap) if !failed.contains(&cap) => return Some(cap),
                _ => {}
            }
        }
    }
}

fn serve(pid: i32, target: f64, version: String, pkg: String) {
    let Some(mem) = Mem::open(pid) else { return };
    let mut meter = None;
    for _ in 0..40 {
        meter = Meter::find(&mem);
        if meter.is_some() || !alive(pid) {
            break;
        }
        sleep(Duration::from_millis(500));
    }
    let Some(meter) = meter else {
        return;
    };
    // Wait silently until this process actually presents frames so helper subprocesses
    // (e.g. shader/PSO compiler services) never clobber the main game's state or logs.
    let start_presents = meter.count(&mem).unwrap_or(0);
    loop {
        if !alive(pid) {
            return;
        }
        let now = meter.count(&mem).unwrap_or(start_presents);
        if now.wrapping_sub(start_presents) >= 2 {
            break;
        }
        sleep(Duration::from_millis(500));
    }

    let game = Game { pkg, version, target, mem, meter };
    game_log(&game.pkg, &format!("pid {pid} version {} to {target}", game.version));
    let state = |s: &str| write_state(&game.pkg, &format!("pid={pid}\ntarget={target}\n{s}"));
    let mut failed: Vec<f64> = Vec::new();
    let mut prev: Vec<(usize, Vec<u64>)> = Vec::new();
    loop {
        state("state=waiting for a steady frame rate\n");
        let Some(cap) = game.steady_cap(&failed) else { break };
        let mut cached = cache_load(&game.pkg).filter(|f| f.version == game.version);
        let mut how = "cache";
        let mut fg_active: Option<(u64, usize, u64)> = None;
        let mut cache_dirty = false;

        let mut hit: Option<(usize, f64, u64)> = None;
        if let Some(ref f) = cached {
            let (rel_cap, rel_fg) = game.relocate(f, cap);
            if let Some((form, slot_cap, addr)) = rel_cap {
                let is_2x = (slot_cap * 2.0 - cap).abs() < 1.0;
                let mut lift_base = cap;
                if is_2x {
                    if let Some((fg_addr, fg_size, fg_on)) = rel_fg {
                        if game.disable_fg_and_verify(fg_addr, fg_size, fg_on, cap, slot_cap) {
                            fg_active = Some((fg_addr, fg_size, fg_on));
                            // With 2x framegen disabled, native FPS lifts from `slot_cap`
                            // (e.g. 45 -> 70..120), even in GPU-bound scenes below `cap`.
                            lift_base = slot_cap;
                        }
                    }
                }
                if game.lifts(&FORMS[form], addr, lift_base, None) {
                    hit = Some((form, slot_cap, addr));
                } else if let Some((fg_addr, fg_size, fg_on)) = fg_active.take() {
                    game.mem.put(fg_addr, fg_size, fg_on);
                }
            }
        }
        if hit.is_none() {
            how = "learned";
            state(&format!("state=learning\ncap={cap}\n"));
            hit = match game.learn(cap, &prev) {
                Learned::Slot(f, sc, a) => Some((f, sc, a)),
                Learned::Moved(p) => {
                    prev = p;
                    continue;
                }
                Learned::Nothing => None,
            };
        }
        let Some((form, slot_cap, addr)) = hit else {
            failed.push(cap);
            state(&format!("state=no cap slot found\ncap={cap}\n"));
            game_log(&game.pkg, &format!("no slot lifts the cap {cap}"));
            continue;
        };

        let f = &FORMS[form];
        let cap_div: u8 = if (slot_cap * 2.0 - cap).abs() < 1.0 { 2 } else { 1 };
        if cap_div == 2 && fg_active.is_none() {
            state(&format!("state=disabling built-in framegen\ncap={cap}\n"));
            if let Some(fg) = game.find_framegen_toggle(f, addr, cap, slot_cap) {
                fg_active = Some(fg);
                cache_dirty = true;
            }
        }

        // Write final holding values before fingerprinting so both objects are in steady state.
        game.mem.put(addr, f.size(), f.bits(target));
        if let Some((fg_addr, fg_size, _)) = fg_active {
            game.mem.put(fg_addr, fg_size, 0);
        }

        if how == "learned" || cache_dirty {
            let regs = game.mem.regions();
            if how == "learned" {
                let fg_req = fg_active.map(|(a, sz, _)| (a, sz));
                let (fp_cap, fp_fg) = game.fingerprint_pair(addr, f.size(), fg_req);
                if let (Some(r), Some((fingerprint, delta))) = (region_of(&regs, addr), fp_cap) {
                    let fg_cache = match (fg_active, fp_fg) {
                        (Some((fa, fsz, fon)), Some((ffp, fdelta))) => region_of(&regs, fa).map(|fr| FrameGenCache {
                            size: fsz,
                            on_bits: fon,
                            lib: fr.lib,
                            label: fr.label.clone(),
                            file_off: fr.file_off,
                            offset: fa - fr.lo,
                            fingerprint: ffp,
                            delta: fdelta,
                        }),
                        _ => None,
                    };
                    let found = Found {
                        version: game.version.clone(),
                        form,
                        lib: r.lib,
                        label: r.label.clone(),
                        file_off: r.file_off,
                        offset: addr - r.lo,
                        fingerprint,
                        delta,
                        cap_div,
                        fg: fg_cache,
                    };
                    cache_save(&game.pkg, &found);
                    cached = Some(found);
                }
            } else if let (Some(mut found), Some((fa, fsz, fon))) = (cached.take(), fg_active) {
                if let (Some(fr), Some((ffp, fdelta))) =
                    (region_of(&regs, fa), game.fingerprint_pair(fa, fsz, None).0)
                {
                    found.cap_div = cap_div;
                    found.fg = Some(FrameGenCache {
                        size: fsz,
                        on_bits: fon,
                        lib: fr.lib,
                        label: fr.label.clone(),
                        file_off: fr.file_off,
                        offset: fa - fr.lo,
                        fingerprint: ffp,
                        delta: fdelta,
                    });
                    cache_save(&game.pkg, &found);
                }
                cached = Some(found);
            }
        }

        sleep(Duration::from_millis(500));
        let fps = game.meter.fps(&game.mem, 1000);
        let fg_note = if fg_active.is_some() { ", framegen=off" } else { "" };
        game_log(
            &game.pkg,
            &format!("{} slot {addr:x} ({how}, slot_cap {slot_cap}{fg_note}) cap {cap} -> {fps:.1} fps", f.name),
        );
        state(&format!("state=holding\nform={}\nhow={how}\ncap={cap}\nfps={fps:.1}\n", f.name));
        let fg_cache_ref = cached.as_ref().and_then(|c| c.fg.as_ref());
        if !game.hold(f, addr, cap, fg_active, fg_cache_ref) {
            break;
        }
        game_log(&game.pkg, &format!("slot {addr:x} left its cap values; finding it again"));
    }
    write_state(&game.pkg, &format!("pid={pid}\nstate=game exited\n"));
}

fn main() {
    recover();
    // Optional single-process debug entry: `lepasd --once <pid> <target> <version> <pkg>`
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 6 && args[1] == "--once" {
        if let Some(s) = parse_start(&format!("{} {} {} {}", args[2], args[3], args[4], args[5])) {
            serve(s.pid, s.target, s.version, s.package);
        }
        return;
    }

    let name = CString::new(START_PROPERTY).unwrap();
    let pi = loop {
        // SAFETY: `name` is a NUL-terminated string that outlives the call.
        let pi = unsafe { __system_property_find(name.as_ptr()) };
        if !pi.is_null() {
            break pi;
        }
        sleep(Duration::from_secs(1));
    };
    let mut seen: HashSet<i32> = HashSet::new();
    let mut serial = 0u32;
    loop {
        // SAFETY: `pi` came from __system_property_find and property infos live as long as the
        // process.
        let now = unsafe { __system_property_serial(pi) };
        if now != serial {
            serial = now;
            if let Some(s) = parse_start(&property(START_PROPERTY)) {
                seen.retain(|&p| alive(p));
                if alive(s.pid) && seen.insert(s.pid) {
                    let _ = thread::Builder::new()
                        .name(format!("lepas {}", s.pid))
                        .spawn(move || serve(s.pid, s.target, s.version, s.package));
                }
            }
        }
        // Blocks until the broker sets the next value: no wakeups between games.
        let mut next = 0u32;
        // SAFETY: `pi` as above; `next` is a live u32; a null timeout waits without a deadline.
        unsafe { __system_property_wait(pi, serial, &mut next, std::ptr::null()) };
    }
}
