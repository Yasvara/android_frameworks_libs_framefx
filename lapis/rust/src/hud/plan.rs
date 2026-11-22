// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The skip table, from the labels of the last frames. A pipeline in a pass whose draws were all
//! HUD skips all of them; one whose HUD draws always come after its other draws skips from the
//! first index past those; one that interleaves them skips nothing. The table goes out only while
//! it covers the HUD draws of the frame; until then Hide HUD keeps the scene copy.

use std::collections::{HashMap, VecDeque};

use super::*;

/// Slots in the open-addressed table the recorder looks up; a power of two.
pub const TABLE_SIZE: usize = 4096;
/// How far a key may sit from its slot. lapis/hud.cpp probes as far.
pub const MAX_PROBE: usize = 16;
/// Pairs past this many only carry `VALUE_KNOWN`: keep the table half empty.
const MAX_ENTRIES: usize = TABLE_SIZE / 2;
const WINDOW: usize = 12;
/// Frames a pair must have been drawn in before it skips.
const MIN_FRAMES: u32 = 4;
/// A HUD label this sure votes to skip; a scene or post label this sure votes against.
const SURE_HUD: f32 = 0.7;
/// In-world UI shares the scene's camera and passes: it needs more.
const SURE_WORLD_UI: f32 = 0.85;
const SURE_OTHER: f32 = 0.5;
const ENGAGE_COVERAGE: f32 = 0.9;
const RELEASE_COVERAGE: f32 = 0.75;
/// Frames in a row under RELEASE_COVERAGE before the table stops skipping.
const RELEASE_FRAMES: u32 = 3;

/// The slot a key starts probing at. lapis/hud.h HudSlot computes the same.
pub fn slot(key: u64) -> usize {
    (key >> 1) as usize & (TABLE_SIZE - 1)
}

#[derive(Clone, Copy, Default)]
struct PairFrame {
    hud: u32,
    unsure: u32,
    other: u32,
    min_hud: u32,
    max_other: Option<u32>,
}

#[derive(Clone, Copy, Default)]
struct Pair {
    frames: u32,
    hud: u32,
    unsure: u32,
    other: u32,
    min_hud: u32,
    max_other: Option<u32>,
}

impl Pair {
    fn add(&mut self, f: &PairFrame) {
        if f.hud > 0 {
            self.min_hud = if self.hud == 0 { f.min_hud } else { self.min_hud.min(f.min_hud) };
        }
        self.frames += 1;
        self.hud += f.hud;
        self.unsure += f.unsure;
        self.other += f.other;
        self.max_other = match (self.max_other, f.max_other) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
    }

    /// The draw indices to skip, from (inclusive).
    fn skips_from(&self, frames: u32) -> Option<u32> {
        if self.frames < frames || self.hud == 0 {
            return None;
        }
        match self.max_other {
            None if self.hud >= self.unsure => Some(0),
            Some(last) if last < self.min_hud => Some(last + 1),
            _ => None,
        }
    }
}

pub struct Planned {
    pub coverage: f32,
    pub entries: u32,
    pub unsampled: u32,
    pub skipping: bool,
    pub changed: bool,
}

pub struct Planner {
    window: VecDeque<HashMap<u64, PairFrame>>,
    table: Box<[HudEntry; TABLE_SIZE]>,
    next: Box<[HudEntry; TABLE_SIZE]>,
    skipping: bool,
    low: u32,
}

fn empty_table() -> Box<[HudEntry; TABLE_SIZE]> {
    vec![HudEntry::default(); TABLE_SIZE].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!())
}

impl Default for Planner {
    fn default() -> Self {
        Planner { window: VecDeque::new(), table: empty_table(), next: empty_table(), skipping: false, low: 0 }
    }
}

fn insert(table: &mut [HudEntry; TABLE_SIZE], key: u64, value: u64) -> bool {
    for probe in 0..MAX_PROBE {
        let e = &mut table[(slot(key) + probe) & (TABLE_SIZE - 1)];
        if e.key == 0 || e.key == key {
            *e = HudEntry { key, value };
            return true;
        }
    }
    false
}

impl Planner {
    pub fn table(&self) -> &[HudEntry; TABLE_SIZE] {
        &self.table
    }

    pub fn skipping(&self) -> bool {
        self.skipping
    }

    /// Folds this frame's labels in and remakes the table. `known` says whether a pair's camera is
    /// known; `allow` whether HUD draws may be skipped at all.
    pub fn update(
        &mut self,
        f: &HudFrame,
        labels: &[u8],
        confidence: &[f32],
        known: impl Fn(u64) -> bool,
        allow: bool,
    ) -> Planned {
        let mut frame: HashMap<u64, PairFrame> = HashMap::new();
        for pass in f.passes {
            for i in draw_range(pass, f.draws.len()) {
                let d = &f.draws[i];
                let e = frame.entry(table_key(pass.key, d.pipeline)).or_default();
                let hud = matches!(labels[i], LABEL_HUD | LABEL_WORLD_UI);
                let sure = if labels[i] == LABEL_WORLD_UI { SURE_WORLD_UI } else { SURE_HUD };
                if hud && confidence[i] >= sure {
                    e.min_hud = if e.hud == 0 { d.index } else { e.min_hud.min(d.index) };
                    e.hud += 1;
                } else if hud {
                    e.unsure += 1;
                } else if confidence[i] >= SURE_OTHER {
                    e.other += 1;
                    e.max_other = Some(e.max_other.map_or(d.index, |m| m.max(d.index)));
                } else {
                    e.unsure += 1;
                }
            }
        }
        if self.window.len() == WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(frame);
        let mut pairs: HashMap<u64, Pair> = HashMap::new();
        for frame in &self.window {
            for (key, pf) in frame {
                pairs.entry(*key).or_default().add(pf);
            }
        }
        let need = MIN_FRAMES.min(self.window.len() as u32).max(1);
        let ready = self.window.len() as u32 >= MIN_FRAMES;

        // This frame's HUD draws the table would skip, and its other draws it would.
        let (mut hud, mut covered) = (0u32, 0u32);
        for pass in f.passes {
            for i in draw_range(pass, f.draws.len()) {
                if !matches!(labels[i], LABEL_HUD | LABEL_WORLD_UI) {
                    continue;
                }
                hud += 1;
                let d = &f.draws[i];
                let from = pairs.get(&table_key(pass.key, d.pipeline)).and_then(|p| p.skips_from(need));
                covered += u32::from(from.is_some_and(|from| d.index >= from));
            }
        }
        let coverage = if hud > 0 { covered as f32 / hud as f32 } else { 0.0 };
        if !allow {
            self.skipping = false;
            self.low = 0;
        } else if hud > 0 {
            if self.skipping {
                self.low = if coverage < RELEASE_COVERAGE { self.low + 1 } else { 0 };
                if self.low >= RELEASE_FRAMES {
                    self.skipping = false;
                    self.low = 0;
                }
            } else if ready && coverage >= ENGAGE_COVERAGE {
                self.skipping = true;
            }
        }

        self.next.fill(HudEntry::default());
        let mut entries = 0u32;
        let mut unsampled = 0u32;
        let mut ordered: Vec<(u64, Option<u32>)> =
            pairs.iter().map(|(&k, p)| (k, if self.skipping { p.skips_from(need) } else { None })).collect();
        // Skips first: they never give way to a full table.
        ordered.sort_by_key(|&(k, from)| (from.is_none(), k));
        for (key, from) in ordered {
            let is_known = known(key);
            if !is_known && self.window.back().is_some_and(|w| w.contains_key(&key)) {
                unsampled += 1;
            }
            let mut value = if is_known { VALUE_KNOWN } else { 0 };
            if let Some(from) = from {
                let from = u64::from(from).min(VALUE_INDEX_MASK);
                value |= VALUE_SKIP | from << VALUE_FROM_SHIFT | VALUE_OPEN << VALUE_TO_SHIFT;
            } else if value == 0 || entries as usize >= MAX_ENTRIES {
                continue;
            }
            entries += u32::from(insert(&mut self.next, key, value));
        }
        let changed = self.next[..] != self.table[..];
        if changed {
            core::mem::swap(&mut self.table, &mut self.next);
        }
        Planned { coverage, entries, unsampled, skipping: self.skipping, changed }
    }
}

/// What the recorder does with `table` for draw `index` of pipeline `pipeline` in pass `pass`:
/// skip it, and whether its camera is known. The reference for lapis/hud.cpp's lookup.
#[cfg(test)]
pub fn lookup(table: &[HudEntry; TABLE_SIZE], pass: u64, pipeline: u64, index: u32) -> (bool, bool) {
    let key = table_key(pass, pipeline);
    for probe in 0..MAX_PROBE {
        let e = table[(slot(key) + probe) & (TABLE_SIZE - 1)];
        if e.key == 0 {
            break;
        }
        if e.key == key {
            let from = (e.value >> VALUE_FROM_SHIFT) & VALUE_INDEX_MASK;
            let to = (e.value >> VALUE_TO_SHIFT) & VALUE_INDEX_MASK;
            let inside = u64::from(index) >= from && (to == VALUE_OPEN || u64::from(index) < to);
            return (e.value & VALUE_SKIP != 0 && inside, e.value & VALUE_KNOWN != 0);
        }
    }
    (false, false)
}
