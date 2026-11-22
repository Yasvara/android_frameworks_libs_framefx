// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The ray-traced scene kept across frames (src/rays.cpp): one bottom-level tree per draw the game
//! repeats, known by its pipeline and draw parameters. A draw seen again with the same vertices is
//! kept as it was, the camera's motion carried by its instance; only draws whose points moved
//! (characters, animation) are refitted, and only new ones built.

use std::collections::{BTreeSet, HashMap};

/// The tree stays as built: the draw's points did not move.
pub const KEEP: u8 = 0;
/// A new draw, or one whose vertex count changed: build its tree.
pub const BUILD: u8 = 1;
/// Its points move: refit its tree to this frame's.
pub const REFIT: u8 = 2;

/// Frames a draw that moved keeps being refitted after it last did.
pub const STICKY: u64 = 30;
/// Refits in a row before the tree is built again: boxes refitted for long grow loose.
pub const MAX_REFITS: u32 = 60;

const NONE: u64 = 0;

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    vertices: u32,
    seen: u64,   // the last frame placed
    moved: u64,  // the last frame its points were seen to move, NONE never
    refits: u32,
    rebuild: bool,  // its points moved too far for a refit
    reachable: bool,
}

#[derive(Default)]
pub struct SceneCache {
    entries: Vec<Entry>,
    free: Vec<u32>,
    by_key: HashMap<u64, u32>,
    occurrences: HashMap<u64, u32>,
    frame: u64,
    /// Out of sight within the age but left out as moving, since the caller last took it.
    pub moving_out_of_sight: u64,
}

fn mix(a: u64, b: u64) -> u64 {
    // splitmix64 over the pair: distinct draws keep distinct keys.
    let mut z = a ^ b.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(a << 6).wrapping_add(a >> 2);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl SceneCache {
    /// A new frame: its draws are placed in the order the game drew them.
    pub fn begin(&mut self) {
        self.frame += 1;
        self.occurrences.clear();
    }

    /// The entry for a draw known by `key`, and what its tree needs this frame. A key drawn
    /// several times in a frame is told apart by the order of its draws.
    pub fn place(&mut self, key: u64, vertices: u32) -> (u32, u8) {
        let n = self.occurrences.entry(key).or_insert(0);
        let full = mix(key, *n as u64);
        *n += 1;
        if let Some(&slot) = self.by_key.get(&full) {
            let e = &mut self.entries[slot as usize];
            if e.vertices == vertices {
                e.seen = self.frame;
                let moving = e.moved != NONE && self.frame - e.moved <= STICKY;
                if e.rebuild || (moving && e.refits >= MAX_REFITS) {
                    e.rebuild = false;
                    e.refits = 0;
                    return (slot, BUILD);
                }
                if moving {
                    e.refits += 1;
                    return (slot, REFIT);
                }
                return (slot, KEEP);
            }
            // Another shape under the same key: the old entry ages out, a frame in flight may
            // still trace it.
            e.reachable = false;
            self.by_key.remove(&full);
        }
        let entry = Entry { key: full, vertices, seen: self.frame, reachable: true, ..Entry::default() };
        let slot = match self.free.pop() {
            Some(s) => {
                self.entries[s as usize] = entry;
                s
            }
            None => {
                self.entries.push(entry);
                (self.entries.len() - 1) as u32
            }
        };
        self.by_key.insert(full, slot);
        (slot, BUILD)
    }

    /// The GPU saw `entry`'s points move in an earlier frame; `far` past what a refit keeps
    /// tight (another mesh drawn under its key).
    pub fn moved(&mut self, entry: u32, far: bool) {
        if let Some(e) = self.entries.get_mut(entry as usize) {
            e.moved = self.frame;
            e.rebuild |= far;
        }
    }

    /// A tree that could not be made: its entry goes, so the next frame builds it again.
    pub fn drop_entry(&mut self, entry: u32) {
        if let Some(e) = self.entries.get_mut(entry as usize) {
            if e.reachable {
                self.by_key.remove(&e.key);
            }
            *e = Entry::default();
            self.free.push(entry);
        }
    }

    /// Entries drawn within `age` frames but not this one, and still when last drawn: what the
    /// game culled or left off screen stands where it was, its shadow and reflection with it.
    pub fn retained(&mut self, age: u64, out: &mut Vec<u32>) {
        for (i, e) in self.entries.iter().enumerate() {
            if !e.reachable || e.seen == NONE || e.seen == self.frame || self.frame - e.seen > age {
                continue;
            }
            if e.moved != NONE && e.seen - e.moved.min(e.seen) <= STICKY {
                self.moving_out_of_sight += 1;
                continue;
            }
            out.push(i as u32);
        }
    }

    /// Entries not placed for more than `age` frames, removed; their trees are the caller's to
    /// free. `age` must cover the frames in flight.
    pub fn evict(&mut self, age: u64, out: &mut Vec<u32>) {
        for (i, e) in self.entries.iter_mut().enumerate() {
            if e.seen == NONE || self.frame - e.seen <= age {
                continue;
            }
            if e.reachable {
                self.by_key.remove(&e.key);
            }
            *e = Entry::default();
            self.free.push(i as u32);
            out.push(i as u32);
        }
    }

    #[cfg(test)]
    fn live(&self) -> usize {
        self.by_key.len()
    }
}

/// A buddy allocator over a power-of-two arena, in blocks of at least `MIN` bytes.
pub struct Arena {
    size: u64,
    free: Vec<BTreeSet<u64>>,      // by order: offsets of free blocks of MIN << order
    used: HashMap<u64, u32>,       // offset -> order
    in_use: u64,
}

pub const MIN: u64 = 256;
pub const NO_SPACE: u64 = u64::MAX;

impl Arena {
    /// `size` rounded down to a power of two, at least MIN.
    pub fn new(size: u64) -> Arena {
        let size = if size < MIN { MIN } else { 1u64 << (63 - size.leading_zeros()) };
        let orders = (size / MIN).trailing_zeros() as usize + 1;
        let mut free = vec![BTreeSet::new(); orders];
        free[orders - 1].insert(0);
        Arena { size, free, used: HashMap::new(), in_use: 0 }
    }

    pub fn in_use(&self) -> u64 {
        self.in_use
    }

    fn order_for(&self, bytes: u64) -> Option<usize> {
        let block = bytes.max(MIN).checked_next_power_of_two()?;
        if block > self.size {
            return None;
        }
        Some((block / MIN).trailing_zeros() as usize)
    }

    /// The offset of `bytes`, aligned to their block, or NO_SPACE.
    pub fn alloc(&mut self, bytes: u64) -> u64 {
        let Some(want) = self.order_for(bytes) else { return NO_SPACE };
        let Some(have) = (want..self.free.len()).find(|&o| !self.free[o].is_empty()) else {
            return NO_SPACE;
        };
        let at = *self.free[have].iter().next().unwrap();
        self.free[have].remove(&at);
        // Halve down to the size wanted, the upper halves left free.
        for o in (want..have).rev() {
            self.free[o].insert(at + (MIN << o));
        }
        self.used.insert(at, want as u32);
        self.in_use += MIN << want;
        at
    }

    pub fn release(&mut self, offset: u64) {
        let Some(order) = self.used.remove(&offset) else { return };
        let mut order = order as usize;
        let mut at = offset;
        self.in_use -= MIN << order;
        while order + 1 < self.free.len() {
            let buddy = at ^ (MIN << order);
            if !self.free[order].remove(&buddy) {
                break;
            }
            at = at.min(buddy);
            order += 1;
        }
        self.free[order].insert(at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_what_repeats_and_builds_what_is_new() {
        let mut c = SceneCache::default();
        c.begin();
        let (a, act) = c.place(7, 300);
        assert_eq!(act, BUILD);
        let (b, act) = c.place(7, 300);
        assert_eq!(act, BUILD);
        assert_ne!(a, b, "the same key twice in a frame is two draws");
        c.begin();
        assert_eq!(c.place(7, 300), (a, KEEP));
        assert_eq!(c.place(7, 300), (b, KEEP));
        c.begin();
        // A new vertex count is another shape: built anew, the old entry left to age out.
        let (n, act) = c.place(7, 303);
        assert_eq!(act, BUILD);
        assert_ne!(n, a);
    }

    #[test]
    fn refits_what_moves_for_a_while() {
        let mut c = SceneCache::default();
        c.begin();
        let (a, _) = c.place(1, 30);
        c.moved(a, false);
        for _ in 0..STICKY {
            c.begin();
            assert_eq!(c.place(1, 30), (a, REFIT));
        }
        c.begin();
        assert_eq!(c.place(1, 30), (a, KEEP));
    }

    #[test]
    fn builds_again_what_moved_far_or_was_refitted_long() {
        let mut c = SceneCache::default();
        c.begin();
        let (a, _) = c.place(1, 30);
        c.moved(a, true);
        c.begin();
        assert_eq!(c.place(1, 30), (a, BUILD));
        c.begin();
        assert_eq!(c.place(1, 30), (a, REFIT), "still moving, now near");
        for _ in 1..MAX_REFITS {
            c.moved(a, false);
            c.begin();
            assert_eq!(c.place(1, 30), (a, REFIT));
        }
        c.moved(a, false);
        c.begin();
        assert_eq!(c.place(1, 30), (a, BUILD));
    }

    #[test]
    fn evicts_what_is_not_drawn() {
        let mut c = SceneCache::default();
        c.begin();
        let (a, _) = c.place(1, 30);
        let (b, _) = c.place(2, 30);
        let mut out = Vec::new();
        for _ in 0..5 {
            c.begin();
            c.place(2, 30);
        }
        c.evict(4, &mut out);
        assert_eq!(out, vec![a]);
        assert_eq!(c.live(), 1);
        c.begin();
        // The freed slot is reused; the kept draw keeps its own.
        let (again, act) = c.place(1, 30);
        assert_eq!((again, act), (a, BUILD));
        assert_eq!(c.place(2, 30), (b, KEEP));
    }

    #[test]
    fn retains_still_draws_out_of_sight() {
        let mut c = SceneCache::default();
        c.begin();
        let (still, _) = c.place(1, 30);
        let (moving, _) = c.place(2, 30);
        c.moved(moving, false);
        c.begin();
        let mut out = Vec::new();
        c.retained(10, &mut out);
        assert_eq!(out, vec![still], "a moving draw out of sight is not where it was");
        c.place(1, 30);
        out.clear();
        c.retained(10, &mut out);
        assert!(out.is_empty(), "drawn this frame: placed, not retained");
        for _ in 0..11 {
            c.begin();
        }
        out.clear();
        c.retained(10, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn dropped_entries_build_again() {
        let mut c = SceneCache::default();
        c.begin();
        let (a, _) = c.place(9, 3);
        c.drop_entry(a);
        c.begin();
        assert_eq!(c.place(9, 3).1, BUILD);
    }

    #[test]
    fn arena_splits_and_merges() {
        let mut a = Arena::new(4096);
        let x = a.alloc(100);
        let y = a.alloc(300);
        let z = a.alloc(1024);
        assert!(x != NO_SPACE && y != NO_SPACE && z != NO_SPACE);
        assert_eq!(x % 256, 0);
        assert_eq!(y % 512, 0);
        assert_eq!(z % 1024, 0);
        assert_eq!(a.in_use(), 256 + 512 + 1024);
        assert_eq!(a.alloc(4096), NO_SPACE);
        a.release(x);
        a.release(y);
        a.release(z);
        assert_eq!(a.in_use(), 0);
        // All merged back: the whole arena is one block again.
        assert_eq!(a.alloc(4096), 0);
    }

    #[test]
    fn arena_fills_without_overlap() {
        let mut a = Arena::new(1 << 16);
        let mut taken = Vec::new();
        loop {
            let at = a.alloc(700);
            if at == NO_SPACE {
                break;
            }
            taken.push(at);
        }
        assert_eq!(taken.len(), (1 << 16) / 1024);
        taken.sort();
        for w in taken.windows(2) {
            assert!(w[1] - w[0] >= 1024);
        }
    }
}
