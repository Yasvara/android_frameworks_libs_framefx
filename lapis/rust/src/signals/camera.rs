// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The scene camera among the constants the scene's draws read: which matrix slot holds the
//! view-projection the game draws with, which holds the same without jitter, and which holds the
//! previous frame's, learned from how the slots relate from one frame to the next. Nothing about a
//! game is assumed beyond the shape tests in `matrix`.

use super::matrix::{self, CameraFit, Mat4};

/// A uniform block a draw read: `slot` names it across frames (set and binding, never the address,
/// which a ring moves), `address` is where its bytes were this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BlockSlot {
    pub set: u32,
    pub binding: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub slot: BlockSlot,
    /// Memory object (opaque) and byte offset of the block's first float in it.
    pub memory: u64,
    pub offset: u64,
    pub floats: Vec<f32>,
}

/// One matrix position: a block and the float it starts at, with the layout it fits in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MatrixSlot {
    pub block: BlockSlot,
    pub float: u32,
    pub row_major: bool,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    slot: MatrixSlot,
    fit: CameraFit,
    memory: u64,
    offset: u64,
}

/// What a slot has been seen to hold, frame over frame.
#[derive(Clone, Copy, Debug, Default)]
struct Votes {
    seen: u32,
    /// Changed while another slot of the same lens stayed: the scene camera moves through it.
    moving: u32,
    /// Equal to the jittered slot with its offset taken out.
    unjittered: u32,
    /// Equal to what a current slot held the frame before.
    previous: u32,
}

/// Where the next frame can find the camera in memory: for a ring, the address moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CameraAddress {
    pub memory: u64,
    pub offset: u64,
    pub row_major: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CameraFrame {
    pub found: bool,
    /// As the scene's draws use it, row-major, jitter included.
    pub jittered: Mat4,
    /// Without jitter: the game's own unjittered matrix when it keeps one, else ours.
    pub unjittered: Mat4,
    /// NDC offset jitter adds; `jitter_exact` when measured against the game's unjittered matrix.
    pub jitter: [f32; 2],
    pub jitter_exact: bool,
    /// The game's own previous-frame matrix when it keeps one (unjittered), else ours.
    pub previous: Mat4,
    pub previous_valid: bool,
    pub previous_from_game: bool,
    pub sx: f32,
    pub sy: f32,
    pub world: bool,
    /// Camera motion since the frame before: view axis turn (deg), eye travel (world units), and
    /// the lens's relative focal change.
    pub turn: f32,
    pub travel: f32,
    pub zoom: f32,
    pub address: CameraAddress,
    pub slot: MatrixSlot,
    /// 0..1: how settled the slot labels are.
    pub confidence: f32,
}

const SETTLE: u32 = 30;
const MATRIX_SLACK: f32 = 1e-5;

fn close(a: &Mat4, b: &Mat4, slack: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= slack * (1.0 + x.abs().max(y.abs())))
}

/// Same w and z rows: the same camera, maybe with another NDC offset.
fn same_but_offset(a: &Mat4, b: &Mat4) -> bool {
    close(&[a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          &[b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          MATRIX_SLACK)
}

#[derive(Default)]
pub struct CameraTracker {
    votes: Vec<(MatrixSlot, Votes)>,
    last: Vec<Candidate>,
    current: Option<MatrixSlot>,
    last_frame: Option<CameraFrame>,
    streak: u32,
    /// A slot a profile names: favoured while it fits, never forced.
    prior: Option<MatrixSlot>,
}

/// Every view-projection in `blocks` whose focal scales are `aspect` apart (0: any).
fn candidates(blocks: &[Block], aspect: f32) -> Vec<Candidate> {
    let mut out = Vec::new();
    for b in blocks {
        let mut i = 0;
        while i + 16 <= b.floats.len() {
            let f = &b.floats[i..i + 16];
            // A view-projection's w row is a unit vector: most windows fail that in either layout,
            // and testing it first skips the full fit on them (NaN fails too).
            let unit = |v: f32| (v - 1.0).abs() <= 0.02;
            let rows = unit(f[12] * f[12] + f[13] * f[13] + f[14] * f[14]);
            let cols = unit(f[3] * f[3] + f[7] * f[7] + f[11] * f[11]);
            if !rows && !cols {
                i += 4;
                continue;
            }
            let fit = rows
                .then(|| matrix::fit_view_projection(f, true, aspect))
                .flatten()
                .or_else(|| cols.then(|| matrix::fit_view_projection(f, false, aspect)).flatten());
            if let Some(fit) = fit {
                out.push(Candidate {
                    slot: MatrixSlot { block: b.slot, float: i as u32, row_major: fit.row_major },
                    fit,
                    memory: b.memory,
                    offset: b.offset + 4 * i as u64,
                });
                // A matrix does not overlap the next one.
                i += 16;
                continue;
            }
            i += 4;
        }
    }
    out
}

impl CameraTracker {
    pub fn reset(&mut self) {
        *self = CameraTracker { prior: self.prior, ..Default::default() };
    }

    pub fn prefer(&mut self, slot: Option<MatrixSlot>) {
        self.prior = slot;
    }

    pub fn chosen(&self) -> Option<MatrixSlot> {
        self.current
    }

    fn votes(&mut self, slot: MatrixSlot) -> &mut Votes {
        if let Some(k) = self.votes.iter().position(|(s, _)| *s == slot) {
            return &mut self.votes[k].1;
        }
        self.votes.push((slot, Votes::default()));
        &mut self.votes.last_mut().unwrap().1
    }

    fn vote_of(&self, slot: MatrixSlot) -> Votes {
        self.votes.iter().find(|(s, _)| *s == slot).map(|(_, v)| *v).unwrap_or_default()
    }

    /// One frame's constants of the pass that draws the scene; `aspect` the scene's.
    pub fn update(&mut self, blocks: &[Block], aspect: f32) -> CameraFrame {
        let now = candidates(blocks, aspect);
        let last = core::mem::take(&mut self.last);

        // Relations with the frame before, as votes.
        for c in &now {
            let before = last.iter().find(|l| l.slot == c.slot);
            let moved = before.is_some_and(|b| !close(&b.fit.m, &c.fit.m, MATRIX_SLACK));
            let was_current = last.iter().any(|l| {
                Some(l.slot) == self.current
                    && (close(&matrix::without_offset(&l.fit.m, l.fit.jx, l.fit.jy), &c.fit.m, MATRIX_SLACK)
                        || close(&l.fit.m, &c.fit.m, MATRIX_SLACK))
            });
            let unjittered_of = now.iter().any(|o| {
                o.slot != c.slot && same_but_offset(&o.fit.m, &c.fit.m) && (o.fit.jx != c.fit.jx || o.fit.jy != c.fit.jy)
                    && c.fit.jx.abs() + c.fit.jy.abs() < o.fit.jx.abs() + o.fit.jy.abs()
            });
            let v = self.votes(c.slot);
            v.seen += 1;
            v.moving += moved as u32;
            v.unjittered += unjittered_of as u32;
            // A slot equal to the camera's slot last frame while that slot moved on.
            v.previous += (was_current && moved) as u32;
        }

        // The scene camera: a world-space matrix that moves, is not the previous frame's copy and is
        // not the unjittered twin; the one already chosen wins ties.
        let score = |t: &Self, c: &Candidate| -> i64 {
            let v = t.vote_of(c.slot);
            let mut s = i64::from(v.seen) + 4 * i64::from(v.moving);
            s -= 8 * i64::from(v.previous) + 8 * i64::from(v.unjittered);
            if c.fit.world {
                s += 1000;
            }
            if Some(c.slot) == t.current {
                s += 2;
            }
            if Some(c.slot) == t.prior {
                s += 3;
            }
            s
        };
        let chosen = now.iter().max_by_key(|c| score(self, c)).copied();
        let mut out = CameraFrame::default();
        let Some(cur) = chosen else {
            self.current = None;
            self.streak = 0;
            self.last_frame = None;
            self.last = now;
            return out;
        };
        if Some(cur.slot) != self.current {
            // A slot the profile named starts half trusted; the frames still have to agree.
            self.streak = if Some(cur.slot) == self.prior { SETTLE / 2 } else { 0 };
        }
        self.current = Some(cur.slot);
        self.streak += 1;

        out.found = true;
        out.jittered = cur.fit.m;
        out.sx = cur.fit.sx;
        out.sy = cur.fit.sy;
        out.world = cur.fit.world;
        out.slot = cur.slot;
        out.address = CameraAddress { memory: cur.memory, offset: cur.offset, row_major: cur.slot.row_major };

        // The game's own unjittered twin gives the exact jitter.
        let twin = now
            .iter()
            .filter(|o| o.slot != cur.slot && same_but_offset(&o.fit.m, &cur.fit.m))
            .filter(|o| self.vote_of(o.slot).unjittered * 2 >= self.vote_of(o.slot).seen.max(1))
            .min_by(|a, b| (a.fit.jx.abs() + a.fit.jy.abs()).total_cmp(&(b.fit.jx.abs() + b.fit.jy.abs())));
        if let Some(t) = twin {
            out.jitter = [cur.fit.jx - t.fit.jx, cur.fit.jy - t.fit.jy];
            out.unjittered = t.fit.m;
            out.jitter_exact = true;
        } else {
            out.jitter = [cur.fit.jx, cur.fit.jy];
            out.unjittered = matrix::without_offset(&cur.fit.m, cur.fit.jx, cur.fit.jy);
        }

        // The game's previous-frame matrix, where it keeps one: it beats ours across a missed frame.
        let prev_slot = now
            .iter()
            .filter(|o| o.slot != cur.slot)
            .filter(|o| {
                let v = self.vote_of(o.slot);
                v.previous >= 2 && v.previous * 2 >= v.moving.max(1)
            })
            .max_by_key(|o| self.vote_of(o.slot).previous);
        if let Some(p) = prev_slot {
            out.previous = matrix::without_offset(&p.fit.m, p.fit.jx, p.fit.jy);
            out.previous_valid = true;
            out.previous_from_game = true;
        } else if let Some(l) = self.last_frame.filter(|l| l.slot == cur.slot) {
            out.previous = l.unjittered;
            out.previous_valid = true;
        }

        if let Some(l) = self.last_frame.filter(|l| l.found) {
            let (a, b) = (axis_of(&l.unjittered), axis_of(&out.unjittered));
            out.turn = matrix::axis_angle(a, b);
            out.travel = match (matrix::eye(&l.unjittered), matrix::eye(&out.unjittered)) {
                (Some(p), Some(q)) if l.world && out.world => {
                    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
                }
                _ => 0.0,
            };
            out.zoom = if l.sy > 0.0 { (out.sy - l.sy).abs() / l.sy } else { 0.0 };
        }
        let labelled = if out.jitter_exact || out.previous_from_game { 1.0 } else { 0.8 };
        out.confidence = labelled * (self.streak.min(SETTLE) as f32 / SETTLE as f32);
        self.last_frame = Some(out);
        self.last = now;
        out
    }
}

fn axis_of(m: &Mat4) -> [f32; 3] {
    let l = (m[12] * m[12] + m[13] * m[13] + m[14] * m[14]).sqrt().max(1e-9);
    [m[12] / l, m[13] / l, m[14] / l]
}

#[cfg(test)]
mod tests {
    use super::super::matrix::test_util::*;
    use super::super::matrix::mul;
    use super::*;

    const ASPECT: f32 = 2.16;

    fn block(binding: u32, offset: u64, mats: &[(usize, Mat4, bool)], len: usize) -> Block {
        let mut floats = vec![0.25f32; len];
        for (at, m, row) in mats {
            let m = if *row { *m } else { transpose(m) };
            floats[*at..*at + 16].copy_from_slice(&m);
        }
        Block { slot: BlockSlot { set: 0, binding }, memory: 7, offset, floats }
    }

    fn camera(i: usize) -> Mat4 {
        mul(&perspective(45.0, ASPECT, 0.25), &view([0.1 * i as f32, 1.5, -0.05 * i as f32], 0.7 * i as f32))
    }

    fn halton(i: usize, base: usize) -> f32 {
        let (mut f, mut r, mut n) = (1.0f32, 0.0f32, i);
        while n > 0 {
            f /= base as f32;
            r += f * (n % base) as f32;
            n /= base;
        }
        r
    }

    fn jitter(i: usize) -> (f32, f32) {
        ((halton(i % 8 + 1, 2) - 0.5) * 2.0 / 1662.0, (halton(i % 8 + 1, 3) - 0.5) * 2.0 / 768.0)
    }

    #[test]
    fn learns_jittered_unjittered_and_previous_slots_like_unity_keeps_them() {
        let mut t = CameraTracker::default();
        let mut out = CameraFrame::default();
        let mut block_at = 0;
        for i in 0..40 {
            let (jx, jy) = jitter(i);
            let cur = camera(i);
            block_at = 256 * (i % 3) as u64;
            let prev = camera(i.saturating_sub(1));
            // Column-major as GLSL uploads it; the ring moves the block every frame.
            let b = block(1, block_at, &[(16, jittered(&cur, jx, jy), false), (48, cur, false), (80, prev, false)], 128);
            out = t.update(&[b], ASPECT);
            assert!(out.found);
            assert_eq!(out.slot.float, 16, "frame {i}");
        }
        let (jx, jy) = jitter(39);
        assert!(out.jitter_exact && (out.jitter[0] - jx).abs() < 1e-7 && (out.jitter[1] - jy).abs() < 1e-7);
        assert!(out.previous_from_game && out.previous_valid);
        assert!(close(&out.previous, &camera(38), 1e-5));
        assert!(close(&out.unjittered, &camera(39), 1e-5));
        assert_eq!(out.address.offset, block_at + 64);
        assert!(out.confidence > 0.99);
        assert!(out.turn > 0.6 && out.turn < 0.8 && out.travel > 0.1);
    }

    #[test]
    fn a_lone_jittered_matrix_still_gives_jitter_and_its_own_previous() {
        let mut t = CameraTracker::default();
        let mut out = CameraFrame::default();
        for i in 0..5 {
            let (jx, jy) = jitter(i);
            out = t.update(&[block(0, 0, &[(0, jittered(&camera(i), jx, jy), true)], 64)], ASPECT);
        }
        let (jx, jy) = jitter(4);
        assert!(!out.jitter_exact && (out.jitter[0] - jx).abs() < 1e-6 && (out.jitter[1] - jy).abs() < 1e-6);
        assert!(out.previous_valid && !out.previous_from_game);
        assert!(close(&out.previous, &camera(3), 1e-4));
    }

    #[test]
    fn a_shadow_or_ui_matrix_is_never_the_camera_and_nothing_found_is_said() {
        let ortho = [0.01, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.001, 0.5, 0.0, 0.0, 0.0, 1.0];
        let mut t = CameraTracker::default();
        assert!(!t.update(&[block(0, 0, &[(0, ortho, true)], 64)], ASPECT).found);
        let out = t.update(&[block(0, 0, &[(0, ortho, true), (16, camera(1), true)], 64)], ASPECT);
        assert!(out.found && out.slot.float == 16);
    }
}
