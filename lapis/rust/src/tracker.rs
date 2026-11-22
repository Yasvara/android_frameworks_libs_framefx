// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Engages the roles once S, the frame before the HUD, holds still for a run of frames.

use core::mem::{offset_of, size_of};

use crate::model::*;

/// framefx::lapis::TrackerState, field for field.
#[repr(C)]
pub struct Tracker {
    last: Roles,
    before: Roles, // the frame before `last` with another S, or just the one before
    have_before: bool,
    streak: i32,
    misses: i32,
    engaged: bool,
    width: u32,
    height: u32,
    miss: u8,
}

const _: () = assert!(size_of::<Tracker>() == 156 && offset_of!(Tracker, miss) == 152);

impl Tracker {
    pub fn reset(&mut self) {
        self.last = Roles::default();
        self.before = Roles::default();
        self.have_before = false;
        self.streak = 0;
        self.misses = 0;
        self.engaged = false;
        self.miss = MISS_NO_FRAME;
    }

    fn matches(&self, r: &Roles) -> bool {
        // The same structure, whichever images play the roles this frame: a game may rotate
        // through any number of them.
        if r.signature != 0 && self.last.signature != 0 {
            return r.signature == self.last.signature
                || !self.have_before
                || r.signature == self.before.signature;
        }
        // A game may show one frame drawn into the swapchain and the next copied into it.
        if r.direct != self.last.direct && !(self.have_before && r.direct == self.before.direct) {
            return false;
        }
        // Each role keeps its image, or takes the one it had two frames back.
        let same = |role: fn(&Roles) -> ImageId| {
            role(r) == role(&self.last) || !self.have_before || role(r) == role(&self.before)
        };
        let kept = |role: fn(&Roles) -> ImageId| role(r) == 0 || role(&self.last) == 0 || same(role);
        // V is not tapped: a game that moves it to another image keeps the run.
        (r.behind_blit || same(|r| r.scene)) && kept(|r| r.depth) && (r.direct || same(|r| r.target))
    }

    /// Whether this frame's `roles` may be tapped.
    pub fn update(&mut self, roles: &Roles, width: u32, height: u32, engage: i32, grace: i32) -> bool {
        if width != self.width || height != self.height {
            let resized = self.width != 0 || self.height != 0;
            self.reset();
            self.width = width;
            self.height = height;
            if resized {
                self.miss = MISS_RESIZED;
            }
        }
        if roles.scene == 0 || roles.miss != MISS_NONE {
            // Engaged or still learning: a game's generated frames may come between those with S.
            if self.streak > 0 && roles.miss != MISS_GAME_GENERATES {
                self.misses += 1;
                if self.misses <= grace {
                    return false;
                }
            }
            self.engaged = false;
            self.streak = 0;
            self.misses = 0;
            self.have_before = false;
            self.last = *roles;
            self.miss = if roles.miss != MISS_NONE { roles.miss } else { MISS_NO_SCENE };
            return false;
        }
        // A frame whose descriptors were reused before its submit hides what it read.
        if self.engaged && roles.finished && !self.matches(roles) {
            self.misses += 1;
            if self.misses <= grace {
                return false;
            }
        }
        self.misses = 0;
        if self.streak > 0 && !self.matches(roles) {
            // A game that was read changes screens: its new roles need one frame to agree, not
            // the whole run again.
            self.streak = if self.engaged { (engage - 2).max(0) } else { 0 };
            self.miss = MISS_CHANGED;
            self.engaged = false;
        }
        // Frames of one kind may come in runs of any length between the others.
        if !self.have_before || roles.scene != self.last.scene {
            self.before = self.last;
        }
        self.have_before = self.streak > 0;
        self.last = *roles;
        self.streak += 1;
        if self.streak >= engage {
            self.engaged = true;
            self.miss = MISS_NONE;
        }
        self.engaged
    }
}
