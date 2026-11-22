// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The adaptive shading-rate governor: how coarse the game's 3D work shades, from how well the
//! game keeps its own pace. Level 0 is full rate; each level up is the next coarser rate the user
//! allowed. It steps up as soon as a window of frames misses the pace, and back down only after
//! several windows in a row hold it, waiting twice as long each time a step down had to be undone.

/// Frames per decision.
pub const WINDOW: usize = 32;
/// A gap this long is a pause or a load, not a slow frame.
const PAUSE_US: u32 = 250_000;
const FIRST_NEED: u8 = 4;
const MAX_NEED: u8 = 32;

/// Mirrored by framefx::lapis::ShadingGovernor in C++; zeroed is a fresh governor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShadingGovernor {
    intervals: [u32; WINDOW],
    last_ns: i64,
    count: u32,
    /// The game's own pace, in microseconds: the fastest whole number of refreshes it held.
    target_us: u32,
    level: u8,
    good: u8,
    need: u8,
    /// Windows since the last step; a step down undone within two is an oscillation.
    since_step: u8,
    stepped_down: u8,
    _pad: [u8; 3],
}

impl ShadingGovernor {
    #[cfg(test)]
    pub fn level(&self) -> u8 {
        self.level
    }

    #[cfg(test)]
    pub fn target_us(&self) -> u32 {
        self.target_us
    }

    /// One real frame of the game, presented at `now_ns`. `refresh_ns` is the panel's period (0
    /// when unknown); `max_level` the coarsest level allowed. Returns the level to shade at.
    pub fn frame(&mut self, now_ns: i64, refresh_ns: i64, max_level: u8) -> u8 {
        if self.need == 0 {
            self.need = FIRST_NEED;
        }
        let last = self.last_ns;
        self.last_ns = now_ns;
        if last == 0 || now_ns <= last {
            return self.clamp(max_level);
        }
        let gap = ((now_ns - last) / 1000).min(u32::MAX as i64) as u32;
        if gap >= PAUSE_US {
            self.count = 0;
            return self.clamp(max_level);
        }
        self.intervals[self.count as usize % WINDOW] = gap;
        self.count += 1;
        if (self.count as usize).is_multiple_of(WINDOW) {
            self.decide(refresh_ns, max_level);
        }
        self.clamp(max_level)
    }

    fn clamp(&mut self, max_level: u8) -> u8 {
        self.level = self.level.min(max_level);
        self.level
    }

    fn decide(&mut self, refresh_ns: i64, max_level: u8) {
        let mut sorted = self.intervals;
        sorted.sort_unstable();
        let median = sorted[WINDOW / 2];
        let p90 = sorted[WINDOW * 9 / 10];
        let refresh_us = (refresh_ns / 1000) as u32;
        let paced = if refresh_us > 0 {
            let n = ((median + refresh_us / 2) / refresh_us).max(1);
            n * refresh_us
        } else {
            median
        };
        // The pace drifts up slowly, so a game that lowers its own cap is followed.
        self.target_us = if self.target_us == 0 {
            paced
        } else {
            paced.min(self.target_us + self.target_us / 100)
        };
        self.since_step = self.since_step.saturating_add(1);
        let target = self.target_us as u64;
        if (p90 as u64) * 100 > target * 110 {
            self.good = 0;
            if self.level < max_level {
                if self.stepped_down != 0 && self.since_step <= 2 {
                    self.need = (self.need.saturating_mul(2)).min(MAX_NEED);
                }
                self.level += 1;
                self.since_step = 0;
                self.stepped_down = 0;
            }
        } else if (p90 as u64) * 100 <= target * 104 {
            self.good = self.good.saturating_add(1);
            if self.level > 0 && self.good >= self.need {
                self.level -= 1;
                self.good = 0;
                self.since_step = 0;
                self.stepped_down = 1;
            }
        } else {
            self.good = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFRESH: i64 = 8_333_333;

    /// A governor that has seen its first present, so every later frame is one interval.
    fn fresh(t: i64) -> ShadingGovernor {
        let mut g = ShadingGovernor::default();
        g.frame(t, REFRESH, 0);
        g
    }

    fn run(g: &mut ShadingGovernor, t: &mut i64, frames: usize, interval_ns: i64, max: u8) -> u8 {
        let mut level = 0;
        for _ in 0..frames {
            *t += interval_ns;
            level = g.frame(*t, REFRESH, max);
        }
        level
    }

    #[test]
    fn steady_game_stays_at_full_rate() {
        let mut t = 1_000_000_000;
        let mut g = fresh(t);
        assert_eq!(run(&mut g, &mut t, 600, 16_666_666, 2), 0);
        assert_eq!(g.target_us(), 16_666);
    }

    #[test]
    fn missing_the_pace_steps_up_to_the_allowed_level() {
        let mut t = 1_000_000_000;
        let mut g = fresh(t);
        run(&mut g, &mut t, 320, 16_666_666, 2);
        assert_eq!(run(&mut g, &mut t, 64, 22_000_000, 2), 2);
        assert_eq!(run(&mut g, &mut t, 320, 22_000_000, 1), 1);
    }

    #[test]
    fn holding_the_pace_steps_back_down() {
        let mut t = 1_000_000_000;
        let mut g = fresh(t);
        run(&mut g, &mut t, 320, 16_666_666, 2);
        run(&mut g, &mut t, 64, 22_000_000, 2);
        assert_eq!(g.level(), 2);
        assert_eq!(run(&mut g, &mut t, WINDOW * 4, 16_666_666, 2), 1);
        assert_eq!(run(&mut g, &mut t, WINDOW * 4, 16_666_666, 2), 0);
    }

    #[test]
    fn an_undone_step_down_waits_longer() {
        let mut t = 1_000_000_000;
        let mut g = fresh(t);
        run(&mut g, &mut t, 320, 16_666_666, 1);
        run(&mut g, &mut t, WINDOW, 22_000_000, 1);
        assert_eq!(g.level(), 1);
        run(&mut g, &mut t, WINDOW * 4, 16_666_666, 1);
        assert_eq!(g.level(), 0);
        run(&mut g, &mut t, WINDOW, 22_000_000, 1);
        assert_eq!(g.level(), 1);
        assert_eq!(g.need, 8);
        assert_eq!(run(&mut g, &mut t, WINDOW * 4, 16_666_666, 1), 1);
        assert_eq!(run(&mut g, &mut t, WINDOW * 4, 16_666_666, 1), 0);
    }

    #[test]
    fn pauses_do_not_count_as_slow_frames() {
        let mut t = 1_000_000_000;
        let mut g = fresh(t);
        run(&mut g, &mut t, 320, 16_666_666, 2);
        t += 2_000_000_000;
        g.frame(t, REFRESH, 2);
        assert_eq!(run(&mut g, &mut t, 64, 16_666_666, 2), 0);
    }
}
