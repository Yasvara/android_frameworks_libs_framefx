// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! What one process runs of the Kelir sections Pentas publishes: the newest valid stack. A section
//! rejected at start leaves Kelir off; one rejected later keeps the plan running, so a bad write
//! never flashes the game unfiltered.

use crate::section::{decode_section, Stack, HEADER_SIZE};

/// Mirrored by framefx::kelir::ControllerCore; zeroed is a controller with Kelir off.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Controller {
    pub stack: Stack,
    pub valid: bool,
    pub applied: u32,
    pub rejected: u32,
    pub reason: u8,
}

pub const KEEP: u8 = 0;
/// Kelir is off: the caller drops its plan.
pub const OFF: u8 = 1;
/// The caller compiles `stack` into its new plan.
pub const COMPILE: u8 = 2;

/// What the caller does with its plan, and whether that changes what runs.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Update {
    pub action: u8,
    pub changed: bool,
}

/// `section`: the channel's Kelir section, None when it has none. `plan_empty`: the plan running
/// now draws nothing.
pub fn update(c: &mut Controller, section: Option<&[u8]>, plan_empty: bool) -> Update {
    let Some(section) = section else {
        let changed = c.valid || !plan_empty;
        c.valid = false;
        c.stack = Stack::default();
        return Update { action: OFF, changed };
    };
    match decode_section(section) {
        Err(r) => {
            // The revision sits at a fixed place: report it when the header got that far.
            c.rejected = if section.len() >= HEADER_SIZE {
                u32::from_le_bytes([section[8], section[9], section[10], section[11]])
            } else {
                0
            };
            c.reason = r as u8;
            Update { action: KEEP, changed: false }
        }
        Ok(next) => {
            if c.valid && next == c.stack {
                return Update { action: KEEP, changed: false };
            }
            c.stack = next;
            c.valid = true;
            c.applied = next.revision;
            Update { action: COMPILE, changed: true }
        }
    }
}

#[cfg(test)]
impl Controller {
    pub fn reject_reason(&self) -> crate::section::Reject {
        use crate::section::Reject;
        match self.reason {
            1 => Reject::Size,
            2 => Reject::Major,
            3 => Reject::Flags,
            4 => Reject::Count,
            5 => Reject::Reserved,
            6 => Reject::Kind,
            7 => Reject::UnusedParam,
            8 => Reject::ParamRange,
            9 => Reject::Repeat,
            10 => Reject::BandLimit,
            _ => Reject::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::section::{encode_section, Reject, MAX_SECTION_SIZE};

    #[test]
    fn a_bad_write_keeps_the_running_stack() {
        let mut c = Controller::default();
        let s = Stack { revision: 3, ..Default::default() };
        let mut buf = [0u8; MAX_SECTION_SIZE];
        let n = encode_section(&s, &mut buf);
        assert_eq!(update(&mut c, Some(&buf[..n]), true), Update { action: COMPILE, changed: true });
        assert_eq!(update(&mut c, Some(&buf[..n]), true), Update { action: KEEP, changed: false });
        buf[0] = 9;
        buf[8] = 4;
        assert_eq!(update(&mut c, Some(&buf[..n]), true).action, KEEP);
        assert_eq!((c.applied, c.rejected, c.reject_reason()), (3, 4, Reject::Major));
        assert_eq!(update(&mut c, None, true), Update { action: OFF, changed: true });
    }
}
