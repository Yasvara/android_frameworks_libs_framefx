// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The Kelir section of a game's channel (framefx/channel.h, "Kelir section v1"): strict, so
//! anything but a well-formed v1 section is rejected whole.

use crate::params::*;

pub const HEADER_SIZE: usize = 16;
pub const NODE_SIZE: usize = 20;
pub const MAX_NODES: usize = 8;
pub const MAX_SECTION_SIZE: usize = 256;
pub const MAJOR: u8 = 1;
pub const PEEK: u8 = 1 << 0;
pub const SPLIT: u8 = 1 << 1;
pub const INSPECT: u8 = 1 << 2;
const HEADER_FLAGS: u8 = PEEK | SPLIT | INSPECT;
const NODE_ENABLED: u8 = 1 << 0;
const MAX_DETAIL: i32 = 3;
const MAX_STYLE: i32 = 1;
const MAX_SCREEN: i32 = 4;
const MAX_RETRO: i32 = 1;

/// framefx::KelirReject.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    None = 0,
    Size = 1,
    Major = 2,
    Flags = 3,
    Count = 4,
    Reserved = 5,
    Kind = 6,
    UnusedParam = 7,
    ParamRange = 8,
    Repeat = 9,
    BandLimit = 10,
}

/// Mirrors framefx::kelir::Node.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Node {
    pub kind: u8,
    pub enabled: bool,
    /// unorm
    pub intensity: u16,
    pub q: [i16; MAX_PARAMS],
}

impl Node {
    pub fn strength(&self) -> f32 {
        self.intensity as f32 / 65535.0
    }
    pub fn value(&self, i: usize) -> f32 {
        decode(&info(self.kind).params[i], self.q[i])
    }
}

/// Mirrors framefx::kelir::Stack.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Stack {
    pub minor: u8,
    pub peek: bool,
    pub split: bool,
    pub inspect: bool,
    /// unorm, display space
    pub split_x: u16,
    pub revision: u32,
    pub count: i32,
    pub nodes: [Node; MAX_NODES],
}

impl PartialEq for Stack {
    fn eq(&self, o: &Self) -> bool {
        let n = self.count.clamp(0, MAX_NODES as i32) as usize;
        self.minor == o.minor
            && self.peek == o.peek
            && self.split == o.split
            && self.inspect == o.inspect
            && self.split_x == o.split_x
            && self.revision == o.revision
            && self.count == o.count
            && self.nodes[..n] == o.nodes[..n]
    }
}

fn u16_at(d: &[u8], i: usize) -> u16 {
    d[i] as u16 | (d[i + 1] as u16) << 8
}

fn u32_at(d: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]])
}

fn decode_node(d: &[u8]) -> Result<Node, Reject> {
    let kind = d[0];
    if !known(kind as i32) {
        return Err(Reject::Kind);
    }
    if d[1] & !NODE_ENABLED != 0 {
        return Err(Reject::Flags);
    }
    let info = info(kind);
    let mut n = Node { kind, enabled: d[1] & NODE_ENABLED != 0, intensity: u16_at(d, 2), q: [0; MAX_PARAMS] };
    for i in 0..MAX_PARAMS {
        n.q[i] = u16_at(d, 4 + 2 * i) as i16;
        if i >= info.param_count as usize {
            if n.q[i] != 0 {
                return Err(Reject::UnusedParam);
            }
        } else if !code_valid(&info.params[i], n.q[i]) {
            return Err(Reject::ParamRange);
        }
    }
    Ok(n)
}

/// The cost envelope: bands, styles, Retro, and kinds that may not repeat.
fn check_limits(s: &Stack) -> Reject {
    let (mut detail, mut style, mut screen, mut retro) = (0, 0, 0, 0);
    let mut seen = 0u32;
    for n in &s.nodes[..s.count as usize] {
        let info = info(n.kind);
        if info.band != BAND_COLOUR {
            if seen & (1 << n.kind) != 0 {
                return Reject::Repeat;
            }
            seen |= 1 << n.kind;
        }
        detail += (info.band == BAND_DETAIL) as i32;
        style += info.style as i32;
        screen += (info.band == BAND_SCREEN) as i32;
        retro += (n.kind == RETRO) as i32;
    }
    if detail > MAX_DETAIL || style > MAX_STYLE || screen > MAX_SCREEN || retro > MAX_RETRO {
        return Reject::BandLimit;
    }
    Reject::None
}

pub fn decode_section(d: &[u8]) -> Result<Stack, Reject> {
    if d.len() < HEADER_SIZE || d.len() > MAX_SECTION_SIZE {
        return Err(Reject::Size);
    }
    if d[0] != MAJOR {
        return Err(Reject::Major);
    }
    if d[2] & !HEADER_FLAGS != 0 {
        return Err(Reject::Flags);
    }
    let count = d[3] as usize;
    if count > MAX_NODES {
        return Err(Reject::Count);
    }
    if d.len() != HEADER_SIZE + NODE_SIZE * count {
        return Err(Reject::Size);
    }
    if u16_at(d, 6) != 0 || u32_at(d, 12) != 0 {
        return Err(Reject::Reserved);
    }
    let mut s = Stack {
        minor: d[1],
        peek: d[2] & PEEK != 0,
        split: d[2] & SPLIT != 0,
        inspect: d[2] & INSPECT != 0,
        split_x: u16_at(d, 4),
        revision: u32_at(d, 8),
        count: count as i32,
        nodes: [Node::default(); MAX_NODES],
    };
    for i in 0..count {
        let at = HEADER_SIZE + NODE_SIZE * i;
        s.nodes[i] = decode_node(&d[at..at + NODE_SIZE])?;
    }
    match check_limits(&s) {
        Reject::None => Ok(s),
        r => Err(r),
    }
}

/// The section for a stack, HEADER_SIZE + NODE_SIZE * count bytes into `out`.
pub fn encode_section(s: &Stack, out: &mut [u8]) -> usize {
    let count = s.count.clamp(0, MAX_NODES as i32) as usize;
    let size = HEADER_SIZE + NODE_SIZE * count;
    out[..size].fill(0);
    out[0] = MAJOR;
    out[1] = s.minor;
    out[2] = (if s.peek { PEEK } else { 0 }) | (if s.split { SPLIT } else { 0 }) | (if s.inspect { INSPECT } else { 0 });
    out[3] = count as u8;
    out[4..6].copy_from_slice(&s.split_x.to_le_bytes());
    out[8..12].copy_from_slice(&s.revision.to_le_bytes());
    for (i, node) in s.nodes[..count].iter().enumerate() {
        let n = &mut out[HEADER_SIZE + NODE_SIZE * i..HEADER_SIZE + NODE_SIZE * (i + 1)];
        n[0] = node.kind;
        n[1] = if node.enabled { NODE_ENABLED } else { 0 };
        n[2..4].copy_from_slice(&node.intensity.to_le_bytes());
        for j in 0..MAX_PARAMS {
            n[4 + 2 * j..6 + 2 * j].copy_from_slice(&node.q[j].to_le_bytes());
        }
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stack_round_trips() {
        let mut s = Stack { minor: 2, split: true, split_x: 30000, revision: 7, count: 2, ..Default::default() };
        s.nodes[0] = Node { kind: COLOR, enabled: true, intensity: 65535, q: [1000, 0, 0, -2000, 0, 0, 0, 0] };
        s.nodes[1] = Node { kind: VIGNETTE, enabled: true, intensity: 40000, q: [0; MAX_PARAMS] };
        let mut buf = [0u8; MAX_SECTION_SIZE];
        let n = encode_section(&s, &mut buf);
        assert_eq!(n, HEADER_SIZE + 2 * NODE_SIZE);
        assert_eq!(decode_section(&buf[..n]), Ok(s));
    }

    #[test]
    fn rejects_whole() {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0] = 2;
        assert_eq!(decode_section(&buf), Err(Reject::Major));
        buf[0] = 1;
        buf[12] = 1;
        assert_eq!(decode_section(&buf), Err(Reject::Reserved));
        assert_eq!(decode_section(&buf[..3]), Err(Reject::Size));
    }
}
