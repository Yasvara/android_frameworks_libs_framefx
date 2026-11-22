// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The game's vertex shaders with their clip position captured by transform feedback: buffer 0,
//! 16 bytes a vertex, exactly what the rasterizer gets. Nothing else in the module changes, so a
//! pipeline built from the copy draws as the original does while capture is off.
//!
//! Refused, and the original kept: anything but one vertex entry point, a module that already
//! uses transform feedback, a position the module never declares, or SPIR-V this reader does not
//! follow.

const MAGIC: u32 = 0x0723_0203;
const HEADER: usize = 5;

const OP_CAPABILITY: u32 = 17;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_EXECUTION_MODE_ID: u32 = 331;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;

const CAPABILITY_TRANSFORM_FEEDBACK: u32 = 53;
const MODEL_VERTEX: u32 = 0;
const MODE_XFB: u32 = 11;
const STORAGE_OUTPUT: u32 = 3;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_XFB_BUFFER: u32 = 36;
const DECORATION_XFB_STRIDE: u32 = 37;
const BUILT_IN_POSITION: u32 = 0;

/// Bytes a captured vertex takes: the clip position, a vec4 of float.
pub const STRIDE: u32 = 16;
/// A vertex's bytes with a varying captured beside its position, at `STRIDE`.
pub const STRIDE_WITH_VARYING: u32 = 32;

/// Why a module keeps its original code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Refusal {
    Malformed = 1,
    NotVertex = 2,
    Captures = 3,
    NoPosition = 4,
    Offsets = 5,
    /// The varying asked for is not a float vector output of its own.
    NoVarying = 6,
}

struct Insn {
    at: usize,
    op: u32,
    len: usize,
}

fn insns(words: &[u32]) -> Result<Vec<Insn>, Refusal> {
    if words.len() < HEADER || words[0] != MAGIC {
        return Err(Refusal::Malformed);
    }
    let mut out = Vec::new();
    let mut at = HEADER;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        if len == 0 || at + len > words.len() {
            return Err(Refusal::Malformed);
        }
        out.push(Insn { at, op: words[at] & 0xffff, len });
        at += len;
    }
    Ok(out)
}

/// Whether `words` is a module with one entry point, a vertex shader.
pub fn is_vertex(words: &[u32]) -> bool {
    let Ok(list) = insns(words) else { return false };
    let mut entries = list.iter().filter(|i| i.op == OP_ENTRY_POINT);
    matches!((entries.next(), entries.next()), (Some(e), None) if e.len >= 3 && words[e.at + 1] == MODEL_VERTEX)
}

const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_ARRAY: u32 = 28;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_RETURN: u32 = 253;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_COMPONENT: u32 = 31;
/// Locations a vertex shader may write (maxVertexOutputComponents / 4 on every GPU we arm).
const MAX_LOCATIONS: u32 = 32;

fn op(code: u32, len: usize) -> u32 {
    ((len as u32) << 16) | code
}

/// Where a declaration's result id sits, by opcode.
fn result_of(words: &[u32], i: &Insn) -> Option<u32> {
    match i.op {
        OP_TYPE_INT | OP_TYPE_FLOAT | OP_TYPE_VECTOR | OP_TYPE_MATRIX | OP_TYPE_ARRAY
        | OP_TYPE_STRUCT | OP_TYPE_POINTER => (i.len >= 2).then(|| words[i.at + 1]),
        OP_CONSTANT | OP_VARIABLE => (i.len >= 3).then(|| words[i.at + 2]),
        _ => None,
    }
}

/// The capturing copy of `words` into `out`. The position goes out a second time, through a
/// varying of its own at a free location that transform feedback captures: Adreno refuses to
/// capture a member of a gl_PerVertex that holds anything besides the position, which every
/// engine's does. Before each return of the entry point, that varying takes the position's value.
#[cfg(test)]
pub fn patch(words: &[u32], out: &mut Vec<u32>) -> Result<(), Refusal> {
    patch_with(words, out, None)
}

/// The same, with the output at location `varying` captured too, 16 bytes past the position: the
/// coordinate an alpha test samples at, so the rays can sample there as well.
pub fn patch_with(words: &[u32], out: &mut Vec<u32>, varying: Option<u32>) -> Result<(), Refusal> {
    let list = insns(words)?;
    let entries: Vec<&Insn> = list.iter().filter(|i| i.op == OP_ENTRY_POINT).collect();
    let [entry] = entries.as_slice() else { return Err(Refusal::NotVertex) };
    if entry.len < 3 || words[entry.at + 1] != MODEL_VERTEX {
        return Err(Refusal::NotVertex);
    }
    let main = words[entry.at + 2];
    if list.iter().any(|i| {
        (i.op == OP_CAPABILITY && i.len == 2 && words[i.at + 1] == CAPABILITY_TRANSFORM_FEEDBACK)
            || (i.op == OP_EXECUTION_MODE && i.len >= 3 && words[i.at + 2] == MODE_XFB)
    }) {
        return Err(Refusal::Captures);
    }
    let def = |id: u32| list.iter().find(|i| result_of(words, i) == Some(id));

    // Where the position lives: a block member (gl_PerVertex) or a variable of its own.
    let mut block: Option<(u32, u32)> = None;
    let mut variable: Option<u32> = None;
    for i in &list {
        if i.op == OP_MEMBER_DECORATE && i.len >= 5 && words[i.at + 3] == DECORATION_BUILT_IN
            && words[i.at + 4] == BUILT_IN_POSITION
        {
            block = Some((words[i.at + 1], words[i.at + 2]));
        }
        if i.op == OP_DECORATE && i.len >= 4 && words[i.at + 2] == DECORATION_BUILT_IN
            && words[i.at + 3] == BUILT_IN_POSITION
        {
            variable = Some(words[i.at + 1]);
        }
    }
    let interface = &words[entry.at + 3..entry.at + entry.len];
    let output_pointer_to = |ty: u32| {
        list.iter()
            .find(|i| {
                i.op == OP_TYPE_POINTER && i.len == 4 && words[i.at + 2] == STORAGE_OUTPUT
                    && words[i.at + 3] == ty
            })
            .map(|i| words[i.at + 1])
    };
    let (target, vec4) = match (block, variable) {
        (Some((ty, member)), _) => {
            let st = def(ty).filter(|i| i.op == OP_TYPE_STRUCT).ok_or(Refusal::Malformed)?;
            if st.len < 3 + member as usize {
                return Err(Refusal::Malformed);
            }
            let vec4 = words[st.at + 2 + member as usize];
            let pointer = output_pointer_to(ty).ok_or(Refusal::NoPosition)?;
            let var = list
                .iter()
                .filter(|i| i.op == OP_VARIABLE && i.len >= 4 && words[i.at + 3] == STORAGE_OUTPUT)
                .map(|i| (words[i.at + 1], words[i.at + 2]))
                .find(|(ptr, id)| *ptr == pointer && interface.contains(id))
                .map(|(_, id)| id)
                .ok_or(Refusal::NoPosition)?;
            (var, vec4)
        }
        (None, Some(v)) if interface.contains(&v) => {
            let var = def(v).filter(|i| i.op == OP_VARIABLE).ok_or(Refusal::Malformed)?;
            let pointer = def(words[var.at + 1]).filter(|i| i.op == OP_TYPE_POINTER && i.len == 4);
            let pointer = pointer.ok_or(Refusal::Malformed)?;
            (v, words[pointer.at + 3])
        }
        _ => return Err(Refusal::NoPosition),
    };
    // A vec4 of 32-bit floats, as the rasterizer takes it.
    let v = def(vec4).filter(|i| i.op == OP_TYPE_VECTOR && i.len == 4).ok_or(Refusal::Malformed)?;
    let f = def(words[v.at + 2]).filter(|i| i.op == OP_TYPE_FLOAT && i.len >= 3);
    if words[v.at + 3] != 4 || f.is_none_or(|f| words[f.at + 2] != 32) {
        return Err(Refusal::Malformed);
    }

    let stride = if varying.is_some() { STRIDE_WITH_VARYING } else { STRIDE };
    let extra = match varying {
        None => None,
        Some(location) => Some(output_at(words, &list, interface, location).ok_or(Refusal::NoVarying)?),
    };

    // The first location no output uses.
    let locations = |ty: u32| locations_of(words, &list, ty, 0);
    let mut free = 0u32;
    for i in &list {
        if i.op == OP_DECORATE && i.len >= 4 && words[i.at + 2] == DECORATION_LOCATION {
            let id = words[i.at + 1];
            if let Some(var) = def(id).filter(|d| d.op == OP_VARIABLE && words[d.at + 3] == STORAGE_OUTPUT) {
                let ptr = def(words[var.at + 1]).filter(|p| p.op == OP_TYPE_POINTER && p.len == 4);
                let size = ptr.map_or(1, |p| locations(words[p.at + 3]));
                free = free.max(words[i.at + 3] + size);
            }
        }
        if i.op == OP_MEMBER_DECORATE && i.len >= 5 && words[i.at + 3] == DECORATION_LOCATION
            && output_pointer_to(words[i.at + 1]).is_some()
        {
            free = free.max(words[i.at + 4] + 4);
        }
    }
    if free >= MAX_LOCATIONS {
        return Err(Refusal::Offsets);
    }

    let mut bound = words[3];
    let mut fresh = || {
        bound += 1;
        bound - 1
    };
    let mut decls: Vec<u32> = Vec::new();
    let pointer = match output_pointer_to(vec4) {
        Some(p) => p,
        None => {
            let p = fresh();
            decls.extend_from_slice(&[op(OP_TYPE_POINTER, 4), p, STORAGE_OUTPUT, vec4]);
            p
        }
    };
    // The member's index, for the access chain into the block.
    let index = match block {
        Some((_, member)) => {
            let int = list
                .iter()
                .find(|i| i.op == OP_TYPE_INT && i.len == 4 && words[i.at + 2] == 32)
                .map(|i| words[i.at + 1]);
            let int = match int {
                Some(t) => t,
                None => {
                    let t = fresh();
                    decls.extend_from_slice(&[op(OP_TYPE_INT, 4), t, 32, 1]);
                    t
                }
            };
            let constant = list
                .iter()
                .find(|i| i.op == OP_CONSTANT && i.len == 4 && words[i.at + 1] == int
                    && words[i.at + 3] == member)
                .map(|i| words[i.at + 2]);
            Some(match constant {
                Some(c) => c,
                None => {
                    let c = fresh();
                    decls.extend_from_slice(&[op(OP_CONSTANT, 4), int, c, member]);
                    c
                }
            })
        }
        None => None,
    };
    let captured = fresh();
    decls.extend_from_slice(&[op(OP_VARIABLE, 4), pointer, captured, STORAGE_OUTPUT]);

    let last_capability = list.iter().rposition(|i| i.op == OP_CAPABILITY).ok_or(Refusal::Malformed)?;
    let last_mode = list
        .iter()
        .rposition(|i| matches!(i.op, OP_ENTRY_POINT | OP_EXECUTION_MODE | OP_EXECUTION_MODE_ID))
        .ok_or(Refusal::Malformed)?;
    let first_annotation = list
        .iter()
        .position(|i| matches!(i.op, OP_DECORATE | OP_MEMBER_DECORATE))
        .ok_or(Refusal::Malformed)?;
    let first_function = list.iter().position(|i| i.op == OP_FUNCTION).ok_or(Refusal::Malformed)?;
    if !(last_capability < last_mode && last_mode < first_annotation && first_annotation < first_function) {
        return Err(Refusal::Malformed);
    }

    // Each return of the entry point first copies the position out.
    let mut copies: Vec<(usize, [u32; 12], usize)> = Vec::new();
    let mut in_main = false;
    for (k, i) in list.iter().enumerate() {
        if i.op == OP_FUNCTION && i.len >= 5 {
            in_main = words[i.at + 2] == main;
        } else if i.op == OP_FUNCTION_END {
            in_main = false;
        } else if in_main && i.op == OP_RETURN {
            let mut code = [0u32; 12];
            let mut n = 0;
            let source = match index {
                Some(index) => {
                    let p = fresh();
                    code[..5].copy_from_slice(&[op(OP_ACCESS_CHAIN, 5), pointer, p, target, index]);
                    n = 5;
                    p
                }
                None => target,
            };
            let value = fresh();
            code[n..n + 4].copy_from_slice(&[op(OP_LOAD, 4), vec4, value, source]);
            code[n + 4..n + 7].copy_from_slice(&[op(OP_STORE, 3), captured, value]);
            copies.push((k, code, n + 7));
        }
    }
    if copies.is_empty() {
        return Err(Refusal::Malformed);
    }

    out.clear();
    out.reserve(words.len() + decls.len() + copies.len() * 12 + 32);
    out.extend_from_slice(&words[..HEADER]);
    out[3] = bound;
    let mut copy = copies.iter().peekable();
    for (k, i) in list.iter().enumerate() {
        if k == first_annotation {
            out.extend_from_slice(&[op(OP_DECORATE, 4), captured, DECORATION_LOCATION, free]);
            out.extend_from_slice(&[op(OP_DECORATE, 4), captured, DECORATION_XFB_BUFFER, 0]);
            out.extend_from_slice(&[op(OP_DECORATE, 4), captured, DECORATION_XFB_STRIDE, stride]);
            out.extend_from_slice(&[op(OP_DECORATE, 4), captured, DECORATION_OFFSET, 0]);
            if let Some(v) = extra {
                out.extend_from_slice(&[op(OP_DECORATE, 4), v, DECORATION_XFB_BUFFER, 0]);
                out.extend_from_slice(&[op(OP_DECORATE, 4), v, DECORATION_XFB_STRIDE, stride]);
                out.extend_from_slice(&[op(OP_DECORATE, 4), v, DECORATION_OFFSET, STRIDE]);
            }
        }
        if k == first_function {
            out.extend_from_slice(&decls);
        }
        if let Some((at, code, n)) = copy.peek() {
            if *at == k {
                out.extend_from_slice(&code[..*n]);
                copy.next();
            }
        }
        if i.op == OP_ENTRY_POINT {
            out.push(op(OP_ENTRY_POINT, i.len + 1));
            out.extend_from_slice(&words[i.at + 1..i.at + i.len]);
            out.push(captured);
        } else {
            out.extend_from_slice(&words[i.at..i.at + i.len]);
        }
        if k == last_capability {
            out.extend_from_slice(&[op(OP_CAPABILITY, 2), CAPABILITY_TRANSFORM_FEEDBACK]);
        }
        if k == last_mode {
            out.extend_from_slice(&[op(OP_EXECUTION_MODE, 3), main, MODE_XFB]);
        }
    }
    Ok(())
}

/// The output variable of the entry point at `location`, whole at its first component: a vector
/// of two to four 32-bit floats.
fn output_at(words: &[u32], list: &[Insn], interface: &[u32], location: u32) -> Option<u32> {
    let decorated = |id: u32, what: u32| {
        list.iter()
            .find(|i| i.op == OP_DECORATE && i.len >= 4 && words[i.at + 1] == id && words[i.at + 2] == what)
            .map(|i| words[i.at + 3])
    };
    let def = |id: u32| list.iter().find(|i| result_of(words, i) == Some(id));
    let var = list
        .iter()
        .filter(|i| i.op == OP_VARIABLE && i.len >= 4 && words[i.at + 3] == STORAGE_OUTPUT)
        .map(|i| words[i.at + 2])
        .find(|&v| interface.contains(&v) && decorated(v, DECORATION_LOCATION) == Some(location))?;
    if decorated(var, DECORATION_COMPONENT).is_some_and(|c| c != 0) {
        return None;
    }
    let var_at = def(var)?;
    let pointer = def(words[var_at.at + 1]).filter(|p| p.op == OP_TYPE_POINTER && p.len == 4)?;
    let vector = def(words[pointer.at + 3]).filter(|v| v.op == OP_TYPE_VECTOR && v.len == 4)?;
    let float = def(words[vector.at + 2]).filter(|f| f.op == OP_TYPE_FLOAT && f.len >= 3)?;
    (words[float.at + 2] == 32 && (2..=4).contains(&words[vector.at + 3])).then_some(var)
}

/// Locations a type takes as a vertex output.
fn locations_of(words: &[u32], list: &[Insn], ty: u32, depth: u32) -> u32 {
    let Some(i) = list.iter().find(|i| result_of(words, i) == Some(ty)) else { return 1 };
    if depth > 8 {
        return 1;
    }
    match i.op {
        OP_TYPE_VECTOR => {
            let wide = list
                .iter()
                .find(|c| result_of(words, c) == Some(words[i.at + 2]))
                .is_some_and(|c| c.len >= 3 && words[c.at + 2] == 64);
            if wide && words[i.at + 3] > 2 { 2 } else { 1 }
        }
        OP_TYPE_MATRIX => words[i.at + 3] * locations_of(words, list, words[i.at + 2], depth + 1),
        OP_TYPE_ARRAY => {
            let len = list
                .iter()
                .find(|c| c.op == OP_CONSTANT && words[c.at + 2] == words[i.at + 3])
                .map_or(1, |c| words[c.at + 3]);
            len.saturating_mul(locations_of(words, list, words[i.at + 2], depth + 1))
        }
        OP_TYPE_STRUCT => (2..i.len).map(|k| locations_of(words, list, words[i.at + k], depth + 1)).sum(),
        _ => 1,
    }
}

const OP_KILL: u32 = 252;
const OP_TERMINATE_INVOCATION: u32 = 4416;
const OP_DEMOTE_TO_HELPER: u32 = 5380;

/// Whether a fragment module can drop its fragments (alpha test: leaves, fences, hair cards).
/// Such geometry is not solid where it is drawn, so it stays out of the scene rays hit.
pub fn discards(words: &[u32]) -> bool {
    let Ok(list) = insns(words) else { return true };
    list.iter().any(|i| matches!(i.op, OP_KILL | OP_TERMINATE_INVOCATION | OP_DEMOTE_TO_HELPER))
}

#[cfg(test)]
mod tests {
    use super::*;

    // glslangValidator -V of:
    //   layout(location=0) in vec3 pos; layout(location=0) out vec2 uv;
    //   layout(set=0,binding=0) uniform U { mat4 m; } u;
    //   void main(){ uv=pos.xy; gl_Position = u.m*vec4(pos,1); }
    include!("xfb_fixture.rs");

    fn find(words: &[u32], seq: &[u32]) -> bool {
        words.windows(seq.len()).any(|w| w == seq)
    }

    #[test]
    fn captures_the_block_position() {
        assert!(is_vertex(VERTEX));
        let mut out = Vec::new();
        patch(VERTEX, &mut out).unwrap();
        assert!(find(&out, &[(2 << 16) | OP_CAPABILITY, CAPABILITY_TRANSFORM_FEEDBACK]));
        assert!(find(&out, &[DECORATION_XFB_STRIDE, STRIDE]));
        // The position copied to the captured varying: an access chain, a load and a store
        // before the entry point's return, and the varying on the interface.
        assert!(out.windows(1).filter(|w| w[0] == op(OP_STORE, 3)).count() > VERTEX.windows(1).filter(|w| w[0] == op(OP_STORE, 3)).count());
        assert!(out[3] > VERTEX[3]);
        // The result parses and a second pass refuses it: it captures already.
        assert!(insns(&out).is_ok());
        assert_eq!(patch(&out, &mut Vec::new()), Err(Refusal::Captures));
    }

    #[test]
    fn captures_a_varying_beside_the_position() {
        let mut out = Vec::new();
        patch_with(VERTEX, &mut out, Some(0)).unwrap();
        assert!(find(&out, &[DECORATION_XFB_STRIDE, STRIDE_WITH_VARYING]));
        assert!(find(&out, &[DECORATION_OFFSET, STRIDE]), "the varying sits past the position");
        assert!(!find(&out, &[DECORATION_XFB_STRIDE, STRIDE]));
        assert!(insns(&out).is_ok());
        assert_eq!(patch_with(VERTEX, &mut out, Some(5)), Err(Refusal::NoVarying));
    }

    #[test]
    fn refuses_what_it_does_not_follow() {
        let mut out = Vec::new();
        assert_eq!(patch(&[1, 2, 3], &mut out), Err(Refusal::Malformed));
        let mut overrun = VERTEX.to_vec();
        *overrun.last_mut().unwrap() = (5 << 16) | 56;
        assert_eq!(patch(&overrun, &mut out), Err(Refusal::Malformed));
        let mut fragment = VERTEX.to_vec();
        let entry = insns(VERTEX).unwrap().into_iter().find(|i| i.op == OP_ENTRY_POINT).unwrap();
        fragment[entry.at + 1] = 4;
        assert!(!is_vertex(&fragment));
        assert_eq!(patch(&fragment, &mut out), Err(Refusal::NotVertex));
    }

    #[test]
    fn finds_alpha_test() {
        assert!(!discards(VERTEX));
        let mut killing = VERTEX.to_vec();
        killing.insert(killing.len() - 2, (1 << 16) | OP_KILL);
        assert!(discards(&killing));
    }
}
