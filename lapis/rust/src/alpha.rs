// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! A fragment shader's alpha test read out of its SPIR-V, so the rays can cut a surface where the
//! game cuts it: leaves, hair and fences cast the shadows they show.
//!
//! The condition guarding the shader's discard is read back into an expression: texture samples,
//! the draw's uniforms, constants, and per-pixel values the rays cannot know (a view angle, a
//! distance), each with the bounds a clamp gives it. One sampled channel `t` at a varying
//! coordinate is the test's; the discard reads as `a·t + b < 0` with `a` and `b` over the rest,
//! and the other conditions it needs (a material switch, the enclosing branches) as gates. At a
//! draw, [`Program::eval`] puts that draw's uniform bytes in and gets `a` and `b` as numbers, a
//! per-pixel value at the middle of its bounds; or that the draw never cuts.
//!
//! Values are followed through component picks and private and function variables: HLSLcc,
//! Unity's translator, routes every value through them. A dither or a computed coordinate is no
//! test the rays can repeat; of several discards, the alpha test is the one that reads as one.

use std::cell::RefCell;
use std::collections::HashMap;

const MAGIC: u32 = 0x0723_0203;
const HEADER: usize = 5;

const OP_EXT_INST: u32 = 12;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_STRUCT: u32 = 30;
const OP_CONSTANT_TRUE: u32 = 41;
const OP_CONSTANT_FALSE: u32 = 42;
const OP_CONSTANT: u32 = 43;
const OP_CONSTANT_COMPOSITE: u32 = 44;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_VECTOR_SHUFFLE: u32 = 79;
const OP_COMPOSITE_CONSTRUCT: u32 = 80;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_COMPOSITE_INSERT: u32 = 82;
const OP_COPY_OBJECT: u32 = 83;
const OP_SAMPLED_IMAGE: u32 = 86;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u32 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u32 = 88;
const OP_BITCAST: u32 = 124;
const OP_F_NEGATE: u32 = 127;
const OP_F_ADD: u32 = 129;
const OP_F_SUB: u32 = 131;
const OP_I_MUL: u32 = 132;
const OP_F_MUL: u32 = 133;
const OP_F_DIV: u32 = 136;
const OP_VECTOR_TIMES_SCALAR: u32 = 142;
const OP_DOT: u32 = 148;
const OP_LOGICAL_OR: u32 = 166;
const OP_LOGICAL_AND: u32 = 167;
const OP_LOGICAL_NOT: u32 = 168;
const OP_SELECT: u32 = 169;
const OP_I_EQUAL: u32 = 170;
const OP_I_NOT_EQUAL: u32 = 171;
const OP_F_ORD_EQUAL: u32 = 180;
const OP_F_UNORD_NOT_EQUAL: u32 = 183;
const OP_F_ORD_LESS_THAN: u32 = 184;
const OP_F_UNORD_GREATER_THAN_EQUAL: u32 = 191;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_KILL: u32 = 252;
const OP_TERMINATE_INVOCATION: u32 = 4416;
const OP_DEMOTE_TO_HELPER: u32 = 5380;

// GLSL.std.450.
const GLSL_FABS: u32 = 4;
const GLSL_FRACT: u32 = 10;
const GLSL_SQRT: u32 = 31;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX: u32 = 40;
const GLSL_FCLAMP: u32 = 43;
const GLSL_FMIX: u32 = 46;
const GLSL_LENGTH: u32 = 66;
const GLSL_NORMALIZE: u32 = 69;
const GLSL_NMIN: u32 = 79;
const GLSL_NMAX: u32 = 80;
const GLSL_NCLAMP: u32 = 81;

const DIM_2D: u32 = 1;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_MATRIX_STRIDE: u32 = 7;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_COMPONENT: u32 = 31;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const STORAGE_INPUT: u32 = 1;
const STORAGE_UNIFORM: u32 = 2;

/// Followed this deep at most: past it a value is per-pixel and unbounded.
const DEPTH: u32 = 48;

/// Where an alpha test samples.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaTest {
    /// The texture: its set and binding, and its sampler's when separate (else equal).
    pub set: u32,
    pub binding: u32,
    pub sampler_set: u32,
    pub sampler_binding: u32,
    /// The varying its coordinate is: location, and the first of the two components read.
    pub location: u32,
    pub component: u32,
    /// The texel's channel tested (0 r .. 3 a).
    pub channel: u32,
    /// 1 when the test also needs conditions of its own (a material switch).
    pub switched: u32,
}

/// Why a shader's discard could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Unread {
    Malformed = 1,
    NoDiscard = 2,
    ManyDiscards = 3,
    NoCondition = 4,
    NotATexel = 5,
    NotAVarying = 6,
    NoBinding = 7,
}

/// A uniform block a program reads: its set and binding, and the bytes it reads from its start.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub set: u32,
    pub binding: u32,
    pub size: u32,
}

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    K(f32),
    /// A float of uniform block `block` at byte `offset`.
    U { block: usize, offset: u32 },
    /// A per-pixel value within these bounds.
    Free(f32, f32),
    /// A texture sample's channel, at instruction `sample`.
    T { sample: usize, channel: u32 },
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Abs(Box<Expr>),
    Clamp(Box<Expr>, Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Select(Box<Cond>, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
enum Cond {
    /// The expression is below zero.
    Less(Expr),
    Eq(Expr, Expr),
    Not(Box<Cond>),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Bool(bool),
    Free,
}

const ANY: Expr = Expr::Free(f32::NEG_INFINITY, f32::INFINITY);

fn add(a: Expr, b: Expr) -> Expr {
    Expr::Add(Box::new(a), Box::new(b))
}

fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Mul(Box::new(a), Box::new(b))
}

fn neg(a: Expr) -> Expr {
    Expr::Neg(Box::new(a))
}

fn not(c: Cond) -> Cond {
    match c {
        Cond::Not(inner) => *inner,
        Cond::Bool(b) => Cond::Bool(!b),
        c => Cond::Not(Box::new(c)),
    }
}

impl Expr {
    fn texels(&self, out: &mut Vec<(usize, u32)>) {
        match self {
            Expr::T { sample, channel } => out.push((*sample, *channel)),
            Expr::K(_) | Expr::U { .. } | Expr::Free(..) => {}
            Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Min(a, b) | Expr::Max(a, b) => {
                a.texels(out);
                b.texels(out);
            }
            Expr::Neg(a) | Expr::Abs(a) => a.texels(out),
            Expr::Clamp(a, b, c) => {
                a.texels(out);
                b.texels(out);
                c.texels(out);
            }
            Expr::Select(c, a, b) => {
                c.texels(out);
                a.texels(out);
                b.texels(out);
            }
        }
    }

    /// As `a·t + b` in texel `t`, `a` None for zero; the other texels in bounds [0, 1]. A clamp
    /// of the texel is taken as the texel: a cutoff inside the clamp's range cuts the same.
    fn linear(&self, t: (usize, u32)) -> Option<(Option<Expr>, Expr)> {
        Some(match self {
            Expr::T { sample, channel } if (*sample, *channel) == t => (Some(Expr::K(1.0)), Expr::K(0.0)),
            Expr::T { .. } => (None, Expr::Free(0.0, 1.0)),
            Expr::K(_) | Expr::U { .. } | Expr::Free(..) => (None, self.clone()),
            Expr::Add(x, y) => {
                let (ax, bx) = x.linear(t)?;
                let (ay, by) = y.linear(t)?;
                let a = match (ax, ay) {
                    (Some(p), Some(q)) => Some(add(p, q)),
                    (p, q) => p.or(q),
                };
                (a, add(bx, by))
            }
            Expr::Neg(x) => {
                let (a, b) = x.linear(t)?;
                (a.map(neg), neg(b))
            }
            Expr::Mul(x, y) => {
                let (ax, bx) = x.linear(t)?;
                let (ay, by) = y.linear(t)?;
                match (ax, ay) {
                    (None, None) => (None, mul(bx, by)),
                    (Some(a), None) => (Some(mul(a, by.clone())), mul(bx, by)),
                    (None, Some(a)) => (Some(mul(a, bx.clone())), mul(by, bx)),
                    (Some(_), Some(_)) => return None,
                }
            }
            Expr::Div(x, y) => {
                let (ax, bx) = x.linear(t)?;
                let (ay, by) = y.linear(t)?;
                if ay.is_some() {
                    return None;
                }
                let d = |e: Expr| Expr::Div(Box::new(e), Box::new(by.clone()));
                (ax.map(d), d(bx))
            }
            Expr::Clamp(x, lo, hi) => {
                let (ax, bx) = x.linear(t)?;
                if ax.is_none() {
                    let (_, l) = lo.linear(t)?;
                    let (_, h) = hi.linear(t)?;
                    (None, Expr::Clamp(Box::new(bx), Box::new(l), Box::new(h)))
                } else {
                    (ax, bx)
                }
            }
            Expr::Abs(_) | Expr::Min(..) | Expr::Max(..) | Expr::Select(..) => {
                let mut seen = Vec::new();
                self.texels(&mut seen);
                if seen.contains(&t) {
                    return None;
                }
                (None, self.clone())
            }
        })
    }

    fn eval(&self, data: &[Option<&[u8]>]) -> (f64, f64) {
        match self {
            Expr::K(v) => (*v as f64, *v as f64),
            Expr::U { block, offset } => {
                let at = *offset as usize;
                match data.get(*block).copied().flatten() {
                    Some(bytes) if at + 4 <= bytes.len() => {
                        let v = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or([0; 4]));
                        if v.is_finite() {
                            (v as f64, v as f64)
                        } else {
                            (f64::NEG_INFINITY, f64::INFINITY)
                        }
                    }
                    _ => (f64::NEG_INFINITY, f64::INFINITY),
                }
            }
            Expr::Free(lo, hi) => (*lo as f64, *hi as f64),
            Expr::T { .. } => (0.0, 1.0),
            Expr::Add(a, b) => {
                let (p, q) = (a.eval(data), b.eval(data));
                (p.0 + q.0, p.1 + q.1)
            }
            Expr::Neg(a) => {
                let p = a.eval(data);
                (-p.1, -p.0)
            }
            Expr::Mul(a, b) => product(a.eval(data), b.eval(data)),
            Expr::Div(a, b) => {
                let q = b.eval(data);
                if q.0 <= 0.0 && q.1 >= 0.0 {
                    return (f64::NEG_INFINITY, f64::INFINITY);
                }
                product(a.eval(data), (1.0 / q.1, 1.0 / q.0))
            }
            Expr::Abs(a) => {
                let p = a.eval(data);
                if p.0 >= 0.0 {
                    p
                } else if p.1 <= 0.0 {
                    (-p.1, -p.0)
                } else {
                    (0.0, (-p.0).max(p.1))
                }
            }
            Expr::Clamp(x, lo, hi) => {
                let (p, l, h) = (x.eval(data), lo.eval(data), hi.eval(data));
                (p.0.max(l.0).min(h.1), p.1.min(h.1).max(l.0))
            }
            Expr::Min(a, b) => {
                let (p, q) = (a.eval(data), b.eval(data));
                (p.0.min(q.0), p.1.min(q.1))
            }
            Expr::Max(a, b) => {
                let (p, q) = (a.eval(data), b.eval(data));
                (p.0.max(q.0), p.1.max(q.1))
            }
            Expr::Select(c, a, b) => match c.eval(data) {
                Some(true) => a.eval(data),
                Some(false) => b.eval(data),
                None => {
                    let (p, q) = (a.eval(data), b.eval(data));
                    (p.0.min(q.0), p.1.max(q.1))
                }
            },
        }
    }
}

fn product(p: (f64, f64), q: (f64, f64)) -> (f64, f64) {
    // 0·∞ is 0 here: a bound times nothing.
    let m = |x: f64, y: f64| if x == 0.0 || y == 0.0 { 0.0 } else { x * y };
    let c = [m(p.0, q.0), m(p.0, q.1), m(p.1, q.0), m(p.1, q.1)];
    let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo.is_nan() || hi.is_nan() {
        (f64::NEG_INFINITY, f64::INFINITY)
    } else {
        (lo, hi)
    }
}

impl Cond {
    fn texels(&self, out: &mut Vec<(usize, u32)>) {
        match self {
            Cond::Less(e) => e.texels(out),
            Cond::Eq(a, b) => {
                a.texels(out);
                b.texels(out);
            }
            Cond::Not(c) => c.texels(out),
            Cond::And(a, b) | Cond::Or(a, b) => {
                a.texels(out);
                b.texels(out);
            }
            Cond::Bool(_) | Cond::Free => {}
        }
    }

    /// True, false, or not known for every pixel.
    fn eval(&self, data: &[Option<&[u8]>]) -> Option<bool> {
        match self {
            Cond::Less(e) => {
                let (lo, hi) = e.eval(data);
                if hi < 0.0 {
                    Some(true)
                } else if lo >= 0.0 {
                    Some(false)
                } else {
                    None
                }
            }
            Cond::Eq(a, b) => {
                let (p, q) = (a.eval(data), b.eval(data));
                if p.0 == p.1 && q.0 == q.1 {
                    Some(p.0 == q.0)
                } else if p.1 < q.0 || q.1 < p.0 {
                    Some(false)
                } else {
                    None
                }
            }
            Cond::Not(c) => c.eval(data).map(|v| !v),
            Cond::And(a, b) => match (a.eval(data), b.eval(data)) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            Cond::Or(a, b) => match (a.eval(data), b.eval(data)) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            Cond::Bool(v) => Some(*v),
            Cond::Free => None,
        }
    }

    /// The terms all of which must hold, negations pushed in where they split.
    fn conjuncts(self, out: &mut Vec<Cond>) {
        match self {
            Cond::And(a, b) => {
                a.conjuncts(out);
                b.conjuncts(out);
            }
            Cond::Not(inner) => match *inner {
                Cond::Or(a, b) => {
                    not(*a).conjuncts(out);
                    not(*b).conjuncts(out);
                }
                Cond::Not(c) => c.conjuncts(out),
                c => out.push(not(c)),
            },
            Cond::Bool(true) => {}
            c => out.push(c),
        }
    }
}

/// What a draw's alpha test does with its uniforms in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// Cut where `a·t + b < 0`, `t` sampled as [`Program::tests`]`[part]` says.
    Cuts { part: usize, a: f32, b: f32 },
    /// Its switches are off: the surface is whole.
    Whole,
    /// A uniform it needs is unknown, or the test has no bounded value.
    Unknown,
}

/// One discard of a shader as `a·t + b < 0` under its gates.
#[derive(Debug)]
struct Part {
    test: AlphaTest,
    a: Expr,
    b: Expr,
    gates: Vec<Cond>,
}

/// A shader's alpha tests: each of its discards that reads as one, all at the same varying (a
/// cutout, a mask beside it), each behind switches of its own.
#[derive(Debug)]
pub struct Program {
    parts: Vec<Part>,
    pub blocks: Vec<Block>,
}

impl Program {
    /// Where each part samples.
    pub fn tests(&self) -> impl Iterator<Item = &AlphaTest> {
        self.parts.iter().map(|p| &p.test)
    }

    /// `data[i]` the bytes of `blocks[i]` as the draw has them bound, None unknown. The first
    /// part whose switches may be on is the draw's test.
    pub fn eval(&self, data: &[Option<&[u8]>]) -> Verdict {
        let mid = |e: &Expr| {
            let (lo, hi) = e.eval(data);
            let m = if lo == hi { lo } else { 0.5 * (lo + hi) };
            m.is_finite().then_some(m as f32)
        };
        for (part, p) in self.parts.iter().enumerate() {
            if p.gates.iter().any(|g| g.eval(data) == Some(false)) {
                continue;
            }
            return match (mid(&p.a), mid(&p.b)) {
                (Some(a), Some(b)) if a != 0.0 => Verdict::Cuts { part, a, b },
                _ => Verdict::Unknown,
            };
        }
        Verdict::Whole
    }
}

struct Module<'a> {
    words: &'a [u32],
    /// Instruction start by result id, values and types alike.
    defs: HashMap<u32, usize>,
    /// Every instruction start, in order.
    order: Vec<usize>,
    decorations: HashMap<(u32, u32), u32>,
    members: HashMap<(u32, u32, u32), u32>,
    /// The uniform blocks the program being read uses.
    blocks: RefCell<Vec<Block>>,
}

impl<'a> Module<'a> {
    fn parse(words: &'a [u32]) -> Option<Module<'a>> {
        if words.len() < HEADER || words[0] != MAGIC {
            return None;
        }
        let mut m = Module {
            words,
            defs: HashMap::new(),
            order: Vec::new(),
            decorations: HashMap::new(),
            members: HashMap::new(),
            blocks: RefCell::new(Vec::new()),
        };
        let mut at = HEADER;
        while at < words.len() {
            let len = (words[at] >> 16) as usize;
            if len == 0 || at + len > words.len() {
                return None;
            }
            let op = words[at] & 0xffff;
            m.order.push(at);
            match op {
                OP_DECORATE if len >= 4 => {
                    m.decorations.insert((words[at + 1], words[at + 2]), words[at + 3]);
                }
                OP_MEMBER_DECORATE if len >= 5 => {
                    m.members.insert((words[at + 1], words[at + 2], words[at + 3]), words[at + 4]);
                }
                // Types and labels: the result is the first operand.
                19..=39 | OP_LABEL if len >= 2 => {
                    m.defs.insert(words[at + 1], at);
                }
                // Instructions with a result type: the result is the second. Of those numbered
                // among them, image writes and barriers have none.
                OP_EXT_INST | OP_CONSTANT_TRUE..=OP_CONSTANT_COMPOSITE | OP_VARIABLE | OP_LOAD
                | OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN | 77..=245
                    if len >= 3 && !matches!(op, 99 | 224 | 225 | 228) =>
                {
                    m.defs.insert(words[at + 2], at);
                }
                _ => {}
            }
            at += len;
        }
        Some(m)
    }

    fn op(&self, at: usize) -> u32 {
        self.words[at] & 0xffff
    }

    fn len(&self, at: usize) -> usize {
        (self.words[at] >> 16) as usize
    }

    fn def(&self, id: u32) -> Option<usize> {
        self.defs.get(&id).copied()
    }

    fn arg(&self, at: usize, i: usize) -> u32 {
        self.words[at + i]
    }

    fn type_of(&self, id: u32) -> Option<usize> {
        self.def(self.arg(self.def(id)?, 1))
    }

    fn is_float(&self, id: u32) -> bool {
        self.type_of(id).is_some_and(|t| {
            let t = if self.op(t) == OP_TYPE_VECTOR { self.def(self.arg(t, 2)) } else { Some(t) };
            t.is_some_and(|t| self.op(t) == OP_TYPE_FLOAT)
        })
    }

    /// Components of a vector value, 1 a scalar.
    fn width(&self, id: u32) -> u32 {
        self.type_of(id)
            .filter(|&t| self.op(t) == OP_TYPE_VECTOR)
            .map_or(1, |t| self.arg(t, 3))
    }

    fn constant_u32(&self, id: u32) -> Option<u32> {
        let at = self.def(id)?;
        (self.op(at) == OP_CONSTANT && self.len(at) == 4).then(|| self.arg(at, 3))
    }

    /// A pointer as a variable and constant indices into it.
    fn place(&self, ptr: u32) -> Option<(u32, Vec<u32>)> {
        let at = self.def(ptr)?;
        match self.op(at) {
            OP_VARIABLE => Some((ptr, Vec::new())),
            OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN => {
                let (base, mut path) = self.place(self.arg(at, 3))?;
                for i in 4..self.len(at) {
                    path.push(self.constant_u32(self.arg(at, i))?);
                }
                Some((base, path))
            }
            _ => None,
        }
    }

    fn storage(&self, var: u32) -> Option<u32> {
        let at = self.def(var)?;
        (self.op(at) == OP_VARIABLE).then(|| self.arg(at, 3))
    }

    /// What a load at `load` reads, component `comp` of it: the value last stored there before
    /// it in the code, and the component of that value.
    fn stored(&self, load: usize, comp: Option<u32>) -> Option<(u32, Option<u32>)> {
        let (base, path) = self.place(self.arg(load, 3))?;
        let index = self.order.iter().position(|&a| a == load)?;
        for &at in self.order[..index].iter().rev() {
            if self.op(at) != OP_STORE {
                continue;
            }
            let Some((b, p)) = self.place(self.arg(at, 1)) else { continue };
            if b != base {
                continue;
            }
            let value = self.arg(at, 2);
            if p == path {
                return Some((value, comp));
            }
            if p.len() == path.len() + 1 && p.starts_with(&path) {
                // One component stored: ours, or another left as it was.
                if comp == p.last().copied() {
                    return Some((value, None));
                }
                if comp.is_some() {
                    continue;
                }
                return None;
            }
            if path.len() == p.len() + 1 && path.starts_with(&p) && comp.is_none() {
                return Some((value, path.last().copied()));
            }
            if p.starts_with(&path) || path.starts_with(&p) {
                return None;
            }
        }
        None
    }

    /// The byte offset of a float at `path` (and `comp`) in uniform block variable `var`.
    fn uniform_offset(&self, var: u32, path: &[u32], comp: Option<u32>) -> Option<u32> {
        let pointer = self.def(self.arg(self.def(var)?, 1))?;
        let mut ty = self.def(self.arg(pointer, 3))?;
        let mut offset = 0u32;
        let mut indices: Vec<u32> = path.to_vec();
        if let Some(c) = comp {
            indices.push(c);
        }
        let mut parent_struct: Option<(u32, u32)> = None;
        for &i in &indices {
            match self.op(ty) {
                OP_TYPE_STRUCT => {
                    let id = self.arg(ty, 1);
                    offset += *self.members.get(&(id, i, DECORATION_OFFSET))?;
                    parent_struct = Some((id, i));
                    ty = self.def(self.arg(ty, 2 + i as usize))?;
                }
                OP_TYPE_ARRAY => {
                    offset += i * *self.decorations.get(&(self.arg(ty, 1), DECORATION_ARRAY_STRIDE))?;
                    ty = self.def(self.arg(ty, 2))?;
                }
                OP_TYPE_MATRIX => {
                    let stride = parent_struct
                        .and_then(|(s, m)| self.members.get(&(s, m, DECORATION_MATRIX_STRIDE)).copied())
                        .unwrap_or(16);
                    offset += i * stride;
                    ty = self.def(self.arg(ty, 2))?;
                }
                OP_TYPE_VECTOR => {
                    offset += i * 4;
                    ty = self.def(self.arg(ty, 2))?;
                }
                _ => return None,
            }
        }
        (self.op(ty) == OP_TYPE_FLOAT && self.arg(ty, 2) == 32).then_some(offset)
    }

    fn block_of(&self, var: u32, size: u32) -> Option<usize> {
        let set = *self.decorations.get(&(var, DECORATION_DESCRIPTOR_SET))?;
        let binding = *self.decorations.get(&(var, DECORATION_BINDING))?;
        let mut blocks = self.blocks.borrow_mut();
        if let Some(i) = blocks.iter().position(|b| b.set == set && b.binding == binding) {
            blocks[i].size = blocks[i].size.max(size);
            return Some(i);
        }
        blocks.push(Block { set, binding, size });
        Some(blocks.len() - 1)
    }

    /// Value `id`, component `comp` of it when a vector, as an expression.
    fn expr(&self, id: u32, comp: Option<u32>, depth: u32) -> Expr {
        if depth > DEPTH {
            return ANY;
        }
        let Some(at) = self.def(id) else { return ANY };
        let d = depth + 1;
        let c = |i: usize| self.expr(self.arg(at, i), comp, d);
        match self.op(at) {
            OP_CONSTANT if self.len(at) == 4 && self.is_float(id) => Expr::K(f32::from_bits(self.arg(at, 3))),
            OP_CONSTANT_COMPOSITE => match comp {
                Some(k) if 3 + (k as usize) < self.len(at) => self.expr(self.arg(at, 3 + k as usize), None, d),
                _ => ANY,
            },
            OP_LOAD => self.loaded(at, comp, d),
            OP_COPY_OBJECT => c(3),
            OP_COMPOSITE_EXTRACT if self.len(at) == 5 && comp.is_none() => {
                self.expr(self.arg(at, 3), Some(self.arg(at, 4)), d)
            }
            OP_COMPOSITE_INSERT if self.len(at) == 6 => match comp {
                Some(k) if k == self.arg(at, 5) => self.expr(self.arg(at, 3), None, d),
                _ => self.expr(self.arg(at, 4), comp, d),
            },
            OP_COMPOSITE_CONSTRUCT => {
                let mut k = comp.unwrap_or(0);
                for i in 3..self.len(at) {
                    let part = self.arg(at, i);
                    let w = self.width(part);
                    if k < w {
                        return self.expr(part, (w > 1).then_some(k), d);
                    }
                    k -= w;
                }
                ANY
            }
            OP_VECTOR_SHUFFLE => {
                let k = comp.unwrap_or(0) as usize;
                if 5 + k >= self.len(at) {
                    return ANY;
                }
                let pick = self.arg(at, 5 + k);
                let first = self.width(self.arg(at, 3));
                if pick < first {
                    self.expr(self.arg(at, 3), Some(pick), d)
                } else {
                    self.expr(self.arg(at, 4), Some(pick - first), d)
                }
            }
            OP_F_NEGATE => neg(c(3)),
            OP_F_ADD => add(c(3), c(4)),
            OP_F_SUB => add(c(3), neg(c(4))),
            OP_F_MUL => mul(c(3), c(4)),
            OP_F_DIV => Expr::Div(Box::new(c(3)), Box::new(c(4))),
            OP_VECTOR_TIMES_SCALAR => mul(c(3), self.expr(self.arg(at, 4), None, d)),
            OP_DOT => {
                let (x, y) = (self.arg(at, 3), self.arg(at, 4));
                (0..self.width(x))
                    .map(|k| mul(self.expr(x, Some(k), d), self.expr(y, Some(k), d)))
                    .reduce(add)
                    .unwrap_or(ANY)
            }
            OP_SELECT => {
                let cond = self.cond(self.arg(at, 3), comp, d);
                Expr::Select(Box::new(cond), Box::new(c(4)), Box::new(c(5)))
            }
            OP_IMAGE_SAMPLE_IMPLICIT_LOD | OP_IMAGE_SAMPLE_EXPLICIT_LOD => match comp {
                Some(channel) if self.flat_2d(self.arg(at, 3)) => Expr::T { sample: at, channel },
                _ => Expr::Free(0.0, 1.0),
            },
            OP_EXT_INST if self.len(at) >= 6 => {
                let e = |i: usize| self.expr(self.arg(at, 5 + i), comp, d);
                let n = self.len(at) - 5;
                match self.arg(at, 4) {
                    GLSL_FABS => Expr::Abs(Box::new(e(0))),
                    GLSL_FMIN | GLSL_NMIN if n == 2 => Expr::Min(Box::new(e(0)), Box::new(e(1))),
                    GLSL_FMAX | GLSL_NMAX if n == 2 => Expr::Max(Box::new(e(0)), Box::new(e(1))),
                    GLSL_FCLAMP | GLSL_NCLAMP if n == 3 => {
                        Expr::Clamp(Box::new(e(0)), Box::new(e(1)), Box::new(e(2)))
                    }
                    GLSL_FMIX if n == 3 => {
                        let w = e(2);
                        add(mul(e(0), add(Expr::K(1.0), neg(w.clone()))), mul(e(1), w))
                    }
                    GLSL_FRACT => Expr::Free(0.0, 1.0),
                    GLSL_NORMALIZE => Expr::Free(-1.0, 1.0),
                    GLSL_SQRT | GLSL_INVERSE_SQRT | GLSL_LENGTH => Expr::Free(0.0, f32::INFINITY),
                    _ => ANY,
                }
            }
            _ => ANY,
        }
    }

    fn loaded(&self, at: usize, comp: Option<u32>, depth: u32) -> Expr {
        let Some((base, path)) = self.place(self.arg(at, 3)) else { return ANY };
        match self.storage(base) {
            Some(STORAGE_UNIFORM) => match self.uniform_offset(base, &path, comp) {
                Some(offset) => match self.block_of(base, offset + 4) {
                    Some(block) => Expr::U { block, offset },
                    None => ANY,
                },
                None => ANY,
            },
            // Varyings, push constants, built-ins: per pixel or unread. Private, function and
            // output variables hold what was stored.
            Some(s) if !matches!(s, 3 | 6 | 7) => ANY,
            _ => match self.stored(at, comp) {
                Some((value, part)) => self.expr(value, part, depth),
                None => ANY,
            },
        }
    }

    /// Bool value `id` (component `comp`) as a condition.
    fn cond(&self, id: u32, comp: Option<u32>, depth: u32) -> Cond {
        if depth > DEPTH {
            return Cond::Free;
        }
        let Some(at) = self.def(id) else { return Cond::Free };
        let d = depth + 1;
        let e = |i: usize| self.expr(self.arg(at, i), comp, d);
        let op = self.op(at);
        match op {
            OP_CONSTANT_TRUE => Cond::Bool(true),
            OP_CONSTANT_FALSE => Cond::Bool(false),
            OP_LOGICAL_NOT => not(self.cond(self.arg(at, 3), comp, d)),
            OP_LOGICAL_AND => Cond::And(
                Box::new(self.cond(self.arg(at, 3), comp, d)),
                Box::new(self.cond(self.arg(at, 4), comp, d)),
            ),
            OP_LOGICAL_OR => Cond::Or(
                Box::new(self.cond(self.arg(at, 3), comp, d)),
                Box::new(self.cond(self.arg(at, 4), comp, d)),
            ),
            OP_LOAD => match self.stored(at, comp) {
                Some((value, part)) => self.cond(value, part, d),
                None => Cond::Free,
            },
            OP_COPY_OBJECT => self.cond(self.arg(at, 3), comp, d),
            OP_COMPOSITE_EXTRACT if self.len(at) == 5 && comp.is_none() => {
                self.cond(self.arg(at, 3), Some(self.arg(at, 4)), d)
            }
            OP_COMPOSITE_INSERT if self.len(at) == 6 => match comp {
                Some(k) if k == self.arg(at, 5) => self.cond(self.arg(at, 3), None, d),
                _ => self.cond(self.arg(at, 4), comp, d),
            },
            OP_COMPOSITE_CONSTRUCT => {
                let k = comp.unwrap_or(0) as usize;
                if 3 + k < self.len(at) {
                    self.cond(self.arg(at, 3 + k), None, d)
                } else {
                    Cond::Free
                }
            }
            OP_VECTOR_SHUFFLE => {
                let k = comp.unwrap_or(0) as usize;
                if 5 + k >= self.len(at) {
                    return Cond::Free;
                }
                let pick = self.arg(at, 5 + k);
                let first = self.width(self.arg(at, 3));
                if pick < first {
                    self.cond(self.arg(at, 3), Some(pick), d)
                } else {
                    self.cond(self.arg(at, 4), Some(pick - first), d)
                }
            }
            // HLSLcc's clip: an int 0 or not, from a select of the condition.
            OP_I_EQUAL | OP_I_NOT_EQUAL => {
                let (a, b) = (self.arg(at, 3), self.arg(at, 4));
                let value = if self.constant_u32(b) == Some(0) {
                    a
                } else if self.constant_u32(a) == Some(0) {
                    b
                } else {
                    return Cond::Free;
                };
                let nonzero = self.nonzero(value, comp, d);
                if op == OP_I_NOT_EQUAL { nonzero } else { not(nonzero) }
            }
            OP_F_ORD_EQUAL..=OP_F_UNORD_GREATER_THAN_EQUAL => {
                let (a, b) = (e(3), e(4));
                match op {
                    180 | 181 => Cond::Eq(a, b),
                    182 | OP_F_UNORD_NOT_EQUAL => not(Cond::Eq(a, b)),
                    // a < b, a <= b: a - b below zero (the edge aside).
                    OP_F_ORD_LESS_THAN | 185 | 188 | 189 => Cond::Less(add(a, neg(b))),
                    _ => Cond::Less(add(b, neg(a))),
                }
            }
            _ => Cond::Free,
        }
    }

    /// An int that is nonzero exactly when this condition holds.
    fn nonzero(&self, id: u32, comp: Option<u32>, depth: u32) -> Cond {
        if depth > DEPTH {
            return Cond::Free;
        }
        let Some(at) = self.def(id) else { return Cond::Free };
        let d = depth + 1;
        match self.op(at) {
            OP_SELECT => {
                let (t, f) = (self.constant_u32(self.arg(at, 4)), self.constant_u32(self.arg(at, 5)));
                let c = self.cond(self.arg(at, 3), comp, d);
                match (t, f) {
                    (Some(t), Some(0)) if t != 0 => c,
                    (Some(0), Some(f)) if f != 0 => not(c),
                    _ => Cond::Free,
                }
            }
            OP_I_MUL => {
                let (a, b) = (self.arg(at, 3), self.arg(at, 4));
                match (self.constant_u32(a), self.constant_u32(b)) {
                    (_, Some(k)) if k != 0 => self.nonzero(a, comp, d),
                    (Some(k), _) if k != 0 => self.nonzero(b, comp, d),
                    _ => Cond::Free,
                }
            }
            OP_BITCAST | OP_COPY_OBJECT => self.nonzero(self.arg(at, 3), comp, d),
            OP_LOAD => match self.stored(at, comp) {
                Some((value, part)) => self.nonzero(value, part, d),
                None => Cond::Free,
            },
            OP_COMPOSITE_EXTRACT if self.len(at) == 5 && comp.is_none() => {
                self.nonzero(self.arg(at, 3), Some(self.arg(at, 4)), d)
            }
            _ => Cond::Free,
        }
    }

    /// Whether a sampled image value is a plain 2D texture, one the rays sample as theirs.
    fn flat_2d(&self, image: u32) -> bool {
        let Some(ty) = self.type_of(image) else { return false };
        if self.op(ty) != OP_TYPE_SAMPLED_IMAGE {
            return false;
        }
        let Some(img) = self.def(self.arg(ty, 2)) else { return false };
        self.op(img) == OP_TYPE_IMAGE && self.len(img) >= 8 && self.arg(img, 3) == DIM_2D
            && self.arg(img, 5) == 0 && self.arg(img, 6) == 0
    }

    /// The input varying and first component a two-component coordinate is, untouched.
    fn varying(&self, id: u32, depth: u32) -> Option<(u32, u32)> {
        if depth > DEPTH {
            return None;
        }
        let at = self.def(id)?;
        match self.op(at) {
            OP_LOAD => {
                let (base, path) = self.place(self.arg(at, 3))?;
                if self.storage(base) == Some(STORAGE_INPUT) && path.is_empty() {
                    let location = *self.decorations.get(&(base, DECORATION_LOCATION))?;
                    let component =
                        self.decorations.get(&(base, DECORATION_COMPONENT)).copied().unwrap_or(0);
                    return Some((location, component));
                }
                let (value, part) = self.stored(at, None)?;
                if part.is_some() {
                    return None;
                }
                self.varying(value, depth + 1)
            }
            OP_VECTOR_SHUFFLE if self.len(at) == 7 => {
                // Two consecutive components of one source.
                let (a, b) = (self.arg(at, 5), self.arg(at, 6));
                let source = self.arg(at, 3);
                if b != a + 1 || a + 1 >= self.width(source) {
                    return None;
                }
                let (location, component) = self.varying(source, depth + 1)?;
                Some((location, component + a))
            }
            OP_COPY_OBJECT => self.varying(self.arg(at, 3), depth + 1),
            _ => None,
        }
    }

    /// The set and binding of a resource loaded from a descriptor.
    fn binding(&self, id: u32) -> Option<(u32, u32)> {
        let at = self.def(id)?;
        if self.op(at) != OP_LOAD {
            return None;
        }
        let var = self.arg(at, 3);
        self.storage(var)?;
        Some((
            *self.decorations.get(&(var, DECORATION_DESCRIPTOR_SET))?,
            *self.decorations.get(&(var, DECORATION_BINDING))?,
        ))
    }

    /// The label of the block instruction `at` is in.
    fn block_label(&self, at: usize) -> Option<u32> {
        let index = self.order.iter().position(|&a| a == at)?;
        self.order[..index].iter().rev().find(|&&a| self.op(a) == OP_LABEL).map(|&a| self.arg(a, 1))
    }

    /// The conditions that hold wherever block `label` runs: up its chain of single
    /// predecessors, the side of each branch taken.
    fn path_conditions(&self, label: u32, out: &mut Vec<Cond>) {
        let mut block = label;
        for _ in 0..16 {
            let into: Vec<usize> = self
                .order
                .iter()
                .copied()
                .filter(|&a| match self.op(a) {
                    OP_BRANCH => self.arg(a, 1) == block,
                    OP_BRANCH_CONDITIONAL => self.arg(a, 2) == block || self.arg(a, 3) == block,
                    _ => false,
                })
                .collect();
            let [from] = into.as_slice() else { return };
            if self.op(*from) == OP_BRANCH_CONDITIONAL {
                let c = self.cond(self.arg(*from, 1), None, 0);
                out.push(if self.arg(*from, 2) == block { c } else { not(c) });
            }
            let Some(up) = self.block_label(*from) else { return };
            block = up;
        }
    }
}

/// Reads the alpha test of fragment shader `words`.
pub fn read(words: &[u32]) -> Result<(AlphaTest, Program), Unread> {
    let m = Module::parse(words).ok_or(Unread::Malformed)?;
    let kills: Vec<usize> = m
        .order
        .iter()
        .copied()
        .filter(|&at| matches!(m.op(at), OP_KILL | OP_TERMINATE_INVOCATION | OP_DEMOTE_TO_HELPER))
        .collect();
    // A shader may discard for other reasons too (a dithered fade, a clip plane): the alpha
    // test is the one discard that reads as one.
    let mut refused = Unread::NoDiscard;
    let mut parts: Vec<Part> = Vec::new();
    for &kill in &kills {
        match read_kill(&m, kill) {
            // The rays capture one varying: parts sampling at another are left out.
            Ok(p) => {
                let first = parts.first().map(|f| (f.test.location, f.test.component));
                if first.is_none_or(|f| f == (p.test.location, p.test.component)) {
                    parts.push(p);
                }
            }
            Err(e) => refused = if kills.len() > 1 { Unread::ManyDiscards } else { e },
        }
    }
    let first = parts.first().ok_or(refused)?.test;
    let blocks = m.blocks.borrow().clone();
    Ok((first, Program { parts, blocks }))
}

fn read_kill(m: &Module, kill: usize) -> Result<Part, Unread> {
    let label = m.block_label(kill).ok_or(Unread::NoCondition)?;
    let mut terms = Vec::new();
    m.path_conditions(label, &mut terms);
    let mut all = Vec::new();
    for t in terms {
        t.conjuncts(&mut all);
    }
    all.dedup();
    let mut unique: Vec<Cond> = Vec::new();
    for t in all {
        if !unique.contains(&t) {
            unique.push(t);
        }
    }
    if unique.is_empty() {
        return Err(Unread::NoCondition);
    }
    // The test: the one term that reads a texel at a varying coordinate.
    let mut test: Option<(Expr, (usize, u32))> = None;
    let mut gates = Vec::new();
    for term in unique {
        let mut seen = Vec::new();
        term.texels(&mut seen);
        let texel = seen.into_iter().find(|&(sample, _)| m.varying(m.arg(sample, 4), 0).is_some());
        match (texel, term) {
            (Some(t), Cond::Less(e)) if test.is_none() => test = Some((e, t)),
            // Not below zero: at or above it, the same cut but for the edge.
            (Some(t), Cond::Not(inner)) if test.is_none() => match *inner {
                Cond::Less(e) => test = Some((neg(e), t)),
                _ => return Err(Unread::NoCondition),
            },
            (Some(_), _) => return Err(Unread::NoCondition),
            (None, term) => gates.push(term),
        }
    }
    let (e, (sample, channel)) = test.ok_or(Unread::NotATexel)?;
    let (a, b) = e.linear((sample, channel)).ok_or(Unread::NotATexel)?;
    let a = a.ok_or(Unread::NotATexel)?;
    let (location, component) = m.varying(m.arg(sample, 4), 0).ok_or(Unread::NotAVarying)?;
    // The sampled image: a combined descriptor, or an image and a sampler put together.
    let image = m.arg(sample, 3);
    let image_at = m.def(image).ok_or(Unread::NoBinding)?;
    let (texture, sampler) = if m.op(image_at) == OP_SAMPLED_IMAGE {
        let t = m.binding(m.arg(image_at, 3)).ok_or(Unread::NoBinding)?;
        let s = m.binding(m.arg(image_at, 4)).ok_or(Unread::NoBinding)?;
        (t, s)
    } else {
        let t = m.binding(image).ok_or(Unread::NoBinding)?;
        (t, t)
    };
    let test = AlphaTest {
        set: texture.0,
        binding: texture.1,
        sampler_set: sampler.0,
        sampler_binding: sampler.1,
        location,
        component,
        channel,
        switched: !gates.is_empty() as u32,
    };
    Ok(Part { test, a, b, gates })
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("alpha_fixture.rs");

    fn cuts(p: &Program, data: &[Option<&[u8]>]) -> (f32, f32) {
        match p.eval(data) {
            Verdict::Cuts { a, b, .. } => (a, b),
            v => panic!("{v:?}"),
        }
    }

    fn floats(values: &[(usize, f32)], size: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; size];
        for &(at, v) in values {
            bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn reads_a_plain_alpha_test() {
        let (t, p) = read(SIMPLE).unwrap();
        assert_eq!((t.set, t.binding, t.sampler_set, t.sampler_binding), (1, 3, 1, 3));
        assert_eq!((t.location, t.component, t.channel, t.switched), (2, 0, 3, 0));
        assert_eq!(cuts(&p, &[]), (1.0, -0.5), "a < 0.5");
    }

    #[test]
    fn reads_unitys_through_its_variables_and_tint() {
        let (t, p) = read(UNITY).unwrap();
        assert_eq!((t.set, t.binding), (0, 2));
        // vs_TEXCOORD0.zw: the second UV packed in one varying.
        assert_eq!((t.location, t.component, t.channel), (0, 2, 3));
        assert_eq!(p.blocks, vec![Block { set: 0, binding: 1, size: 20 }]);
        // a·_Color.w < _Cutoff, _Color.w at 12 and _Cutoff at 16.
        let bytes = floats(&[(12, 0.8), (16, 0.4)], 20);
        assert_eq!(cuts(&p, &[Some(&bytes)]), (0.8, -0.4));
        assert_eq!(p.eval(&[None]), Verdict::Unknown, "the uniforms unread");
    }

    #[test]
    fn reads_a_separate_sampler_and_a_negated_test() {
        let (t, p) = read(SEPARATE).unwrap();
        assert_eq!((t.set, t.binding, t.sampler_set, t.sampler_binding), (2, 0, 2, 1));
        assert_eq!((t.location, t.component, t.channel), (3, 2, 1));
        let (a, b) = cuts(&p, &[]);
        assert_eq!(a, 1.0, "!(a >= c) cuts below c");
        assert!((b + 0.33).abs() < 1e-6);
    }

    #[test]
    fn reads_a_clip() {
        let (t, p) = read(CLIP).unwrap();
        assert_eq!((t.binding, t.location, t.channel), (5, 1, 3));
        assert_eq!(cuts(&p, &[]), (1.0, -0.25));
    }

    #[test]
    fn refuses_a_dither() {
        assert_eq!(read(DITHER).err(), Some(Unread::NotATexel));
    }

    #[test]
    fn reads_hlslccs_clip_past_a_dithered_fade() {
        // int(x < 0) * -1 != 0 on x = alpha + -cutoff, inside a branch on the same condition,
        // after a fade's discard.
        let (t, p) = read(HLSLCC).unwrap();
        assert_eq!((t.set, t.binding, t.location, t.component, t.channel), (0, 1, 3, 0, 3));
        // _Cutoff at 20 of PGlobals (set 1).
        assert_eq!(p.blocks, vec![Block { set: 1, binding: 0, size: 24 }]);
        let bytes = floats(&[(20, 0.3)], 24);
        assert_eq!(cuts(&p, &[Some(&bytes)]), (1.0, -0.3));
    }

    #[test]
    fn reads_a_foliage_test_under_its_switch() {
        // Genshin's leaves: the mask in blue, a cutoff the material scales by the view angle, a
        // distance that fills far leaves in, and a switch.
        let (t, p) = read(FOLIAGE).unwrap();
        assert_eq!((t.set, t.binding, t.location, t.channel, t.switched), (0, 1, 0, 2, 1));
        assert_eq!(p.blocks, vec![Block { set: 1, binding: 0, size: 36 }]);
        let on = floats(&[(16, 30.0), (20, 0.2), (24, 0.5), (28, 0.6), (32, 1.0)], 36);
        let (a, b) = cuts(&p, &[Some(&on)]);
        // t + [0, 0.2] - [0.5, 1]·0.6, at the middle of its bounds.
        assert_eq!(a, 1.0);
        assert!((b + 0.35).abs() < 1e-5, "{b}");
        let off = floats(&[(16, 30.0), (20, 0.2), (24, 0.5), (28, 0.6), (32, 0.0)], 36);
        assert_eq!(p.eval(&[Some(&off)]), Verdict::Whole);
    }

    #[test]
    fn refuses_what_has_no_discard() {
        let mut words = SIMPLE.to_vec();
        for w in words.iter_mut() {
            if *w & 0xffff == OP_KILL && *w >> 16 == 1 {
                *w = (1 << 16) | 253; // OpReturn
            }
        }
        assert_eq!(read(&words).err(), Some(Unread::NoDiscard));
        assert_eq!(read(&[1, 2, 3]).err(), Some(Unread::Malformed));
    }
}
