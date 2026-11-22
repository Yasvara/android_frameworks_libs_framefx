// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! A stack compiled for the passes: the bands in their fixed order Detail -> Colour -> Screen,
//! stack order inside each, every node that would change nothing dropped. The colour band's LUT
//! is baked by `lut`; who owns it is the caller's business.

use crate::colour::{is_identity_colour, luma_weights, Step};
use crate::lut::ends_linear;
use crate::params::*;
use crate::section::{Node, Stack, MAX_NODES};

pub const MAX_DETAIL: usize = 3;
pub const MAX_SCREEN: usize = 4;
pub const MAX_SCENE: usize = 8;
pub const PLAN_VEC4S: usize = 16;
pub const SCENE_VEC4S: usize = 6;
pub const FRAME_VEC4S: usize = 4;

pub const FORMAT_UNORM10: u8 = 2;
pub const ROLE_STAGE_A: u8 = 1;
pub const ROLE_STAGE_B: u8 = 2;

/// Everything in framefx::kelir::Plan but the LUTs it owns.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanCore {
    pub revision: u32,
    pub peek: bool,
    pub split: bool,
    pub inspect: bool,
    /// display space
    pub split_x: f32,
    pub detail_count: i32,
    pub detail: [Step; MAX_DETAIL],
    pub colour_count: i32,
    pub colour: [Step; MAX_NODES],
    pub screen_count: i32,
    pub screen: [Step; MAX_SCREEN],
    /// Run by Luwes on the game's scene before its HUD, never by Kelir's own passes.
    pub scene_count: i32,
    pub scene: [Step; MAX_SCENE],
    /// Stack index of each colour step, for the per-node curves.
    pub colour_node: [i32; MAX_NODES],
    /// The colour band's LUT holds linear light.
    pub lut_linear: bool,
}

impl PlanCore {
    pub fn detail(&self) -> &[Step] {
        &self.detail[..self.detail_count as usize]
    }
    pub fn colour(&self) -> &[Step] {
        &self.colour[..self.colour_count as usize]
    }
    pub fn screen(&self) -> &[Step] {
        &self.screen[..self.screen_count as usize]
    }
    pub fn scene(&self) -> &[Step] {
        &self.scene[..self.scene_count as usize]
    }
    pub fn empty(&self) -> bool {
        self.detail_count + self.colour_count + self.screen_count == 0
    }
    pub fn bypass(&self) -> bool {
        self.empty() || self.peek
    }
    pub fn has(&self, kind: u8) -> bool {
        self.detail().iter().chain(self.screen()).any(|s| s.kind == kind)
    }
    pub fn scene_step(&self, kind: u8) -> Option<&Step> {
        self.scene().iter().find(|s| s.kind == kind)
    }
    /// The shared quarter-resolution pyramid: Details, Tilt-Shift, Painterly, Watercolor.
    pub fn needs_pyramid(&self) -> bool {
        self.has(DETAILS) || self.has(TILT_SHIFT) || self.has(PAINTERLY) || self.has(WATERCOLOR)
    }
    /// Neighbouring source pixels: Sharpen, Watercolor, Retro. On GLES this rules out fetch.
    pub fn needs_source(&self) -> bool {
        self.has(SHARPEN) || self.has(WATERCOLOR) || self.has(RETRO) || self.needs_pyramid()
    }
    pub fn same_colour_band(&self, o: &PlanCore) -> bool {
        self.colour() == o.colour()
    }
}

fn decode_node(n: &Node) -> Step {
    let mut s = Step { kind: n.kind, intensity: n.strength(), v: [0.0; MAX_PARAMS] };
    for i in 0..info(n.kind).param_count as usize {
        s.v[i] = n.value(i);
    }
    s
}

/// A node that cannot change a pixel; dropped, so it costs nothing.
fn inert(n: &Node, s: &Step) -> bool {
    if !n.enabled || n.intensity == 0 {
        return true;
    }
    match info(n.kind).band {
        BAND_COLOUR => is_identity_colour(s),
        BAND_DETAIL => {
            if n.kind == DETAILS {
                return s.v[p::CLARITY] == 0.0 && s.v[p::HDR_TONING] == 0.0 && s.v[p::BLOOM] == 0.0;
            }
            n.kind == TILT_SHIFT && s.v[p::BLUR] == 0.0
        }
        BAND_SCENE => {
            (n.kind == DEPTH_FOG && s.v[p::DENSITY] == 0.0)
                || (n.kind == DEPTH_OF_FIELD && s.v[p::DOF_BLUR] == 0.0)
                || (n.kind == GLOBAL_ILLUMINATION && s.v[p::BOUNCE] == 0.0)
                || (n.kind == SKY && s.v[..4].iter().all(|&v| v == 0.0))
                || (n.kind == OUTLINE && s.v[p::LINE_DARKNESS] == 0.0)
                || (n.kind == PATH_TRACED_LIGHT && s.v[p::PATH_BOUNCE] == 0.0)
        }
        _ => false,
    }
}

/// Compiles a validated stack.
pub fn compile(stack: &Stack) -> PlanCore {
    let mut plan = PlanCore {
        revision: stack.revision,
        peek: stack.peek,
        split: stack.split,
        inspect: stack.inspect,
        split_x: stack.split_x as f32 / 65535.0,
        ..Default::default()
    };
    let count = stack.count.clamp(0, MAX_NODES as i32) as usize;
    for (i, n) in stack.nodes[..count].iter().enumerate() {
        let s = decode_node(n);
        if inert(n, &s) {
            continue;
        }
        match info(n.kind).band {
            BAND_DETAIL if (plan.detail_count as usize) < MAX_DETAIL => {
                plan.detail[plan.detail_count as usize] = s;
                plan.detail_count += 1;
            }
            BAND_COLOUR => {
                plan.colour_node[plan.colour_count as usize] = i as i32;
                plan.colour[plan.colour_count as usize] = s;
                plan.colour_count += 1;
            }
            BAND_SCREEN if (plan.screen_count as usize) < MAX_SCREEN => {
                plan.screen[plan.screen_count as usize] = s;
                plan.screen_count += 1;
            }
            BAND_SCENE if (plan.scene_count as usize) < MAX_SCENE => {
                plan.scene[plan.scene_count as usize] = s;
                plan.scene_count += 1;
            }
            _ => {}
        }
    }
    plan.lut_linear = ends_linear(plan.colour());
    plan
}

/// Mirrors framefx::kelir::Target.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Target {
    pub format: u8,
    pub reads_linear: bool,
    pub fetch: bool,
    pub msaa: bool,
    pub role: u8,
}

/// Mirrors framefx::kelir::Program.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Program {
    pub target: Target,
    pub lut: bool,
    pub detail: [i32; MAX_DETAIL],
    pub sharpen_plus: bool,
    pub screen: [i32; MAX_SCREEN],
}

fn detail_code(kind: u8) -> u64 {
    (kind - SHARPEN) as u64 + 1
}

fn screen_code(kind: u8) -> u64 {
    (kind - VIGNETTE) as u64 + 1
}

/// Bits: 0-1 format class, 2 fetch, 3 MSAA, 4-5 role, 6 LUT, 7 linear reads, 8-19 detail kinds
/// (4 bits each), 20 Sharpen Plus, 24-35 screen kinds (3 bits each). Exact, so never colliding.
pub fn signature(plan: &PlanCore, t: &Target) -> u64 {
    let mut s = t.format as u64
        | (t.fetch as u64) << 2
        | (t.msaa as u64) << 3
        | (t.role as u64) << 4
        | (t.reads_linear as u64) << 7;
    if t.role != ROLE_STAGE_B {
        s |= ((plan.colour_count > 0) as u64) << 6;
        for (i, d) in plan.detail().iter().enumerate() {
            s |= detail_code(d.kind) << (8 + 4 * i);
            if d.kind == SHARPEN && d.v[p::QUALITY] as i32 == SHARPEN_PLUS {
                s |= 1 << 20;
            }
        }
    }
    // Stage A leaves the screen band to the presents, so a screen edit never rebuilds it.
    if t.role != ROLE_STAGE_A {
        for (i, sc) in plan.screen().iter().enumerate() {
            s |= screen_code(sc.kind) << (24 + 3 * i);
        }
    }
    s
}

pub fn unpack(s: u64) -> Program {
    let mut p = Program {
        target: Target {
            format: (s & 3) as u8,
            fetch: s >> 2 & 1 != 0,
            msaa: s >> 3 & 1 != 0,
            role: (s >> 4 & 3) as u8,
            reads_linear: s >> 7 & 1 != 0,
        },
        lut: s >> 6 & 1 != 0,
        detail: [-1; MAX_DETAIL],
        sharpen_plus: s >> 20 & 1 != 0,
        screen: [-1; MAX_SCREEN],
    };
    for i in 0..MAX_DETAIL {
        let code = (s >> (8 + 4 * i) & 0xf) as i32;
        p.detail[i] = if code == 0 { -1 } else { code - 1 + SHARPEN as i32 };
    }
    for i in 0..MAX_SCREEN {
        let code = (s >> (24 + 3 * i) & 0x7) as i32;
        p.screen[i] = if code == 0 { -1 } else { code - 1 + VIGNETTE as i32 };
    }
    p
}

/// Plan constants the pass reads, PLAN_VEC4S vec4s: per detail slot two vec4s of parameters
/// (0..5), per screen slot two (6..13), detail intensities (14.xyz), 1 in 14.w when the LUT holds
/// linear light, screen intensities (15).
pub fn pack_plan(plan: &PlanCore, out: &mut [f32; PLAN_VEC4S * 4]) {
    out.fill(0.0);
    for (i, d) in plan.detail().iter().enumerate() {
        out[8 * i..8 * i + MAX_PARAMS].copy_from_slice(&d.v);
        out[14 * 4 + i] = d.intensity;
    }
    for (i, s) in plan.screen().iter().enumerate() {
        out[6 * 4 + 8 * i..6 * 4 + 8 * i + MAX_PARAMS].copy_from_slice(&s.v);
        out[15 * 4 + i] = s.intensity;
    }
    out[14 * 4 + 3] = if plan.colour_count > 0 && plan.lut_linear { 1.0 } else { 0.0 };
}

/// The scene band's constants for Luwes's scene passes (luwes/shaders/scene.frag).
pub fn pack_scene(plan: &PlanCore, out: &mut [f32; SCENE_VEC4S * 4]) {
    out.fill(0.0);
    if let Some(ao) = plan.scene_step(AMBIENT_OCCLUSION) {
        out[0] = ao.v[p::RADIUS];
        out[1] = ao.intensity;
    }
    if let Some(fog) = plan.scene_step(DEPTH_FOG) {
        let density = fog.v[p::DENSITY];
        out[4] = fog.v[p::FOG_START];
        out[5] = 0.02 * density * density;
        out[6] = fog.intensity;
        // HSV, hue in degrees.
        let h = (fog.v[p::FOG_HUE] % 360.0) / 60.0;
        let sat = fog.v[p::FOG_SATURATION];
        let v = fog.v[p::FOG_BRIGHTNESS];
        let x = 1.0 - ((h % 2.0) - 1.0).abs();
        let sector = (h as i32 % 6) as usize;
        let rgb = [[1.0, x, 0.0], [x, 1.0, 0.0], [0.0, 1.0, x], [0.0, x, 1.0], [x, 0.0, 1.0], [1.0, 0.0, x]];
        for i in 0..3 {
            out[8 + i] = v * (1.0 - sat + sat * rgb[sector][i]);
        }
    }
    if let Some(dof) = plan.scene_step(DEPTH_OF_FIELD) {
        out[12] = if dof.v[p::FOCUS] as i32 == FIXED_FOCUS { 1.0 } else { 0.0 };
        out[13] = dof.v[p::DISTANCE];
        out[14] = 0.02 * dof.v[p::DOF_BLUR];
        out[15] = dof.v[p::RANGE];
        out[16] = dof.intensity;
    }
    if let Some(gi) = plan.scene_step(GLOBAL_ILLUMINATION) {
        out[20] = gi.v[p::REACH];
        out[21] = gi.v[p::BOUNCE];
        out[22] = gi.intensity;
    }
}

/// Mirrors framefx::kelir::FrameInfo.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInfo {
    pub buffer_width: i32,
    pub buffer_height: i32,
    pub display_width: f32,
    pub display_height: f32,
    pub turns: i32,
    pub flip_y: bool,
    pub seed: u32,
    pub primaries: u8,
    pub format: u8,
    pub edges: bool,
}

/// One present's target, for the frame constants (kelir/shaders/common.glsl, FRAME).
pub fn pack_frame(plan: &PlanCore, f: &FrameInfo, out: &mut [f32; FRAME_VEC4S * 4]) {
    let luma = luma_weights(f.primaries);
    let lsb = if f.format == FORMAT_UNORM10 { 1.0 / 1023.0 } else { 1.0 / 255.0 };
    out.fill(0.0);
    let values = [
        f.buffer_width as f32,
        f.buffer_height as f32,
        f.display_width,
        f.display_height,
        (f.turns & 3) as f32,
        if f.flip_y { 1.0 } else { 0.0 },
        if plan.split { plan.split_x } else { -1.0 },
        // The seed travels as an exact float: 23 bits wrap after a day at 135 Hz.
        (f.seed & 0x7fffff) as f32,
        luma[0],
        luma[1],
        luma[2],
        0.5 * lsb,
        if f.edges { 1.0 } else { 0.0 },
    ];
    out[..values.len()].copy_from_slice(&values);
}

/// Which stages run on one present. Mirrors framefx::kelir::Stages.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stages {
    pub full: bool,
    pub stage_a: bool,
    pub stage_b: bool,
}

/// `generation`: Luwes presents this swapchain. `real`: the present carries the game's own frame.
pub fn schedule(plan: Option<&PlanCore>, generation: bool, real: bool) -> Stages {
    let mut s = Stages::default();
    let Some(plan) = plan.filter(|p| !p.bypass()) else { return s };
    if !generation {
        s.full = real;
        return s;
    }
    s.stage_a = real && plan.detail_count + plan.colour_count > 0;
    s.stage_b = true;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_unpack_to_what_they_pack() {
        let mut stack = Stack { count: 3, ..Default::default() };
        stack.nodes[0] = Node { kind: SHARPEN, enabled: true, intensity: 65535, q: [0, 1, 0, 0, 0, 0, 0, 0] };
        stack.nodes[1] = Node { kind: COLOR, enabled: true, intensity: 65535, q: [0, 0, 0, 10000, 0, 0, 0, 0] };
        stack.nodes[2] = Node { kind: VIGNETTE, enabled: true, intensity: 30000, q: [0; MAX_PARAMS] };
        let plan = compile(&stack);
        assert_eq!((plan.detail_count, plan.colour_count, plan.screen_count), (1, 1, 1));
        let t = Target { format: 0, reads_linear: false, fetch: true, msaa: false, role: 0 };
        let p = unpack(signature(&plan, &t));
        assert_eq!(p.target, t);
        assert!(p.lut && p.sharpen_plus);
        assert_eq!(p.detail, [SHARPEN as i32, -1, -1]);
        assert_eq!(p.screen, [VIGNETTE as i32, -1, -1, -1]);
    }

    #[test]
    fn inert_nodes_drop() {
        let mut stack = Stack { count: 2, ..Default::default() };
        stack.nodes[0] = Node { kind: COLOR, enabled: true, intensity: 65535, q: [0; MAX_PARAMS] };
        stack.nodes[1] = Node { kind: VIGNETTE, enabled: false, intensity: 65535, q: [0; MAX_PARAMS] };
        assert!(compile(&stack).empty());
    }
}
