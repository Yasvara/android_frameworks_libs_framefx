// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! What a frame's pass graph says beyond Lapis's roles: where the post-processing chain starts and
//! what it does, the scene colour before it, how much of the scene is transparent, the render scale,
//! and whether the game writes per-object stencil. Read off attachments, sizes, formats, reads and
//! draw state; never a shader or a game's image ids.

use crate::model::{is_depth_format, is_untoned, Roles, Size, View, STORE_OP_STORE};

/// Draw state bits the recorder files each draw under.
pub const DRAW_BLEND: usize = 1 << 0;
pub const DRAW_DEPTH_TEST: usize = 1 << 1;
pub const DRAW_DEPTH_WRITE: usize = 1 << 2;
pub const DRAW_STENCIL_WRITE: usize = 1 << 3;
pub const DRAW_STATES: usize = 16;

/// Per step, what the classifier's frame does not carry: draws counted by state, and the first
/// draw of each state in the pass. The recorder only counts; meaning is given here. Zero for ops.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PassExtra {
    pub draws_by_state: [u32; DRAW_STATES],
    /// u32::MAX when no draw had that state.
    pub first_by_state: [u32; DRAW_STATES],
    /// The depth attachment's stencil store op.
    pub stencil_store: u32,
    /// The first draw's viewport: x, y, width, height (height signed).
    pub viewport: [f32; 4],
}

impl PassExtra {
    fn count(&self, f: impl Fn(usize) -> bool) -> u32 {
        (0..DRAW_STATES).filter(|&k| f(k)).map(|k| self.draws_by_state[k]).sum()
    }

    /// Blended draws that test depth without writing it: particles, glass, foliage cards.
    pub fn transparent_draws(&self) -> u32 {
        self.count(transparent)
    }

    pub fn first_transparent_draw(&self) -> u32 {
        (0..DRAW_STATES).filter(|&k| transparent(k) && self.draws_by_state[k] > 0).map(|k| self.first_by_state[k]).min().unwrap_or(u32::MAX)
    }

    pub fn stencil_writes(&self) -> u32 {
        self.count(|k| k & DRAW_STENCIL_WRITE != 0)
    }
}

fn transparent(k: usize) -> bool {
    k & DRAW_BLEND != 0 && k & DRAW_DEPTH_TEST != 0 && k & DRAW_DEPTH_WRITE == 0
}

pub const POST_TEMPORAL: u32 = 1 << 0;
pub const POST_BLOOM: u32 = 1 << 1;
/// Reads scene depth after the scene is lit: depth of field, fog, SSR.
pub const POST_DEPTH_AWARE: u32 = 1 << 2;
pub const POST_MOTION_BLUR: u32 = 1 << 3;
pub const POST_TONEMAP: u32 = 1 << 4;
/// Full-screen passes after the tonemap: anti-aliasing, grain, aberration, sharpening.
pub const POST_SCREEN: u32 = 1 << 5;
pub const POST_UPSCALE: u32 = 1 << 6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PassGraph {
    pub structure: u64,
    pub scene_depth_writes: u32,
    pub scene_draws: u32,
    pub transparent_draws: u32,
    /// The scene pass that draws transparency last, and its first transparent draw: where a split
    /// would see the opaque scene alone. -1 / 0 when none.
    pub transparent_step: i32,
    pub transparent_from: u32,
    pub post: u32,
    /// The scene colour before post-processing: what the tonemap's input was before bloom, depth of
    /// field, motion blur. 0 when not found.
    pub pre_post: u32,
    pub pre_post_untoned: bool,
    /// The step that finishes it, and whether a later step writes it again (it must be copied there).
    pub pre_post_step: i32,
    pub pre_post_overwritten: bool,
    /// The tonemap step, -1 when none.
    pub tonemap: i32,
    /// Scene size over presented size, and the drawn part of the scene's image (dynamic resolution).
    pub render_scale: [f32; 2],
    pub drawn: [f32; 2],
    /// The scene's pass writes stencil per draw and keeps it.
    pub stencil_ids: bool,
    pub stencil_writes: u32,
}

const FNV: u64 = 0xcbf29ce484222325;

fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x100000001b3)
}

fn log2_bucket(n: u32) -> u64 {
    u64::from(32 - n.leading_zeros())
}

/// The graph's shape: steps, formats, sizes relative to the scene, load/store, draw-count scale.
/// Independent of image ids, which a game reallocates.
pub fn structure_hash(v: &View, scene: Size) -> u64 {
    let rel = |s: Size| -> u64 {
        if scene.width == 0 || s.width == 0 {
            return 0;
        }
        u64::from((s.width * 8 / scene.width.max(1)).min(64)) << 8 | u64::from((s.height * 8 / scene.height.max(1)).min(64))
    };
    let mut h = FNV;
    for s in v.steps {
        h = mix(h, u64::from(s.is_pass) << 8 | u64::from(s.op_kind));
        if !s.is_pass {
            continue;
        }
        for t in v.colors(s) {
            h = mix(h, u64::from(t.format) << 32 | rel(v.size_of(t.image)) << 4 | u64::from(t.load) << 2 | u64::from(t.store));
        }
        if s.has_depth {
            h = mix(h, 1 << 40 | u64::from(s.depth.format) << 8 | u64::from(s.depth.store));
        }
        h = mix(h, log2_bucket(s.draws) << 8 | log2_bucket(s.depth_writes));
    }
    h
}

fn writes(v: &View, step: usize, id: u32) -> bool {
    let s = &v.steps[step];
    if s.is_pass {
        v.colors(s).iter().any(|t| t.image == id) || (s.has_depth && s.depth.image == id)
    } else {
        s.transfers() && s.dst == id
    }
}

fn last_writer(v: &View, before: usize, id: u32) -> Option<usize> {
    (0..before).rev().find(|&k| writes(v, k, id))
}

fn same_size(a: Size, b: Size) -> bool {
    a.width.abs_diff(b.width) <= 2 && a.height.abs_diff(b.height) <= 2 && a.width > 0
}

fn full_float_read(v: &View, step: usize, scene: Size) -> Option<u32> {
    let s = &v.steps[step];
    v.reads(s)
        .iter()
        .copied()
        .filter(|&r| is_untoned(v.format_of(r)) && same_size(v.size_of(r), scene))
        .max_by_key(|&r| last_writer(v, step, r))
}

pub fn analyse(v: &View, extras: &[PassExtra], roles: &Roles) -> PassGraph {
    let mut g = PassGraph { transparent_step: -1, pre_post_step: -1, tonemap: -1, ..Default::default() };
    let extra = |k: usize| extras.get(k).copied().unwrap_or_default();
    let depth = roles.depth;
    let scene = if depth != 0 { v.size_of(depth) } else { v.size_of(roles.scene) };
    g.structure = structure_hash(v, scene);

    // Scene draws: passes that attach the scene's depth.
    for (k, s) in v.steps.iter().enumerate() {
        if !s.is_pass || !s.has_depth || depth == 0 || s.depth.image != depth {
            continue;
        }
        let e = extra(k);
        g.scene_depth_writes += s.depth_writes;
        g.scene_draws += s.draws;
        let t = e.transparent_draws();
        g.transparent_draws += t;
        if t > 0 {
            g.transparent_step = k as i32;
            g.transparent_from = e.first_transparent_draw();
        }
    }
    if depth == 0 {
        // No scene depth found: the biggest depth-writing pass stands for the scene.
        if let Some(s) = v.steps.iter().filter(|s| s.is_pass).max_by_key(|s| s.depth_writes) {
            g.scene_depth_writes = s.depth_writes;
            g.scene_draws = s.draws;
        }
    }
    if roles.geometry >= 0 {
        let k = roles.geometry as usize;
        let s = &v.steps[k];
        let e = extra(k);
        g.stencil_writes = e.stencil_writes();
        g.stencil_ids = s.has_depth && g.stencil_writes >= 8 && e.stencil_store == STORE_OP_STORE;
        let [_, _, w, h] = e.viewport;
        let image = v.size_of(s.depth.image);
        if image.width > 0 && w > 0.0 {
            g.drawn = [(w / image.width as f32).min(1.0), (h.abs() / image.height as f32).min(1.0)];
        }
    }
    if v.width > 0 && scene.width > 0 {
        let (pw, ph) = (v.width.max(v.height), v.width.min(v.height));
        let (sw, sh) = (scene.long(), scene.short());
        g.render_scale = [sw as f32 / pw as f32, sh as f32 / ph as f32];
    }

    // The tonemap: the pass that writes S (or what S came from through same-size unorm passes) from
    // a full-size float image.
    let ui = if roles.ui >= 0 { roles.ui as usize } else { v.steps.len() };
    let mut at = if roles.scene != 0 { last_writer(v, ui.min(v.steps.len()), roles.scene) } else { None };
    let mut screen_passes = 0;
    while let Some(k) = at {
        let s = &v.steps[k];
        if !s.is_pass {
            at = if s.transfers() { last_writer(v, k, s.src) } else { None };
            continue;
        }
        if full_float_read(v, k, scene).is_some() || v.reads(s).iter().any(|&r| is_untoned(v.format_of(r))) {
            g.tonemap = k as i32;
            break;
        }
        // An unorm pass between the tonemap and S: anti-aliasing, grain and the like.
        let input = v.reads(s).iter().copied().find(|&r| !is_depth_format(v.format_of(r)) && v.size_of(r).area() * 4 >= scene.area());
        screen_passes += 1;
        at = input.and_then(|r| last_writer(v, k, r));
        if screen_passes > 8 {
            break;
        }
    }
    if g.tonemap >= 0 {
        g.post |= POST_TONEMAP;
        if screen_passes > 0 {
            g.post |= POST_SCREEN;
        }
        let tk = g.tonemap as usize;
        let s_size = v.size_of(roles.scene);
        if s_size.area() > scene.area() + scene.area() / 8 {
            g.post |= POST_UPSCALE;
        }
        // Back from the tonemap's full-size input, through full-size float passes that are post
        // effects, to the scene colour before them; the temporal resolve stays (it is the clean frame).
        let mut colour = full_float_read(v, tk, scene).or_else(|| {
            v.reads(&v.steps[tk]).iter().copied().filter(|&r| is_untoned(v.format_of(r))).max_by_key(|&r| v.size_of(r).area())
        });
        let mut writer = colour.and_then(|c| last_writer(v, tk, c));
        let mut hops = 0;
        while let (Some(c), Some(w)) = (colour, writer) {
            let s = &v.steps[w];
            if w as i32 == roles.resolve || !s.is_pass || s.draws > 4 || (s.has_depth && s.depth_writes > 0) {
                break;
            }
            let reads_velocity = roles.velocity != 0 && v.reads(s).contains(&roles.velocity);
            // Motion plus two full-size float scenes (this frame's and the history): a temporal
            // resolve the classifier did not name. Its output is the clean frame.
            let floats = v.reads(s).iter().filter(|&&r| is_untoned(v.format_of(r)) && same_size(v.size_of(r), scene)).count();
            if reads_velocity && floats >= 2 {
                g.post |= POST_TEMPORAL;
                break;
            }
            let reads_depth = depth != 0 && v.reads(s).contains(&depth);
            let input = full_float_read(v, w, scene).filter(|&i| i != c);
            match input {
                Some(i) if reads_velocity || reads_depth => {
                    g.post |= if reads_velocity { POST_MOTION_BLUR } else { POST_DEPTH_AWARE };
                    colour = Some(i);
                    writer = last_writer(v, w, i);
                }
                _ => break,
            }
            hops += 1;
            if hops > 6 {
                break;
            }
        }
        if let (Some(c), Some(w)) = (colour, writer) {
            g.pre_post = c;
            g.pre_post_untoned = is_untoned(v.format_of(c));
            g.pre_post_step = w as i32;
            g.pre_post_overwritten = ((w + 1)..tk).any(|k| writes(v, k, c));
        }
        // Between the scene and the tonemap: small float targets fed from the scene are bloom.
        let from = g.pre_post_step.max(0) as usize;
        for k in from..tk {
            let s = &v.steps[k];
            let target = if s.is_pass { v.colors(s).first().map(|t| t.image) } else { None };
            let small = target.is_some_and(|t| {
                let z = v.size_of(t);
                z.area() > 0 && z.area() * 3 <= scene.area() && is_untoned(v.format_of(t))
            });
            if small && (k > from) && v.reads(s).iter().any(|&r| is_untoned(v.format_of(r))) {
                g.post |= POST_BLOOM;
            }
        }
    }
    if roles.resolve >= 0 {
        g.post |= POST_TEMPORAL;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Frame, Image, Span, Step, Target};

    const D24S8: u32 = 129;
    const R11G11B10: u32 = 122;
    const RGBA8: u32 = 37;
    const BGR10A2: u32 = 64;

    struct Builder {
        steps: Vec<Step>,
        targets: Vec<Target>,
        reads: Vec<u32>,
        images: Vec<Image>,
        extras: Vec<PassExtra>,
    }

    fn target(image: u32, format: u32) -> Target {
        Target { image, format, load: 1, store: STORE_OP_STORE, samples: 1 }
    }

    impl Builder {
        fn image(&mut self, id: u32, format: u32, w: u32, h: u32) {
            self.images.push(Image { id, format, width: w, height: h });
        }

        fn pass(&mut self, colors: &[(u32, u32)], depth: Option<u32>, reads: &[u32], draws: u32, writes: u32, extra: PassExtra) {
            let c = Span { start: self.targets.len() as u32, len: colors.len() as u32 };
            for &(i, f) in colors {
                self.targets.push(target(i, f));
            }
            let r = Span { start: self.reads.len() as u32, len: reads.len() as u32 };
            self.reads.extend_from_slice(reads);
            self.steps.push(Step {
                is_pass: true,
                has_depth: depth.is_some(),
                op_kind: 0,
                flips: 0,
                depth: target(depth.unwrap_or(0), D24S8),
                clear_depth: f32::NAN,
                draws,
                depth_writes: writes,
                depth_clears: 0,
                last_clear_draw: 0,
                colors: c,
                reads: r,
                src: 0,
                dst: 0,
            });
            self.extras.push(extra);
        }
    }

    /// Genshin's graph as the survey recorded it: G-buffer, lighting + transparency, velocity, TAA,
    /// a bloom chain, the tonemap into S, then the HUD into the swapchain.
    #[test]
    fn a_deferred_frame_gives_its_post_chain_and_the_colour_before_it() {
        let mut b = Builder { steps: vec![], targets: vec![], reads: vec![], images: vec![], extras: vec![] };
        for (id, f, w, h) in [
            (1, RGBA8, 2119, 979),
            (3000, D24S8, 2118, 978),
            (3087, RGBA8, 2118, 978),
            (2999, R11G11B10, 2118, 978),
            (3113, BGR10A2, 2118, 978),
            (3114, R11G11B10, 2118, 978),
            (3117, R11G11B10, 530, 245),
            (3122, R11G11B10, 530, 245),
            (3123, RGBA8, 2118, 978),
        ] {
            b.image(id, f, w, h);
        }
        let mut gbuf = PassExtra { first_by_state: [u32::MAX; DRAW_STATES], ..Default::default() };
        gbuf.draws_by_state[DRAW_DEPTH_TEST | DRAW_DEPTH_WRITE | DRAW_STENCIL_WRITE] = 330;
        gbuf.stencil_store = STORE_OP_STORE;
        gbuf.viewport = [0.0, 0.0, 2118.0, 978.0];
        b.pass(&[(3087, RGBA8)], Some(3000), &[], 330, 330, gbuf);
        let mut light = PassExtra { first_by_state: [u32::MAX; DRAW_STATES], ..Default::default() };
        light.draws_by_state[0] = 1;
        light.draws_by_state[DRAW_BLEND | DRAW_DEPTH_TEST] = 22;
        light.first_by_state[DRAW_BLEND | DRAW_DEPTH_TEST] = 1;
        b.pass(&[(2999, R11G11B10)], Some(3000), &[3087, 3000], 23, 0, light);
        b.pass(&[(3113, BGR10A2)], Some(3000), &[3000], 12, 0, PassExtra::default());
        b.pass(&[(3114, R11G11B10)], None, &[3113, 3000, 2999], 1, 0, PassExtra::default());
        b.pass(&[(3117, R11G11B10)], None, &[3114], 1, 0, PassExtra::default());
        b.pass(&[(3122, R11G11B10)], None, &[3117], 1, 0, PassExtra::default());
        b.pass(&[(3123, RGBA8)], None, &[3122, 3114], 1, 0, PassExtra::default());
        b.pass(&[(1, RGBA8)], None, &[3123], 98, 0, PassExtra::default());
        b.images.sort_by_key(|i| i.id);
        let frame = Frame {
            steps: b.steps.as_ptr(),
            step_count: b.steps.len() as u32,
            targets: b.targets.as_ptr(),
            target_count: b.targets.len() as u32,
            reads: b.reads.as_ptr(),
            read_count: b.reads.len() as u32,
            images: b.images.as_ptr(),
            image_count: b.images.len() as u32,
            presented: 1,
            width: 2119,
            height: 979,
        };
        // SAFETY: the arrays outlive the view.
        let view = unsafe { View::new(&frame) };
        let roles = Roles { depth: 3000, scene: 3123, velocity: 3113, ui: 7, resolve: 3, geometry: 0, ..Default::default() };
        let g = analyse(&view, &b.extras, &roles);
        assert_eq!(g.tonemap, 6);
        assert_eq!(g.pre_post, 3114, "the TAA output, before bloom");
        assert!(g.pre_post_untoned && !g.pre_post_overwritten);
        assert_eq!(g.post, POST_TEMPORAL | POST_BLOOM | POST_TONEMAP);
        assert_eq!((g.transparent_draws, g.transparent_step, g.transparent_from), (22, 1, 1));
        assert_eq!(g.scene_depth_writes, 330);
        assert!(g.stencil_ids);
        assert!((g.render_scale[0] - 2118.0 / 2119.0).abs() < 1e-4);
        assert_eq!(g.drawn, [1.0, 1.0]);
    }
}
