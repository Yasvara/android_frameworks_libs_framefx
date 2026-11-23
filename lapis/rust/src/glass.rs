// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The transparency cut: where in the frame the opaque scene is whole and the transparent draws
//! (glass, water, particles, bubbles) have not begun, so Luwes can take the two layers apart.
//! Every game, no table: the pass graph says which class the frame is, and the cut must hold
//! still for a run of frames before the recorder acts on it.
//!
//! Classes:
//! - In one pass: opaque and transparent draws share the scene pass, transparent last. Cut at the
//!   first transparent draw after the pass's last depth write.
//! - A pass of their own: the transparent draws load the opaque target in a later pass. No split:
//!   copy the target before that pass begins.
//! - Refraction copy: the game already copies the opaque target between the two (an opaque
//!   texture for water and glass). That copy is the opaque layer; tap it.

use crate::model::*;

/// Per pass, what the recorder counted. Mirrored by framefx::lapis::GlassPass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GlassPass {
    /// The first blended, depth-tested, non-writing draw after the pass's last depth write;
    /// u32::MAX when none.
    pub glass_from: u32,
    /// The pass's last depth-writing draw, u32::MAX when none.
    pub last_depth_write: u32,
}

pub const CLASS_NONE: u8 = 0;
pub const CLASS_IN_PASS: u8 = 1;
pub const CLASS_OWN_PASS: u8 = 2;
pub const CLASS_REFRACTION: u8 = 3;

pub const REFUSED_NONE: u8 = 0;
pub const REFUSED_NO_SCENE: u8 = 1;
pub const REFUSED_NO_GLASS: u8 = 2;
pub const REFUSED_READS_TARGET: u8 = 3;
pub const REFUSED_MSAA: u8 = 4;
pub const REFUSED_UNSTABLE: u8 = 5;

/// The cut this frame shows. Mirrored by framefx::lapis::GlassCut.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlassCut {
    pub class: u8,
    pub refused: u8,
    /// Held still for long enough: the recorder acts on it.
    pub armed: bool,
    /// Riding out an odd frame via patience: keep the armed cut, never index this frame's steps.
    pub held: bool,
    /// The step: the pass to split (in pass), the pass to copy before (own pass), the copy.
    /// Negative when none or when `held`.
    pub step: i32,
    /// In pass: the draw to split at.
    pub draw: u32,
    /// The colour target the opaque scene is in (or the refraction copy's destination).
    pub target: ImageId,
}

const _: () = assert!(
    core::mem::size_of::<GlassCut>() == 16
        && core::mem::offset_of!(GlassCut, held) == 3
        && core::mem::offset_of!(GlassCut, step) == 4
        && core::mem::offset_of!(GlassCut, target) == 12
);

/// Frames a cut must hold before it is armed, and misses before an armed one is dropped.
pub const STABLE_FRAMES: u32 = 8;
pub const PATIENCE: u32 = 3;

/// Finds the cut in one frame.
pub fn find(v: &View, passes: &[GlassPass], depth: ImageId) -> GlassCut {
    let mut cut = GlassCut { step: -1, ..Default::default() };
    if depth == 0 {
        cut.refused = REFUSED_NO_SCENE;
        return cut;
    }
    let pass = |k: usize| passes.get(k).copied().unwrap_or(GlassPass { glass_from: u32::MAX, last_depth_write: u32::MAX });
    // The last pass that writes the scene's depth holds the opaque scene.
    let Some(opaque) = (0..v.steps.len())
        .rev()
        .find(|&k| v.steps[k].is_pass && v.steps[k].has_depth && v.steps[k].depth.image == depth && v.steps[k].depth_writes > 0)
    else {
        cut.refused = REFUSED_NO_SCENE;
        return cut;
    };
    let s = &v.steps[opaque];
    let Some(target) = v.colors(s).first().copied() else {
        cut.refused = REFUSED_NO_SCENE;
        return cut;
    };
    if target.samples > 1 {
        cut.refused = REFUSED_MSAA;
        return cut;
    }
    let p = pass(opaque);
    if p.glass_from != u32::MAX && p.glass_from < s.draws {
        if v.reads(s).contains(&target.image) {
            cut.refused = REFUSED_READS_TARGET;
            return cut;
        }
        cut.class = CLASS_IN_PASS;
        cut.step = opaque as i32;
        cut.draw = p.glass_from;
        cut.target = target.image;
        return cut;
    }
    // After the opaque pass: a copy of its target (refraction), or a pass that loads it and draws
    // transparency over it.
    for k in opaque + 1..v.steps.len() {
        let t = &v.steps[k];
        if t.transfers() && t.src == target.image {
            cut.class = CLASS_REFRACTION;
            cut.step = k as i32;
            cut.target = t.dst;
            return cut;
        }
        if t.is_pass && v.colors(t).iter().any(|c| c.image == target.image) {
            let q = pass(k);
            let tests_scene_depth = t.has_depth && t.depth.image == depth;
            if tests_scene_depth && q.glass_from != u32::MAX && t.depth_writes == 0 {
                cut.class = CLASS_OWN_PASS;
                cut.step = k as i32;
                cut.target = target.image;
                return cut;
            }
            break;
        }
    }
    cut.refused = REFUSED_NO_GLASS;
    cut
}

/// The cut across frames: armed once it holds still, dropped after a few misses.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GlassTracker {
    last: GlassCut,
    same: u32,
    missed: u32,
}

impl GlassTracker {
    pub fn update(&mut self, now: GlassCut) -> GlassCut {
        let same = now.class != CLASS_NONE
            && now.class == self.last.class
            && now.step == self.last.step
            && now.draw == self.last.draw
            && now.target == self.last.target;
        if same {
            self.same = self.same.saturating_add(1);
            self.missed = 0;
        } else if self.last.armed && self.missed < PATIENCE {
            // An armed cut rides out a few odd frames (a loading flash, a menu). Do not hand back
            // the previous frame's step index: the current frame may have a different step count.
            self.missed += 1;
            return GlassCut { held: true, step: -1, ..self.last };
        } else {
            self.same = if now.class != CLASS_NONE { 1 } else { 0 };
            self.missed = 0;
        }
        let mut out = now;
        out.armed = now.class != CLASS_NONE && self.same >= STABLE_FRAMES;
        if now.class != CLASS_NONE && !out.armed && out.refused == REFUSED_NONE {
            out.refused = REFUSED_UNSTABLE;
        }
        self.last = out;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct B {
        steps: Vec<Step>,
        targets: Vec<Target>,
        reads: Vec<ImageId>,
        passes: Vec<GlassPass>,
    }

    fn blank() -> Step {
        Step {
            is_pass: false,
            has_depth: false,
            op_kind: 0,
            flips: 0,
            depth: Target { image: 0, format: 0, load: 0, store: 0, samples: 1 },
            clear_depth: f32::NAN,
            draws: 0,
            depth_writes: 0,
            depth_clears: 0,
            last_clear_draw: 0,
            colors: Span { start: 0, len: 0 },
            reads: Span { start: 0, len: 0 },
            src: 0,
            dst: 0,
        }
    }

    impl B {
        fn new() -> Self {
            B { steps: vec![], targets: vec![], reads: vec![], passes: vec![] }
        }
        fn pass(&mut self, color: ImageId, depth: ImageId, draws: u32, writes: u32, reads: &[ImageId], glass: GlassPass) {
            let c = Span { start: self.targets.len() as u32, len: 1 };
            self.targets.push(Target { image: color, format: 97, load: 0, store: 0, samples: 1 });
            let r = Span { start: self.reads.len() as u32, len: reads.len() as u32 };
            self.reads.extend_from_slice(reads);
            let mut s = Step { is_pass: true, colors: c, reads: r, draws, depth_writes: writes, ..blank() };
            if depth != 0 {
                s.has_depth = true;
                s.depth.image = depth;
            }
            self.steps.push(s);
            self.passes.push(glass);
        }
        fn copy(&mut self, src: ImageId, dst: ImageId) {
            self.steps.push(Step { op_kind: OP_COPY, src, dst, ..blank() });
            self.passes.push(GlassPass::default());
        }
        fn find(&self, depth: ImageId) -> GlassCut {
            let frame = Frame {
                steps: self.steps.as_ptr(),
                step_count: self.steps.len() as u32,
                targets: self.targets.as_ptr(),
                target_count: self.targets.len() as u32,
                reads: self.reads.as_ptr(),
                read_count: self.reads.len() as u32,
                images: core::ptr::null(),
                image_count: 0,
                presented: 9,
                width: 0,
                height: 0,
            };
            // SAFETY: the vectors outlive the view.
            let v = unsafe { View::new(&frame) };
            find(&v, &self.passes, depth)
        }
    }

    const NONE: GlassPass = GlassPass { glass_from: u32::MAX, last_depth_write: u32::MAX };

    #[test]
    fn genshin_like_cuts_inside_the_scene_pass() {
        let mut b = B::new();
        b.pass(5, 3, 400, 300, &[], GlassPass { glass_from: 378, last_depth_write: 377 });
        b.pass(9, 0, 30, 0, &[5], NONE);
        let c = b.find(3);
        assert_eq!((c.class, c.step, c.draw, c.target), (CLASS_IN_PASS, 0, 378, 5));
    }

    #[test]
    fn a_separate_transparent_pass_copies_before_it() {
        let mut b = B::new();
        b.pass(5, 3, 300, 300, &[], NONE);
        b.pass(5, 3, 40, 0, &[], GlassPass { glass_from: 0, last_depth_write: u32::MAX });
        let c = b.find(3);
        assert_eq!((c.class, c.step, c.target), (CLASS_OWN_PASS, 1, 5));
    }

    #[test]
    fn a_refraction_copy_is_the_opaque_layer() {
        let mut b = B::new();
        b.pass(5, 3, 300, 300, &[], NONE);
        b.copy(5, 7);
        b.pass(5, 3, 40, 0, &[7], GlassPass { glass_from: 0, last_depth_write: u32::MAX });
        let c = b.find(3);
        assert_eq!((c.class, c.step, c.target), (CLASS_REFRACTION, 1, 7));
    }

    #[test]
    fn refuses_what_it_cannot_cut() {
        let mut b = B::new();
        b.pass(5, 3, 300, 300, &[5], GlassPass { glass_from: 290, last_depth_write: 289 });
        assert_eq!(b.find(3).refused, REFUSED_READS_TARGET);
        let mut c = B::new();
        c.pass(5, 3, 300, 300, &[], NONE);
        assert_eq!(c.find(3).refused, REFUSED_NO_GLASS);
        assert_eq!(c.find(0).refused, REFUSED_NO_SCENE);
    }

    #[test]
    fn arms_after_holding_still_and_rides_out_a_miss() {
        let mut t = GlassTracker::default();
        let cut = GlassCut { class: CLASS_IN_PASS, step: 2, draw: 50, target: 5, ..Default::default() };
        for _ in 0..STABLE_FRAMES - 1 {
            assert!(!t.update(cut).armed);
        }
        assert!(t.update(cut).armed);
        let held = t.update(GlassCut::default());
        assert!(held.armed);
        assert!(held.held);
        assert_eq!(held.step, -1);
        let resumed = t.update(cut);
        assert!(resumed.armed);
        assert!(!resumed.held);
        assert_eq!(resumed.step, 2);
        for _ in 0..=PATIENCE {
            t.update(GlassCut::default());
        }
        assert!(!t.update(GlassCut::default()).armed);
    }
}
