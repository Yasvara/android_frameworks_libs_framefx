// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Which way the scene reaches the screen, read from the game's own commands instead of guessed
//! from pictures. Copies and resolves keep it; a blit turns it as its offsets say. A pass that
//! samples it into another image (the final blit drawn as a quad) turns it by its viewport only
//! if its UVs are the plain ones, which the commands do not show: that is a hint, not a fact.

use crate::model::*;

/// What the commands say: the flips (FLIP_Y, FLIP_X), and whether they are certain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    Exact(u8),
    Hint(u8),
    Unknown,
}

/// From `from`, as it stands after step `after` (or from the start when None), to the presented
/// image.
pub fn orientation(v: &View, from: ImageId, after: Option<usize>) -> Orientation {
    if from == 0 || v.presented == 0 {
        return Orientation::Unknown;
    }
    let mut cur = from;
    let mut flips = 0u8;
    let mut hinted = false;
    let start = after.map_or(0, |a| a + 1);
    for s in v.steps.iter().skip(start) {
        if s.transfers() {
            if s.src == cur && s.dst != 0 && s.dst != cur {
                cur = s.dst;
                if s.op_kind == OP_BLIT {
                    flips ^= s.flips & (FLIP_Y | FLIP_X);
                }
            } else if s.dst == cur {
                // Something else lands where the scene was.
                return Orientation::Unknown;
            }
        } else if s.is_pass && v.reads(s).contains(&cur) {
            let colors = v.colors(s);
            if colors.iter().any(|t| t.image == cur) {
                continue;  // drawn over in place
            }
            // A composite into one image by a few draws: the quad that puts the scene on screen.
            if colors.len() == 1 && s.draws > 0 && s.draws <= 4 {
                cur = colors[0].image;
                flips ^= s.flips & FLIP_Y;
                hinted = true;
            } else {
                return Orientation::Unknown;
            }
        }
    }
    if cur != v.presented {
        Orientation::Unknown
    } else if hinted {
        Orientation::Hint(flips)
    } else {
        Orientation::Exact(flips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct B {
        steps: Vec<Step>,
        targets: Vec<Target>,
        reads: Vec<ImageId>,
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
            B { steps: vec![], targets: vec![], reads: vec![] }
        }
        fn op(&mut self, kind: u8, src: ImageId, dst: ImageId, flips: u8) {
            self.steps.push(Step { op_kind: kind, src, dst, flips, ..blank() });
        }
        fn pass(&mut self, colors: &[ImageId], reads: &[ImageId], draws: u32, flips: u8) {
            let c = Span { start: self.targets.len() as u32, len: colors.len() as u32 };
            for &i in colors {
                self.targets.push(Target { image: i, format: 37, load: 0, store: 0, samples: 1 });
            }
            let r = Span { start: self.reads.len() as u32, len: reads.len() as u32 };
            self.reads.extend_from_slice(reads);
            self.steps.push(Step { is_pass: true, colors: c, reads: r, draws, flips, ..blank() });
        }
        fn run(&self, presented: ImageId, from: ImageId, after: Option<usize>) -> Orientation {
            let frame = Frame {
                steps: self.steps.as_ptr(),
                step_count: self.steps.len() as u32,
                targets: self.targets.as_ptr(),
                target_count: self.targets.len() as u32,
                reads: self.reads.as_ptr(),
                read_count: self.reads.len() as u32,
                images: core::ptr::null(),
                image_count: 0,
                presented,
                width: 0,
                height: 0,
            };
            // SAFETY: the vectors outlive the view.
            let v = unsafe { View::new(&frame) };
            orientation(&v, from, after)
        }
    }

    #[test]
    fn drawn_straight_into_the_presented_image_is_upright() {
        let mut b = B::new();
        b.pass(&[9], &[], 300, 0);
        assert_eq!(b.run(9, 9, Some(0)), Orientation::Exact(0));
    }

    #[test]
    fn a_flipped_blit_turns_it() {
        let mut b = B::new();
        b.pass(&[5], &[], 300, 0);
        b.pass(&[5], &[], 20, 0); // the HUD, drawn over in place
        b.op(OP_BLIT, 5, 9, FLIP_Y);
        assert_eq!(b.run(9, 5, Some(0)), Orientation::Exact(FLIP_Y));
    }

    #[test]
    fn copies_keep_it_and_two_flips_cancel() {
        let mut b = B::new();
        b.op(OP_BLIT, 5, 6, FLIP_Y);
        b.op(OP_COPY, 6, 7, 0);
        b.op(OP_BLIT, 7, 9, FLIP_Y);
        assert_eq!(b.run(9, 5, None), Orientation::Exact(0));
    }

    #[test]
    fn a_quad_composite_is_only_a_hint() {
        let mut b = B::new();
        b.pass(&[9], &[5], 1, FLIP_Y);
        assert_eq!(b.run(9, 5, None), Orientation::Hint(FLIP_Y));
        let mut c = B::new();
        c.pass(&[9], &[5], 1, 0);
        assert_eq!(c.run(9, 5, None), Orientation::Hint(0));
    }

    #[test]
    fn a_busy_pass_or_an_overwrite_says_nothing() {
        let mut b = B::new();
        b.pass(&[9], &[5], 40, 0);
        assert_eq!(b.run(9, 5, None), Orientation::Unknown);
        let mut c = B::new();
        c.op(OP_COPY, 3, 5, 0);
        c.op(OP_COPY, 5, 9, 0);
        assert_eq!(c.run(9, 5, None), Orientation::Unknown);
        let mut d = B::new();
        d.op(OP_COPY, 5, 6, 0);
        assert_eq!(d.run(9, 5, None), Orientation::Unknown);
    }
}
