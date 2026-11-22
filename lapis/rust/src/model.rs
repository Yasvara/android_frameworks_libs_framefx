// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! The flat frame the engine hands over and the roles it gets back. Layouts are shared with
//! lapis/classifier.h and lapis/classifier.cpp, which assert the same sizes and offsets.

use core::mem::{offset_of, size_of};
use core::ops::Range;
use core::slice;

pub type ImageId = u32;

pub const LOAD_OP_LOAD: u32 = 0;
pub const STORE_OP_STORE: u32 = 0;
pub const STORE_OP_DONT_CARE: u32 = 1;
pub const SAMPLE_COUNT_1: u32 = 1;

pub const FLIP_Y: u8 = 1;
pub const FLIP_X: u8 = 2;
pub const OP_COPY: u8 = 1;
pub const OP_BLIT: u8 = 2;
pub const OP_RESOLVE: u8 = 3;
pub const OP_CLEAR: u8 = 5;
pub const OP_OPTICAL_FLOW: u8 = 8;

pub const TIER_NONE: u8 = 0;
pub const TIER_A: u8 = 1;
pub const TIER_B: u8 = 2;
pub const TIER_C: u8 = 3;

pub const MISS_NONE: u8 = 0;
pub const MISS_NO_FRAME: u8 = 1;
pub const MISS_NO_SCENE: u8 = 2;
pub const MISS_GAME_GENERATES: u8 = 3;
pub const MISS_CHANGED: u8 = 4;
pub const MISS_RESIZED: u8 = 5;
pub const MISS_UNTONED: u8 = 6;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Image {
    pub id: ImageId,
    pub format: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Target {
    pub image: ImageId,
    pub format: u32,
    pub load: u32,
    pub store: u32,
    pub samples: u32,
}

/// A run of `Frame::targets` or `Frame::reads`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Span {
    pub start: u32,
    pub len: u32,
}

impl Span {
    fn range(self) -> Range<usize> {
        let start = self.start as usize;
        start..start.saturating_add(self.len as usize)
    }
}

/// A render pass, or the transfer, clear or dispatch `op_kind` names.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Step {
    pub is_pass: bool,
    pub has_depth: bool,
    pub op_kind: u8,
    /// FLIP_Y, FLIP_X: a blit's offsets, or a pass's first viewport.
    pub flips: u8,
    pub depth: Target,
    /// The pass's depth clear value, or the clear op's; NaN when none.
    pub clear_depth: f32,
    pub draws: u32,
    pub depth_writes: u32,
    pub depth_clears: u32,
    pub last_clear_draw: u32,
    pub colors: Span,
    pub reads: Span,
    pub src: ImageId,
    pub dst: ImageId,
}

impl Step {
    pub fn transfers(&self) -> bool {
        !self.is_pass && matches!(self.op_kind, OP_COPY | OP_BLIT | OP_RESOLVE)
    }
}

#[repr(C)]
pub struct Frame {
    pub steps: *const Step,
    pub step_count: u32,
    pub targets: *const Target,
    pub target_count: u32,
    pub reads: *const ImageId,
    pub read_count: u32,
    /// Sorted by id.
    pub images: *const Image,
    pub image_count: u32,
    pub presented: ImageId,
    pub width: u32,
    pub height: u32,
}

/// What a `Frame` points at.
pub struct View<'a> {
    pub steps: &'a [Step],
    targets: &'a [Target],
    reads: &'a [ImageId],
    images: &'a [Image],
    pub presented: ImageId,
    pub width: u32,
    pub height: u32,
}

unsafe fn parts<'a, T>(data: *const T, len: u32) -> &'a [T] {
    if data.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: the caller guarantees `len` valid elements at `data` for 'a.
        unsafe { slice::from_raw_parts(data, len as usize) }
    }
}

impl<'a> View<'a> {
    /// # Safety
    /// Every pointer in `frame` holds its count of elements, alive for 'a.
    pub unsafe fn new(frame: &'a Frame) -> Self {
        // SAFETY: per this function's contract.
        unsafe {
            View {
                steps: parts(frame.steps, frame.step_count),
                targets: parts(frame.targets, frame.target_count),
                reads: parts(frame.reads, frame.read_count),
                images: parts(frame.images, frame.image_count),
                presented: frame.presented,
                width: frame.width,
                height: frame.height,
            }
        }
    }

    pub fn colors(&self, s: &Step) -> &'a [Target] {
        self.targets.get(s.colors.range()).unwrap_or(&[])
    }

    pub fn reads(&self, s: &Step) -> &'a [ImageId] {
        self.reads.get(s.reads.range()).unwrap_or(&[])
    }

    fn image(&self, id: ImageId) -> Option<&'a Image> {
        self.images.binary_search_by_key(&id, |i| i.id).ok().map(|i| &self.images[i])
    }

    pub fn size_of(&self, id: ImageId) -> Size {
        self.image(id).map_or(Size::default(), |i| Size { width: i.width, height: i.height })
    }

    pub fn format_of(&self, id: ImageId) -> u32 {
        self.image(id).map_or(0, |i| i.format)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    pub fn aspect(self) -> f32 {
        let lo = self.width.min(self.height);
        if lo == 0 {
            0.0
        } else {
            self.width.max(self.height) as f32 / lo as f32
        }
    }

    pub fn long(self) -> u32 {
        self.width.max(self.height)
    }

    pub fn short(self) -> u32 {
        self.width.min(self.height)
    }

}

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub struct Roles {
    pub depth: ImageId,
    pub scene: ImageId,
    pub velocity: ImageId,
    pub target: ImageId,
    pub direct: bool,
    pub ui: i32,
    pub split: i32,
    pub untoned: ImageId,
    pub clear_split: bool,
    pub resolve: i32,
    pub jittered: ImageId,
    pub resolved: ImageId,
    pub geometry: i32,
    pub depth_stored: bool,
    pub reversed: bool,
    pub game_generates: bool,
    pub finished: bool,
    pub behind_blit: bool,
    pub tier: u8,
    pub miss: u8,
    /// The shapes of S and D, hashed; never 0 once classified. What a game keeps from frame to
    /// frame, while the images that play the roles change.
    pub signature: u32,
}

impl Default for Roles {
    fn default() -> Self {
        Roles {
            depth: 0,
            scene: 0,
            velocity: 0,
            target: 0,
            direct: false,
            ui: -1,
            split: -1,
            untoned: 0,
            clear_split: false,
            resolve: -1,
            jittered: 0,
            resolved: 0,
            geometry: -1,
            depth_stored: false,
            reversed: false,
            game_generates: false,
            finished: false,
            behind_blit: false,
            tier: TIER_NONE,
            miss: MISS_NO_FRAME,
            signature: 0,
        }
    }
}

const _: () = assert!(size_of::<Image>() == 16);
const _: () = assert!(size_of::<Target>() == 20);
const _: () = assert!(size_of::<Step>() == 68 && offset_of!(Step, depth) == 4);
const _: () = assert!(offset_of!(Step, dst) == 64);
const _: () = assert!(size_of::<Roles>() == 64 && offset_of!(Roles, miss) == 58);

pub fn is_depth_format(f: u32) -> bool {
    // D16 .. D32_S8.
    (124..=130).contains(&f)
}

/// Float colour: the scene before tonemapping.
pub fn is_untoned(f: u32) -> bool {
    // B10G11R11_UFLOAT, E5B9G9R9_UFLOAT, R16G16B16A16_SFLOAT, R32G32B32A32_SFLOAT.
    matches!(f, 122 | 123 | 97 | 109)
}

/// Colour formats engines keep linear or reversed depth in.
pub fn is_depth_like_format(f: u32) -> bool {
    // R32_SFLOAT, R16_SFLOAT, R16_UNORM.
    matches!(f, 100 | 76 | 70)
}

pub fn is_velocity_format(f: u32) -> bool {
    // R16G16_UNORM, R16G16_SNORM, R16G16_SFLOAT, A2R10G10B10_UNORM, A2B10G10R10_UNORM.
    matches!(f, 77 | 78 | 83 | 58 | 64)
}

pub fn is_colour(f: u32) -> bool {
    f != 0 && !is_depth_format(f) && !is_depth_like_format(f)
}
