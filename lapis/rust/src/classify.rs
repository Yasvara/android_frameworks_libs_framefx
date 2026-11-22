// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Finds, in one frame, the scene depth D, the scene before the HUD S, the pass U that draws the
//! HUD over it, and the velocity buffer V.

use crate::model::*;

const MAX_TRANSFER_HOPS: i32 = 4;
const MAX_BLITS: i32 = 3;
const MAX_LINEAGE_HOPS: i32 = 16;
// How far back an effect the presenting pass reads is traced to the scene: a bloom's chain.
const EFFECT_HOPS: i32 = 4;
const MIN_STRETCHED_AREA: f32 = 0.04; // of the target's: a fifth of each side
const ASPECT_TOLERANCE: f32 = 0.02;
const MIN_DEPTH_AREA: f32 = 0.2; // of the swapchain's
const MIN_SCENE_SIDE: f32 = 0.9; // of the scene depth's sides
const SIZE_SLACK: u32 = 2; // pixels, between a velocity buffer and D

fn near(a: u32, b: u32) -> bool {
    a.abs_diff(b) <= SIZE_SLACK
}

fn colour_target<'a>(v: &View<'a>, s: &Step, id: ImageId) -> Option<&'a Target> {
    v.colors(s).iter().find(|t| t.image == id)
}

/// Step indices that write each image: attachments, transfer destinations and clears.
struct Writes {
    // (image, step), by image, steps ascending.
    at: Vec<(ImageId, i32)>,
}

impl Writes {
    fn new(v: &View) -> Self {
        let mut at = Vec::with_capacity(v.steps.len() * 2);
        for (i, s) in v.steps.iter().enumerate() {
            let mut add = |id: ImageId| {
                if id != 0 {
                    at.push((id, i as i32));
                }
            };
            if s.is_pass {
                v.colors(s).iter().for_each(|t| add(t.image));
                if s.has_depth && s.depth_writes > 0 {
                    add(s.depth.image);
                }
            } else if s.transfers() || s.op_kind == OP_CLEAR {
                add(s.dst);
            }
        }
        at.sort_by_key(|&(id, _)| id);
        Writes { at }
    }

    /// The last step before `at` that wrote `id`, or -1.
    fn before(&self, id: ImageId, at: i32) -> i32 {
        let lo = self.at.partition_point(|&(i, _)| i < id);
        let hi = self.at.partition_point(|&(i, _)| i <= id);
        let run = &self.at[lo..hi];
        match run.partition_point(|&(_, step)| step < at) {
            0 => -1,
            n => run[n - 1].1,
        }
    }
}

/// The scene depth: the depth attachment with the most depth-writing draws, shaped like the
/// swapchain either way up (shadow maps are not) and covering a real share of it.
fn find_depth(v: &View, screen: Size) -> ImageId {
    let mut writes: Vec<(ImageId, u32)> = Vec::new();
    for s in v.steps {
        if s.is_pass && s.has_depth && s.depth.image != 0 && s.depth_writes > 0 {
            match writes.iter_mut().find(|(id, _)| *id == s.depth.image) {
                Some((_, count)) => *count += s.depth_writes,
                None => writes.push((s.depth.image, s.depth_writes)),
            }
        }
    }
    let (mut best, mut most) = (0, 0);
    for &(id, count) in &writes {
        let size = v.size_of(id);
        // A game may draw across a swapchain it turns its frame into last.
        if size.area() == 0 {
            continue;
        }
        if (size.aspect() / screen.aspect() - 1.0).abs() > ASPECT_TOLERANCE {
            continue;
        }
        if (size.area() as f32) < MIN_DEPTH_AREA * screen.area() as f32 {
            continue;
        }
        if count > most || (count == most && id < best) {
            best = id;
            most = count;
        }
    }
    best
}

/// Whether D still holds this frame's scene at present: its last depth-writing pass stores it,
/// and nothing after clears, discards or overwrites it. Also returns whether D clears to 0.
fn depth_kept(v: &View, depth: ImageId) -> (bool, bool) {
    let mut last_write = None;
    let mut reversed = false;
    let mut cleared = false;
    for (i, s) in v.steps.iter().enumerate() {
        if s.is_pass && s.has_depth && s.depth.image == depth {
            if !cleared && !s.clear_depth.is_nan() {
                reversed = s.clear_depth < 0.5;
                cleared = true;
            }
            if s.depth_writes > 0 {
                last_write = Some(i);
            }
        } else if !s.is_pass && s.op_kind == OP_CLEAR && s.dst == depth && !cleared {
            reversed = s.clear_depth < 0.5;
            cleared = true;
        }
    }
    let Some(last) = last_write else { return (false, reversed) };
    if v.steps[last].depth.store != STORE_OP_STORE {
        return (false, reversed);
    }
    for s in &v.steps[last + 1..] {
        if s.is_pass && s.has_depth && s.depth.image == depth {
            if s.depth.load != LOAD_OP_LOAD || s.depth.store == STORE_OP_DONT_CARE {
                return (false, reversed);
            }
        } else if !s.is_pass && (s.transfers() || s.op_kind == OP_CLEAR) && s.dst == depth {
            return (false, reversed);
        }
    }
    (true, reversed)
}

/// The single-sample image the scene's depth is in as the pass `geometry` ends: its depth
/// attachment when it stores it, or what a multisampled pass resolves its own into and stores,
/// where a game renders into attachments it never stores (ANGLE's render to texture).
fn depth_as_drawn(v: &View, geometry: usize) -> Option<ImageId> {
    let s = &v.steps[geometry];
    if !s.has_depth {
        return None;
    }
    if s.depth.samples <= 1 {
        return (s.depth.store == STORE_OP_STORE).then_some(s.depth.image);
    }
    let size = v.size_of(s.depth.image);
    v.colors(s)
        .iter()
        .find(|c| {
            is_depth_format(c.format)
                && c.samples <= 1
                && c.store == STORE_OP_STORE
                && v.size_of(c.image) == size
        })
        .map(|c| c.image)
}

/// Where `image`, as step `after` left it, is at present: itself, or the image a copy taken
/// before anything drew over it put it in. None when a pass started it over, a transfer or clear
/// wrote it, or a pass resolved into it again first: a later pass of the game's that brings a
/// resolved depth back and resolves anew is not trusted to leave the scene's depth there.
fn kept_since(v: &View, image: ImageId, after: usize) -> Option<ImageId> {
    let mut at = image;
    for s in &v.steps[after + 1..] {
        if s.is_pass {
            let drawn = s.has_depth && s.depth.image == at && s.depth.load != LOAD_OP_LOAD;
            if drawn || v.colors(s).iter().any(|c| c.image == at) {
                return None;
            }
        } else if (s.transfers() || s.op_kind == OP_CLEAR) && s.dst == at {
            return None;
        } else if s.op_kind == OP_COPY && s.src == at && at == image {
            at = s.dst;
        }
    }
    Some(at)
}

/// A velocity buffer: written at D's size and read by a pass that also reads an image this frame
/// did not make, at the same size: the temporal resolve with its history, whose index comes back
/// with it.
fn find_velocity(v: &View, writes: &Writes, scene: Size, before: i32) -> Option<(ImageId, i32)> {
    let before = (before.max(0) as usize).min(v.steps.len());
    for i in 0..before {
        let s = &v.steps[i];
        if !s.is_pass {
            continue;
        }
        for t in v.colors(s) {
            if !is_velocity_format(t.format) || t.store != STORE_OP_STORE {
                continue;
            }
            let size = v.size_of(t.image);
            if !near(size.width, scene.width) || !near(size.height, scene.height) {
                continue;
            }
            for j in i + 1..before {
                let r = &v.steps[j];
                if !r.is_pass || !v.reads(r).contains(&t.image) {
                    continue;
                }
                for &h in v.reads(r) {
                    let hs = v.size_of(h);
                    if h != t.image
                        && is_colour(v.format_of(h))
                        && writes.before(h, j as i32) < 0
                        && near(hs.width, scene.width)
                        && near(hs.height, scene.height)
                    {
                        return Some((t.image, j as i32));
                    }
                }
            }
        }
    }
    None
}

/// How well an image a presenting pass samples stands for the scene before the HUD.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Score {
    drawn: bool,
    made: i32,
    area: u64,
}

/// The depth `id`, as it is before step `at`, was drawn with: that of the depth-writing pass with
/// the most such draws among the passes it comes from, directly or through the passes and
/// transfers between (post effects, resolves, upscales). None when no draw behind it wrote depth.
fn drawn_with(v: &View, writes: &Writes, id: ImageId, at: i32) -> Option<ImageId> {
    let mut seen = vec![false; v.steps.len()];
    let mut queue: Vec<(i32, i32, ImageId)> = Vec::new();
    let mut push = |queue: &mut Vec<(i32, i32, ImageId)>, id: ImageId, at: i32, hops: i32| {
        let made = writes.before(id, at);
        if made >= 0 && hops < MAX_LINEAGE_HOPS && !seen[made as usize] {
            seen[made as usize] = true;
            queue.push((made, hops, id));
        }
    };
    push(&mut queue, id, at, 0);
    let mut best: Option<(u32, ImageId)> = None;
    let mut next = 0;
    while next < queue.len() {
        let (made, hops, id) = queue[next];
        next += 1;
        let s = &v.steps[made as usize];
        if !s.is_pass {
            if s.transfers() {
                push(&mut queue, s.src, made, hops + 1);
            }
            continue;
        }
        if s.has_depth && s.depth_writes > 0 {
            if best.is_none_or(|(most, _)| s.depth_writes > most) {
                best = Some((s.depth_writes, s.depth.image));
            }
            continue;
        }
        // A pass that loads the image draws over what the pass before it left there.
        if colour_target(v, s, id).is_some_and(|t| t.load == LOAD_OP_LOAD) {
            push(&mut queue, id, made, hops);
        }
        for &from in v.reads(s) {
            if is_colour(v.format_of(from)) {
                push(&mut queue, from, made, hops + 1);
            }
        }
    }
    best.map(|(_, depth)| depth)
}

/// The roles the passes that build `target`, last drawn by step `last`, give: D and S.
fn scan_target(v: &View, writes: &Writes, target: ImageId, last: usize, screen: Size) -> Roles {
    let steps = v.steps;
    let mut r = Roles { target, direct: target == v.presented, ..Roles::default() };
    r.depth = find_depth(v, screen);
    if r.depth != 0 {
        (r.depth_stored, r.reversed) = depth_kept(v, r.depth);
    }

    // The passes that build the target this frame, back to the one that starts it over.
    let mut presenting: Vec<usize> = Vec::new();
    for i in (0..=last.min(steps.len().saturating_sub(1))).rev() {
        if !steps[i].is_pass {
            continue;
        }
        let Some(t) = colour_target(v, &steps[i], target) else { continue };
        presenting.push(i);
        if t.load != LOAD_OP_LOAD {
            break;
        }
    }

    // S: among the full-size, tonemapped colour images this frame made that a presenting pass
    // samples, the one that scores best: drawn from the 3D scene, then the newest, then the
    // largest.
    let mut scene = v.size_of(r.depth);
    if scene.area() == 0 {
        scene = Size { width: screen.width / 2, height: screen.height / 2 };
    }
    let mut best = Score::default();
    let mut untoned_made = -1;
    // A pass that starts the target over puts what it samples under the rest. One that loads it
    // may only add a layer of the game's own, such as its HUD, over a frame from before: there
    // only an image drawn from the 3D scene is the scene.
    let started = presenting.last().is_some_and(|&p| {
        colour_target(v, &steps[p], target).is_some_and(|t| t.load != LOAD_OP_LOAD)
    });
    for &p in presenting.iter().rev() {
        if r.scene != 0 {
            break;
        }
        for &id in v.reads(&steps[p]) {
            let made = writes.before(id, p as i32);
            let format = v.format_of(id);
            if made < 0 || id == target || !is_colour(format) {
                continue;
            }
            let size = v.size_of(id);
            if (size.long() as f32) < MIN_SCENE_SIDE * scene.long() as f32
                || (size.short() as f32) < MIN_SCENE_SIDE * scene.short() as f32
            {
                continue;
            }
            if is_untoned(format) {
                if started && made > untoned_made {
                    r.untoned = id;
                    r.split = p as i32;
                    untoned_made = made;
                }
                continue;
            }
            let score = Score {
                drawn: drawn_with(v, writes, id, p as i32).is_some(),
                made,
                area: size.area(),
            };
            if (started || score.drawn) && score > best {
                best = score;
                r.scene = id;
                r.ui = p as i32;
            }
        }
    }

    // S read beside effects made from it this frame (a bloom, a blur), or kept deeper than the
    // target it goes into: the presenting pass finishes the look, whatever S's format, so S is the
    // scene before its tonemap and the split goes after the draws that read it.
    if r.scene != 0
        && started
        && (finished_later(v, writes, r.scene, r.ui as usize)
            || is_deeper(v.format_of(r.scene), v.format_of(target)))
    {
        r.untoned = r.scene;
        r.split = r.ui;
        r.scene = 0;
        r.ui = -1;
    }

    // Without S, a copy of the target taken between two passes that build it: the scene before
    // what the later pass draws over it (Lapis's own split, or a game keeping its frame).
    let mut k = 0;
    while r.scene == 0 && k + 1 < presenting.len() {
        for s in &steps[presenting[k + 1] + 1..presenting[k]] {
            if s.is_pass || !s.transfers() || s.src != target || s.dst == 0 || s.dst == target {
                continue;
            }
            if v.size_of(s.dst) == screen {
                r.scene = s.dst;
                r.ui = presenting[k] as i32;
            }
        }
        k += 1;
    }
    if r.scene != 0 {
        r.split = -1;
        r.untoned = 0;
    } else if r.untoned == 0 && presenting.len() == 1 {
        let p = &steps[presenting[0]];
        if p.depth_clears > 0 && p.last_clear_draw > 0 && p.last_clear_draw < p.draws {
            r.split = presenting[0] as i32;
            r.clear_split = true;
        }
    }
    r
}

/// A 10-bit colour image drawn into an 8-bit one: what the draw does on the way is the game's last
/// colour step.
fn is_deeper(scene: u32, target: u32) -> bool {
    // A2R10G10B10_UNORM, A2B10G10R10_UNORM into R8G8B8A8 / B8G8R8A8, UNORM or SRGB.
    matches!(scene, 58 | 64) && matches!(target, 37 | 43 | 44 | 50)
}

/// Whether pass `at`, which reads `scene`, also reads a colour image made this frame from it: an
/// effect of the scene's own the pass combines with it.
fn finished_later(v: &View, writes: &Writes, scene: ImageId, at: usize) -> bool {
    let reads_scene = |step: i32, id: ImageId| -> bool {
        // The step that made `id`, or a step before it in its lineage, sampled the scene.
        let mut seen = vec![false; v.steps.len()];
        let mut queue = vec![(step, id, 0)];
        let mut next = 0;
        while next < queue.len() {
            let (made, id, hops) = queue[next];
            next += 1;
            if std::mem::replace(&mut seen[made as usize], true) {
                continue;
            }
            let s = &v.steps[made as usize];
            if !s.is_pass {
                continue;
            }
            let reads = v.reads(s);
            if reads.contains(&scene) {
                return true;
            }
            if hops + 1 >= EFFECT_HOPS {
                continue;
            }
            for &from in reads {
                if from != id && is_colour(v.format_of(from)) {
                    let m = writes.before(from, made);
                    if m >= 0 {
                        queue.push((m, from, hops + 1));
                    }
                }
            }
        }
        false
    };
    let p = &v.steps[at];
    v.reads(p).iter().any(|&id| {
        if id == scene || !is_colour(v.format_of(id)) {
            return false;
        }
        let made = writes.before(id, at as i32);
        made >= 0 && reads_scene(made, id)
    })
}

/// The image a pass only blits into its target, turned or not: one draw over a target it does
/// not load (or a multisampled one resolved into it), sampling a single colour image of this
/// frame as large as the target, or of its shape. Bound descriptor sets may name smaller ones too.
fn blit_source(v: &View, writes: &Writes, target: ImageId, pass: usize) -> Option<ImageId> {
    let p = &v.steps[pass];
    let t = colour_target(v, p, target)?;
    if t.load == LOAD_OP_LOAD || p.draws != 1 {
        return None;
    }
    // A multisampled attachment the pass starts over and resolves into the target is the target;
    // a depth resolve beside it adds no colour.
    let others_resolve = v.colors(p).iter().filter(|c| c.image != target).all(|c| {
        !is_colour(c.format) || (c.samples != SAMPLE_COUNT_1 && c.load != LOAD_OP_LOAD)
    });
    if !others_resolve {
        return None;
    }
    let to = v.size_of(target);
    let mut source = None;
    for &id in v.reads(p) {
        if id == target || writes.before(id, pass as i32) < 0 || !is_colour(v.format_of(id)) {
            continue;
        }
        let from = v.size_of(id);
        let same = near(from.width, to.width) && near(from.height, to.height);
        let turned = near(from.width, to.height) && near(from.height, to.width);
        // A game that keeps the size it started at stretches its frame to a window grown since.
        let stretched = from.area() != 0
            && (from.aspect() / to.aspect() - 1.0).abs() <= ASPECT_TOLERANCE
            && from.area() as f32 >= MIN_STRETCHED_AREA * to.area() as f32;
        if !same && !turned && !stretched {
            continue;
        }
        if source.is_some() {
            return None;
        }
        source = Some(id);
    }
    source
}

fn signature(v: &View, r: &Roles) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for id in [r.scene, r.depth] {
        let size = v.size_of(id);
        for word in [size.width, size.height, v.format_of(id)] {
            hash = (hash ^ word).wrapping_mul(0x0100_0193);
        }
    }
    hash | 1
}

pub fn classify(v: &View) -> Roles {
    let steps = v.steps;

    // Back from the present to the pass that last drew the presented image, through copies.
    let mut target = v.presented;
    let mut last = None;
    let mut hops = 0;
    for (i, s) in steps.iter().enumerate().rev() {
        if target == 0 {
            break;
        }
        if s.is_pass && colour_target(v, s, target).is_some() {
            last = Some(i);
            break;
        }
        if s.transfers() && s.dst == target {
            hops += 1;
            if hops > MAX_TRANSFER_HOPS {
                break;
            }
            target = s.src;
        }
    }
    let Some(mut last) = last else { return Roles::default() };

    let screen = if v.width != 0 {
        Size { width: v.width, height: v.height }
    } else {
        v.size_of(target)
    };
    let writes = Writes::new(v);
    let mut r = scan_target(v, &writes, target, last, screen);

    // What a last pass only blits holds the finished frame, HUD and all, however it looks like S.
    // The frame's HUD is on it when the passes that build it sample a scene of their own;
    // otherwise the source stays S: a frame with no HUD to draw.
    // A pass-through may feed another (a multisampled copy, then the swapchain blit), and one at
    // the HUD's size may sit behind a scene drawn larger: walk them all before deciding.
    let (mut at, mut at_last) = (r.target, last);
    for _ in 0..MAX_BLITS {
        let Some(source) = blit_source(v, &writes, at, at_last) else { break };
        let made = writes.before(source, at_last as i32);
        if made < 0 || !steps[made as usize].is_pass {
            break;
        }
        let made = made as usize;
        let behind = scan_target(v, &writes, source, made, v.size_of(source));
        if behind.scene != 0 {
            r = Roles { behind_blit: true, ..behind };
            last = made;
        } else if blit_source(v, &writes, source, made)
            .is_none_or(|deeper| is_untoned(v.format_of(deeper)))
        {
            // Nothing further, or a tonemap: real work, not a pass-through.
            r.finished = r.scene == source;
            break;
        }
        (at, at_last) = (source, made);
    }

    // D is the depth S was drawn with, where its passes show it: a game may keep several of the
    // scene's size, and which has the most draws changes from frame to frame.
    if r.scene != 0 && r.ui >= 0 {
        if let Some(depth) = drawn_with(v, &writes, r.scene, r.ui) {
            if depth != 0 && depth != r.depth {
                r.depth = depth;
                (r.depth_stored, r.reversed) = depth_kept(v, depth);
            }
        }
    }

    if r.depth != 0 {
        let depth = v.size_of(r.depth);
        let before = if r.ui >= 0 { r.ui } else { last as i32 };
        if let Some((velocity, resolve)) = find_velocity(v, &writes, depth, before) {
            r.velocity = velocity;
            r.resolve = resolve;
            // The resolve's untoned input made this frame, and the untoned image it writes.
            let t = &steps[resolve as usize];
            for &id in v.reads(t) {
                let size = v.size_of(id);
                if id != r.velocity
                    && is_untoned(v.format_of(id))
                    && writes.before(id, resolve) >= 0
                    && near(size.width, depth.width)
                    && near(size.height, depth.height)
                {
                    r.jittered = id;
                }
            }
            if let Some(c) = v.colors(t).iter().find(|c| is_untoned(c.format)) {
                r.resolved = c.image;
            }
            if r.jittered == 0 || r.resolved == 0 {
                r.jittered = 0;
                r.resolved = 0;
            }
        }
        let mut most = 0;
        for (i, s) in steps.iter().enumerate() {
            if s.is_pass && s.has_depth && s.depth.image == r.depth && s.depth_writes > most {
                most = s.depth_writes;
                r.geometry = i as i32;
            }
        }
        if !r.depth_stored && r.geometry >= 0 {
            if let Some(drawn) = depth_as_drawn(v, r.geometry as usize) {
                // Not kept: D is still the image a copy would have to be taken of.
                let kept = kept_since(v, drawn, r.geometry as usize);
                r.depth = kept.unwrap_or(drawn);
                r.depth_stored = kept.is_some();
            }
        }
    }

    r.tier = if r.depth == 0 {
        TIER_C
    } else if r.depth_stored && r.scene != 0 {
        TIER_A
    } else {
        TIER_B
    };
    r.miss = if r.scene != 0 {
        MISS_NONE
    } else if r.untoned != 0 {
        MISS_UNTONED
    } else {
        MISS_NO_SCENE
    };
    r.signature = signature(v, &r);
    r.game_generates = steps.iter().any(|s| !s.is_pass && s.op_kind == OP_OPTICAL_FLOW);
    if r.game_generates && r.scene == 0 {
        r.miss = MISS_GAME_GENERATES;
    }
    r
}
