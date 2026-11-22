/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstdint>

#include "lapis/model.h"

// Finds, in one frame, what Luwes and Kelir can use from inside the game's rendering: the scene
// depth D, the scene before the HUD S, the pass U that draws the HUD over it, and the velocity
// buffer V. A Tracker engages the roles once they hold still for a run of frames.
namespace framefx::lapis {

enum class Tier : uint8_t {
    kNone = 0,
    kA = 1,  // D stored and S exists: read-only taps
    kB = 2,  // depth exists but is not kept, or no S: needs the game's render pass changed
    kC = 3,  // no scene depth
};

// Why a frame has no usable roles, or why the tracker let go.
enum class Miss : uint8_t {
    kNone = 0,
    kNoFrame = 1,       // nothing reached the presented image this interval
    kNoScene = 2,       // the presenting pass samples no full-size image made this frame
    kGameGenerates = 3, // the game runs optical flow itself and offers no S
    kChanged = 4,       // roles differ from the frames before
    kResized = 5,       // swapchain extent changed
    kUntoned = 6,       // the presenting pass samples the scene only before its tonemap
};

struct Roles {
    ImageId depth = 0;
    ImageId scene = 0;
    ImageId velocity = 0;
    ImageId target = 0;  // what U renders into: the presented image, or the image copied to it
    bool direct = false; // U renders into the presented image, which changes every frame
    int ui = -1;         // index of U in Frame::steps
    // Without S: the presenting pass that tonemaps this untoned image into the target, where a
    // split at its first HUD draw would reach the scene before the HUD.
    int split = -1;
    ImageId untoned = 0;
    // That pass tonemaps nothing here: it draws the scene and the HUD itself, and clears depth
    // between them. The split goes at that clear.
    bool clearSplit = false;
    // The temporal resolve: the pass that reads V and its own history, its jittered input (the
    // scene before anti-aliasing) and what it resolves to, all at D's size.
    int resolve = -1;
    ImageId jittered = 0;
    ImageId resolved = 0;
    int geometry = -1;  // the pass that writes most of D, whose draws carry the camera
    bool depthStored = false;
    bool reversed = false;  // D cleared to 0
    bool gameGenerates = false;  // the game runs optical flow itself: its own frame generation
    // S is only what the last pass blits, with nothing behind it to show it has no HUD.
    bool finished = false;
    // S was found behind that blit, which is proof enough: a game may cycle through several.
    bool behindBlit = false;
    Tier tier = Tier::kNone;
    Miss miss = Miss::kNoFrame;
    // The shapes of S and D, hashed; never 0 once classified.
    uint32_t signature = 0;

    bool sameAs(const Roles& o) const {
        return depth == o.depth && scene == o.scene && velocity == o.velocity &&
                direct == o.direct && (direct || target == o.target) &&
                depthStored == o.depthStored && tier == o.tier;
    }
};

Roles Classify(const Frame& frame, const Images& images);

// Engages once S, the frame before the HUD, holds still for kEngageFrames frames. D being kept or
// not does not count: S is read in place either way, and D is offered only by frames that keep it.
// A run is frames of one structure: S and D of the same shapes, whichever images play them.
// Roles without a signature are matched by image instead, two alternating at most. Up to kGraceFrames frames in a row without
// S are ridden out, untapped and not counted, before letting go. Once engaged, changed roles
// engage again at their second frame: a game makes its images again from screen to screen.
// The tracker's state, shared field for field with the Rust tracker that updates it.
struct TrackerState {
    Roles last;
    Roles before;  // the frame before `last` with another S, or just the one before
    bool haveBefore = false;
    int32_t streak = 0;
    int32_t misses = 0;
    bool engaged = false;
    uint32_t width = 0;
    uint32_t height = 0;
    Miss miss = Miss::kNoFrame;
};

class Tracker {
  public:
    static constexpr int kEngageFrames = 8;
    static constexpr int kGraceFrames = 2;

    // Returns whether this frame's `roles` may be tapped.
    bool update(const Roles& roles, uint32_t width, uint32_t height);
    void reset();

    // Engaged, possibly riding out frames without S.
    bool engaged() const { return state_.engaged; }
    const Roles& roles() const { return state_.last; }
    Miss lastMiss() const { return state_.miss; }

  private:
    TrackerState state_;
};

// The transparency cut of one frame (lapis/rust/src/glass.rs), through a tracker kept across
// frames in `tracker`.
inline constexpr uint8_t kGlassInPass = 1;
inline constexpr uint8_t kGlassOwnPass = 2;
inline constexpr uint8_t kGlassRefraction = 3;
struct GlassCut {
    uint8_t kind;
    uint8_t refused;
    bool armed;
    int32_t step;
    uint32_t draw;
    ImageId target;
};
GlassCut FindGlass(const Frame& frame, const Images& images, ImageId depth, uint8_t* tracker);
// Bytes the tracker takes.
size_t GlassTrackerSize();

// Which way the scene reaches the screen, from the frame's own commands: kFlipY/kFlipX bits,
// with `exact` false when a composite pass's viewport only hints at it. -1 when they don't say.
int SceneOrientation(const Frame& frame, const Images& images, const Roles& roles, bool* exact);

}  // namespace framefx::lapis
