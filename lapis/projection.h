/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "lapis/model.h"

// The game's camera projection P, found among its per-frame constants in host-mapped memory: a
// perspective matrix [sx 0 jx 0; 0 sy jy 0; 0 0 a b; 0 0 w 0] (w = ±1) in either layout. Shadow
// and UI projections are orthographic and fail the w test.
namespace framefx::lapis {

enum class DepthRange : uint8_t {
    kReversed = 0,  // [0, 1], near at 1 (cleared to 0)
    kForward = 1,   // [0, 1], near at 0
    kForwardGl = 2, // [-1, 1] clip depth, stored as [0, 1]
};

struct Projection {
    float sx = 0, sy = 0;  // focal scales
    float jx = 0, jy = 0;  // jitter, in clip units
    float a = 0, b = 0;    // clip depth = c + b / view distance, c = w * a
    float w = 0;
    bool rowMajor = false;

    float fovY() const;    // degrees
    float aspect() const;  // width over height of the view
    // The planes under `range`; far is infinity for an infinite projection. False when the
    // matrix does not describe a frustum in that range.
    bool planes(DepthRange range, float* near, float* far) const;
};

bool AsProjection(const float* m, bool rowMajor, Projection* out);

// The range a game's stored depth is in: a GL game's clip depth stays [-1, 1] whatever draws it,
// ANGLE on Vulkan included, unless it draws reversed, which GL only does over [0, 1].
inline DepthRange RangeOf(bool reversed, bool glClip) {
    return reversed ? DepthRange::kReversed : glClip ? DepthRange::kForwardGl : DepthRange::kForward;
}

// The view distance a stored depth reads as under `p`, as the depth pass computes it.
inline float ViewDistance(const Projection& p, DepthRange range, float stored) {
    const float clip = range == DepthRange::kForwardGl ? 2.0f * stored - 1.0f : stored;
    return p.b / (clip - (p.w < 0 ? -p.a : p.a));
}

// A view-projection P V (V rigid) the scene's draws are made with, as the game hands it to them:
// jitter included. Rows in `m`, the jitter split out as the NDC offset it adds.
struct ViewProjection {
    float m[16] = {};  // row-major: clip = m * world
    float jx = 0, jy = 0;  // NDC
    bool world = false;    // the translation is the camera's, not zero (camera-relative)
};

// The view-projection among `blocks` whose focal scales are `aspect` (the scene's width over
// height) apart, a world-space one before a camera-relative one; false when there is none.
bool FindViewProjection(const std::vector<Constants>& blocks, float aspect, ViewProjection* out);

// The direction toward the light whose shadow map `blocks` were drawn with: the depth axis of an
// orthographic, rotated matrix among them (a sun's cascade), the nearest one; `reversed` when
// that depth is cleared to 0. With `eye` (the scene camera's position, in the space the light's
// matrix reads), only a matrix whose map holds the camera counts: objects' own rotations do not.
// `scales`, when given, gets the lengths of the found matrix's x, y and depth rows, and `matrix`
// the matrix itself, row-major. False when there is none.
bool FindLightDirection(const std::vector<Constants>& blocks, bool reversed, const float* eye,
                        float out[3], float* scales = nullptr, float* matrix = nullptr);

// Every orthographic light matrix in `blocks` (AsLightMatrix's test, the camera not asked), for
// the shadow log: block, float offset, x row length and translation.
struct LightMatrix {
    size_t block;
    size_t offset;
    float scale;
    float tx, ty, tz;
};
std::vector<LightMatrix> LightMatrices(const std::vector<Constants>& blocks);

// Where the camera of a row-major view-projection (clip = vp * world) stands: the point it maps
// to x = y = w = 0. False when it has none.
bool EyeOf(const float vp[16], float out[3]);

// Collects projections across scans and names the one the game draws its scene with: the most
// common frustum that matches the scene depth's aspect and depth range. Jitter is ignored for the
// count; the newest jitter is kept.
class ProjectionVote {
  public:
    // `floats` is 16-byte aligned memory; `aspect` is the scene depth's long over short side.
    void scan(const float* floats, size_t count, float aspect, DepthRange range);
    bool best(Projection* out, float* near, float* far) const;
    void clear() { seen_.clear(); }

  private:
    struct Seen {
        Projection p;
        float near = 0;
        float far = 0;
        uint32_t count = 0;
    };
    // fovY in 0.1°, aspect in 0.001, near and far in 1/1000 of their log2.
    using Key = std::tuple<int, int, int, int>;
    std::map<Key, Seen> seen_;
    DepthRange range_ = DepthRange::kReversed;
};

// The sun across a frame's shadow maps and from frame to frame. Games draw more than one light's:
// a character's own beside the sun's cascades. The sun's maps hold the most of the scene: the
// light whose maps drew the most is it, and a new one takes over only after winning a while, so
// the light does not flip between frames.
class LightVote {
public:
    // One of the sun's shadow maps: the image, the matrix it was drawn with (row-major, world to
    // its clip space), its depth convention and its x row's length (larger nearer the camera).
    struct Cascade {
        ImageId image = 0;
        float matrix[16] = {};
        bool reversed = false;
        bool flipsY = false;  // drawn with a negative viewport height
        float scale = 0;
        uint64_t frame = 0;   // the vote's frame it was last drawn in
        // Where in the image it was drawn, in texture coordinates: x, y its corner, z, w its
        // size. Games keep their cascades side by side in one image, each redrawn in its turn.
        float region[4] = {0, 0, 1, 1};
    };

    // A shadow pass of this frame (FindLightDirection's arguments), how many draws it made, the
    // image it drew into and the region of it its viewport held.
    void add(const std::vector<Constants>& blocks, bool reversed, const float* eye, uint32_t draws,
             ImageId image = 0, bool flipsY = false, const float region[4] = nullptr);
    // Ends the frame. True with the light toward the sun, and its nearest cascade's row lengths,
    // while there is one; it holds through a short while of frames without.
    bool settle(float out[3], float scales[3]);

    // The sun's maps as last drawn, nearest first: a cascade not redrawn this frame still holds
    // what its matrix drew.
    std::vector<Cascade> cascades() const;
    uint64_t frame() const { return frames_; }

    static constexpr int kTakeOver = 15;  // frames a new light must win in a row
    static constexpr int kHold = 120;     // frames the light holds without any

private:
    struct Light {
        float axis[3];
        float scales[3];
        uint64_t draws;
        std::vector<Cascade> maps;
    };
    // By image and region: a cascade stays until its region of the image is drawn again.
    std::map<std::tuple<ImageId, int, int>, Cascade> maps_;
    uint64_t frames_ = 0;
    std::vector<Light> frame_;
    std::vector<Light> far_;  // this frame's cascades that do not hold the camera
    bool has_ = false;
    Light current_ = {};
    Light challenger_ = {};
    int wins_ = 0;
    int missing_ = 0;
};

}  // namespace framefx::lapis
