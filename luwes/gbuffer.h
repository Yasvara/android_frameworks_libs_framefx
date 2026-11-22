/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// The game's G-buffer as frame generation sees it: per channel confidence, the camera
// reprojection between real frames, and a per-package profile so a known game starts trusted.
// Scoring is lapis/rust/src/gbuffer.rs; this is its bridge, free of any graphics API.
namespace framefx::luwes {

// lapis/rust/src/gbuffer.rs GBufferInput, field for field.
struct GBufferInput {
    uint32_t signature = 0;
    uint32_t sceneWidth = 0, sceneHeight = 0;
    uint32_t depthWidth = 0, depthHeight = 0, depthFormat = 0;
    uint32_t velocityWidth = 0, velocityHeight = 0, velocityFormat = 0;
    float clearDepth = __builtin_nanf("");
    bool hasScene = false;
    bool hasDepth = false;
    bool depthSampled = false;
    bool reversed = false;
    bool hasVelocity = false;
    bool temporal = false;
    bool cameraKnown = false;
    bool cameraMatrices = false;
    bool held = false;
    bool gameGenerates = false;
    bool cameraWorld = false;  // vp holds the camera's position, not camera-relative
    uint8_t pad = 0;
    float sx = 0, sy = 0, c = 0, b = 0;
    float vp[16] = {};
    float jitter[2] = {};
};
static_assert(sizeof(GBufferInput) == 140 && offsetof(GBufferInput, vp) == 68);

enum class DepthRangeKind : uint8_t { kReversed = 0, kForward = 1, kForwardGl = 2 };

struct GBufferOutput {
    float depth = 0, velocity = 0, camera = 0, scene = 0;
    float reproject[16] = {};  // row-major: previous clip = reproject * current clip, unjittered
    float jitter[2] = {};
    float previousJitter[2] = {};
    bool reprojectValid = false;
    bool cut = false;
    DepthRangeKind range = DepthRangeKind::kReversed;
    uint8_t pad = 0;
};
static_assert(sizeof(GBufferOutput) == 100 && offsetof(GBufferOutput, range) == 98);

struct alignas(4) GBufferState {
    uint8_t bytes[108] = {};
};

// Shape of S, D and V hashed: what a game keeps from frame to frame, never 0.
uint32_t GBufferSignature(const GBufferInput& in);

class GBufferFeed {
public:
    // `packages` is debug.framefx.gbuffer: comma-separated packages, plus the words `afme` (feed
    // AFME depth and the reprojection) and `inverse` (hand AFME the inverse matrix).
    void configure(const std::string& packages, const std::string& package);
    bool on() const { return on_; }
    bool feedsAfme() const { return on_ && afme_; }
    bool inverse() const { return inverse_; }

    // One real frame; `gl` when the game's depth is GL clip depth. Loads the profile from
    // `profilePath` once and writes it back once the frame is fully trusted.
    const GBufferOutput& update(const GBufferInput& in, bool gl, const std::string& profilePath,
                                int64_t now);
    const GBufferOutput& last() const { return out_; }
    // A summary once every `periodNs`, or empty.
    std::string report(int64_t now, int64_t periodNs);
    void reset();

private:
    bool on_ = false, afme_ = false, inverse_ = false;
    bool profileRead_ = false, profileWritten_ = false, warmed_ = false;
    GBufferState state_;
    GBufferOutput out_;
    GBufferInput in_;
    int64_t reportedAt_ = 0;
    uint64_t frames_ = 0, cuts_ = 0, reprojected_ = 0, updateNs_ = 0;
};

// Wrappers over the Rust ABI, for tests.
GBufferOutput GBufferUpdate(GBufferState* state, const GBufferInput& in, bool gl);
std::string GBufferProfileLine(const GBufferInput& in);
bool GBufferWarm(GBufferState* state, const std::string& line);
// Row-major 4x4 inverse; false when singular.
bool Invert4x4(const float* m, float* out);

}  // namespace framefx::luwes
