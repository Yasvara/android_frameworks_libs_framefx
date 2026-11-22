/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lapis/classifier.h"
#include "lapis/flat.h"
#include "lapis/model.h"

// Engine signals for frame generation: the camera (current, previous, next frame's, jitter),
// scene cuts, the post chain and the colour before it, transparency, render scale, stencil IDs,
// the simulation clock, object motion and exposure, each with a confidence. All of it is
// lapis/rust/src/signals; this only carries one frame's records across, once per present.
namespace framefx::lapis {

// lapis/rust/src/signals/mod.rs EngineSignals, field for field. Matrices row-major, clip = M *
// world, without jitter.
struct EngineSignals {
    uint64_t structure;
    double simSeconds;
    int64_t nextSeenBeforePresentNs;
    int64_t nextLeadMedianNs;
    int64_t nextOwnLeadMedianNs;

    float cameraConfidence;
    uint32_t cameraFlags;  // kCamera*
    float viewProjection[16];
    float previousViewProjection[16];
    float turnDegrees, travel, zoom;

    float nextConfidence;
    uint32_t nextValid;
    float nextHitRate, nextEarlyRate;
    float nextViewProjection[16];

    float jitterConfidence;
    uint32_t jitterPattern;  // 0 none, 1 periodic, 2 Halton, 3 random
    uint32_t jitterPeriod;
    uint32_t jitterNextValid;
    float jitterNdc[2];
    float jitterPixels[2];
    float nextJitterPixels[2];

    uint32_t cut;
    uint32_t cutReasons;
    uint32_t sceneState;  // 0 unknown, 1 3D scene, 2 flat (menu, loading, video)
    uint32_t framesSinceCut;
    float cutConfidence;

    uint32_t postEffects;
    ImageId prePostImage;
    int32_t prePostStep;
    uint32_t prePostFlags;
    int32_t tonemapStep;
    uint32_t sceneDraws;
    uint32_t transparentDraws;
    int32_t transparentStep;
    uint32_t transparentFrom;
    float renderScale[2];
    float drawn[2];
    uint32_t stencilIds;
    uint32_t stencilWrites;

    float clockConfidence;
    uint32_t clockFlags;
    float simStep;
    float presentInterval;

    float objectConfidence;
    uint32_t objectsTransformed;
    uint32_t objectsMatched;
    uint32_t objectsMoving;
    float objects[16][4];  // NDC x, y, motion dx, dy

    float exposureConfidence;
    float exposureGain;
};
#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(EngineSignals) == 688 && offsetof(EngineSignals, cameraConfidence) == 40 &&
              offsetof(EngineSignals, nextViewProjection) == 204 &&
              offsetof(EngineSignals, objects) == 420);
#endif

constexpr uint32_t kCameraFound = 1u << 0;
constexpr uint32_t kCameraWorld = 1u << 1;
constexpr uint32_t kCameraJitterExact = 1u << 2;
constexpr uint32_t kCameraPrevious = 1u << 3;
constexpr uint32_t kCameraPreviousFromGame = 1u << 4;
constexpr uint32_t kClockFound = 1u << 0;
constexpr uint32_t kClockPaused = 1u << 1;
constexpr uint32_t kPrePostUntoned = 1u << 0;
constexpr uint32_t kPrePostOverwritten = 1u << 1;
constexpr uint32_t kPostTemporal = 1u << 0;
constexpr uint32_t kPostBloom = 1u << 1;
constexpr uint32_t kPostDepthAware = 1u << 2;
constexpr uint32_t kPostMotionBlur = 1u << 3;
constexpr uint32_t kPostTonemap = 1u << 4;
constexpr uint32_t kPostScreen = 1u << 5;
constexpr uint32_t kPostUpscale = 1u << 6;

// An address to read 16 floats from at each submit until the next present.
struct SignalWatch {
    uint64_t memory;
    uint64_t offset;
    uint32_t rowMajor;
    uint32_t pad;
};

struct SignalSample {
    uint64_t memory;
    uint64_t offset;
    int64_t timeNs;
    uint32_t rowMajor;
    float floats[16];
};

// One frame's records, gathered by the recorder.
struct SignalFrame {
    const Frame* frame = nullptr;
    const Images* images = nullptr;
    Roles roles;
    const std::vector<Constants>* cameraBlocks = nullptr;   // the scene pass's first draw
    const std::vector<Constants>* tonemapBlocks = nullptr;  // the tonemap pass's first draw
    const std::vector<SignalSample>* samples = nullptr;
    int64_t presentNs = 0;
    uint32_t sceneWidth = 0, sceneHeight = 0;
};

class SignalFeed {
  public:
    SignalFeed();
    ~SignalFeed();
    SignalFeed(const SignalFeed&) = delete;
    SignalFeed& operator=(const SignalFeed&) = delete;

    // prepare() copies what it needs of `f` (call it under the lock that guards f's images); run()
    // decides, off that lock. update() is both.
    void prepare(const SignalFrame& f);
    const EngineSignals& run();
    const EngineSignals& update(const SignalFrame& f) {
        prepare(f);
        return run();
    }
    const EngineSignals& last() const { return out_; }
    // Where the next frame's camera may land; read them at each submit.
    const std::vector<SignalWatch>& watch() const { return watch_; }
    void reset();
    // The per-package profile line, empty until something is learned; warm() takes one back.
    std::string profile() const;
    bool warm(const std::string& line);

  private:
    void* state_;
    EngineSignals out_;
    std::vector<SignalWatch> watch_;
    FlatScratch flat_;
    std::vector<uint8_t> extras_;
    std::vector<float> floats_;
    std::vector<uint8_t> blocks_;  // camera blocks then tonemap blocks
    std::vector<SignalSample> samples_;
    uint32_t cameraBlocks_ = 0, tonemapBlocks_ = 0, steps_ = 0;
    FfiFrame flat_frame_ = {};
    Roles roles_;
    int64_t presentNs_ = 0;
    uint32_t sceneWidth_ = 0, sceneHeight_ = 0;
};

}  // namespace framefx::lapis
