/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lapis/classifier.h"
#include "lapis/flat.h"
#include "lapis/model.h"

// Hide HUD by not drawing it. The recorder captures each draw (Pass::hudDraws); once per present
// HudLabeler hands the frame to lapis/rust/src/hud, which labels every draw and plans the skips,
// and publishes the skip table the draw hooks look up lock-free for the next frame's draws.
namespace framefx::lapis {

// HudDraw::state, past the kDraw* bits.
constexpr uint16_t kHudDepthCompare = 1u << 4;  // tests depth with a compare other than ALWAYS
constexpr uint16_t kHudIndexed = 1u << 5;
constexpr uint16_t kHudIndirect = 1u << 6;
// HudDraw::textures.
constexpr uint8_t kHudTexSingleChannel = 1u << 0;
constexpr uint8_t kHudTexRenderTarget = 1u << 1;
constexpr uint8_t kHudTexDepth = 1u << 2;
constexpr uint8_t kHudTexUploaded = 1u << 3;
constexpr uint8_t kHudTexUntoned = 1u << 4;
// HudDraw::vertex; the low nibble counts attributes.
constexpr uint8_t kHudVertexAttributes = 0x0f;
constexpr uint8_t kHudVertexColour8 = 1u << 4;
constexpr uint8_t kHudVertex2d = 1u << 5;
constexpr uint8_t kHudVertexNone = 1u << 6;

enum class HudLabel : uint8_t { kScene = 0, kPost = 1, kHud = 2, kWorldUi = 3 };

enum class HudCross : uint8_t {
    kNone = 0,
    kAgrees = 1,   // the screen shows the HUD the labels find, gone while skipping
    kResidue = 2,  // skipping left HUD on screen: no skipping for a while
    kNoHud = 3,    // the labels find HUD draws the screen does not show
};

// lapis/rust/src/hud/mod.rs HudOutput.
struct HudOutput {
    uint32_t labelled[4];  // draws per HudLabel
    float coverage;        // share of this frame's HUD draws the table skips
    float confidence;      // mean confidence of the HUD labels
    float hudShare;        // share of the screen holding still under rotation, unskipped
    float residue;         // and while skipping
    uint32_t entries;
    uint32_t unsampled;
    bool skipping;
    bool applied;  // draws were skipped by the table while this frame was recorded
    bool changed;
    HudCross cross;
    bool dirty;  // the profile has something new to write
};
static_assert(sizeof(HudOutput) == 48);

struct HudEntry {
    uint64_t key;
    uint64_t value;
};

constexpr size_t kHudTableSize = 4096;
constexpr size_t kHudMaxProbe = 16;
constexpr uint64_t kHudIndexMask = 0xffffff;
constexpr uint64_t kHudOpen = kHudIndexMask;
constexpr uint64_t kHudSkip = 1ull << 48;
constexpr uint64_t kHudKnown = 1ull << 49;

inline uint64_t HudMix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// A pipeline in a pass, as the table keys it; never 0. The same as the Rust table_key.
inline uint64_t HudKey(uint64_t pass, uint64_t pipeline) {
    return HudMix(pass ^ HudMix(pipeline ^ 0x9e3779b97f4a7c15ull)) | 1;
}

inline size_t HudSlot(uint64_t key) {
    return static_cast<size_t>(key >> 1) & (kHudTableSize - 1);
}

// A pass's shape: what stays the same from frame to frame while its images change.
uint64_t HudPassKey(const Pass& p, bool swapchainTarget);

struct HudLookup {
    bool skip = false;
    bool known = false;  // the pair's camera is known: no need to sample its constants
};

// The table the draw hooks read while the present thread replaces it: a sequence count around
// each publish, and a lookup that meets one reads as nothing to skip.
class HudTable {
  public:
    HudLookup find(uint64_t key, uint32_t index) const;
    void publish(const HudEntry* entries);
    void clear();

  private:
    std::atomic<uint32_t> seq_{0};
    std::atomic<uint64_t> keys_[kHudTableSize] = {};
    std::atomic<uint64_t> values_[kHudTableSize] = {};
};

// The process's table, and what the draw hooks need around it.
HudTable& HudSkipTable();
bool HudCapturing();
void SetHudCapturing(bool on);
// A draw may have its uniform bytes kept for its camera: a few per frame.
bool HudTakeSample();
void HudNoteSkipped();
// Draws skipped since the last call; resets the sample budget for the next frame.
uint32_t HudFrameDone();

// The presented frame, low resolution, of an earlier present: what the cross-check reads.
struct HudGrid {
    const uint8_t* luma = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t seq = 0;
    bool hasCamera = false;
    float camera[16] = {};  // row-major, clip = M * world
};

struct HudFrameIn {
    const Frame* frame = nullptr;
    const Images* images = nullptr;
    Roles roles;
    bool hideHud = false;
    uint32_t skipped = 0;
    const HudGrid* grid = nullptr;
};

// lapis/rust/src/hud/mod.rs HudPass and HudBlock.
struct HudFfiPass {
    uint64_t key;
    uint32_t drawStart;
    uint32_t drawCount;
};
struct HudFfiBlock {
    uint32_t step;
    uint32_t draw;
    uint32_t start;
    uint32_t len;
};

class HudLabeler {
  public:
    explicit HudLabeler(HudTable* table);
    ~HudLabeler();
    HudLabeler(const HudLabeler&) = delete;
    HudLabeler& operator=(const HudLabeler&) = delete;

    // prepare() copies what it needs of `in` (call it under the lock that guards its images);
    // run() labels, plans and publishes the table when it changed, off that lock.
    void prepare(const HudFrameIn& in);
    const HudOutput& run();
    const HudOutput& update(const HudFrameIn& in) {
        prepare(in);
        return run();
    }
    const HudOutput& last() const { return out_; }
    void reset();

    // The last frame's draws in step order, their labels, confidences, the labels the frame's
    // structure alone gives, and feature rows (featureCount() floats each); and where step i's
    // draws start.
    const std::vector<HudDraw>& draws() const { return draws_; }
    const std::vector<uint8_t>& labels() const { return labels_; }
    const std::vector<float>& confidences() const { return confidence_; }
    const std::vector<uint8_t>& weakLabels() const { return weak_; }
    const std::vector<float>& features() const { return features_; }
    uint32_t firstDraw(size_t step) const;
    // Fills features() at the next update; off, they cost nothing.
    void keepFeatures(bool on) { keepFeatures_ = on; }

    static size_t featureCount();
    static std::string featureName(size_t index);

    std::string profile() const;
    void profileWritten();
    bool warm(const std::string& line);
    // A game's own `hudweights` line, from lab/hud-detect's trainer.
    bool setWeights(const std::string& line);

  private:
    void* state_;
    HudTable* table_;
    HudOutput out_ = {};
    FlatScratch flat_;
    std::vector<HudFfiPass> passes_;  // one per step
    std::vector<HudDraw> draws_;
    std::vector<HudFfiBlock> blocks_;
    std::vector<float> floats_;
    std::vector<HudEntry> entries_;
    std::vector<uint8_t> labels_;
    std::vector<float> confidence_;
    std::vector<uint8_t> weak_;
    std::vector<float> features_;
    std::vector<uint32_t> firstDraw_;
    std::vector<uint8_t> grid_;
    HudGrid gridInfo_;
    bool haveGrid_ = false;
    FfiFrame flatFrame_ = {};
    Roles roles_;
    bool hideHud_ = false;
    uint32_t skipped_ = 0;
    bool keepFeatures_ = false;
};

}  // namespace framefx::lapis
