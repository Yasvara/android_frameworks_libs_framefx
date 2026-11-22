/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "lapis/hud.h"

#include <cstring>

// Carries a recorded frame to lapis/rust/src/hud in one call, and its skip table back to the draw
// hooks; no logic of its own beyond the lookup.
namespace framefx::lapis {
namespace {

constexpr uint32_t kInputHideHud = 1u << 0;
constexpr uint32_t kInputGrid = 1u << 1;
constexpr uint32_t kInputGridCamera = 1u << 2;
// Draws per frame whose uniform bytes are kept to find their camera.
constexpr int kSamplesPerFrame = 32;

struct FfiHudInput {
    FfiFrame frame;
    const HudFfiPass* passes;
    uint32_t passCount;
    const HudDraw* draws;
    uint32_t drawCount;
    const HudFfiBlock* blocks;
    uint32_t blockCount;
    const float* floats;
    uint32_t floatCount;
    const uint8_t* grid;
    uint32_t gridWidth;
    uint32_t gridHeight;
    uint32_t gridSeq;
    Roles roles;
    uint32_t flags;
    uint32_t skipped;
    float gridViewProjection[16];
};

static_assert(sizeof(HudFfiPass) == 16 && sizeof(HudFfiBlock) == 16 && sizeof(HudEntry) == 16);
#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(FfiHudInput) == 296 && offsetof(FfiHudInput, roles) == 156 &&
              offsetof(FfiHudInput, gridViewProjection) == 228);
#endif

extern "C" {
void* framefx_hud_new();
void framefx_hud_free(void* state);
void framefx_hud_reset(void* state);
void framefx_hud_update(void* state, const FfiHudInput* input, HudOutput* out);
size_t framefx_hud_table(const void* state, HudEntry* out, size_t capacity);
size_t framefx_hud_labels(const void* state, uint8_t* labels, float* confidence, uint8_t* weak,
                          size_t capacity);
size_t framefx_hud_features(const void* state, float* out, size_t capacity);
size_t framefx_hud_feature_count();
size_t framefx_hud_feature_name(size_t index, uint8_t* out, size_t capacity);
size_t framefx_hud_profile(const void* state, uint8_t* out, size_t capacity);
void framefx_hud_profile_written(void* state);
bool framefx_hud_warm(void* state, const uint8_t* line, size_t len);
bool framefx_hud_set_weights(void* state, const uint8_t* line, size_t len);
}

std::atomic<bool> gCapturing{false};
std::atomic<int> gSamples{kSamplesPerFrame};
std::atomic<uint32_t> gSkipped{0};

}  // namespace

uint64_t HudPassKey(const Pass& p, bool swapchainTarget) {
    uint64_t k = HudMix(p.renderPass ^ (p.dynamic ? 0x5bd1e995ull : 0));
    k = HudMix(k ^ (static_cast<uint64_t>(p.width) << 32 | p.height));
    k = HudMix(k ^ (static_cast<uint64_t>(p.colors.size()) << 40 | p.subpasses << 8 |
                    (swapchainTarget ? 1 : 0)));
    const uint64_t colour = p.colors.empty() ? 0 : static_cast<uint64_t>(p.colors[0].format);
    const uint64_t depth = p.hasDepth ? static_cast<uint64_t>(p.depth.format) : 0;
    k = HudMix(k ^ (colour << 32) ^ depth);
    // A pass that starts its target over and one that draws on: dynamic rendering tells them
    // apart only here.
    const uint64_t colourLoad = p.colors.empty() ? 0 : static_cast<uint64_t>(p.colors[0].load);
    const uint64_t depthLoad = p.hasDepth ? static_cast<uint64_t>(p.depth.load) : 0;
    return HudMix(k ^ (colourLoad << 8 | depthLoad)) | 1;
}

HudLookup HudTable::find(uint64_t key, uint32_t index) const {
    const uint32_t seq = seq_.load(std::memory_order_acquire);
    if (seq & 1) return {};
    HudLookup found;
    for (size_t probe = 0; probe < kHudMaxProbe; probe++) {
        const size_t at = (HudSlot(key) + probe) & (kHudTableSize - 1);
        const uint64_t k = keys_[at].load(std::memory_order_relaxed);
        if (k == 0) break;
        if (k != key) continue;
        const uint64_t v = values_[at].load(std::memory_order_relaxed);
        const uint64_t from = v & kHudIndexMask;
        const uint64_t to = (v >> 24) & kHudIndexMask;
        found.skip = (v & kHudSkip) && index >= from && (to == kHudOpen || index < to);
        found.known = (v & kHudKnown) != 0;
        break;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (seq_.load(std::memory_order_relaxed) != seq) return {};
    return found;
}

void HudTable::publish(const HudEntry* entries) {
    const uint32_t seq = seq_.load(std::memory_order_relaxed);
    seq_.store(seq + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    for (size_t i = 0; i < kHudTableSize; i++) {
        keys_[i].store(entries == nullptr ? 0 : entries[i].key, std::memory_order_relaxed);
        values_[i].store(entries == nullptr ? 0 : entries[i].value, std::memory_order_relaxed);
    }
    seq_.store(seq + 2, std::memory_order_release);
}

void HudTable::clear() {
    publish(nullptr);
}

HudTable& HudSkipTable() {
    static HudTable table;
    return table;
}

bool HudCapturing() {
    return gCapturing.load(std::memory_order_relaxed);
}

void SetHudCapturing(bool on) {
    gCapturing.store(on, std::memory_order_relaxed);
}

bool HudTakeSample() {
    return gSamples.fetch_sub(1, std::memory_order_relaxed) > 0;
}

void HudNoteSkipped() {
    gSkipped.fetch_add(1, std::memory_order_relaxed);
}

uint32_t HudFrameDone() {
    gSamples.store(kSamplesPerFrame, std::memory_order_relaxed);
    return gSkipped.exchange(0, std::memory_order_relaxed);
}

HudLabeler::HudLabeler(HudTable* table) : state_(framefx_hud_new()), table_(table) {
    entries_.resize(kHudTableSize);
}

HudLabeler::~HudLabeler() {
    framefx_hud_free(state_);
}

void HudLabeler::reset() {
    framefx_hud_reset(state_);
    out_ = {};
    if (table_ != nullptr) table_->clear();
}

void HudLabeler::prepare(const HudFrameIn& in) {
    const Frame& frame = *in.frame;
    passes_.clear();
    draws_.clear();
    blocks_.clear();
    floats_.clear();
    firstDraw_.clear();
    for (size_t i = 0; i < frame.steps.size(); i++) {
        const Step& s = frame.steps[i];
        HudFfiPass p = {0, static_cast<uint32_t>(draws_.size()), 0};
        firstDraw_.push_back(p.drawStart);
        if (s.isPass) {
            p.key = s.pass.hudKey;
            p.drawCount = static_cast<uint32_t>(s.pass.hudDraws.size());
            draws_.insert(draws_.end(), s.pass.hudDraws.begin(), s.pass.hudDraws.end());
            for (const Constants& c : s.pass.hudConstants) {
                blocks_.push_back({static_cast<uint32_t>(i), c.draw,
                                   static_cast<uint32_t>(floats_.size()),
                                   static_cast<uint32_t>(c.values.size())});
                floats_.insert(floats_.end(), c.values.begin(), c.values.end());
            }
        }
        passes_.push_back(p);
    }
    flatFrame_ = Flatten(frame, *in.images, &flat_);
    roles_ = in.roles;
    hideHud_ = in.hideHud;
    skipped_ = in.skipped;
    haveGrid_ = in.grid != nullptr && in.grid->luma != nullptr;
    if (haveGrid_) {
        gridInfo_ = *in.grid;
        grid_.assign(in.grid->luma, in.grid->luma + size_t{in.grid->width} * in.grid->height);
        gridInfo_.luma = grid_.data();
    }
}

const HudOutput& HudLabeler::run() {
    FfiHudInput input = {};
    input.frame = flatFrame_;
    input.passes = passes_.data();
    input.passCount = static_cast<uint32_t>(passes_.size());
    input.draws = draws_.data();
    input.drawCount = static_cast<uint32_t>(draws_.size());
    input.blocks = blocks_.data();
    input.blockCount = static_cast<uint32_t>(blocks_.size());
    input.floats = floats_.data();
    input.floatCount = static_cast<uint32_t>(floats_.size());
    input.roles = roles_;
    input.flags = hideHud_ ? kInputHideHud : 0;
    input.skipped = skipped_;
    if (haveGrid_) {
        input.flags |= kInputGrid;
        input.grid = gridInfo_.luma;
        input.gridWidth = gridInfo_.width;
        input.gridHeight = gridInfo_.height;
        input.gridSeq = gridInfo_.seq;
        if (gridInfo_.hasCamera) {
            input.flags |= kInputGridCamera;
            memcpy(input.gridViewProjection, gridInfo_.camera, sizeof(input.gridViewProjection));
        }
    }
    framefx_hud_update(state_, &input, &out_);
    if (out_.changed && table_ != nullptr &&
        framefx_hud_table(state_, entries_.data(), entries_.size()) == kHudTableSize) {
        table_->publish(entries_.data());
    }
    const size_t n = draws_.size();
    labels_.resize(n);
    confidence_.resize(n);
    weak_.resize(n);
    framefx_hud_labels(state_, labels_.data(), confidence_.data(), weak_.data(), n);
    if (keepFeatures_) {
        features_.resize(n * featureCount());
        framefx_hud_features(state_, features_.data(), features_.size());
    } else {
        features_.clear();
    }
    return out_;
}

uint32_t HudLabeler::firstDraw(size_t step) const {
    return step < firstDraw_.size() ? firstDraw_[step] : static_cast<uint32_t>(draws_.size());
}

size_t HudLabeler::featureCount() {
    return framefx_hud_feature_count();
}

std::string HudLabeler::featureName(size_t index) {
    char name[64] = {};
    const size_t n =
            framefx_hud_feature_name(index, reinterpret_cast<uint8_t*>(name), sizeof(name));
    return std::string(name, n);
}

std::string HudLabeler::profile() const {
    std::string line(64 * 1024, '\0');
    const size_t n =
            framefx_hud_profile(state_, reinterpret_cast<uint8_t*>(line.data()), line.size());
    line.resize(n);
    return line;
}

void HudLabeler::profileWritten() {
    framefx_hud_profile_written(state_);
}

bool HudLabeler::warm(const std::string& line) {
    return framefx_hud_warm(state_, reinterpret_cast<const uint8_t*>(line.data()), line.size());
}

bool HudLabeler::setWeights(const std::string& line) {
    return framefx_hud_set_weights(state_, reinterpret_cast<const uint8_t*>(line.data()),
                                   line.size());
}

}  // namespace framefx::lapis
