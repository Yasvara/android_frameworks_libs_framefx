/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "lapis/signals.h"

#include <cstring>

// Carries a recorded frame to lapis/rust/src/signals in one call; no logic of its own.
namespace framefx::lapis {
namespace {

struct FfiPassExtra {
    uint32_t drawsByState[Pass::kDrawStates];
    uint32_t firstByState[Pass::kDrawStates];
    uint32_t stencilStore;
    float viewport[4];
};

struct FfiBlock {
    uint64_t memory;
    uint64_t offset;
    uint32_t set;
    uint32_t binding;
    uint32_t start;
    uint32_t len;
};

struct FfiDraw {
    uint64_t key;
    uint32_t start;
    uint32_t len;
};

struct FfiInput {
    FfiFrame frame;
    const FfiPassExtra* extras;
    uint32_t extraCount;
    Roles roles;
    const FfiBlock* cameraBlocks;
    uint32_t cameraBlockCount;
    const FfiBlock* tonemapBlocks;
    uint32_t tonemapBlockCount;
    const float* floats;
    uint32_t floatCount;
    const FfiDraw* draws;
    uint32_t drawCount;
    const SignalSample* samples;
    uint32_t sampleCount;
    int64_t presentNs;
    uint32_t sceneWidth;
    uint32_t sceneHeight;
};

static_assert(sizeof(FfiPassExtra) == 148 && sizeof(FfiBlock) == 32 && sizeof(FfiDraw) == 16);
#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(FfiFrame) == 72 && sizeof(SignalSample) == 96 && sizeof(SignalWatch) == 24);
static_assert(sizeof(FfiInput) == 248 && offsetof(FfiInput, roles) == 84 &&
              offsetof(FfiInput, presentNs) == 232);
#endif

extern "C" {
void* framefx_signals_new();
void framefx_signals_free(void* state);
void framefx_signals_reset(void* state);
void framefx_signals_update(void* state, const FfiInput* input, EngineSignals* out);
size_t framefx_signals_watch(const void* state, SignalWatch* out, size_t capacity);
size_t framefx_signals_profile(const void* state, uint8_t* out, size_t capacity);
bool framefx_signals_warm(void* state, const uint8_t* line, size_t len);
}

constexpr size_t kMaxWatch = 8;

void appendBlocks(const std::vector<Constants>* in, std::vector<FfiBlock>* blocks,
                  std::vector<float>* floats) {
    if (in == nullptr) return;
    for (const Constants& c : *in) {
        blocks->push_back({c.memory, c.offset, c.set, c.binding,
                           static_cast<uint32_t>(floats->size()),
                           static_cast<uint32_t>(c.values.size())});
        floats->insert(floats->end(), c.values.begin(), c.values.end());
    }
}

}  // namespace

SignalFeed::SignalFeed() : state_(framefx_signals_new()), out_{} {
    watch_.reserve(kMaxWatch);
}

SignalFeed::~SignalFeed() {
    framefx_signals_free(state_);
}

void SignalFeed::prepare(const SignalFrame& f) {
    static const Images kNone;
    flat_frame_ = Flatten(*f.frame, f.images != nullptr ? *f.images : kNone, &flat_);
    steps_ = static_cast<uint32_t>(f.frame->steps.size());
    extras_.resize(steps_ * sizeof(FfiPassExtra));
    auto* extras = reinterpret_cast<FfiPassExtra*>(extras_.data());
    for (size_t k = 0; k < steps_; k++) {
        FfiPassExtra& e = extras[k];
        memset(&e, 0, sizeof(e));
        const Step& s = f.frame->steps[k];
        if (!s.isPass) continue;
        memcpy(e.drawsByState, s.pass.drawsByState, sizeof(e.drawsByState));
        memcpy(e.firstByState, s.pass.firstByState, sizeof(e.firstByState));
        e.stencilStore = static_cast<uint32_t>(s.pass.stencilStore);
        e.viewport[0] = s.pass.viewportX;
        e.viewport[1] = s.pass.viewportY;
        e.viewport[2] = s.pass.viewportWidth;
        e.viewport[3] = s.pass.viewportHeight;
    }
    thread_local std::vector<FfiBlock> blocks;
    blocks.clear();
    floats_.clear();
    appendBlocks(f.cameraBlocks, &blocks, &floats_);
    cameraBlocks_ = static_cast<uint32_t>(blocks.size());
    appendBlocks(f.tonemapBlocks, &blocks, &floats_);
    tonemapBlocks_ = static_cast<uint32_t>(blocks.size()) - cameraBlocks_;
    blocks_.assign(reinterpret_cast<const uint8_t*>(blocks.data()),
                   reinterpret_cast<const uint8_t*>(blocks.data() + blocks.size()));
    samples_.clear();
    if (f.samples != nullptr) samples_.insert(samples_.end(), f.samples->begin(), f.samples->end());
    roles_ = f.roles;
    presentNs_ = f.presentNs;
    sceneWidth_ = f.sceneWidth;
    sceneHeight_ = f.sceneHeight;
}

const EngineSignals& SignalFeed::run() {
    const auto* blocks = reinterpret_cast<const FfiBlock*>(blocks_.data());
    const FfiInput in{flat_frame_,
                      reinterpret_cast<const FfiPassExtra*>(extras_.data()),
                      steps_,
                      roles_,
                      blocks,
                      cameraBlocks_,
                      blocks + cameraBlocks_,
                      tonemapBlocks_,
                      floats_.data(),
                      static_cast<uint32_t>(floats_.size()),
                      nullptr,
                      0,
                      samples_.data(),
                      static_cast<uint32_t>(samples_.size()),
                      presentNs_,
                      sceneWidth_,
                      sceneHeight_};
    framefx_signals_update(state_, &in, &out_);
    watch_.resize(kMaxWatch);
    watch_.resize(framefx_signals_watch(state_, watch_.data(), kMaxWatch));
    return out_;
}

void SignalFeed::reset() {
    framefx_signals_reset(state_);
    out_ = {};
    watch_.clear();
}

std::string SignalFeed::profile() const {
    uint8_t line[128];
    const size_t n = framefx_signals_profile(state_, line, sizeof(line));
    return std::string(reinterpret_cast<const char*>(line), n);
}

bool SignalFeed::warm(const std::string& line) {
    return framefx_signals_warm(state_, reinterpret_cast<const uint8_t*>(line.data()), line.size());
}

}  // namespace framefx::lapis
