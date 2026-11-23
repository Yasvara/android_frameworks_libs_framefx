/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "lapis/classifier.h"
#include "lapis/flat.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

// The classifier and tracker live in Rust (lapis/rust). This flattens a recorded frame into the
// arrays it reads.
namespace framefx::lapis {
namespace {

static_assert(std::is_standard_layout_v<Roles> && sizeof(Roles) == 64 && offsetof(Roles, miss) == 58);
static_assert(std::is_standard_layout_v<TrackerState> && sizeof(TrackerState) == 156 &&
              offsetof(TrackerState, miss) == 152);
static_assert(std::is_standard_layout_v<GlassCut> && sizeof(GlassCut) == 16 &&
              offsetof(GlassCut, held) == 3 && offsetof(GlassCut, step) == 4 &&
              offsetof(GlassCut, target) == 12);

extern "C" {
void framefx_lapis_classify(const FfiFrame* frame, Roles* out);
int32_t framefx_lapis_orientation(const FfiFrame* frame, uint32_t from, int32_t after);
struct FfiGlassPass {
    uint32_t glassFrom;
    uint32_t lastDepthWrite;
};
void framefx_lapis_glass(const FfiFrame* frame, const FfiGlassPass* passes, uint32_t depth,
                         uint8_t* tracker, framefx::lapis::GlassCut* out);
size_t framefx_lapis_glass_tracker_size();
bool framefx_lapis_tracker_update(TrackerState* tracker, const Roles* roles, uint32_t width,
                                  uint32_t height, int32_t engage, int32_t grace);
void framefx_lapis_tracker_reset(TrackerState* tracker);
bool framefx_lapis_is_depth_format(uint32_t format);
bool framefx_lapis_is_depth_like_format(uint32_t format);
bool framefx_lapis_is_untoned(uint32_t format);
}

}  // namespace

size_t GlassTrackerSize() {
    return framefx_lapis_glass_tracker_size();
}

GlassCut FindGlass(const Frame& frame, const Images& images, ImageId depth, uint8_t* tracker) {
    thread_local FlatScratch scratch;
    thread_local std::vector<FfiGlassPass> passes;
    const FfiFrame flat = Flatten(frame, images, &scratch);
    passes.resize(frame.steps.size());
    for (size_t k = 0; k < frame.steps.size(); k++) {
        const Step& s = frame.steps[k];
        passes[k] = s.isPass ? FfiGlassPass{s.pass.glassFrom, s.pass.lastDepthWrite}
                             : FfiGlassPass{UINT32_MAX, UINT32_MAX};
    }
    GlassCut out = {};
    out.step = -1;
    framefx_lapis_glass(&flat, passes.data(), depth, tracker, &out);
    return out;
}

int SceneOrientation(const Frame& frame, const Images& images, const Roles& roles, bool* exact) {
    // Where the scene was copied out of an image the frame goes on to show, the chain starts at
    // that copy; otherwise after the last pass that drew the scene.
    ImageId from = roles.scene;
    int after = -1;
    for (int i = static_cast<int>(frame.steps.size()) - 1; i >= 0; i--) {
        const Step& s = frame.steps[i];
        if (!s.isPass && s.op.transfers() && s.op.dst == roles.scene) {
            from = s.op.src;
            after = i;
            break;
        }
        if (s.isPass && s.pass.writes(roles.scene)) {
            after = i;
            break;
        }
    }
    thread_local FlatScratch scratch;
    const FfiFrame flat = Flatten(frame, images, &scratch);
    const int32_t r = framefx_lapis_orientation(&flat, from, after);
    if (r < 0) return -1;
    *exact = (r & 0x100) != 0;
    return r & (kFlipY | kFlipX);
}

Roles Classify(const Frame& frame, const Images& images) {
    thread_local FlatScratch scratch;
    const FfiFrame flat = Flatten(frame, images, &scratch);
    Roles r;
    framefx_lapis_classify(&flat, &r);
    return r;
}

FfiFrame Flatten(const Frame& frame, const Images& images, FlatScratch* s) {
    s->steps.clear();
    s->targets.clear();
    s->reads.clear();
    s->ids.clear();
    s->images.clear();
    s->ids.push_back(frame.presented);
    const auto flat = [](const Target& t) {
        return FfiTarget{t.image, static_cast<uint32_t>(t.format), static_cast<uint32_t>(t.load),
                         static_cast<uint32_t>(t.store), static_cast<uint32_t>(t.samples)};
    };
    for (const Step& step : frame.steps) {
        FfiStep f{};
        f.isPass = step.isPass;
        if (step.isPass) {
            const Pass& p = step.pass;
            f.hasDepth = p.hasDepth;
            f.flips = p.viewportHeight < 0 ? kFlipY : 0;
            f.depth = flat(p.depth);
            f.clearDepth = p.clearDepth;
            f.draws = p.draws;
            f.depthWrites = p.depthWrites;
            f.depthClears = p.depthClears;
            f.lastClearDraw = p.lastClearDraw;
            f.colors = {static_cast<uint32_t>(s->targets.size()),
                        static_cast<uint32_t>(p.colors.size())};
            for (const Target& t : p.colors) {
                s->targets.push_back(flat(t));
                s->ids.push_back(t.image);
            }
            f.reads = {static_cast<uint32_t>(s->reads.size()), static_cast<uint32_t>(p.reads.size())};
            s->reads.insert(s->reads.end(), p.reads.begin(), p.reads.end());
            s->ids.insert(s->ids.end(), p.reads.begin(), p.reads.end());
            if (p.hasDepth) s->ids.push_back(p.depth.image);
        } else {
            f.opKind = static_cast<uint8_t>(step.op.kind);
            f.flips = step.op.flips;
            f.clearDepth = step.op.depth;
            f.src = step.op.src;
            f.dst = step.op.dst;
            s->ids.push_back(step.op.src);
            s->ids.push_back(step.op.dst);
        }
        s->steps.push_back(f);
    }
    std::sort(s->ids.begin(), s->ids.end());
    s->ids.erase(std::unique(s->ids.begin(), s->ids.end()), s->ids.end());
    for (ImageId id : s->ids) {
        const auto i = images.find(id);
        if (i == images.end()) continue;
        s->images.push_back({id, static_cast<uint32_t>(i->second.format), i->second.width,
                             i->second.height});
    }
    return {s->steps.data(),   static_cast<uint32_t>(s->steps.size()),
            s->targets.data(), static_cast<uint32_t>(s->targets.size()),
            s->reads.data(),   static_cast<uint32_t>(s->reads.size()),
            s->images.data(),  static_cast<uint32_t>(s->images.size()),
            frame.presented,   frame.width,
            frame.height};
}

bool Tracker::update(const Roles& roles, uint32_t width, uint32_t height) {
    return framefx_lapis_tracker_update(&state_, &roles, width, height, kEngageFrames,
                                        kGraceFrames);
}

void Tracker::reset() {
    framefx_lapis_tracker_reset(&state_);
}

bool IsDepthFormat(VkFormat f) {
    return framefx_lapis_is_depth_format(static_cast<uint32_t>(f));
}

bool IsDepthLikeFormat(VkFormat f) {
    return framefx_lapis_is_depth_like_format(static_cast<uint32_t>(f));
}

bool IsUntoned(VkFormat f) {
    return framefx_lapis_is_untoned(static_cast<uint32_t>(f));
}

}  // namespace framefx::lapis
