/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "tests/parity/kelir_legacy/plan.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace framefx::kelir_legacy {
namespace {

Step decode(const Node& n) {
    Step s;
    s.kind = n.kind;
    s.intensity = n.strength();
    for (int i = 0; i < Info(n.kind).paramCount; i++) s.v[i] = n.value(i);
    return s;
}

// A node that cannot change a pixel; dropped, so it costs nothing.
bool inert(const Node& n, const Step& s) {
    if (!n.enabled || n.intensity == 0) return true;
    switch (Info(n.kind).band) {
        case Band::kColour:
            return IsIdentityColour(s);
        case Band::kDetail:
            if (n.kind == kDetails) {
                return s.v[p::kClarity] == 0.0f && s.v[p::kHdrToning] == 0.0f &&
                        s.v[p::kBloom] == 0.0f;
            }
            return n.kind == kTiltShift && s.v[p::kBlur] == 0.0f;
        case Band::kScreen:
            return false;
        case Band::kScene:
            return (n.kind == kDepthFog && s.v[p::kDensity] == 0.0f) ||
                    (n.kind == kDepthOfField && s.v[p::kDofBlur] == 0.0f) ||
                    (n.kind == kGlobalIllumination && s.v[p::kBounce] == 0.0f);
    }
    return false;
}

bool sameColourBand(const Plan& a, const Plan& b) {
    return a.colourCount == b.colourCount &&
            std::equal(a.colour, a.colour + a.colourCount, b.colour);
}

// Kind codes per band, 1-based inside the band, 0 for an empty slot.
uint64_t detailCode(uint8_t kind) { return kind - kSharpen + 1u; }
uint64_t screenCode(uint8_t kind) { return kind - kVignette + 1u; }

}  // namespace

bool Plan::has(uint8_t kind) const {
    const auto is = [kind](const Step& s) { return s.kind == kind; };
    return std::any_of(detail, detail + detailCount, is) ||
            std::any_of(screen, screen + screenCount, is);
}

const Step* Plan::sceneStep(uint8_t kind) const {
    for (int i = 0; i < sceneCount; i++) {
        if (scene[i].kind == kind) return &scene[i];
    }
    return nullptr;
}

bool Plan::needsPyramid() const {
    return has(kDetails) || has(kTiltShift) || has(kPainterly) || has(kWatercolor);
}

bool Plan::needsSource() const {
    return has(kSharpen) || has(kWatercolor) || has(kRetro) || needsPyramid();
}

std::shared_ptr<const Plan> Compile(const Stack& stack, const Plan* previous) {
    auto plan = std::make_shared<Plan>();
    plan->revision = stack.revision;
    plan->peek = stack.peek;
    plan->split = stack.split;
    plan->inspect = stack.inspect;
    plan->splitX = stack.splitX / 65535.0f;
    for (int i = 0; i < stack.count; i++) {
        const Node& n = stack.nodes[i];
        const Step s = decode(n);
        if (inert(n, s)) continue;
        switch (Info(n.kind).band) {
            case Band::kDetail:
                plan->detail[plan->detailCount++] = s;
                break;
            case Band::kColour:
                plan->colourNode[plan->colourCount] = i;
                plan->colour[plan->colourCount++] = s;
                break;
            case Band::kScreen:
                plan->screen[plan->screenCount++] = s;
                break;
            case Band::kScene:
                plan->scene[plan->sceneCount++] = s;
                break;
        }
    }
    if (plan->colourCount > 0) {
        plan->luts = previous != nullptr && previous->luts != nullptr &&
                        sameColourBand(*plan, *previous)
                ? previous->luts
                : std::make_shared<const Luts>(plan->colour, plan->colourCount);
    }
    return plan;
}

// Bits: 0-1 format class, 2 fetch, 3 MSAA, 4-5 role, 6 LUT, 7 linear reads, 8-19 detail kinds
// (4 bits each), 20 Sharpen Plus, 24-35 screen kinds (3 bits each). Exact, so never colliding.
uint64_t Signature(const Plan& plan, const Target& t) {
    uint64_t s = static_cast<uint64_t>(t.format) | (t.fetch ? 1u : 0u) << 2 |
            (t.msaa ? 1u : 0u) << 3 | static_cast<uint64_t>(t.role) << 4 |
            (t.readsLinear ? 1u : 0u) << 7;
    if (t.role != Role::kStageB) {
        s |= (plan.colourCount > 0 ? 1ull : 0ull) << 6;
        for (int i = 0; i < plan.detailCount; i++) {
            s |= detailCode(plan.detail[i].kind) << (8 + 4 * i);
            if (plan.detail[i].kind == kSharpen &&
                static_cast<int>(plan.detail[i].v[p::kQuality]) == kPlus) {
                s |= 1ull << 20;
            }
        }
    }
    // Stage A leaves the screen band to the presents, so a screen edit never rebuilds it.
    if (t.role != Role::kStageA) {
        for (int i = 0; i < plan.screenCount; i++) {
            s |= screenCode(plan.screen[i].kind) << (24 + 3 * i);
        }
    }
    return s;
}

bool Program::has(int kind) const {
    return std::find(detail, detail + kMaxDetail, kind) != detail + kMaxDetail ||
            std::find(screen, screen + kMaxScreen, kind) != screen + kMaxScreen;
}

Program Unpack(uint64_t s) {
    Program p;
    p.target.format = static_cast<FormatClass>(s & 3u);
    p.target.fetch = (s >> 2 & 1u) != 0;
    p.target.msaa = (s >> 3 & 1u) != 0;
    p.target.role = static_cast<Role>(s >> 4 & 3u);
    p.lut = (s >> 6 & 1u) != 0;
    p.target.readsLinear = (s >> 7 & 1u) != 0;
    for (int i = 0; i < kMaxDetail; i++) {
        const int code = static_cast<int>(s >> (8 + 4 * i) & 0xfu);
        p.detail[i] = code == 0 ? -1 : code - 1 + kSharpen;
    }
    p.sharpenPlus = (s >> 20 & 1u) != 0;
    for (int i = 0; i < kMaxScreen; i++) {
        const int code = static_cast<int>(s >> (24 + 3 * i) & 0x7u);
        p.screen[i] = code == 0 ? -1 : code - 1 + kVignette;
    }
    return p;
}

void PackPlan(const Plan& plan, float out[kPlanVec4s * 4]) {
    memset(out, 0, sizeof(float) * kPlanVec4s * 4);
    for (int i = 0; i < plan.detailCount; i++) {
        memcpy(out + 8 * i, plan.detail[i].v, sizeof(float) * kMaxParams);
        out[14 * 4 + i] = plan.detail[i].intensity;
    }
    for (int i = 0; i < plan.screenCount; i++) {
        memcpy(out + 6 * 4 + 8 * i, plan.screen[i].v, sizeof(float) * kMaxParams);
        out[15 * 4 + i] = plan.screen[i].intensity;
    }
    out[14 * 4 + 3] = plan.luts != nullptr && plan.luts->linear() ? 1.0f : 0.0f;
}

void PackScene(const Plan& plan, float out[kSceneVec4s * 4]) {
    memset(out, 0, sizeof(float) * kSceneVec4s * 4);
    if (const Step* ao = plan.sceneStep(kAmbientOcclusion)) {
        out[0] = ao->v[p::kRadius];
        out[1] = ao->intensity;
    }
    if (const Step* fog = plan.sceneStep(kDepthFog)) {
        const float density = fog->v[p::kDensity];
        out[4] = fog->v[p::kFogStart];
        out[5] = 0.02f * density * density;
        out[6] = fog->intensity;
        // HSV, hue in degrees.
        const float h = std::fmod(fog->v[p::kFogHue], 360.0f) / 60.0f;
        const float sat = fog->v[p::kFogSaturation];
        const float v = fog->v[p::kFogBrightness];
        const float x = 1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f);
        const int sector = static_cast<int>(h) % 6;
        const float rgb[6][3] = {{1, x, 0}, {x, 1, 0}, {0, 1, x}, {0, x, 1}, {x, 0, 1}, {1, 0, x}};
        for (int i = 0; i < 3; i++) out[8 + i] = v * (1.0f - sat + sat * rgb[sector][i]);
    }
    if (const Step* dof = plan.sceneStep(kDepthOfField)) {
        out[12] = static_cast<int>(dof->v[p::kFocus]) == kFixedFocus ? 1.0f : 0.0f;
        out[13] = dof->v[p::kDistance];
        out[14] = 0.02f * dof->v[p::kDofBlur];
        out[15] = dof->v[p::kRange];
        out[16] = dof->intensity;
    }
    if (const Step* gi = plan.sceneStep(kGlobalIllumination)) {
        out[20] = gi->v[p::kReach];
        out[21] = gi->v[p::kBounce];
        out[22] = gi->intensity;
    }
}

void PackFrame(const Plan& plan, const FrameInfo& f, float out[kFrameVec4s * 4]) {
    const float* luma = LumaWeights(f.primaries);
    const float lsb = f.format == FormatClass::kUnorm10 ? 1.0f / 1023.0f : 1.0f / 255.0f;
    // The seed travels as an exact float: 23 bits wrap after a day at 135 Hz.
    const float values[kFrameVec4s * 4] = {
            static_cast<float>(f.bufferWidth),
            static_cast<float>(f.bufferHeight),
            f.displayWidth,
            f.displayHeight,
            static_cast<float>(f.turns & 3),
            f.flipY ? 1.0f : 0.0f,
            plan.split ? plan.splitX : -1.0f,
            static_cast<float>(f.seed & 0x7fffffu),
            luma[0],
            luma[1],
            luma[2],
            0.5f * lsb,
            f.edges ? 1.0f : 0.0f,
    };
    memcpy(out, values, sizeof(values));
}

Stages Schedule(const Plan* plan, bool generation, bool real) {
    Stages s;
    if (plan == nullptr || plan->bypass()) return s;
    if (!generation) {
        s.full = real;
        return s;
    }
    s.stageA = real && plan->detailCount + plan->colourCount > 0;
    s.stageB = true;
    return s;
}

}  // namespace framefx::kelir_legacy
