/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "kelir/colour.h"
#include "kelir/plan.h"
#include "tests/host/kelir_test_util.h"
#include "tests/ref/kelir_ref.h"

namespace framefx::kelir {
namespace {

using test::MakeNode;
using test::MakeStack;
using test::Set;

constexpr int kWidth = 96;
constexpr int kHeight = 64;

// Gradients, saturated patches, hard edges, highlights and low-contrast noise: something for
// every kind to act on.
ref::Image TestImage() {
    ref::Image img(kWidth, kHeight);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> noise(-0.015f, 0.015f);
    for (int y = 0; y < kHeight; y++) {
        for (int x = 0; x < kWidth; x++) {
            Rgb c = {x / (kWidth - 1.0f), y / (kHeight - 1.0f), 0.5f};
            if (x > 60 && y < 20) c = {0.9f, 0.2f, 0.1f};
            if (x > 60 && y >= 20 && y < 40) c = {0.15f, 0.6f, 0.95f};
            if (((x / 8) + (y / 8)) % 2 == 0 && x < 30 && y > 40) c = {1.0f, 1.0f, 0.98f};
            if (x < 30 && y < 30) c = {0.4f + noise(rng), 0.42f + noise(rng), 0.38f + noise(rng)};
            img.set(x, y, c);
        }
    }
    return img;
}

ref::Frame TestFrame(uint32_t seed) {
    ref::Frame f;
    f.displayWidth = kWidth * 4.0f;
    f.displayHeight = kHeight * 4.0f;
    f.seed = seed;
    f.dither = false;
    return f;
}

float Difference(const ref::Image& a, const ref::Image& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.rgb.size(); i++) d = std::max(d, std::fabs(a.rgb[i] - b.rgb[i]));
    return d;
}

// Whether two nodes of a kind give different pictures: colour kinds over a spread of colours,
// the others rendered, at a few presents for the kinds that draw new noise on each.
bool Differs(const Node& a, const Node& b) {
    const auto pa = Compile(MakeStack({a}), nullptr);
    const auto pb = Compile(MakeStack({b}), nullptr);
    if (Info(a.kind).band == Band::kScene) {
        // Luwes draws these on the game's depth; what reaches its passes is the packed band.
        // Three blocks: the depth kinds, the look pass and the ray-traced kinds.
        constexpr int kAll = (kSceneVec4s + kLookVec4s + kRaysVec4s) * 4;
        float va[kAll];
        float vb[kAll];
        PackScene(*pa, va);
        PackScene(*pb, vb);
        PackSceneLook(*pa, va + kSceneVec4s * 4);
        PackSceneLook(*pb, vb + kSceneVec4s * 4);
        PackSceneRays(*pa, va + (kSceneVec4s + kLookVec4s) * 4);
        PackSceneRays(*pb, vb + (kSceneVec4s + kLookVec4s) * 4);
        for (int i = 0; i < kAll; i++) {
            if (std::fabs(va[i] - vb[i]) > 1e-6f) return true;
        }
        return false;
    }
    if (Info(a.kind).band == Band::kColour) {
        if (pa->colourCount == 0 || pb->colourCount == 0) return pa->colourCount != pb->colourCount;
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (int i = 0; i < 4096; i++) {
            const Rgb x = {u(rng), u(rng), u(rng)};
            const Rgb ya = Blend(pa->colour[0], x, Primaries::kSrgb);
            const Rgb yb = Blend(pb->colour[0], x, Primaries::kSrgb);
            if (std::max({std::fabs(ya.r - yb.r), std::fabs(ya.g - yb.g), std::fabs(ya.b - yb.b)}) >
                1e-4f) {
                return true;
            }
        }
        return false;
    }
    const ref::Image src = TestImage();
    for (uint32_t seed : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        if (Difference(ref::Render(*pa, src, TestFrame(seed)),
                       ref::Render(*pb, src, TestFrame(seed))) > 1e-4f) {
            return true;
        }
    }
    return false;
}

// A value of the parameter away from its default: halfway to an end, since a hue's far end is
// its default again.
float Away(const Param& p) {
    if (p.choices > 0) return 1.0f;
    return p.def + 0.5f * ((p.hi != p.def ? p.hi : p.lo) - p.def);
}

// Every parameter of every kind changes the picture, given the parameters it depends on: no knob
// in the table is dead.
TEST(KelirParam, EveryParamIsLive) {
    for (int kind = 0; kind < kKindCount; kind++) {
        const KindInfo& info = Info(kind);
        for (int i = 0; i < info.paramCount; i++) {
            const Param& p = info.params[i];
            Node base = MakeNode(kind);
            if (p.needs >= 0) Set(base, p.needs, Away(info.params[p.needs]));
            // Colour kinds at their identity need something else on to show a change in intensity
            // or in this parameter; the parameter itself is enough.
            Node changed = base;
            Set(changed, i, Away(p));
            EXPECT_TRUE(Differs(base, changed)) << info.name << " " << p.name;
        }
    }
}

TEST(KelirParam, EveryIntensityIsLive) {
    for (int kind = 0; kind < kKindCount; kind++) {
        const KindInfo& info = Info(kind);
        Node full = MakeNode(kind, 1.0f);
        // Kinds whose defaults change nothing get one parameter moved.
        if (kind == kBrightnessContrast) Set(full, p::kContrast, 0.5f);
        if (kind == kColor) Set(full, p::kSaturation, 0.5f);
        if (kind == kLevels) Set(full, p::kBlack, 0.2f);
        if (kind == kDetails) Set(full, p::kClarity, 1.0f);
        if (kind == kSky) Set(full, 0, 0.5f);
        Node half = full;
        half.intensity = 32768;
        EXPECT_TRUE(Differs(full, half)) << info.name;
    }
}

}  // namespace
}  // namespace framefx::kelir
