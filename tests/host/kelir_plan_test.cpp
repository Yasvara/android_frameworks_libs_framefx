/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "kelir/cache.h"
#include "kelir/plan.h"
#include "tests/host/kelir_test_util.h"

namespace framefx::kelir {
namespace {

using test::MakeNode;
using test::MakeStack;
using test::Set;

Node Contrast(float v) {
    Node n = MakeNode(kBrightnessContrast);
    return Set(n, p::kContrast, v);
}

Node Hue(float degrees) {
    Node n = MakeNode(kColor);
    return Set(n, p::kHue, degrees);
}

std::vector<uint8_t> Kinds(const Step* steps, int count) {
    std::vector<uint8_t> k;
    for (int i = 0; i < count; i++) k.push_back(steps[i].kind);
    return k;
}

using Kinds_ = std::vector<uint8_t>;

// ---- KelirPlan ------------------------------------------------------------------------------

TEST(KelirPlan, BandsCanonical) {
    const auto plan = Compile(MakeStack({MakeNode(kVignette), Hue(20.0f), MakeNode(kSharpen),
                                         Contrast(0.3f), MakeNode(kFilmGrain)}),
                              nullptr);
    EXPECT_EQ((Kinds_{kSharpen}), Kinds(plan->detail, plan->detailCount));
    EXPECT_EQ((Kinds_{kColor, kBrightnessContrast}), Kinds(plan->colour, plan->colourCount));
    EXPECT_EQ((Kinds_{kVignette, kFilmGrain}), Kinds(plan->screen, plan->screenCount));
    EXPECT_EQ(1, plan->colourNode[0]);
    EXPECT_EQ(3, plan->colourNode[1]);
}

TEST(KelirPlan, OrderWithinBandPreserved) {
    const auto plan = Compile(MakeStack({MakeNode(kTiltShift), MakeNode(kSharpen),
                                         MakeNode(kHalftone), MakeNode(kVignette)}),
                              nullptr);
    EXPECT_EQ((Kinds_{kTiltShift, kSharpen}), Kinds(plan->detail, plan->detailCount));
    EXPECT_EQ((Kinds_{kHalftone, kVignette}), Kinds(plan->screen, plan->screenCount));
}

TEST(KelirPlan, StageAssignment) {
    for (int kind = 0; kind < kKindCount; kind++) {
        Node n = MakeNode(kind);
        if (kind == kBrightnessContrast) Set(n, p::kContrast, 0.5f);
        if (kind == kColor) Set(n, p::kHue, 10.0f);
        if (kind == kLevels) Set(n, p::kBlack, 0.1f);
        if (kind == kDetails) Set(n, p::kClarity, 0.5f);
        if (kind == kSky) Set(n, 0, 0.3f);  // Brightness: Sky at its defaults changes nothing
        const auto plan = Compile(MakeStack({n}), nullptr);
        EXPECT_EQ(Info(kind).band == Band::kDetail ? 1 : 0, plan->detailCount) << Info(kind).name;
        EXPECT_EQ(Info(kind).band == Band::kColour ? 1 : 0, plan->colourCount) << Info(kind).name;
        EXPECT_EQ(Info(kind).band == Band::kScreen ? 1 : 0, plan->screenCount) << Info(kind).name;
        EXPECT_EQ(Info(kind).band == Band::kScene ? 1 : 0, plan->sceneCount) << Info(kind).name;
    }
}

TEST(KelirPlan, SceneBandIsNotKelirsToDraw) {
    const auto plan = Compile(MakeStack({MakeNode(kAmbientOcclusion), MakeNode(kDepthFog),
                                         MakeNode(kDepthOfField)}),
                              nullptr);
    EXPECT_EQ(3, plan->sceneCount);
    EXPECT_TRUE(plan->empty());
    EXPECT_EQ(0u, Signature(*plan, Target()) & ~0xffull);
}

TEST(KelirPlan, SceneBandInertWithoutEffect) {
    Node fog = MakeNode(kDepthFog);
    Set(fog, p::kDensity, 0.0f);
    Node dof = MakeNode(kDepthOfField);
    Set(dof, p::kDofBlur, 0.0f);
    EXPECT_EQ(0, Compile(MakeStack({fog, dof}), nullptr)->sceneCount);
}

TEST(KelirPlan, PacksTheSceneBand) {
    Node dof = MakeNode(kDepthOfField, 0.5f);
    Set(dof, p::kFocus, 1.0f);
    Set(dof, p::kDistance, 25.0f);
    const auto plan = Compile(MakeStack({MakeNode(kAmbientOcclusion), dof}), nullptr);
    float v[kSceneVec4s * 4];
    PackScene(*plan, v);
    EXPECT_FLOAT_EQ(0.6f, v[0]);
    EXPECT_FLOAT_EQ(1.0f, v[1]);
    EXPECT_EQ(0.0f, v[6]);  // no fog
    EXPECT_EQ(1.0f, v[12]);
    EXPECT_NEAR(25.0f, v[13], 0.01f);
    EXPECT_NEAR(0.5f, v[16], 1e-4f);
}

TEST(KelirPlan, FogColourFromHueSaturationBrightness) {
    Node fog = MakeNode(kDepthFog);
    Set(fog, p::kFogHue, 0.0f);
    Set(fog, p::kFogSaturation, 1.0f);
    Set(fog, p::kFogBrightness, 1.0f);
    float v[kSceneVec4s * 4];
    PackScene(*Compile(MakeStack({fog}), nullptr), v);
    EXPECT_NEAR(1.0f, v[8], 1e-4f);
    EXPECT_NEAR(0.0f, v[9], 1e-4f);
    EXPECT_NEAR(0.0f, v[10], 1e-4f);
}

TEST(KelirPlan, DisabledExcluded) {
    Node n = MakeNode(kSepia);
    n.enabled = false;
    EXPECT_TRUE(Compile(MakeStack({n}), nullptr)->empty());
}

TEST(KelirPlan, IntensityZeroExcluded) {
    EXPECT_TRUE(Compile(MakeStack({MakeNode(kVignette, 0.0f), MakeNode(kSharpen, 0.0f)}), nullptr)
                        ->empty());
}

TEST(KelirPlan, AllIdentityIsPassthrough) {
    Node tint = MakeNode(kColor);
    Set(tint, p::kTintColour, 200.0f);  // no tint intensity: still nothing
    Node tilt = MakeNode(kTiltShift);
    Set(tilt, p::kBlur, 0.0f);
    const auto plan = Compile(MakeStack({MakeNode(kBrightnessContrast), tint, MakeNode(kLevels),
                                         MakeNode(kDetails), tilt}),
                              nullptr);
    EXPECT_TRUE(plan->empty());
    EXPECT_TRUE(plan->bypass());
    EXPECT_EQ(nullptr, plan->luts);
}

TEST(KelirPlan, PeekBypassesANonEmptyPlan) {
    Stack s = MakeStack({MakeNode(kSepia)});
    s.peek = true;
    const auto plan = Compile(s, nullptr);
    EXPECT_FALSE(plan->empty());
    EXPECT_TRUE(plan->bypass());
}

TEST(KelirPlan, PyramidOnlyWhenNeeded) {
    for (int kind = 0; kind < kKindCount; kind++) {
        Node n = MakeNode(kind);
        if (kind == kDetails) Set(n, p::kBloom, 0.5f);
        const auto plan = Compile(MakeStack({n}), nullptr);
        const bool wants = kind == kDetails || kind == kTiltShift || kind == kPainterly ||
                kind == kWatercolor;
        EXPECT_EQ(wants, plan->needsPyramid()) << Info(kind).name;
    }
}

TEST(KelirPlan, SourceTextureOnlyWhenNeeded) {
    for (int kind = 0; kind < kKindCount; kind++) {
        Node n = MakeNode(kind);
        if (kind == kDetails) Set(n, p::kBloom, 0.5f);
        if (kind == kColor) Set(n, p::kHue, 10.0f);
        const auto plan = Compile(MakeStack({n}), nullptr);
        const bool neighbours = Info(kind).band == Band::kDetail || kind == kRetro;
        EXPECT_EQ(neighbours, plan->needsSource()) << Info(kind).name;
    }
}

TEST(KelirPlan, OneLutForOneToEightColourNodes) {
    std::shared_ptr<const Plan> previous;
    for (int n = 1; n <= kMaxKelirNodes; n++) {
        Stack s;
        for (int i = 0; i < n; i++) s.nodes[s.count++] = Hue(5.0f * (i + 1));
        const auto plan = Compile(s, nullptr);
        ASSERT_NE(nullptr, plan->luts);
        EXPECT_EQ(kLutTexels * 4, plan->luts->get(Primaries::kSrgb).texels.size());
        EXPECT_EQ(kLutTexels * 4, plan->luts->get(Primaries::kP3).texels.size());
        EXPECT_EQ(n, plan->colourCount);
    }
    // A screen edit keeps the colour band, and with it the baked LUTs.
    Node v = MakeNode(kVignette);
    const auto a = Compile(MakeStack({Hue(30.0f), v}), nullptr);
    Set(v, p::kVignetteSize, 0.9f);
    const auto b = Compile(MakeStack({Hue(30.0f), v}), a.get());
    EXPECT_EQ(a->luts, b->luts);
    const auto c = Compile(MakeStack({Hue(31.0f), v}), b.get());
    EXPECT_NE(b->luts, c->luts);
}

TEST(KelirPlan, LuwesSplitsGenerationOutput) {
    const auto plan = Compile(MakeStack({MakeNode(kSharpen), Hue(10.0f), MakeNode(kVignette)}),
                              nullptr);
    Target full;
    Target a = full;
    a.role = Role::kStageA;
    Target b = full;
    b.role = Role::kStageB;
    EXPECT_NE(Signature(*plan, full), Signature(*plan, a));
    // Stage B over a generated frame has no detail and no LUT.
    const Program p = Unpack(Signature(*plan, b));
    EXPECT_FALSE(p.lut);
    EXPECT_EQ(-1, p.detail[0]);
    EXPECT_EQ(kVignette, p.screen[0]);
}

TEST(KelirPlan, PackPlanPlacesSlots) {
    Node tilt = MakeNode(kTiltShift, 0.25f);
    Set(tilt, p::kAngle, 45.0f);
    Node grain = MakeNode(kFilmGrain, 0.5f);
    Set(grain, p::kGrainSize, 2.0f);
    const auto plan = Compile(MakeStack({MakeNode(kSharpen), tilt, MakeNode(kVignette), grain}),
                              nullptr);
    float packed[kPlanVec4s * 4];
    PackPlan(*plan, packed);
    EXPECT_NEAR(45.0f, packed[8 + p::kAngle], 0.01);           // detail slot 1
    EXPECT_NEAR(2.0f, packed[6 * 4 + 8 + p::kGrainSize], 0.01);  // screen slot 1
    EXPECT_NEAR(0.25f, packed[14 * 4 + 1], 1e-4);
    EXPECT_NEAR(0.5f, packed[15 * 4 + 1], 1e-4);
    EXPECT_EQ(0.0f, packed[14 * 4 + 2]);
}

// ---- KelirSignature -------------------------------------------------------------------------

TEST(KelirSignature, ParamsDoNotChange) {
    Node sharpen = MakeNode(kSharpen);
    Node vignette = MakeNode(kVignette);
    Stack s = MakeStack({sharpen, Hue(10.0f), vignette});
    const uint64_t before = Signature(*Compile(s, nullptr), Target());
    Set(s.nodes[0], p::kIgnoreGrain, 0.7f);
    s.nodes[0].intensity = 1000;
    Set(s.nodes[2], p::kRoundness, 0.1f);
    s.split = true;
    s.splitX = 12345;
    EXPECT_EQ(before, Signature(*Compile(s, nullptr), Target()));
}

TEST(KelirSignature, ColourEditsNeverChange) {
    const uint64_t one = Signature(*Compile(MakeStack({Hue(10.0f)}), nullptr), Target());
    Stack s;
    for (int i = 0; i < kMaxKelirNodes; i++) {
        s.nodes[s.count++] = i % 2 == 0 ? Contrast(0.1f * (i + 1)) : MakeNode(kSepia, 0.3f);
        EXPECT_EQ(one, Signature(*Compile(s, nullptr), Target())) << i;
    }
}

TEST(KelirSignature, KindOrderChanges) {
    const auto ab = Compile(MakeStack({MakeNode(kSharpen), MakeNode(kTiltShift)}), nullptr);
    const auto ba = Compile(MakeStack({MakeNode(kTiltShift), MakeNode(kSharpen)}), nullptr);
    EXPECT_NE(Signature(*ab, Target()), Signature(*ba, Target()));
    const auto vg = Compile(MakeStack({MakeNode(kVignette), MakeNode(kFilmGrain)}), nullptr);
    const auto gv = Compile(MakeStack({MakeNode(kFilmGrain), MakeNode(kVignette)}), nullptr);
    EXPECT_NE(Signature(*vg, Target()), Signature(*gv, Target()));
}

TEST(KelirSignature, TargetClassChanges) {
    const auto plan = Compile(MakeStack({Hue(10.0f)}), nullptr);
    std::set<uint64_t> seen;
    for (FormatClass f : {FormatClass::kUnorm8, FormatClass::kSrgb8, FormatClass::kUnorm10}) {
        for (bool reads : {false, true}) {
            for (bool fetch : {false, true}) {
                for (bool msaa : {false, true}) {
                    Target t;
                    t.format = f;
                    t.readsLinear = reads;
                    t.fetch = fetch;
                    t.msaa = msaa;
                    EXPECT_TRUE(seen.insert(Signature(*plan, t)).second);
                }
            }
        }
    }
}

TEST(KelirSignature, GoldenValues) {
    Node plus = MakeNode(kSharpen);
    Set(plus, p::kQuality, kPlus);
    const auto plan = Compile(MakeStack({plus, MakeNode(kWatercolor), Hue(5.0f),
                                         MakeNode(kRetro), MakeNode(kLetterbox)}),
                              nullptr);
    Target t;
    t.format = FormatClass::kSrgb8;
    t.readsLinear = true;
    // Letterbox 3 and Retro 5 in the screen slots, Sharpen Plus, Watercolor 5 and Sharpen 1 in
    // the detail slots, linear reads, a LUT, sRGB.
    EXPECT_EQ(0x1d1051c1ull, Signature(*plan, t));
    t.format = FormatClass::kUnorm8;
    t.readsLinear = false;
    t.fetch = true;
    EXPECT_EQ(0x1d105144ull, Signature(*plan, t));
    EXPECT_EQ(0x0ull, Signature(*Compile(Stack(), nullptr), Target()));
}

// Every ordered detail list the limits allow, every screen list, both sharpen qualities, with and
// without a LUT, under every target: all distinct, and each unpacks to what made it.
TEST(KelirSignature, NoCollisionsAcrossAllValidDetailScreenPermutations) {
    std::vector<std::vector<uint8_t>> details = {{}};
    for (size_t len = 1; len <= kMaxDetail; len++) {
        std::vector<std::vector<uint8_t>> next;
        for (const auto& d : details) {
            if (d.size() != len - 1) continue;
            for (uint8_t k = kSharpen; k <= kWatercolor; k++) {
                if (std::find(d.begin(), d.end(), k) != d.end()) continue;
                int styles = Info(k).style;
                for (uint8_t e : d) styles += Info(e).style;
                if (styles > 1) continue;
                auto n = d;
                n.push_back(k);
                next.push_back(n);
            }
        }
        details.insert(details.end(), next.begin(), next.end());
    }
    std::vector<std::vector<uint8_t>> screens = {{}};
    for (size_t len = 1; len <= kMaxScreen; len++) {
        std::vector<std::vector<uint8_t>> next;
        for (const auto& s : screens) {
            if (s.size() != len - 1) continue;
            for (uint8_t k = kVignette; k <= kHalftone; k++) {
                if (std::find(s.begin(), s.end(), k) != s.end()) continue;
                auto n = s;
                n.push_back(k);
                next.push_back(n);
            }
        }
        screens.insert(screens.end(), next.begin(), next.end());
    }
    std::set<uint64_t> seen;
    size_t total = 0;
    for (const auto& d : details) {
        const bool hasSharpen = std::find(d.begin(), d.end(), kSharpen) != d.end();
        for (int quality = 0; quality <= (hasSharpen ? 1 : 0); quality++) {
            for (const auto& s : screens) {
                for (bool lut : {false, true}) {
                    // Built directly: a signature reads the kinds, not the baked LUT.
                    auto plan = std::make_unique<Plan>();
                    for (uint8_t k : d) {
                        Step& step = plan->detail[plan->detailCount++];
                        step.kind = k;
                        if (k == kSharpen) step.v[p::kQuality] = static_cast<float>(quality);
                    }
                    for (uint8_t k : s) plan->screen[plan->screenCount++].kind = k;
                    if (lut) plan->colour[plan->colourCount++].kind = kColor;
                    for (FormatClass f : {FormatClass::kUnorm8, FormatClass::kSrgb8}) {
                        for (bool fetch : {false, true}) {
                            Target t;
                            t.format = f;
                            t.fetch = fetch;
                            const uint64_t sig = Signature(*plan, t);
                            ASSERT_TRUE(seen.insert(sig).second);
                            const Program back = Unpack(sig);
                            for (int i = 0; i < kMaxDetail; i++) {
                                ASSERT_EQ(i < static_cast<int>(d.size()) ? d[i] : -1,
                                          back.detail[i]);
                            }
                            for (int i = 0; i < kMaxScreen; i++) {
                                ASSERT_EQ(i < static_cast<int>(s.size()) ? s[i] : -1,
                                          back.screen[i]);
                            }
                            ASSERT_EQ(lut, back.lut);
                            ASSERT_EQ(quality == 1, back.sharpenPlus);
                            ASSERT_EQ(f, back.target.format);
                            ASSERT_EQ(fetch, back.target.fetch);
                            total++;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(total, seen.size());
    EXPECT_GT(total, 10000u);
}

// ---- KelirCache -----------------------------------------------------------------------------

TEST(KelirCache, LruBoundedAt8) {
    Lru<int> lru;
    for (int i = 0; i < 20; i++) lru.insert(static_cast<uint64_t>(i), i + 1);
    EXPECT_EQ(8u, lru.size());
}

TEST(KelirCache, HitOnRepeat) {
    Lru<int> lru;
    lru.insert(42, 7);
    ASSERT_NE(nullptr, lru.find(42));
    EXPECT_EQ(7, *lru.find(42));
    EXPECT_EQ(nullptr, lru.find(43));
}

TEST(KelirCache, EvictsLeastRecent) {
    Lru<int> lru;
    for (int i = 0; i < 8; i++) lru.insert(static_cast<uint64_t>(i), i + 100);
    lru.find(0);  // 1 is now the oldest
    EXPECT_EQ(101, lru.insert(99, 199));
    EXPECT_EQ(nullptr, lru.find(1));
    EXPECT_NE(nullptr, lru.find(0));
    EXPECT_NE(nullptr, lru.find(99));
}

TEST(KelirCache, NoGrowthOver10kRandomEdits) {
    Lru<int> lru;
    uint64_t state = 1;
    int evictions = 0;
    for (int i = 0; i < 10000; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const uint64_t key = (state >> 33) % 23;
        if (lru.find(key) == nullptr) evictions += lru.insert(key, 1) != 0;
    }
    EXPECT_EQ(8u, lru.size());
    EXPECT_GT(evictions, 0);
}

}  // namespace
}  // namespace framefx::kelir
