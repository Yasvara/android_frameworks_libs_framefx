/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "kelir/lut.h"
#include "kelir/plan.h"
#include "tests/host/kelir_test_util.h"

namespace framefx::kelir {
namespace {

using test::MakeNode;
using test::MakeStack;
using test::Set;

float MaxError(Rgb a, Rgb b) {
    return std::max({std::fabs(a.r - b.r), std::fabs(a.g - b.g), std::fabs(a.b - b.b)});
}

// The error that reaches the display, which clamps.
float ShownError(Rgb a, Rgb b) {
    const auto c = [](float v) { return std::clamp(v, 0.0f, 1.0f); };
    return MaxError({c(a.r), c(a.g), c(a.b)}, {c(b.r), c(b.g), c(b.b)});
}

// Each colour kind at settings across its range: what the editor can reach.
std::vector<Node> ColourSettings() {
    std::vector<Node> out;
    const auto with = [&](uint8_t kind, std::initializer_list<std::pair<int, float>> values,
                          float intensity = 1.0f) {
        Node n = MakeNode(kind, intensity);
        for (const auto& [param, value] : values) Set(n, param, value);
        out.push_back(n);
    };
    with(kBrightnessContrast, {{p::kExposure, 1.0f}});
    with(kBrightnessContrast, {{p::kExposure, -2.0f}, {p::kContrast, 0.5f}});
    with(kBrightnessContrast, {{p::kContrast, 1.0f}, {p::kShadows, 1.0f}, {p::kHighlights, -1.0f}});
    with(kBrightnessContrast, {{p::kGamma, 0.5f}});
    with(kBrightnessContrast, {{p::kGamma, 1.6f}});
    with(kColor, {{p::kTemperature, 1.0f}, {p::kTint, -0.6f}});
    with(kColor, {{p::kHue, 120.0f}, {p::kSaturation, 0.5f}});
    with(kColor, {{p::kVibrance, 1.0f}});
    with(kColor, {{p::kVibrance, -1.0f}, {p::kSaturation, -0.5f}});
    with(kColor, {{p::kTintColour, 30.0f}, {p::kTintIntensity, 0.6f}});
    with(kLevels, {{p::kBlack, 0.1f}, {p::kWhite, 0.9f}});
    with(kLevels, {{p::kMidtones, 0.6f}, {p::kBrightness, 0.2f}});
    with(kLevels, {{p::kMidtones, 1.6f}});
    with(kBlackWhite, {});
    with(kSepia, {}, 0.7f);
    with(kNightMode, {});
    for (int t = 0; t < 3; t++) with(kColorblind, {{p::kType, static_cast<float>(t)}});
    return out;
}

// Uniform over the cube, then as many within its darkest tenth, where curves are steepest.
std::vector<Rgb> Samples(int count) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<Rgb> s;
    for (int i = 0; i < count; i++) s.push_back({u(rng), u(rng), u(rng)});
    for (int i = 0; i < count; i++) s.push_back({0.1f * u(rng), 0.1f * u(rng), 0.1f * u(rng)});
    return s;
}

// ---- KelirLut -------------------------------------------------------------------------------

TEST(KelirLut, IdentityBakesIdentity) {
    Lut lut;
    BakeLut(nullptr, 0, Primaries::kSrgb, &lut);
    for (int b = 0; b < kLutSize; b++) {
        for (int g = 0; g < kLutSize; g++) {
            for (int r = 0; r < kLutSize; r++) {
                const uint16_t* t = &lut.texels[((b * kLutSize + g) * kLutSize + r) * 4];
                const auto at = [](int i) {
                    return ToHalf(std::pow(i / (kLutSize - 1.0f), 1.0f / kLutShaper));
                };
                ASSERT_EQ(at(r), t[0]);
                ASSERT_EQ(at(g), t[1]);
                ASSERT_EQ(at(b), t[2]);
                ASSERT_EQ(ToHalf(1.0f), t[3]);
            }
        }
    }
    EXPECT_FALSE(lut.linear);
    for (const Rgb& x : Samples(4096)) EXPECT_LE(MaxError(SampleLut(lut, x), x) * 255.0f, 0.5f);
}

// The LUT against the colour band evaluated exactly, as the hardware's trilinear filter reads it.
// Measured on 33^3 with the shaper: the worst colour sits within 2.5 LSB at 8 bits (strong
// negative vibrance), the mean within 0.3 LSB.
TEST(KelirLut, EachColourKindMatchesDirectEval) {
    const auto samples = Samples(4096);
    for (const Node& n : ColourSettings()) {
        const auto plan = Compile(MakeStack({n}), nullptr);
        ASSERT_EQ(1, plan->colourCount) << Info(n.kind).name;
        for (Primaries p : {Primaries::kSrgb, Primaries::kP3}) {
            const Lut& lut = plan->luts->get(p);
            float worst = 0.0f;
            double sum = 0.0;
            for (const Rgb& x : samples) {
                const float e = ShownError(SampleLut(lut, x), EvaluateColour(plan->colour, 1, p, x));
                worst = std::max(worst, e);
                sum += e;
            }
            EXPECT_LE(worst * 255.0f, 2.5f) << Info(n.kind).name << " q " << n.q[0] << "," << n.q[1];
            EXPECT_LE(sum / samples.size() * 255.0, 0.3) << Info(n.kind).name;
        }
    }
}

TEST(KelirLut, CompositionEqualsSequential) {
    Node a = MakeNode(kBrightnessContrast);
    Set(a, p::kContrast, 0.4f);
    Node b = MakeNode(kColor, 0.7f);
    Set(b, p::kHue, -40.0f);
    Set(b, p::kSaturation, 0.3f);
    const auto plan = Compile(MakeStack({a, MakeNode(kSepia, 0.5f), b}), nullptr);
    for (const Rgb& x : Samples(512)) {
        Rgb y = x;
        for (int i = 0; i < plan->colourCount; i++) y = Blend(plan->colour[i], y, Primaries::kSrgb);
        EXPECT_LE(ShownError(SampleLut(plan->luts->get(Primaries::kSrgb), x), y) * 255.0f, 2.5f);
    }
}

TEST(KelirLut, OrderMatters) {
    Node levels = MakeNode(kLevels);
    Set(levels, p::kBlack, 0.2f);
    Node contrast = MakeNode(kBrightnessContrast);
    Set(contrast, p::kContrast, 0.8f);
    const auto ab = Compile(MakeStack({levels, contrast}), nullptr);
    const auto ba = Compile(MakeStack({contrast, levels}), nullptr);
    const Rgb x = {0.3f, 0.5f, 0.7f};
    EXPECT_GT(MaxError(SampleLut(ab->luts->get(Primaries::kSrgb), x), SampleLut(ba->luts->get(Primaries::kSrgb), x)), 0.01f);
}

TEST(KelirLut, SrgbRoundTrip) {
    for (int i = -64; i <= 320; i++) {
        const float x = i / 256.0f;
        EXPECT_NEAR(x, Encode(Decode(x)), 2e-6f) << x;
        EXPECT_NEAR(x, Decode(Encode(x)), 2e-6f) << x;
    }
    EXPECT_FLOAT_EQ(0.5f, Decode(Encode(0.5f)));
    EXPECT_NEAR(0.21404114f, Decode(0.5f), 1e-6f);
    EXPECT_NEAR(1.0f, Decode(1.0f), 1e-6f);
}

TEST(KelirLut, P3PrimariesForP3Targets) {
    const auto plan = Compile(MakeStack({MakeNode(kBlackWhite)}), nullptr);
    const Rgb green = {0.0f, 1.0f, 0.0f};
    const Rgb srgb = SampleLut(plan->luts->get(Primaries::kSrgb), green);
    const Rgb p3 = SampleLut(plan->luts->get(Primaries::kP3), green);
    EXPECT_NEAR(Encode(0.7152f), srgb.g, 2e-3f);
    EXPECT_NEAR(Encode(0.6917385f), p3.g, 2e-3f);
    EXPECT_GT(std::fabs(srgb.g - p3.g), 0.005f);
}

// Machado et al. 2009 at severity 1.0 on linear sRGB, then Fidaner's error shift.
TEST(KelirLut, ColorblindInLinearLight) {
    const float deutan[9] = {0.367322f, 0.860646f, -0.227968f, 0.280085f, 0.672501f,
                             0.047413f, -0.011820f, 0.042940f, 0.968881f};
    Node n = MakeNode(kColorblind);
    Set(n, p::kType, kDeutan);
    const auto plan = Compile(MakeStack({n}), nullptr);
    const Rgb x = {0.8f, 0.3f, 0.2f};
    const float lin[3] = {Decode(x.r), Decode(x.g), Decode(x.b)};
    float sim[3];
    for (int i = 0; i < 3; i++) {
        sim[i] = deutan[3 * i] * lin[0] + deutan[3 * i + 1] * lin[1] + deutan[3 * i + 2] * lin[2];
    }
    const float e[3] = {lin[0] - sim[0], lin[1] - sim[1], lin[2] - sim[2]};
    const Rgb want = {Encode(lin[0]), Encode(lin[1] + 0.7f * e[0] + e[1]),
                      Encode(lin[2] + 0.7f * e[0] + e[2])};
    const Rgb got = ApplyColour(plan->colour[0], x, Primaries::kSrgb);
    EXPECT_LE(MaxError(want, got), 1e-5f);
    // The same matrices on encoded values give a visibly different colour.
    float simEncoded[3];
    const float enc[3] = {x.r, x.g, x.b};
    for (int i = 0; i < 3; i++) {
        simEncoded[i] = deutan[3 * i] * enc[0] + deutan[3 * i + 1] * enc[1] + deutan[3 * i + 2] * enc[2];
    }
    const Rgb encodedWay = {enc[0], enc[1] + 0.7f * (enc[0] - simEncoded[0]) + enc[1] - simEncoded[1],
                            enc[2] + 0.7f * (enc[0] - simEncoded[0]) + enc[2] - simEncoded[2]};
    EXPECT_GT(MaxError(encodedWay, got), 0.01f);
}

TEST(KelirLut, IntensityMixesInEncodedSpace) {
    Node full = MakeNode(kNightMode);
    Node half = MakeNode(kNightMode, 0.5f);
    const auto a = Compile(MakeStack({full}), nullptr);
    const auto b = Compile(MakeStack({half}), nullptr);
    const Rgb x = {0.6f, 0.6f, 0.9f};
    const Rgb y = ApplyColour(a->colour[0], x, Primaries::kSrgb);
    const Rgb m = Blend(b->colour[0], x, Primaries::kSrgb);
    EXPECT_NEAR((x.b + y.b) * 0.5f, m.b, 1e-4f);
    EXPECT_NEAR((x.g + y.g) * 0.5f, m.g, 1e-4f);
}

TEST(KelirLut, BakeFitsTheWatcherThread) {
    Stack s;
    for (const Node& n : ColourSettings()) {
        if (s.count == kMaxKelirNodes) break;
        s.nodes[s.count++] = n;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto plan = Compile(s, nullptr);
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
    EXPECT_EQ(kMaxKelirNodes, plan->colourCount);
    // Both primaries, eight nodes; the device budget is a few ms, a host build is no slower.
    EXPECT_LT(ms.count(), 50.0);
}

TEST(KelirLut, HalfConversionIsExactWhereHalfIs) {
    for (uint32_t h = 0; h < 0x7c00u; h++) {
        ASSERT_EQ(h, ToHalf(FromHalf(static_cast<uint16_t>(h))));
        ASSERT_EQ(h | 0x8000u, ToHalf(FromHalf(static_cast<uint16_t>(h | 0x8000u))));
    }
    EXPECT_EQ(0x3c00u, ToHalf(1.0f));
    EXPECT_EQ(0x7bffu, ToHalf(1e9f));
    EXPECT_EQ(ToHalf(1.0f), ToHalf(1.0f + 1.0f / 4096.0f));  // ties to even
}

TEST(KelirLut, SwatchAndCurvesFollowTheBand) {
    Node bw = MakeNode(kBlackWhite);
    const auto plan = Compile(MakeStack({bw}), nullptr);
    uint8_t swatch[kKelirSwatchColours * 3];
    Swatch(plan->colour, plan->colourCount, Primaries::kSrgb, swatch);
    for (int i = 0; i < kKelirSwatchColours; i++) {
        EXPECT_EQ(swatch[3 * i], swatch[3 * i + 1]);
        EXPECT_EQ(swatch[3 * i + 1], swatch[3 * i + 2]);
    }
    EXPECT_EQ(0, swatch[0]);
    EXPECT_EQ(255, swatch[12]);
    Node lift = MakeNode(kLevels);
    Set(lift, p::kBrightness, 0.25f);
    const auto lifted = Compile(MakeStack({lift}), nullptr);
    uint8_t curve[kKelirCurvePoints];
    Curve(lifted->colour[0], Primaries::kSrgb, curve);
    EXPECT_NEAR(64, curve[0], 1);
    EXPECT_EQ(255, curve[kKelirCurvePoints - 1]);
}

}  // namespace
}  // namespace framefx::kelir
