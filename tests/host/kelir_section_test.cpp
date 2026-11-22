/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>
#include <json/json.h>

#include <cmath>
#include <fstream>
#include <string>

#include "kelir/plan.h"
#include "kelir/section.h"
#include "tests/host/kelir_test_util.h"

namespace framefx::kelir {
namespace {

using test::Bytes;
using test::MakeNode;
using test::MakeStack;
using test::Set;

KelirReject Decode(const std::vector<uint8_t>& bytes, Stack* out = nullptr) {
    Stack s;
    return DecodeSection(bytes.data(), bytes.size(), out != nullptr ? out : &s);
}

void Put16(std::vector<uint8_t>& b, size_t at, uint16_t v) {
    b[at] = static_cast<uint8_t>(v);
    b[at + 1] = static_cast<uint8_t>(v >> 8);
}

constexpr size_t kNode0 = kKelirHeaderSize;

Json::Value ReadJson(const std::string& path) {
    std::ifstream f(path);
    Json::Value v;
    f >> v;
    return v;
}

const char* BandName(Band b) {
    switch (b) {
        case Band::kDetail:
            return "detail";
        case Band::kColour:
            return "colour";
        case Band::kScreen:
            return "screen";
        case Band::kScene:
            return "scene";
    }
    return "?";
}

// ---- KelirSection ---------------------------------------------------------------------------

TEST(KelirSection, ParsesEmpty) {
    const auto bytes = test::ReadFile(test::GoldenDir() + "/kelir/empty.bin");
    Stack s;
    ASSERT_EQ(KelirReject::kNone, Decode(bytes, &s));
    EXPECT_EQ(0, s.count);
    EXPECT_TRUE(Compile(s, nullptr)->empty());
}

TEST(KelirSection, ParsesEveryKindAtDefaults) {
    for (int kind = 0; kind < kKindCount; kind++) {
        Stack s;
        ASSERT_EQ(KelirReject::kNone, Decode(Bytes(MakeStack({MakeNode(kind)})), &s))
                << Info(kind).name;
        EXPECT_EQ(kind, s.nodes[0].kind);
        for (int i = 0; i < Info(kind).paramCount; i++) {
            EXPECT_EQ(Info(kind).params[i].def, s.nodes[0].value(i)) << Info(kind).name;
        }
    }
}

TEST(KelirSection, ParsesMaxStack) {
    const auto bytes = test::ReadFile(test::GoldenDir() + "/kelir/max_stack.bin");
    Stack s;
    ASSERT_EQ(KelirReject::kNone, Decode(bytes, &s));
    EXPECT_EQ(kMaxKelirNodes, s.count);
    EXPECT_EQ(kKelirHeaderSize + kKelirNodeSize * kMaxKelirNodes, bytes.size());
}

TEST(KelirSection, RejectsMajor2) {
    auto b = Bytes(MakeStack({MakeNode(kSepia)}));
    b[0] = 2;
    EXPECT_EQ(KelirReject::kMajor, Decode(b));
}

TEST(KelirSection, AcceptsMinorAhead) {
    auto b = Bytes(MakeStack({MakeNode(kSepia)}));
    b[1] = 9;
    Stack s;
    EXPECT_EQ(KelirReject::kNone, Decode(b, &s));
    EXPECT_EQ(9, s.minor);
}

TEST(KelirSection, RejectsTruncatedHeader) {
    auto b = Bytes(MakeStack({}));
    b.pop_back();
    EXPECT_EQ(KelirReject::kSize, Decode(b));
    EXPECT_EQ(KelirReject::kSize, DecodeSection(nullptr, 0, nullptr));
}

TEST(KelirSection, RejectsCountSizeMismatch) {
    auto b = Bytes(MakeStack({MakeNode(kSepia)}));
    b[3] = 2;
    EXPECT_EQ(KelirReject::kSize, Decode(b));
    b[3] = 0;
    EXPECT_EQ(KelirReject::kSize, Decode(b));
}

TEST(KelirSection, RejectsTrailingBytes) {
    auto b = Bytes(MakeStack({MakeNode(kSepia)}));
    b.push_back(0);
    EXPECT_EQ(KelirReject::kSize, Decode(b));
}

TEST(KelirSection, RejectsCountAbove8) {
    std::vector<uint8_t> b(kKelirHeaderSize + kKelirNodeSize * 9, 0);
    b[0] = kKelirMajor;
    b[3] = 9;
    for (int i = 0; i < 9; i++) b[kNode0 + kKelirNodeSize * i] = kSepia;
    EXPECT_EQ(KelirReject::kCount, Decode(b));
}

TEST(KelirSection, RejectsUnknownKind) {
    auto b = Bytes(MakeStack({MakeNode(kSepia)}));
    b[kNode0] = kKindCount;
    EXPECT_EQ(KelirReject::kKind, Decode(b));
    b[kNode0] = 0xff;
    EXPECT_EQ(KelirReject::kKind, Decode(b));
}

TEST(KelirSection, RejectsReservedFlagBits) {
    auto b = Bytes(MakeStack({MakeNode(kSepia)}));
    for (int bit = 3; bit < 8; bit++) {
        auto h = b;
        h[2] |= static_cast<uint8_t>(1u << bit);
        EXPECT_EQ(KelirReject::kFlags, Decode(h)) << "header bit " << bit;
    }
    for (int bit = 1; bit < 8; bit++) {
        auto n = b;
        n[kNode0 + 1] |= static_cast<uint8_t>(1u << bit);
        EXPECT_EQ(KelirReject::kFlags, Decode(n)) << "node bit " << bit;
    }
}

TEST(KelirSection, RejectsNonZeroReserved) {
    const auto b = Bytes(MakeStack({MakeNode(kSepia)}));
    for (size_t at : {6u, 7u, 12u, 13u, 14u, 15u}) {
        auto r = b;
        r[at] = 1;
        EXPECT_EQ(KelirReject::kReserved, Decode(r)) << "byte " << at;
    }
}

TEST(KelirSection, RejectsNonZeroUnusedParam) {
    for (int kind = 0; kind < kKindCount; kind++) {
        for (int i = Info(kind).paramCount; i < kMaxParams; i++) {
            auto b = Bytes(MakeStack({MakeNode(kind)}));
            Put16(b, kNode0 + 4 + 2 * i, 1);
            EXPECT_EQ(KelirReject::kUnusedParam, Decode(b)) << Info(kind).name << " p" << i;
        }
    }
}

TEST(KelirSection, RejectsQMin) {
    auto b = Bytes(MakeStack({MakeNode(kBrightnessContrast)}));
    Put16(b, kNode0 + 4, 0x8000);
    EXPECT_EQ(KelirReject::kParamRange, Decode(b));
}

TEST(KelirSection, RejectsNegativeOnOneSidedParam) {
    for (int kind = 0; kind < kKindCount; kind++) {
        const KindInfo& info = Info(kind);
        for (int i = 0; i < info.paramCount; i++) {
            const Param& p = info.params[i];
            if (p.choices > 0) continue;
            auto b = Bytes(MakeStack({MakeNode(kind)}));
            Put16(b, kNode0 + 4 + 2 * i, static_cast<uint16_t>(-1));
            EXPECT_EQ(p.lo == p.def ? KelirReject::kParamRange : KelirReject::kNone, Decode(b))
                    << info.name << " " << p.name;
            Put16(b, kNode0 + 4 + 2 * i, 1);
            EXPECT_EQ(p.hi == p.def ? KelirReject::kParamRange : KelirReject::kNone, Decode(b))
                    << info.name << " " << p.name;
        }
    }
}

TEST(KelirSection, RejectsDiscreteOutOfRange) {
    for (int kind = 0; kind < kKindCount; kind++) {
        const KindInfo& info = Info(kind);
        for (int i = 0; i < info.paramCount; i++) {
            const Param& p = info.params[i];
            if (p.choices == 0) continue;
            for (int q : {-1, static_cast<int>(p.choices), 1000}) {
                auto b = Bytes(MakeStack({MakeNode(kind)}));
                Put16(b, kNode0 + 4 + 2 * i, static_cast<uint16_t>(q));
                EXPECT_EQ(KelirReject::kParamRange, Decode(b)) << info.name << " " << q;
            }
            auto b = Bytes(MakeStack({MakeNode(kind)}));
            Put16(b, kNode0 + 4 + 2 * i, static_cast<uint16_t>(p.choices - 1));
            EXPECT_EQ(KelirReject::kNone, Decode(b));
        }
    }
}

TEST(KelirSection, RejectsRepeatedNonColourKind) {
    for (int kind = 0; kind < kKindCount; kind++) {
        const auto r = Decode(Bytes(MakeStack({MakeNode(kind), MakeNode(kind)})));
        if (Info(kind).band == Band::kColour) {
            EXPECT_EQ(KelirReject::kNone, r) << Info(kind).name;
        } else {
            EXPECT_EQ(KelirReject::kRepeat, r) << Info(kind).name;
        }
    }
}

TEST(KelirSection, RejectsBandLimits) {
    EXPECT_EQ(KelirReject::kBandLimit,
              Decode(Bytes(MakeStack({MakeNode(kSharpen), MakeNode(kDetails),
                                      MakeNode(kTiltShift), MakeNode(kPainterly)}))));
    EXPECT_EQ(KelirReject::kBandLimit,
              Decode(Bytes(MakeStack({MakeNode(kPainterly), MakeNode(kWatercolor)}))));
    EXPECT_EQ(KelirReject::kBandLimit,
              Decode(Bytes(MakeStack({MakeNode(kVignette), MakeNode(kFilmGrain),
                                      MakeNode(kLetterbox), MakeNode(kOldFilm),
                                      MakeNode(kHalftone)}))));
    EXPECT_NE(KelirReject::kNone, Decode(Bytes(MakeStack({MakeNode(kRetro), MakeNode(kRetro)}))));
    // At the limits.
    EXPECT_EQ(KelirReject::kNone,
              Decode(Bytes(MakeStack({MakeNode(kSharpen), MakeNode(kDetails), MakeNode(kPainterly),
                                      MakeNode(kVignette), MakeNode(kFilmGrain),
                                      MakeNode(kLetterbox), MakeNode(kRetro),
                                      MakeNode(kColor)}))));
}

TEST(KelirSection, RejectsOversizeBeforeParse) {
    std::vector<uint8_t> b(kMaxKelirSectionSize + 1, 0);
    b[0] = kKelirMajor;
    EXPECT_EQ(KelirReject::kSize, Decode(b));
}

TEST(KelirSection, RejectionLeavesTheOutputAlone) {
    Stack s = MakeStack({MakeNode(kSepia)}, 5);
    auto b = Bytes(MakeStack({MakeNode(kColor)}));
    b[0] = 2;
    const Stack before = s;
    EXPECT_NE(KelirReject::kNone, Decode(b, &s));
    EXPECT_EQ(before, s);
}

TEST(KelirSection, EncodeRoundTrips) {
    Node a = MakeNode(kTiltShift, 0.4f);
    Set(a, p::kShape, kRadial);
    Set(a, p::kAngle, -45.0f);
    Node b = MakeNode(kColor, 0.9f);
    Set(b, p::kHue, 30.0f);
    b.enabled = false;
    Stack s = MakeStack({a, b}, 0x12345678);
    s.peek = true;
    s.inspect = true;
    s.splitX = 40000;
    Stack back;
    ASSERT_EQ(KelirReject::kNone, Decode(Bytes(s), &back));
    EXPECT_EQ(s, back);
    EXPECT_EQ(Bytes(s), Bytes(back));
}

void ExpectPlanMatches(const Json::Value& want, const Plan& plan, const std::string& name) {
    EXPECT_EQ(want["revision"].asUInt(), plan.revision) << name;
    EXPECT_EQ(want["peek"].asBool(), plan.peek) << name;
    EXPECT_EQ(want["split"].asBool(), plan.split) << name;
    EXPECT_EQ(want["inspect"].asBool(), plan.inspect) << name;
    EXPECT_NEAR(want["split_x"].asFloat(), plan.splitX, 1e-5) << name;
    const auto band = [&](const char* key, const Step* steps, int count) {
        const Json::Value& w = want[key];
        ASSERT_EQ(static_cast<int>(w.size()), count) << name << " " << key;
        for (int i = 0; i < count; i++) {
            EXPECT_EQ(w[i]["kind"].asString(), Info(steps[i].kind).name) << name;
            EXPECT_NEAR(w[i]["intensity"].asFloat(), steps[i].intensity, 1e-5) << name;
            const Json::Value& values = w[i]["values"];
            ASSERT_EQ(static_cast<int>(values.size()), Info(steps[i].kind).paramCount);
            for (Json::ArrayIndex j = 0; j < values.size(); j++) {
                EXPECT_NEAR(values[j].asFloat(), steps[i].v[j], 1e-5)
                        << name << " " << key << i << " p" << j;
            }
        }
    };
    band("detail", plan.detail, plan.detailCount);
    band("colour", plan.colour, plan.colourCount);
    band("screen", plan.screen, plan.screenCount);
}

TEST(KelirSection, GoldenVectors) {
    const std::string dir = test::GoldenDir() + "/kelir/";
    int checked = 0;
    for (const char* name : {"empty", "flags", "colour_band", "detail_band", "screen_band",
                             "max_stack", "inert_nodes"}) {
        const auto bytes = test::ReadFile(dir + name + ".bin");
        ASSERT_FALSE(bytes.empty()) << name;
        Stack s;
        ASSERT_EQ(KelirReject::kNone, Decode(bytes, &s)) << name;
        EXPECT_EQ(bytes, Bytes(s)) << name;
        ExpectPlanMatches(ReadJson(dir + name + ".json"), *Compile(s, nullptr), name);
        checked++;
    }
    EXPECT_EQ(7, checked);
}

// ---- KelirParam -----------------------------------------------------------------------------

TEST(KelirParam, DefaultRoundTripsExactly) {
    for (int kind = 0; kind < kKindCount; kind++) {
        for (int i = 0; i < Info(kind).paramCount; i++) {
            const Param& p = Info(kind).params[i];
            EXPECT_EQ(p.def, Decode(p, 0)) << Info(kind).name << " " << p.name;
            EXPECT_EQ(0, Encode(p, p.def)) << Info(kind).name << " " << p.name;
            if (p.choices == 0) {
                EXPECT_EQ(p.hi, Decode(p, Encode(p, p.hi))) << p.name;
                EXPECT_EQ(p.lo, Decode(p, Encode(p, p.lo))) << p.name;
            }
        }
    }
}

TEST(KelirParam, DecodeMonotonic) {
    for (int kind = 0; kind < kKindCount; kind++) {
        for (int i = 0; i < Info(kind).paramCount; i++) {
            const Param& p = Info(kind).params[i];
            float last = -INFINITY;
            for (int q = -kQMax; q <= kQMax; q++) {
                if (!CodeValid(p, static_cast<int16_t>(q))) continue;
                const float v = Decode(p, static_cast<int16_t>(q));
                ASSERT_GE(v, last) << p.name << " " << q;
                last = v;
            }
        }
    }
}

TEST(KelirParam, ValuesInsideRange) {
    for (int kind = 0; kind < kKindCount; kind++) {
        for (int i = 0; i < Info(kind).paramCount; i++) {
            const Param& p = Info(kind).params[i];
            int valid = 0;
            for (int q = INT16_MIN; q <= INT16_MAX; q++) {
                if (!CodeValid(p, static_cast<int16_t>(q))) continue;
                valid++;
                const float v = Decode(p, static_cast<int16_t>(q));
                ASSERT_TRUE(std::isfinite(v));
                if (p.choices > 0) {
                    ASSERT_TRUE(v >= 0 && v < p.choices);
                } else {
                    ASSERT_GE(v, p.lo) << p.name << " " << q;
                    ASSERT_LE(v, p.hi) << p.name << " " << q;
                }
            }
            if (p.choices > 0) {
                EXPECT_EQ(p.choices, valid);
            } else {
                EXPECT_EQ((p.lo < p.def ? kQMax : 0) + (p.hi > p.def ? kQMax : 0) + 1, valid);
            }
        }
    }
}

TEST(KelirParam, ClampsLevelsCrossover) {
    Node n = MakeNode(kLevels);
    Set(n, p::kBlack, 0.5f);
    Set(n, p::kWhite, 0.5f);
    const auto plan = Compile(MakeStack({n}), nullptr);
    ASSERT_EQ(1, plan->colourCount);
    float last = -INFINITY;
    for (int i = 0; i <= 64; i++) {
        const float x = i / 64.0f;
        const Rgb y = Blend(plan->colour[0], {x, x, x}, Primaries::kSrgb);
        ASSERT_TRUE(std::isfinite(y.r));
        ASSERT_GE(y.r, last);
        last = y.r;
    }
}

TEST(KelirParam, TableMatchesGoldenJson) {
    const Json::Value table = ReadJson(test::GoldenDir() + "/kelir_params.json");
    ASSERT_EQ(static_cast<Json::ArrayIndex>(kKindCount), table.size());
    for (int kind = 0; kind < kKindCount; kind++) {
        const KindInfo& info = Info(kind);
        const Json::Value& k = table[kind];
        EXPECT_EQ(k["name"].asString(), info.name);
        EXPECT_EQ(k["band"].asString(), BandName(info.band)) << info.name;
        EXPECT_EQ(k["style"].asBool(), info.style) << info.name;
        ASSERT_EQ(k["params"].size(), static_cast<Json::ArrayIndex>(info.paramCount)) << info.name;
        for (int i = 0; i < info.paramCount; i++) {
            const Json::Value& pj = k["params"][i];
            const Param& p = info.params[i];
            EXPECT_EQ(pj["name"].asString(), p.name);
            EXPECT_EQ(pj["choices"].asInt(), p.choices) << p.name;
            EXPECT_EQ(pj["lo"].asFloat(), p.lo) << p.name;
            EXPECT_EQ(pj["default"].asFloat(), p.def) << p.name;
            EXPECT_EQ(pj["hi"].asFloat(), p.hi) << p.name;
            EXPECT_EQ(pj["needs"].asInt(), p.needs) << p.name;
        }
    }
}

}  // namespace
}  // namespace framefx::kelir
