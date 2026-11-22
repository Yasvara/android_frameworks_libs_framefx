/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

// Kelir's logic in kelir/rust against the C++ it replaced (tests/parity/kelir_legacy), bit for bit:
// sections, codes, the transfer pair, every colour kind, the LUT bake, plans and their packing.

#include <gtest/gtest.h>

#include <cstring>
#include <random>

#include "kelir/lut.h"
#include "kelir/plan.h"
#include "kelir/section.h"
#include "tests/parity/kelir_legacy/lut.h"
#include "tests/parity/kelir_legacy/plan.h"
#include "tests/parity/kelir_legacy/section.h"

namespace framefx {
namespace {

namespace now = kelir;
namespace old = kelir_legacy;

uint32_t bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// A random valid stack: random kinds within the band limits, random valid codes.
bool randomStack(std::mt19937& rng, uint8_t* section, size_t* length) {
    old::Stack s;
    s.revision = rng();
    s.count = static_cast<int>(rng() % (kMaxKelirNodes + 1));
    s.split = rng() & 1;
    s.splitX = static_cast<uint16_t>(rng());
    for (int i = 0; i < s.count; i++) {
        old::Node& n = s.nodes[i];
        n.kind = static_cast<uint8_t>(rng() % old::kKindCount);
        n.enabled = (rng() % 8) != 0;
        n.intensity = static_cast<uint16_t>(rng() % 4 == 0 ? 65535 : rng());
        const old::KindInfo& info = old::Info(n.kind);
        for (int j = 0; j < info.paramCount; j++) {
            const old::Param& p = info.params[j];
            if (p.choices > 0) {
                n.q[j] = static_cast<int16_t>(rng() % p.choices);
            } else {
                int16_t q = static_cast<int16_t>(rng() % 65535 - 32767);
                if (rng() % 5 == 0) q = 0;
                if (q > 0 && p.hi == p.def) q = 0;
                if (q < 0 && p.lo == p.def) q = 0;
                n.q[j] = q;
            }
        }
    }
    *length = old::EncodeSection(s, section);
    old::Stack check;
    return old::DecodeSection(section, *length, &check) == KelirReject::kNone;
}

TEST(KelirParity, SectionsAndCodes) {
    std::mt19937 rng(1);
    uint8_t section[kMaxKelirSectionSize];
    int valid = 0;
    for (int i = 0; i < 20000; i++) {
        size_t length = 0;
        randomStack(rng, section, &length);
        // Corrupt a byte now and then, so rejections are compared too.
        if (i % 3 == 0) section[rng() % length] ^= static_cast<uint8_t>(1u << (rng() % 8));
        old::Stack a;
        now::Stack b;
        const KelirReject ra = old::DecodeSection(section, length, &a);
        const KelirReject rb = now::DecodeSection(section, length, &b);
        // A kind appended since is unknown to the old table only.
        bool appended = false;
        for (size_t n = kKelirHeaderSize; n + kKelirNodeSize <= length; n += kKelirNodeSize) {
            appended |= section[n] >= old::kKindCount && section[n] < now::kKindCount;
        }
        if (appended && ra == KelirReject::kKind) continue;
        ASSERT_EQ(ra, rb) << i;
        if (ra != KelirReject::kNone) continue;
        valid++;
        uint8_t ea[kMaxKelirSectionSize], eb[kMaxKelirSectionSize];
        ASSERT_EQ(old::EncodeSection(a, ea), now::EncodeSection(b, eb));
        ASSERT_EQ(0, memcmp(ea, eb, length));
    }
    EXPECT_GT(valid, 5000);
    for (int k = 0; k < old::kKindCount; k++) {
        const old::KindInfo& a = old::Info(k);
        const now::KindInfo& b = now::Info(k);
        ASSERT_STREQ(a.name, b.name);
        ASSERT_EQ(static_cast<int>(a.band), static_cast<int>(b.band));
        ASSERT_EQ(a.style, b.style);
        ASSERT_EQ(a.paramCount, b.paramCount);
        for (int j = 0; j < a.paramCount; j++) {
            const old::Param& pa = a.params[j];
            const now::Param& pb = b.params[j];
            ASSERT_STREQ(pa.name, pb.name);
            ASSERT_EQ(pa.choices, pb.choices);
            ASSERT_EQ(bits(pa.lo), bits(pb.lo));
            ASSERT_EQ(bits(pa.def), bits(pb.def));
            ASSERT_EQ(bits(pa.hi), bits(pb.hi));
            ASSERT_EQ(pa.needs, pb.needs);
            for (int q = -32768; q <= 32767; q += 7) {
                const auto c = static_cast<int16_t>(q);
                ASSERT_EQ(old::CodeValid(pa, c), now::CodeValid(pb, c));
                if (old::CodeValid(pa, c)) ASSERT_EQ(bits(old::Decode(pa, c)), bits(now::Decode(pb, c)));
            }
            for (int v = 0; v <= 1000; v++) {
                const float x = pa.lo - 1.0f + (pa.hi - pa.lo + 2.0f) * v / 1000.0f;
                ASSERT_EQ(old::Encode(pa, x), now::Encode(pb, x)) << a.name << " " << x;
            }
        }
    }
}

TEST(KelirParity, TransferAndHalves) {
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> any(-3.0f, 3.0f);
    for (int i = 0; i < 200000; i++) {
        const float x = any(rng);
        ASSERT_EQ(bits(old::Decode(x)), bits(now::Decode(x))) << x;
        ASSERT_EQ(bits(old::Encode(x)), bits(now::Encode(x))) << x;
        ASSERT_EQ(old::ToHalf(x * 1000.0f), now::ToHalf(x * 1000.0f));
    }
    for (uint32_t h = 0; h <= 0xffff; h++) {
        ASSERT_EQ(bits(old::FromHalf(static_cast<uint16_t>(h))), bits(now::FromHalf(static_cast<uint16_t>(h))));
    }
}

TEST(KelirParity, PlansLutsAndPacking) {
    std::mt19937 rng(3);
    uint8_t section[kMaxKelirSectionSize];
    int baked = 0;
    for (int i = 0; i < 400; i++) {
        size_t length = 0;
        if (!randomStack(rng, section, &length)) continue;
        old::Stack sa;
        now::Stack sb;
        ASSERT_EQ(old::DecodeSection(section, length, &sa), KelirReject::kNone);
        ASSERT_EQ(now::DecodeSection(section, length, &sb), KelirReject::kNone);
        const auto a = old::Compile(sa, nullptr);
        const auto b = now::Compile(sb, nullptr);
        ASSERT_EQ(a->detailCount, b->detailCount);
        ASSERT_EQ(a->colourCount, b->colourCount);
        ASSERT_EQ(a->screenCount, b->screenCount);
        ASSERT_EQ(a->sceneCount, b->sceneCount);
        ASSERT_EQ(0, memcmp(a->colourNode, b->colourNode, sizeof(a->colourNode)));
        for (int t = 0; t < 3; t++) {
            old::Target ta;
            now::Target tb;
            ta.role = static_cast<old::Role>(t);
            tb.role = static_cast<now::Role>(t);
            ta.fetch = tb.fetch = (i & 1) != 0;
            ASSERT_EQ(old::Signature(*a, ta), now::Signature(*b, tb));
            const old::Program pa = old::Unpack(old::Signature(*a, ta));
            const now::Program pb = now::Unpack(now::Signature(*b, tb));
            ASSERT_EQ(0, memcmp(pa.detail, pb.detail, sizeof(pa.detail)));
            ASSERT_EQ(0, memcmp(pa.screen, pb.screen, sizeof(pa.screen)));
            ASSERT_EQ(pa.lut, pb.lut);
            ASSERT_EQ(pa.sharpenPlus, pb.sharpenPlus);
        }
        float fa[old::kPlanVec4s * 4], fb[now::kPlanVec4s * 4];
        old::PackPlan(*a, fa);
        now::PackPlan(*b, fb);
        ASSERT_EQ(0, memcmp(fa, fb, sizeof(fa)));
        float ga[old::kSceneVec4s * 4], gb[now::kSceneVec4s * 4];
        old::PackScene(*a, ga);
        now::PackScene(*b, gb);
        ASSERT_EQ(0, memcmp(ga, gb, sizeof(ga)));
        old::FrameInfo ia;
        now::FrameInfo ib;
        ia.bufferWidth = ib.bufferWidth = 2772;
        ia.seed = ib.seed = rng();
        ia.flipY = ib.flipY = true;
        float ha[old::kFrameVec4s * 4], hb[now::kFrameVec4s * 4];
        old::PackFrame(*a, ia, ha);
        now::PackFrame(*b, ib, hb);
        ASSERT_EQ(0, memcmp(ha, hb, sizeof(ha)));
        for (int gen = 0; gen < 2; gen++) {
            for (int real = 0; real < 2; real++) {
                const old::Stages x = old::Schedule(a.get(), gen, real);
                const now::Stages y = now::Schedule(b.get(), gen, real);
                ASSERT_EQ(x.full, y.full);
                ASSERT_EQ(x.stageA, y.stageA);
                ASSERT_EQ(x.stageB, y.stageB);
            }
        }
        ASSERT_EQ(a->needsPyramid(), b->needsPyramid());
        ASSERT_EQ(a->needsSource(), b->needsSource());
        if (a->colourCount == 0 || baked >= 60) continue;
        baked++;
        for (int p = 0; p < 2; p++) {
            const old::Lut& la = a->luts->get(static_cast<old::Primaries>(p));
            const now::Lut& lb = b->luts->get(static_cast<now::Primaries>(p));
            ASSERT_EQ(la.linear, lb.linear);
            ASSERT_EQ(la.texels, lb.texels) << "stack " << i << " primaries " << p;
            uint8_t wa[kKelirSwatchColours * 3], wb[kKelirSwatchColours * 3];
            old::Swatch(a->colour, a->colourCount, static_cast<old::Primaries>(p), wa);
            now::Swatch(b->colour, b->colourCount, static_cast<now::Primaries>(p), wb);
            ASSERT_EQ(0, memcmp(wa, wb, sizeof(wa)));
            uint8_t ca[kKelirCurvePoints], cb[kKelirCurvePoints];
            old::Curve(a->colour[0], static_cast<old::Primaries>(p), ca);
            now::Curve(b->colour[0], static_cast<now::Primaries>(p), cb);
            ASSERT_EQ(0, memcmp(ca, cb, sizeof(ca)));
            const old::Rgb x = {0.3f, 0.55f, 0.9f};
            const now::Rgb y = {0.3f, 0.55f, 0.9f};
            const old::Rgb sa2 = old::SampleLut(la, x);
            const now::Rgb sb2 = now::SampleLut(lb, y);
            ASSERT_EQ(bits(sa2.r), bits(sb2.r));
            ASSERT_EQ(bits(sa2.g), bits(sb2.g));
            ASSERT_EQ(bits(sa2.b), bits(sb2.b));
        }
    }
    EXPECT_GT(baked, 20);
}

}  // namespace
}  // namespace framefx
