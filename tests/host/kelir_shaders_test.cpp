/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <random>
#include <regex>
#include <set>
#include <string>

#include "framefx_kelir_shader_sources.h"
#include "kelir/plan.h"
#include "kelir/shaders.h"
#include "kelir/shaders/spirv.inc"

namespace framefx::kelir {
namespace {

uint32_t Fnv1a(uint32_t h, const std::string& s) {
    for (unsigned char c : s) h = (h ^ c) * 0x01000193u;
    return h;
}

uint64_t RandomSignature(std::mt19937& rng) {
    Plan plan;
    std::vector<uint8_t> detail = {kSharpen, kDetails, kTiltShift, kPainterly, kWatercolor};
    std::shuffle(detail.begin(), detail.end(), rng);
    bool style = false;
    for (uint8_t k : detail) {
        if (plan.detailCount == kMaxDetail || rng() % 2 == 0) continue;
        if (Info(k).style && style) continue;
        style |= Info(k).style;
        plan.detail[plan.detailCount].kind = k;
        plan.detail[plan.detailCount++].v[p::kQuality] = static_cast<float>(rng() % 2);
    }
    std::vector<uint8_t> screen = {kVignette, kFilmGrain, kLetterbox, kOldFilm, kRetro, kHalftone};
    std::shuffle(screen.begin(), screen.end(), rng);
    for (uint8_t k : screen) {
        if (plan.screenCount < kMaxScreen && rng() % 2 == 0) plan.screen[plan.screenCount++].kind = k;
    }
    plan.colourCount = static_cast<int>(rng() % 2);
    Target t;
    t.format = static_cast<FormatClass>(rng() % 3);
    t.readsLinear = rng() % 2 == 0;
    t.fetch = !plan.needsSource() && rng() % 2 == 0;
    return Signature(plan, t);
}

// ---- KelirShaderSources ---------------------------------------------------------------------

TEST(KelirShaderSources, GeneratedSpirvNotStale) {
    namespace src = source;
    std::string module;
    for (const char* part :
         {src::kSpecGlsl, src::kCommonGlsl, src::kResourcesGlsl, src::kSharpenGlsl,
          src::kDetailsGlsl, src::kTiltshiftGlsl, src::kPainterlyGlsl, src::kWatercolorGlsl,
          src::kVignetteGlsl, src::kGrainGlsl, src::kLetterboxGlsl, src::kOldfilmGlsl,
          src::kRetroGlsl, src::kHalftoneGlsl, src::kStageBGlsl, src::kFinishGlsl, src::kKelirFrag}) {
        module += part;
    }
    EXPECT_EQ(spirv::kKelirFragSource, Fnv1a(0x811c9dc5u, module));
    EXPECT_EQ(spirv::kPyramidFragSource,
              Fnv1a(0x811c9dc5u, std::string(src::kCommonGlsl) + src::kPyramidFrag));
    EXPECT_EQ(0x07230203u, shaders::kKelirFragSpirv.code[0]);
    EXPECT_EQ(0x07230203u, shaders::kPyramidFragSpirv.code[0]);
}

// Every kind outside the colour band has a snippet, every snippet is dispatched, and the Vulkan
// module holds them all.
TEST(KelirShaderSources, EverySnippetReachable) {
    size_t count = 0;
    const shaders::Snippet* snippets = shaders::Snippets(&count);
    std::set<int> kinds;
    const std::string dispatch = std::string(source::kKelirFrag) +
            source::kStageBGlsl;
    for (size_t i = 0; i < count; i++) {
        const shaders::Snippet& s = snippets[i];
        kinds.insert(s.kind);
        EXPECT_NE(Band::kColour, Info(s.kind).band) << s.function;
        EXPECT_NE(std::string::npos, std::string(s.source).find(std::string("vec3 ") + s.function))
                << s.function;
        EXPECT_NE(std::string::npos, dispatch.find(std::string("#ifdef ") + s.define)) << s.define;
        EXPECT_NE(std::string::npos, dispatch.find(std::string(s.function) + "(slot")) << s.function;
        // Some signature takes it into a GLES program.
        Plan plan;
        if (Info(s.kind).band == Band::kDetail) {
            plan.detail[plan.detailCount++].kind = static_cast<uint8_t>(s.kind);
        } else {
            plan.screen[plan.screenCount++].kind = static_cast<uint8_t>(s.kind);
        }
        EXPECT_NE(std::string::npos,
                  shaders::GlesKelir(Signature(plan, Target())).find(std::string("vec3 ") + s.function));
    }
    for (int k = 0; k < kKindCount; k++) {
        const bool drawn = Info(k).band == Band::kDetail || Info(k).band == Band::kScreen;
        EXPECT_EQ(drawn, kinds.count(k) == 1) << Info(k).name;
    }
}

TEST(KelirShaderSources, GlesSourceHasOnlySignatureKinds) {
    size_t count = 0;
    const shaders::Snippet* snippets = shaders::Snippets(&count);
    std::mt19937 rng(11);
    for (int i = 0; i < 500; i++) {
        const uint64_t sig = RandomSignature(rng);
        const Program p = Unpack(sig);
        const std::string source = shaders::GlesKelir(sig);
        for (size_t j = 0; j < count; j++) {
            const bool defined = source.find(std::string("vec3 ") + snippets[j].function + "(int") !=
                    std::string::npos;
            ASSERT_EQ(p.has(snippets[j].kind), defined) << snippets[j].function << " " << sig;
        }
        ASSERT_EQ(p.target.fetch, source.find("GL_EXT_shader_framebuffer_fetch") != std::string::npos);
        ASSERT_EQ(0u, source.find("#version 300 es\n"));
    }
}

TEST(KelirShaderSources, SpecIdsInKelirRange) {
    EXPECT_GE(shaders::kFirstSpecId, 64u);
    EXPECT_LE(shaders::kFirstSpecId + shaders::kSpecCount - 1, 79u);
    const std::string spec = source::kSpecGlsl;
    const std::regex id("constant_id = (\\d+)");
    int seen = 0;
    for (auto it = std::sregex_iterator(spec.begin(), spec.end(), id); it != std::sregex_iterator();
         ++it) {
        const int v = std::stoi((*it)[1]);
        EXPECT_EQ(static_cast<int>(shaders::kFirstSpecId) + seen, v);
        seen++;
    }
    EXPECT_EQ(shaders::kSpecCount, seen);
}

TEST(KelirShaderSources, SpecValuesFollowTheSignature) {
    Plan plan;
    plan.detail[plan.detailCount].kind = kSharpen;
    plan.detail[plan.detailCount++].v[p::kQuality] = kPlus;
    plan.screen[plan.screenCount++].kind = kHalftone;
    plan.colourCount = 1;
    Target t;
    t.format = FormatClass::kSrgb8;
    t.readsLinear = true;
    int32_t v[shaders::kSpecCount];
    shaders::SpecValues(Signature(plan, t), v);
    const int32_t want[] = {1, 1, 1, kSharpen, -1, -1, 1, kHalftone, -1, -1, -1, 0};
    for (int i = 0; i < shaders::kSpecCount; i++) EXPECT_EQ(want[i], v[i]) << i;
    t.role = Role::kStageA;
    shaders::SpecValues(Signature(plan, t), v);
    EXPECT_EQ(static_cast<int32_t>(Role::kStageA), v[11]);
}

TEST(KelirShaderSources, PushConstantsWithin128) {
    EXPECT_LE(shaders::kFramePushOffset + shaders::kFramePushSize, 128u);
}

}  // namespace
}  // namespace framefx::kelir
