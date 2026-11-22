/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "kelir/shaders.h"

#include <iterator>

#include "framefx_kelir_shader_sources.h"
#include "framefx_shader_sources.h"
#include "kelir/plan.h"
#include "kelir/shaders/spirv.inc"

namespace framefx::kelir::shaders {
namespace {

namespace src = source;

constexpr uint32_t Fnv1a(uint32_t h, const char* s) {
    for (; *s != '\0'; s++) h = (h ^ static_cast<uint8_t>(*s)) * 0x01000193u;
    return h;
}

template <typename... Parts>
constexpr uint32_t Hash(const Parts&... parts) {
    uint32_t h = 0x811c9dc5u;
    ((h = Fnv1a(h, parts)), ...);
    return h;
}

// The Vulkan module's sources, in tools/gen_spirv.py's order.
static_assert(Hash(src::kSpecGlsl, src::kCommonGlsl, src::kResourcesGlsl, src::kSharpenGlsl,
                   src::kDetailsGlsl, src::kTiltshiftGlsl, src::kPainterlyGlsl,
                   src::kWatercolorGlsl, src::kVignetteGlsl, src::kGrainGlsl,
                   src::kLetterboxGlsl, src::kOldfilmGlsl, src::kRetroGlsl, src::kHalftoneGlsl,
                   src::kStageBGlsl, src::kFinishGlsl, src::kKelirFrag) == spirv::kKelirFragSource,
              "kelir/shaders changed: run tools/gen_spirv.py");
static_assert(Hash(src::kCommonGlsl, src::kPyramidFrag) == spirv::kPyramidFragSource,
              "kelir/shaders changed: run tools/gen_spirv.py");
static_assert(Hash(src::kSpecGlsl, src::kCommonGlsl, src::kVignetteGlsl, src::kGrainGlsl,
                   src::kLetterboxGlsl, src::kOldfilmGlsl, src::kRetroGlsl, src::kHalftoneGlsl,
                   src::kStageBGlsl, src::kFinishGlsl, luwes::source::kOutputFrag) ==
                      spirv::kLuwesFinishFragSource,
              "kelir/shaders or luwes/shaders/output.frag changed: run tools/gen_spirv.py");

constexpr Snippet kSnippets[] = {
        {kSharpen, "KELIR_HAS_SHARPEN", "kelirSharpen", src::kSharpenGlsl},
        {kDetails, "KELIR_HAS_DETAILS", "kelirDetails", src::kDetailsGlsl},
        {kTiltShift, "KELIR_HAS_TILT_SHIFT", "kelirTiltShift", src::kTiltshiftGlsl},
        {kPainterly, "KELIR_HAS_PAINTERLY", "kelirPainterly", src::kPainterlyGlsl},
        {kWatercolor, "KELIR_HAS_WATERCOLOR", "kelirWatercolor", src::kWatercolorGlsl},
        {kVignette, "KELIR_HAS_VIGNETTE", "kelirVignette", src::kVignetteGlsl},
        {kFilmGrain, "KELIR_HAS_FILM_GRAIN", "kelirFilmGrain", src::kGrainGlsl},
        {kLetterbox, "KELIR_HAS_LETTERBOX", "kelirLetterbox", src::kLetterboxGlsl},
        {kOldFilm, "KELIR_HAS_OLD_FILM", "kelirOldFilm", src::kOldfilmGlsl},
        {kRetro, "KELIR_HAS_RETRO", "kelirRetro", src::kRetroGlsl},
        {kHalftone, "KELIR_HAS_HALFTONE", "kelirHalftone", src::kHalftoneGlsl},
};

// Must match GLES_PRELUDE in tools/gen_spirv.py.
constexpr char kGlesPrelude[] = R"(precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;
#define KELIR_SAMPLER(set_, binding_) uniform
#define KELIR_PLAN uniform vec4 uKelirPlan[16]
#define PLAN(i) uKelirPlan[i]
#define KELIR_FRAME uniform vec4 uKelirFrame[4]
#define FRAME(i) uKelirFrame[i]
#define PARAMS uniform vec4 uParams[4]
#define P(i) uParams[i]
#define VERTEX_ID gl_VertexID
)";

constexpr const char* kSpecNames[kSpecCount] = {
        "KELIR_READS_LINEAR", "KELIR_WRITES_LINEAR", "KELIR_LUT_ON", "KELIR_D0",
        "KELIR_D1",           "KELIR_D2",            "KELIR_SHARPEN_PLUS", "KELIR_S0",
        "KELIR_S1",           "KELIR_S2",            "KELIR_S3",         "KELIR_ROLE",
};

std::string constant(const char* name, int value) {
    return std::string("const int ") + name + " = " + std::to_string(value) + ";\n";
}

std::string version(bool fetch) {
    std::string s = "#version 300 es\n";
    if (fetch) s += "#extension GL_EXT_shader_framebuffer_fetch : require\n";
    return s + kGlesPrelude;
}

}  // namespace

const Snippet* Snippets(size_t* count) {
    *count = std::size(kSnippets);
    return kSnippets;
}

std::string GlesVertex() {
    return version(false) + luwes::source::kFullscreenVert;
}

std::string GlesPyramid() {
    return version(false) + src::kCommonGlsl + src::kPyramidFrag;
}

std::string GlesKelir(uint64_t signature) {
    const Program p = Unpack(signature);
    int32_t spec[kSpecCount];
    SpecValues(signature, spec);
    std::string s = version(p.target.fetch);
    if (p.target.fetch) s += "#define KELIR_FETCH\n";
    for (int i = 0; i < kSpecCount; i++) s += constant(kSpecNames[i], spec[i]);
    for (const Snippet& snippet : kSnippets) {
        if (p.has(snippet.kind)) s += std::string("#define ") + snippet.define + "\n";
    }
    s += src::kCommonGlsl;
    s += src::kResourcesGlsl;
    for (const Snippet& snippet : kSnippets) {
        if (p.has(snippet.kind)) s += snippet.source;
    }
    s += src::kStageBGlsl;
    s += src::kFinishGlsl;
    s += src::kKelirFrag;
    return s;
}

std::string GlesFinish(uint64_t signature) {
    const Program p = Unpack(signature);
    int32_t spec[kSpecCount];
    SpecValues(signature, spec);
    std::string s = kGlesPrelude;
    s += "#define KELIR_FINISH\n";
    for (int i = 0; i < kSpecCount; i++) s += constant(kSpecNames[i], spec[i]);
    for (const Snippet& snippet : kSnippets) {
        if (p.has(snippet.kind)) s += std::string("#define ") + snippet.define + "\n";
    }
    s += src::kCommonGlsl;
    for (const Snippet& snippet : kSnippets) {
        if (p.has(snippet.kind)) s += snippet.source;
    }
    s += src::kStageBGlsl;
    s += src::kFinishGlsl;
    return s;
}

void SpecValues(uint64_t signature, int32_t out[kSpecCount]) {
    const Program p = Unpack(signature);
    out[0] = p.target.readsLinear ? 1 : 0;
    out[1] = p.target.format == FormatClass::kSrgb8 ? 1 : 0;
    out[2] = p.lut ? 1 : 0;
    for (int i = 0; i < kMaxDetail; i++) out[3 + i] = p.detail[i];
    out[6] = p.sharpenPlus ? 1 : 0;
    for (int i = 0; i < kMaxScreen; i++) out[7 + i] = p.screen[i];
    out[11] = static_cast<int32_t>(p.target.role);
}

template <size_t N>
constexpr Spirv Module(const uint32_t (&code)[N]) {
    return {code, N * sizeof(uint32_t)};
}

const Spirv kKelirFragSpirv = Module(spirv::kKelirFrag);
const Spirv kPyramidFragSpirv = Module(spirv::kPyramidFrag);
const Spirv kLuwesFinishFragSpirv = Module(spirv::kLuwesFinishFrag);

}  // namespace framefx::kelir::shaders
