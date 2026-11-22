/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "shaders.h"

#include "framefx_shader_sources.h"
#include "shaders/spirv.inc"

namespace framefx::luwes::shaders {
namespace {

constexpr uint32_t Fnv1a(const char* s, size_t n) {
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < n; i++) h = (h ^ static_cast<uint8_t>(s[i])) * 0x01000193u;
    return h;
}

template <size_t N>
constexpr uint32_t Hash(const char (&source)[N]) {
    return Fnv1a(source, N - 1);
}

#define FRESH(text, module)                                    \
    static_assert(Hash(source::text) == spirv::module##Source, \
                  "luwes/shaders changed: run tools/gen_spirv.py")

FRESH(kFullscreenVert, kFullscreenVert);
FRESH(kPyramidFrag, kPyramidFrag);
FRESH(kMotionComp, kMotionComp);
FRESH(kOutputFrag, kOutputFrag);
FRESH(kUpscaleFrag, kUpscaleFrag);
FRESH(kHudFrag, kHudFrag);
FRESH(kDepthFrag, kDepthFrag);
FRESH(kAoFrag, kAoFrag);
FRESH(kFocusFrag, kFocusFrag);
FRESH(kSceneFrag, kSceneFrag);
FRESH(kSunFrag, kSunFrag);
FRESH(kRaysComp, kRaysComp);
FRESH(kPositionsComp, kPositionsComp);
FRESH(kHistoryComp, kHistoryComp);
FRESH(kTemporalFrag, kTemporalFrag);
FRESH(kTemporalUpFrag, kTemporalUpFrag);
FRESH(kGiFrag, kGiFrag);
FRESH(kTransposeFrag, kTransposeFrag);
FRESH(kYcbcrFrag, kYcbcrFrag);
FRESH(kHintFrag, kHintFrag);
FRESH(kDenseFrag, kDenseFrag);
FRESH(kSplatComp, kSplatComp);

template <size_t N>
constexpr Spirv Module(const uint32_t (&code)[N]) {
    return {code, N * sizeof(uint32_t)};
}

}  // namespace

// Must match GLES_PRELUDE in tools/gen_spirv.py.
const char kGlesPrelude[] = R"(#version 320 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp image2D;
#define BINDING(vk, gl) binding = gl
#define VERTEX_ID gl_VertexID
#define PARAMS uniform vec4 uParams[4]
#define PARAMS8 uniform vec4 uParams[8]
#define P(i) uParams[i]
)";

const char* const kFullscreenVert = source::kFullscreenVert;
const char* const kPyramidFrag = source::kPyramidFrag;
const char* const kMotionComp = source::kMotionComp;
const char* const kOutputFrag = source::kOutputFrag;
const char* const kUpscaleFrag = source::kUpscaleFrag;
const char* const kHudFrag = source::kHudFrag;
const char* const kDepthFrag = source::kDepthFrag;
const char* const kAoFrag = source::kAoFrag;
const char* const kFocusFrag = source::kFocusFrag;
const char* const kSceneFrag = source::kSceneFrag;
const char* const kSunFrag = source::kSunFrag;
const char* const kRaysComp = source::kRaysComp;
const char* const kPositionsComp = source::kPositionsComp;
const char* const kHistoryComp = source::kHistoryComp;
const char* const kTemporalFrag = source::kTemporalFrag;
const char* const kTemporalUpFrag = source::kTemporalUpFrag;
const char* const kGiFrag = source::kGiFrag;
const char* const kTransposeFrag = source::kTransposeFrag;
const char* const kYcbcrFrag = source::kYcbcrFrag;
const char* const kHintFrag = source::kHintFrag;
const char* const kDenseFrag = source::kDenseFrag;
const char* const kSplatComp = source::kSplatComp;

const Spirv kFullscreenVertSpirv = Module(spirv::kFullscreenVert);
const Spirv kPyramidFragSpirv = Module(spirv::kPyramidFrag);
const Spirv kMotionCompSpirv = Module(spirv::kMotionComp);
const Spirv kOutputFragSpirv = Module(spirv::kOutputFrag);
const Spirv kUpscaleFragSpirv = Module(spirv::kUpscaleFrag);
const Spirv kHudFragSpirv = Module(spirv::kHudFrag);
const Spirv kDepthFragSpirv = Module(spirv::kDepthFrag);
const Spirv kAoFragSpirv = Module(spirv::kAoFrag);
const Spirv kFocusFragSpirv = Module(spirv::kFocusFrag);
const Spirv kSceneFragSpirv = Module(spirv::kSceneFrag);
const Spirv kSunFragSpirv = Module(spirv::kSunFrag);
const Spirv kRaysCompSpirv = Module(spirv::kRaysComp);
const Spirv kPositionsCompSpirv = Module(spirv::kPositionsComp);
const Spirv kHistoryCompSpirv = Module(spirv::kHistoryComp);
const Spirv kTemporalFragSpirv = Module(spirv::kTemporalFrag);
const Spirv kTemporalUpFragSpirv = Module(spirv::kTemporalUpFrag);
const Spirv kGiFragSpirv = Module(spirv::kGiFrag);
const Spirv kTransposeFragSpirv = Module(spirv::kTransposeFrag);
const Spirv kYcbcrFragSpirv = Module(spirv::kYcbcrFrag);
const Spirv kHintFragSpirv = Module(spirv::kHintFrag);
const Spirv kDenseFragSpirv = Module(spirv::kDenseFrag);
const Spirv kSplatCompSpirv = Module(spirv::kSplatComp);

}  // namespace framefx::luwes::shaders
