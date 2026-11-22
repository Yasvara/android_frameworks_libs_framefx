/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

// The Luwes shaders. One GLSL source per pass in luwes/shaders serves both APIs: GLES compiles
// it at run time behind kGlesPrelude, Vulkan takes the SPIR-V tools/gen_spirv.py made from it.
namespace framefx::luwes::shaders {

extern const char kGlesPrelude[];

extern const char* const kFullscreenVert;
extern const char* const kPyramidFrag;
extern const char* const kMotionComp;
extern const char* const kOutputFrag;
extern const char* const kUpscaleFrag;
extern const char* const kHudFrag;
extern const char* const kDepthFrag;
extern const char* const kAoFrag;
extern const char* const kFocusFrag;
extern const char* const kSceneFrag;
extern const char* const kSunFrag;
extern const char* const kRaysComp;
extern const char* const kPositionsComp;
extern const char* const kHistoryComp;
extern const char* const kTemporalFrag;
extern const char* const kTemporalUpFrag;
extern const char* const kGiFrag;
extern const char* const kTransposeFrag;
extern const char* const kYcbcrFrag;
extern const char* const kHintFrag;
extern const char* const kDenseFrag;
extern const char* const kSplatComp;

struct Spirv {
    const uint32_t* code;
    size_t size;  // bytes
};

extern const Spirv kFullscreenVertSpirv;
extern const Spirv kPyramidFragSpirv;
extern const Spirv kMotionCompSpirv;
extern const Spirv kOutputFragSpirv;
extern const Spirv kUpscaleFragSpirv;
extern const Spirv kHudFragSpirv;
extern const Spirv kDepthFragSpirv;
extern const Spirv kAoFragSpirv;
extern const Spirv kFocusFragSpirv;
extern const Spirv kSceneFragSpirv;
extern const Spirv kSunFragSpirv;
// Ray query: built only on a device made with it (extensions::Granted::rays).
extern const Spirv kRaysCompSpirv;
extern const Spirv kPositionsCompSpirv;
extern const Spirv kHistoryCompSpirv;
extern const Spirv kTemporalFragSpirv;
extern const Spirv kTemporalUpFragSpirv;
extern const Spirv kGiFragSpirv;
extern const Spirv kTransposeFragSpirv;
extern const Spirv kYcbcrFragSpirv;
extern const Spirv kHintFragSpirv;
extern const Spirv kDenseFragSpirv;
extern const Spirv kSplatCompSpirv;

}  // namespace framefx::luwes::shaders
