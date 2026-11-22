/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Kelir's shaders. One GLSL source per kind in kelir/shaders serves both APIs: a GLES program
// is assembled at run time from the snippets its signature names, with the signature as
// constants; Vulkan takes one SPIR-V module of every snippet that tools/gen_spirv.py made, and
// specialises it per signature.
namespace framefx::kelir::shaders {

struct Snippet {
    int kind;
    const char* define;  // KELIR_HAS_<KIND>, guarding its dispatch
    const char* function;
    const char* source;
};

// Every kind outside the colour band and the snippet that implements it.
const Snippet* Snippets(size_t* count);

std::string GlesVertex();
std::string GlesPyramid();
// The Kelir pass for a signature: its snippets only.
std::string GlesKelir(uint64_t signature);

// What a pass of frame generation puts before its own source to finish every frame it writes like
// Kelir would: `kelirFinish(rgb, px)` for a stage-B signature. Version line not included.
std::string GlesFinish(uint64_t signature);

struct Spirv {
    const uint32_t* code;
    size_t size;  // bytes
};

extern const Spirv kKelirFragSpirv;
extern const Spirv kPyramidFragSpirv;
// Luwes's output pass with Kelir's finish: Luwes's set 0 and push bytes, Kelir's set 1 and frame
// block, specialised by a stage-B signature.
extern const Spirv kLuwesFinishFragSpirv;

// Specialisation constants 64..64 + kSpecCount - 1 for a signature, in ID order.
inline constexpr uint32_t kFirstSpecId = 64;
inline constexpr int kSpecCount = 12;
void SpecValues(uint64_t signature, int32_t out[kSpecCount]);

// Push constants, Vulkan: Kelir's frame block sits after the 64 bytes a pass of its own or Luwes's
// output pass uses.
inline constexpr uint32_t kFramePushOffset = 64;
inline constexpr uint32_t kFramePushSize = 64;

}  // namespace framefx::kelir::shaders
