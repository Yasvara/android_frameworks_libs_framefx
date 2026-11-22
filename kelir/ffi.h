/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "kelir/colour.h"
#include "kelir/params.h"

// The C ABI of Kelir's logic in kelir/rust (src/ffi.rs). Every struct crossing it is laid out the
// same on both sides; kelir_layout_test holds the sizes to Rust's.
namespace framefx::kelir {

struct Stack;
struct PlanCore;
struct Program;
struct FrameInfo;
struct Stages;
struct Target;
struct ControllerCore;

struct ControllerUpdate {
    uint8_t action;  // kControllerKeep, kControllerOff, kControllerCompile
    bool changed;
};
inline constexpr uint8_t kControllerKeep = 0;
inline constexpr uint8_t kControllerOff = 1;
inline constexpr uint8_t kControllerCompile = 2;

// Rgb has member initializers but is trivially copyable: it passes as the plain C struct it mirrors.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern "C" {
const KindInfo* framefx_kelir_info(int32_t kind);
bool framefx_kelir_code_valid(const Param* p, int16_t q);
float framefx_kelir_param_decode(const Param* p, int16_t q);
int16_t framefx_kelir_param_encode(const Param* p, float value);
uint8_t framefx_kelir_decode_section(const uint8_t* data, size_t length, Stack* out);
size_t framefx_kelir_encode_section(const Stack* stack, uint8_t* out);
const float* framefx_kelir_luma_weights(uint8_t primaries);
float framefx_kelir_srgb_decode(float encoded);
float framefx_kelir_srgb_encode(float linear);
Rgb framefx_kelir_apply_colour(const Step* step, Rgb x, uint8_t primaries);
Rgb framefx_kelir_blend(const Step* step, Rgb x, uint8_t primaries);
bool framefx_kelir_is_identity_colour(const Step* step);
Rgb framefx_kelir_evaluate(const Step* steps, int32_t count, uint8_t primaries, Rgb x);
bool framefx_kelir_bake_lut(const Step* steps, int32_t count, uint8_t primaries, uint16_t* out);
Rgb framefx_kelir_sample_lut(const uint16_t* texels, bool linear, Rgb x);
uint16_t framefx_kelir_to_half(float v);
float framefx_kelir_from_half(uint16_t h);
void framefx_kelir_swatch(const Step* steps, int32_t count, uint8_t primaries, uint8_t* out);
void framefx_kelir_curve(const Step* step, uint8_t primaries, uint8_t* out);
void framefx_kelir_compile(const Stack* stack, PlanCore* out);
bool framefx_kelir_plan_has(const PlanCore* plan, uint8_t kind);
int32_t framefx_kelir_scene_step(const PlanCore* plan, uint8_t kind);
bool framefx_kelir_needs_pyramid(const PlanCore* plan);
bool framefx_kelir_needs_source(const PlanCore* plan);
bool framefx_kelir_same_colour_band(const PlanCore* a, const PlanCore* b);
uint64_t framefx_kelir_signature(const PlanCore* plan, const Target* target);
void framefx_kelir_unpack(uint64_t signature, Program* out);
void framefx_kelir_pack_plan(const PlanCore* plan, float* out);
void framefx_kelir_pack_scene(const PlanCore* plan, float* out);
bool framefx_kelir_pack_scene_look(const PlanCore* plan, float* out);
bool framefx_kelir_pack_scene_rays(const PlanCore* plan, float* out);
void framefx_kelir_pack_frame(const PlanCore* plan, const FrameInfo* frame, float* out);
void framefx_kelir_schedule(const PlanCore* plan, bool generation, bool real, Stages* out);
ControllerUpdate framefx_kelir_controller_update(ControllerCore* c, const uint8_t* data,
                                                 size_t length, bool planEmpty);
// Sizes of KindInfo, Stack, Step, PlanCore, Program, FrameInfo, ControllerCore, Target.
void framefx_kelir_sizes(size_t out[8]);
}
#pragma clang diagnostic pop

}  // namespace framefx::kelir
