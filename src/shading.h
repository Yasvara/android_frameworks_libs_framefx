/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <cstdint>

#include "src/channel.h"

// Variable rate shading of the game's depth-tested work. A game that starts with shading on gets
// the shading-rate feature on its device and, on every pipeline that tests depth into a depth
// attachment, the rate as dynamic state; each bind of such a pipeline then sets the frame's rate.
// Pipelines without depth (UI, post-processing) keep full rate. On GLES the rate is coarse while
// GL_DEPTH_TEST is on. The rate follows Pentas live; adaptive mode keeps full rate while the game
// holds its own pace (lapis/rust/src/shading.rs).
namespace framefx::shading {

// The fragment size of a rate code: 0 1x1, 1 2x1, 2 2x2, 3 4x4.
VkExtent2D sizeOf(int rate);

// --- Vulkan ---------------------------------------------------------------------------------

// `granted`: the device was made with pipelineFragmentShadingRate (extensions::DeviceRequest).
void onDevice(const framefx_vk_instance& in, VkPhysicalDevice physical, VkDevice device,
              bool granted);
void onDestroyDevice(VkDevice device);
// The wrapper for a command shading watches, around `next`; nullptr keeps `next`. Only in a game
// that started with shading on.
PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next, bool shadingOn);

// --- Both APIs --------------------------------------------------------------------------------

// A real frame of the game presented at `nowNs`: picks the rate its next frame shades at.
void onPresent(const TextureSettings& settings, int64_t nowNs, int64_t refreshNs);
// The rate code in force: 0 full.
int rate();

// --- GLES -------------------------------------------------------------------------------------

void attachGl(const framefx_gl_host* host);
// The calling thread presents the game's GLES frames: its depth-tested draws may shade coarse.
void onGlPresent();
void* interceptGl(const char* name, void* next, bool shadingOn);

}  // namespace framefx::shading
