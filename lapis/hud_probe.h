/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

#include "lapis/hud.h"

// A low-resolution luma copy of the presented frame for Hide HUD's cross-check: two linear blits
// and a copy to host memory, recorded into the game's command buffer as the pass that draws the
// presented image ends, read at a later present once the GPU has written it.
namespace framefx::lapis::hud_probe {

void OnDevice(VkDevice device, PFN_vkGetDeviceProcAddr get,
              const VkPhysicalDeviceMemoryProperties& memory);
void OnDestroyDevice(VkDevice device);

// At present: the pass whose end leaves the presented image (render pass handle bits, 0 for
// dynamic rendering), its target's format, size and layout there. The next frames copy it.
void Arm(VkDevice device, uint64_t renderPass, VkFormat format, uint32_t width, uint32_t height,
         VkImageLayout final);
void Disarm();

// At a pass's end, cheap: whether this frame still wants the copy, from a pass of this shape.
bool Wants(uint64_t renderPass, VkFormat format, uint32_t width, uint32_t height);
// Records the copy of `image` (in the armed layout) into `cb`, once per frame.
void Record(VkCommandBuffer cb, VkImage image);

// At present: the camera of the frame being presented, then the oldest grid the GPU finished and
// was not taken yet, so that grids come in order. `grid` points into the probe until the next call.
void Presented(bool hasCamera, const float* camera);
bool Take(HudGrid* grid);

}  // namespace framefx::lapis::hud_probe
