/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <memory>

#include "kelir/plan.h"

// Kelir on Vulkan, native on the game's device. The game's image is never written: one pass reads
// it and writes a spare image of the same swapchain, which is presented in its place, and the
// game's image goes back to the queue unpresented. One submission per present, ordered by
// semaphores; the CPU never waits on the GPU in steady state.
namespace framefx::kelir::vulkan {

void attach(const framefx_vk_host* host);

void onDevice(const framefx_vk_instance* instance, VkPhysicalDevice physical, VkDevice device);
void destroyDevice(VkDevice device);
void getDeviceQueue(VkDevice device, uint32_t family, VkQueue queue);

// The creation Kelir needs of a swapchain it can take: the usage of its pass, and one image of its
// own unless one is already `reserved`. False, and `info` untouched, for one it cannot take.
bool adjustSwapchain(VkDevice device, VkSwapchainCreateInfoKHR* info, bool reserved);
// `asked` is the creation as the game asked for it; `spare` whether the swapchain holds an image
// beyond the game's.
void onSwapchain(VkDevice device, VkSwapchainKHR swapchain, const VkSwapchainCreateInfoKHR* asked,
                 bool spare);
void destroySwapchain(VkDevice device, VkSwapchainKHR swapchain);

struct Claim {
    VkDevice device = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    void* target = nullptr;
};

// Whether Kelir presents this frame: a stack to apply, and frame generation off unless it passes
// this frame on (`fromLuwes`).
bool claim(VkQueue queue, const VkPresentInfoKHR* info, Claim* out, bool fromLuwes = false);
VkResult present(VkQueue queue, const VkPresentInfoKHR* info, const Claim& claim);

// ---- With frame generation: Luwes presents, Kelir works inside its passes ----

// One real frame Luwes processes with Kelir on.
struct Stage {
    void* target = nullptr;
    int ubo = -1;
    VkPipeline stageA = VK_NULL_HANDLE;  // null when stage A has nothing to do
    uint64_t serial = 0;
    uint64_t signatureB = 0;  // what Luwes's output pass specialises for
};

// Plan constants, LUT and the stage-A pipeline for the frame; false when Kelir has nothing to
// apply to this swapchain. `family` is the present queue's.
bool beginStage(VkDevice device, VkSwapchainKHR swapchain, uint32_t family,
                const std::shared_ptr<const Plan>& plan, Stage* out);

// Records stage A of the game's image `app` into `view` (of `image`, the swapchain's format and
// size), left in SHADER_READ_ONLY_OPTIMAL; the game's image goes back to PRESENT_SRC_KHR.
void recordStageA(const Stage& stage, VkCommandBuffer cb, uint32_t app, VkImage image,
                  VkImageView view);

// Records stage A of `source` (readable, the swapchain's format and size) into `view` the same
// way, left in SHADER_READ_ONLY_OPTIMAL. `edges`: the source's alpha marks depth edges, where
// Sharpen holds back. False when it cannot take another source.
bool recordStageAFrom(const Stage& stage, VkCommandBuffer cb, VkImageView source, VkImage image,
                      VkImageView view, bool edges = false);

// For Luwes's output pass with the finish: Kelir's set 1, and the frame constants it pushes at
// shaders::kFramePushOffset, a new seed on every call.
VkDescriptorSet planSet(const Stage& stage);
void frameConstants(const Stage& stage, float out[kFrameVec4s * 4]);
VkDescriptorSetLayout planLayout(VkDevice device);

// Luwes's frame that carried the stage retired on the GPU; `gpuNs`, when not 0, is what Kelir's
// work cost in it.
void stageRetired(VkDevice device, VkSwapchainKHR swapchain, uint64_t serial, uint32_t gpuNs = 0);
// The caller destroyed the views it handed to recordStageA and recordStageAFrom, with no frame
// in flight: a view made later may carry the same handle, and nothing kept by view may outlive it.
void forgetViews(VkDevice device, VkSwapchainKHR swapchain);

// ---- Without a swapchain: a video's frames, into images the caller owns ----

// A target of `format` and `extent` under `key`, which stands where a swapchain's handle would and
// goes through the calls above: destroySwapchain, recordStageAFrom and stageRetired. `turns` are the
// quarter turns from its images to the display.
void onOffscreen(VkDevice device, VkSwapchainKHR key, VkFormat format, VkExtent2D extent, int turns);

// What a pass of `plan` is compiled for, on an offscreen target. kFull when nothing is generated;
// with generation, kStageA once per real frame and kStageB on each present.
uint64_t offscreenSignature(const Plan& plan, Role role);
// Whether that pass is built, and building it. Compiling takes long enough to be kept off the
// frames: until it is built, they stay on the plan before.
bool offscreenReady(VkDevice device, VkFormat format, uint64_t signature);
bool offscreenCompile(VkDevice device, VkFormat format, uint64_t signature);

// The plan constants, LUT and the pass of `role` in Stage::stageA, for recordOffscreen. False when
// the target cannot run `plan` now.
bool beginOffscreen(VkDevice device, VkSwapchainKHR key, uint32_t family,
                    const std::shared_ptr<const Plan>& plan, Role role, Stage* out);

// Draws that pass from `source` (readable, the target's format and size) into `view`, left in
// SHADER_READ_ONLY_OPTIMAL.
bool recordOffscreen(const Stage& stage, VkCommandBuffer cb, VkImageView source, VkImageView view,
                     Role role);

}  // namespace framefx::kelir::vulkan
