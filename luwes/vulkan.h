/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <cstdint>
#include <vector>

// Luwes on Vulkan, native on the game's own device: motion from a block search on a luma pyramid
// in compute, generated frames rendered into images acquired from the game's swapchain.
//
// No hidden GLES context and no interop: every pass is recorded on the game's present queue and
// ordered by semaphores and barriers, so the CPU never waits on the GPU in steady state.
namespace framefx::luwes::vulkan {

void attach(const framefx_vk_host* host);
VkResult createDevice(const framefx_vk_instance* instance, VkPhysicalDevice physical,
                      const VkDeviceCreateInfo* info, const VkAllocationCallbacks* allocator,
                      VkDevice* device);
void destroyDevice(VkDevice device);
void getDeviceQueue(VkDevice device, uint32_t family, VkQueue queue);
// Whether Luwes takes the swapchain: then it reserves at least one image beyond the game's.
bool accepts(VkDevice device, const VkSwapchainCreateInfoKHR* info);
VkResult createSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                         const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain);
// Images the swapchain holds beyond what the game asked for.
uint32_t spareImages(VkDevice device, VkSwapchainKHR swapchain);
// Luwes presents the swapchain's frames, or has to release what it held.
bool running(VkDevice device, VkSwapchainKHR swapchain);
// This swapchain's frame wants the scene layer now (the game's 3D upscaled, graded apart from its
// HUD, through Kelir's scene band, or shown alone), which Luwes does even with frame generation
// off.
bool layers(VkDevice device, VkSwapchainKHR swapchain);
void destroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
                      const VkAllocationCallbacks* allocator);
VkResult queuePresent(VkQueue queue, const VkPresentInfoKHR* info);

// A game drawing at another size than the display's (its own resolution setting, or a render
// scale) draws into images of its own, and each frame is resampled into a swapchain at the
// display's size: upscaled with the scene under its HUD as sharp as Luwes makes it, or
// downsampled with a proper filter. The wrapper for a swapchain entry point that has to see
// through that, or nullptr to keep `next`.
PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next);
// The images the game draws into, when they are not the swapchain's own, and the size shown.
bool gameImages(VkDevice device, VkSwapchainKHR swapchain, std::vector<VkImage>* images,
                VkExtent2D* shown);
// Drains in-flight Luwes work referencing `imageId` and drops cached views of it before the game
// destroys the image or frees its backing memory.
void releaseImage(VkDevice device, uint32_t imageId, VkImage image = VK_NULL_HANDLE);

}  // namespace framefx::luwes::vulkan
