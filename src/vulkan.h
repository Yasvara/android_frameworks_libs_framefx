/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

// The Vulkan side of an armed process: each swapchain is created with what both features need,
// Luwes presents while it generates frames, and Kelir redirects the frame the game presents
// otherwise.
namespace framefx::vulkan {

void attach(const framefx_vk_host* host);
VkResult createDevice(const framefx_vk_instance* instance, VkPhysicalDevice physical,
                      const VkDeviceCreateInfo* info, const VkAllocationCallbacks* allocator,
                      VkDevice* device);
void destroyDevice(VkDevice device);
void getDeviceQueue(VkDevice device, uint32_t family, VkQueue queue);
VkResult createSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                         const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain);
void destroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
                      const VkAllocationCallbacks* allocator);
VkResult queuePresent(VkQueue queue, const VkPresentInfoKHR* info);
PFN_vkVoidFunction intercept(VkDevice device, const char* name, PFN_vkVoidFunction next);

}  // namespace framefx::vulkan
