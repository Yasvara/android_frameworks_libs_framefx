/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <vulkan/vulkan.h>

#include <cmath>

// A game drawing at another size than the display's: the size its swapchain is made at instead,
// and the images it then draws into. No Vulkan call; the backend makes what these describe.
namespace framefx::luwes {

inline constexpr float kResizeMin = 1.1f;
inline constexpr float kResizeMax = 2.0f;  // the downsample's kernel reaches 2 source texels per side
inline constexpr float kAspectSlack = 0.015f;

// The panel's size the way the game holds it, when the game draws its whole screen at the panel's
// aspect and a size far enough from it; zero to leave the swapchain as asked. `panel` is longer
// side first.
inline VkExtent2D ResizedExtent(VkExtent2D game, VkExtent2D panel) {
    const VkExtent2D none = {};
    if (panel.width == 0 || panel.height == 0 || game.width == 0 || game.height == 0) return none;
    const VkExtent2D o = game.width >= game.height ? panel : VkExtent2D{panel.height, panel.width};
    const float aspect = (static_cast<float>(game.width) / game.height) /
            (static_cast<float>(o.width) / o.height);
    const float ratio = static_cast<float>(o.width) / game.width;
    if (std::fabs(aspect - 1.0f) > kAspectSlack) return none;
    if (ratio < kResizeMin && 1.0f / ratio < kResizeMin) return none;
    if (1.0f / ratio > kResizeMax) return none;
    return o;
}

// The image a game draws into in place of a swapchain image made by `info`: what the game asked
// for, readable by the resample. A game that views its swapchain images in other formats (ANGLE,
// for sRGB) views these the same way: `formats` is filled and chained for it, and must outlive
// the result's use.
inline VkImageCreateInfo GameImageInfo(const VkSwapchainCreateInfoKHR& info,
                                       VkImageFormatListCreateInfo* formats) {
    VkImageCreateInfo image = {};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = info.imageFormat;
    image.extent = {info.imageExtent.width, info.imageExtent.height, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = info.imageUsage | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image.sharingMode = info.imageSharingMode;
    image.queueFamilyIndexCount = info.queueFamilyIndexCount;
    image.pQueueFamilyIndices = info.pQueueFamilyIndices;
    if ((info.flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR) == 0) return image;
    image.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext); next != nullptr;
         next = next->pNext) {
        if (next->sType != VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) continue;
        *formats = *reinterpret_cast<const VkImageFormatListCreateInfo*>(next);
        formats->pNext = nullptr;
        image.pNext = formats;
    }
    return image;
}

}  // namespace framefx::luwes
