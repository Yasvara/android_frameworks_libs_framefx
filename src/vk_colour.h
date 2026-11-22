/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <vulkan/vulkan.h>

namespace framefx {

// What a swapchain's colour space shows. Every nonlinear SDR space the display composes as sRGB
// or P3 is one of those; pass-through and extended sRGB on a UNORM format are sRGB on Android, as
// a UNORM value cannot leave [0, 1]. PQ, HLG, linear spaces and float formats are HDR.
enum class SwapchainColour { kSrgb, kP3, kHdr };

inline SwapchainColour ColourOf(VkColorSpaceKHR space, VkFormat format) {
    const bool unorm = format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_B8G8R8A8_UNORM ||
            format == VK_FORMAT_R8G8B8A8_SRGB || format == VK_FORMAT_B8G8R8A8_SRGB ||
            format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    switch (space) {
        case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
        case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
            return SwapchainColour::kSrgb;
        case VK_COLOR_SPACE_PASS_THROUGH_EXT:
        case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
            return unorm ? SwapchainColour::kSrgb : SwapchainColour::kHdr;
        case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
        case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
            return SwapchainColour::kP3;
        default:
            return SwapchainColour::kHdr;
    }
}

}  // namespace framefx
