/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include "luwes/resize.h"

namespace framefx::luwes {
namespace {

constexpr VkExtent2D kPanel = {2772, 1280};

bool Same(VkExtent2D a, VkExtent2D b) {
    return a.width == b.width && a.height == b.height;
}

TEST(Resize, AGameAtAQuarterIsShownAtThePanelsSizeTheWayItHoldsIt) {
    EXPECT_TRUE(Same(ResizedExtent({291, 629}, kPanel), {1280, 2772}));
    EXPECT_TRUE(Same(ResizedExtent({629, 291}, kPanel), {2772, 1280}));
    EXPECT_TRUE(Same(ResizedExtent({1080, 2338}, kPanel), {1280, 2772}));
}

TEST(Resize, AGameAtThePanelsSizeIsLeftAlone) {
    EXPECT_TRUE(Same(ResizedExtent({1280, 2772}, kPanel), {}));
    EXPECT_TRUE(Same(ResizedExtent({1164, 2516}, kPanel), {}));  // under 1.1x
}

TEST(Resize, SupersampledUpToTwiceIsDownsampled) {
    EXPECT_TRUE(Same(ResizedExtent({1920, 4158}, kPanel), {1280, 2772}));
    EXPECT_TRUE(Same(ResizedExtent({2880, 6237}, kPanel), {}));
}

TEST(Resize, AnotherAspectOrNothingKnownIsLeftAlone) {
    EXPECT_TRUE(Same(ResizedExtent({720, 1280}, kPanel), {}));
    EXPECT_TRUE(Same(ResizedExtent({291, 629}, {}), {}));
    EXPECT_TRUE(Same(ResizedExtent({0, 0}, kPanel), {}));
}

VkSwapchainCreateInfoKHR Swapchain() {
    VkSwapchainCreateInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
    info.imageExtent = {291, 629};
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    return info;
}

TEST(GameImage, IsWhatTheGameAskedForAndReadable) {
    const VkSwapchainCreateInfoKHR info = Swapchain();
    VkImageFormatListCreateInfo formats = {};
    const VkImageCreateInfo image = GameImageInfo(info, &formats);
    EXPECT_EQ(image.format, VK_FORMAT_R8G8B8A8_UNORM);
    EXPECT_EQ(image.extent.width, 291u);
    EXPECT_EQ(image.extent.height, 629u);
    EXPECT_EQ(image.usage, VkImageUsageFlags{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT} | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    EXPECT_EQ(image.flags, 0u);
    EXPECT_EQ(image.pNext, nullptr);
}

// ANGLE makes its swapchain with a mutable format and the list of the formats it views it in,
// for sRGB: the images it draws into instead take both, or its views of them are invalid.
TEST(GameImage, AngleViewsItsImagesInTheFormatsItListed) {
    const VkFormat views[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB};
    VkImageFormatListCreateInfo list = {};
    list.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
    list.viewFormatCount = 2;
    list.pViewFormats = views;
    VkSwapchainPresentModesCreateInfoEXT other = {};
    other.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT;
    other.pNext = &list;
    VkSwapchainCreateInfoKHR info = Swapchain();
    info.flags = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
    info.pNext = &other;

    VkImageFormatListCreateInfo formats = {};
    const VkImageCreateInfo image = GameImageInfo(info, &formats);
    EXPECT_NE(image.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, 0u);
    ASSERT_EQ(image.pNext, &formats);
    EXPECT_EQ(formats.sType, VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO);
    EXPECT_EQ(formats.pNext, nullptr);  // the swapchain's own chain stays with the swapchain
    ASSERT_EQ(formats.viewFormatCount, 2u);
    EXPECT_EQ(formats.pViewFormats[1], VK_FORMAT_R8G8B8A8_SRGB);
}

TEST(GameImage, AMutableSwapchainWithoutAListStaysMutable) {
    VkSwapchainCreateInfoKHR info = Swapchain();
    info.flags = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
    VkImageFormatListCreateInfo formats = {};
    const VkImageCreateInfo image = GameImageInfo(info, &formats);
    EXPECT_NE(image.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, 0u);
    EXPECT_EQ(image.pNext, nullptr);
}

}  // namespace
}  // namespace framefx::luwes
