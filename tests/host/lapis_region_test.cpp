/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include "lapis/model.h"

namespace framefx::lapis {
namespace {

bool Whole(VkRect2D r) {
    return r.extent.width == 0 && r.extent.height == 0;
}

TEST(DrawnRegion, ADynamicResolutionsCornerIsTheRegion) {
    // Roblox at ~80 %: 1144x2479 image, viewport from the top left.
    const VkRect2D r = DrawnRegion(0, 0, 912, 1983, 1144, 2479);
    EXPECT_EQ(0, r.offset.x);
    EXPECT_EQ(0, r.offset.y);
    EXPECT_EQ(912u, r.extent.width);
    EXPECT_EQ(1983u, r.extent.height);
}

TEST(DrawnRegion, ANegativeHeightSpansUpFromItsY) {
    // The Y-flipped viewport of the same corner: y at its bottom edge.
    const VkRect2D r = DrawnRegion(0, 1983, 912, -1983, 1144, 2479);
    EXPECT_EQ(0, r.offset.y);
    EXPECT_EQ(1983u, r.extent.height);
}

TEST(DrawnRegion, AllOfItOrNothingKnownIsWhole) {
    EXPECT_TRUE(Whole(DrawnRegion(0, 0, 1144, 2479, 1144, 2479)));
    EXPECT_TRUE(Whole(DrawnRegion(0, 0, 1143, 2478, 1144, 2479)));  // rounding
    EXPECT_TRUE(Whole(DrawnRegion(0, 0, 0, 0, 1144, 2479)));         // unknown
    EXPECT_TRUE(Whole(DrawnRegion(0, 0, 1280, 2772, 1144, 2479)));   // past the image
    EXPECT_TRUE(Whole(DrawnRegion(-4, 0, 900, 1900, 1144, 2479)));
}

}  // namespace
}  // namespace framefx::lapis
