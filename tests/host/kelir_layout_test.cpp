/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include "kelir/controller.h"
#include "kelir/ffi.h"
#include "kelir/plan.h"

namespace framefx::kelir {
namespace {

// The structs crossing into kelir/rust have one layout on both sides.
TEST(KelirLayout, MatchesRust) {
    size_t rust[8] = {};
    framefx_kelir_sizes(rust);
    EXPECT_EQ(rust[0], sizeof(KindInfo));
    EXPECT_EQ(rust[1], sizeof(Stack));
    EXPECT_EQ(rust[2], sizeof(Step));
    EXPECT_EQ(rust[3], sizeof(PlanCore));
    EXPECT_EQ(rust[4], sizeof(Program));
    EXPECT_EQ(rust[5], sizeof(FrameInfo));
    EXPECT_EQ(rust[6], sizeof(ControllerCore));
    EXPECT_EQ(rust[7], sizeof(Target));
    EXPECT_EQ(offsetof(PlanCore, lutLinear) + 1 <= sizeof(PlanCore), true);
}

}  // namespace
}  // namespace framefx::kelir
