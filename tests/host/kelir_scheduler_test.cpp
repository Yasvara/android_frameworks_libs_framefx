/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <memory>

#include "kelir/plan.h"
#include "tests/host/kelir_test_util.h"

namespace framefx::kelir {
namespace {

using test::MakeNode;
using test::MakeStack;
using test::Set;

// Frame generation as the backends drive Kelir: per real frame, the real present and then
// `multiplier - 1` generated ones, each asking Schedule what to run.
struct FakeLuwes {
    int multiplier = 1;
    int stageA = 0;
    int stageB = 0;
    int full = 0;
    int stageAOnGenerated = 0;

    void frame(const Plan* plan) {
        for (int i = 0; i < multiplier; i++) {
            const bool real = i == 0;
            const Stages s = Schedule(plan, multiplier > 1, real);
            stageA += s.stageA;
            stageB += s.stageB;
            full += s.full;
            if (!real && s.stageA) stageAOnGenerated++;
        }
    }
};

std::shared_ptr<const Plan> Mixed() {
    Node hue = MakeNode(kColor);
    Set(hue, p::kHue, 20.0f);
    return Compile(MakeStack({MakeNode(kSharpen), hue, MakeNode(kVignette)}), nullptr);
}

TEST(KelirScheduler, RealFrameRunsAOnceAndB) {
    const auto plan = Mixed();
    FakeLuwes luwes;
    luwes.multiplier = 3;
    for (int f = 0; f < 10; f++) luwes.frame(plan.get());
    EXPECT_EQ(10, luwes.stageA);
    EXPECT_EQ(30, luwes.stageB);
    EXPECT_EQ(0, luwes.full);
}

TEST(KelirScheduler, SynthFramesRunBOnly) {
    const auto plan = Mixed();
    const Stages s = Schedule(plan.get(), true, false);
    EXPECT_FALSE(s.stageA);
    EXPECT_TRUE(s.stageB);
    EXPECT_FALSE(s.full);
    FakeLuwes luwes;
    luwes.multiplier = 4;
    for (int f = 0; f < 5; f++) luwes.frame(plan.get());
    EXPECT_EQ(0, luwes.stageAOnGenerated);
}

TEST(KelirScheduler, WithoutGenerationOnePassDoesBoth) {
    const auto plan = Mixed();
    FakeLuwes off;
    for (int f = 0; f < 7; f++) off.frame(plan.get());
    EXPECT_EQ(7, off.full);
    EXPECT_EQ(0, off.stageA + off.stageB);
}

TEST(KelirScheduler, ScreenOnlyStackSkipsStageA) {
    const auto plan = Compile(MakeStack({MakeNode(kVignette), MakeNode(kFilmGrain)}), nullptr);
    const Stages real = Schedule(plan.get(), true, true);
    EXPECT_FALSE(real.stageA);
    EXPECT_TRUE(real.stageB);
}

TEST(KelirScheduler, GenerationSourceHasNoBEffects) {
    const auto plan = Mixed();
    Target a;
    a.role = Role::kStageA;
    const Program source = Unpack(Signature(*plan, a));
    EXPECT_TRUE(source.lut);
    EXPECT_EQ(kSharpen, source.detail[0]);
    for (int k : source.screen) EXPECT_EQ(-1, k);
    // ...and stage B carries no detail or colour, which generation has already seen.
    Target b;
    b.role = Role::kStageB;
    const Program finish = Unpack(Signature(*plan, b));
    EXPECT_FALSE(finish.lut);
    for (int k : finish.detail) EXPECT_EQ(-1, k);
    EXPECT_EQ(kVignette, finish.screen[0]);
}

TEST(KelirScheduler, PeekBypassesBoth) {
    Stack stack = MakeStack({MakeNode(kSharpen), MakeNode(kSepia), MakeNode(kVignette)});
    stack.peek = true;
    const auto plan = Compile(stack, nullptr);
    for (bool generation : {false, true}) {
        for (bool real : {false, true}) {
            const Stages s = Schedule(plan.get(), generation, real);
            EXPECT_FALSE(s.full || s.stageA || s.stageB) << generation << real;
        }
    }
    const Stages none = Schedule(nullptr, true, true);
    EXPECT_FALSE(none.full || none.stageA || none.stageB);
}

// Retro samples the held real frame, so it fuses into generation's output like any screen node.
TEST(KelirScheduler, RetroFusesIntoGenerationOutput) {
    const auto plan = Compile(MakeStack({MakeNode(kRetro)}), nullptr);
    EXPECT_TRUE(Schedule(plan.get(), true, false).stageB);
    Target b;
    b.role = Role::kStageB;
    EXPECT_EQ(kRetro, Unpack(Signature(*plan, b)).screen[0]);
}

}  // namespace
}  // namespace framefx::kelir
