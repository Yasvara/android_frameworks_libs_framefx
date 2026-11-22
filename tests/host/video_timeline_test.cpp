/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include "video/timeline.h"

namespace framefx::video {
namespace {

constexpr int64_t kMs = 1'000'000;
constexpr int64_t k120 = 8'333'333;

TEST(Timeline, TheFirstFrameGoesAlone) {
    Timeline line;
    Present p[kMaxPresents];
    ASSERT_EQ(1, line.plan(1000 * kMs, 960 * kMs, k120, 16 * kMs, 1.0f, p));
    EXPECT_TRUE(p[0].real);
    EXPECT_EQ(1000 * kMs, p[0].when);
}

// The player's lead covers the whole interval: no delay, every frame of the grid.
TEST(Timeline, ThirtyOnAHundredTwentyIsFourPresents) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 900 * kMs, k120, 16 * kMs, 1.0f, p);
    const int64_t t = 1000 * kMs + 33'333'333;
    ASSERT_EQ(4, line.plan(t, t - 50 * kMs, k120, 16 * kMs, 1.0f, p));
    EXPECT_EQ(0, line.delay());
    for (int i = 0; i < 3; i++) {
        EXPECT_FALSE(p[i].real);
        EXPECT_NEAR(0.25f * static_cast<float>(i + 1), p[i].phase, 1e-3f);
        EXPECT_EQ(1000 * kMs + (i + 1) * k120, p[i].when);
    }
    EXPECT_TRUE(p[3].real);
    EXPECT_EQ(t, p[3].when);
}

TEST(Timeline, TwentyFourOnAHundredTwentyIsFivePresents) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 900 * kMs, k120, 10 * kMs, 1.0f, p);
    const int64_t t = 1000 * kMs + 41'666'666;
    EXPECT_EQ(5, line.plan(t, t - 60 * kMs, k120, 10 * kMs, 1.0f, p));
}

TEST(Timeline, AVideoAtTheDisplayRateHasNothingToGenerate) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 900 * kMs, k120, 10 * kMs, 1.0f, p);
    EXPECT_EQ(1, line.plan(1000 * kMs + k120, 1000 * kMs - 40 * kMs, k120, 10 * kMs, 1.0f, p));
}

// YouTube's measured lead at 24 fps, 40.7 ms: the first frame of the interval needs a delay.
TEST(Timeline, AShortLeadDelaysTheVideoInsteadOfDroppingFrames) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 960 * kMs, k120, 16 * kMs, 1.0f, p);
    const int64_t t = 1000 * kMs + 41'666'666;
    const int64_t now = t - 40'700'000;
    ASSERT_EQ(5, line.plan(t, now, k120, 16 * kMs, 1.0f, p));
    EXPECT_GT(line.delay(), 8 * kMs);
    EXPECT_LE(line.delay(), kMaxDelayNs);
    EXPECT_GE(p[0].when - 16 * kMs, now);
    EXPECT_EQ(t + line.delay(), p[4].when);
}

TEST(Timeline, TheDelayNeverPassesWhatTheAudioHides) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 999 * kMs, k120, 16 * kMs, 1.0f, p);
    const int64_t t = 1000 * kMs + 41'666'666;
    const int count = line.plan(t, t - kMs, k120, 16 * kMs, 1.0f, p);
    EXPECT_EQ(kMaxDelayNs, line.delay());
    EXPECT_TRUE(p[count - 1].real);
    // What the capped delay cannot save goes; the order holds.
    for (int i = 1; i < count; i++) EXPECT_GT(p[i].when, p[i - 1].when);
    EXPECT_LT(count, 5);
}

TEST(Timeline, ADelayShrinksSlowlyWhenTheLeadComesBack) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 960 * kMs, k120, 16 * kMs, 1.0f, p);
    int64_t t = 1000 * kMs + 41'666'666;
    line.plan(t, t - 30 * kMs, k120, 16 * kMs, 1.0f, p);
    const int64_t high = line.delay();
    t += 41'666'666;
    line.plan(t, t - 100 * kMs, k120, 16 * kMs, 1.0f, p);
    EXPECT_EQ(high - kDelayFallNs, line.delay());
}

TEST(Timeline, ASeekOrPauseGoesAloneAndAdoptsTheNewTime) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 950 * kMs, k120, 16 * kMs, 1.0f, p);
    EXPECT_EQ(1, line.plan(900 * kMs, 850 * kMs, k120, 16 * kMs, 1.0f, p));
    EXPECT_EQ(1, line.plan(5000 * kMs, 4950 * kMs, k120, 16 * kMs, 1.0f, p));
    const int64_t t = 5000 * kMs + 33'333'333;
    EXPECT_EQ(4, line.plan(t, t - 50 * kMs, k120, 16 * kMs, 1.0f, p));
}

TEST(Timeline, ACutGoesAloneButKeepsTheTime) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 950 * kMs, k120, 16 * kMs, 1.0f, p);
    int64_t t = 1000 * kMs + 33'333'333;
    EXPECT_EQ(1, line.plan(t, t - 50 * kMs, 0, 16 * kMs, 1.0f, p));
    t += 33'333'333;
    EXPECT_EQ(4, line.plan(t, t - 50 * kMs, k120, 16 * kMs, 1.0f, p));
}

TEST(Timeline, SmoothnessKeepsSomeOfTheCadence) {
    Timeline line;
    Present p[kMaxPresents];
    line.plan(1000 * kMs, 950 * kMs, k120, 16 * kMs, 0.5f, p);
    const int64_t t = 1000 * kMs + 33'333'333;
    ASSERT_EQ(4, line.plan(t, t - 50 * kMs, k120, 16 * kMs, 0.5f, p));
    EXPECT_NEAR(0.125f, p[0].phase, 1e-3f);
    EXPECT_NEAR(0.375f, p[2].phase, 1e-3f);
    EXPECT_FLOAT_EQ(1.0f, p[3].phase);
}

TEST(Timeline, TwentyFourOnSixtyAlternatesTheGrid) {
    Timeline line;
    Present p[kMaxPresents];
    const int64_t k60 = 16'666'667;
    line.plan(1000 * kMs, 900 * kMs, k60, 16 * kMs, 1.0f, p);
    const int64_t t = 1000 * kMs + 41'666'666;
    ASSERT_EQ(3, line.plan(t, t - 60 * kMs, k60, 16 * kMs, 1.0f, p));
    EXPECT_NEAR(0.4f, p[0].phase, 1e-3f);
    EXPECT_NEAR(0.8f, p[1].phase, 1e-3f);
}

}  // namespace
}  // namespace framefx::video
