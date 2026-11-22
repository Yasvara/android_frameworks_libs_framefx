/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include "src/target.h"

namespace framefx {
namespace {

constexpr int64_t kMs = 1'000'000;
constexpr int64_t kRefresh135 = 7'407'407;

// What a backend does with the pacer after presenting: schedule, then any limiter wait.
void Pace(Target& target, int64_t* now, const Target::Decision& d) {
    luwes::Present order[luwes::kMaxPresents];
    target.pacer().schedule(*now, d.plan, Mode::kExtrapolate, d.generate, kRefresh135, order);
    const int64_t wake = target.pacer().limit(*now);
    if (wake > *now) *now = wake;
    target.pacer().resumed(*now);
}

// Frames generated per real frame at the base, 0 when it generates none.
int Generated(const Target::Decision& d) {
    return d.generate ? d.plan.multiplier - 1 : 0;
}

LuwesSettings luwes(int multiplier, Mode mode = Mode::kExtrapolate) {
    LuwesSettings s;
    s.multiplier = multiplier;
    s.mode = mode;
    return s;
}

// A game loop at 30 fps on a 135 Hz panel: slower than every base, so generation runs from the
// first frame with history and the limiter stays out of the way.
struct Loop {
    Target target;
    int64_t now = 0;
    int width = 1968;
    int height = 908;
    uint32_t transform = 0;

    Target::Decision present(const LuwesSettings& s) {
        now += 33 * kMs;
        Target::Frame f;
        f.now = now;
        f.width = width;
        f.height = height;
        f.transform = transform;
        f.refreshNs = kRefresh135;
        const Target::Decision d = target.begin(f, s);
        if (d.process) {
            target.captured();
            Pace(target, &now, d);
        }
        return d;
    }

    void engage(const LuwesSettings& s) {
        for (int i = 0; i < 30; i++) present(s);
    }
};

TEST(Target, OffNeverProcesses) {
    Loop loop;
    for (int i = 0; i < 100; i++) {
        const Target::Decision d = loop.present(luwes(0));
        EXPECT_FALSE(d.process);
        EXPECT_EQ(0, Generated(d));
    }
}

TEST(Target, ArmWaitsForAGameLoopThenPrimesHistory) {
    Loop loop;
    int firstProcessed = -1;
    for (int i = 0; i < 40 && firstProcessed < 0; i++) {
        if (loop.present(luwes(3)).process) firstProcessed = i;
    }
    ASSERT_GE(firstProcessed, 20);  // the engagement gate needs a sustained loop
    const Target::Decision next = loop.present(luwes(3));
    EXPECT_TRUE(next.process);
    EXPECT_EQ(2, Generated(next));
}

TEST(Target, FirstProcessedFrameHasNoHistory) {
    Loop loop;
    Target::Decision d;
    do {
        d = loop.present(luwes(2));
    } while (!d.process);
    EXPECT_EQ(0, Generated(d));
}

TEST(Target, SlowWindowNeverEngages) {
    Loop loop;
    for (int i = 0; i < 100; i++) {
        loop.now += 80 * kMs;  // an on-demand window, not a loop
        Target::Frame f{loop.now, 1080, 2400, 0, kRefresh135};
        EXPECT_FALSE(loop.target.begin(f, luwes(2)).process);
    }
}

TEST(Target, DisarmReleasesAndRearmPrimesAgain) {
    Loop loop;
    loop.engage(luwes(2));
    ASSERT_EQ(1, Generated(loop.present(luwes(2))));

    const Target::Decision off = loop.present(luwes(0));
    EXPECT_FALSE(off.process);
    EXPECT_TRUE(off.releaseLuwes);
    EXPECT_FALSE(loop.target.history());
    EXPECT_FALSE(loop.present(luwes(0)).releaseLuwes);  // once

    const Target::Decision on = loop.present(luwes(2));
    EXPECT_TRUE(on.process);  // still engaged: the loop never stopped
    EXPECT_EQ(0, Generated(on));
    EXPECT_EQ(1, Generated(loop.present(luwes(2))));
}

TEST(Target, LiveMultiplierAndModeChangesApplyNextFrame) {
    Loop loop;
    loop.engage(luwes(2));
    ASSERT_EQ(1, Generated(loop.present(luwes(2))));
    const Target::Decision four = loop.present(luwes(4, Mode::kInterpolate));
    EXPECT_FALSE(four.resize);
    EXPECT_EQ(3, Generated(four));
}

TEST(Target, ResizeDropsHistoryButKeepsEngagement) {
    Loop loop;
    loop.engage(luwes(3));
    ASSERT_EQ(2, Generated(loop.present(luwes(3))));
    loop.width = 2400;
    loop.height = 1080;
    const Target::Decision resized = loop.present(luwes(3));
    EXPECT_TRUE(resized.resize);
    EXPECT_TRUE(resized.process);
    EXPECT_EQ(0, Generated(resized));
    const Target::Decision next = loop.present(luwes(3));
    EXPECT_FALSE(next.resize);
    EXPECT_EQ(2, Generated(next));
}

TEST(Target, RotationIsAResize) {
    Loop loop;
    loop.engage(luwes(2));
    loop.transform = 4;  // NATIVE_WINDOW_TRANSFORM_ROT_90
    const Target::Decision d = loop.present(luwes(2));
    EXPECT_TRUE(d.resize);
    EXPECT_EQ(0, Generated(d));
}

TEST(Target, RenderScaleChangeRestartsHistoryAndKeepsPacing) {
    // A paced game: 6 ms of work against a 67.5 fps base.
    Target target;
    int64_t now = 0;
    Target::Decision d;
    auto frame = [&](int width, int height) {
        now += 6 * kMs;
        d = target.begin({now, width, height, 0, kRefresh135}, luwes(2));
        if (!d.process) return;
        target.captured();
        Pace(target, &now, d);
    };
    for (int i = 0; i < 200; i++) frame(1968, 908);
    ASSERT_TRUE(d.plan.paced);
    ASSERT_EQ(1, Generated(d));
    frame(1386, 640);  // render scale 0.7: the fitted buffer shrinks
    EXPECT_TRUE(d.resize);
    EXPECT_EQ(0, Generated(d));
    EXPECT_TRUE(d.plan.paced);
    frame(1386, 640);
    EXPECT_EQ(1, Generated(d));
}

TEST(Target, RecreatedSwapchainInheritsEngagement) {
    Loop loop;
    loop.engage(luwes(2));
    Target replacement;
    replacement.inherit(loop.target);
    loop.now += 33 * kMs;
    const Target::Decision first =
            replacement.begin({loop.now, loop.width, loop.height, 0, kRefresh135}, luwes(2));
    EXPECT_TRUE(first.process);
    EXPECT_TRUE(first.resize);
    EXPECT_EQ(0, Generated(first));
    replacement.captured();
    loop.now += 33 * kMs;
    EXPECT_EQ(1, Generated(replacement.begin({loop.now, loop.width, loop.height, 0, kRefresh135},
                                             luwes(2))));
}

TEST(Target, FreshSwapchainWithoutPredecessorWaitsForTheLoop) {
    Target fresh;
    EXPECT_FALSE(fresh.begin({33 * kMs, 1968, 908, 0, kRefresh135}, luwes(2)).process);
}

TEST(Meter, ReportsRatesOncePerSecond) {
    Meter meter;
    int reports = 0;
    for (int i = 1; i <= 90; i++) {
        if (meter.frame(i * 33 * kMs, 3)) reports++;  // 30 real frames a second, 3x
    }
    EXPECT_EQ(2, reports);
    EXPECT_NEAR(30.3, meter.realFps(), 0.5);
    EXPECT_NEAR(90.9, meter.presentedFps(), 1.5);
}

TEST(Sleep, WaitsUntilTheDeadline) {
    const int64_t start = Now();
    SleepUntil(start + 2 * kMs);
    EXPECT_GE(Now() - start, 2 * kMs);
    SleepUntil(start);  // already past
}

}  // namespace
}  // namespace framefx
