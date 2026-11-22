/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <tuple>
#include <vector>

#include "luwes/pacer.h"

namespace framefx::luwes {
namespace {

constexpr int64_t kMs = 1'000'000;

int64_t refreshOf(double hz) { return static_cast<int64_t>(std::llround(1e9 / hz)); }

TEST(Multiplier, KeepsBaseAtOrAbove30) {
    EXPECT_EQ(2, EffectiveMultiplier(2, 135.0));
    EXPECT_EQ(3, EffectiveMultiplier(3, 135.0));
    EXPECT_EQ(4, EffectiveMultiplier(4, 135.0));
    EXPECT_EQ(2, EffectiveMultiplier(2, 90.0));
    EXPECT_EQ(3, EffectiveMultiplier(3, 90.0));
    EXPECT_EQ(3, EffectiveMultiplier(4, 90.0));
    EXPECT_EQ(2, EffectiveMultiplier(2, 60.0));
    EXPECT_EQ(2, EffectiveMultiplier(3, 60.0));
    EXPECT_EQ(2, EffectiveMultiplier(4, 60.0));
    EXPECT_EQ(1, EffectiveMultiplier(0, 135.0));
    EXPECT_EQ(1, EffectiveMultiplier(2, 50.0));
    EXPECT_EQ(4, EffectiveMultiplier(9, 135.0));
}

TEST(Multiplier, BaseAgainstPanel) {
    EXPECT_DOUBLE_EQ(67.5, BaseFps(2, 135.0));
    EXPECT_DOUBLE_EQ(45.0, BaseFps(3, 135.0));
    EXPECT_DOUBLE_EQ(33.75, BaseFps(4, 135.0));
    EXPECT_DOUBLE_EQ(45.0, BaseFps(2, 90.0));
    EXPECT_DOUBLE_EQ(30.0, BaseFps(2, 60.0));
}

// SurfaceFlinger and the BufferQueue as the pacer sees them. A composition `compositor` before
// every refresh latches queued presents in order while each is due (desired time before the
// refresh) and its buffer is ready; the last one latched shows at the refresh and any before it
// are dropped. A buffer is free again once its successor is latched. The producer holds at most
// `buffers` of them queued or latched; past that, queueing blocks until a composition frees one.
struct Display {
    int64_t refresh = 0;
    int64_t phase = 0;  // a refresh
    int64_t compositor = 0;
    size_t buffers = 0;

    struct Queued {
        int64_t desired;
        int64_t ready;
        double content;
        size_t id;
    };
    // What became of each present: shown at a refresh, or dropped, and when that was known.
    struct Fate {
        int64_t shown = 0;
        int64_t known = 0;
    };
    std::vector<Fate> fates;
    struct Shown {
        int64_t at;
        double content;
        int64_t desired;
        int64_t ready;
    };
    std::deque<Queued> queue;
    bool latched = false;
    int64_t next = 0;  // the next refresh to compose for
    std::vector<Shown> screen;  // what each refresh showed
    int dropped = 0;

    int64_t refreshAfter(int64_t t) const {
        const int64_t k = (t - phase + refresh - 1) / refresh;
        return phase + k * refresh;
    }

    // Composes every refresh whose composition is due by `t`.
    void run(int64_t t) {
        if (next == 0) next = refreshAfter(t + compositor);
        for (; next - compositor <= t; next += refresh) {
            int applied = 0;
            Queued shown{};
            while (!queue.empty() && queue.front().desired < next &&
                   queue.front().ready <= next - compositor) {
                if (applied > 0) fates[shown.id] = {0, next - compositor};
                shown = queue.front();
                queue.pop_front();
                applied++;
            }
            if (applied > 0) {
                fates[shown.id] = {next, next};
                dropped += applied - 1;
                screen.push_back({next, shown.content, shown.desired, shown.ready});
                latched = true;
            } else if (latched) {
                Shown again = screen.back();
                again.at = next;
                screen.push_back(again);
            }
        }
    }

    // Queues a present at `*now`, blocking while every buffer is taken; returns the time spent
    // blocked.
    int64_t queuePresent(int64_t* now, int64_t desired, int64_t gpu, double content) {
        const int64_t start = *now;
        run(*now);
        while (queue.size() + (latched ? 1 : 0) >= buffers) {
            *now = next - compositor;
            run(*now);
        }
        queue.push_back({desired, *now + gpu, content, fates.size()});
        fates.push_back({});
        return *now - start;
    }

    // Deadline and latency as SurfaceFlinger's compositor timing reports them at `now`.
    void timing(int64_t now, int64_t* deadline, int64_t* latency) const {
        *deadline = refreshAfter(now + compositor + 1) - compositor;
        *latency = compositor;
    }
};

struct Game {
    double workMs = 6.0;  // building a frame
    double capMs = 0.0;   // the game's own frame limit, 0 for none
    double jitterMs = 0.0;
    double gpuMs = 4.0;   // queue to buffer ready
};

struct Outcome {
    double realIntervalMs = 0;
    double presentsPerSecond = 0;
    double shownPerSecond = 0;  // refreshes that showed a present they had not shown before
    Pacer::Plan last;
    int repeats = 0;            // refreshes that showed the previous refresh's present again
    double judderMs = 0;        // worst deviation of content advance per refresh from a refresh
    double latencySpreadMs = 0; // display time minus content time, max - min
    double latencyMs = 0;       // its mean, from the game presenting a frame to showing it
    int64_t longestWaitNs = 0;  // the limiter
    int64_t blockedNs = 0;      // queueing on a full queue
    int untouched = 0;          // the game's frames shown as rendered
    int dropped = 0;
};

Outcome Simulate(double hz, int multiplier, Mode mode, const Game& game, size_t buffers = 8,
                 int frames = 900) {
    Pacer pacer;
    pacer.buffers(static_cast<int>(buffers));
    const int64_t refresh = refreshOf(hz);
    Display display;
    display.refresh = refresh;
    display.phase = 3 * kMs;
    display.compositor = 12 * kMs + refresh / 3;
    display.buffers = buffers;
    std::mt19937 random(7);
    std::uniform_real_distribution<double> jitter(-game.jitterMs, game.jitterMs);
    const bool interpolate = mode == Mode::kInterpolate;
    int64_t now = 1'000 * kMs;
    int64_t lastArrival = 0;
    int64_t previousArrival = 0;
    int64_t steadyStart = 0;
    int steadyFrames = 0;
    int steadyPresents = 0;
    size_t steadyScreen = 0;
    bool history = false;
    struct Asked {
        size_t id;
        int64_t desired;
        int64_t ready;
    };
    std::deque<Asked> feedback;
    Outcome run;
    for (int f = 0; f < frames; f++) {
        int64_t arrival = now + static_cast<int64_t>((game.workMs + jitter(random)) * kMs);
        if (game.capMs > 0 && lastArrival > 0) {
            arrival = std::max(arrival, lastArrival + static_cast<int64_t>(game.capMs * kMs));
        }
        now = arrival;
        display.run(now);
        int64_t deadline = 0;
        int64_t latency = 0;
        display.timing(now, &deadline, &latency);
        pacer.compositor(deadline, latency);
        // The display reports each present, in queue order, once shown or dropped.
        while (!feedback.empty()) {
            const Asked& asked = feedback.front();
            const Display::Fate& fate = display.fates[asked.id];
            if (fate.known == 0 || fate.known > now) break;
            pacer.presented(asked.desired, asked.ready, fate.shown, refresh);
            feedback.pop_front();
        }
        const Pacer::Plan plan = pacer.begin(now, multiplier, refresh);
        if (getenv("TRACE") && f % 50 == 0) printf("f %d lead %.2f window %.2f shown %zu dropped %d\n", f, pacer.leadNs() / 1e6, pacer.windowNs() / 1e6, display.screen.size(), display.dropped);
        const bool generate = history && plan.multiplier > 1;
        Present order[kMaxPresents];
        const int count = pacer.schedule(now, plan, mode, generate, refresh, order);
        const bool steady = f > frames / 2;
        const int64_t a = previousArrival;
        const int64_t b = now;
        for (int i = 0; i < count; i++) {
            const Present& p = order[i];
            // Content time of what the present shows, the game's frames at their arrival.
            double content = static_cast<double>(b);
            if (p.generated || p.phase != (interpolate ? 1.0f : 0.0f)) {
                content = interpolate ? a + p.phase * static_cast<double>(b - a)
                                      : b + p.phase * static_cast<double>(b - a);
            }
            if (!p.generated && steady) run.untouched++;
            const int64_t blocked = display.queuePresent(
                    &now, p.desired, static_cast<int64_t>(game.gpuMs * kMs), content);
            if (steady) run.blockedNs += blocked;
            feedback.push_back({display.queue.back().id, p.desired, display.queue.back().ready});
        }
        history = true;
        previousArrival = b;
        lastArrival = b;
        const int64_t wake = pacer.limit(now);
        if (wake > now) {
            run.longestWaitNs = std::max(run.longestWaitNs, wake - now);
            now = wake;
        }
        pacer.resumed(now);
        run.last = plan;
        if (f == frames / 2) {
            steadyStart = now;
            display.run(now);
            steadyScreen = display.screen.size();
        }
        if (steady) {
            steadyFrames++;
            steadyPresents += count;
        }
    }
    const double seconds = static_cast<double>(now - steadyStart) / 1e9;
    run.realIntervalMs = seconds * 1000.0 / steadyFrames;
    run.presentsPerSecond = steadyPresents / seconds;
    // Judge the refreshes composed in steady state, leaving out the tail still in flight.
    display.run(now);
    const size_t end = display.screen.size() > 20 ? display.screen.size() - 20 : 0;
    double minLatency = 1e18;
    double maxLatency = -1e18;
    double sumLatency = 0;
    int fresh = 0;
    for (size_t i = steadyScreen + 1; i < end; i++) {
        const auto& s = display.screen[i];
        const auto& before = display.screen[i - 1];
        if (s.desired == before.desired) {
            run.repeats++;
        } else {
            fresh++;
        }
        const double advance = (s.content - before.content) / 1e6;
        run.judderMs = std::max(run.judderMs, std::abs(advance - refresh / 1e6));
        const double latency = (static_cast<double>(s.at) - s.content) / 1e6;
        minLatency = std::min(minLatency, latency);
        maxLatency = std::max(maxLatency, latency);
        sumLatency += latency;
    }
    const size_t judged = end > steadyScreen + 1 ? end - steadyScreen - 1 : 1;
    run.latencySpreadMs = maxLatency - minLatency;
    run.latencyMs = sumLatency / judged;
    run.shownPerSecond = fresh / (judged * refresh / 1e9);
    run.dropped = display.dropped;
    return run;
}

class Grid : public testing::TestWithParam<std::tuple<double, int, Mode>> {};

// A game with headroom is held at Hz/n: every refresh shows a new present, every real frame
// shows untouched, and content advances exactly a refresh per refresh.
TEST_P(Grid, CapableGameIsHeldAtTheBaseAndFillsEveryRefresh) {
    const auto [hz, multiplier, mode] = GetParam();
    const int effective = EffectiveMultiplier(multiplier, hz);
    const Outcome run = Simulate(hz, multiplier, mode, Game{.workMs = 6.0});
    EXPECT_TRUE(run.last.paced);
    EXPECT_EQ(effective, run.last.multiplier);
    EXPECT_NEAR(1000.0 / BaseFps(effective, hz), run.realIntervalMs, run.realIntervalMs * 0.03);
    EXPECT_NEAR(hz, run.shownPerSecond, hz * 0.02);
    EXPECT_EQ(0, run.repeats);
    EXPECT_EQ(0, run.dropped);
    EXPECT_LT(run.judderMs, 0.05);
    EXPECT_GT(run.untouched, 0);
}

// A game slower than the base never waits for Luwes, and every refresh it can fill shows content
// a refresh further on.
TEST_P(Grid, SlowGameRunsAtItsOwnRateAndFillsTheRefreshes) {
    const auto [hz, multiplier, mode] = GetParam();
    const int effective = EffectiveMultiplier(multiplier, hz);
    const double base = BaseFps(effective, hz);
    const double intervalMs = 1000.0 / (base * 0.7);
    const Outcome run = Simulate(hz, multiplier, mode, Game{.workMs = intervalMs});
    EXPECT_FALSE(run.last.paced);
    EXPECT_EQ(0, run.longestWaitNs);
    EXPECT_EQ(0, run.blockedNs);
    EXPECT_NEAR(intervalMs, run.realIntervalMs, intervalMs * 0.01);
    const double fillable = std::min(hz, 1000.0 / intervalMs * kMaxPresents);
    EXPECT_NEAR(fillable, run.shownPerSecond, hz * 0.02);
    if (fillable >= hz * 0.99) {
        EXPECT_EQ(0, run.repeats);
        EXPECT_LT(run.judderMs, refreshOf(hz) / 8e6);
    }
    EXPECT_EQ(0, run.dropped);
}

INSTANTIATE_TEST_SUITE_P(
        PanelsAndMultipliers, Grid,
        testing::Combine(testing::Values(135.0, 90.0, 60.0), testing::Values(2, 3, 4),
                         testing::Values(Mode::kExtrapolate, Mode::kInterpolate)),
        [](const testing::TestParamInfo<Grid::ParamType>& info) {
            return std::to_string(static_cast<int>(std::get<0>(info.param))) + "Hz_" +
                    std::to_string(std::get<1>(info.param)) + "x_" +
                    (std::get<2>(info.param) == Mode::kExtrapolate ? "Extrapolate"
                                                                   : "Interpolate");
        });

// A 60 fps game on a 135 Hz panel, 2x: below the 67.5 base, so it keeps its own 60, and all 135
// refreshes show a frame of their own, content advancing evenly.
TEST(Pacer, SixtyOnOneThirtyFiveFillsEveryRefresh) {
    for (Mode mode : {Mode::kInterpolate, Mode::kExtrapolate}) {
        const Outcome run =
                Simulate(135.0, 2, mode, Game{.workMs = 5.0, .capMs = 1000.0 / 60.0});
        EXPECT_FALSE(run.last.paced);
        EXPECT_EQ(0, run.longestWaitNs);
        EXPECT_EQ(0, run.blockedNs);
        EXPECT_NEAR(1000.0 / 60.0, run.realIntervalMs, 0.1);
        EXPECT_NEAR(135.0, run.shownPerSecond, 1.0);
        EXPECT_EQ(0, run.repeats);
        EXPECT_EQ(0, run.dropped);
        EXPECT_LT(run.judderMs, 0.3);
        EXPECT_LT(run.latencySpreadMs, 0.5);
    }
}

// Interpolation shows content an interval and the lead behind; extrapolation only the lead.
TEST(Pacer, LatencyIsTheLeadPlusTheIntervalWhenInterpolating) {
    const Game game{.workMs = 5.0, .capMs = 1000.0 / 60.0};
    const Outcome interpolate = Simulate(135.0, 2, Mode::kInterpolate, game);
    const Outcome extrapolate = Simulate(135.0, 2, Mode::kExtrapolate, game);
    const double refreshMs = 1000.0 / 135.0;
    EXPECT_NEAR(interpolate.latencyMs - extrapolate.latencyMs, 1000.0 / 60.0, 1.0);
    // Composition, half a refresh for the lead, half to the refresh, a quarter for jitter.
    EXPECT_LT(extrapolate.latencyMs, 12.0 + refreshMs / 3 + 1.25 * refreshMs + 2.0);
}

// Arrivals that wander by a couple of milliseconds barely move the content timeline.
TEST(Pacer, ArrivalJitterStaysOffTheTimeline) {
    const Outcome run = Simulate(
            135.0, 2, Mode::kInterpolate,
            Game{.workMs = 5.0, .capMs = 1000.0 / 60.0, .jitterMs = 2.0});
    EXPECT_EQ(0, run.repeats);
    EXPECT_EQ(0, run.dropped);
    EXPECT_NEAR(135.0, run.shownPerSecond, 1.0);
}

// Too few buffers for every refresh: the windows take only free ones, so the game never blocks
// and keeps its rate, and still gets more than its own frames shown.
TEST(Pacer, FewBuffersNeverBlockTheGame) {
    for (Mode mode : {Mode::kInterpolate, Mode::kExtrapolate}) {
        const Outcome run = Simulate(135.0, 2, mode, Game{.workMs = 5.0, .capMs = 1000.0 / 60.0},
                                     /*buffers=*/4);
        EXPECT_EQ(0, run.blockedNs);
        EXPECT_NEAR(1000.0 / 60.0, run.realIntervalMs, 0.1);
        EXPECT_GT(run.shownPerSecond, 90.0);
        EXPECT_EQ(0, run.dropped);
    }
}

// Buffers that take longer than the starting lead to be ready: the lead grows until presents
// are shown when they were asked for.
TEST(Pacer, LeadGrowsToCoverGpuLatency) {
    const Outcome run =
            Simulate(135.0, 2, Mode::kExtrapolate, Game{.workMs = 25.0, .gpuMs = 18.0});
    EXPECT_EQ(0, run.repeats);
    EXPECT_EQ(0, run.dropped);
}

TEST(Pacer, LeadShrinksBackWhenFramesAreOnTime) {
    Pacer pacer;
    const int64_t refresh = refreshOf(135.0);
    pacer.compositor(0, 12 * kMs);
    int64_t now = kMs;
    Present order[kMaxPresents];
    const Pacer::Plan plan = pacer.begin(now, 2, refresh);
    pacer.schedule(now, plan, Mode::kInterpolate, true, refresh, order);
    const int64_t start = pacer.leadNs();
    // Buffers ready two refreshes after they were due.
    const int64_t desired = 100 * kMs;
    const int64_t due = desired + refresh / 2;
    for (int i = 0; i < 20; i++) pacer.presented(desired, due + 2 * refresh, due + 3 * refresh, refresh);
    EXPECT_GT(pacer.leadNs(), start);
    const int64_t grown = pacer.leadNs();
    for (int i = 0; i < 400; i++) pacer.presented(desired, due - 13 * kMs - refresh, due, refresh);
    EXPECT_LT(pacer.leadNs(), grown);
    EXPECT_GE(pacer.leadNs(), 12 * kMs + refresh / 2);
}

TEST(Pacer, ReleasesPacingWhenTheGameFallsBehind) {
    Pacer pacer;
    const int64_t refresh = refreshOf(135.0);
    int64_t now = kMs;
    Pacer::Plan plan;
    Present order[kMaxPresents];
    auto frame = [&](double workMs) {
        now += static_cast<int64_t>(workMs * kMs);
        plan = pacer.begin(now, 3, refresh);
        pacer.schedule(now, plan, Mode::kExtrapolate, true, refresh, order);
        const int64_t wake = pacer.limit(now);
        if (wake > now) now = wake;
        pacer.resumed(now);
    };
    for (int f = 0; f < 200; f++) frame(8.0);
    ASSERT_TRUE(plan.paced);
    int released = -1;
    for (int f = 0; f < 200 && released < 0; f++) {
        frame(60.0);
        if (!plan.paced) released = f;
    }
    EXPECT_GE(released, 0);
    EXPECT_LT(released, 40);
}

TEST(Pacer, PassKeepsEstimatesButDropsTheSchedule) {
    Pacer pacer;
    const int64_t refresh = refreshOf(90.0);
    int64_t now = kMs;
    Pacer::Plan plan;
    Present order[kMaxPresents];
    for (int f = 0; f < 120; f++) {
        now += 5 * kMs;
        plan = pacer.begin(now, 2, refresh);
        pacer.schedule(now, plan, Mode::kExtrapolate, true, refresh, order);
        const int64_t wake = pacer.limit(now);
        if (wake > now) now = wake;
        pacer.resumed(now);
    }
    ASSERT_TRUE(plan.paced);
    const float interval = pacer.intervalMs();
    now += 5 * kMs;
    pacer.pass(now);  // the same cadence, untouched: the estimate carries on
    EXPECT_NEAR(interval, pacer.intervalMs(), interval * 0.02);
    EXPECT_EQ(0, pacer.limit(now));
}

// Paced, a window is whole refreshes: generated frames at even phases, the game's frame last
// (interpolating) or first (extrapolating) and untouched.
TEST(Pacer, PacedWindowsAreWholeRefreshes) {
    for (Mode mode : {Mode::kInterpolate, Mode::kExtrapolate}) {
        Pacer pacer;
        const int64_t refresh = refreshOf(135.0);
        pacer.compositor(0, 12 * kMs);
        pacer.presented(0, 0, 0, refresh);
        int64_t now = kMs;
        Pacer::Plan plan;
        Present order[kMaxPresents];
        int count = 0;
        for (int f = 0; f < 300; f++) {
            now += 6 * kMs;
            plan = pacer.begin(now, 4, refresh);
            count = pacer.schedule(now, plan, mode, true, refresh, order);
            const int64_t wake = pacer.limit(now);
            if (wake > now) now = wake;
            pacer.resumed(now);
        }
        ASSERT_TRUE(plan.paced);
        ASSERT_EQ(4, count);
        const int own = mode == Mode::kInterpolate ? 3 : 0;
        for (int i = 0; i < 4; i++) {
            if (i > 0) EXPECT_EQ(refresh, order[i].desired - order[i - 1].desired);
            EXPECT_EQ(i == own, order[i].own);
            EXPECT_EQ(i != own, order[i].generated);
            const float phase = mode == Mode::kInterpolate ? (i + 1) / 4.0f : i / 4.0f;
            EXPECT_FLOAT_EQ(phase, order[i].phase);
        }
    }
}

// Without history the game's frame goes out alone, untouched.
TEST(Pacer, NothingToGenerateIsJustTheGamesFrame) {
    for (Mode mode : {Mode::kInterpolate, Mode::kExtrapolate}) {
        Pacer pacer;
        const int64_t refresh = refreshOf(135.0);
        const Pacer::Plan plan = pacer.begin(kMs, 2, refresh);
        Present order[kMaxPresents];
        ASSERT_EQ(1, pacer.schedule(kMs, plan, mode, false, refresh, order));
        EXPECT_TRUE(order[0].own);
        EXPECT_FALSE(order[0].generated);
    }
}

}  // namespace
}  // namespace framefx::luwes
