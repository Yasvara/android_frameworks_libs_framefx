/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "pacer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace framefx::luwes {
namespace {

// Pacing engages after this many frames that reach the base, and lets go after kRelease that do
// not: a game that falls short must not be held further below its own rate.
constexpr int kEngage = 12;
constexpr int kRelease = 6;

// Real rate against the base that engages pacing, and that keeps it engaged.
constexpr float kReachRatio = 0.97f;
constexpr float kKeepRatio = 0.94f;

// Lead bounds past the compositor's latency, in half refreshes, and its starting point.
constexpr int64_t kMinLeadHalves = 1;
constexpr int64_t kMaxLeadHalves = 8;
constexpr int64_t kStartLeadHalves = 4;

// Unpaced, each interval moves the timeline this fraction of the way to its target.
constexpr int64_t kLockDivisor = 8;
// The game's frame goes out untouched within this fraction of a refresh of its content time.
constexpr int64_t kSnapDivisor = 8;
// Arrival jitter the unpaced timeline absorbs without moving, as a fraction of a refresh.
constexpr int64_t kJitterDivisor = 4;
// The lead shrinks by what the tightest of kJudged on-time presents left spare, past
// 1/kSpareDivisor of a refresh kept for GPU jitter, halved; at most 1/kShrinkDivisor of a refresh
// at a time.
constexpr int kJudged = 8;
constexpr int64_t kSpareDivisor = 4;
constexpr int64_t kShrinkDivisor = 8;

// The limiter never sleeps longer than this; past it the estimate is wrong, not the game fast.
constexpr int64_t kMaxLimiterWaitNs = 100'000'000;

int64_t clampRefresh(int64_t refreshNs) {
    return std::clamp(refreshNs, kMinRefreshNs, kMaxRefreshNs);
}

int64_t floorDiv(int64_t a, int64_t b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

}  // namespace

int EffectiveMultiplier(int multiplier, double hz) {
    int m = std::clamp(multiplier, 1, kMaxMultiplier);
    while (m > 1 && hz / m < kMinBaseFps - 1e-6) m--;
    return m;
}

void Pacer::observe(int64_t now) {
    if (last_ > 0) {
        const float ms = static_cast<float>(now - last_) / 1e6f;
        if (ms > 0.05f && ms < 200.0f) {
            intervalMs_ = intervalMs_ <= 0.0f ? ms : intervalMs_ * 0.9f + ms * 0.1f;
        }
    }
    last_ = now;
}

int64_t Pacer::refreshAfter(int64_t t, int64_t refresh) const {
    return grid_ - floorDiv(grid_ - t, refresh) * refresh;
}

Pacer::Plan Pacer::begin(int64_t now, int multiplier, int64_t refreshNs) {
    observe(now);
    if (resumed_ > 0 && now > resumed_) {
        const int64_t work = now - resumed_;
        work_ = work_ <= 0 ? work : (work_ * 7 + work) / 8;
    }
    resumed_ = 0;
    const double hz = 1e9 / static_cast<double>(clampRefresh(refreshNs));
    Plan plan;
    plan.multiplier = EffectiveMultiplier(multiplier, hz);
    if (plan.multiplier <= 1) {
        paced_ = false;
        qualify_ = miss_ = 0;
        return plan;
    }
    const double base = hz / plan.multiplier;
    const double realFps = intervalMs_ > 0.0f ? 1000.0 / intervalMs_ : 0.0;
    const bool holds = realFps >= base * static_cast<double>(paced_ ? kKeepRatio : kReachRatio);
    if (holds) {
        miss_ = 0;
        if (++qualify_ >= kEngage) paced_ = true;
    } else {
        qualify_ = 0;
        if (++miss_ >= kRelease) paced_ = false;
    }
    plan.paced = paced_;
    return plan;
}

void Pacer::pass(int64_t now) {
    observe(now);
    anchor_ = 0;
}

int Pacer::schedule(int64_t now, const Plan& plan, Mode mode, bool generate, int64_t refreshNs,
                    Present out[kMaxPresents]) {
    const int64_t refresh = clampRefresh(refreshNs);
    refresh_ = refresh;
    const int64_t half = refresh / 2;
    if (lead_ <= 0) lead_ = compositor_ + kStartLeadHalves * half;
    lead_ = std::clamp(lead_, compositor_ + kMinLeadHalves * half,
                       compositor_ + kMaxLeadHalves * half);
    if (grid_ <= 0) grid_ = now;  // stands in until the display reports a refresh
    const bool interpolate = mode == Mode::kInterpolate;
    // Paced, the window is exactly the base's interval, the limiter's; otherwise the game's own.
    const int64_t interval = intervalMs_ > 0.0f ? static_cast<int64_t>(intervalMs_ * 1e6f)
                                                : plan.multiplier * refresh;
    const int64_t window = plan.paced ? plan.multiplier * refresh : std::max<int64_t>(interval, 1);

    // The first refresh a present queued now still makes, and the lowest anchor whose window
    // takes no earlier one: interpolating, the window opens after its anchor, extrapolating at it.
    const int64_t earliest = refreshAfter(now + lead_ + half, refresh);
    const int64_t lower = interpolate ? earliest - refresh : earliest - refresh + 1;
    // Whatever the refreshes' phase, a window opening here has every refresh within reach, with
    // some arrival jitter to spare.
    const int64_t target = now + lead_ + half + refresh / kJitterDivisor;
    int64_t anchor = target;
    if (anchor_ > 0) {
        // Continue the timeline; unpaced, lean toward the target. A schedule ahead of the game
        // only adds latency, and one too late skips content: the bounds.
        anchor = anchor_ + window_;
        if (!plan.paced) anchor += (target - anchor) / kLockDivisor;
    }
    anchor = std::clamp(anchor, lower, std::max(lower, target + refresh));
    const bool shifted = shift_ > 0;
    anchor += shift_;
    shift_ = 0;
    if (plan.paced) {
        // On a refresh: the windows are whole refreshes and the game's frame lands on one.
        anchor = std::max(refreshAfter(anchor - half, refresh),
                          interpolate ? earliest - refresh : earliest);
    }

    // The refreshes the window covers, never one an earlier interval took. Interpolating keeps
    // the latest of too many, so the game's frame stays at the end.
    int64_t first = refreshAfter(interpolate ? anchor + 1 : anchor, refresh);
    if (lastShown_ > 0) first = std::max(first, refreshAfter(lastShown_ + half, refresh));
    first = std::max(first, earliest);
    const int64_t span = anchor + window - first;
    int count = interpolate ? (span >= 0 ? static_cast<int>(span / refresh) + 1 : 0)
                            : (span > 0 ? static_cast<int>((span + refresh - 1) / refresh) : 0);
    // A game faster than the panel can leave a window without a refresh; its frame still goes
    // out, at the next one, as it is.
    const bool empty = count == 0;
    if (empty) count = 1;
    if (!generate && count > 1) {
        if (interpolate) first += (count - 1) * refresh;
        count = 1;
    }
    // Every present holds its buffer until the next one is latched, a compositor latency before
    // that one's refresh. The window takes only buffers free now, so queueing never blocks the
    // game; a refresh it leaves out shows the previous present again.
    int room = kMaxPresents;
    if (buffers_ > 0) {
        const int64_t freed = now + compositor_ - refresh;
        const int held = lastShown_ > freed ? static_cast<int>((lastShown_ - freed - 1) / refresh) + 1
                                            : 0;
        room = std::clamp(buffers_ - held, 1, kMaxPresents);
    }
    if (count > room) {
        if (interpolate) first += (count - room) * refresh;
        count = room;
    }
    const int own = interpolate ? count - 1 : 0;
    const int64_t snap = refresh / kSnapDivisor;
    for (int i = 0; i < count; i++) {
        const int64_t at = first + i * refresh;
        const int64_t content = at - anchor;  // into the window
        Present& p = out[i];
        p.desired = at - half;
        p.own = i == own;
        const bool real = interpolate ? content >= window - snap : content <= snap;
        p.generated = generate && !empty && (!p.own || !real);
        p.phase = p.generated ? std::clamp(static_cast<float>(content) /
                                                   static_cast<float>(window),
                                           0.0f, 1.0f)
                              : (interpolate ? 1.0f : 0.0f);
    }
    if (shifted) settled_ = out[0].desired;
    anchor_ = anchor;
    window_ = window;
    lastShown_ = first + (count - 1) * refresh;
    return count;
}

int64_t Pacer::limit(int64_t now) const {
    if (!paced_ || anchor_ <= 0) return 0;
    // The next window's first refresh follows this one's last. Its present must be queued the
    // lead and half a refresh before it, and building the frame takes the game about work_;
    // half a refresh spare.
    const int64_t wake = lastShown_ - lead_ - work_;
    return wake > now && wake - now < kMaxLimiterWaitNs ? wake : 0;
}

void Pacer::resumed(int64_t now) {
    resumed_ = now;
}

void Pacer::compositor(int64_t deadline, int64_t latency) {
    compositor_ = std::max<int64_t>(latency, 0);
    if (!gridKnown_ && deadline > 0 && latency > 0) grid_ = deadline + latency;
}

void Pacer::presented(int64_t desired, int64_t ready, int64_t shown, int64_t refreshNs) {
    const int64_t refresh = clampRefresh(refreshNs);
    if (desired <= 0 || lead_ <= 0) return;
    if (shown > 0) {
        grid_ = shown;
        gridKnown_ = true;
    }
    // On time, a present shows at the refresh half a refresh past its desired time, its buffer
    // ready before the compositor's deadline for that refresh.
    const int64_t due = desired + refresh / 2;
    const int64_t deadline = due - compositor_ - refresh / 4;
    if (ready > deadline) {
        // The buffer itself came late, shown late or dropped for a later one: the lead is
        // shorter than the GPU needs.
        lead_ += std::min(ready - deadline, kMaxLeadHalves * refresh / 2) / 2;
        judged_ = 0;
        spare_ = INT64_MAX;
    } else if (shown <= 0) {
        return;  // ready and still dropped: whatever took its refresh is judged on its own
    } else if (shown - due > refresh / 2) {
        // Ready in time and still late: a backlog of earlier presents. Push the schedule back
        // once, and judge again only presents scheduled after that.
        if (desired >= settled_ && shift_ == 0) shift_ = std::min(shown - due, 4 * refresh);
    } else if (ready > 0) {
        // On time: give back part of what the GPU left spare, keeping some for its jitter. The
        // first present of a window has the least; a lead that shrinks until a present misses
        // would drop one every few frames.
        spare_ = std::min(spare_, deadline - ready);
        if (++judged_ >= kJudged) {
            const int64_t give = (spare_ - refresh / kSpareDivisor) / 2;
            lead_ -= std::clamp(give, int64_t{0}, refresh / kShrinkDivisor);
            judged_ = 0;
            spare_ = INT64_MAX;
        }
    } else {
        lead_ -= refresh / 64;
    }
    lead_ = std::clamp(lead_, compositor_ + kMinLeadHalves * refresh / 2,
                       compositor_ + kMaxLeadHalves * refresh / 2);
}

void Pacer::reset() {
    *this = Pacer();
}

}  // namespace framefx::luwes
