/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/channel.h>

#include <algorithm>
#include <cstdint>

namespace framefx::luwes {

// Lowest real frame rate generation runs at. A multiplier whose base would fall below it is
// lowered on a slower panel, and Pentas hides it.
inline constexpr double kMinBaseFps = 30.0;

// Presents one real frame's interval can hold: a slow game on a fast panel fills more refreshes
// than the multiplier.
inline constexpr int kMaxPresents = kMaxMultiplier + 1;
inline constexpr int64_t kMinRefreshNs = 2'000'000;    // 500 Hz
inline constexpr int64_t kMaxRefreshNs = 50'000'000;   // 20 Hz

// Presents per real frame that keep the base at or above kMinBaseFps on a `hz` panel; 1 means
// no generation.
int EffectiveMultiplier(int multiplier, double hz);

// Real frame rate the limiter holds for `multiplier` presents per real frame.
inline double BaseFps(int multiplier, double hz) { return multiplier > 0 ? hz / multiplier : 0.0; }

// One present of a real frame's interval, in issue order.
struct Present {
    int64_t desired = 0;     // desired present time, half a refresh before the one it is for
    float phase = 0.0f;      // content time: interpolating, 0 the previous real frame and 1 this
                             // one; extrapolating, 0 this one and 1 an interval past it
    bool own = false;        // the game's image goes out here
    bool generated = false;  // carrying a generated frame; an own present without one is the
                             // game's frame as rendered
};

// Presentation timing for one target: which refreshes each real frame fills, and the content
// time each of them shows.
//
// Every refresh shows a present, and each carries the content of its own display time: real
// frames sit on a content timeline that runs a fixed latency behind the display, and the presents
// of one real frame are its window of that timeline, one per refresh, each generated at the phase
// its refresh falls on. A 60 fps game on a 135 Hz panel gets 135 evenly advancing frames, where a
// fixed two presents per frame would leave a refresh every ninth one showing the same frame
// twice. The game's own image goes out at the real end of the window; when no refresh falls
// there, it carries a generated frame too.
//
// The timeline follows the game's smoothed interval and is pulled gently toward the earliest
// time a present can make, so arrival jitter barely moves it. Refreshes come from the display's
// own present times; each present asks for the middle of the refresh before its own, so small
// errors never move it to a neighbour. A game that reaches the base is held there by the
// limiter, the one wait left, and its windows are whole refreshes: every real frame shows
// untouched. Feedback tells two kinds of lateness apart: a buffer not ready in time needs a
// longer lead, a ready buffer shown late sits behind a backlog, which one later schedule drains.
// Times are CLOCK_MONOTONIC ns; the caller sleeps and presents, so the control law has no clock
// of its own.
class Pacer {
public:
    struct Plan {
        int multiplier = 1;  // presents per real frame at the base
        bool paced = false;  // the game reaches the base: the limiter holds it there
    };

    // A real frame arrived at `now`.
    Plan begin(int64_t now, int multiplier, int64_t refreshNs);

    // The frame went out untouched; keep the estimates, drop the schedule.
    void pass(int64_t now);

    // This interval's presents, in issue order; returns the count, at least one. Without
    // `generate` it is the game's frame alone.
    int schedule(int64_t now, const Plan& plan, Mode mode, bool generate, int64_t refreshNs,
                 Present out[kMaxPresents]);

    // When the game's next frame may begin, or 0 to let it run.
    int64_t limit(int64_t now) const;

    // The game resumed at `now`, after any wait limit() asked for.
    void resumed(int64_t now);

    // A present asked for `desired` had its buffer ready at `ready` and reached the display at
    // `shown`, 0 when it was dropped; `ready` is 0 when unknown.
    void presented(int64_t desired, int64_t ready, int64_t shown, int64_t refreshNs);

    // SurfaceFlinger composes `latency` before a frame reaches the display, and its next
    // composition is due at `deadline`. A present must be in SurfaceFlinger, its buffer ready, by
    // then: queued later, it misses its refresh and lands on the next present's, which replaces
    // it unseen. Deadline plus latency is a refresh, which places them until a present is shown.
    void compositor(int64_t deadline, int64_t latency);

    // Buffers the target's presents may hold at once, queued or on screen; 0 when unknown.
    void buffers(int count) { buffers_ = count; }

    void reset();

    float intervalMs() const { return intervalMs_; }
    int64_t leadNs() const { return lead_; }
    int64_t windowNs() const { return window_; }

private:
    void observe(int64_t now);
    // The first refresh at or after t.
    int64_t refreshAfter(int64_t t, int64_t refresh) const;

    int64_t last_ = 0;
    float intervalMs_ = 0.0f;
    bool paced_ = false;
    int qualify_ = 0;
    int miss_ = 0;

    int64_t anchor_ = 0;      // display time of the window's phase 0
    int64_t window_ = 0;      // its length
    int64_t lastShown_ = 0;   // the last refresh scheduled
    int64_t refresh_ = 0;
    int64_t grid_ = 0;        // a refresh the display reported, or a stand-in until it does
    bool gridKnown_ = false;
    int64_t lead_ = 0;        // queue to display
    int64_t spare_ = INT64_MAX;  // least an on-time present left between ready and deadline
    int judged_ = 0;
    int64_t shift_ = 0;       // added once to the next anchor, to drain a backlog
    int64_t settled_ = 0;     // presents desired before this were scheduled before the last shift
    int64_t resumed_ = 0;
    int64_t work_ = 0;        // the game's frame, from resuming to presenting
    int64_t compositor_ = 0;  // composition to present
    int buffers_ = 0;
};

}  // namespace framefx::luwes
