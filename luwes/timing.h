/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <system/window.h>

#include <algorithm>
#include <cstdint>
#include <deque>

#include "luwes/pacer.h"

namespace framefx::luwes {

// The desired present times one target sets on its window, and when real frames reached the
// display, read back a few frames later for the pacer.
class WindowTiming {
public:
    // Luwes presents on the window this frame. BufferQueue's EGL throttle makes each queue wait
    // for the previous queued buffer's rendering; with a generated and a real present per frame
    // the second would wait for the frame just submitted, serialising the game's CPU with the
    // GPU. Without it, dequeueing a free buffer is the back-pressure.
    void take(ANativeWindow* window) {
        if (window == nullptr || unthrottled_) return;
        unthrottled_ = native_window_set_producer_throttling_enabled(window, false) == 0;
    }

    // Desired present time of the window's next queue; 0 for as soon as possible.
    void stamp(ANativeWindow* window, int64_t desired) {
        if (window == nullptr || (desired <= 0 && !stamped_)) return;
        native_window_set_buffers_timestamp(window,
                                            desired > 0 ? desired : NATIVE_WINDOW_TIMESTAMP_AUTO);
        stamped_ = desired > 0;
    }

    // The next queue is asked for `desired`: remember it, once frame timestamps are on.
    void expect(ANativeWindow* window, int64_t desired) {
        if (window == nullptr) return;
        if (!enabled_) enabled_ = native_window_enable_frame_timestamps(window, true) == 0;
        uint64_t id = 0;
        if (enabled_ && native_window_get_next_frame_id(window, &id) == 0) {
            if (pending_.size() >= kMaxPending) pending_.pop_front();
            pending_.push_back({id, desired});
        }
    }

    // Feeds the pacer the compositor's timing and the presents old enough to have reached the
    // display.
    void learn(ANativeWindow* window, Pacer& pacer, int64_t refreshNs) {
        int64_t deadline = 0;
        int64_t interval = 0;
        int64_t latency = 0;
        if (enabled_ && window != nullptr &&
            native_window_get_compositor_timing(window, &deadline, &interval, &latency) == 0) {
            pacer.compositor(deadline, latency);
        }
        // In queue order, each once the display showed or dropped it: a dropped frame never
        // gets a present time, only its release. Frames fall out of the window's short history
        // unanswered.
        while (window != nullptr && !pending_.empty()) {
            const Pending p = pending_.front();
            int64_t ready = 0;
            int64_t shown = 0;
            int64_t released = 0;
            if (native_window_get_frame_timestamps(window, p.id, nullptr, &ready, nullptr,
                                                   nullptr, nullptr, nullptr, &shown, nullptr,
                                                   &released) != 0) {
                pending_.pop_front();
                continue;
            }
            if (shown <= 0 && released <= 0) break;
            pending_.pop_front();
            pacer.presented(p.desired, std::max<int64_t>(ready, 0), std::max<int64_t>(shown, 0),
                            refreshNs);
        }
    }

    // Luwes went off: the game's own presents go out as soon as possible again, throttled as
    // before.
    void reset(ANativeWindow* window) {
        stamp(window, 0);
        pending_.clear();
        if (window != nullptr && unthrottled_) {
            native_window_set_producer_throttling_enabled(window, true);
        }
        unthrottled_ = false;
    }

private:
    static constexpr size_t kMaxPending = 32;

    struct Pending {
        uint64_t id;
        int64_t desired;
    };

    bool stamped_ = false;
    bool enabled_ = false;
    bool unthrottled_ = false;
    std::deque<Pending> pending_;
};

}  // namespace framefx::luwes
