/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "target.h"

#include <errno.h>
#include <time.h>

namespace framefx {

bool EngagementGate::check(int64_t now) {
    if (engaged_) return true;
    if (last_ > 0) {
        const int64_t interval = now - last_;
        run_ = interval > 50'000 && interval < kMaxIntervalNs ? run_ + 1 : 0;
    }
    last_ = now;
    engaged_ = run_ >= kRunFrames;
    return engaged_;
}

Target::Decision Target::begin(const Frame& frame, const LuwesSettings& luwes) {
    Decision d;
    if (frame.width != width_ || frame.height != height_ || frame.transform != transform_) {
        // A new size, a rotation or a render-scale change: last frame no longer lines up.
        d.resize = true;
        width_ = frame.width;
        height_ = frame.height;
        transform_ = frame.transform;
        history_ = false;
    }
    const bool engaged = gate_.check(frame.now);
    if (luwesOn_ && !luwes.on()) {
        d.releaseLuwes = true;
        history_ = false;
        pacer_.reset();
    }
    luwesOn_ = luwes.on();
    if (!luwesOn_ || !engaged) {
        pacer_.pass(frame.now);
        history_ = false;
        return d;
    }
    d.process = true;
    d.plan = pacer_.begin(frame.now, luwes.multiplier, frame.refreshNs);
    d.generate = history_ && d.plan.multiplier > 1;
    return d;
}

void Target::inherit(const Target& old) {
    gate_ = old.gate_;
    pacer_ = old.pacer_;
    luwesOn_ = old.luwesOn_;
}

bool Meter::frame(int64_t now, int presents) {
    if (start_ == 0) start_ = now;
    frames_++;
    presents_ += presents;
    const int64_t elapsed = now - start_;
    if (elapsed < 1'000'000'000) return false;
    const float seconds = static_cast<float>(elapsed) / 1e9f;
    realFps_ = static_cast<float>(frames_) / seconds;
    presentedFps_ = static_cast<float>(presents_) / seconds;
    start_ = now;
    frames_ = 0;
    presents_ = 0;
    return true;
}

void SleepUntil(int64_t when) {
    timespec ts;
    ts.tv_sec = static_cast<time_t>(when / 1'000'000'000);
    ts.tv_nsec = static_cast<long>(when % 1'000'000'000);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
    }
}

int64_t Now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

}  // namespace framefx
