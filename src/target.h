/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstdint>

#include "channel.h"
#include "luwes/pacer.h"

namespace framefx {

// Tells a game render loop from a window that redraws on demand: the loop presents continuously.
// Thread names cannot tell them apart (engines name their render thread like HWUI's). A loading
// screen presenting slowly only delays engagement.
class EngagementGate {
public:
    bool check(int64_t now);

private:
    static constexpr int64_t kMaxIntervalNs = 50'000'000;
    static constexpr int kRunFrames = 24;

    bool engaged_ = false;
    int run_ = 0;
    int64_t last_ = 0;
};

// Real and presented frame rates over about a second, for the status page.
class Meter {
public:
    // Counts one real frame and the presents it made. True about once a second, when the rates
    // were just refreshed.
    bool frame(int64_t now, int presents);
    float realFps() const { return realFps_; }
    float presentedFps() const { return presentedFps_; }

private:
    int64_t start_ = 0;
    int frames_ = 0;
    int presents_ = 0;
    float realFps_ = 0.0f;
    float presentedFps_ = 0.0f;
};

// Sleeps until `when` on CLOCK_MONOTONIC; returns at once for a time already past.
void SleepUntil(int64_t when);

// CLOCK_MONOTONIC now, ns.
int64_t Now();

// Lifecycle of one presentation target (an EGL window surface or a swapchain), apart from the
// graphics API: when to run, when size-dependent resources are stale, when history is usable,
// and whether to generate.
class Target {
public:
    struct Frame {
        int64_t now = 0;
        int width = 0;
        int height = 0;
        uint32_t transform = 0;
        int64_t refreshNs = 0;
    };

    struct Decision {
        bool process = false;       // capture and analyse this real frame
        bool resize = false;        // (re)create size-dependent resources first
        bool releaseLuwes = false;  // Luwes went off: free what only it needs
        bool generate = false;      // generate frames around it
        luwes::Pacer::Plan plan;
    };

    Decision begin(const Frame& frame, const LuwesSettings& luwes);

    // The real frame was captured and analysed: the next one may generate from it.
    void captured() { history_ = true; }

    // A swapchain replaced its predecessor: keep the engagement and the pacing estimates, start
    // over on everything that depends on the images.
    void inherit(const Target& old);

    bool history() const { return history_; }
    luwes::Pacer& pacer() { return pacer_; }

private:
    EngagementGate gate_;
    luwes::Pacer pacer_;
    int width_ = 0;
    int height_ = 0;
    uint32_t transform_ = 0;
    bool luwesOn_ = false;
    bool history_ = false;
};

}  // namespace framefx
