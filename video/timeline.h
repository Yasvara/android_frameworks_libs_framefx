/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstdint>

// When each frame of a video goes out, generated ones included. A player releases a frame ahead of
// its time, and interpolation only has the pair once the newer frame is here, so the frames between
// the two are due from about the moment it arrives. The timeline lays them on a grid of the display's
// period from the frame before, drops those it can no longer make, and where the player's lead is too
// short for the first of them, delays the whole video by the difference, within what A/V sync hides.
namespace framefx::video {

inline constexpr int kMaxPresents = 6;  // 24 fps on a 120 Hz panel is five
inline constexpr int64_t kMaxGapNs = 100'000'000;  // a longer gap is a pause or a seek
inline constexpr int64_t kMaxDelayNs = 40'000'000;  // BT.1359: sound 45 ms early goes unnoticed
inline constexpr int64_t kDelayFallNs = 100'000;  // per frame: a delay in use shrinks unnoticed

struct Present {
    int64_t when = 0;   // display time
    float phase = 1.0f; // 0 the frame before, 1 this one
    bool real = true;
};

class Timeline {
public:
    // The frame with timestamp `t` arrived at `now`. `periodNs` is the display period to fill, 0
    // for this frame alone (off, or a cut). `readyNs` is how long before its time a present must
    // start. Writes the presents, the real frame last; returns how many.
    int plan(int64_t t, int64_t now, int64_t periodNs, int64_t readyNs, float smoothness,
             Present out[kMaxPresents]);

    void reset();
    int64_t delay() const { return delay_; }

private:
    int64_t last_ = 0;
    bool hasLast_ = false;
    int64_t delay_ = 0;
};

}  // namespace framefx::video
