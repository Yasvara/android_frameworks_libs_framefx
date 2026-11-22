/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "video/timeline.h"

#include <algorithm>

namespace framefx::video {

void Timeline::reset() {
    hasLast_ = false;
    delay_ = 0;
}

int Timeline::plan(int64_t t, int64_t now, int64_t periodNs, int64_t readyNs, float smoothness,
                   Present out[kMaxPresents]) {
    const int64_t gap = t - last_;
    const bool continuous = hasLast_ && gap > 0 && gap <= kMaxGapNs;
    last_ = t;
    hasLast_ = true;
    int count = 0;
    if (periodNs <= 0) delay_ = 0;
    // A frame the display period already covers has nothing to generate between.
    if (continuous && periodNs > 0 && gap > periodNs * 5 / 4) {
        // The first generated frame is due a period after the frame before.
        const int64_t need = std::clamp(gap - periodNs + readyNs - (t - now), int64_t{0}, kMaxDelayNs);
        delay_ = need > delay_ ? need : std::max(need, delay_ - kDelayFallNs);
        const int64_t base = t - gap;
        for (int k = 1; count < kMaxPresents - 1; k++) {
            const int64_t at = base + k * periodNs;
            if (at >= t - periodNs / 3) break;
            const int64_t when = at + delay_;
            if (when - readyNs < now) continue;
            out[count].when = when;
            out[count].phase = static_cast<float>(k * periodNs) / static_cast<float>(gap) * smoothness;
            out[count].real = false;
            count++;
        }
    }
    out[count].when = t + delay_;
    out[count].phase = 1.0f;
    out[count].real = true;
    return count + 1;
}

}  // namespace framefx::video
