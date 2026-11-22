/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "src/safety.h"

#include <android-base/stringprintf.h>
#include <fcntl.h>
#include <log/log.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <string>

#include "src/engine.h"

namespace framefx::safety {
namespace {

constexpr const char* kNames[kFeatureCount] = {"shading-rate", "rays", "transparency-cut"};
// About ten seconds of play.
constexpr uint32_t kCleanPresents = 600;

std::mutex gLock;
bool gDecided[kFeatureCount] = {};
bool gAllowed[kFeatureCount] = {};
bool gArmed[kFeatureCount] = {};
std::atomic<uint32_t> gPresents{0};
std::atomic<bool> gCleared{false};

std::string markerOf(Feature f) {
    return android::base::StringPrintf("/data/user/%u/%s/cache/framefx-running-%s",
                                       getuid() / 100000, Engine::get().package().c_str(),
                                       kNames[f]);
}

}  // namespace

bool allowed(Feature f) {
    std::lock_guard lock(gLock);
    if (gDecided[f]) return gAllowed[f];
    gDecided[f] = true;
    const std::string marker = markerOf(f);
    if (access(marker.c_str(), F_OK) == 0) {
        // The last start ended with this feature on and the marker still set.
        unlink(marker.c_str());
        ALOGW("%s: %s off for this start: the game stopped before it ran clean last time",
              Engine::get().package().c_str(), kNames[f]);
        gAllowed[f] = false;
        return false;
    }
    const int fd = open(marker.c_str(), O_CREAT | O_WRONLY | O_CLOEXEC, 0600);
    if (fd >= 0) close(fd);
    gArmed[f] = fd >= 0;
    gAllowed[f] = true;
    return true;
}

void onPresent() {
    if (gCleared.load(std::memory_order_relaxed)) return;
    if (gPresents.fetch_add(1, std::memory_order_relaxed) + 1 < kCleanPresents) return;
    std::lock_guard lock(gLock);
    if (gCleared.exchange(true)) return;
    for (int f = 0; f < kFeatureCount; f++) {
        if (gArmed[f]) unlink(markerOf(static_cast<Feature>(f)).c_str());
        gArmed[f] = false;
    }
}

}  // namespace framefx::safety
