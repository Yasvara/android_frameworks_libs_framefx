/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "channel.h"
#include "kelir/controller.h"

namespace framefx {

// What Pentas asked of this process, as of one channel sequence.
struct Config {
    uint32_t sequence = 0;
    LuwesSettings luwes;
    std::shared_ptr<const kelir::Plan> kelir = std::make_shared<kelir::Plan>();
    LapisSettings lapis;
    UpscaleSettings upscale;
    TextureSettings texture;
    // What a decoder's frames get; apart from `kelir`, which the app's own presents get.
    std::shared_ptr<const kelir::Plan> videoKelir = std::make_shared<kelir::Plan>();
    VideoSettings video;
};

// What a present path reports of Kelir on its target.
struct KelirTargetStatus {
    KelirState state = KelirState::kOff;
    KelirReason reason = KelirReason::kNone;
    Api api = Api::kNone;
    FormatClass format = FormatClass::kUnorm8;
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t rotation = 0;
    KelirPath path = KelirPath::kNone;
    uint64_t signature = 0;
    uint32_t gpuNs = 0;

    bool operator==(const KelirTargetStatus& o) const;
};

// One per armed process: the channel from Pentas and the status page back to it.
//
// The broker hands the channel over once; a thread then sleeps on the channel's futex and
// republishes the decoded settings whenever Pentas changes them. Presents read them through
// Watch, which costs one atomic load until something changes.
class Engine {
public:
    static Engine& get();

    // Opens the channel through the broker. Idempotent; later calls return at once.
    void attach();

    // Maps a channel (read-only) and a status page (read-write) and starts watching the
    // channel. The broker's descriptors take this path; a test hands in its own.
    bool adopt(int channelFd, int statusFd);

    StatusPage& status() { return status_; }
    const std::string& package() const { return package_; }

    uint32_t generation() const { return generation_.load(std::memory_order_acquire); }
    Config config() const;
    // The memory shared with org.yasvara.alih: read-write maps of framefx/alih.h's regions.
    bool openAlih(uint8_t** ring, uint8_t** table);

    // Whether any feature is on: the one check an idle present makes.
    bool active() const { return active_.load(std::memory_order_relaxed); }

    // Whether a decoder's frames have anything to go through: the one check a rendered frame makes.
    bool videoWanted() const { return videoWanted_.load(std::memory_order_relaxed); }

    // Kelir's state on a target, for the status page; cheap when nothing changed.
    void reportKelir(const KelirTargetStatus& status);
    // Lapis's state, for the status page; cheap when nothing changed.
    void reportLapis(const LapisStatus& status);
    // Upscaling's state, for the status page; cheap when nothing changed.
    void reportUpscale(const UpscaleStatus& status);
    // The video path's state, for the status page.
    void reportVideo(const VideoStatus& status);

    // Runs `listener` on the watcher thread after each publication, off every present path.
    void listen(std::function<void(const Config&)> listener);

    // A present path's view of the configuration; refresh() is the per-present check.
    class Watch {
    public:
        // True when the configuration changed since the last call.
        bool refresh(const Engine& engine) {
            const uint32_t g = engine.generation();
            if (g == seen_) return false;
            seen_ = g;
            config_ = engine.config();
            return true;
        }
        const Config& config() const { return config_; }

    private:
        uint32_t seen_ = 0;
        Config config_;
    };

private:
    Engine();
    void watch();
    void publish(const uint8_t* snapshot);

    std::once_flag once_;
    const std::string package_;
    const uint8_t* channel_ = nullptr;  // read-only shared mapping
    StatusPage status_;

    mutable std::mutex lock_;
    Config config_;
    kelir::Controller kelir_;
    uint32_t kelirShape_ = 0;
    kelir::Controller videoKelir_;
    std::atomic<uint32_t> generation_{0};
    std::atomic<bool> active_{false};
    std::atomic<bool> videoWanted_{false};

    std::mutex listenersLock_;
    std::vector<std::function<void(const Config&)>> listeners_;

    std::mutex statusLock_;
    KelirStatus kelirStatus_;
    KelirTargetStatus kelirTarget_;
    LapisStatus lapisStatus_;
    UpscaleStatus upscaleStatus_;
};

}  // namespace framefx
