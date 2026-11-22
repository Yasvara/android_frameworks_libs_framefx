/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include <android-base/properties.h>
#include <ucontext.h>
#include <signal.h>
#include <dlfcn.h>
#include "engine.h"

#include <android-base/file.h>
#include <android-base/unique_fd.h>
#include <binder/Binder.h>
#include <binder/IServiceManager.h>
#include <binder/Parcel.h>
#include <framefx/alih.h>
#include <linux/futex.h>
#include <log/log.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <thread>

namespace framefx {
namespace {

const char* modeName(Mode mode) {
    return mode == Mode::kInterpolate ? "interpolate" : "extrapolate";
}

// The editor's readouts of a plan: the swatch after the whole colour band and each colour node's
// own curve, at the node's place in the stack.
void inspect(const kelir::Plan& plan, KelirStatus* s) {
    memset(s->swatch, 0, sizeof(s->swatch));
    memset(s->curves, 0, sizeof(s->curves));
    if (!plan.inspect) return;
    kelir::Swatch(plan.colour, plan.colourCount, kelir::Primaries::kSrgb, s->swatch);
    for (int i = 0; i < plan.colourCount; i++) {
        kelir::Curve(plan.colour[i], kelir::Primaries::kSrgb, s->curves[plan.colourNode[i]]);
    }
}

std::string processName() {
    std::string cmdline;
    if (!android::base::ReadFileToString("/proc/self/cmdline", &cmdline)) return "?";
    return cmdline.substr(0, cmdline.find('\0'));
}

}  // namespace

namespace {
// Debug aid: games that defeat debuggerd's unwinder still say where they fault.
struct sigaction gPrevSegv;
void LogFrame(const char* what, uintptr_t pc) {
    Dl_info info = {};
    if (pc != 0 && dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_fname != nullptr) {
        ALOGE("segv probe %s %s+0x%zx (%s)", what, info.dli_fname,
              pc - reinterpret_cast<uintptr_t>(info.dli_fbase),
              info.dli_sname != nullptr ? info.dli_sname : "?");
    } else {
        ALOGE("segv probe %s 0x%zx", what, pc);
    }
}
void OnSegv(int sig, siginfo_t* si, void* uc) {
    auto* ctx = static_cast<ucontext_t*>(uc);
    ALOGE("segv probe: addr %p tid %d", si->si_addr, gettid());
    LogFrame("pc", ctx->uc_mcontext.pc);
    LogFrame("lr", ctx->uc_mcontext.regs[30]);
    // Frame-pointer walk, bounded: enough to name framefx's caller.
    auto* fp = reinterpret_cast<uintptr_t*>(ctx->uc_mcontext.regs[29]);
    for (int i = 0; i < 12 && fp != nullptr && (reinterpret_cast<uintptr_t>(fp) & 7) == 0; i++) {
        LogFrame("fp", fp[1]);
        auto* next = reinterpret_cast<uintptr_t*>(fp[0]);
        if (next <= fp) break;
        fp = next;
    }
    sigaction(sig, &gPrevSegv, nullptr);
    raise(sig);
}
void MaybeProbeSegv() {
    if (!android::base::GetBoolProperty("debug.framefx.segv", false)) return;
    struct sigaction sa = {};
    sa.sa_sigaction = OnSegv;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, &gPrevSegv);
}
}  // namespace

Engine::Engine() : package_(processName()) { MaybeProbeSegv(); }

Engine& Engine::get() {
    static Engine* engine = new Engine();
    return *engine;
}

void Engine::attach() {
    std::call_once(once_, [this] {
        if (channel_ != nullptr) return;
        using android::Parcel;
        const android::sp<android::IBinder> broker =
                android::defaultServiceManager()->checkService(android::String16(kServiceName));
        if (broker == nullptr) {
            ALOGW("%s: armed but the broker is not running; staying off", package_.c_str());
            return;
        }
        Parcel data;
        Parcel reply;
        data.writeInterfaceToken(android::String16(kInterface));
        android::base::unique_fd channelFd;
        android::base::unique_fd statusFd;
        if (broker->transact(kOpen, data, &reply) != android::OK || reply.readInt32() != 0 ||
            reply.readUniqueFileDescriptor(&channelFd) != android::OK ||
            reply.readUniqueFileDescriptor(&statusFd) != android::OK) {
            ALOGW("%s: the broker has no channel for this process", package_.c_str());
            return;
        }
        adopt(channelFd.get(), statusFd.get());
    });
}

bool Engine::openAlih(uint8_t** ring, uint8_t** table) {
    using android::Parcel;
    const android::sp<android::IBinder> broker =
            android::defaultServiceManager()->checkService(android::String16(kServiceName));
    if (broker == nullptr) return false;
    // The broker watches it to learn this process is gone.
    static const android::sp<android::BBinder> token = android::sp<android::BBinder>::make();
    Parcel data;
    Parcel reply;
    data.writeInterfaceToken(android::String16(kInterface));
    data.writeStrongBinder(token);
    android::base::unique_fd ringFd;
    android::base::unique_fd tableFd;
    if (broker->transact(kAlihOpen, data, &reply) != android::OK || reply.readInt32() != 0 ||
        reply.readUniqueFileDescriptor(&ringFd) != android::OK ||
        reply.readUniqueFileDescriptor(&tableFd) != android::OK) {
        return false;
    }
    void* r = mmap(nullptr, alih::kRingSize, PROT_READ | PROT_WRITE, MAP_SHARED, ringFd.get(), 0);
    void* t = mmap(nullptr, alih::kTableSize, PROT_READ | PROT_WRITE, MAP_SHARED, tableFd.get(), 0);
    if (r == MAP_FAILED || t == MAP_FAILED) {
        if (r != MAP_FAILED) munmap(r, alih::kRingSize);
        if (t != MAP_FAILED) munmap(t, alih::kTableSize);
        return false;
    }
    *ring = static_cast<uint8_t*>(r);
    *table = static_cast<uint8_t*>(t);
    return true;
}

bool Engine::adopt(int channelFd, int statusFd) {
    void* channel = mmap(nullptr, kChannelSize, PROT_READ, MAP_SHARED, channelFd, 0);
    void* page = mmap(nullptr, kStatusSize, PROT_READ | PROT_WRITE, MAP_SHARED, statusFd, 0);
    if (channel == MAP_FAILED || page == MAP_FAILED) {
        ALOGW("%s: channel map failed", package_.c_str());
        if (channel != MAP_FAILED) munmap(channel, kChannelSize);
        if (page != MAP_FAILED) munmap(page, kStatusSize);
        return false;
    }
    status_.attach(static_cast<uint8_t*>(page));
    channel_ = static_cast<const uint8_t*>(channel);
    uint8_t snapshot[kChannelSize];
    if (SnapshotChannel(channel_, snapshot)) publish(snapshot);
    std::thread([this] { watch(); }).detach();
    ALOGI("%s: armed, channel open", package_.c_str());
    return true;
}

void Engine::watch() {
    pthread_setname_np(pthread_self(), "framefx");
    auto* word = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(channel_ + kSequenceOffset));
    uint32_t seen = config().sequence;
    uint8_t snapshot[kChannelSize];
    for (;;) {
        const uint32_t now = __atomic_load_n(word, __ATOMIC_ACQUIRE);
        if (now != seen && (now & 1u) == 0 && SnapshotChannel(channel_, snapshot)) {
            publish(snapshot);
            seen = ReadU32(snapshot + kSequenceOffset);
            continue;
        }
        // Returns on the broker's wake, or at once if the word already moved on.
        syscall(SYS_futex, word, FUTEX_WAIT, now, nullptr, nullptr, 0);
    }
}

void Engine::publish(const uint8_t* snapshot) {
    Sections sections;
    if (!DecodeChannel(snapshot, &sections)) {
        ALOGW("%s: malformed channel ignored", package_.c_str());
        return;
    }
    Config next;
    next.sequence = sections.sequence;
    if (sections.luwes != nullptr &&
        !DecodeLuwes(sections.luwes, sections.luwesLength, &next.luwes)) {
        ALOGW("%s: malformed Luwes section; Luwes off", package_.c_str());
        next.luwes = LuwesSettings();
    }
    if (sections.lapis != nullptr &&
        !DecodeLapis(sections.lapis, sections.lapisLength, &next.lapis)) {
        ALOGW("%s: malformed Lapis section; Lapis off", package_.c_str());
        next.lapis.mode = LapisMode::kOff;
    }
    // Lapis's auto default feeds the other features; with none of them on, it records nothing.
    if (sections.none()) next.lapis.mode = LapisMode::kOff;
    if (sections.upscale != nullptr &&
        !DecodeUpscale(sections.upscale, sections.upscaleLength, &next.upscale)) {
        ALOGW("%s: malformed Upscale section; upscaling off", package_.c_str());
        next.upscale.mode = UpscaleMode::kOff;
    }
    if (sections.texture != nullptr &&
        !DecodeTexture(sections.texture, sections.textureLength, &next.texture)) {
        ALOGW("%s: malformed Texture section; the game's sampling", package_.c_str());
        next.texture = TextureSettings();
    }
    // debug.framefx.shading = <rate 1..3>[a]: the shading rate for A/B runs, over Pentas's.
    static const std::string shading = android::base::GetProperty("debug.framefx.shading", "");
    if (!shading.empty() && shading[0] >= '0' && shading[0] <= '3') {
        next.texture.shadingRate = shading[0] - '0';
        next.texture.shadingAdaptive = next.texture.shadingRate != 0 && shading.size() > 1 &&
                shading[1] == 'a';
    }
    const uint32_t rejected = kelir_.rejectedRevision();
    const KelirReject reason = kelir_.rejectReason();
    if (kelir_.update(sections.kelir, sections.kelirLength)) {
        // Live edits arrive at the editor's frame rate: log the stack's shape, not every value.
        const kelir::Plan& plan = *kelir_.plan();
        const uint32_t shape = static_cast<uint32_t>(plan.detailCount) |
                static_cast<uint32_t>(plan.colourCount) << 4 |
                static_cast<uint32_t>(plan.screenCount) << 8 |
                static_cast<uint32_t>(plan.sceneCount) << 12;
        if (shape != kelirShape_) {
            kelirShape_ = shape;
            ALOGI("%s: kelir %d detail, %d colour, %d screen, %d scene", package_.c_str(),
                  plan.detailCount, plan.colourCount, plan.screenCount, plan.sceneCount);
        }
    } else if (kelir_.rejectedRevision() != rejected || kelir_.rejectReason() != reason) {
        ALOGW("%s: kelir revision %u rejected (%d); keeping revision %u", package_.c_str(),
              kelir_.rejectedRevision(), static_cast<int>(kelir_.rejectReason()),
              kelir_.appliedRevision());
    }
    next.kelir = kelir_.plan();
    const uint32_t videoRejected = videoKelir_.rejectedRevision();
    if (videoKelir_.update(sections.videoKelir, sections.videoKelirLength)) {
        ALOGV("%s: video kelir revision %u", package_.c_str(), videoKelir_.appliedRevision());
    } else if (videoKelir_.rejectedRevision() != videoRejected) {
        ALOGW("%s: video kelir revision %u rejected (%d); keeping revision %u", package_.c_str(),
              videoKelir_.rejectedRevision(), static_cast<int>(videoKelir_.rejectReason()),
              videoKelir_.appliedRevision());
    }
    next.videoKelir = videoKelir_.plan();
    if (sections.video != nullptr &&
        !DecodeVideo(sections.video, sections.videoLength, &next.video)) {
        ALOGW("%s: malformed Video section; motion and upscaling off", package_.c_str());
        next.video = VideoSettings();
    }
    videoWanted_.store(!next.videoKelir->empty() || next.video.on(), std::memory_order_relaxed);
    {
        // The editor reads one Kelir block: an app filtered for its video reports that stack.
        const kelir::Controller& shown = sections.videoKelir != nullptr ? videoKelir_ : kelir_;
        std::lock_guard lock(statusLock_);
        kelirStatus_.appliedRevision = shown.appliedRevision();
        kelirStatus_.rejectedRevision = shown.rejectedRevision();
        kelirStatus_.reject = shown.rejectReason();
        if (shown.plan()->empty()) {
            kelirStatus_.state = KelirState::kOff;
            kelirTarget_ = KelirTargetStatus();
        } else if (kelirStatus_.state == KelirState::kOff) {
            kelirStatus_.state = KelirState::kWaiting;
        }
        inspect(*shown.plan(), &kelirStatus_);
        status_.writeKelir(kelirStatus_);
    }
    {
        std::lock_guard lock(lock_);
        if (config_.luwes != next.luwes) {
            if (next.luwes.on()) {
                ALOGI("%s: luwes %dx %s, HUD protection %s, %s motion", package_.c_str(),
                      next.luwes.multiplier, modeName(next.luwes.mode),
                      next.luwes.hudProtection ? "on" : "off",
                      next.luwes.ownEngine ? "own" : "AFME");
            } else {
                ALOGI("%s: luwes off", package_.c_str());
            }
        }
        if (config_.lapis != next.lapis) {
            ALOGI("%s: lapis %s%s%s", package_.c_str(), next.lapis.on() ? "auto" : "off",
                  next.lapis.hideHud ? ", HUD hidden" : "",
                  next.lapis.wholeScreen ? ", Kelir on the whole screen" : "");
        }
        if (config_.upscale != next.upscale) {
            ALOGI("%s: upscale %s", package_.c_str(), next.upscale.on() ? "auto" : "off");
        }
        if (config_.video != next.video) {
            ALOGI("%s: video motion %d Hz (%s), smoothness %.2f, upscale %s", package_.c_str(),
                  next.video.motionHz, next.video.afme ? "AFME" : "own", next.video.smoothness,
                  next.video.upscale ? "on" : "off");
        }
        if (config_.texture != next.texture) {
            ALOGI("%s: texture AF %dx, LOD bias %+.3f, shading rate %d%s%s", package_.c_str(),
                  next.texture.anisotropy, next.texture.lodBiasEighths / 8.0,
                  next.texture.shadingRate, next.texture.shadingAdaptive ? " adaptive" : "",
                  config_.sequence != 0 ? " (from the next start)" : "");
        }
        config_ = next;
    }
    generation_.fetch_add(1, std::memory_order_release);
    active_.store(next.luwes.on() || !next.kelir->empty() || next.kelir->sceneCount > 0 ||
                          (next.upscale.written && next.upscale.on()),
                  std::memory_order_relaxed);
    std::lock_guard lock(listenersLock_);
    for (const auto& listener : listeners_) listener(next);
}

void Engine::listen(std::function<void(const Config&)> listener) {
    std::lock_guard lock(listenersLock_);
    listeners_.push_back(std::move(listener));
}

bool KelirTargetStatus::operator==(const KelirTargetStatus& o) const {
    return state == o.state && reason == o.reason && api == o.api && format == o.format &&
            width == o.width && height == o.height && rotation == o.rotation && path == o.path &&
            signature == o.signature && gpuNs == o.gpuNs;
}

void Engine::reportUpscale(const UpscaleStatus& s) {
    std::lock_guard lock(statusLock_);
    if (s == upscaleStatus_) return;
    upscaleStatus_ = s;
    status_.writeUpscale(s);
}

void Engine::reportLapis(const LapisStatus& s) {
    std::lock_guard lock(statusLock_);
    if (s == lapisStatus_) return;
    lapisStatus_ = s;
    status_.writeLapis(s);
}

void Engine::reportVideo(const VideoStatus& s) {
    std::lock_guard lock(statusLock_);
    status_.writeVideo(s);
}

void Engine::reportKelir(const KelirTargetStatus& t) {
    std::lock_guard lock(statusLock_);
    if (t == kelirTarget_) return;
    kelirTarget_ = t;
    KelirStatus& s = kelirStatus_;
    s.state = t.state;
    s.reason = t.reason;
    s.api = t.api;
    s.format = t.format;
    s.width = t.width;
    s.height = t.height;
    s.rotation = t.rotation;
    s.path = t.path;
    s.signature = t.signature;
    s.gpuNs = t.gpuNs;
    status_.writeKelir(s);
}

Config Engine::config() const {
    std::lock_guard lock(lock_);
    return config_;
}

}  // namespace framefx
