/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "video/proxy.h"

#include <android/hardware_buffer.h>
#include <gui/BufferItem.h>
#include <gui/BufferItemConsumer.h>
#include <gui/Surface.h>
#include <log/log.h>
#include <pthread.h>
#include <sys/resource.h>
#include <system/graphics.h>
#include <system/window.h>
#include <ui/Fence.h>
#include <ui/GraphicBuffer.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "kelir/vulkan.h"
#include "luwes/shaders.h"
#include "src/engine.h"
#include "video/afme_engine.h"
#include "video/gpu.h"
#include "video/motion.h"
#include "video/timeline.h"

namespace framefx::video {
namespace {

using android::BufferItem;
using android::BufferItemConsumer;
using android::Fence;
using android::GraphicBuffer;
using android::sp;
using android::Surface;

constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr int kRing = 8;              // submissions in flight: a real frame's and its presents'
constexpr size_t kMaxSources = 40;    // decoder buffers imported; the decoder keeps fewer
constexpr size_t kMaxOutputs = 12;    // window buffers imported
constexpr int kMaxDequeued = 4;       // queued frames wait for their time on the window's side
constexpr int kMinSide = 16;          // smaller is a blank buffer pushed at shutdown, not video
constexpr uint32_t kTileWidth = 16;   // RGBA8888 UBWC tile
constexpr uint32_t kTileHeight = 4;
// The SR filter builds its model on the DSP for 4 s: not for a video scrolled past.
constexpr uint32_t kSuperresAfterFrames = 45;
constexpr int kAfmeSettleFrames = 5;
constexpr int64_t kAfmeMarginNs = 3'000'000;
constexpr int kAfmeJudgeFrames = 60;
constexpr uint64_t kFenceTimeoutNs = 1'000'000'000;
constexpr nsecs_t kDequeueTimeoutNs = 50'000'000;
constexpr int64_t kReportNs = 1'000'000'000;
constexpr int64_t kClockFarNs = 1'000'000'000;  // past this a timestamp is a media time, not a clock
constexpr int64_t kClockSlipNs = 200'000'000;
constexpr int64_t kMediaLeadNs = 16'000'000;
constexpr int64_t kBehindNs = 50'000'000;
constexpr int64_t kRenderBudgetNs = 3'000'000;  // to get one present through the GPU
constexpr int64_t kMaxRenderBudgetNs = 10'000'000;
constexpr int64_t kMaxQueueAheadNs = 20'000'000;

// What Xiaomi's SR filter multiplies a decoder's frame by: the sizes its SM8735 models take, which
// libmialgo holds in a table of its own.
uint32_t superresScale(uint32_t width, uint32_t height) {
    const uint32_t longSide = std::max(width, height);
    const uint32_t shortSide = std::min(width, height);
    if (longSide <= 640 && shortSide <= 384) return 5;
    if (longSide <= 960 && shortSide <= 544) return 3;
    if (longSide <= 1280 && shortSide <= 720) return 2;
    return 1;
}

int64_t now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1'000'000'000LL + t.tv_nsec;
}

// Decoders tag the frame's standard and transfer, or nothing for an unknown one.
android_dataspace_t knownDataspace(android_dataspace_t asked, uint32_t height) {
    if ((asked & HAL_DATASPACE_STANDARD_MASK) != 0) return asked;
    return static_cast<android_dataspace_t>(
            (height < 720 ? HAL_DATASPACE_STANDARD_BT601_625 : HAL_DATASPACE_STANDARD_BT709) |
            HAL_DATASPACE_TRANSFER_SMPTE_170M | HAL_DATASPACE_RANGE_LIMITED);
}

bool isHdr(android_dataspace_t dataspace) {
    const int transfer = dataspace & HAL_DATASPACE_TRANSFER_MASK;
    return transfer == HAL_DATASPACE_TRANSFER_ST2084 || transfer == HAL_DATASPACE_TRANSFER_HLG;
}

VkSamplerYcbcrModelConversion modelOf(android_dataspace_t dataspace) {
    switch (dataspace & HAL_DATASPACE_STANDARD_MASK) {
        case HAL_DATASPACE_STANDARD_BT601_625:
        case HAL_DATASPACE_STANDARD_BT601_625_UNADJUSTED:
        case HAL_DATASPACE_STANDARD_BT601_525:
        case HAL_DATASPACE_STANDARD_BT601_525_UNADJUSTED:
            return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
        case HAL_DATASPACE_STANDARD_BT2020:
        case HAL_DATASPACE_STANDARD_BT2020_CONSTANT_LUMINANCE:
            return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020;
        default:
            return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
    }
}

VkSamplerYcbcrRange rangeOf(android_dataspace_t dataspace) {
    return (dataspace & HAL_DATASPACE_RANGE_MASK) == HAL_DATASPACE_RANGE_FULL
            ? VK_SAMPLER_YCBCR_RANGE_ITU_FULL
            : VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
}

int turnsOf(uint32_t transform) {
    switch (transform & (HAL_TRANSFORM_ROT_90 | HAL_TRANSFORM_ROT_180)) {
        case HAL_TRANSFORM_ROT_90:
            return 1;
        case HAL_TRANSFORM_ROT_180:
            return 2;
        case HAL_TRANSFORM_ROT_270:
            return 3;
        default:
            return 0;
    }
}

// AFME outlives a proxy: a feed's next video is usually shown at the same size.
std::mutex gIdleAfmeLock;
std::unique_ptr<AfmeEngine> gIdleAfme;

class Proxy : public BufferItemConsumer::FrameAvailableListener {
public:
    Proxy() {
        std::tie(consumer_, input_) =
                BufferItemConsumer::create(AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE);
        consumer_->setName(android::String8("framefx-video"));
    }

    void start() {
        consumer_->setFrameAvailableListener(this);
        worker_ = std::thread([this] { run(); });
    }

    int32_t state() const { return state_.load(std::memory_order_acquire); }
    ANativeWindow* input() const { return input_.get(); }
    bool wantsSuperres() const { return superres_.load(std::memory_order_relaxed); }

    int32_t attach(ANativeWindow* window) {
        int err;
        {
            std::lock_guard target(targetLock_);
            disconnectLocked();
            const sp<Surface> surface = static_cast<Surface*>(window);
            err = native_window_api_connect(window, NATIVE_WINDOW_API_MEDIA);
            if (err == 0) {
                surface->setDequeueTimeout(kDequeueTimeoutNs);
                surface->setMaxDequeuedBufferCount(kMaxDequeued);
                native_window_set_usage(window, AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT |
                                                AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE);
                native_window_set_buffers_format(window, HAL_PIXEL_FORMAT_RGBA_8888);
                if (outWidth_ != 0) {
                    native_window_set_buffers_dimensions(window, static_cast<int>(outWidth_),
                                                         static_cast<int>(outHeight_));
                }
                target_ = surface;
            } else {
                ALOGW("video: the app's surface refused a connection: %d", err);
            }
        }
        std::lock_guard lock(lock_);
        open_ = true;
        frames_.notify_all();
        return err;
    }

    void detach() {
        {
            std::lock_guard target(targetLock_);
            disconnectLocked();
        }
        std::lock_guard lock(lock_);
        open_ = true;
        frames_.notify_all();
    }

    void close() {
        {
            std::lock_guard lock(lock_);
            stopping_ = true;
            frames_.notify_all();
        }
        worker_.join();
        consumer_.clear();  // abandons the queue
        std::lock_guard target(targetLock_);
        disconnectLocked();
    }

    void onFrameAvailable(const BufferItem&) override {
        std::lock_guard lock(lock_);
        pending_++;
        frames_.notify_all();
    }

private:
    struct Source {
        sp<GraphicBuffer> buffer;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    struct Output {
        sp<GraphicBuffer> buffer;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
    };

    struct Slot {
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore sourceReady = VK_NULL_HANDLE;
        VkSemaphore outputFree = VK_NULL_HANDLE;
        VkSemaphore drawn = VK_NULL_HANDLE;
        bool pending = false;
        bool timed = false;
        bool present = false;
        uint64_t stage = 0;
    };

    void run() {
        pthread_setname_np(pthread_self(), "framefx-video");
        const int64_t began = now();
        gpu_ = GetGpu();
        const int64_t gpu = now();
        if (gpu_ == nullptr || !warm()) {
            decline(VideoReason::kGpu);
            state_.store(FRAMEFX_VIDEO_FAILED, std::memory_order_release);
        } else {
            state_.store(FRAMEFX_VIDEO_READY, std::memory_order_release);
        }
        ALOGI("%s: video path warm: device %.0f ms, the rest %.0f ms",
              Engine::get().package().c_str(), (gpu - began) / 1e6, (now() - gpu) / 1e6);
        setpriority(PRIO_PROCESS, 0, -4);
        for (;;) {
            {
                std::unique_lock lock(lock_);
                frames_.wait(lock, [this] { return stopping_ || (pending_ > 0 && open_); });
                if (stopping_) break;
                pending_--;
            }
            frame();
        }
        if (gpu_ != nullptr) destroy();
    }

    // What every frame needs but the decoder's format, so the first frame costs little.
    bool warm() {
        const Gpu& g = *gpu_;
        pass_ = MakePass(g, kFormat);
        if (pass_ == VK_NULL_HANDLE) return false;
        namespace shaders = luwes::shaders;
        vertex_ = MakeModule(g, shaders::kFullscreenVertSpirv.code,
                             shaders::kFullscreenVertSpirv.size);
        fragment_ = MakeModule(g, shaders::kYcbcrFragSpirv.code, shaders::kYcbcrFragSpirv.size);
        upscaleModule_ = MakeModule(g, shaders::kUpscaleFragSpirv.code,
                                    shaders::kUpscaleFragSpirv.size);
        linear_ = MakeSampler(g, VK_FILTER_LINEAR);
        upscaleLayout_ = MakeSetLayout(g, 1, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
        if (vertex_ == VK_NULL_HANDLE || fragment_ == VK_NULL_HANDLE ||
            upscaleModule_ == VK_NULL_HANDLE || linear_ == VK_NULL_HANDLE ||
            upscaleLayout_ == VK_NULL_HANDLE) {
            return false;
        }
        upscalePipe_ = MakePipelineLayout(g, upscaleLayout_);
        upscale_ = MakeFullscreen(g, vertex_, upscaleModule_, upscalePipe_, pass_);
        if (upscale_ == VK_NULL_HANDLE) return false;
        if (g.motion) {
            motion_ = std::make_unique<Motion>(g);
            if (!motion_->warm(pass_)) motion_.reset();
        }

        VkCommandPoolCreateInfo commands = {};
        commands.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commands.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commands.queueFamilyIndex = g.family;
        if (g.CreateCommandPool(g.device, &commands, nullptr, &commandPool_) != VK_SUCCESS) {
            return false;
        }
        if (g.timestampPeriod > 0.0f) {
            VkQueryPoolCreateInfo timer = {};
            timer.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            timer.queryType = VK_QUERY_TYPE_TIMESTAMP;
            timer.queryCount = 2 * kRing;
            if (g.CreateQueryPool(g.device, &timer, nullptr, &timer_) != VK_SUCCESS) {
                timer_ = VK_NULL_HANDLE;
            }
        }
        VkExportSemaphoreCreateInfo exportable = {};
        exportable.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        exportable.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        VkSemaphoreCreateInfo plain = {};
        plain.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphoreCreateInfo signalled = plain;
        signalled.pNext = &exportable;
        VkFenceCreateInfo fence = {};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        for (Slot& s : ring_) {
            VkCommandBufferAllocateInfo allocate = {};
            allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocate.commandPool = commandPool_;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            if (g.AllocateCommandBuffers(g.device, &allocate, &s.commands) != VK_SUCCESS ||
                g.CreateFence(g.device, &fence, nullptr, &s.fence) != VK_SUCCESS ||
                g.CreateSemaphore(g.device, &plain, nullptr, &s.sourceReady) != VK_SUCCESS ||
                g.CreateSemaphore(g.device, &plain, nullptr, &s.outputFree) != VK_SUCCESS ||
                g.CreateSemaphore(g.device, &signalled, nullptr, &s.drawn) != VK_SUCCESS) {
                return false;
            }
        }

        // What is asked for now: compiling it later would hold frames back.
        watch_.refresh(Engine::get());
        desired_ = watch_.config().videoKelir;
        const int hz = motion_ != nullptr ? watch_.config().video.motionHz : 0;
        if (!desired_->bypass()) {
            uint64_t signatures[2];
            const int count = signaturesOf(*desired_, hz > 0, signatures);
            for (int i = 0; i < count; i++) {
                kelir::vulkan::offscreenCompile(g.device, kFormat, signatures[i]);
            }
        }
        running_ = desired_;
        hz_ = hz;
        return true;
    }

    // Kelir's passes for a plan: one over the whole frame, or with generation the grade once per
    // real frame and what goes over every present.
    static bool gradesRealFrames(const kelir::Plan& plan) {
        return plan.detailCount + plan.colourCount > 0;
    }
    static bool finishesPresents(const kelir::Plan& plan) {
        return plan.screenCount > 0 || plan.split;
    }
    static int signaturesOf(const kelir::Plan& plan, bool motion, uint64_t out[2]) {
        using kelir::Role;
        if (!motion) {
            out[0] = kelir::vulkan::offscreenSignature(plan, Role::kFull);
            return 1;
        }
        int count = 0;
        if (gradesRealFrames(plan)) {
            out[count++] = kelir::vulkan::offscreenSignature(plan, Role::kStageA);
        }
        if (finishesPresents(plan)) {
            out[count++] = kelir::vulkan::offscreenSignature(plan, Role::kStageB);
        }
        return count;
    }

    void decline(VideoReason reason) {
        status_.state = VideoState::kDeclined;
        status_.reason = reason;
        Engine::get().reportVideo(status_);
        KelirTargetStatus t;
        t.state = KelirState::kDeclined;
        t.reason = reason == VideoReason::kHdr ? KelirReason::kHdr : KelirReason::kResources;
        t.api = Api::kVulkan;
        Engine::get().reportKelir(t);
    }

    void frame() {
        frameBegan_ = now();
        Present presents[kMaxPresents];
        int count = 0;
        bool newerWaiting;
        {
            std::lock_guard lock(lock_);
            newerWaiting = pending_ > 0;
        }
        {
            std::lock_guard target(targetLock_);
            BufferItem item;
            if (consumer_->acquireBuffer(&item, 0, false) != android::OK) return;
            // Far behind with newer frames waiting, after a stall of any kind: going through them
            // all would show as the video rushing to catch up.
            if (newerWaiting && isBehind(item, frameBegan_)) {
                status_.dropped++;
                if (afmeOn_) judgeAfme(true);
                consumer_->releaseBuffer(item, item.mFence);
                return;
            }
            sp<Fence> done = Fence::NO_FENCE;
            bool ok = state() == FRAMEFX_VIDEO_READY && prepare(item);
            if (ok) ok = hz_ > 0 ? analyze(item, presents, &count) : draw(item, &done);
            if (!ok) status_.dropped++;
            consumer_->releaseBuffer(item, done);
        }
        for (int i = 0; i < count; i++) {
            const Present& p = presents[i];
            const int64_t start = p.when - readyNs_ - (p.real ? earlyNs_ : 0);
            {
                std::unique_lock lock(lock_);
                if (stopping_) return;
                // A generated frame is what goes when the frames behind it pile up.
                if (!p.real && pending_ > 1) continue;
                const auto due = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(start));
                frames_.wait_until(lock, due, [this] { return stopping_; });
                if (stopping_) return;
            }
            std::lock_guard target(targetLock_);
            if (!present(p)) status_.dropped += p.real ? 1 : 0;
        }
        std::lock_guard target(targetLock_);
        report();
    }

    bool isBehind(const BufferItem& item, int64_t t0) const {
        if (item.mIsAutoTimestamp) return false;
        int64_t t = item.mTimestamp;
        if (std::abs(t - t0) > kClockFarNs) {
            t += mediaOffset_;
            if (std::abs(t - t0) > kClockSlipNs) return false;
        }
        return t + kBehindNs < t0;
    }

    // Takes what Pentas asks for and sizes everything by this frame. Holds targetLock_.
    bool prepare(const BufferItem& item) {
        const sp<GraphicBuffer>& buffer = item.mGraphicBuffer;
        if (buffer == nullptr || target_ == nullptr) return false;
        const android_dataspace_t dataspace =
                knownDataspace(item.mDataSpace, static_cast<uint32_t>(buffer->getHeight()));
        if (isHdr(dataspace)) {
            fail(VideoReason::kHdr);
            return false;
        }
        android::Rect crop = item.mCrop;
        if (crop.isEmpty()) crop = buffer->getBounds();
        crop.intersect(buffer->getBounds(), &crop);
        const auto width = static_cast<uint32_t>(crop.width());
        const auto height = static_cast<uint32_t>(crop.height());
        if (width < kMinSide || height < kMinSide) return false;
        applySettings();
        const int turns = turnsOf(item.mTransform);
        const bool multiplied = isMultiplied(width, height);
        uint32_t outWidth = width;
        uint32_t outHeight = height;
        bool superres = false;
        if (upscaleOn_) {
            int w = 0;
            int h = 0;
            ANativeWindow* window = target_.get();
            if (window->query(window, NATIVE_WINDOW_DEFAULT_WIDTH, &w) == 0 &&
                window->query(window, NATIVE_WINDOW_DEFAULT_HEIGHT, &h) == 0 && w > 0 && h > 0) {
                if ((turns & 1) != 0) std::swap(w, h);
                // Some apps fix the surface at the video's size and let the display scale it: all
                // that says is that the video is shown as large as it fits.
                if (static_cast<uint32_t>(w) == decodedWidth_ &&
                    static_cast<uint32_t>(h) == decodedHeight_) {
                    int hint = 0;
                    window->query(window, NATIVE_WINDOW_TRANSFORM_HINT, &hint);
                    const bool sideways = (hint & HAL_TRANSFORM_ROT_90) != 0;
                    fitDisplay(width, height, turns, sideways, &w, &h);
                }
                const auto shownWidth = static_cast<uint32_t>(w);
                const auto shownHeight = static_cast<uint32_t>(h);
                // The window scales the frame anyway: only this much is worth redrawing for. The
                // filter's frames can outgrow it, and go no further through the chain than it shows.
                if ((shownWidth * 100 >= width * 105 && shownHeight * 100 >= height * 105) ||
                    (multiplied && shownWidth <= width && shownHeight <= height)) {
                    outWidth = shownWidth;
                    outHeight = shownHeight;
                }
                superres = status_.frames >= kSuperresAfterFrames &&
                           superresScale(decodedWidth_, decodedHeight_) > 1 &&
                           shownWidth * 100 >= decodedWidth_ * 125 &&
                           shownHeight * 100 >= decodedHeight_ * 125;
            }
        }
        superres_.store(superres, std::memory_order_relaxed);
        // Adreno corrupts a UBWC tile that a pass with heavy sampling covers only in part: the
        // window's buffers are whole tiles, the frame stretched over the few lines more.
        outWidth = (outWidth + kTileWidth - 1) / kTileWidth * kTileWidth;
        outHeight = (outHeight + kTileHeight - 1) / kTileHeight * kTileHeight;
        const bool motion = hz_ > 0;
        const uint32_t sceneWidth = std::min(width, outWidth);
        const uint32_t sceneHeight = std::min(height, outHeight);
        if ((sceneWidth != width_ || sceneHeight != height_ || outWidth != outWidth_ ||
             outHeight != outHeight_ || turns != turns_ || motion != chainMotion_) &&
            !resize(sceneWidth, sceneHeight, outWidth, outHeight, turns, motion)) {
            fail(VideoReason::kGpu);
            return false;
        }
        const bool afme = motion && watch_.config().video.afme;
        if (afme) {
            takeAfme(outWidth, outHeight);
        } else if (afmeWanted_) {
            dropAfme();
        }
        afmeWanted_ = afme;
        afmeOn_ = afme && afmeReady_ && !afmeTooSlow_;
        source_ = sourceFor(buffer, dataspace);
        if (source_ == nullptr) {
            fail(VideoReason::kGpu);
            return false;
        }
        crop_ = crop;
        dataspace_ = dataspace;
        transform_ = item.mTransform;
        scalingMode_ = item.mScalingMode;
        status_.sourceWidth = static_cast<uint16_t>(decodedWidth_);
        status_.sourceHeight = static_cast<uint16_t>(decodedHeight_);
        status_.width = static_cast<uint16_t>(outWidth);
        status_.height = static_cast<uint16_t>(outHeight);
        status_.upscaled = outWidth != decodedWidth_ || outHeight != decodedHeight_;
        status_.superres = multiplied;
        return true;
    }

    // Whether this frame is the decoder's own size times what the SR filter multiplies that by.
    // Otherwise it is what the decoder makes now: a stream can be exactly that much larger than the
    // last one, and an app that fixes its surface at the video's size moves it along.
    bool isMultiplied(uint32_t width, uint32_t height) {
        int w = 0;
        int h = 0;
        ANativeWindow* window = target_.get();
        window->query(window, NATIVE_WINDOW_DEFAULT_WIDTH, &w);
        window->query(window, NATIVE_WINDOW_DEFAULT_HEIGHT, &h);
        const uint32_t windowSide = static_cast<uint32_t>(std::max(w, h));
        const uint32_t scale = superresScale(decodedWidth_, decodedHeight_);
        if (scale > 1 && superres_.load(std::memory_order_relaxed) &&
            width == decodedWidth_ * scale && height == decodedHeight_ * scale &&
            (windowSide == decodedWindowSide_ || windowSide != std::max(width, height))) {
            return true;
        }
        decodedWidth_ = width;
        decodedHeight_ = height;
        decodedWindowSide_ = windowSide;
        return false;
    }

    // The frame at its aspect, as large as fits the display as it is turned now.
    void fitDisplay(uint32_t width, uint32_t height, int turns, bool sideways, int* w, int* h) {
        const VideoSettings& v = watch_.config().video;
        if (v.displayLong <= 0 || v.displayShort <= 0) return;
        int fitW = sideways ? v.displayLong : v.displayShort;
        int fitH = sideways ? v.displayShort : v.displayLong;
        // In the frame's axes, before its transform turns it.
        if ((turns & 1) != 0) std::swap(fitW, fitH);
        const double scale = std::min(static_cast<double>(fitW) / width,
                                      static_cast<double>(fitH) / height);
        *w = static_cast<int>(width * scale + 0.5);
        *h = static_cast<int>(height * scale + 0.5);
    }

    // Draws one frame straight into the window, under its own timestamp. `done` is when the GPU has
    // finished reading the frame.
    bool draw(const BufferItem& item, sp<Fence>* done) {
        const Gpu& g = *gpu_;
        ANativeWindow* window = target_.get();
        ANativeWindowBuffer* anb = nullptr;
        int dequeueFence = -1;
        const int dequeued = window->dequeueBuffer(window, &anb, &dequeueFence);
        if (dequeued != 0) {
            ALOGW_IF(dequeued != -ETIMEDOUT, "video: window dequeue: %d", dequeued);
            return false;
        }
        Output* output = outputFor(GraphicBuffer::from(anb));
        Slot* s = output != nullptr ? beginSlot() : nullptr;
        if (s == nullptr) {
            window->cancelBuffer(window, anb, dequeueFence);
            fail(VideoReason::kGpu);
            return false;
        }
        takeIn(s->commands, *source_);
        takeOut(s->commands, *output);
        const std::shared_ptr<const kelir::Plan> plan = running_;
        kelir::vulkan::Stage stage;
        const bool staged = !plan->bypass() &&
                kelir::vulkan::beginOffscreen(g.device, kelirKey(), g.family, plan,
                                              kelir::Role::kFull, &stage);
        const bool upscaled = outWidth_ != width_ || outHeight_ != height_;
        if (!staged && !upscaled) {
            convert(s->commands, *source_, output->framebuffer);
        } else {
            convert(s->commands, *source_, sceneTarget_);
            if (upscaled && staged) {
                upscaleInto(s->commands, upTarget_, !upscaleOn_);
                kelir::vulkan::recordOffscreen(stage, s->commands, up_.view, output->view,
                                               kelir::Role::kFull);
            } else if (upscaled) {
                upscaleInto(s->commands, output->framebuffer, !upscaleOn_);
            } else {
                kelir::vulkan::recordOffscreen(stage, s->commands, scene_.view, output->view,
                                               kelir::Role::kFull);
            }
        }
        if (staged) s->stage = stage.serial;
        giveBack(s->commands, *source_);
        giveOut(s->commands, *output);
        int drawn = -1;
        const int sourceFence = item.mFence != nullptr ? item.mFence->dup() : -1;
        if (!submit(*s, sourceFence, dequeueFence, &drawn)) {
            window->cancelBuffer(window, anb, -1);
            fail(VideoReason::kGpu);
            return false;
        }
        *done = drawn >= 0 ? sp<Fence>::make(dup(drawn)) : Fence::NO_FENCE;
        if (!queue(window, anb, item.mIsAutoTimestamp ? NATIVE_WINDOW_TIMESTAMP_AUTO : item.mTimestamp,
                   drawn)) {
            return false;
        }
        status_.state = VideoState::kRunning;
        status_.reason = VideoReason::kNone;
        status_.frames++;
        presented_++;
        return true;
    }

    // One real frame on its way to being a pair: converted, upscaled and graded into the pair, its
    // motion searched. The presents are planned from what that found.
    bool analyze(const BufferItem& item, Present presents[kMaxPresents], int* count) {
        const Gpu& g = *gpu_;
        Slot* s = beginSlot();
        if (s == nullptr) {
            fail(VideoReason::kGpu);
            return false;
        }
        const VkCommandBuffer cb = s->commands;
        takeIn(cb, *source_);
        convert(cb, *source_, sceneTarget_);
        giveBack(cb, *source_);
        if (!pairPrimed_) {
            for (const Image& image : history_) {
                Barrier(g, cb, image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            }
            pairPrimed_ = true;
        }
        current_ = 1 - current_;
        const std::shared_ptr<const kelir::Plan> plan = running_;
        kelir::vulkan::Stage stage;
        const bool graded = !plan->bypass() && gradesRealFrames(*plan) &&
                kelir::vulkan::beginOffscreen(g.device, kelirKey(), g.family, plan,
                                              kelir::Role::kStageA, &stage);
        const bool upscaled = outWidth_ != width_ || outHeight_ != height_;
        if (upscaled && graded) {
            upscaleInto(cb, upTarget_, !upscaleOn_);
            kelir::vulkan::recordOffscreen(stage, cb, up_.view, history_[current_].view,
                                           kelir::Role::kStageA);
        } else if (graded) {
            kelir::vulkan::recordOffscreen(stage, cb, scene_.view, history_[current_].view,
                                           kelir::Role::kStageA);
        } else {
            upscaleInto(cb, historyTarget_[current_], !upscaled || !upscaleOn_);
        }
        if (graded) s->stage = stage.serial;
        motion_->record(cb, current_, hasPair_);
        afmeFrame_ = afmeOn_ && hasPair_ &&
                afme_->record(cb, history_[1 - current_].view, history_[current_].view);
        // An engine that comes from another video still holds that video's last frames.
        if (afmeFrame_ && afmeFresh_) {
            afmeFresh_ = false;
            afmeFrame_ = false;
        }
        const int sourceFence = item.mFence != nullptr ? item.mFence->dup() : -1;
        int drawn = -1;
        if (!submit(*s, sourceFence, -1, &drawn)) {
            fail(VideoReason::kGpu);
            return false;
        }
        if (drawn >= 0) ::close(drawn);
        // The presents hang on what the search found, and the decoder's buffer goes back free.
        if (!retire(*s)) return false;
        const bool cut = hasPair_ && motion_->cut();
        ALOGD_IF(cut, "video path: a cut");
        hasPair_ = true;

        const int64_t t0 = now();
        int64_t t = item.mIsAutoTimestamp ? t0 + kMediaLeadNs : item.mTimestamp;
        // A player that renders under its media time has no clock of ours: time from arrival.
        if (std::abs(t - t0) > kClockFarNs) {
            if (std::abs(t + mediaOffset_ - t0) > kClockSlipNs) mediaOffset_ = t0 + kMediaLeadNs - t;
            t += mediaOffset_;
        }
        int64_t period = 1'000'000'000LL / hz_;
        const int64_t ahead = std::min(period * 3 / 2, kMaxQueueAheadNs);
        analysisNs_ = std::max(t0 - frameBegan_, analysisNs_ - analysisNs_ / 8);
        // AFME has the one frame, halfway. The next real frame starts here once this one has gone,
        // so a search longer than half the gap would make every such frame late, whatever the delay:
        // this one goes that much sooner, under the same timestamp.
        const int64_t half = (t - lastTime_) / 2;
        earlyNs_ = 0;
        if (afmeOn_ && half > 0 && half * 2 <= kMaxGapNs) {
            period = std::max(period, half);
            earlyNs_ = std::clamp(analysisNs_ + kAfmeMarginNs - half, int64_t{0}, half);
        }
        lastTime_ = t;
        // A present's GPU work has to end before the composer latches it.
        readyNs_ = ahead + std::clamp(presentNs_ * 3 / 2, kRenderBudgetNs, kMaxRenderBudgetNs);
        *count = timeline_.plan(t, t0, cut || (afmeOn_ && !afmeFrame_) ? 0 : period, readyNs_,
                                smoothness_, presents);
        if (afmeOn_) judgeAfme(t + timeline_.delay() < t0);
        status_.delayUs = static_cast<uint32_t>(timeline_.delay() / 1000);
        status_.state = VideoState::kRunning;
        status_.reason = VideoReason::kNone;
        status_.frames++;
        return true;
    }

    // Real frames falling behind: Luwes searches for less, and takes over at this size.
    void judgeAfme(bool late) {
        afmeLate_ += late ? 1 : 0;
        if (++afmeJudged_ < kAfmeJudgeFrames) return;
        if (afmeLate_ * 10 >= afmeJudged_) {
            ALOGW("video path: AFME left %d of %d real frames late: Luwes at this size", afmeLate_,
                  afmeJudged_);
            afmeTooSlow_ = true;
        }
        afmeLate_ = 0;
        afmeJudged_ = 0;
    }

    // One frame of the pair at its phase into the window, under its display time. Holds targetLock_.
    bool present(const Present& p) {
        const Gpu& g = *gpu_;
        if (target_ == nullptr || state() != FRAMEFX_VIDEO_READY || !hasPair_) return false;
        ANativeWindow* window = target_.get();
        ANativeWindowBuffer* anb = nullptr;
        int dequeueFence = -1;
        const int dequeued = window->dequeueBuffer(window, &anb, &dequeueFence);
        if (dequeued != 0) {
            ALOGW_IF(dequeued != -ETIMEDOUT, "video: window dequeue: %d", dequeued);
            return false;
        }
        Output* output = outputFor(GraphicBuffer::from(anb));
        Slot* s = output != nullptr ? beginSlot() : nullptr;
        if (s == nullptr) {
            window->cancelBuffer(window, anb, dequeueFence);
            fail(VideoReason::kGpu);
            return false;
        }
        s->present = true;
        takeOut(s->commands, *output);
        const std::shared_ptr<const kelir::Plan> plan = running_;
        kelir::vulkan::Stage stage;
        const bool finished = !plan->bypass() && finishesPresents(*plan) &&
                kelir::vulkan::beginOffscreen(g.device, kelirKey(), g.family, plan,
                                              kelir::Role::kStageB, &stage);
        const VkFramebuffer into = finished ? tempTarget_ : output->framebuffer;
        if (afmeOn_ && !p.real) {
            const float params[16] = {
                    1.0f / static_cast<float>(outWidth_), 1.0f / static_cast<float>(outHeight_),
                    static_cast<float>(outWidth_), static_cast<float>(outHeight_),
                    static_cast<float>(outWidth_), static_cast<float>(outHeight_), 0.0f, 0.0f,
                    1.0f,
            };
            Fullscreen(g, s->commands, pass_, into, outWidth_, outHeight_, upscale_, upscalePipe_,
                       afmeSet_, params);
        } else {
            motion_->recordOutput(s->commands, into, current_, p.phase);
        }
        if (finished) {
            kelir::vulkan::recordOffscreen(stage, s->commands, temp_.view, output->view,
                                           kelir::Role::kStageB);
            s->stage = stage.serial;
        }
        giveOut(s->commands, *output);
        int drawn = -1;
        if (!submit(*s, -1, dequeueFence, &drawn)) {
            window->cancelBuffer(window, anb, -1);
            fail(VideoReason::kGpu);
            return false;
        }
        if (!queue(window, anb, p.when, drawn)) return false;
        presented_++;
        if (!p.real) status_.generated++;
        return true;
    }

    bool queue(ANativeWindow* window, ANativeWindowBuffer* anb, int64_t when, int drawn) {
        native_window_set_buffers_timestamp(window, when);
        native_window_set_buffers_transform(window, static_cast<int>(transform_));
        native_window_set_scaling_mode(window, static_cast<int>(scalingMode_));
        // What the matrix produced: the frame's standard and transfer, now at full range.
        native_window_set_buffers_data_space(
                window, static_cast<android_dataspace_t>(
                                (dataspace_ & ~HAL_DATASPACE_RANGE_MASK) | HAL_DATASPACE_RANGE_FULL));
        const int queued = window->queueBuffer(window, anb, drawn);
        ALOGW_IF(queued != 0, "video: window queue: %d", queued);
        return queued == 0;
    }

    void fail(VideoReason reason) {
        decline(reason);
        state_.store(FRAMEFX_VIDEO_FAILED, std::memory_order_release);
    }

    // The ring's next slot, retired, with its command buffer open and the frame's timer started.
    Slot* beginSlot() {
        const Gpu& g = *gpu_;
        const int index = ringIndex_;
        Slot& s = ring_[index];
        if (!retire(s)) return nullptr;
        ringIndex_ = (ringIndex_ + 1) % kRing;
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        g.BeginCommandBuffer(s.commands, &begin);
        s.timed = timer_ != VK_NULL_HANDLE;
        s.present = false;
        if (s.timed) {
            const uint32_t base = static_cast<uint32_t>(index) * 2;
            g.CmdResetQueryPool(s.commands, timer_, base, 2);
            g.CmdWriteTimestamp(s.commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timer_, base);
        }
        return &s;
    }

    // Ends the slot's commands and submits them behind the fds, which it takes over. `drawn` gets
    // the sync fd of the GPU's end, or -1 when that has already passed.
    bool submit(Slot& s, int sourceFence, int outputFence, int* drawn) {
        const Gpu& g = *gpu_;
        if (s.timed) {
            const auto base = static_cast<uint32_t>(&s - ring_) * 2;
            g.CmdWriteTimestamp(s.commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timer_, base + 1);
        }
        g.EndCommandBuffer(s.commands);
        VkSemaphore waits[2];
        VkPipelineStageFlags stages[2];
        uint32_t waitCount = 0;
        if (importFence(s.sourceReady, sourceFence)) {
            waits[waitCount] = s.sourceReady;
            stages[waitCount++] = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        if (importFence(s.outputFree, outputFence)) {
            waits[waitCount] = s.outputFree;
            stages[waitCount++] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = waitCount;
        submit.pWaitSemaphores = waits;
        submit.pWaitDstStageMask = stages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &s.commands;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &s.drawn;
        VkResult result;
        {
            std::lock_guard queue(g.queueLock);
            result = g.QueueSubmit(g.queue, 1, &submit, s.fence);
        }
        if (result != VK_SUCCESS) {
            ALOGW("video: submit: %d", result);
            return false;
        }
        s.pending = true;
        VkSemaphoreGetFdInfoKHR get = {};
        get.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        get.semaphore = s.drawn;
        get.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        if (g.GetSemaphoreFdKHR(g.device, &get, drawn) != VK_SUCCESS) {
            // Nothing to wait on left: what follows must see the GPU done.
            g.WaitForFences(g.device, 1, &s.fence, VK_TRUE, kFenceTimeoutNs);
            *drawn = -1;
        }
        return true;
    }

    // Both images come from outside the device. The source's contents must survive it; the output's
    // are all overwritten.
    void takeIn(VkCommandBuffer cb, const Source& source) {
        transfer(cb, source.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_QUEUE_FAMILY_FOREIGN_EXT, gpu_->family, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    void giveBack(VkCommandBuffer cb, const Source& source) {
        transfer(cb, source.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                 gpu_->family, VK_QUEUE_FAMILY_FOREIGN_EXT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    }
    void takeOut(VkCommandBuffer cb, const Output& output) {
        transfer(cb, output.image, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT, gpu_->family,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    }
    void giveOut(VkCommandBuffer cb, const Output& output) {
        transfer(cb, output.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, gpu_->family, VK_QUEUE_FAMILY_FOREIGN_EXT,
                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    }

    // SGSR 1 from the converted frame to the output size; `plain` copies it instead, same size.
    void upscaleInto(VkCommandBuffer cb, VkFramebuffer target, bool plain) {
        const float params[16] = {
                1.0f / static_cast<float>(width_), 1.0f / static_cast<float>(height_),
                static_cast<float>(width_), static_cast<float>(height_),
                static_cast<float>(outWidth_), static_cast<float>(outHeight_), 0.0f, 0.0f,
                plain ? 1.0f : 0.0f,
        };
        Fullscreen(*gpu_, cb, pass_, target, outWidth_, outHeight_, upscale_, upscalePipe_, sceneSet_,
                   params);
    }

    // Takes up what Pentas last asked for. A stack that changes a pass compiles beside the frames,
    // which stay on what they had until it is built; one that only moves a value goes straight on.
    void applySettings() {
        const Gpu& g = *gpu_;
        if (watch_.refresh(Engine::get())) desired_ = watch_.config().videoKelir;
        upscaleOn_ = watch_.config().video.upscale;
        smoothness_ = watch_.config().video.smoothness;
        const int hz = motion_ != nullptr ? watch_.config().video.motionHz : 0;
        if (desired_ != running_ || hz != hz_) {
            bool ready = true;
            uint64_t signatures[2] = {};
            int count = 0;
            if (!desired_->bypass()) {
                count = signaturesOf(*desired_, hz > 0, signatures);
                for (int i = 0; i < count; i++) {
                    ready = ready && kelir::vulkan::offscreenReady(g.device, kFormat, signatures[i]);
                }
            }
            if (ready) {
                running_ = desired_;
                hz_ = hz;
            } else if (!compiling_.exchange(true)) {
                const sp<Proxy> self = sp<Proxy>::fromExisting(this);
                std::thread([self, signatures, count] {
                    const int64_t began = now();
                    for (int i = 0; i < count; i++) {
                        kelir::vulkan::offscreenCompile(self->gpu_->device, kFormat, signatures[i]);
                    }
                    ALOGI("video path: a stack's passes compiled in %.0f ms",
                          static_cast<double>(now() - began) / 1e6);
                    self->compiling_.store(false);
                }).detach();
            }
        }
        // The panel at the rate the frames are laid out for. A surface drawn into the app's own
        // window, a TextureView, has no layer to vote with, and its redraws speak for it.
        if (target_ != nullptr && hz_ != votedHz_) {
            target_->setFrameRate(static_cast<float>(hz_),
                                  hz_ > 0 ? ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE
                                          : ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_DEFAULT,
                                  ANATIVEWINDOW_CHANGE_FRAME_RATE_ONLY_IF_SEAMLESS);
            votedHz_ = hz_;
        }
    }

    void convert(VkCommandBuffer cb, const Source& source, VkFramebuffer target) {
        const float params[16] = {1.0f / static_cast<float>(source.buffer->getWidth()),
                                  1.0f / static_cast<float>(source.buffer->getHeight()),
                                  static_cast<float>(crop_.left),
                                  static_cast<float>(crop_.top),
                                  static_cast<float>(crop_.width()) / static_cast<float>(width_),
                                  static_cast<float>(crop_.height()) / static_cast<float>(height_)};
        Fullscreen(*gpu_, cb, pass_, target, width_, height_, pipeline_, layout_, source.set, params);
    }

    void transfer(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
                  uint32_t fromFamily, uint32_t toFamily, VkPipelineStageFlags srcStage,
                  VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                  VkAccessFlags dstAccess) {
        VkImageMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = fromFamily;
        b.dstQueueFamilyIndex = toFamily;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        gpu_->CmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // Takes over a sync fd as the semaphore's next wait; false when there is nothing to wait on.
    bool importFence(VkSemaphore semaphore, int fd) {
        if (fd < 0) return false;
        VkImportSemaphoreFdInfoKHR info = {};
        info.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
        info.semaphore = semaphore;
        info.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
        info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        info.fd = fd;
        if (gpu_->ImportSemaphoreFdKHR(gpu_->device, &info) == VK_SUCCESS) return true;
        // The wait is lost, so sit it out here.
        sp<Fence> fence = sp<Fence>::make(fd);
        fence->wait(1000);
        return false;
    }

    bool retire(Slot& s) {
        if (!s.pending) return true;
        const Gpu& g = *gpu_;
        if (g.WaitForFences(g.device, 1, &s.fence, VK_TRUE, kFenceTimeoutNs) != VK_SUCCESS) {
            return false;
        }
        g.ResetFences(g.device, 1, &s.fence);
        s.pending = false;
        if (s.timed) {
            uint64_t ticks[2] = {};
            const auto base = static_cast<uint32_t>(&s - ring_) * 2;
            if (g.GetQueryPoolResults(g.device, timer_, base, 2, sizeof(ticks), ticks,
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                const auto ns = static_cast<int64_t>(static_cast<double>(ticks[1] - ticks[0]) *
                                                     g.timestampPeriod);
                gpuTotalNs_ += static_cast<uint64_t>(ns);
                if (s.present) presentNs_ = presentNs_ == 0 ? ns : (presentNs_ * 7 + ns) / 8;
            }
        }
        if (s.stage != 0) {
            kelir::vulkan::stageRetired(g.device, kelirKey(), s.stage);
            s.stage = 0;
        }
        return true;
    }

    void waitRing() {
        for (Slot& s : ring_) retire(s);
    }

    VkSwapchainKHR kelirKey() { return reinterpret_cast<VkSwapchainKHR>(this); }

    // Everything sized by the frame. Holds targetLock_.
    bool resize(uint32_t width, uint32_t height, uint32_t outWidth, uint32_t outHeight, int turns,
                bool motion) {
        const Gpu& g = *gpu_;
        waitRing();
        freeSized();
        width_ = width;
        height_ = height;
        outWidth_ = outWidth;
        outHeight_ = outHeight;
        turns_ = turns;
        chainMotion_ = motion;
        if (target_ != nullptr) {
            native_window_set_buffers_dimensions(target_.get(), static_cast<int>(outWidth),
                                                 static_cast<int>(outHeight));
        }
        ALOGI("%s: video path %ux%u to %ux%u%s", Engine::get().package().c_str(), width, height,
              outWidth, outHeight, motion ? ", generating" : "");
        const VkImageUsageFlags drawn =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const auto make = [&](Image* image, VkFramebuffer* framebuffer, uint32_t w, uint32_t h) {
            if (!MakeImage(g, image, kFormat, w, h, drawn)) return false;
            *framebuffer = MakeFramebuffer(g, pass_, image->view, w, h);
            return *framebuffer != VK_NULL_HANDLE;
        };
        if (!make(&scene_, &sceneTarget_, width, height)) return false;
        const bool upscaled = outWidth != width || outHeight != height;
        if (upscaled && !make(&up_, &upTarget_, outWidth, outHeight)) return false;
        if (motion) {
            for (int p = 0; p < 2; p++) {
                if (!make(&history_[p], &historyTarget_[p], outWidth, outHeight)) return false;
            }
            if (!make(&temp_, &tempTarget_, outWidth, outHeight)) return false;
            const VkImageView pair[2] = {history_[0].view, history_[1].view};
            if (!motion_->resize(width, height, scene_.view, pair, outWidth, outHeight)) return false;
        }
        const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2};
        VkDescriptorPoolCreateInfo pool = {};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = 2;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        if (g.CreateDescriptorPool(g.device, &pool, nullptr, &chainPool_) != VK_SUCCESS) return false;
        sceneSet_ = AllocateSet(g, chainPool_, upscaleLayout_);
        if (sceneSet_ == VK_NULL_HANDLE) return false;
        WriteSet(g, sceneSet_, {Binding{scene_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, linear_}},
                 nullptr);
        kelir::vulkan::onOffscreen(g.device, kelirKey(), kFormat, {outWidth, outHeight}, turns);
        hasKelirTarget_ = true;
        timeline_.reset();
        hasPair_ = false;
        pairPrimed_ = false;
        afmeAt_ = {};
        afmeFrame_ = false;
        return true;
    }

    // What the plain copy reads AFME's frame through.
    // The engine for this output size, once there is one. Luwes's own motion goes on meanwhile, and
    // wherever AFME says no: frames do not wait for its pipelines. Holds targetLock_.
    void takeAfme(uint32_t width, uint32_t height) {
        const std::pair<uint32_t, uint32_t> size(width, height);
        if (afmeAt_ == size) return;
        afmeReady_ = false;
        if (afmeSettling_ != size) {
            afmeSettling_ = size;
            afmeSettled_ = 0;
        }
        afmeSettled_++;
        if (afme_ == nullptr || !afme_->sized(width, height)) {
            std::unique_ptr<AfmeEngine> found;
            bool refused = false;
            {
                std::lock_guard lock(afmeLock_);
                if (afmeBuilt_ != nullptr && afmeBuilt_->sized(width, height)) {
                    found = std::move(afmeBuilt_);
                }
                refused = afmeRefused_ == size;
            }
            if (found == nullptr) {
                std::lock_guard lock(gIdleAfmeLock);
                if (gIdleAfme != nullptr && gIdleAfme->sized(width, height)) {
                    found = std::move(gIdleAfme);
                }
            }
            if (found == nullptr) {
                // A window often settles on its size a few frames in.
                if (!refused && afmeSettled_ >= kAfmeSettleFrames &&
                    !afmeBuilding_.load(std::memory_order_acquire)) {
                    buildAfme(width, height);
                }
                return;
            }
            waitRing();
            afme_ = std::move(found);
        }
        afmeAt_ = size;
        afmeFresh_ = true;
        afmeTooSlow_ = false;
        afmeLate_ = 0;
        afmeJudged_ = 0;
        afmeReady_ = bindAfme();
    }

    void buildAfme(uint32_t width, uint32_t height) {
        afmeBuilding_.store(true, std::memory_order_release);
        const sp<Proxy> self = sp<Proxy>::fromExisting(this);
        std::thread([self, width, height] {
            auto engine = std::make_unique<AfmeEngine>(*self->gpu_);
            const int64_t began = now();
            const bool ok = engine->resize(width, height);
            ALOGI("video path: AFME for %ux%u built in %.0f ms: %s", width, height,
                  static_cast<double>(now() - began) / 1e6, ok ? "ok" : "refused");
            {
                std::lock_guard lock(self->afmeLock_);
                if (ok) {
                    self->afmeBuilt_ = std::move(engine);
                } else {
                    self->afmeRefused_ = {width, height};
                }
            }
            self->afmeBuilding_.store(false, std::memory_order_release);
        }).detach();
    }

    // AFME is off: its images go. Holds targetLock_.
    void dropAfme() {
        waitRing();
        afme_.reset();
        afmeAt_ = {};
        afmeReady_ = false;
        {
            std::lock_guard lock(afmeLock_);
            afmeBuilt_.reset();
        }
        std::lock_guard lock(gIdleAfmeLock);
        gIdleAfme.reset();
    }

    bool bindAfme() {
        const Gpu& g = *gpu_;
        if (afmeSet_ != VK_NULL_HANDLE) g.FreeDescriptorSets(g.device, chainPool_, 1, &afmeSet_);
        afmeSet_ = AllocateSet(g, chainPool_, upscaleLayout_);
        if (afmeSet_ == VK_NULL_HANDLE) return false;
        WriteSet(g, afmeSet_,
                 {Binding{afme_->output(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, linear_}},
                 nullptr);
        return true;
    }

    void freeSized() {
        const Gpu& g = *gpu_;
        if (hasKelirTarget_) kelir::vulkan::destroySwapchain(g.device, kelirKey());
        hasKelirTarget_ = false;
        for (auto& [id, o] : outputs_) freeOutput(o);
        outputs_.clear();
        const auto free = [&](Image& image, VkFramebuffer& framebuffer) {
            if (framebuffer != VK_NULL_HANDLE) g.DestroyFramebuffer(g.device, framebuffer, nullptr);
            framebuffer = VK_NULL_HANDLE;
            FreeImage(g, image);
        };
        free(scene_, sceneTarget_);
        free(up_, upTarget_);
        free(temp_, tempTarget_);
        for (int p = 0; p < 2; p++) free(history_[p], historyTarget_[p]);
        if (chainPool_ != VK_NULL_HANDLE) g.DestroyDescriptorPool(g.device, chainPool_, nullptr);
        chainPool_ = VK_NULL_HANDLE;
        sceneSet_ = VK_NULL_HANDLE;
        afmeSet_ = VK_NULL_HANDLE;
    }

    // Imports `buffer`'s memory into `image`, dedicated. False leaves what was made for the caller.
    bool importBuffer(const sp<GraphicBuffer>& buffer, const VkImageCreateInfo& info,
                      VkImage* image, VkDeviceMemory* memory,
                      const VkAndroidHardwareBufferPropertiesANDROID& properties) {
        const Gpu& g = *gpu_;
        if (g.CreateImage(g.device, &info, nullptr, image) != VK_SUCCESS) return false;
        VkImportAndroidHardwareBufferInfoANDROID imported = {};
        imported.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
        imported.buffer = buffer->toAHardwareBuffer();
        VkMemoryDedicatedAllocateInfo dedicated = {};
        dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated.pNext = &imported;
        dedicated.image = *image;
        VkMemoryAllocateInfo allocate = {};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.pNext = &dedicated;
        allocate.allocationSize = properties.allocationSize;
        allocate.memoryTypeIndex = MemoryType(*gpu_, properties.memoryTypeBits, 0);
        return allocate.memoryTypeIndex != UINT32_MAX &&
                g.AllocateMemory(g.device, &allocate, nullptr, memory) == VK_SUCCESS &&
                g.BindImageMemory(g.device, *image, *memory, 0) == VK_SUCCESS;
    }

    Source* sourceFor(const sp<GraphicBuffer>& buffer, android_dataspace_t dataspace) {
        const Gpu& g = *gpu_;
        const auto found = sources_.find(buffer->getId());
        VkAndroidHardwareBufferFormatPropertiesANDROID format = {};
        format.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
        VkAndroidHardwareBufferPropertiesANDROID properties = {};
        properties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
        properties.pNext = &format;
        if (found != sources_.end() && streamDataspace_ == dataspace) return &found->second;
        if (g.GetAndroidHardwareBufferPropertiesANDROID(g.device, buffer->toAHardwareBuffer(),
                                                        &properties) != VK_SUCCESS) {
            return nullptr;
        }
        if ((format.externalFormat != streamFormat_ || dataspace != streamDataspace_ ||
             format.format != streamVkFormat_) &&
            !conversion(format, dataspace)) {
            return nullptr;
        }
        const auto again = sources_.find(buffer->getId());
        if (again != sources_.end()) return &again->second;
        if (sources_.size() >= kMaxSources) {
            waitRing();
            for (auto& [id, s] : sources_) freeSource(s);
            sources_.clear();
        }

        Source s;
        s.buffer = buffer;
        VkExternalFormatANDROID external = {};
        external.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
        external.externalFormat = format.format == VK_FORMAT_UNDEFINED ? format.externalFormat : 0;
        VkExternalMemoryImageCreateInfo memory = {};
        memory.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        memory.pNext = &external;
        memory.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
        VkImageCreateInfo image = {};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.pNext = &memory;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = format.format;
        image.extent = {static_cast<uint32_t>(buffer->getWidth()),
                        static_cast<uint32_t>(buffer->getHeight()), 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!importBuffer(buffer, image, &s.image, &s.memory, properties)) {
            freeSource(s);
            return nullptr;
        }
        VkSamplerYcbcrConversionInfo convert = {};
        convert.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
        convert.conversion = conversion_;
        s.view = MakeView(g, s.image, format.format, &convert);
        VkDescriptorSetAllocateInfo allocate = {};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = pool_;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &setLayout_;
        if (s.view == VK_NULL_HANDLE ||
            g.AllocateDescriptorSets(g.device, &allocate, &s.set) != VK_SUCCESS) {
            freeSource(s);
            return nullptr;
        }
        VkDescriptorImageInfo descriptor = {VK_NULL_HANDLE, s.view,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = s.set;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &descriptor;
        g.UpdateDescriptorSets(g.device, 1, &write, 0, nullptr);
        return &sources_.emplace(buffer->getId(), std::move(s)).first->second;
    }

    // The sampler, and what is built on it, for the stream's format and colour.
    bool conversion(const VkAndroidHardwareBufferFormatPropertiesANDROID& format,
                    android_dataspace_t dataspace) {
        const Gpu& g = *gpu_;
        waitRing();
        for (auto& [id, s] : sources_) freeSource(s);
        sources_.clear();
        freeConversion();

        VkExternalFormatANDROID external = {};
        external.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
        external.externalFormat = format.format == VK_FORMAT_UNDEFINED ? format.externalFormat : 0;
        VkSamplerYcbcrConversionCreateInfo create = {};
        create.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;
        create.pNext = &external;
        create.format = format.format;
        // The driver knows an RGB buffer from a Y'CbCr one; the frame knows its colour.
        const bool rgb = format.suggestedYcbcrModel == VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY;
        create.ycbcrModel = rgb ? format.suggestedYcbcrModel : modelOf(dataspace);
        create.ycbcrRange = rgb ? format.suggestedYcbcrRange : rangeOf(dataspace);
        create.components = format.samplerYcbcrConversionComponents;
        create.xChromaOffset = format.suggestedXChromaOffset;
        create.yChromaOffset = format.suggestedYChromaOffset;
        const bool linear = (format.formatFeatures &
                             VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT) != 0;
        const VkFilter filter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        create.chromaFilter = filter;
        if (g.CreateSamplerYcbcrConversion(g.device, &create, nullptr, &conversion_) != VK_SUCCESS) {
            return false;
        }
        VkSamplerYcbcrConversionInfo convert = {};
        convert.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
        convert.conversion = conversion_;
        VkSamplerCreateInfo sampler = {};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.pNext = &convert;
        sampler.magFilter = sampler.minFilter = filter;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (g.CreateSampler(g.device, &sampler, nullptr, &sampler_) != VK_SUCCESS) return false;
        VkDescriptorSetLayoutBinding binding = {};
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.pImmutableSamplers = &sampler_;
        VkDescriptorSetLayoutCreateInfo set = {};
        set.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set.bindingCount = 1;
        set.pBindings = &binding;
        if (g.CreateDescriptorSetLayout(g.device, &set, nullptr, &setLayout_) != VK_SUCCESS) {
            return false;
        }
        layout_ = MakePipelineLayout(g, setLayout_);
        if (layout_ == VK_NULL_HANDLE) return false;
        // A multi-planar view can take several descriptors of the pool per set.
        const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                           3 * static_cast<uint32_t>(kMaxSources)};
        VkDescriptorPoolCreateInfo pool = {};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = kMaxSources;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        if (g.CreateDescriptorPool(g.device, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        pipeline_ = MakeFullscreen(g, vertex_, fragment_, layout_, pass_);
        if (pipeline_ == VK_NULL_HANDLE) return false;
        streamFormat_ = format.externalFormat;
        streamVkFormat_ = format.format;
        streamDataspace_ = dataspace;
        ALOGI("%s: video stream format %#llx, model %d, range %d", Engine::get().package().c_str(),
              static_cast<unsigned long long>(format.externalFormat), create.ycbcrModel,
              create.ycbcrRange);
        return true;
    }

    Output* outputFor(const sp<GraphicBuffer>& buffer) {
        const Gpu& g = *gpu_;
        const auto found = outputs_.find(buffer->getId());
        if (found != outputs_.end()) return &found->second;
        if (outputs_.size() >= kMaxOutputs) {
            // The window's buffers came and went: Kelir keeps framebuffers over their views.
            waitRing();
            kelir::vulkan::destroySwapchain(g.device, kelirKey());
            kelir::vulkan::onOffscreen(g.device, kelirKey(), kFormat, {outWidth_, outHeight_},
                                       turns_);
            for (auto& [id, o] : outputs_) freeOutput(o);
            outputs_.clear();
        }
        VkAndroidHardwareBufferFormatPropertiesANDROID format = {};
        format.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
        VkAndroidHardwareBufferPropertiesANDROID properties = {};
        properties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
        properties.pNext = &format;
        if (g.GetAndroidHardwareBufferPropertiesANDROID(g.device, buffer->toAHardwareBuffer(),
                                                        &properties) != VK_SUCCESS) {
            return nullptr;
        }
        Output o;
        o.buffer = buffer;
        VkExternalMemoryImageCreateInfo memory = {};
        memory.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        memory.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
        VkImageCreateInfo image = {};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.pNext = &memory;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = kFormat;
        image.extent = {static_cast<uint32_t>(buffer->getWidth()),
                        static_cast<uint32_t>(buffer->getHeight()), 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!importBuffer(buffer, image, &o.image, &o.memory, properties)) {
            ALOGW("video: a window buffer (%dx%d, format %d) could not be imported",
                  buffer->getWidth(), buffer->getHeight(), buffer->getPixelFormat());
            freeOutput(o);
            return nullptr;
        }
        o.view = MakeView(g, o.image, kFormat, nullptr);
        o.framebuffer = o.view != VK_NULL_HANDLE
                ? MakeFramebuffer(g, pass_, o.view, outWidth_, outHeight_)
                : VK_NULL_HANDLE;
        if (o.framebuffer == VK_NULL_HANDLE) {
            freeOutput(o);
            return nullptr;
        }
        return &outputs_.emplace(buffer->getId(), std::move(o)).first->second;
    }

    void freeSource(Source& s) {
        const Gpu& g = *gpu_;
        if (s.set != VK_NULL_HANDLE) g.FreeDescriptorSets(g.device, pool_, 1, &s.set);
        if (s.view != VK_NULL_HANDLE) g.DestroyImageView(g.device, s.view, nullptr);
        if (s.image != VK_NULL_HANDLE) g.DestroyImage(g.device, s.image, nullptr);
        if (s.memory != VK_NULL_HANDLE) g.FreeMemory(g.device, s.memory, nullptr);
        s = Source();
    }

    void freeOutput(Output& o) {
        const Gpu& g = *gpu_;
        if (o.framebuffer != VK_NULL_HANDLE) g.DestroyFramebuffer(g.device, o.framebuffer, nullptr);
        if (o.view != VK_NULL_HANDLE) g.DestroyImageView(g.device, o.view, nullptr);
        if (o.image != VK_NULL_HANDLE) g.DestroyImage(g.device, o.image, nullptr);
        if (o.memory != VK_NULL_HANDLE) g.FreeMemory(g.device, o.memory, nullptr);
        o = Output();
    }

    void freeConversion() {
        const Gpu& g = *gpu_;
        if (pipeline_ != VK_NULL_HANDLE) g.DestroyPipeline(g.device, pipeline_, nullptr);
        if (pool_ != VK_NULL_HANDLE) g.DestroyDescriptorPool(g.device, pool_, nullptr);
        if (layout_ != VK_NULL_HANDLE) g.DestroyPipelineLayout(g.device, layout_, nullptr);
        if (setLayout_ != VK_NULL_HANDLE) g.DestroyDescriptorSetLayout(g.device, setLayout_, nullptr);
        if (sampler_ != VK_NULL_HANDLE) g.DestroySampler(g.device, sampler_, nullptr);
        if (conversion_ != VK_NULL_HANDLE) {
            g.DestroySamplerYcbcrConversion(g.device, conversion_, nullptr);
        }
        pipeline_ = VK_NULL_HANDLE;
        pool_ = VK_NULL_HANDLE;
        layout_ = VK_NULL_HANDLE;
        setLayout_ = VK_NULL_HANDLE;
        sampler_ = VK_NULL_HANDLE;
        conversion_ = VK_NULL_HANDLE;
        streamFormat_ = 0;
        streamVkFormat_ = VK_FORMAT_UNDEFINED;
        streamDataspace_ = HAL_DATASPACE_UNKNOWN;
    }

    void destroy() {
        const Gpu& g = *gpu_;
        // A plan compile may still be on another thread; it holds this object and touches none of
        // what goes here.
        waitRing();
        if (afme_ != nullptr) {
            std::lock_guard lock(gIdleAfmeLock);
            gIdleAfme = std::move(afme_);
        }
        motion_.reset();
        freeSized();
        for (auto& [id, s] : sources_) freeSource(s);
        sources_.clear();
        freeConversion();
        for (Slot& s : ring_) {
            if (s.fence != VK_NULL_HANDLE) g.DestroyFence(g.device, s.fence, nullptr);
            for (VkSemaphore sem : {s.sourceReady, s.outputFree, s.drawn}) {
                if (sem != VK_NULL_HANDLE) g.DestroySemaphore(g.device, sem, nullptr);
            }
        }
        if (timer_ != VK_NULL_HANDLE) g.DestroyQueryPool(g.device, timer_, nullptr);
        if (commandPool_ != VK_NULL_HANDLE) g.DestroyCommandPool(g.device, commandPool_, nullptr);
        if (upscale_ != VK_NULL_HANDLE) g.DestroyPipeline(g.device, upscale_, nullptr);
        if (upscalePipe_ != VK_NULL_HANDLE) g.DestroyPipelineLayout(g.device, upscalePipe_, nullptr);
        if (upscaleLayout_ != VK_NULL_HANDLE) {
            g.DestroyDescriptorSetLayout(g.device, upscaleLayout_, nullptr);
        }
        if (linear_ != VK_NULL_HANDLE) g.DestroySampler(g.device, linear_, nullptr);
        for (VkShaderModule m : {upscaleModule_, fragment_, vertex_}) {
            if (m != VK_NULL_HANDLE) g.DestroyShaderModule(g.device, m, nullptr);
        }
        if (pass_ != VK_NULL_HANDLE) g.DestroyRenderPass(g.device, pass_, nullptr);
    }

    void report() {
        const int64_t t = now();
        if (reportedAt_ == 0) {
            reportedAt_ = t;
            reportedFrames_ = status_.frames;
            reportedPresents_ = presented_;
            gpuTotalNs_ = 0;
            return;
        }
        if (t - reportedAt_ < kReportNs) return;
        const uint32_t frames = status_.frames - reportedFrames_;
        const auto seconds = static_cast<float>(t - reportedAt_) / 1e9f;
        status_.fps = static_cast<float>(frames) / seconds;
        status_.presentedFps = static_cast<float>(presented_ - reportedPresents_) / seconds;
        if (frames > 0) status_.gpuNs = static_cast<uint32_t>(gpuTotalNs_ / frames);
        status_.appliedRevision = running_ != nullptr ? running_->revision : 0;
        reportedAt_ = t;
        reportedFrames_ = status_.frames;
        reportedPresents_ = presented_;
        gpuTotalNs_ = 0;
        Engine::get().reportVideo(status_);
        if (running_ == nullptr || running_->empty()) return;
        KelirTargetStatus k;
        k.state = KelirState::kRunning;
        k.api = Api::kVulkan;
        k.width = status_.width;
        k.height = status_.height;
        k.rotation = static_cast<uint8_t>(turns_);
        k.path = KelirPath::kVulkan;
        k.signature = kelir::vulkan::offscreenSignature(*running_, kelir::Role::kFull);
        k.gpuNs = status_.gpuNs;
        Engine::get().reportKelir(k);
    }

    void disconnectLocked() {
        if (target_ == nullptr) return;
        // The window's buffers belong to that connection.
        if (state() == FRAMEFX_VIDEO_READY) {
            waitRing();
            if (hasKelirTarget_) {
                kelir::vulkan::destroySwapchain(gpu_->device, kelirKey());
                kelir::vulkan::onOffscreen(gpu_->device, kelirKey(), kFormat,
                                           {outWidth_, outHeight_}, turns_);
            }
            for (auto& [id, o] : outputs_) freeOutput(o);
            outputs_.clear();
        }
        if (votedHz_ != 0) {
            target_->setFrameRate(0.0f, ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_DEFAULT,
                                  ANATIVEWINDOW_CHANGE_FRAME_RATE_ONLY_IF_SEAMLESS);
        }
        votedHz_ = 0;
        target_->setDequeueTimeout(-1);
        native_window_api_disconnect(target_.get(), NATIVE_WINDOW_API_MEDIA);
        target_.clear();
    }

    sp<BufferItemConsumer> consumer_;
    sp<Surface> input_;
    std::thread worker_;
    std::atomic<int32_t> state_{FRAMEFX_VIDEO_WARMING};

    // Never held across a frame, so the decoder's queue never waits on the GPU.
    std::mutex lock_;
    std::condition_variable frames_;
    // Frames wait for the window only until the first attach. After that nothing may hold the
    // decoder back: without a window its frames go.
    bool open_ = false;
    bool stopping_ = false;
    int pending_ = 0;

    // Held across a frame, so a window changes between frames. Covers everything below.
    std::mutex targetLock_;
    sp<Surface> target_;

    const Gpu* gpu_ = nullptr;
    VkRenderPass pass_ = VK_NULL_HANDLE;
    VkShaderModule vertex_ = VK_NULL_HANDLE;
    VkShaderModule fragment_ = VK_NULL_HANDLE;
    VkShaderModule upscaleModule_ = VK_NULL_HANDLE;
    VkSampler linear_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout upscaleLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout upscalePipe_ = VK_NULL_HANDLE;
    VkPipeline upscale_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkQueryPool timer_ = VK_NULL_HANDLE;
    Slot ring_[kRing];
    int ringIndex_ = 0;

    // The stream's format and colour.
    uint64_t streamFormat_ = 0;
    VkFormat streamVkFormat_ = VK_FORMAT_UNDEFINED;
    android_dataspace_t streamDataspace_ = HAL_DATASPACE_UNKNOWN;
    VkSamplerYcbcrConversion conversion_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, Source> sources_;

    // This frame.
    Source* source_ = nullptr;
    android::Rect crop_;
    android_dataspace_t dataspace_ = HAL_DATASPACE_UNKNOWN;
    uint32_t transform_ = 0;
    uint32_t scalingMode_ = 0;

    // The frame's size and what it goes through. The scene is no larger than what is shown.
    uint32_t decodedWidth_ = 0;
    uint32_t decodedHeight_ = 0;
    uint32_t decodedWindowSide_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t outHeight_ = 0;
    int turns_ = 0;
    bool chainMotion_ = false;
    Image scene_;  // the frame as RGB
    VkFramebuffer sceneTarget_ = VK_NULL_HANDLE;
    VkDescriptorPool chainPool_ = VK_NULL_HANDLE;
    VkDescriptorSet sceneSet_ = VK_NULL_HANDLE;
    Image up_;     // upscaled, for Kelir to grade
    VkFramebuffer upTarget_ = VK_NULL_HANDLE;
    bool hasKelirTarget_ = false;
    std::unordered_map<uint64_t, Output> outputs_;

    // With generation: the pair, and a frame of it on its way to Kelir's finish.
    std::unique_ptr<Motion> motion_;
    Image history_[2];
    VkFramebuffer historyTarget_[2] = {};
    Image temp_;
    VkFramebuffer tempTarget_ = VK_NULL_HANDLE;
    int current_ = 0;
    bool hasPair_ = false;
    bool pairPrimed_ = false;
    Timeline timeline_;
    std::unique_ptr<AfmeEngine> afme_;
    std::pair<uint32_t, uint32_t> afmeAt_;  // the size afme_ is bound at
    std::pair<uint32_t, uint32_t> afmeSettling_;
    int afmeSettled_ = 0;
    std::atomic<bool> afmeBuilding_{false};
    std::mutex afmeLock_;
    std::unique_ptr<AfmeEngine> afmeBuilt_;  // guarded by afmeLock_
    std::pair<uint32_t, uint32_t> afmeRefused_;  // guarded by afmeLock_
    bool afmeWanted_ = false;
    bool afmeFresh_ = false;
    bool afmeTooSlow_ = false;
    int afmeLate_ = 0;
    int afmeJudged_ = 0;
    int64_t frameBegan_ = 0;
    int64_t analysisNs_ = 0;
    int64_t earlyNs_ = 0;
    VkDescriptorSet afmeSet_ = VK_NULL_HANDLE;
    bool afmeReady_ = false;
    bool afmeOn_ = false;
    bool afmeFrame_ = false;  // this real frame's pair went through it
    int64_t lastTime_ = 0;
    int64_t readyNs_ = 0;
    int64_t presentNs_ = 0;
    int64_t mediaOffset_ = 0;

    Engine::Watch watch_;
    std::shared_ptr<const kelir::Plan> desired_;
    std::shared_ptr<const kelir::Plan> running_;
    int hz_ = 0;
    int votedHz_ = 0;
    bool upscaleOn_ = false;
    std::atomic<bool> superres_{false};
    float smoothness_ = 1.0f;
    std::atomic<bool> compiling_{false};

    VideoStatus status_;
    int64_t reportedAt_ = 0;
    uint32_t reportedFrames_ = 0;
    uint32_t presented_ = 0;
    uint32_t reportedPresents_ = 0;
    uint64_t gpuTotalNs_ = 0;
};

Proxy* from(framefx_video* video) { return reinterpret_cast<Proxy*>(video); }

int32_t wanted() { return Engine::get().videoWanted() ? 1 : 0; }

int32_t superres(framefx_video* video) { return from(video)->wantsSuperres() ? 1 : 0; }

Proxy* makeProxy() {
    auto* proxy = new Proxy();
    proxy->incStrong(reinterpret_cast<void*>(makeProxy));
    proxy->start();
    return proxy;
}

// One waits warm for the next decoder, so that its first frame already goes through: a feed of
// short videos starts a decoder per video. It holds no image, and lives as long as the process.
std::mutex gSpareLock;
Proxy* gSpare = nullptr;

framefx_video* open() {
    std::lock_guard lock(gSpareLock);
    Proxy* proxy = gSpare != nullptr ? gSpare : makeProxy();
    gSpare = proxy->state() != FRAMEFX_VIDEO_FAILED ? makeProxy() : nullptr;
    return reinterpret_cast<framefx_video*>(proxy);
}

int32_t state(framefx_video* video) { return from(video)->state(); }

ANativeWindow* input(framefx_video* video) { return from(video)->input(); }

int32_t attach(framefx_video* video, ANativeWindow* target) { return from(video)->attach(target); }

void detach(framefx_video* video) { from(video)->detach(); }

void close(framefx_video* video) {
    Proxy* proxy = from(video);
    proxy->close();
    proxy->decStrong(reinterpret_cast<void*>(makeProxy));
}

const framefx_video_api kApi = {
        .abi = FRAMEFX_VIDEO_ABI,
        .wanted = wanted,
        .open = open,
        .state = state,
        .input = input,
        .attach = attach,
        .detach = detach,
        .close = close,
        .superres = superres,
};

}  // namespace

const framefx_video_api* api() { return &kApi; }

}  // namespace framefx::video
