/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

// A game in this process, driving libframefx through the entry points the graphics loaders call:
// its channel as the broker writes it, and its window an in-process image queue, so every frame
// that reaches the display is read back.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <framefx/hook.h>
#include <media/NdkImageReader.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <vector>

#include "src/channel.h"

namespace framefx::test {

const framefx_api* Entry();

class Channel {
public:
    Channel();
    // Rewrites the channel under its sequence lock and wakes the engine's watcher.
    void write(const LuwesSettings& luwes, const std::vector<uint8_t>& kelir);
    int channelFd() const { return channel_; }
    int statusFd() const { return status_; }
    const uint8_t* status() const { return page_; }

private:
    int channel_ = -1;
    int status_ = -1;
    uint8_t* map_ = nullptr;
    const uint8_t* page_ = nullptr;
};

// The channel the engine adopted, once per process.
Channel& TheChannel();

// Publishes both sections and waits until the engine's watcher has applied them.
void Apply(const LuwesSettings& luwes, const std::vector<uint8_t>& kelir = {});

// What reached the queue: RGBA8 bytes, or RGB10A2 widened to 16 bits per channel; top row first.
struct Frame {
    int width = 0;
    int height = 0;
    bool tenBit = false;
    std::vector<uint16_t> rgba;

    // A channel as a value in [0, 1].
    float at(int x, int y, int c) const;
};

class Queue {
public:
    Queue(int width, int height, bool tenBit);
    Queue(const Queue&) = delete;
    ~Queue();
    ANativeWindow* window() const { return window_; }
    // Every frame queued since the last call, top row first: GLES's bottom-up default framebuffer
    // reaches the queue flipped like any other window buffer.
    std::vector<Frame> drain();

private:
    bool tenBit_;
    AImageReader* reader_ = nullptr;
    ANativeWindow* window_ = nullptr;
};

// The game's frame number and buffer size -> its pixels, RGBA8 or (ten-bit) RGB10A2 words, top
// row first.
using Source = std::function<void(int frame, int width, int height, std::vector<uint8_t>* pixels)>;

struct GlOptions {
    int width = 256;
    int height = 128;
    bool srgb = false;    // EGL_GL_COLORSPACE_SRGB_KHR
    bool tenBit = false;  // RGB10A2 config and window
    int samples = 0;      // a multisampled default framebuffer
    bool preserved = false;  // EGL_BUFFER_PRESERVED swaps
};

void* GlProc(const char* name);

struct GlGame {
    GlOptions options;
    Source source;
    Queue queue;
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLConfig config = nullptr;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    void* state = nullptr;
    GLuint texture = 0;
    GLuint framebuffer = 0;
    GLuint program = 0;  // draws the frame where a blit cannot: into a multisampled framebuffer
    int frame = 0;
    // A game's real frame time: present() waits out the rest of it. 0 presents at once.
    int64_t periodNs = 0;
    int64_t startedNs = 0;
    // The last present: when the engine was entered and when it called the swap.
    int64_t enteredNs = 0;
    int64_t swappedNs = 0;

    GlGame(Source source, GlOptions options = {});
    ~GlGame();
    // Renders the next frame, presents it through the engine and returns what reached the queue.
    std::vector<Frame> present();

    static EGLBoolean Swap(void* cookie);
};

struct VkOptions {
    int width = 256;
    int height = 128;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkSurfaceTransformFlagBitsKHR transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    VkSwapchainCreateFlagsKHR flags = 0;
    // Khronos validation on the instance, the engine's calls included (the layer in
    // /data/local/debug/vulkan).
    bool validation = false;
};

// Validation errors and warnings since the last reset, each printed as it comes.
int ValidationMessages();
void ResetValidationMessages();

struct VkGame {
    VkOptions options;
    Source source;
    Queue queue;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> images;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    void* stagingMap = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    std::vector<VkSemaphore> rendered;  // per image: a present waits on it until it comes back
    VkFence done = VK_NULL_HANDLE;
    int frame = 0;
    int64_t periodNs = 0;  // as GlGame's
    int64_t startedNs = 0;
    VkResult lastPresent = VK_SUCCESS;
    // The last present: when the engine was entered and when it handed the driver its present.
    int64_t enteredNs = 0;
    int64_t presentedNs = 0;

    static constexpr int64_t kRefreshNs = 7'407'407;  // a 135 Hz panel

    VkGame(Source source, VkOptions options = {});
    ~VkGame();
    void createSwapchain();
    std::vector<Frame> present();
};

int64_t NowNs();

// Waits until `period` after `*started`, then marks the new start. A game paced to its frame time.
void Pace(int64_t period, int64_t* started);

// Process state, for leak checks.
int OpenFds();
long RssKb();

}  // namespace framefx::test
