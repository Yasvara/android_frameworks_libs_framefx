/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "tests/device/harness.h"

#include <android/hardware_buffer.h>
#include <dirent.h>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <linux/futex.h>
#include <media/NdkImage.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <string>

#include "src/engine.h"
#include "src/vk_colour.h"

namespace framefx::test {
namespace {

void put16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}

void put32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

constexpr uint32_t kTenBitFormat = AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM;
constexpr int32_t kDataspaceSrgb = 142671872;  // ADATASPACE_SRGB

// The game whose swapchain the Vulkan host hooks serve; tests run one at a time.
VkGame* gVkGame = nullptr;

}  // namespace

const framefx_api* Entry() {
    static const framefx_api* api = framefx_entry(FRAMEFX_ABI);
    return api;
}

// ---- Channel ---------------------------------------------------------------------------------

Channel::Channel() {
    channel_ = memfd_create("framefx-test-channel", MFD_CLOEXEC);
    status_ = memfd_create("framefx-test-status", MFD_CLOEXEC);
    ftruncate(channel_, kChannelSize);
    ftruncate(status_, kStatusSize);
    map_ = static_cast<uint8_t*>(
            mmap(nullptr, kChannelSize, PROT_READ | PROT_WRITE, MAP_SHARED, channel_, 0));
    page_ = static_cast<const uint8_t*>(
            mmap(nullptr, kStatusSize, PROT_READ, MAP_SHARED, status_, 0));
    write(LuwesSettings(), {});
}

void Channel::write(const LuwesSettings& luwes, const std::vector<uint8_t>& kelir) {
    uint8_t image[kChannelSize] = {};
    put32(image, kChannelMagic);
    put16(image + 4, kChannelVersion);
    put16(image + 6, kChannelSize);
    size_t next = kSectionDataOffset;
    int count = 0;
    const auto section = [&](uint8_t id, const uint8_t* bytes, size_t length) {
        uint8_t* entry = image + kSectionTableOffset + 8 * count++;
        entry[0] = id;
        put16(entry + 2, static_cast<uint16_t>(next));
        put16(entry + 4, static_cast<uint16_t>(length));
        memcpy(image + next, bytes, length);
        next += length;
    };
    if (luwes.on()) {
        const uint8_t d[kLuwesSectionSize] = {kLuwesMajor, static_cast<uint8_t>(luwes.multiplier),
                                              static_cast<uint8_t>(luwes.mode),
                                              luwes.hudProtection ? kLuwesHudProtection : uint8_t{0}};
        section(kSectionLuwes, d, sizeof(d));
    }
    if (!kelir.empty()) section(kSectionKelir, kelir.data(), kelir.size());
    image[12] = static_cast<uint8_t>(count);
    auto* word = reinterpret_cast<uint32_t*>(map_ + kSequenceOffset);
    const uint32_t sequence = __atomic_load_n(word, __ATOMIC_RELAXED);
    __atomic_store_n(word, sequence + 1, __ATOMIC_RELAXED);
    std::atomic_thread_fence(std::memory_order_release);
    memcpy(map_, image, kSequenceOffset);
    memcpy(map_ + kSequenceOffset + 4, image + kSequenceOffset + 4,
           kChannelSize - kSequenceOffset - 4);
    __atomic_store_n(word, sequence + 2, __ATOMIC_RELEASE);
    syscall(SYS_futex, word, FUTEX_WAKE, INT32_MAX, nullptr, nullptr, 0);
}

Channel& TheChannel() {
    static Channel* channel = [] {
        auto* c = new Channel();
        EXPECT_TRUE(Engine::get().adopt(c->channelFd(), c->statusFd()));
        return c;
    }();
    return *channel;
}

void Apply(const LuwesSettings& luwes, const std::vector<uint8_t>& kelir) {
    Channel& channel = TheChannel();
    const uint32_t before = Engine::get().generation();
    channel.write(luwes, kelir);
    for (int i = 0; i < 2000 && Engine::get().generation() == before; i++) usleep(1000);
    ASSERT_NE(before, Engine::get().generation()) << "the watcher never published";
    ASSERT_EQ(luwes, Engine::get().config().luwes);
}

// ---- Frames and the queue --------------------------------------------------------------------

float Frame::at(int x, int y, int c) const {
    return rgba[(static_cast<size_t>(y) * width + x) * 4 + c] / (tenBit ? 1023.0f : 255.0f);
}

Queue::Queue(int width, int height, bool tenBit) : tenBit_(tenBit) {
    const uint64_t usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
            AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
    const media_status_t s = tenBit
            ? AImageReader_newWithDataSpace(width, height, usage, 8, kTenBitFormat, kDataspaceSrgb,
                                            &reader_)
            : AImageReader_newWithUsage(width, height, AIMAGE_FORMAT_RGBA_8888, usage, 8, &reader_);
    EXPECT_EQ(AMEDIA_OK, s);
    EXPECT_EQ(AMEDIA_OK, AImageReader_getWindow(reader_, &window_));
}

Queue::~Queue() {
    AImageReader_delete(reader_);
}

std::vector<Frame> Queue::drain() {
    std::vector<Frame> frames;
    AImage* image = nullptr;
    while (AImageReader_acquireNextImage(reader_, &image) == AMEDIA_OK) {
        AHardwareBuffer* buffer = nullptr;
        AImage_getHardwareBuffer(image, &buffer);
        AHardwareBuffer_Desc desc = {};
        AHardwareBuffer_describe(buffer, &desc);
        void* data = nullptr;
        EXPECT_EQ(0, AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr,
                                          &data));
        // A pre-rotated swapchain queues buffers turned against the window.
        Frame f;
        f.width = static_cast<int>(desc.width);
        f.height = static_cast<int>(desc.height);
        f.tenBit = tenBit_;
        f.rgba.resize(static_cast<size_t>(f.width) * f.height * 4);
        for (int y = 0; y < f.height; y++) {
            const auto* src = static_cast<const uint8_t*>(data) +
                    static_cast<size_t>(y) * desc.stride * 4;
            for (int x = 0; x < f.width; x++) {
                uint16_t* d = &f.rgba[(static_cast<size_t>(y) * f.width + x) * 4];
                if (tenBit_) {
                    uint32_t w;
                    memcpy(&w, src + x * 4, 4);
                    d[0] = w & 0x3ff;
                    d[1] = (w >> 10) & 0x3ff;
                    d[2] = (w >> 20) & 0x3ff;
                    d[3] = static_cast<uint16_t>((w >> 30) * 341);
                } else {
                    for (int c = 0; c < 4; c++) d[c] = src[x * 4 + c];
                }
            }
        }
        AHardwareBuffer_unlock(buffer, nullptr);
        frames.push_back(std::move(f));
        AImage_delete(image);
    }
    return frames;
}

// ---- GLES ------------------------------------------------------------------------------------

namespace {

GLuint Shader(GLenum type, const char* source) {
    const GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &source, nullptr);
    glCompileShader(s);
    return s;
}

// A fullscreen triangle that copies the frame's texture texel for texel.
GLuint CopyProgram() {
    const GLuint program = glCreateProgram();
    glAttachShader(program, Shader(GL_VERTEX_SHADER, R"(#version 300 es
        void main() {
            vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
        })"));
    glAttachShader(program, Shader(GL_FRAGMENT_SHADER, R"(#version 300 es
        precision highp float;
        uniform highp sampler2D frame;
        out vec4 color;
        void main() { color = texelFetch(frame, ivec2(gl_FragCoord.xy), 0); })"));
    glLinkProgram(program);
    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    EXPECT_EQ(GL_TRUE, linked);
    return program;
}

}  // namespace

void* GlProc(const char* name) {
    void* p = dlsym(RTLD_DEFAULT, name);
    return p != nullptr ? p : reinterpret_cast<void*>(eglGetProcAddress(name));
}

GlGame::GlGame(Source s, GlOptions o)
    : options(o), source(std::move(s)), queue(o.width, o.height, o.tenBit) {
    eglInitialize(display, nullptr, nullptr);
    const EGLint bits = o.tenBit ? 10 : 8;
    const EGLint surfaceType = EGL_WINDOW_BIT | (o.preserved ? EGL_SWAP_BEHAVIOR_PRESERVED_BIT : 0);
    const EGLint attribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                              EGL_SURFACE_TYPE,    surfaceType,
                              EGL_RED_SIZE,        bits,
                              EGL_GREEN_SIZE,      bits,
                              EGL_BLUE_SIZE,       bits,
                              EGL_ALPHA_SIZE,      o.tenBit ? 2 : 8,
                              EGL_SAMPLE_BUFFERS,  o.samples > 0 ? 1 : 0,
                              EGL_SAMPLES,         o.samples,
                              EGL_NONE};
    EGLint count = 0;
    eglChooseConfig(display, attribs, &config, 1, &count);
    EXPECT_EQ(1, count);
    const EGLint version[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, version);
    const EGLint srgb[] = {EGL_GL_COLORSPACE_KHR, EGL_GL_COLORSPACE_SRGB_KHR, EGL_NONE};
    surface = eglCreateWindowSurface(display, config, queue.window(), o.srgb ? srgb : nullptr);
    EXPECT_NE(EGL_NO_SURFACE, surface);
    if (o.preserved) {
        EXPECT_TRUE(eglSurfaceAttrib(display, surface, EGL_SWAP_BEHAVIOR, EGL_BUFFER_PRESERVED));
    }
    EXPECT_TRUE(eglMakeCurrent(display, surface, surface, context));
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, o.tenBit ? GL_RGB10_A2 : o.srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8,
                   o.width, o.height);
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    if (o.samples > 0) program = CopyProgram();
    const framefx_gl_host host = {GlProc};
    Entry()->gl_attach(&host);
}

GlGame::~GlGame() {
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    // libEGL releases the engine's state with the context.
    if (state != nullptr) Entry()->gl_release(state);
    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
}

EGLBoolean GlGame::Swap(void* cookie) {
    auto* g = static_cast<GlGame*>(cookie);
    g->swappedNs = NowNs();
    return eglSwapBuffers(g->display, g->surface);
}

std::vector<Frame> GlGame::present() {
    Pace(periodNs, &startedNs);
    const int w = options.width;
    const int h = options.height;
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
    source(frame, w, h, &pixels);
    // GL's first row is the bottom one: upload bottom-up so the frame reads top-down.
    std::vector<uint8_t> flipped(pixels.size());
    for (int y = 0; y < h; y++) {
        memcpy(&flipped[static_cast<size_t>(y) * w * 4], &pixels[static_cast<size_t>(h - 1 - y) * w * 4],
               static_cast<size_t>(w) * 4);
    }
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA,
                    options.tenBit ? GL_UNSIGNED_INT_2_10_10_10_REV : GL_UNSIGNED_BYTE,
                    flipped.data());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    if (program != 0) {
        glUseProgram(program);
        glActiveTexture(GL_TEXTURE0);
        glViewport(0, 0, w, h);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glUseProgram(0);
    } else {
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }
    const framefx_gl_frame f = {&state, display, surface, config, queue.window(), Swap, this};
    enteredNs = NowNs();
    if (Entry()->gl_present(&f) == FRAMEFX_PASS) Swap(this);
    EXPECT_EQ(static_cast<GLenum>(GL_NO_ERROR), glGetError());
    frame++;
    return queue.drain();
}

// ---- Vulkan ----------------------------------------------------------------------------------

namespace {

VkResult Acquire(VkDevice d, VkSwapchainKHR s, VkSemaphore sem, uint32_t* index) {
    return vkAcquireNextImageKHR(d, s, UINT64_MAX, sem, VK_NULL_HANDLE, index);
}

std::atomic<int> gValidationMessages{0};

VKAPI_ATTR VkBool32 VKAPI_CALL OnValidation(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                            VkDebugUtilsMessageTypeFlagsEXT,
                                            const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    gValidationMessages++;
    printf("validation: %s\n", data->pMessage);
    return VK_FALSE;
}

VkResult Present(VkQueue queue, const VkPresentInfoKHR* info) {
    if (gVkGame != nullptr) gVkGame->presentedNs = NowNs();
    return vkQueuePresentKHR(queue, info);
}

int64_t Refresh(VkSwapchainKHR) {
    return VkGame::kRefreshNs;
}

ANativeWindow* Window(VkSwapchainKHR) {
    return gVkGame != nullptr ? gVkGame->queue.window() : nullptr;
}

// The loader's own release is internal: here it is swapchain_maintenance1's, once the GPU is
// past the semaphore.
VkResult Release(VkQueue queue, VkSwapchainKHR swapchain, uint32_t index, VkSemaphore semaphore) {
    VkGame& g = *gVkGame;
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(g.device, &fci, nullptr, &fence);
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = semaphore != VK_NULL_HANDLE ? 1 : 0;
    submit.pWaitSemaphores = &semaphore;
    submit.pWaitDstStageMask = &stage;
    vkQueueSubmit(queue, 1, &submit, fence);
    vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(g.device, fence, nullptr);
    VkReleaseSwapchainImagesInfoEXT info = {};
    info.sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT;
    info.swapchain = swapchain;
    info.imageIndexCount = 1;
    info.pImageIndices = &index;
    const auto release = reinterpret_cast<PFN_vkReleaseSwapchainImagesEXT>(
            vkGetDeviceProcAddr(g.device, "vkReleaseSwapchainImagesEXT"));
    return release(g.device, &info);
}

uint32_t BytesPerPixel(VkFormat) {
    return 4;
}

}  // namespace

VkGame::VkGame(Source s, VkOptions o)
    : options(o),
      source(std::move(s)),
      queue(o.width, o.height, o.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
    gVkGame = this;
    const char* instanceExtensions[] = {VK_KHR_SURFACE_EXTENSION_NAME,
                                        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
                                        VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
                                        VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
                                        VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
                                        VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_1;
    VkDebugUtilsMessengerCreateInfoEXT messages = {};
    messages.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messages.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messages.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messages.pfnUserCallback = OnValidation;
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pNext = o.validation ? &messages : nullptr;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = o.validation ? 1 : 0;
    ici.ppEnabledLayerNames = layers;
    ici.enabledExtensionCount = o.validation ? 6 : 5;
    ici.ppEnabledExtensionNames = instanceExtensions;
    EXPECT_EQ(VK_SUCCESS, vkCreateInstance(&ici, nullptr, &instance)) << "validation " << o.validation;
    if (o.validation) {
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        EXPECT_EQ(VK_SUCCESS, create(instance, &messages, nullptr, &messenger));
    }
    uint32_t count = 1;
    vkEnumeratePhysicalDevices(instance, &count, &physical);

    const framefx_vk_instance in = {
            reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                    vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2")),
            reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                    vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2")),
            reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties2>(
                    vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFormatProperties2")),
            reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                    vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2")),
            vkEnumerateDeviceExtensionProperties,
            vkCreateDevice,
            vkGetDeviceProcAddr,
    };
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                      VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME};
    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT maintenance = {};
    maintenance.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
    maintenance.swapchainMaintenance1 = VK_TRUE;
    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &maintenance;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = deviceExtensions;
    EXPECT_EQ(VK_SUCCESS, Entry()->vk_create_device(&in, physical, &dci, nullptr, &device));
    vkGetDeviceQueue(device, 0, 0, &graphics);
    Entry()->vk_get_device_queue(device, 0, graphics);

    const framefx_vk_host host = {
            vkCreateSwapchainKHR, vkDestroySwapchainKHR, vkGetSwapchainImagesKHR,
            Present,              Acquire,               Release,
            Refresh,              Window,
    };
    Entry()->vk_attach(&host);

    VkAndroidSurfaceCreateInfoKHR sci = {};
    sci.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    sci.window = queue.window();
    EXPECT_EQ(VK_SUCCESS, vkCreateAndroidSurfaceKHR(instance, &sci, nullptr, &surface));
    createSwapchain();

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = static_cast<VkDeviceSize>(o.width) * o.height * BytesPerPixel(o.format);
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(device, &bci, nullptr, &staging);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, staging, &req);
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(physical, &mem);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags want =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((req.memoryTypeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want) {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    vkAllocateMemory(device, &mai, nullptr, &stagingMemory);
    vkBindBufferMemory(device, staging, stagingMemory, 0);
    vkMapMemory(device, stagingMemory, 0, VK_WHOLE_SIZE, 0, &stagingMap);

    VkCommandPoolCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cai = {};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cai, &cb);
    VkSemaphoreCreateInfo sem = {};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(device, &sem, nullptr, &acquired);
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device, &fci, nullptr, &done);
}

void VkGame::createSwapchain() {
    VkSwapchainCreateInfoKHR ci = {};
    ci.flags = options.flags;
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface;
    VkSurfaceCapabilitiesKHR caps = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);
    ci.minImageCount = caps.minImageCount;
    ci.imageFormat = options.format;
    ci.imageColorSpace = options.colorSpace;
    const bool turned = options.transform == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
            options.transform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR;
    ci.imageExtent = {static_cast<uint32_t>(turned ? options.height : options.width),
                      static_cast<uint32_t>(turned ? options.width : options.height)};
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.preTransform = options.transform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain;
    VkSwapchainKHR next = VK_NULL_HANDLE;
    EXPECT_EQ(VK_SUCCESS, Entry()->vk_create_swapchain(device, &ci, nullptr, &next));
    if (swapchain != VK_NULL_HANDLE) Entry()->vk_destroy_swapchain(device, swapchain, nullptr);
    swapchain = next;
    uint32_t count = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr);
    images.resize(count);
    vkGetSwapchainImagesKHR(device, swapchain, &count, images.data());
    VkSemaphoreCreateInfo sem = {};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    while (rendered.size() < count) {
        VkSemaphore s = VK_NULL_HANDLE;
        vkCreateSemaphore(device, &sem, nullptr, &s);
        rendered.push_back(s);
    }
    // Framefx brings a spare image only to a swapchain it can take.
    if (ColourOf(options.colorSpace, options.format) != SwapchainColour::kHdr) {
        EXPECT_GE(count, caps.minImageCount + 1) << "one image more than asked";
    }
}

VkGame::~VkGame() {
    vkDeviceWaitIdle(device);
    vkDestroyFence(device, done, nullptr);
    vkDestroySemaphore(device, acquired, nullptr);
    for (VkSemaphore s : rendered) vkDestroySemaphore(device, s, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    Entry()->vk_destroy_swapchain(device, swapchain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    Entry()->vk_destroy_device(device);
    vkDestroyDevice(device, nullptr);
    if (messenger != VK_NULL_HANDLE) {
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"))(instance, messenger,
                                                                                   nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    gVkGame = nullptr;
}

std::vector<Frame> VkGame::present() {
    Pace(periodNs, &startedNs);
    vkWaitForFences(device, 1, &done, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &done);
    const bool turned = options.transform == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
            options.transform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR;
    const uint32_t w = static_cast<uint32_t>(turned ? options.height : options.width);
    const uint32_t h = static_cast<uint32_t>(turned ? options.width : options.height);
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
    source(frame, static_cast<int>(w), static_cast<int>(h), &pixels);
    memcpy(stagingMap, pixels.data(), pixels.size());
    uint32_t index = 0;
    EXPECT_EQ(VK_SUCCESS,
              vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &index));
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &begin);
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = images[index];
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy copy = {};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cb, staging, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copy);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
    vkEndCommandBuffer(cb);
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &acquired;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &rendered[index];
    EXPECT_EQ(VK_SUCCESS, vkQueueSubmit(graphics, 1, &submit, done));
    VkPresentInfoKHR present = {};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &rendered[index];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &index;
    enteredNs = NowNs();
    lastPresent = Entry()->vk_queue_present(graphics, &present);
    vkQueueWaitIdle(graphics);
    if (lastPresent == VK_ERROR_OUT_OF_DATE_KHR) {
        createSwapchain();
    } else {
        EXPECT_GE(lastPresent, VK_SUCCESS);  // a pre-rotated swapchain may be suboptimal
    }
    frame++;
    return queue.drain();
}

// ---- Process state ---------------------------------------------------------------------------

int OpenFds() {
    int n = 0;
    if (DIR* d = opendir("/proc/self/fd")) {
        while (readdir(d) != nullptr) n++;
        closedir(d);
    }
    return n;
}

long RssKb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
    }
    return 0;
}

int64_t NowNs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

void Pace(int64_t period, int64_t* started) {
    if (period > 0 && *started > 0) {
        const int64_t until = *started + period;
        for (int64_t now = NowNs(); now < until; now = NowNs()) usleep(static_cast<useconds_t>((until - now) / 1000));
    }
    *started = NowNs();
}

int ValidationMessages() {
    return gValidationMessages.load();
}

void ResetValidationMessages() {
    gValidationMessages = 0;
}

}  // namespace framefx::test
