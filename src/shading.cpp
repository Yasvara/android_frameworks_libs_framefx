/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "src/shading.h"

#include <GLES2/gl2.h>
#include <log/log.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "lapis/recorder.h"
#include "src/pipeline_set.h"
#include "src/engine.h"

extern "C" {
uint8_t framefx_lapis_shading_frame(void* governor, int64_t now_ns, int64_t refresh_ns,
                                    uint8_t max_level);
size_t framefx_lapis_shading_governor_size();
}

#ifndef GL_SHADING_RATE_1X1_PIXELS_QCOM
#define GL_SHADING_RATE_1X1_PIXELS_QCOM 0x96A6
#define GL_SHADING_RATE_2X1_PIXELS_QCOM 0x96A8
#define GL_SHADING_RATE_2X2_PIXELS_QCOM 0x96A9
#define GL_SHADING_RATE_4X4_PIXELS_QCOM 0x96AE
#endif

namespace framefx::shading {
namespace {

// The rate code of the frame being drawn, written once per present, read at every bind.
std::atomic<int> gRate{0};

const VkBaseInStructure* find(const void* chain, VkStructureType type) {
    for (auto* p = static_cast<const VkBaseInStructure*>(chain); p != nullptr; p = p->pNext) {
        if (p->sType == type) return p;
    }
    return nullptr;
}

// --- Pipelines that carry the rate as dynamic state -------------------------------------------

// Depth-tested pipelines with the rate as dynamic state, ours or the game's: they shade at the
// frame's rate, or the game's own when that is coarser.
PipelineSet& scenePipelines() {
    static PipelineSet* set = new PipelineSet();
    return *set;
}
// The game's own dynamic-rate pipelines that test no depth: they get the game's rate back.
PipelineSet& flatPipelines() {
    static PipelineSet* set = new PipelineSet();
    return *set;
}

uint64_t bits(VkPipeline p) {
    return reinterpret_cast<uint64_t>(p);
}

// Render passes by subpass: bit i set where subpass i has a depth attachment.
std::mutex gPassLock;
std::unordered_map<uint64_t, uint64_t> gPassDepth;
std::unordered_map<VkDevice, bool> gDevices;  // granted devices
std::atomic<PFN_vkCmdSetFragmentShadingRateKHR> gSetRate{nullptr};
std::atomic<uint32_t> gSeen{0};

bool granted(VkDevice device) {
    std::lock_guard lock(gPassLock);
    const auto it = gDevices.find(device);
    return it != gDevices.end() && it->second;
}

template <typename Desc>
uint64_t depthMask(const Desc* subpasses, uint32_t count) {
    uint64_t mask = 0;
    for (uint32_t i = 0; i < count && i < 64; i++) {
        const auto* d = subpasses[i].pDepthStencilAttachment;
        if (d != nullptr && d->attachment != VK_ATTACHMENT_UNUSED) mask |= 1ull << i;
    }
    return mask;
}

void rememberPass(VkRenderPass pass, uint64_t mask) {
    std::lock_guard lock(gPassLock);
    gPassDepth[reinterpret_cast<uint64_t>(pass)] = mask;
}

std::atomic<PFN_vkCreateRenderPass> gCreateRenderPass{nullptr};
std::atomic<PFN_vkCreateRenderPass2> gCreateRenderPass2{nullptr};
std::atomic<PFN_vkDestroyRenderPass> gDestroyRenderPass{nullptr};
std::atomic<PFN_vkCreateGraphicsPipelines> gCreateGraphics{nullptr};
std::atomic<PFN_vkDestroyPipeline> gDestroyPipeline{nullptr};
std::atomic<PFN_vkCmdBindPipeline> gBindPipeline{nullptr};

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* info,
                                                const VkAllocationCallbacks* allocator,
                                                VkRenderPass* pass) {
    const VkResult r = gCreateRenderPass.load(std::memory_order_relaxed)(device, info, allocator, pass);
    if (r == VK_SUCCESS) rememberPass(*pass, depthMask(info->pSubpasses, info->subpassCount));
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass2(VkDevice device,
                                                 const VkRenderPassCreateInfo2* info,
                                                 const VkAllocationCallbacks* allocator,
                                                 VkRenderPass* pass) {
    const VkResult r = gCreateRenderPass2.load(std::memory_order_relaxed)(device, info, allocator, pass);
    if (r == VK_SUCCESS) rememberPass(*pass, depthMask(info->pSubpasses, info->subpassCount));
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyRenderPass(VkDevice device, VkRenderPass pass,
                                             const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(gPassLock);
        gPassDepth.erase(reinterpret_cast<uint64_t>(pass));
    }
    gDestroyRenderPass.load(std::memory_order_relaxed)(device, pass, allocator);
}

bool hasDynamic(const VkPipelineDynamicStateCreateInfo* d, VkDynamicState state) {
    if (d == nullptr) return false;
    for (uint32_t i = 0; i < d->dynamicStateCount; i++) {
        if (d->pDynamicStates[i] == state) return true;
    }
    return false;
}

// What a pipeline is to shading, counted for the log.
enum Verdict {
    kShades,      // depth-tested; we add the dynamic rate
    kGameScene,   // depth-tested with the game's own dynamic rate
    kGameFlat,    // the game's own dynamic rate, no depth test
    kOwnRate,     // a fixed rate of the game's, or a library
    kNoDepthTest,
    kUnknownPass,
    kNoDepthAttachment,
    kVerdicts
};
std::atomic<uint32_t> gVerdicts[kVerdicts] = {};

// A depth test that passes everything (UI drawn over the scene in its pass) is none.
bool testsDepth(const VkGraphicsPipelineCreateInfo& info) {
    const VkPipelineDepthStencilStateCreateInfo* ds = info.pDepthStencilState;
    if (ds == nullptr ||
        (info.pRasterizationState != nullptr && info.pRasterizationState->rasterizerDiscardEnable)) {
        return false;
    }
    const bool on =
            ds->depthTestEnable || hasDynamic(info.pDynamicState, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
    const bool always = ds->depthCompareOp == VK_COMPARE_OP_ALWAYS &&
            !hasDynamic(info.pDynamicState, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);
    return on && !always;
}

// 3D work: it tests depth into a depth attachment.
Verdict depthVerdict(const VkGraphicsPipelineCreateInfo& info) {
    if (!testsDepth(info)) return kNoDepthTest;
    if (info.renderPass == VK_NULL_HANDLE) {
        const auto* rendering = reinterpret_cast<const VkPipelineRenderingCreateInfo*>(
                find(info.pNext, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO));
        return rendering != nullptr && rendering->depthAttachmentFormat != VK_FORMAT_UNDEFINED
                ? kShades
                : kNoDepthAttachment;
    }
    std::lock_guard lock(gPassLock);
    const auto it = gPassDepth.find(reinterpret_cast<uint64_t>(info.renderPass));
    if (it == gPassDepth.end()) return kUnknownPass;
    return info.subpass < 64 && (it->second >> info.subpass & 1) != 0 ? kShades
                                                                        : kNoDepthAttachment;
}

Verdict verdictOf(const VkGraphicsPipelineCreateInfo& info) {
    if ((info.flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) != 0) return kOwnRate;
    const bool gameDynamic = hasDynamic(info.pDynamicState, VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR);
    if (!gameDynamic) {
        // A fixed rate other than full is the game's choice; a full one we override dynamically.
        const auto* own = reinterpret_cast<const VkPipelineFragmentShadingRateStateCreateInfoKHR*>(
                find(info.pNext, VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR));
        if (own != nullptr && (own->fragmentSize.width != 1 || own->fragmentSize.height != 1)) {
            return kOwnRate;
        }
    }
    const Verdict depth = depthVerdict(info);
    if (gameDynamic) return depth == kShades ? kGameScene : kGameFlat;
    return depth;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache,
                                                       uint32_t n,
                                                       const VkGraphicsPipelineCreateInfo* infos,
                                                       const VkAllocationCallbacks* allocator,
                                                       VkPipeline* out) {
    const PFN_vkCreateGraphicsPipelines next = gCreateGraphics.load(std::memory_order_relaxed);
    if (!granted(device)) return next(device, cache, n, infos, allocator, out);
    std::vector<Verdict> verdicts(n);
    std::vector<bool> flagged(n, false);
    size_t count = 0;
    size_t tracked = 0;
    for (uint32_t i = 0; i < n; i++) {
        verdicts[i] = verdictOf(infos[i]);
        gVerdicts[verdicts[i]].fetch_add(1, std::memory_order_relaxed);
        flagged[i] = verdicts[i] == kShades;
        count += flagged[i];
        tracked += verdicts[i] == kGameScene || verdicts[i] == kGameFlat;
    }
    const uint32_t seen = gSeen.fetch_add(n, std::memory_order_relaxed) + n;
    if ((seen & (seen - 1)) == 0 || (seen - n) >> 8 != seen >> 8) {
        ALOGI("shading: %u pipelines: %u shade, %u 3D and %u flat at the game's own dynamic "
              "rate, %u fixed rate, %u test no depth, %u in unknown passes, %u in passes "
              "without depth",
              seen, gVerdicts[kShades].load(), gVerdicts[kGameScene].load(),
              gVerdicts[kGameFlat].load(), gVerdicts[kOwnRate].load(),
              gVerdicts[kNoDepthTest].load(), gVerdicts[kUnknownPass].load(),
              gVerdicts[kNoDepthAttachment].load());
    }
    // The game's own dynamic-rate pipelines need nothing changed, only remembering.
    if (count + tracked == 0 || !scenePipelines().reserve(count + tracked)) {
        return next(device, cache, n, infos, allocator, out);
    }
    std::vector<VkGraphicsPipelineCreateInfo> changed(infos, infos + n);
    std::vector<VkPipelineDynamicStateCreateInfo> dynamics(n);
    std::vector<std::vector<VkDynamicState>> states(n);
    for (uint32_t i = 0; i < n; i++) {
        if (!flagged[i]) continue;
        const VkPipelineDynamicStateCreateInfo* d = infos[i].pDynamicState;
        if (d != nullptr) {
            dynamics[i] = *d;
            states[i].assign(d->pDynamicStates, d->pDynamicStates + d->dynamicStateCount);
        } else {
            dynamics[i] = {};
            dynamics[i].sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        }
        states[i].push_back(VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR);
        dynamics[i].dynamicStateCount = static_cast<uint32_t>(states[i].size());
        dynamics[i].pDynamicStates = states[i].data();
        changed[i].pDynamicState = &dynamics[i];
    }
    const VkResult r = next(device, cache, n, changed.data(), allocator, out);
    for (uint32_t i = 0; i < n; i++) {
        if (out[i] == VK_NULL_HANDLE) continue;
        if (flagged[i] || verdicts[i] == kGameScene) scenePipelines().insert(bits(out[i]));
        if (verdicts[i] == kGameFlat) flatPipelines().insert(bits(out[i]));
    }
    scenePipelines().unreserve(count + tracked);
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline,
                                           const VkAllocationCallbacks* allocator) {
    if (pipeline != VK_NULL_HANDLE) {
        scenePipelines().erase(bits(pipeline));
        flatPipelines().erase(bits(pipeline));
    }
    gDestroyPipeline.load(std::memory_order_relaxed)(device, pipeline, allocator);
}

// The rate state of the command buffer this thread records: the game's own last rate, and whether
// what is bound now shades at ours. A thread that switches buffers starts over from full rate.
struct Recording {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkExtent2D game = {1, 1};
    VkFragmentShadingRateCombinerOpKHR ops[2] = {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                                                 VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR};
    bool scene = false;      // a depth-tested dynamic-rate pipeline is bound
    bool overridden = false;  // the buffer holds our rate, not the game's
};
thread_local Recording tRecording;

Recording& recordingOf(VkCommandBuffer cb) {
    if (tRecording.cb != cb) tRecording = Recording{cb};
    return tRecording;
}

std::atomic<PFN_vkCmdSetFragmentShadingRateKHR> gGameSetRate{nullptr};

// The coarser of the game's rate and the frame's.
VkExtent2D sceneSize(const Recording& r) {
    const VkExtent2D ours = sizeOf(gRate.load(std::memory_order_relaxed));
    return {std::max(r.game.width, ours.width), std::max(r.game.height, ours.height)};
}

void emit(VkCommandBuffer cb, Recording& r, bool scene) {
    const PFN_vkCmdSetFragmentShadingRateKHR set = gSetRate.load(std::memory_order_relaxed);
    if (set == nullptr) return;
    const VkExtent2D size = scene ? sceneSize(r) : r.game;
    set(cb, &size, r.ops);
    r.overridden = scene && (size.width != r.game.width || size.height != r.game.height);
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point,
                                           VkPipeline pipeline) {
    gBindPipeline.load(std::memory_order_relaxed)(cb, point, pipeline);
    if (point != VK_PIPELINE_BIND_POINT_GRAPHICS) return;
    if (scenePipelines().contains(bits(pipeline))) {
        // Where Lapis has found the scene, only its pass before the tonemap and HUD shades
        // coarse: a HUD drawn with depth (Genshin's) keeps the game's rate. Every bind sets the
        // rate, as the pipelines we made dynamic have no other.
        Recording& r = recordingOf(cb);
        r.scene = lapis::sceneState(cb) != 0;
        emit(cb, r, r.scene);
    } else if (flatPipelines().contains(bits(pipeline))) {
        Recording& r = recordingOf(cb);
        r.scene = false;
        if (r.overridden) emit(cb, r, false);
    } else if (tRecording.cb == cb) {
        tRecording.scene = false;
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetFragmentShadingRate(
        VkCommandBuffer cb, const VkExtent2D* size,
        const VkFragmentShadingRateCombinerOpKHR ops[2]) {
    Recording& r = recordingOf(cb);
    r.game = *size;
    r.ops[0] = ops[0];
    r.ops[1] = ops[1];
    static std::atomic<uint32_t> logged{0};
    if ((size->width != 1 || size->height != 1) && logged.fetch_add(1) < 4) {
        ALOGI("shading: the game sets %ux%u itself", size->width, size->height);
    }
    if (!r.scene) {
        gGameSetRate.load(std::memory_order_relaxed)(cb, size, ops);
        r.overridden = false;
        return;
    }
    emit(cb, r, true);
}

template <typename Fn>
PFN_vkVoidFunction wrap(std::atomic<Fn>* slot, PFN_vkVoidFunction next, Fn own) {
    Fn none = nullptr;
    const auto want = reinterpret_cast<Fn>(next);
    if (!slot->compare_exchange_strong(none, want) && none != want) return nullptr;
    return reinterpret_cast<PFN_vkVoidFunction>(own);
}

// --- Rate per frame ---------------------------------------------------------------------------

struct Governor {
    alignas(8) uint8_t state[256] = {};
};
std::mutex gGovernorLock;
Governor gGovernor;
int gLogged = -1;

// --- GLES ---------------------------------------------------------------------------------------

using EnableFn = void (*)(GLenum);
using ShadingRateFn = void (*)(GLenum);
std::atomic<EnableFn> gEnable{nullptr};
std::atomic<EnableFn> gDisable{nullptr};
std::atomic<ShadingRateFn> gGlRate{nullptr};
// A copy: libEGL hands the host struct over from its stack.
framefx_gl_host gGlHost = {};
std::once_flag gGlOnce;

GLenum glRateOf(int rate) {
    switch (rate) {
        case 1: return GL_SHADING_RATE_2X1_PIXELS_QCOM;
        case 2: return GL_SHADING_RATE_2X2_PIXELS_QCOM;
        case 3: return GL_SHADING_RATE_4X4_PIXELS_QCOM;
        default: return GL_SHADING_RATE_1X1_PIXELS_QCOM;
    }
}

// The driver's rate entry point, looked up from inside the game's GL call so its context is
// current: GL_QCOM_shading_rate, else the EXT one whose rate values match.
ShadingRateFn glShadingRate() {
    std::call_once(gGlOnce, [] {
        if (gGlHost.proc == nullptr) return;
        using GetString = const GLubyte* (*)(GLenum);
        const auto getString = reinterpret_cast<GetString>(gGlHost.proc("glGetString"));
        const char* all = getString != nullptr
                ? reinterpret_cast<const char*>(getString(GL_EXTENSIONS))
                : nullptr;
        if (all == nullptr) return;
        void* fn = nullptr;
        if (strstr(all, "GL_QCOM_shading_rate") != nullptr) fn = gGlHost.proc("glShadingRateQCOM");
        if (fn == nullptr && strstr(all, "GL_EXT_fragment_shading_rate") != nullptr) {
            fn = gGlHost.proc("glShadingRateEXT");
        }
        gGlRate.store(reinterpret_cast<ShadingRateFn>(fn), std::memory_order_relaxed);
        ALOGI("shading: GLES %s", fn != nullptr ? "rate while depth test is on" : "has no shading rate");
    });
    return gGlRate.load(std::memory_order_relaxed);
}

// GLES shading only in a game that draws with GLES, on the thread that presents its frames: a
// Vulkan game's WebView or a helper context keeps full rate.
std::atomic<bool> gVulkanGame{false};
thread_local bool tGamePresents = false;

bool glShades() {
    return tGamePresents && !gVulkanGame.load(std::memory_order_relaxed);
}

void Enable(GLenum cap) {
    gEnable.load(std::memory_order_relaxed)(cap);
    if (cap != GL_DEPTH_TEST || !glShades()) return;
    if (const ShadingRateFn set = glShadingRate()) set(glRateOf(gRate.load(std::memory_order_relaxed)));
}

void Disable(GLenum cap) {
    gDisable.load(std::memory_order_relaxed)(cap);
    if (cap != GL_DEPTH_TEST || !glShades()) return;
    if (const ShadingRateFn set = glShadingRate()) set(GL_SHADING_RATE_1X1_PIXELS_QCOM);
}

}  // namespace

VkExtent2D sizeOf(int rate) {
    switch (rate) {
        case 1: return {2, 1};
        case 2: return {2, 2};
        case 3: return {4, 4};
        default: return {1, 1};
    }
}

void onDevice(const framefx_vk_instance& in, VkPhysicalDevice physical, VkDevice device,
              bool on) {
    gVulkanGame.store(true, std::memory_order_relaxed);
    if (on && in.get_device_proc_addr != nullptr) {
        gSetRate.store(reinterpret_cast<PFN_vkCmdSetFragmentShadingRateKHR>(
                in.get_device_proc_addr(device, "vkCmdSetFragmentShadingRateKHR")));
        on = gSetRate.load() != nullptr;
    }
    if (on && in.get_properties2 != nullptr) {
        VkPhysicalDeviceFragmentShadingRatePropertiesKHR fsr = {};
        fsr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props = {};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &fsr;
        in.get_properties2(physical, &props);
        ALOGI("shading: device %p, fragments up to %ux%u", device, fsr.maxFragmentSize.width,
              fsr.maxFragmentSize.height);
    }
    std::lock_guard lock(gPassLock);
    gDevices[device] = on;
}

void onDestroyDevice(VkDevice device) {
    std::lock_guard lock(gPassLock);
    gDevices.erase(device);
}

PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next, bool shadingOn) {
    if (next == nullptr || !shadingOn) return nullptr;
    if (strcmp(name, "vkCreateRenderPass") == 0) {
        return wrap(&gCreateRenderPass, next, &CreateRenderPass);
    }
    if (strcmp(name, "vkCreateRenderPass2") == 0 || strcmp(name, "vkCreateRenderPass2KHR") == 0) {
        return wrap(&gCreateRenderPass2, next, &CreateRenderPass2);
    }
    if (strcmp(name, "vkDestroyRenderPass") == 0) {
        return wrap(&gDestroyRenderPass, next, &DestroyRenderPass);
    }
    if (strcmp(name, "vkCreateGraphicsPipelines") == 0) {
        return wrap(&gCreateGraphics, next, &CreateGraphicsPipelines);
    }
    if (strcmp(name, "vkDestroyPipeline") == 0) {
        return wrap(&gDestroyPipeline, next, &DestroyPipeline);
    }
    if (strcmp(name, "vkCmdBindPipeline") == 0) {
        return wrap(&gBindPipeline, next, &CmdBindPipeline);
    }
    if (strcmp(name, "vkCmdSetFragmentShadingRateKHR") == 0) {
        return wrap(&gGameSetRate, next, &CmdSetFragmentShadingRate);
    }
    return nullptr;
}

void onPresent(const TextureSettings& s, int64_t nowNs, int64_t refreshNs) {
    int next = s.shadingRate;
    std::lock_guard lock(gGovernorLock);
    if (s.shadingRate != 0 && s.shadingAdaptive) {
        static const bool fits = framefx_lapis_shading_governor_size() <= sizeof(gGovernor.state);
        next = fits ? framefx_lapis_shading_frame(gGovernor.state, nowNs, refreshNs,
                                                  static_cast<uint8_t>(s.shadingRate))
                    : s.shadingRate;
    }
    gRate.store(next, std::memory_order_relaxed);
    if (next != gLogged) {
        const VkExtent2D size = sizeOf(next);
        ALOGI("shading: %ux%u%s", size.width, size.height,
              s.shadingAdaptive ? " (adaptive)" : "");
        gLogged = next;
    }
}

int rate() {
    return gRate.load(std::memory_order_relaxed);
}

void onGlPresent() {
    tGamePresents = true;
}

void attachGl(const framefx_gl_host* host) {
    gGlHost = *host;
}

void* interceptGl(const char* name, void* next, bool shadingOn) {
    if (next == nullptr || !shadingOn) return nullptr;
    if (strcmp(name, "glEnable") == 0) {
        gEnable.store(reinterpret_cast<EnableFn>(next));
        return reinterpret_cast<void*>(&Enable);
    }
    if (strcmp(name, "glDisable") == 0) {
        gDisable.store(reinterpret_cast<EnableFn>(next));
        return reinterpret_cast<void*>(&Disable);
    }
    return nullptr;
}

}  // namespace framefx::shading
