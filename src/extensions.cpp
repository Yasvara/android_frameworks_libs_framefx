/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "src/extensions.h"

#include <EGL/eglext.h>
#include <android-base/properties.h>
#include <log/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace framefx::extensions {
namespace {

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

const VkBaseInStructure* find(const void* chain, VkStructureType type) {
    for (auto* p = static_cast<const VkBaseInStructure*>(chain); p != nullptr; p = p->pNext) {
        if (p->sType == type) return p;
    }
    return nullptr;
}

const char* priorityName(VkQueueGlobalPriorityKHR p) {
    switch (p) {
        case VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR: return "low";
        case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR: return "medium";
        case VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR: return "high";
        case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR: return "realtime";
        default: return "default";
    }
}

struct DeviceState {
    Granted granted;
    PFN_vkGetDeviceFaultInfoEXT getFaultInfo = nullptr;
    bool faultLogged = false;
};

std::mutex gLock;
std::unordered_map<VkDevice, DeviceState> gDevices;
std::unordered_map<VkQueue, VkDevice> gQueues;

bool granted(VkDevice device, bool Granted::*what) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    return it != gDevices.end() && it->second.granted.*what;
}

// --- Lost device ------------------------------------------------------------------------------

const char* addressTypeName(VkDeviceFaultAddressTypeEXT t) {
    switch (t) {
        case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "read invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "write invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "execute invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "ip unknown";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "ip invalid";
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "ip fault";
        default: return "none";
    }
}

void logFault(VkDevice device, PFN_vkGetDeviceFaultInfoEXT get) {
    VkDeviceFaultCountsEXT counts = {};
    counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    if (get(device, &counts, nullptr) != VK_SUCCESS) {
        ALOGE("device lost on %p; the driver gave no fault report", device);
        return;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
    VkDeviceFaultInfoEXT info = {};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    info.pAddressInfos = addresses.data();
    info.pVendorInfos = vendor.data();
    const size_t binary = counts.vendorBinarySize;
    counts.vendorBinarySize = 0;
    const VkResult got = get(device, &counts, &info);
    if (got != VK_SUCCESS && got != VK_INCOMPLETE) {
        ALOGE("device lost on %p; fault report unreadable (%d)", device, got);
        return;
    }
    ALOGE("device lost on %p: %s (%u addresses, %u vendor records, %zu bytes vendor binary)",
          device, info.description, counts.addressInfoCount, counts.vendorInfoCount, binary);
    for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
        ALOGE("  address %s 0x%llx ±0x%llx", addressTypeName(addresses[i].addressType),
              static_cast<unsigned long long>(addresses[i].reportedAddress),
              static_cast<unsigned long long>(addresses[i].addressPrecision));
    }
    for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
        ALOGE("  vendor %s code 0x%llx data 0x%llx", vendor[i].description,
              static_cast<unsigned long long>(vendor[i].vendorFaultCode),
              static_cast<unsigned long long>(vendor[i].vendorFaultData));
    }
}

// --- ASTC decode precision --------------------------------------------------------------------

bool astcUnorm(VkFormat f) {
    switch (f) {
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
            return true;
        default:
            // sRGB views already decode to 8 bits; HDR (SFLOAT) ones need the float range.
            return false;
    }
}

std::atomic<PFN_vkCreateImageView> gCreateImageView{nullptr};
std::atomic<uint32_t> gAstcViews{0};

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice device, const VkImageViewCreateInfo* info,
                                               const VkAllocationCallbacks* allocator,
                                               VkImageView* view) {
    const PFN_vkCreateImageView next = gCreateImageView.load(std::memory_order_relaxed);
    if (!astcUnorm(info->format) ||
        find(info->pNext, VK_STRUCTURE_TYPE_IMAGE_VIEW_ASTC_DECODE_MODE_EXT) != nullptr ||
        !granted(device, &Granted::astcDecode)) {
        return next(device, info, allocator, view);
    }
    VkImageViewASTCDecodeModeEXT mode = {};
    mode.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_ASTC_DECODE_MODE_EXT;
    mode.pNext = info->pNext;
    mode.decodeMode = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageViewCreateInfo changed = *info;
    changed.pNext = &mode;
    const uint32_t n = gAstcViews.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & (n - 1)) == 0) ALOGI("astc: %u views decode to RGBA8", n);
    return next(device, &changed, allocator, view);
}

// --- Pipeline compile times -------------------------------------------------------------------

struct PipelineStats {
    std::atomic<uint32_t> created{0};
    std::atomic<uint32_t> cacheHits{0};
    std::atomic<uint32_t> slow{0};  // over kSlowNs
    std::atomic<int64_t> totalNs{0};
    std::atomic<int64_t> longestNs{0};
    std::atomic<int64_t> windowStart{0};
};
PipelineStats gPipelines;
constexpr int64_t kSlowNs = 4'000'000;  // a quarter of a 60 Hz frame
constexpr int64_t kWindowNs = 10'000'000'000;

void recordPipelines(const VkPipelineCreationFeedback* feedback, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if ((feedback[i].flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) == 0) continue;
        const auto ns = static_cast<int64_t>(feedback[i].duration);
        gPipelines.created.fetch_add(1, std::memory_order_relaxed);
        if (feedback[i].flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) {
            gPipelines.cacheHits.fetch_add(1, std::memory_order_relaxed);
        }
        if (ns >= kSlowNs) gPipelines.slow.fetch_add(1, std::memory_order_relaxed);
        gPipelines.totalNs.fetch_add(ns, std::memory_order_relaxed);
        int64_t longest = gPipelines.longestNs.load(std::memory_order_relaxed);
        while (ns > longest &&
               !gPipelines.longestNs.compare_exchange_weak(longest, ns, std::memory_order_relaxed)) {
        }
    }
    const int64_t now = nowNs();
    int64_t start = gPipelines.windowStart.load(std::memory_order_relaxed);
    if (start == 0) {
        gPipelines.windowStart.compare_exchange_strong(start, now, std::memory_order_relaxed);
        return;
    }
    if (now - start < kWindowNs ||
        !gPipelines.windowStart.compare_exchange_strong(start, now, std::memory_order_relaxed)) {
        return;
    }
    const uint32_t created = gPipelines.created.exchange(0, std::memory_order_relaxed);
    const uint32_t hits = gPipelines.cacheHits.exchange(0, std::memory_order_relaxed);
    const uint32_t slow = gPipelines.slow.exchange(0, std::memory_order_relaxed);
    const int64_t total = gPipelines.totalNs.exchange(0, std::memory_order_relaxed);
    const int64_t longest = gPipelines.longestNs.exchange(0, std::memory_order_relaxed);
    ALOGI("pipelines: %u made in %.1f s (%u from the app's cache), %.1f ms compiling, longest "
          "%.1f ms, %u over %.0f ms",
          created, static_cast<double>(now - start) / 1e9, hits, static_cast<double>(total) / 1e6,
          static_cast<double>(longest) / 1e6, slow, static_cast<double>(kSlowNs) / 1e6);
}

// The game's create infos with a pipeline-level feedback struct added where it has none.
template <typename Info>
struct WithFeedback {
    std::vector<Info> infos;
    std::vector<VkPipelineCreationFeedbackCreateInfo> chains;
    std::vector<VkPipelineCreationFeedback> feedback;

    const Info* add(const Info* asked, uint32_t n) {
        infos.assign(asked, asked + n);
        chains.assign(n, {});
        feedback.assign(n, {});
        for (uint32_t i = 0; i < n; i++) {
            if (find(asked[i].pNext, VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO) !=
                nullptr) {
                continue;
            }
            chains[i].sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
            chains[i].pNext = asked[i].pNext;
            chains[i].pPipelineCreationFeedback = &feedback[i];
            infos[i].pNext = &chains[i];
        }
        return infos.data();
    }
};

std::atomic<PFN_vkCreateGraphicsPipelines> gCreateGraphics{nullptr};
std::atomic<PFN_vkCreateComputePipelines> gCreateCompute{nullptr};

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache,
                                                       uint32_t n,
                                                       const VkGraphicsPipelineCreateInfo* infos,
                                                       const VkAllocationCallbacks* allocator,
                                                       VkPipeline* pipelines) {
    const PFN_vkCreateGraphicsPipelines next = gCreateGraphics.load(std::memory_order_relaxed);
    if (!granted(device, &Granted::pipelineFeedback)) {
        return next(device, cache, n, infos, allocator, pipelines);
    }
    WithFeedback<VkGraphicsPipelineCreateInfo> with;
    const VkResult result = next(device, cache, n, with.add(infos, n), allocator, pipelines);
    recordPipelines(with.feedback.data(), n);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice device, VkPipelineCache cache,
                                                      uint32_t n,
                                                      const VkComputePipelineCreateInfo* infos,
                                                      const VkAllocationCallbacks* allocator,
                                                      VkPipeline* pipelines) {
    const PFN_vkCreateComputePipelines next = gCreateCompute.load(std::memory_order_relaxed);
    if (!granted(device, &Granted::pipelineFeedback)) {
        return next(device, cache, n, infos, allocator, pipelines);
    }
    WithFeedback<VkComputePipelineCreateInfo> with;
    const VkResult result = next(device, cache, n, with.add(infos, n), allocator, pipelines);
    recordPipelines(with.feedback.data(), n);
    return result;
}

// One slot per command for every device, as Lapis keeps its own: a second device that brings
// another function keeps it unwrapped.
template <typename Fn>
PFN_vkVoidFunction wrap(std::atomic<Fn>* slot, PFN_vkVoidFunction next, Fn own) {
    Fn none = nullptr;
    const auto want = reinterpret_cast<Fn>(next);
    if (!slot->compare_exchange_strong(none, want) && none != want) return nullptr;
    return reinterpret_cast<PFN_vkVoidFunction>(own);
}

}  // namespace

const Settings& settings() {
    static const Settings s = [] {
        Settings out;
        const std::string priority = android::base::GetProperty("debug.framefx.vk_priority", "");
        if (priority == "high") out.queuePriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
        if (priority == "realtime") out.queuePriority = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR;
        out.deviceFault = android::base::GetProperty("debug.framefx.device_fault", "") != "off";
        out.astcRgba8 = android::base::GetProperty("debug.framefx.astc_decode", "") == "rgba8";
        out.pipelineLog = android::base::GetBoolProperty("debug.framefx.pipeline_log", false);
        const std::string gl = android::base::GetProperty("debug.framefx.gl_priority", "");
        if (gl == "high") out.glPriority = EGL_CONTEXT_PRIORITY_HIGH_IMG;
        if (gl == "realtime") out.glPriority = EGL_CONTEXT_PRIORITY_REALTIME_NV;
        out.glNoError = android::base::GetBoolProperty("debug.framefx.gl_no_error", false);
        out.rays = android::base::GetBoolProperty("debug.framefx.rays", false);
        return out;
    }();
    return s;
}

DeviceRequest::DeviceRequest(const framefx_vk_instance& in, VkPhysicalDevice physical,
                             const VkDeviceCreateInfo& asked, bool shading, bool rays)
    : asked_(&asked),
      info_(asked),
      fault_{},
      shading_{},
      structures_{},
      rayQuery_{},
      address_{},
      feedback_{},
      fetch_{} {
    const Settings& want = settings();
    rays = rays || want.rays;
    if (want.queuePriority == 0 && !want.deviceFault && !want.astcRgba8 && !want.pipelineLog &&
        !shading && !rays) {
        return;
    }
    if (in.enumerate_extensions == nullptr) return;
    uint32_t count = 0;
    in.enumerate_extensions(physical, nullptr, &count, nullptr);
    supported_.resize(count);
    in.enumerate_extensions(physical, nullptr, &count, supported_.data());
    supported_.resize(count);
    extensions_.assign(asked.ppEnabledExtensionNames,
                       asked.ppEnabledExtensionNames + asked.enabledExtensionCount);

    // Every queue the game makes, at the asked level, unless it set one itself. The driver may
    // refuse a level above what the process is allowed: the whole request then falls back.
    if (want.queuePriority != 0) {
        const char* ext = nullptr;
        for (const char* name :
             {VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME}) {
            if (listed(name)) {
                ext = name;
                break;
            }
            if (ext == nullptr && supports(name)) ext = name;
        }
        if (ext != nullptr) {
            queues_.assign(asked.pQueueCreateInfos,
                           asked.pQueueCreateInfos + asked.queueCreateInfoCount);
            priorities_.resize(queues_.size());
            bool any = false;
            for (size_t i = 0; i < queues_.size(); i++) {
                if (find(queues_[i].pNext,
                         VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR) != nullptr) {
                    continue;
                }
                priorities_[i] = {};
                priorities_[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
                priorities_[i].pNext = queues_[i].pNext;
                priorities_[i].globalPriority = want.queuePriority;
                queues_[i].pNext = &priorities_[i];
                any = true;
            }
            if (any) {
                enable(ext);
                info_.pQueueCreateInfos = queues_.data();
                granted_.queuePriority = want.queuePriority;
            }
        }
    }

    // A game that turned the fault report on itself shares it; one that lists the extension
    // without the feature gets the feature, unless it chained it off.
    const auto* ownFault = reinterpret_cast<const VkPhysicalDeviceFaultFeaturesEXT*>(
            find(asked.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT));
    if (want.deviceFault && ownFault != nullptr) {
        granted_.deviceFault = ownFault->deviceFault;
    } else if (want.deviceFault && supports(VK_EXT_DEVICE_FAULT_EXTENSION_NAME) &&
               in.get_features2 != nullptr) {
        VkPhysicalDeviceFaultFeaturesEXT has = {};
        has.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 features = {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &has;
        in.get_features2(physical, &features);
        if (has.deviceFault) {
            fault_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
            fault_.pNext = const_cast<void*>(info_.pNext);
            fault_.deviceFault = VK_TRUE;
            info_.pNext = &fault_;
            enable(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
            granted_.deviceFault = true;
        }
    }

    // Pipeline shading rate, for the depth-tested pipelines src/shading.cpp makes dynamic. A game
    // that chained the feature itself keeps its own choice.
    const auto* ownShading = reinterpret_cast<const VkPhysicalDeviceFragmentShadingRateFeaturesKHR*>(
            find(asked.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR));
    if (shading && ownShading != nullptr) {
        granted_.shading = ownShading->pipelineFragmentShadingRate;
    } else if (shading && supports(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME) &&
               in.get_features2 != nullptr) {
        VkPhysicalDeviceFragmentShadingRateFeaturesKHR has = {};
        has.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
        VkPhysicalDeviceFeatures2 features = {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &has;
        in.get_features2(physical, &features);
        if (has.pipelineFragmentShadingRate) {
            shading_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
            shading_.pNext = const_cast<void*>(info_.pNext);
            shading_.pipelineFragmentShadingRate = VK_TRUE;
            info_.pNext = &shading_;
            enable(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
            // Its dependency, core from 1.2 but not for a game that asked for less.
            if (supports(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME)) {
                enable(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
            }
            granted_.shading = true;
        }
    }

    // Ray query over acceleration structures built from device addresses. A game that chains the
    // Vulkan 1.2 features or the address feature itself keeps its device as asked: the separate
    // struct may not sit beside the 1.2 one.
    if (rays && supports(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        supports(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
        supports(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) && in.get_features2 != nullptr &&
        find(asked.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR) ==
                nullptr &&
        find(asked.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR) == nullptr) {
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as = {};
        as.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        VkPhysicalDeviceRayQueryFeaturesKHR rq = {};
        rq.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        rq.pNext = &as;
        VkPhysicalDeviceBufferDeviceAddressFeatures bda = {};
        bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bda.pNext = &rq;
        VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb = {};
        xfb.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT;
        as.pNext = &xfb;
        VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetch = {};
        fetch.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR;
        xfb.pNext = &fetch;
        VkPhysicalDeviceFeatures2 features = {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &bda;
        in.get_features2(physical, &features);
        const auto* own12 = reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(
                find(asked.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES));
        const auto* ownAddress = find(asked.pNext,
                                      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES);
        if (as.accelerationStructure && rq.rayQuery && bda.bufferDeviceAddress && own12 == nullptr &&
            ownAddress == nullptr) {
            address_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
            address_.bufferDeviceAddress = VK_TRUE;
            address_.pNext = const_cast<void*>(info_.pNext);
            rayQuery_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            rayQuery_.rayQuery = VK_TRUE;
            rayQuery_.pNext = &address_;
            structures_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            structures_.accelerationStructure = VK_TRUE;
            structures_.pNext = &rayQuery_;
            info_.pNext = &structures_;
            // The game's own positions, captured from its vertex shaders for the rays to hit.
            if (xfb.transformFeedback && fetch.rayTracingPositionFetch &&
                supports(VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME) &&
                supports(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME) &&
                find(asked.pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT) ==
                        nullptr) {
                feedback_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT;
                feedback_.transformFeedback = VK_TRUE;
                feedback_.pNext = const_cast<void*>(info_.pNext);
                fetch_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR;
                fetch_.rayTracingPositionFetch = VK_TRUE;
                fetch_.pNext = &feedback_;
                info_.pNext = &fetch_;
                enable(VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME);
                enable(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME);
                granted_.capture = true;
            }
            for (const char* name :
                 {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                  VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                  VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_SPIRV_1_4_EXTENSION_NAME,
                  VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME}) {
                if (supports(name)) enable(name);
            }
            granted_.rays = true;
        } else if (own12 != nullptr || ownAddress != nullptr) {
            ALOGI("extensions: the game chains its own 1.2 or address features; no ray query");
        }
    }

    auto offer = [this](const char* name) {
        if (listed(name)) return true;
        if (!supports(name)) return false;
        enable(name);
        return true;
    };
    if (want.astcRgba8) granted_.astcDecode = offer(VK_EXT_ASTC_DECODE_MODE_EXTENSION_NAME);
    if (want.pipelineLog) {
        granted_.pipelineFeedback = offer(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
    }

    info_.enabledExtensionCount = static_cast<uint32_t>(extensions_.size());
    info_.ppEnabledExtensionNames = extensions_.data();
}

bool DeviceRequest::supports(const char* name) const {
    return std::any_of(supported_.begin(), supported_.end(),
                       [name](const auto& e) { return strcmp(e.extensionName, name) == 0; });
}

bool DeviceRequest::listed(const char* name) const {
    for (uint32_t i = 0; i < asked_->enabledExtensionCount; i++) {
        if (strcmp(asked_->ppEnabledExtensionNames[i], name) == 0) return true;
    }
    for (const char* e : extensions_) {
        if (strcmp(e, name) == 0) return true;
    }
    return false;
}

void DeviceRequest::enable(const char* name) {
    if (!listed(name)) extensions_.push_back(name);
}

void onDevice(const framefx_vk_instance& in, VkDevice device, const Granted& g) {
    DeviceState state;
    state.granted = g;
    if (g.deviceFault && in.get_device_proc_addr != nullptr) {
        state.getFaultInfo = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
                in.get_device_proc_addr(device, "vkGetDeviceFaultInfoEXT"));
    }
    if (g.any()) {
        ALOGI("extensions: device %p with queue priority %s, fault report %s, ASTC RGBA8 %s, "
              "pipeline log %s, shading rate %s, ray query %s, geometry capture %s",
              device, priorityName(g.queuePriority), state.getFaultInfo ? "on" : "off",
              g.astcDecode ? "on" : "off", g.pipelineFeedback ? "on" : "off",
              g.shading ? "on" : "off", g.rays ? "on" : "off", g.capture ? "on" : "off");
    }
    std::lock_guard lock(gLock);
    gDevices[device] = state;
}

void onDestroyDevice(VkDevice device) {
    std::lock_guard lock(gLock);
    gDevices.erase(device);
    for (auto it = gQueues.begin(); it != gQueues.end();) {
        it = it->second == device ? gQueues.erase(it) : std::next(it);
    }
}

void onQueue(VkDevice device, VkQueue queue) {
    if (queue == VK_NULL_HANDLE) return;
    std::lock_guard lock(gLock);
    gQueues[queue] = device;
}

void onResult(VkQueue queue, VkResult result) {
    if (result != VK_ERROR_DEVICE_LOST) return;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceFaultInfoEXT get = nullptr;
    {
        std::lock_guard lock(gLock);
        const auto q = gQueues.find(queue);
        if (q == gQueues.end()) return;
        device = q->second;
        const auto d = gDevices.find(device);
        if (d == gDevices.end() || d->second.faultLogged) return;
        d->second.faultLogged = true;
        get = d->second.getFaultInfo;
    }
    if (get == nullptr) {
        ALOGE("device lost on %p (no fault report on this device)", device);
        return;
    }
    logFault(device, get);
}

PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next) {
    if (next == nullptr) return nullptr;
    const Settings& s = settings();
    if (s.astcRgba8 && strcmp(name, "vkCreateImageView") == 0) {
        return wrap(&gCreateImageView, next, &CreateImageView);
    }
    if (s.pipelineLog && strcmp(name, "vkCreateGraphicsPipelines") == 0) {
        return wrap(&gCreateGraphics, next, &CreateGraphicsPipelines);
    }
    if (s.pipelineLog && strcmp(name, "vkCreateComputePipelines") == 0) {
        return wrap(&gCreateCompute, next, &CreateComputePipelines);
    }
    return nullptr;
}

}  // namespace framefx::extensions
