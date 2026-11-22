/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "luwes/afme.h"

#include <dlfcn.h>
#include <link.h>
#include <log/log.h>

#include <cstring>
#include <mutex>
#include <vector>

namespace framefx::luwes::afme {
namespace {

constexpr char kLibrary[] = "libframeestimationVK.so";

bool HasExtension(const std::vector<VkExtensionProperties>& all, const char* name) {
    for (const VkExtensionProperties& e : all) {
        if (strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

}  // namespace

bool BehindAngle() {
    return dl_iterate_phdr(
                   [](dl_phdr_info* info, size_t, void*) {
                       return info->dlpi_name != nullptr &&
                                       strstr(info->dlpi_name, "libGLESv2_angle") != nullptr
                               ? 1
                               : 0;
                   },
                   nullptr) != 0;
}

const Api* Load() {
    static std::once_flag once;
    static Api api;
    static bool loaded = false;
    std::call_once(once, [] {
        void* lib = dlopen(kLibrary, RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            ALOGW("luwes: AFME not loaded: %s", dlerror());
            return;
        }
        api.isSupported = reinterpret_cast<decltype(api.isSupported)>(
                dlsym(lib, "AFME_VK_IsFeatureSupported"));
        api.init = reinterpret_cast<decltype(api.init)>(dlsym(lib, "AFME_VK_Init"));
        api.update = reinterpret_cast<decltype(api.update)>(dlsym(lib, "AFME_VK_Update"));
        api.estimate = reinterpret_cast<decltype(api.estimate)>(dlsym(lib, "AFME_VK_Estimate"));
        api.destroy = reinterpret_cast<decltype(api.destroy)>(dlsym(lib, "AFME_VK_Destroy"));
        loaded = api.isSupported && api.init && api.update && api.estimate && api.destroy;
        if (!loaded) ALOGW("luwes: AFME library lacks its entry points");
    });
    return loaded ? &api : nullptr;
}

bool Extend(const framefx_vk_instance& in, VkPhysicalDevice physical,
            const VkDeviceCreateInfo& asked, DeviceInfo* out) {
    if (in.enumerate_extensions == nullptr || in.get_features2 == nullptr) return false;
    uint32_t count = 0;
    in.enumerate_extensions(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> all(count);
    in.enumerate_extensions(physical, nullptr, &count, all.data());
    if (!HasExtension(all, VK_NV_OPTICAL_FLOW_EXTENSION_NAME)) return false;
    VkPhysicalDeviceOpticalFlowFeaturesNV supported = {};
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &supported;
    in.get_features2(physical, &features);
    if (!supported.opticalFlow) return false;

    // A game that already enables the feature keeps its own structure.
    bool chained = false;
    for (auto* p = static_cast<const VkBaseInStructure*>(asked.pNext); p != nullptr; p = p->pNext) {
        if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV) {
            if (!reinterpret_cast<const VkPhysicalDeviceOpticalFlowFeaturesNV*>(p)->opticalFlow) {
                return false;
            }
            chained = true;
        }
    }
    out->info = asked;
    out->extensions.assign(asked.ppEnabledExtensionNames,
                           asked.ppEnabledExtensionNames + asked.enabledExtensionCount);
    for (const char* name :
         {VK_NV_OPTICAL_FLOW_EXTENSION_NAME, VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,
          VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME}) {
        bool listed = false;
        for (const char* e : out->extensions) listed = listed || strcmp(e, name) == 0;
        if (!listed && HasExtension(all, name)) out->extensions.push_back(name);
    }
    out->info.enabledExtensionCount = static_cast<uint32_t>(out->extensions.size());
    out->info.ppEnabledExtensionNames = out->extensions.data();
    if (!chained) {
        out->opticalFlow = {};
        out->opticalFlow.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
        out->opticalFlow.opticalFlow = VK_TRUE;
        out->opticalFlow.pNext = const_cast<void*>(asked.pNext);
        out->info.pNext = &out->opticalFlow;
    }
    return true;
}

}  // namespace framefx::luwes::afme
