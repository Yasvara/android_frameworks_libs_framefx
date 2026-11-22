/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "src/vulkan.h"

#include <log/log.h>
#include <utils/Timers.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "kelir/plan.h"
#include "kelir/vulkan.h"
#include "lapis/recorder.h"
#include "luwes/vulkan.h"
#include "src/engine.h"
#include "src/extensions.h"
#include "src/rays.h"
#include "src/safety.h"
#include "src/shading.h"
#include "src/survey.h"
#include "src/texture.h"
#include "video/gpu.h"

namespace framefx::vulkan {
namespace {

framefx_vk_host gHost = {};

// Kelir's stack at the game's start holds a ray-traced kind: the device needs ray query.
bool startsWithRaysNow() {
    const Config config = Engine::get().config();
    const auto& plan = config.kelir;
    if (plan == nullptr) return false;
    for (uint8_t kind : {kelir::kRtShadows, kelir::kRtReflections, kelir::kPathTracedLight}) {
        if (plan->sceneStep(kind) != nullptr) return true;
    }
    return false;
}

// Asked once: the device and every wrapper agree for the process's life.
bool startsWithRays() {
    static const bool on = [] {
        Engine::get().attach();
        return startsWithRaysNow() && safety::allowed(safety::kRays);
    }();
    return on;
}

// What Pentas asked of the game's textures as the game started: samplers are made at load, so a
// change waits for the next start.
const TextureSettings& textureSettings() {
    static const TextureSettings settings = [] {
        Engine& engine = Engine::get();
        engine.attach();
        return engine.config().texture;
    }();
    return settings;
}

std::mutex gTextureLock;
std::unordered_map<VkDevice, texture::Limits> gTextureLimits;
PFN_vkCreateSampler gCreateSampler = nullptr;

// `asked` with samplerAnisotropy on, where the GPU has it and the game did not ask for it. A game
// that lists its features in a VkPhysicalDeviceFeatures2 keeps them; it gets no anisotropy unless
// it asked.
struct Features {
    VkDeviceCreateInfo info;
    VkPhysicalDeviceFeatures features;
};
const VkDeviceCreateInfo* withAnisotropy(const framefx_vk_instance* in, VkPhysicalDevice physical,
                                         const VkDeviceCreateInfo* asked, Features* out,
                                         texture::Limits* limits) {
    if (in->get_features2 == nullptr || in->get_properties2 == nullptr) return asked;
    VkPhysicalDeviceFeatures2 supported = {};
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    in->get_features2(physical, &supported);
    VkPhysicalDeviceProperties2 props = {};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    in->get_properties2(physical, &props);
    limits->maxAnisotropy = props.properties.limits.maxSamplerAnisotropy;
    limits->maxLodBias = props.properties.limits.maxSamplerLodBias;
    if (asked->pEnabledFeatures != nullptr) {
        limits->anisotropy = asked->pEnabledFeatures->samplerAnisotropy || supported.features.samplerAnisotropy;
        if (asked->pEnabledFeatures->samplerAnisotropy || !supported.features.samplerAnisotropy) {
            return asked;
        }
        out->features = *asked->pEnabledFeatures;
    } else {
        for (auto* p = static_cast<const VkBaseInStructure*>(asked->pNext); p != nullptr;
             p = p->pNext) {
            if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
                limits->anisotropy = reinterpret_cast<const VkPhysicalDeviceFeatures2*>(p)
                                             ->features.samplerAnisotropy;
                return asked;
            }
        }
        if (!supported.features.samplerAnisotropy) return asked;
        limits->anisotropy = true;
        out->features = {};
    }
    out->features.samplerAnisotropy = VK_TRUE;
    out->info = *asked;
    out->info.pEnabledFeatures = &out->features;
    return &out->info;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(VkDevice device, const VkSamplerCreateInfo* info,
                                             const VkAllocationCallbacks* allocator,
                                             VkSampler* sampler) {
    texture::Limits limits;
    bool known = false;
    {
        std::lock_guard lock(gTextureLock);
        const auto it = gTextureLimits.find(device);
        if (it != gTextureLimits.end()) {
            limits = it->second;
            known = true;
        }
    }
    VkSamplerCreateInfo changed = *info;
    if (known && texture::Override(textureSettings(), limits, &changed)) {
        return gCreateSampler(device, &changed, allocator, sampler);
    }
    return gCreateSampler(device, info, allocator, sampler);
}

}  // namespace

void attach(const framefx_vk_host* host) {
    gHost = *host;
    luwes::vulkan::attach(host);
    kelir::vulkan::attach(host);
    lapis::attach(host);
}

VkResult createDevice(const framefx_vk_instance* instance, VkPhysicalDevice physical,
                      const VkDeviceCreateInfo* info, const VkAllocationCallbacks* allocator,
                      VkDevice* device) {
    // The video path's own device, in a process also armed for its presents.
    if (video::creatingDevice()) return instance->create_device(physical, info, allocator, device);
    const TextureSettings& texture = textureSettings();
    Features features;
    texture::Limits limits;
    const VkDeviceCreateInfo* create =
            texture.on() ? withAnisotropy(instance, physical, info, &features, &limits) : info;
    // Driver extensions framefx adds on top; a driver that refuses them gets the request without.
    // A feature that took the game down last start sits this one out (src/safety.h).
    const bool shadingOn = texture.shading() && safety::allowed(safety::kShadingRate);
    const bool raysOn = startsWithRays();
    extensions::DeviceRequest extras(*instance, physical, *create, shadingOn, raysOn);
    extensions::Granted granted = extras.granted();
    VkResult result =
            luwes::vulkan::createDevice(instance, physical, extras.info(), allocator, device);
    if (result != VK_SUCCESS && granted.any()) {
        ALOGW("extensions: device with framefx's extras refused (%d); without them", result);
        granted = {};
        result = luwes::vulkan::createDevice(instance, physical, create, allocator, device);
    }
    if (result != VK_SUCCESS && create != info) {
        ALOGW("texture: device with anisotropy refused (%d); without it", result);
        limits.anisotropy = false;
        result = luwes::vulkan::createDevice(instance, physical, info, allocator, device);
    }
    if (result == VK_SUCCESS) {
        extensions::onDevice(*instance, *device, granted);
        shading::onDevice(*instance, physical, *device, granted.shading);
        rays::onDevice(*instance, physical, *device, granted.rays && granted.capture);
        kelir::vulkan::onDevice(instance, physical, *device);
        lapis::onDevice(instance, physical, *device);
        if (texture.on()) {
            ALOGI("texture: AF %dx%s, LOD bias %+.3f for device %p", texture.anisotropy,
                  texture.anisotropy > 1 && !limits.anisotropy ? " (not on this device)" : "",
                  texture.lodBiasEighths / 8.0, *device);
            std::lock_guard lock(gTextureLock);
            gTextureLimits[*device] = limits;
        }
    }
    return result;
}

void destroyDevice(VkDevice device) {
    {
        std::lock_guard lock(gTextureLock);
        gTextureLimits.erase(device);
    }
    lapis::onDestroyDevice(device);
    extensions::onDestroyDevice(device);
    rays::onDestroyDevice(device);
    shading::onDestroyDevice(device);
    kelir::vulkan::destroyDevice(device);
    luwes::vulkan::destroyDevice(device);
}

void getDeviceQueue(VkDevice device, uint32_t family, VkQueue queue) {
    extensions::onQueue(device, queue);
    rays::onQueue(device, queue);
    luwes::vulkan::getDeviceQueue(device, family, queue);
    kelir::vulkan::getDeviceQueue(device, family, queue);
}

VkResult createSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                         const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain) {
    // Luwes keeps at least one image beyond the game's whenever it takes a swapchain; Kelir's
    // output shares it, and brings its own only where Luwes does not.
    VkSwapchainCreateInfoKHR asked = *info;
    const bool luwes = luwes::vulkan::accepts(device, info);
    const bool kelir = kelir::vulkan::adjustSwapchain(device, &asked, luwes);
    // Both views every image up front; deferred memory only binds an image at its first acquire.
    if (luwes || kelir) asked.flags &= ~VK_SWAPCHAIN_CREATE_DEFERRED_MEMORY_ALLOCATION_BIT_EXT;
    // Lapis may copy the scene out of an image the game tonemaps straight into.
    if (lapis::recording()) asked.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkResult result = luwes::vulkan::createSwapchain(device, &asked, allocator, swapchain);
    if (result == VK_SUCCESS) {
        // A swapchain Kelir cannot take is still tracked, so its status says why. Where the game
        // draws into images of its own, Kelir works at the size shown, inside Luwes's passes, and
        // Lapis follows the game's images.
        std::vector<VkImage> images;
        VkSwapchainCreateInfoKHR shown = *info;
        const bool resized =
                luwes::vulkan::gameImages(device, *swapchain, &images, &shown.imageExtent);
        const bool spare = kelir && (!luwes || luwes::vulkan::spareImages(device, *swapchain) > 0);
        kelir::vulkan::onSwapchain(device, *swapchain, &shown, spare);
        lapis::onSwapchain(device, *swapchain, &asked, resized ? &images : nullptr);
    }
    return result;
}

void destroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
                      const VkAllocationCallbacks* allocator) {
    lapis::onDestroySwapchain(swapchain);
    kelir::vulkan::destroySwapchain(device, swapchain);
    luwes::vulkan::destroySwapchain(device, swapchain, allocator);
}

namespace {

// The shading rate follows Pentas live, in a game that started with it on.
void paceShading(const VkPresentInfoKHR* info) {
    if (!textureSettings().shading()) return;
    thread_local Engine::Watch watch;
    watch.refresh(Engine::get());
    const int64_t refresh =
            gHost.refresh_ns != nullptr ? gHost.refresh_ns(info->pSwapchains[0]) : 0;
    shading::onPresent(watch.config().texture, systemTime(SYSTEM_TIME_MONOTONIC), refresh);
}

}  // namespace

VkResult queuePresent(VkQueue queue, const VkPresentInfoKHR* info) {
    Engine::get().status().countPresent();
    safety::onPresent();
    paceShading(info);
    lapis::onPresent(queue, info);
    rays::onPresent(queue);
    kelir::vulkan::Claim claim;
    thread_local std::vector<VkImage> images;
    VkExtent2D shown;
    // Luwes releases what it held before Kelir takes over its swapchain.
    if (kelir::vulkan::claim(queue, info, &claim) &&
        !luwes::vulkan::running(claim.device, claim.swapchain) &&
        !luwes::vulkan::layers(claim.device, claim.swapchain) &&
        !luwes::vulkan::gameImages(claim.device, claim.swapchain, &images, &shown)) {
        if (survey::shooting()) ALOGI("survey   layer: none, Kelir presents alone");
        const VkResult result = kelir::vulkan::present(queue, info, claim);
        extensions::onResult(queue, result);
        return result;
    }
    const VkResult result = luwes::vulkan::queuePresent(queue, info);
    extensions::onResult(queue, result);
    return result;
}

PFN_vkVoidFunction intercept(VkDevice device, const char* name, PFN_vkVoidFunction next) {
    if (PFN_vkVoidFunction own = luwes::vulkan::intercept(name, next)) return own;
    if (next != nullptr && strcmp(name, "vkCreateSampler") == 0) {
        if (!textureSettings().on()) return nullptr;
        gCreateSampler = reinterpret_cast<PFN_vkCreateSampler>(next);
        return reinterpret_cast<PFN_vkVoidFunction>(CreateSampler);
    }
    // Shading's and the extras' wrappers go in front of Lapis's for the commands they share.
    PFN_vkVoidFunction chain = next;
    PFN_vkVoidFunction own = nullptr;
    if (PFN_vkVoidFunction recorder = lapis::intercept(device, name, next)) chain = own = recorder;
    if (PFN_vkVoidFunction extra = extensions::intercept(name, chain)) chain = own = extra;
    if (PFN_vkVoidFunction rate = shading::intercept(name, chain, textureSettings().shading())) {
        chain = own = rate;
    }
    // The capture goes in front of all: transform feedback ends before any of them binds.
    if (startsWithRays()) {
        if (PFN_vkVoidFunction capture = rays::intercept(name, chain)) own = capture;
    }
    return own;
}

}  // namespace framefx::vulkan
