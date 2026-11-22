/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <framefx/hook.h>
#include <framefx/video.h>

#include "src/engine.h"
#include "src/gles.h"
#include "src/vulkan.h"
#include "video/proxy.h"

namespace {

namespace gles = framefx::gles;
namespace vulkan = framefx::vulkan;

const framefx_api kApi = {
        .abi = FRAMEFX_ABI,
        .gl_attach = gles::attach,
        .gl_present = gles::present,
        .gl_release = gles::release,
        .gl_swap_interval = gles::swapInterval,
        .gl_redrawn = gles::redrawn,
        .vk_attach = vulkan::attach,
        .vk_create_device = vulkan::createDevice,
        .vk_destroy_device = vulkan::destroyDevice,
        .vk_get_device_queue = vulkan::getDeviceQueue,
        .vk_create_swapchain = vulkan::createSwapchain,
        .vk_destroy_swapchain = vulkan::destroySwapchain,
        .vk_queue_present = vulkan::queuePresent,
        .vk_intercept = vulkan::intercept,
        .gl_context_attribs = gles::contextAttribs,
        .gl_intercept = gles::intercept,
};

}  // namespace

extern "C" __attribute__((visibility("default"))) const framefx_api* framefx_entry(uint32_t abi) {
    return abi >= 3 && abi <= FRAMEFX_ABI ? &kApi : nullptr;
}

extern "C" __attribute__((visibility("default"))) const framefx_video_api* framefx_video_entry(
        uint32_t abi) {
    if (abi != FRAMEFX_VIDEO_ABI) return nullptr;
    framefx::Engine::get().attach();
    return framefx::video::api();
}
