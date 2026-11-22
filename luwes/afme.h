/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <vector>

// Qualcomm's Adreno Frame Motion Engine 2.0 (libframeestimationVK.so, proprietary and optional)
// as Luwes's motion engine: it estimates motion on the optical-flow hardware and writes the frame
// at a phase between or past two real frames. The API is reverse-engineered from the library and
// a game that calls it (lab/luwes/afme/README.md); only what Luwes uses is declared.
namespace framefx::luwes::afme {

struct Context {
    VkInstance instance;  // looks up the optical-flow entry points
    VkDevice device;
    VkPhysicalDevice gpu;
};

struct InitInfo {
    uint32_t colorWidth, colorHeight;
    uint32_t depthWidth, depthHeight;
    uint32_t flags;     // kFlags: the set a shipping game uses
    float scaleFactor;  // phase from the current frame: 0 itself, -0.5 halfway back, (-1, 1)
};

// The flags Wuthering Waves initialises AFME with.
inline constexpr uint32_t kFlags = 0x26000d;

struct FEInfo {
    VkImageView src1Color, src2Color;  // previous and current frame, shader-read
    VkImageView src1Depth, src2Depth;  // depth-aspect views, depth-read
    VkImageView src1Motion, src2Motion;  // RG32F, shader-read
    VkImageView unused[3];
    VkImageView output;  // colour attachment; left as one
    float matrix[16];    // clip of the current frame to clip of the previous
    float jitter[4];
    VkCommandBuffer cmd;  // recording; AFME neither begins nor ends it with kFlags. Must come
                          // from the loader, as AFME records through it.
    uint64_t reserved;
};
static_assert(sizeof(FEInfo) == 22 * 8, "AFME reads 22 qwords");

struct Api {
    bool (*isSupported)(Context*);
    void* (*init)(Context*, InitInfo*);
    bool (*update)(void*, InitInfo*);
    bool (*estimate)(void*, FEInfo*);
    bool (*destroy)(void*);
};

// The library, or nullptr when it is not installed.
const Api* Load();
// Whether the game's GL is ANGLE. The library makes and destroys an EGL context of its own,
// which is then ANGLE's, and a present arrives from inside ANGLE's swap with its locks held:
// destroying the context there never returns.
bool BehindAngle();

// A device create info with what AFME needs added: VK_NV_optical_flow and its feature, and the
// fd-sharing extensions it looks up. The copies live in `storage`. False when the device lacks
// optical flow, so the game's own create info stands.
struct DeviceInfo {
    VkDeviceCreateInfo info;
    std::vector<const char*> extensions;
    VkPhysicalDeviceOpticalFlowFeaturesNV opticalFlow;
};
bool Extend(const framefx_vk_instance& in, VkPhysicalDevice physical,
            const VkDeviceCreateInfo& asked, DeviceInfo* storage);

}  // namespace framefx::luwes::afme
