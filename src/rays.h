/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <cstdint>

// The game's scene as triangles the GPU can trace rays against, for Kelir's ray-traced kinds and
// any other effect that needs geometry the screen does not show.
//
// Capture: in a game that starts with rays, every vertex shader gets a copy that writes its clip
// position by transform feedback (lapis/rust/src/xfb.rs), and every opaque, depth-writing pipeline
// is made from that copy. Their draws in the pass Lapis knows as the scene's (before the tonemap
// and the HUD) then write the exact positions the rasterizer gets, characters' skinning included,
// into a ring this module owns. An alpha-tested pipeline (leaves, hair, fences) whose test reads
// as one (lapis/rust/src/alpha.rs) captures the coordinate it samples at too, and the texture its
// draw has bound goes with the draw: the rays test the same texel and pass where the game cut.
//
// Build: each draw the game repeats keeps its own bottom-level tree (lapis/rust/src/scene.rs),
// built for tracing the first time it is drawn. At the present that closes a frame, the positions
// of its draws become camera-relative world points (luwes/shaders/positions.comp), compared with
// the ones their trees hold: a draw that stayed put keeps its tree, the camera's motion carried by
// its instance; one that moved is refitted while it keeps moving. One top level over this frame's
// draws is what the rays query.
namespace framefx::rays {

// `granted`: the device was made with ray query and transform feedback (extensions::Granted).
void onDevice(const framefx_vk_instance& in, VkPhysicalDevice physical, VkDevice device,
              bool granted);
void onDestroyDevice(VkDevice device);
void onQueue(VkDevice device, VkQueue queue);
// The wrapper for a command the capture watches, around `next`; nullptr keeps `next`.
PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next);
// The game presents on `queue`: the geometry submitted since its last present is the frame's.
void onPresent(VkQueue queue);

// Textures a frame's alpha-tested draws sample, at most; draws past them stay out of its rays.
constexpr uint32_t kTextures = 64;
// What a build gives the rays.
struct Scene {
    VkAccelerationStructureKHR top = VK_NULL_HANDLE;
    uint32_t triangles = 0;
    bool refitted = false;
    // Per instance of `top`, by its index, 32 bytes: uv (its first vertex in `uvs`), texture
    // (into `textures`, ~0 solid), test (channel | the varying's first component << 3), a word
    // unused, then a and b: the game cuts where a·t + b < 0, t the texel's channel. Its
    // vertices' coordinates, two floats each.
    VkDescriptorBufferInfo surfaces = {};
    VkDescriptorBufferInfo uvs = {};
    VkSampler sampler = VK_NULL_HANDLE;
    VkImageView textures[kTextures] = {};
    uint32_t textureCount = 0;
};

// Records into `cb` the frame's tree: `inverse` maps the frame's clip space to world around the
// camera at `eye` (column-major, as GLSL reads it). False when the device has no capture or the
// frame captured nothing; the caller's later work must wait for
// VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT reads, which this adds the barrier for.
bool build(VkDevice device, VkCommandBuffer cb, const float inverse[16], const double eye[3],
           Scene* out);

// This frame's constants beside its tree, for a uniform block the rays read: `data` (at most 4 KB)
// copied into the slot the last build() used. False without one.
bool frameConstants(VkDevice device, const void* data, size_t size, VkDescriptorBufferInfo* out);

// A storage block of `size` bytes (at most 256) the rays add to this frame, cleared, in the slot
// the last build() used; what it held when that slot was last used, frames ago and done, into
// `earlier`. False without one.
// `label` (at most 64 bytes) is kept with it and handed back with what it held, in
// `earlierLabel`: what the earlier frame's stats were of.
bool frameStats(VkDevice device, size_t size, const void* label, size_t labelSize, void* earlier,
                void* earlierLabel, VkDescriptorBufferInfo* out);

// Counters for the log: pipelines made capturing and the ones that stayed as they were.
struct Stats {
    uint32_t vertexModules = 0;
    uint32_t refusedModules = 0;
    uint32_t capturing = 0;
    uint32_t alphaTested = 0;
    uint32_t otherPipelines = 0;
};
Stats stats();

}  // namespace framefx::rays
