/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "lapis/classifier.h"
#include "lapis/model.h"
#include "lapis/signals.h"

// Records an armed game's Vulkan work into frames (lapis/model.h) from the entry points libvulkan
// routes through framefx, and classifies each frame at its present.
//
// Runs in an armed process unless Pentas turned Lapis off for the game, and wherever the survey
// runs. Decided once per process, at its first Vulkan lookup; turning Lapis off later only stops
// classifying and tapping.
namespace framefx::lapis {

// Whether this process records.
bool recording();

void attach(const framefx_vk_host* host);
void onDevice(const framefx_vk_instance* instance, VkPhysicalDevice physical, VkDevice device);
void onDestroyDevice(VkDevice device);
// `images`: the ones the game draws into, when they are not the swapchain's own.
void onSwapchain(VkDevice device, VkSwapchainKHR swapchain, const VkSwapchainCreateInfoKHR* info,
                 const std::vector<VkImage>* images = nullptr);
void onDestroySwapchain(VkSwapchainKHR swapchain);
// The recorder's wrapper for a device-level entry point, or nullptr to keep `next`.
PFN_vkVoidFunction intercept(VkDevice device, const char* name, PFN_vkVoidFunction next);
// Closes the frame the game presents with `info` and classifies it.
void onPresent(VkQueue queue, const VkPresentInfoKHR* info);

// An image of the game's frame that framefx may read at present, in place.
struct Tap {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;  // at the end of the frame; restore it
    VkImageUsageFlags usage = 0;
    // Never reused, unlike a handle the driver may hand out again once the game frees the image.
    ImageId id = 0;
    // A copy of what the game presents, so stored the way the screen shows it.
    bool upright = false;
    // Which way the frame's commands carry it to the screen (kFlipY, kFlipX), -1 when they don't
    // say; `orientationExact` false when only a composite pass's viewport hints at it.
    int orientation = -1;
    bool orientationExact = false;
    // What of the image the game drew, when less than all of it (a dynamic resolution); else
    // empty.
    VkRect2D region = {};
};

// The camera projection, as depth needs it: a stored depth d reads as the view distance
// b / (d' - c), where d' is d, or 2d - 1 for GL's clip depth.
struct Camera {
    bool known = false;
    bool gl = false;
    float c = 0, b = 0;
    float sx = 0, sy = 0;  // focal scales: view x / distance to clip x
    float jx = 0, jy = 0;  // jitter, in clip units
    // The view-projection this frame's scene was drawn with, when the game's constants show it:
    // row-major, its jitter included, and that jitter as the NDC offset it adds.
    bool matrices = false;
    float vp[16] = {};
    float ndcJitterX = 0, ndcJitterY = 0;
    bool flipsY = false;  // the scene's viewport is upside down: NDC y up the image
    // `vp` carries the camera's position; camera-relative ones rotate only, so cannot reproject a
    // moving camera.
    bool world = false;
    // Toward the sun, from the matrix the game draws its shadow map with (FindLightDirection), in
    // the axes of `vp`.
    bool light = false;
    float toLight[3] = {};
};

// One of the sun's shadow maps as the game drew it: sampled with `matrix` (row-major, world in
// `Camera::vp`'s axes to the map's clip space), x and y over [-1, 1], depth as `reversed` says.
struct ShadowMap {
    Tap map;
    float matrix[16] = {};
    bool reversed = false;
    bool flipsY = false;
    float scale = 0;  // its x row's length: larger, nearer the camera
    float region[4] = {0, 0, 1, 1};  // where in the image it was drawn: corner, size
};
constexpr int kShadowMaps = 4;
constexpr int kGBufferTargets = 4;

struct Taps {
    Tap scene;  // S, the frame before the HUD
    Tap depth;  // D, when kept
    // The game's temporal anti-aliasing, when it has one: the jittered scene it resolves, what it
    // resolves to, and the motion it reads (V, the game's own encoding).
    Tap jittered;
    Tap resolved;
    Tap velocity;
    // The transparency cut: the scene with its opaque draws alone, and with its transparent ones
    // too, at the scene's render size. Empty while no cut runs; `lit` empty when the game's own
    // copy is the opaque layer.
    Tap opaque;
    Tap lit;
    bool reversed = false;
    Camera camera;
    // The sun's maps, nearest first, while Camera::light holds.
    ShadowMap shadows[kShadowMaps];
    int shadowCount = 0;
    // What else the pass that draws the scene's geometry stores beside its colour: a deferred
    // renderer's G-buffer (albedo, normals, material), each sampleable where the frame left it.
    // Which is which is for the reader to find out.
    Tap gbuffer[kGBufferTargets];
    int gbufferCount = 0;
    // The game runs its own frame generation: S is good to show, not to generate from.
    bool gameGenerates = false;
    // This frame made no S: `scene` is the last frame's, still in place. Good to show, nothing else.
    bool held = false;
    // `scene` is a copy with x and y swapped; depth is still as the game stored it.
    bool turned = false;
    // Hide HUD skipped the HUD's draws in this frame: the presented image has no HUD to take off.
    bool hudSkipped = false;
};

// What the frame the game just presented on `swapchain` offers, while Lapis is engaged there.
// Valid until the game's next present.
bool taps(VkSwapchainKHR swapchain, Taps* out);

// The engine signals of the frame just presented on `swapchain`, while debug.framefx.signals names
// the game and Lapis is engaged there. Valid until the game's next present.
bool signals(VkSwapchainKHR swapchain, EngineSignals* out);

// Whether `cb`'s open pass draws the scene: it writes the depth the classifier found, and has not
// yet drawn its tonemap or reached its HUD. 1 yes, 0 no, -1 unknown (nothing classified yet).
int sceneState(VkCommandBuffer cb);

// The image view `cb`'s bound graphics descriptor set `set` holds at `binding` (its first
// element), as the game wrote it; null when unknown or since destroyed.
VkImageView sampledView(VkCommandBuffer cb, uint32_t set, uint32_t binding);
// Whether `view` is still the game's: not destroyed since it was made.
bool viewAlive(VkImageView view);

// Up to `size` bytes of the uniform buffer `cb`'s bound graphics set `set` holds at `binding`
// (its first element, at its dynamic offset), as they stand now, into `out`; how many.
size_t uniformBytes(VkCommandBuffer cb, uint32_t set, uint32_t binding, size_t size, uint8_t* out);

// Command buffers framefx records into itself, which a library it calls may still record into
// through the loader: never part of the game's frames.
void ignore(const VkCommandBuffer* buffers, uint32_t count);

// Calls seen per entry point family since the process started, to tell a path the recorder
// misses from one the game skips.
struct Calls {
    uint32_t renderPass = 0;
    uint32_t renderPass2 = 0;
    uint32_t rendering = 0;
    uint32_t draw = 0;
    uint32_t submit = 0;
    uint32_t submit2 = 0;
    uint32_t execute = 0;
};
Calls calls();

// The survey's `hud` word, at present: a line per draw of step `step` of `frame` with its label
// and features, and the frame's summary; nothing unless Hide HUD labelled that frame.
std::vector<std::string> hudSurveyLines(const Frame& frame, size_t step);
std::string hudSurveySummary(const Frame& frame);

// Every host mapping the game holds, with its memory type, while the mappings cannot go away.
void forEachMapping(const std::function<void(const uint8_t* data, size_t size, uint32_t type)>& fn);

}  // namespace framefx::lapis
