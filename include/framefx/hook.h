/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <EGL/egl.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

// Present hook between the graphics loaders and libframefx.
//
// In a process Pentas armed (see Armed()), libEGL and libvulkan load libframefx once and route
// present, plus the calls that frame a presentation target's life, through it inside their own
// dispatch. Every other process keeps the untouched entry points. One callback per API serves
// every present-time feature: Kelir stage A on the real frame, Luwes generation, then Kelir
// stage B on every frame presented, fused into the pass that writes it.

#define FRAMEFX_ABI 5
#define FRAMEFX_LIBRARY "libframefx.so"
#define FRAMEFX_ENTRY "framefx_entry"

// gl_present: libEGL presents the frame itself, unchanged.
#define FRAMEFX_PASS (-1)

#ifdef __cplusplus
extern "C" {
#endif

struct ANativeWindow;

typedef struct framefx_gl_host {
    // A GLES or EGL entry point as the driver implements it, below libEGL's per-context tables.
    void* (*proc)(const char* name);
} framefx_gl_host;

typedef struct framefx_gl_frame {
    void** state;  // engine state of the current context; libEGL keeps it with the context
    EGLDisplay display;  // driver objects
    EGLSurface surface;
    EGLConfig config;
    struct ANativeWindow* window;
    // Queues what the default framebuffer holds, with full damage.
    EGLBoolean (*swap)(void* cookie);
    void* cookie;
} framefx_gl_frame;

typedef struct framefx_vk_host {
    // libvulkan's own swapchain entry points, below every hook.
    PFN_vkCreateSwapchainKHR create_swapchain;
    PFN_vkDestroySwapchainKHR destroy_swapchain;
    PFN_vkGetSwapchainImagesKHR get_swapchain_images;
    PFN_vkQueuePresentKHR queue_present;
    // Dequeues an image to present instead of or besides the app's; `semaphore` signals once it
    // is free to write.
    VkResult (*acquire)(VkDevice device, VkSwapchainKHR swapchain, VkSemaphore semaphore,
                        uint32_t* index);
    // Hands an app image back to the swapchain unpresented once `semaphore` signals.
    VkResult (*release)(VkQueue queue, VkSwapchainKHR swapchain, uint32_t index,
                        VkSemaphore semaphore);
    // Refresh period of the swapchain's display, the value VK_GOOGLE_display_timing reports.
    int64_t (*refresh_ns)(VkSwapchainKHR swapchain);
    // The window the swapchain queues to: presents set their desired present time on it and
    // read back when frames reached the display.
    struct ANativeWindow* (*window)(VkSwapchainKHR swapchain);
    // ABI 4. The loader's allocator: command buffers that a library calling through the loader
    // records into must come from it.
    PFN_vkAllocateCommandBuffers allocate_command_buffers;
} framefx_vk_host;

// Driver entry points for one physical device, as libvulkan's driver level holds them.
typedef struct framefx_vk_instance {
    PFN_vkGetPhysicalDeviceProperties2 get_properties2;
    PFN_vkGetPhysicalDeviceFeatures2 get_features2;
    PFN_vkGetPhysicalDeviceFormatProperties2 get_format_properties2;
    PFN_vkGetPhysicalDeviceMemoryProperties2 get_memory_properties2;
    PFN_vkEnumerateDeviceExtensionProperties enumerate_extensions;
    PFN_vkCreateDevice create_device;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
    VkInstance instance;  // ABI 4: the app's instance, for libraries that look entry points up
} framefx_vk_instance;

typedef struct framefx_api {
    uint32_t abi;

    void (*gl_attach)(const framefx_gl_host* host);
    // FRAMEFX_PASS, or the result of the real frame's present.
    int32_t (*gl_present)(const framefx_gl_frame* frame);
    void (*gl_release)(void* state);
    // Swap interval to apply for the one the game asked for.
    int32_t (*gl_swap_interval)(int32_t requested);
    // 1 when the surface's last frame went out redrawn (filtered or generated): its buffer age
    // must read 0.
    int32_t (*gl_redrawn)(EGLSurface surface);

    void (*vk_attach)(const framefx_vk_host* host);
    // Creates the device with what generation needs, or as asked when the driver refuses that.
    VkResult (*vk_create_device)(const framefx_vk_instance* instance, VkPhysicalDevice physical,
                                 const VkDeviceCreateInfo* info,
                                 const VkAllocationCallbacks* allocator, VkDevice* device);
    void (*vk_destroy_device)(VkDevice device);
    void (*vk_get_device_queue)(VkDevice device, uint32_t family, VkQueue queue);
    VkResult (*vk_create_swapchain)(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                                    const VkAllocationCallbacks* allocator,
                                    VkSwapchainKHR* swapchain);
    void (*vk_destroy_swapchain)(VkDevice device, VkSwapchainKHR swapchain,
                                 const VkAllocationCallbacks* allocator);
    VkResult (*vk_queue_present)(VkQueue queue, const VkPresentInfoKHR* info);

    // ABI 4. The engine's wrapper for a device-level `name` it watches, or NULL to keep `next`.
    // Asked while the loader builds `device`'s dispatch, and with VK_NULL_HANDLE when the game
    // looks the command up through vkGetInstanceProcAddr: that wrapper serves every device.
    PFN_vkVoidFunction (*vk_intercept)(VkDevice device, const char* name, PFN_vkVoidFunction next);

    // ABI 5. The attributes to create a context with in place of `asked` (EGL_NONE-terminated,
    // may be NULL): written to `out`, which holds `capacity` values, and returned; or `asked`
    // itself. libEGL creates the context as asked when the driver refuses the returned list.
    const EGLint* (*gl_context_attribs)(const EGLint* asked, EGLint* out, int32_t capacity);
    // ABI 5. The engine's wrapper for a GLES entry point it watches, or NULL to keep `next`. Asked
    // once per name as libEGL fills its dispatch tables.
    void* (*gl_intercept)(const char* name, void* next);
} framefx_api;

typedef const framefx_api* (*framefx_entry_fn)(uint32_t abi);

// libframefx's export for the loaders: the table for `abi`, or NULL when it speaks another. Tables
// only grow, so a loader built for an older ABI reads the prefix it knows.
const framefx_api* framefx_entry(uint32_t abi);

#ifdef __cplusplus
}  // extern "C"

#include <dlfcn.h>
#include <framefx/channel.h>
#include <unistd.h>

namespace framefx {

// Whether Pentas armed this uid: one access() on its marker.
inline bool Armed(uid_t uid) { return Marked(kMarkerDir, uid); }

// The engine for an armed process, nullptr otherwise. *error is set when the process is armed
// but the engine could not be loaded.
inline const framefx_api* LoadEngine(const char** error) {
    *error = nullptr;
    if (!Armed(getuid())) return nullptr;
    void* lib = dlopen(FRAMEFX_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        *error = dlerror();
        return nullptr;
    }
    const auto entry = reinterpret_cast<framefx_entry_fn>(dlsym(lib, FRAMEFX_ENTRY));
    const framefx_api* api = entry != nullptr ? entry(FRAMEFX_ABI) : nullptr;
    if (api == nullptr) *error = "ABI mismatch";
    return api;
}

}  // namespace framefx
#endif
