/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <vector>

// Driver extensions framefx turns on in the game's device beyond what the game asked for, and the
// wrappers that use them. Each is chosen by a debug property, read once per process:
//
//   debug.framefx.vk_priority    high | realtime   global priority of every queue the game makes
//   debug.framefx.device_fault   off               the driver's fault report on a lost device (on)
//   debug.framefx.astc_decode    rgba8             LDR ASTC views decode to 8 bits, not float16
//   debug.framefx.pipeline_log   1                 pipeline compile times, summed every 10 s
//   debug.framefx.gl_priority    high | realtime   priority of every EGL context the game makes
//   debug.framefx.gl_no_error    1                 GLES contexts without error checking
//   debug.framefx.rays           1                 ray query and acceleration structures, for
//                                                  Kelir's ray-traced kinds (also on whenever the
//                                                  game starts with one in its stack)
//
// A device or context the driver refuses with the extras is made again as the game asked.
namespace framefx::extensions {

struct Settings {
    VkQueueGlobalPriorityKHR queuePriority = static_cast<VkQueueGlobalPriorityKHR>(0);
    bool deviceFault = true;
    bool astcRgba8 = false;
    bool pipelineLog = false;
    EGLint glPriority = 0;  // EGL_CONTEXT_PRIORITY_*, 0: the game's
    bool glNoError = false;
    bool rays = false;
};
const Settings& settings();

// What one device got beyond the game's request.
struct Granted {
    VkQueueGlobalPriorityKHR queuePriority = static_cast<VkQueueGlobalPriorityKHR>(0);
    bool deviceFault = false;
    bool astcDecode = false;
    bool pipelineFeedback = false;
    bool shading = false;  // pipelineFragmentShadingRate, for src/shading.h
    bool rays = false;     // accelerationStructure + rayQuery + bufferDeviceAddress
    // transformFeedback and ray-tracing position fetch, beside rays: the game's geometry for
    // src/rays.h, its triangles readable where a ray meets them.
    bool capture = false;

    bool any() const {
        return queuePriority != 0 || deviceFault || astcDecode || pipelineFeedback || shading ||
                rays;
    }
};

// The game's create info with the extras its device supports. Holds the copies it points into,
// so it must outlive the create call.
class DeviceRequest {
  public:
    // `shading`: the game starts with variable rate shading on. `rays`: with a ray-traced kind.
    DeviceRequest(const framefx_vk_instance& in, VkPhysicalDevice physical,
                  const VkDeviceCreateInfo& asked, bool shading, bool rays = false);
    DeviceRequest(const DeviceRequest&) = delete;
    DeviceRequest& operator=(const DeviceRequest&) = delete;

    // The extended info, or the game's own when nothing was added.
    const VkDeviceCreateInfo* info() const { return granted_.any() ? &info_ : asked_; }
    const Granted& granted() const { return granted_; }

  private:
    bool supports(const char* name) const;
    bool listed(const char* name) const;
    void enable(const char* name);

    const VkDeviceCreateInfo* asked_;
    VkDeviceCreateInfo info_;
    Granted granted_;
    std::vector<VkExtensionProperties> supported_;
    std::vector<const char*> extensions_;
    std::vector<VkDeviceQueueCreateInfo> queues_;
    std::vector<VkDeviceQueueGlobalPriorityCreateInfoKHR> priorities_;
    VkPhysicalDeviceFaultFeaturesEXT fault_;
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR shading_;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR structures_;
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery_;
    VkPhysicalDeviceBufferDeviceAddressFeatures address_;
    VkPhysicalDeviceTransformFeedbackFeaturesEXT feedback_;
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetch_;
};

void onDevice(const framefx_vk_instance& in, VkDevice device, const Granted& granted);
void onDestroyDevice(VkDevice device);
void onQueue(VkDevice device, VkQueue queue);
// What a present on `queue` returned: a lost device logs the driver's fault report, once.
void onResult(VkQueue queue, VkResult result);

// The wrapper for a device command the extras use, around `next`; nullptr keeps `next`.
PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next);

}  // namespace framefx::extensions
