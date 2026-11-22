/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <mutex>

// The video path's own Vulkan device: one per process, made on the first proxy's thread and kept
// for the process's life. It shares nothing with what the app draws through.
namespace framefx::video {

#define VIDEO_VK(X)                                                                          \
    X(QueueSubmit) X(CreateImage) X(DestroyImage) X(GetImageMemoryRequirements)             \
    X(AllocateMemory) X(FreeMemory) X(BindImageMemory) X(CreateImageView)                   \
    X(DestroyImageView) X(CreateSampler) X(DestroySampler) X(CreateSamplerYcbcrConversion)  \
    X(DestroySamplerYcbcrConversion) X(CreateDescriptorSetLayout)                           \
    X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout)          \
    X(CreateShaderModule) X(DestroyShaderModule) X(CreateGraphicsPipelines)                 \
    X(DestroyPipeline) X(CreateRenderPass) X(DestroyRenderPass) X(CreateFramebuffer)        \
    X(DestroyFramebuffer) X(CreateDescriptorPool) X(DestroyDescriptorPool)                  \
    X(AllocateDescriptorSets) X(FreeDescriptorSets) X(UpdateDescriptorSets)                 \
    X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers)                    \
    X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdPipelineBarrier) X(CmdBeginRenderPass)   \
    X(CmdEndRenderPass) X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants)     \
    X(CmdDraw) X(CmdSetViewport) X(CmdSetScissor) X(CreateFence) X(DestroyFence)            \
    X(WaitForFences) X(ResetFences) X(CreateSemaphore) X(DestroySemaphore)                  \
    X(ImportSemaphoreFdKHR) X(GetSemaphoreFdKHR) X(CreateQueryPool) X(DestroyQueryPool)     \
    X(CmdResetQueryPool) X(CmdWriteTimestamp) X(GetQueryPoolResults)                        \
    X(GetAndroidHardwareBufferPropertiesANDROID) X(CreateComputePipelines) X(CmdDispatch)   \
    X(CmdCopyImageToBuffer) X(CreateBuffer) X(DestroyBuffer)                                \
    X(GetBufferMemoryRequirements) X(BindBufferMemory) X(MapMemory) X(UnmapMemory)          \
    X(CmdClearColorImage) X(CmdClearDepthStencilImage)

struct Gpu {
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memory = {};
    float timestampPeriod = 0.0f;  // 0: no timing
    bool motion = false;           // compute subgroups of the 64 lanes motion.comp runs a block on
    bool opticalFlow = false;      // enabled on the device, for AFME
    VkInstance instance = VK_NULL_HANDLE;
    // Every decoder in the process submits to the one queue.
    mutable std::mutex queueLock;

#define DECLARE(name) PFN_vk##name name = nullptr;
    VIDEO_VK(DECLARE)
#undef DECLARE
};

// An image of the device's own memory, with a view of its one level.
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

uint32_t MemoryType(const Gpu& g, uint32_t bits, VkMemoryPropertyFlags want);
bool MakeImage(const Gpu& g, Image* out, VkFormat format, uint32_t width, uint32_t height,
               VkImageUsageFlags usage);
void FreeImage(const Gpu& g, Image& image);
VkImageView MakeView(const Gpu& g, VkImage image, VkFormat format, const void* next);
VkFramebuffer MakeFramebuffer(const Gpu& g, VkRenderPass pass, VkImageView view, uint32_t width,
                              uint32_t height);
// One color attachment, fully overwritten, left for a later pass to sample.
VkRenderPass MakePass(const Gpu& g, VkFormat format);
VkShaderModule MakeModule(const Gpu& g, const uint32_t* code, size_t size);
VkPipeline MakeFullscreen(const Gpu& g, VkShaderModule vertex, VkShaderModule fragment,
                          VkPipelineLayout layout, VkRenderPass pass);
VkSampler MakeSampler(const Gpu& g, VkFilter filter);
// `sampled` combined image samplers, then `storage` storage images, from binding 0.
VkDescriptorSetLayout MakeSetLayout(const Gpu& g, uint32_t sampled, uint32_t storage,
                                    VkShaderStageFlags stages);
// The 64 push bytes every pass of Luwes's shaders takes.
VkPipelineLayout MakePipelineLayout(const Gpu& g, VkDescriptorSetLayout set);
VkDescriptorSet AllocateSet(const Gpu& g, VkDescriptorPool pool, VkDescriptorSetLayout layout);

// One combined image sampler of a set.
struct Binding {
    VkImageView view;
    VkImageLayout layout;
    VkSampler sampler;
};
// `sampled` from binding 0, then `storage` after them when not null.
void WriteSet(const Gpu& g, VkDescriptorSet set, std::initializer_list<Binding> sampled,
              const Binding* storage);

// One fullscreen triangle with the shader's 64 push bytes.
void Fullscreen(const Gpu& g, VkCommandBuffer cb, VkRenderPass pass, VkFramebuffer target,
                uint32_t width, uint32_t height, VkPipeline pipeline, VkPipelineLayout layout,
                VkDescriptorSet set, const float params[16]);

void Barrier(const Gpu& g, VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
             VkAccessFlags dstAccess);

// The device, nullptr where this GPU cannot do the path. Blocks while the first call brings it up.
const Gpu* GetGpu();

// True on a thread inside GetGpu's vkCreateDevice: the present hook keeps its hands off that device.
bool creatingDevice();

}  // namespace framefx::video
