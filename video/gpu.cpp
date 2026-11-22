/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "video/gpu.h"

#include <framefx/hook.h>
#include <log/log.h>

#include <cstring>
#include <vector>

#include "kelir/vulkan.h"
#include "luwes/afme.h"

namespace framefx::video {
namespace {

thread_local bool gCreating = false;

constexpr const char* kExtensions[] = {
        VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        // Kelir's render passes name PRESENT_SRC, whichever image they end up writing.
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

bool bringUp(Gpu* g) {
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "framefx video";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo create = {};
    create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create, nullptr, &instance) != VK_SUCCESS) {
        ALOGW("video: no Vulkan instance");
        return false;
    }
    uint32_t count = 1;
    vkEnumeratePhysicalDevices(instance, &count, &g->physical);
    if (g->physical == VK_NULL_HANDLE) return false;

    uint32_t supported = 0;
    vkEnumerateDeviceExtensionProperties(g->physical, nullptr, &supported, nullptr);
    std::vector<VkExtensionProperties> extensions(supported);
    vkEnumerateDeviceExtensionProperties(g->physical, nullptr, &supported, extensions.data());
    for (const char* wanted : kExtensions) {
        bool found = false;
        for (const auto& e : extensions) found = found || strcmp(e.extensionName, wanted) == 0;
        if (!found) {
            ALOGW("video: the GPU lacks %s", wanted);
            return false;
        }
    }

    VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr = {};
    ycbcr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &ycbcr;
    vkGetPhysicalDeviceFeatures2(g->physical, &features);
    if (!ycbcr.samplerYcbcrConversion) {
        ALOGW("video: the GPU has no Y'CbCr sampler");
        return false;
    }

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g->physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> queues(families);
    vkGetPhysicalDeviceQueueFamilyProperties(g->physical, &families, queues.data());
    for (uint32_t i = 0; i < families && g->family == UINT32_MAX; i++) {
        if ((queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) g->family = i;
    }
    if (g->family == UINT32_MAX) return false;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue = {};
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = g->family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkPhysicalDeviceSamplerYcbcrConversionFeatures enabled = {};
    enabled.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
    enabled.samplerYcbcrConversion = VK_TRUE;
    VkDeviceCreateInfo device = {};
    device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device.pNext = &enabled;
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &queue;
    device.enabledExtensionCount = static_cast<uint32_t>(std::size(kExtensions));
    device.ppEnabledExtensionNames = kExtensions;
    const framefx_vk_instance table = {
            .get_properties2 = vkGetPhysicalDeviceProperties2,
            .get_features2 = vkGetPhysicalDeviceFeatures2,
            .get_format_properties2 = vkGetPhysicalDeviceFormatProperties2,
            .get_memory_properties2 = vkGetPhysicalDeviceMemoryProperties2,
            .enumerate_extensions = vkEnumerateDeviceExtensionProperties,
            .create_device = vkCreateDevice,
            .get_device_proc_addr = vkGetDeviceProcAddr,
            .instance = instance,
    };
    g->instance = instance;
    // What AFME needs, where the library and the optical-flow hardware are both there.
    luwes::afme::DeviceInfo extended;
    g->opticalFlow = luwes::afme::Load() != nullptr &&
            luwes::afme::Extend(table, g->physical, device, &extended);
    gCreating = true;
    VkResult made = vkCreateDevice(g->physical, g->opticalFlow ? &extended.info : &device, nullptr,
                                   &g->device);
    if (made != VK_SUCCESS && g->opticalFlow) {
        g->opticalFlow = false;
        made = vkCreateDevice(g->physical, &device, nullptr, &g->device);
    }
    gCreating = false;
    if (made != VK_SUCCESS) {
        ALOGW("video: no device: %d", made);
        return false;
    }
    vkGetDeviceQueue(g->device, g->family, 0, &g->queue);

    bool ok = g->queue != VK_NULL_HANDLE;
#define LOAD(name)                                                                 \
    g->name = reinterpret_cast<PFN_vk##name>(vkGetDeviceProcAddr(g->device, "vk" #name)); \
    ok = ok && g->name != nullptr;
    VIDEO_VK(LOAD)
#undef LOAD
    if (!ok) {
        ALOGW("video: entry points missing");
        return false;
    }

    vkGetPhysicalDeviceMemoryProperties(g->physical, &g->memory);
    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(g->physical, &properties);
    if (properties.limits.timestampComputeAndGraphics) {
        g->timestampPeriod = properties.limits.timestampPeriod;
    }
    VkPhysicalDeviceSubgroupProperties subgroup = {};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2 = {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(g->physical, &properties2);
    constexpr VkSubgroupFeatureFlags kOps =
            VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
            VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    g->motion = subgroup.subgroupSize >= 64 &&
            (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
            (subgroup.supportedOperations & kOps) == kOps;

    // Kelir draws here with its own objects, as on a game's device.
    kelir::vulkan::onDevice(&table, g->physical, g->device);
    return true;
}

}  // namespace

uint32_t MemoryType(const Gpu& g, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.memory.memoryTypeCount; i++) {
        if ((bits & (1u << i)) != 0 && (g.memory.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkImageView MakeView(const Gpu& g, VkImage image, VkFormat format, const void* next) {
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.pNext = next;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    g.CreateImageView(g.device, &info, nullptr, &view);
    return view;
}

bool MakeImage(const Gpu& g, Image* out, VkFormat format, uint32_t width, uint32_t height,
               VkImageUsageFlags usage) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (g.CreateImage(g.device, &info, nullptr, &out->image) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements;
    g.GetImageMemoryRequirements(g.device, out->image, &requirements);
    VkMemoryAllocateInfo allocate = {};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex =
            MemoryType(g, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocate.memoryTypeIndex == UINT32_MAX ||
        g.AllocateMemory(g.device, &allocate, nullptr, &out->memory) != VK_SUCCESS ||
        g.BindImageMemory(g.device, out->image, out->memory, 0) != VK_SUCCESS) {
        return false;
    }
    out->view = MakeView(g, out->image, format, nullptr);
    return out->view != VK_NULL_HANDLE;
}

void FreeImage(const Gpu& g, Image& image) {
    if (image.view != VK_NULL_HANDLE) g.DestroyImageView(g.device, image.view, nullptr);
    if (image.image != VK_NULL_HANDLE) g.DestroyImage(g.device, image.image, nullptr);
    if (image.memory != VK_NULL_HANDLE) g.FreeMemory(g.device, image.memory, nullptr);
    image = Image();
}

VkFramebuffer MakeFramebuffer(const Gpu& g, VkRenderPass pass, VkImageView view, uint32_t width,
                              uint32_t height) {
    VkFramebufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass = pass;
    info.attachmentCount = 1;
    info.pAttachments = &view;
    info.width = width;
    info.height = height;
    info.layers = 1;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    g.CreateFramebuffer(g.device, &info, nullptr, &framebuffer);
    return framebuffer;
}

VkRenderPass MakePass(const Gpu& g, VkFormat format) {
    VkAttachmentDescription attachment = {};
    attachment.format = format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkAttachmentReference reference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstStageMask =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 2;
    info.pDependencies = deps;
    VkRenderPass pass = VK_NULL_HANDLE;
    g.CreateRenderPass(g.device, &info, nullptr, &pass);
    return pass;
}

VkShaderModule MakeModule(const Gpu& g, const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    g.CreateShaderModule(g.device, &info, nullptr, &module);
    return module;
}

VkPipeline MakeFullscreen(const Gpu& g, VkShaderModule vertex, VkShaderModule fragment,
                          VkPipelineLayout layout, VkRenderPass pass) {
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo input = {};
    input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo assembly = {};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport = {};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster = {};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState attachment = {};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend = {};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &attachment;
    const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    VkGraphicsPipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = pass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    g.CreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    return pipeline;
}

VkSampler MakeSampler(const Gpu& g, VkFilter filter) {
    VkSamplerCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = info.minFilter = filter;
    info.addressModeU = info.addressModeV = info.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler = VK_NULL_HANDLE;
    g.CreateSampler(g.device, &info, nullptr, &sampler);
    return sampler;
}

VkDescriptorSetLayout MakeSetLayout(const Gpu& g, uint32_t sampled, uint32_t storage,
                                    VkShaderStageFlags stages) {
    VkDescriptorSetLayoutBinding bindings[8] = {};
    const uint32_t count = sampled + storage;
    for (uint32_t i = 0; i < count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = i < sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                 : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = stages;
    }
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = count;
    info.pBindings = bindings;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    g.CreateDescriptorSetLayout(g.device, &info, nullptr, &layout);
    return layout;
}

VkPipelineLayout MakePipelineLayout(const Gpu& g, VkDescriptorSetLayout set) {
    const VkPushConstantRange range = {VK_SHADER_STAGE_ALL, 0, 64};
    VkPipelineLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 1;
    info.pSetLayouts = &set;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    g.CreatePipelineLayout(g.device, &info, nullptr, &layout);
    return layout;
}

VkDescriptorSet AllocateSet(const Gpu& g, VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    g.AllocateDescriptorSets(g.device, &info, &set);
    return set;
}

void WriteSet(const Gpu& g, VkDescriptorSet set, std::initializer_list<Binding> sampled,
              const Binding* storage) {
    VkDescriptorImageInfo images[8];
    VkWriteDescriptorSet writes[8] = {};
    uint32_t n = 0;
    for (const Binding& b : sampled) {
        images[n] = {b.sampler, b.view, b.layout};
        writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet = set;
        writes[n].dstBinding = n;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[n].pImageInfo = &images[n];
        n++;
    }
    if (storage != nullptr) {
        images[n] = {VK_NULL_HANDLE, storage->view, storage->layout};
        writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet = set;
        writes[n].dstBinding = n;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[n].pImageInfo = &images[n];
        n++;
    }
    g.UpdateDescriptorSets(g.device, n, writes, 0, nullptr);
}

void Fullscreen(const Gpu& g, VkCommandBuffer cb, VkRenderPass pass, VkFramebuffer target,
                uint32_t width, uint32_t height, VkPipeline pipeline, VkPipelineLayout layout,
                VkDescriptorSet set, const float params[16]) {
    VkRenderPassBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = pass;
    begin.framebuffer = target;
    begin.renderArea = {{0, 0}, {width, height}};
    g.CmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport = {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height),
                                 0.0f, 1.0f};
    const VkRect2D scissor = {{0, 0}, {width, height}};
    g.CmdSetViewport(cb, 0, 1, &viewport);
    g.CmdSetScissor(cb, 0, 1, &scissor);
    g.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    g.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
    g.CmdPushConstants(cb, layout, VK_SHADER_STAGE_ALL, 0, 64, params);
    g.CmdDraw(cb, 3, 1, 0, 0);
    g.CmdEndRenderPass(cb);
}

void Barrier(const Gpu& g, VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
             VkAccessFlags dstAccess) {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    g.CmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

const Gpu* GetGpu() {
    static Gpu* gpu = [] {
        auto* g = new Gpu();
        if (bringUp(g)) return g;
        // A failed bring-up keeps what it made: the process never asks again.
        return static_cast<Gpu*>(nullptr);
    }();
    return gpu;
}

bool creatingDevice() { return gCreating; }

}  // namespace framefx::video
