/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "kelir/vulkan.h"

#include <log/log.h>
#include <system/window.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "kelir/cache.h"
#include "kelir/shaders.h"
#include "luwes/shaders.h"
#include "src/engine.h"
#include "src/target.h"
#include "src/vk_colour.h"

namespace framefx::kelir::vulkan {
namespace {

constexpr int kFrames = 3;     // presents in flight per swapchain
constexpr int kUbos = 4;       // plan constants in flight
constexpr int kLuts = 3;       // LUTs in flight: a drag changes the band every frame
constexpr int kLevels = 5;     // pyramid, a quarter of the frame and four halvings
constexpr int kMaxSources = 4;  // other images stage A grades: Luwes's two scenes, resized once
constexpr size_t kMaxStageTargets = 8;
constexpr uint64_t kFenceTimeoutNs = 1'000'000'000;
constexpr int64_t kSizeCheckNs = 1'000'000'000;
constexpr VkDeviceSize kUboSize = kPlanVec4s * 16;
constexpr VkDeviceSize kLutBytes = kLutTexels * 8;

#define KELIR_VK(X)                                                                          \
    X(QueueSubmit) X(DeviceWaitIdle) X(CreateImage) X(DestroyImage)                         \
    X(GetImageMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindImageMemory)        \
    X(CreateImageView) X(DestroyImageView) X(CreateSampler) X(DestroySampler)               \
    X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreatePipelineLayout)      \
    X(DestroyPipelineLayout) X(CreateShaderModule) X(DestroyShaderModule)                   \
    X(CreateGraphicsPipelines) X(DestroyPipeline) X(CreatePipelineCache)                    \
    X(DestroyPipelineCache) X(CreateRenderPass) X(DestroyRenderPass) X(CreateFramebuffer)   \
    X(DestroyFramebuffer) X(CreateDescriptorPool) X(DestroyDescriptorPool)                  \
    X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateCommandPool)                  \
    X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer)                   \
    X(EndCommandBuffer) X(CmdPipelineBarrier) X(CmdBeginRenderPass) X(CmdEndRenderPass)     \
    X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDraw)              \
    X(CmdSetViewport) X(CmdSetScissor) X(CmdCopyBufferToImage) X(CreateFence)               \
    X(DestroyFence) X(WaitForFences) X(ResetFences) X(GetFenceStatus) X(CreateSemaphore)    \
    X(DestroySemaphore) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements)     \
    X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(CreateQueryPool) X(DestroyQueryPool)  \
    X(CmdResetQueryPool) X(CmdWriteTimestamp) X(GetQueryPoolResults)

// The driver's device entry points: Kelir's objects never pass through the loader or the game's
// layers, so nothing that intercepts the game's pipelines touches them.
struct Fns {
#define DECLARE(name) PFN_vk##name name = nullptr;
    KELIR_VK(DECLARE)
#undef DECLARE

    bool load(VkDevice device, PFN_vkGetDeviceProcAddr proc) {
        bool ok = true;
#define LOAD(name)                                                        \
    name = reinterpret_cast<PFN_vk##name>(proc(device, "vk" #name)); \
    ok = ok && name != nullptr;
        KELIR_VK(LOAD)
#undef LOAD
        return ok;
    }
};

framefx_vk_host gHost;

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* map = nullptr;
};

struct LutSlot {
    Image image;
    Buffer staging;
    std::shared_ptr<const Luts> luts;
    Primaries primaries = Primaries::kSrgb;
    uint64_t lastUse = 0;
    bool filled = false;  // staging holds defined texels
    bool upload = false;  // which the image does not have yet
};

struct UboSlot {
    Buffer buffer;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::shared_ptr<const Plan> plan;
    int lut = -1;
    uint64_t lastUse = 0;
};

struct Frame {
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    bool pending = false;
    bool timed = false;
    uint64_t serial = 0;
};

struct Device;

struct Swapchain {
    Device* device = nullptr;
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    ANativeWindow* window = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent = {};
    int turns = 0;
    Primaries primaries = Primaries::kSrgb;
    FormatClass formatClass = FormatClass::kUnorm8;
    bool spare = false;
    KelirReason declined = KelirReason::kNone;
    std::vector<VkImage> images;

    Engine::Watch watch;
    float displayWidth = 0.0f;
    float displayHeight = 0.0f;
    int64_t sizeCheckedAt = 0;
    uint32_t seed = 0;
    KelirTargetStatus reported;

    bool prepared = false;
    VkRenderPass pass = VK_NULL_HANDLE;  // the device's, for this format
    std::vector<VkImageView> views;
    std::vector<VkFramebuffer> targets;
    // Per image, not per frame: a present's or release's wait is only known done once its image
    // comes back.
    std::vector<VkSemaphore> drawn;  // the pass wrote the output image
    std::vector<VkSemaphore> read;   // the pass read the game's image
    std::vector<VkDescriptorSet> passSets;   // per image: it as the source, and the pyramid
    std::vector<VkDescriptorSet> firstLevel;  // per image: it as the pyramid's source
    Image pyramid;
    bool pyramidDefined = false;  // in SHADER_READ_ONLY, built or not
    VkImageView levelViews[kLevels] = {};
    VkFramebuffer levelTargets[kLevels] = {};
    VkDescriptorSet levelSets[kLevels] = {};  // 1..: the level above as the source
    VkDescriptorPool pool = VK_NULL_HANDLE;
    LutSlot luts[kLuts];
    UboSlot ubos[kUbos];
    int ubo = -1;

    uint32_t family = UINT32_MAX;
    VkCommandPool commands = VK_NULL_HANDLE;
    Frame frames[kFrames];
    int frameIndex = 0;
    uint64_t serial = 0;
    uint64_t completed = 0;
    VkQueryPool timer = VK_NULL_HANDLE;  // inspect: two timestamps per frame
    uint32_t gpuNs = 0;

    // With Luwes: framebuffers over the images stage A writes, by view.
    std::vector<std::pair<VkImageView, VkFramebuffer>> stageTargets;
    // With Lapis: other images stage A grades the same way, by view: it as the source and as the
    // pyramid's.
    struct Source {
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet pass = VK_NULL_HANDLE;
        VkDescriptorSet first = VK_NULL_HANDLE;
    };
    std::vector<Source> sources;
};

struct Pass {
    VkFormat format;
    VkRenderPass pass;
};

// A pipeline evicted while frames may still use it: destroyed once they retired.
struct Retired {
    VkPipeline pipeline;
    std::vector<std::pair<const Swapchain*, uint64_t>> until;
};

struct Device {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    framefx_vk_instance instance = {};
    Fns vk;
    bool usable = false;
    VkPhysicalDeviceMemoryProperties memory = {};
    float timestampPeriod = 0.0f;
    std::vector<std::pair<VkQueue, uint32_t>> queues;
    // Taken briefly, after `lock` when both are held.
    std::mutex swapchainsLock;
    std::unordered_map<VkSwapchainKHR, std::unique_ptr<Swapchain>> swapchains;

    // Shared by every swapchain, built on first use; the lock covers them and the pipelines. A
    // pipeline compiles outside it, listed in `building` until it lands.
    std::mutex lock;
    std::condition_variable compiled;
    std::set<uint64_t> building;
    bool destroyed = false;  // the game destroys it next: build nothing more
    bool built = false;
    bool broken = false;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout sources = VK_NULL_HANDLE;  // set 0: source, pyramid
    VkDescriptorSetLayout plan = VK_NULL_HANDLE;     // set 1: plan constants, LUT
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule vertex = VK_NULL_HANDLE;
    VkShaderModule fragment = VK_NULL_HANDLE;
    VkShaderModule pyramidFragment = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    VkRenderPass levelPass = VK_NULL_HANDLE;
    VkPipeline levelPipeline = VK_NULL_HANDLE;
    std::vector<Pass> passes;
    std::vector<Pass> stagePasses;  // into an image a later pass samples
    Lru<VkPipeline> pipelines;
    std::vector<Retired> retired;
};

std::mutex gLock;
std::unordered_map<VkDevice, std::shared_ptr<Device>> gDevices;

Device* deviceOf(VkDevice device) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    return it == gDevices.end() ? nullptr : it->second.get();
}

// ---- Object helpers --------------------------------------------------------------------------

uint32_t memoryType(const Device& d, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; i++) {
        if ((bits & (1u << i)) != 0 && (d.memory.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool allocate(const Device& d, const VkMemoryRequirements& req, VkMemoryPropertyFlags want,
              VkDeviceMemory* out) {
    VkMemoryAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = req.size;
    info.memoryTypeIndex = memoryType(d, req.memoryTypeBits, want);
    return info.memoryTypeIndex != UINT32_MAX &&
            d.vk.AllocateMemory(d.device, &info, nullptr, out) == VK_SUCCESS;
}

VkImageView makeView(const Device& d, VkImage image, VkImageViewType type, VkFormat format,
                     uint32_t level, uint32_t levels) {
    VkImageViewCreateInfo view = {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = image;
    view.viewType = type;
    view.format = format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, levels, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    const VkResult r = d.vk.CreateImageView(d.device, &view, nullptr, &v);
    if (r != VK_SUCCESS) ALOGW("kelir: image view of format %d: %d", format, r);
    return r == VK_SUCCESS ? v : VK_NULL_HANDLE;
}

bool makeImage(const Device& d, Image* out, VkImageType type, VkFormat format, VkExtent3D extent,
               uint32_t levels, VkImageUsageFlags usage) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = type;
    info.format = format;
    info.extent = extent;
    info.mipLevels = levels;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d.vk.CreateImage(d.device, &info, nullptr, &out->image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d.vk.GetImageMemoryRequirements(d.device, out->image, &req);
    if (!allocate(d, req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &out->memory) ||
        d.vk.BindImageMemory(d.device, out->image, out->memory, 0) != VK_SUCCESS) {
        return false;
    }
    out->view = makeView(d, out->image,
                         type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
                         format, 0, levels);
    return out->view != VK_NULL_HANDLE;
}

void freeImage(const Device& d, Image& image) {
    if (image.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, image.view, nullptr);
    if (image.image != VK_NULL_HANDLE) d.vk.DestroyImage(d.device, image.image, nullptr);
    if (image.memory != VK_NULL_HANDLE) d.vk.FreeMemory(d.device, image.memory, nullptr);
    image = Image();
}

bool makeBuffer(const Device& d, Buffer* out, VkDeviceSize size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d.vk.CreateBuffer(d.device, &info, nullptr, &out->buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d.vk.GetBufferMemoryRequirements(d.device, out->buffer, &req);
    return allocate(d, req,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &out->memory) &&
            d.vk.BindBufferMemory(d.device, out->buffer, out->memory, 0) == VK_SUCCESS &&
            d.vk.MapMemory(d.device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->map) == VK_SUCCESS;
}

void freeBuffer(const Device& d, Buffer& b) {
    if (b.map != nullptr) d.vk.UnmapMemory(d.device, b.memory);
    if (b.buffer != VK_NULL_HANDLE) d.vk.DestroyBuffer(d.device, b.buffer, nullptr);
    if (b.memory != VK_NULL_HANDLE) d.vk.FreeMemory(d.device, b.memory, nullptr);
    b = Buffer();
}

VkShaderModule makeModule(const Device& d, const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    d.vk.CreateShaderModule(d.device, &info, nullptr, &module);
    return module;
}

// One color attachment, fully overwritten, handed on to `finalLayout`.
VkRenderPass makeRenderPass(const Device& d, VkFormat format, VkImageLayout finalLayout) {
    VkAttachmentDescription attachment = {};
    attachment.format = format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = finalLayout;
    const VkAttachmentReference ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;
    VkSubpassDependency deps[2] = {};
    // In: earlier reads of the image, and the acquire semaphore's wait for a swapchain image.
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    // Out: the next pass samples it, or the present takes it.
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
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
    d.vk.CreateRenderPass(d.device, &info, nullptr, &pass);
    return pass;
}

VkFramebuffer makeFramebuffer(const Device& d, VkRenderPass pass, VkImageView view, uint32_t width,
                              uint32_t height) {
    VkFramebufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass = pass;
    info.attachmentCount = 1;
    info.pAttachments = &view;
    info.width = width;
    info.height = height;
    info.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    d.vk.CreateFramebuffer(d.device, &info, nullptr, &fb);
    return fb;
}

VkPipeline makePipeline(Device& d, VkShaderModule fragment, VkRenderPass pass,
                        const VkSpecializationInfo* spec) {
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = d.vertex;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment;
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = spec;
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
    info.layout = d.layout;
    info.renderPass = pass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    d.vk.CreateGraphicsPipelines(d.device, d.cache, 1, &info, nullptr, &pipeline);
    return pipeline;
}

VkDescriptorSetLayout makeSetLayout(const Device& d, bool withUniform) {
    VkDescriptorSetLayoutBinding bindings[2] = {};
    for (uint32_t i = 0; i < 2; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = withUniform && i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                           : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 2;
    info.pBindings = bindings;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    d.vk.CreateDescriptorSetLayout(d.device, &info, nullptr, &layout);
    return layout;
}

// A step of building Kelir's objects failed: which one, for the decline that follows.
bool Failed(const char* what) {
    ALOGW("%s: kelir could not create its %s", Engine::get().package().c_str(), what);
    return false;
}
// Everything every swapchain of the device shares. Holds d.lock.
bool build(Device& d) {
    if (d.built || d.broken) return d.built;
    VkSamplerCreateInfo sampler = {};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = VK_LOD_CLAMP_NONE;
    d.vk.CreateSampler(d.device, &sampler, nullptr, &d.sampler);
    d.sources = makeSetLayout(d, false);
    d.plan = makeSetLayout(d, true);
    const VkDescriptorSetLayout sets[] = {d.sources, d.plan};
    const VkPushConstantRange push = {VK_SHADER_STAGE_ALL_GRAPHICS, 0,
                                      shaders::kFramePushOffset + shaders::kFramePushSize};
    VkPipelineLayoutCreateInfo layout = {};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 2;
    layout.pSetLayouts = sets;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (d.sampler != VK_NULL_HANDLE && d.sources != VK_NULL_HANDLE && d.plan != VK_NULL_HANDLE) {
        d.vk.CreatePipelineLayout(d.device, &layout, nullptr, &d.layout);
    }
    const luwes::shaders::Spirv& vert = luwes::shaders::kFullscreenVertSpirv;
    d.vertex = makeModule(d, vert.code, vert.size);
    d.fragment = makeModule(d, shaders::kKelirFragSpirv.code, shaders::kKelirFragSpirv.size);
    d.pyramidFragment =
            makeModule(d, shaders::kPyramidFragSpirv.code, shaders::kPyramidFragSpirv.size);
    VkPipelineCacheCreateInfo cache = {};
    cache.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    d.vk.CreatePipelineCache(d.device, &cache, nullptr, &d.cache);
    bool ok = d.layout != VK_NULL_HANDLE && d.vertex != VK_NULL_HANDLE &&
            d.fragment != VK_NULL_HANDLE && d.pyramidFragment != VK_NULL_HANDLE;
    if (ok) {
        d.levelPass = makeRenderPass(d, VK_FORMAT_R16G16B16A16_SFLOAT,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (d.levelPass != VK_NULL_HANDLE) {
            d.levelPipeline = makePipeline(d, d.pyramidFragment, d.levelPass, nullptr);
        }
        ok = d.levelPipeline != VK_NULL_HANDLE;
    }
    d.built = ok;
    d.broken = !ok;
    if (!ok) Failed("shared pipeline objects");
    if (!ok) ALOGE("kelir: Vulkan objects could not be built");
    return ok;
}

VkRenderPass passFor(Device& d, VkFormat format, size_t* index) {
    for (size_t i = 0; i < d.passes.size(); i++) {
        if (d.passes[i].format == format) {
            *index = i;
            return d.passes[i].pass;
        }
    }
    const VkRenderPass pass = makeRenderPass(d, format, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    if (pass == VK_NULL_HANDLE) return pass;
    d.passes.push_back({format, pass});
    *index = d.passes.size() - 1;
    return pass;
}

// Destroys evicted pipelines no frame can still use. Holds d.lock.
void sweep(Device& d) {
    std::lock_guard swapchains(d.swapchainsLock);
    std::erase_if(d.retired, [&](const Retired& r) {
        for (const auto& [sc, serial] : r.until) {
            for (const auto& [handle, live] : d.swapchains) {
                if (live.get() == sc && live->completed < serial) return false;
            }
        }
        d.vk.DestroyPipeline(d.device, r.pipeline, nullptr);
        return true;
    });
}

// The pass for a signature on a swapchain's format, built through the device's pipeline cache.
// Holds d.lock.
// The pipeline for a signature on a format: cached, or compiled by the caller with the lock
// released, so no present waits on another thread's compile. Null while another thread builds
// it: that present goes out as the game rendered it.
VkPipeline pipelineFor(Device& d, VkFormat format, uint64_t signature) {
    uint64_t key = 0;
    VkRenderPass pass = VK_NULL_HANDLE;
    {
        std::lock_guard lock(d.lock);
        if (!d.built || d.destroyed) return VK_NULL_HANDLE;
        size_t index = 0;
        pass = passFor(d, format, &index);
        if (pass == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        key = signature | static_cast<uint64_t>(index) << 48;
        if (VkPipeline* p = d.pipelines.find(key)) return *p;
        if (!d.building.insert(key).second) return VK_NULL_HANDLE;
    }
    int32_t values[shaders::kSpecCount];
    shaders::SpecValues(signature, values);
    VkSpecializationMapEntry entries[shaders::kSpecCount];
    for (int i = 0; i < shaders::kSpecCount; i++) {
        entries[i] = {shaders::kFirstSpecId + i, static_cast<uint32_t>(i * sizeof(int32_t)),
                      sizeof(int32_t)};
    }
    const VkSpecializationInfo spec = {shaders::kSpecCount, entries, sizeof(values), values};
    VkPipeline pipeline = makePipeline(d, d.fragment, pass, &spec);
    std::lock_guard lock(d.lock);
    d.building.erase(key);
    d.compiled.notify_all();
    if (pipeline == VK_NULL_HANDLE) return pipeline;
    if (d.destroyed) {
        d.vk.DestroyPipeline(d.device, pipeline, nullptr);
        return VK_NULL_HANDLE;
    }
    const VkPipeline evicted = d.pipelines.insert(key, pipeline);
    if (evicted != VK_NULL_HANDLE) {
        Retired r = {evicted, {}};
        std::lock_guard swapchains(d.swapchainsLock);
        for (const auto& [handle, sc] : d.swapchains) r.until.emplace_back(sc.get(), sc->serial);
        d.retired.push_back(std::move(r));
    }
    sweep(d);
    return pipeline;
}

Target targetOf(FormatClass format) {
    Target t;
    t.format = format;
    t.readsLinear = format == FormatClass::kSrgb8;
    return t;
}

// On the watcher thread, after a stack changed: the pipelines every live swapchain will need, so a
// present never compiles one.
void prebuild(const Config& config) {
    if (!Schedule(config.kelir.get(), config.luwes.on(), true).full) return;
    const Plan& plan = *config.kelir;
    std::vector<std::shared_ptr<Device>> devices;
    {
        std::lock_guard lock(gLock);
        for (const auto& [handle, d] : gDevices) devices.push_back(d);
    }
    for (const auto& d : devices) {
        std::vector<std::pair<VkFormat, FormatClass>> targets;
        {
            std::lock_guard swapchains(d->swapchainsLock);
            for (const auto& [handle, sc] : d->swapchains) {
                if (sc->declined == KelirReason::kNone) targets.emplace_back(sc->format, sc->formatClass);
            }
        }
        for (const auto& [format, formatClass] : targets) {
            pipelineFor(*d, format, Signature(plan, targetOf(formatClass)));
        }
    }
}

void destroyDeviceObjects(Device& d) {
    d.pipelines.forEach([&](VkPipeline p) { d.vk.DestroyPipeline(d.device, p, nullptr); });
    for (const Retired& r : d.retired) d.vk.DestroyPipeline(d.device, r.pipeline, nullptr);
    for (const Pass& p : d.passes) d.vk.DestroyRenderPass(d.device, p.pass, nullptr);
    for (const Pass& p : d.stagePasses) d.vk.DestroyRenderPass(d.device, p.pass, nullptr);
    if (d.levelPipeline != VK_NULL_HANDLE) d.vk.DestroyPipeline(d.device, d.levelPipeline, nullptr);
    if (d.levelPass != VK_NULL_HANDLE) d.vk.DestroyRenderPass(d.device, d.levelPass, nullptr);
    for (VkShaderModule m : {d.vertex, d.fragment, d.pyramidFragment}) {
        if (m != VK_NULL_HANDLE) d.vk.DestroyShaderModule(d.device, m, nullptr);
    }
    if (d.cache != VK_NULL_HANDLE) d.vk.DestroyPipelineCache(d.device, d.cache, nullptr);
    if (d.layout != VK_NULL_HANDLE) d.vk.DestroyPipelineLayout(d.device, d.layout, nullptr);
    for (VkDescriptorSetLayout l : {d.sources, d.plan}) {
        if (l != VK_NULL_HANDLE) d.vk.DestroyDescriptorSetLayout(d.device, l, nullptr);
    }
    if (d.sampler != VK_NULL_HANDLE) d.vk.DestroySampler(d.device, d.sampler, nullptr);
}

// ---- Per-swapchain resources ------------------------------------------------------------------

uint32_t levelWidth(const Swapchain& sc, int level) {
    return std::max(1u, ((sc.extent.width + 3) / 4) >> level);
}

uint32_t levelHeight(const Swapchain& sc, int level) {
    return std::max(1u, ((sc.extent.height + 3) / 4) >> level);
}

void retire(Device& d, Swapchain& sc, Frame& f) {
    if (f.timed && sc.timer != VK_NULL_HANDLE) {
        uint64_t ticks[2] = {};
        const uint32_t base = static_cast<uint32_t>(&f - sc.frames) * 2;
        if (d.vk.GetQueryPoolResults(d.device, sc.timer, base, 2, sizeof(ticks), ticks,
                                     sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const auto ns = static_cast<uint32_t>(static_cast<double>(ticks[1] - ticks[0]) *
                                                  d.timestampPeriod);
            sc.gpuNs = sc.gpuNs == 0 ? ns : (sc.gpuNs * 7 + ns) / 8;
        }
    }
    f.timed = false;
    f.pending = false;
    sc.completed = std::max(sc.completed, f.serial);
}

void waitFrames(Device& d, Swapchain& sc) {
    for (Frame& f : sc.frames) {
        if (!f.pending) continue;
        d.vk.WaitForFences(d.device, 1, &f.fence, VK_TRUE, kFenceTimeoutNs);
        d.vk.ResetFences(d.device, 1, &f.fence);
        retire(d, sc, f);
    }
}

void freeSwapchain(Device& d, Swapchain& sc) {
    waitFrames(d, sc);
    for (Frame& f : sc.frames) {
        if (f.acquired != VK_NULL_HANDLE) d.vk.DestroySemaphore(d.device, f.acquired, nullptr);
        if (f.fence != VK_NULL_HANDLE) d.vk.DestroyFence(d.device, f.fence, nullptr);
        f = Frame();
    }
    if (sc.commands != VK_NULL_HANDLE) d.vk.DestroyCommandPool(d.device, sc.commands, nullptr);
    if (sc.timer != VK_NULL_HANDLE) d.vk.DestroyQueryPool(d.device, sc.timer, nullptr);
    sc.commands = VK_NULL_HANDLE;
    sc.timer = VK_NULL_HANDLE;
    sc.family = UINT32_MAX;
    for (LutSlot& l : sc.luts) {
        freeImage(d, l.image);
        freeBuffer(d, l.staging);
        l = LutSlot();
    }
    for (UboSlot& u : sc.ubos) {
        freeBuffer(d, u.buffer);
        u = UboSlot();
    }
    sc.ubo = -1;
    for (VkFramebuffer fb : sc.targets) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
    for (VkImageView v : sc.views) d.vk.DestroyImageView(d.device, v, nullptr);
    for (VkSemaphore s : sc.drawn) d.vk.DestroySemaphore(d.device, s, nullptr);
    for (VkSemaphore s : sc.read) d.vk.DestroySemaphore(d.device, s, nullptr);
    sc.drawn.clear();
    sc.read.clear();
    for (const auto& [view, fb] : sc.stageTargets) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
    sc.stageTargets.clear();
    for (int k = 0; k < kLevels; k++) {
        if (sc.levelTargets[k] != VK_NULL_HANDLE) {
            d.vk.DestroyFramebuffer(d.device, sc.levelTargets[k], nullptr);
        }
        if (sc.levelViews[k] != VK_NULL_HANDLE) {
            d.vk.DestroyImageView(d.device, sc.levelViews[k], nullptr);
        }
        sc.levelTargets[k] = VK_NULL_HANDLE;
        sc.levelViews[k] = VK_NULL_HANDLE;
        sc.levelSets[k] = VK_NULL_HANDLE;
    }
    freeImage(d, sc.pyramid);
    sc.pyramidDefined = false;
    if (sc.pool != VK_NULL_HANDLE) d.vk.DestroyDescriptorPool(d.device, sc.pool, nullptr);
    sc.pool = VK_NULL_HANDLE;
    sc.targets.clear();
    sc.views.clear();
    sc.passSets.clear();
    sc.firstLevel.clear();
    sc.sources.clear();
    sc.prepared = false;
}

void writeImage(std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorImageInfo>& infos,
                VkDescriptorSet set, uint32_t binding, VkSampler sampler, VkImageView view) {
    infos.push_back({sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &infos.back();
    writes.push_back(w);
}


// Views, targets, descriptor sets, the pyramid, the plan rings and the per-frame objects.
bool prepare(Device& d, Swapchain& sc, uint32_t family) {
    if (sc.prepared && sc.family == family) return true;
    freeSwapchain(d, sc);
    size_t index = 0;
    {
        std::lock_guard lock(d.lock);
        sc.pass = passFor(d, sc.format, &index);
    }
    if (sc.pass == VK_NULL_HANDLE) return Failed("render pass");
    const VkRenderPass pass = sc.pass;
    const uint32_t n = static_cast<uint32_t>(sc.images.size());
    for (VkImage image : sc.images) {
        const VkImageView v = makeView(d, image, VK_IMAGE_VIEW_TYPE_2D, sc.format, 0, 1);
        if (v == VK_NULL_HANDLE) return Failed("image view");
        sc.views.push_back(v);
        const VkFramebuffer fb =
                makeFramebuffer(d, pass, v, sc.extent.width, sc.extent.height);
        if (fb == VK_NULL_HANDLE) return Failed("framebuffer");
        sc.targets.push_back(fb);
        VkSemaphoreCreateInfo semaphore = {};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (auto* list : {&sc.drawn, &sc.read}) {
            VkSemaphore s = VK_NULL_HANDLE;
            if (d.vk.CreateSemaphore(d.device, &semaphore, nullptr, &s) != VK_SUCCESS) {
                return Failed("present semaphore");
            }
            list->push_back(s);
        }
    }
    const VkExtent3D level0 = {levelWidth(sc, 0), levelHeight(sc, 0), 1};
    if (!makeImage(d, &sc.pyramid, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, level0, kLevels,
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
        return Failed("pyramid");
    }
    for (int k = 0; k < kLevels; k++) {
        sc.levelViews[k] = makeView(d, sc.pyramid.image, VK_IMAGE_VIEW_TYPE_2D,
                                    VK_FORMAT_R16G16B16A16_SFLOAT, static_cast<uint32_t>(k), 1);
        if (sc.levelViews[k] == VK_NULL_HANDLE) return Failed("pyramid view");
        sc.levelTargets[k] = makeFramebuffer(d, d.levelPass, sc.levelViews[k], levelWidth(sc, k),
                                             levelHeight(sc, k));
        if (sc.levelTargets[k] == VK_NULL_HANDLE) return Failed("pyramid framebuffer");
    }

    const uint32_t sets = 2 * n + kLevels + kUbos + 2 * kMaxSources;
    const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * sets},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kUbos},
    };
    VkDescriptorPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = sets;
    pool.poolSizeCount = 2;
    pool.pPoolSizes = sizes;
    if (d.vk.CreateDescriptorPool(d.device, &pool, nullptr, &sc.pool) != VK_SUCCESS) {
        return Failed("descriptor pool");
    }
    bool ok = true;
    const auto allocateSet = [&](VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = sc.pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        ok = ok && d.vk.AllocateDescriptorSets(d.device, &info, &set) == VK_SUCCESS;
        return set;
    };
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> infos;
    infos.reserve(2 * sets);
    for (uint32_t i = 0; i < n; i++) {
        sc.passSets.push_back(allocateSet(d.sources));
        writeImage(writes, infos, sc.passSets.back(), 0, d.sampler, sc.views[i]);
        writeImage(writes, infos, sc.passSets.back(), 1, d.sampler, sc.pyramid.view);
        sc.firstLevel.push_back(allocateSet(d.sources));
        writeImage(writes, infos, sc.firstLevel.back(), 0, d.sampler, sc.views[i]);
        writeImage(writes, infos, sc.firstLevel.back(), 1, d.sampler, sc.pyramid.view);
    }
    for (int k = 1; k < kLevels; k++) {
        sc.levelSets[k] = allocateSet(d.sources);
        writeImage(writes, infos, sc.levelSets[k], 0, d.sampler, sc.levelViews[k - 1]);
        writeImage(writes, infos, sc.levelSets[k], 1, d.sampler, sc.pyramid.view);
    }
    for (LutSlot& l : sc.luts) {
        ok = ok && makeImage(d, &l.image, VK_IMAGE_TYPE_3D, VK_FORMAT_R16G16B16A16_SFLOAT,
                             {kLutSize, kLutSize, kLutSize}, 1,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
                makeBuffer(d, &l.staging, kLutBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    }
    for (UboSlot& u : sc.ubos) {
        ok = ok && makeBuffer(d, &u.buffer, kUboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        u.set = allocateSet(d.plan);
    }
    if (!ok) return Failed("descriptor sets or plan buffers");
    d.vk.UpdateDescriptorSets(d.device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                              nullptr);

    VkCommandPoolCreateInfo commands = {};
    commands.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commands.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commands.queueFamilyIndex = family;
    if (d.vk.CreateCommandPool(d.device, &commands, nullptr, &sc.commands) != VK_SUCCESS) {
        return Failed("command pool");
    }
    sc.family = family;
    if (d.timestampPeriod > 0.0f) {
        VkQueryPoolCreateInfo timer = {};
        timer.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        timer.queryType = VK_QUERY_TYPE_TIMESTAMP;
        timer.queryCount = 2 * kFrames;
        if (d.vk.CreateQueryPool(d.device, &timer, nullptr, &sc.timer) != VK_SUCCESS) {
            sc.timer = VK_NULL_HANDLE;
        }
    }
    VkFenceCreateInfo fence = {};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkSemaphoreCreateInfo semaphore = {};
    semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (Frame& f : sc.frames) {
        VkCommandBufferAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = sc.commands;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        if (d.vk.AllocateCommandBuffers(d.device, &alloc, &f.commands) != VK_SUCCESS ||
            d.vk.CreateFence(d.device, &fence, nullptr, &f.fence) != VK_SUCCESS ||
            d.vk.CreateSemaphore(d.device, &semaphore, nullptr, &f.acquired) != VK_SUCCESS) {
            return Failed("frame objects");
        }
    }
    sc.prepared = true;
    return true;
}

bool retired(const Swapchain& sc, uint64_t lastUse) {
    return lastUse <= sc.completed;
}

// A LUT slot for the plan: the one holding its band, else a retired one with the band staged. For
// a plan without a band, any slot with defined texels; the first time, an identity. -1 when every
// slot is still in flight.
int selectLut(Swapchain& sc, const Plan& plan, int current) {
    for (int i = 0; i < kLuts; i++) {
        const LutSlot& l = sc.luts[i];
        if (l.filled && l.luts == plan.luts && (plan.luts == nullptr || l.primaries == sc.primaries)) {
            return i;
        }
    }
    if (plan.luts == nullptr && current >= 0) return current;
    for (int i = 0; i < kLuts; i++) {
        LutSlot& l = sc.luts[i];
        if (i == current || !retired(sc, l.lastUse)) continue;
        Lut identity;
        const Lut* texels = &identity;
        if (plan.luts != nullptr) {
            texels = &plan.luts->get(sc.primaries);
        } else {
            BakeLut(nullptr, 0, sc.primaries, &identity);
        }
        memcpy(l.staging.map, texels->texels.data(), kLutBytes);
        l.luts = plan.luts;
        l.primaries = sc.primaries;
        l.filled = true;
        l.upload = true;
        return i;
    }
    return -1;
}

// The UBO slot holding `plan`, with its LUT. When every slot is still in flight the frame keeps
// the plan before, so a fast drag never waits on the GPU.
int selectPlan(Device& d, Swapchain& sc, const std::shared_ptr<const Plan>& plan) {
    if (sc.ubo >= 0 && sc.ubos[sc.ubo].plan == plan) return sc.ubo;
    int ubo = -1;
    for (int i = 0; i < kUbos && ubo < 0; i++) {
        if (i != sc.ubo && retired(sc, sc.ubos[i].lastUse)) ubo = i;
    }
    if (ubo < 0) return sc.ubo;
    const int lut = selectLut(sc, *plan, sc.ubo >= 0 ? sc.ubos[sc.ubo].lut : -1);
    if (lut < 0) return sc.ubo;
    UboSlot& u = sc.ubos[ubo];
    float packed[kPlanVec4s * 4];
    PackPlan(*plan, packed);
    memcpy(u.buffer.map, packed, sizeof(packed));
    u.plan = plan;
    u.lut = lut;
    const VkDescriptorBufferInfo buffer = {u.buffer.buffer, 0, kUboSize};
    const VkDescriptorImageInfo image = {d.sampler, sc.luts[lut].image.view,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2] = {};
    for (auto& w : writes) {
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = u.set;
        w.descriptorCount = 1;
    }
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &buffer;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &image;
    d.vk.UpdateDescriptorSets(d.device, 2, writes, 0, nullptr);
    sc.ubo = ubo;
    return ubo;
}

// ---- Recording ------------------------------------------------------------------------------

void barrier(const Fns& vk, VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
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
    vk.CmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void fullscreen(const Device& d, VkCommandBuffer cb, VkRenderPass pass, VkFramebuffer fb,
                uint32_t width, uint32_t height, VkPipeline pipeline,
                std::initializer_list<VkDescriptorSet> sets, const float* params,
                const float* frame) {
    const Fns& vk = d.vk;
    VkRenderPassBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = pass;
    info.framebuffer = fb;
    info.renderArea = {{0, 0}, {width, height}};
    vk.CmdBeginRenderPass(cb, &info, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport = {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height),
                                 0.0f, 1.0f};
    const VkRect2D scissor = {{0, 0}, {width, height}};
    vk.CmdSetViewport(cb, 0, 1, &viewport);
    vk.CmdSetScissor(cb, 0, 1, &scissor);
    vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, d.layout, 0,
                             static_cast<uint32_t>(sets.size()), sets.begin(), 0, nullptr);
    if (params != nullptr) {
        vk.CmdPushConstants(cb, d.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, shaders::kFramePushOffset,
                            params);
    }
    if (frame != nullptr) {
        vk.CmdPushConstants(cb, d.layout, VK_SHADER_STAGE_ALL_GRAPHICS, shaders::kFramePushOffset,
                            shaders::kFramePushSize, frame);
    }
    vk.CmdDraw(cb, 3, 1, 0, 0);
    vk.CmdEndRenderPass(cb);
}

// Kelir's pass over the game's image `app` into `fb`: the LUT when it is new, the pyramid when the
// plan reads it, then one fullscreen draw. The game's image is read only, and left as presented.
// From `source` (with `first` naming it as the pyramid's source); `app` is the game's image
// behind it, moved between present and read layouts, or null for an image already readable.
void recordPass(Device& d, Swapchain& sc, VkCommandBuffer cb, VkImage app, VkDescriptorSet source,
                VkDescriptorSet first, VkRenderPass pass, VkFramebuffer fb, VkPipeline pipeline,
                const Plan& plan, bool pyramid, bool edges = false) {
    const Fns& vk = d.vk;
    UboSlot& u = sc.ubos[sc.ubo];
    LutSlot& lut = sc.luts[u.lut];
    if (lut.upload) {
        barrier(vk, cb, lut.image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy copy = {};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kLutSize, kLutSize, kLutSize};
        vk.CmdCopyBufferToImage(cb, lut.staging.buffer, lut.image.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier(vk, cb, lut.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        lut.upload = false;
    }
    // The game's writes are visible through its semaphores, which this submission waits on.
    if (app != VK_NULL_HANDLE) {
        barrier(vk, cb, app, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    if (pyramid) {
        for (int k = 0; k < kLevels; k++) {
            const float sw = static_cast<float>(k == 0 ? sc.extent.width : levelWidth(sc, k - 1));
            const float sh = static_cast<float>(k == 0 ? sc.extent.height : levelHeight(sc, k - 1));
            const bool linear = k == 0 && sc.formatClass == FormatClass::kSrgb8;
            const float params[16] = {1.0f / sw, 1.0f / sh, k == 0 ? 1.0f : 0.0f,
                                      linear ? 1.0f : 0.0f};
            fullscreen(d, cb, d.levelPass, sc.levelTargets[k], levelWidth(sc, k),
                       levelHeight(sc, k), d.levelPipeline,
                       {k == 0 ? first : sc.levelSets[k], u.set}, params, nullptr);
        }
        sc.pyramidDefined = true;
    } else if (!sc.pyramidDefined) {
        // Never built: the descriptor still names it, so give it a defined layout.
        VkImageMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = sc.pyramid.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, kLevels, 0, 1};
        vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                              &b);
        sc.pyramidDefined = true;
    }
    FrameInfo info;
    info.bufferWidth = static_cast<int>(sc.extent.width);
    info.bufferHeight = static_cast<int>(sc.extent.height);
    info.displayWidth = sc.displayWidth;
    info.displayHeight = sc.displayHeight;
    info.turns = sc.turns;
    info.seed = ++sc.seed;
    info.primaries = sc.primaries;
    info.format = sc.formatClass;
    info.edges = edges;
    float frame[kFrameVec4s * 4];
    PackFrame(plan, info, frame);
    fullscreen(d, cb, pass, fb, sc.extent.width, sc.extent.height, pipeline,
               {source, u.set}, nullptr, frame);
    if (app != VK_NULL_HANDLE) {
        barrier(vk, cb, app, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    }
}

void record(Device& d, Swapchain& sc, Frame& f, uint32_t app, uint32_t out, VkPipeline pipeline,
            const Plan& plan, bool timed) {
    const Fns& vk = d.vk;
    const VkCommandBuffer cb = f.commands;
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.BeginCommandBuffer(cb, &begin);
    const uint32_t base = static_cast<uint32_t>(&f - sc.frames) * 2;
    if (timed) {
        vk.CmdResetQueryPool(cb, sc.timer, base, 2);
        vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sc.timer, base);
    }
    recordPass(d, sc, cb, sc.images[app], sc.passSets[app], sc.firstLevel[app], sc.pass,
               sc.targets[out], pipeline, plan, plan.needsPyramid());
    if (timed) vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.timer, base + 1);
    vk.EndCommandBuffer(cb);
}

// The present chain for the redirected image: everything the game asked for but its damage,
// which says nothing about a frame Kelir redrew whole. Copies the structs libvulkan reads.
struct Chain {
    union Node {
        VkBaseInStructure base;
        VkPresentTimesInfoGOOGLE googleTimes;
        VkPresentTimingsInfoEXT timings;
        VkSwapchainPresentFenceInfoEXT fences;
        VkSwapchainPresentModeInfoEXT modes;
        VkPresentId2KHR ids;
    };
    Node nodes[8];

    const void* build(const void* next) {
        const auto* p = static_cast<const VkBaseInStructure*>(next);
        bool regions = false;
        for (const auto* q = p; q != nullptr; q = q->pNext) {
            regions = regions || q->sType == VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR;
        }
        if (!regions) return next;
        int count = 0;
        const void* head = nullptr;
        Node* tail = nullptr;
        for (; p != nullptr && count < 8; p = p->pNext) {
            size_t size = 0;
            switch (p->sType) {
                case VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE:
                    size = sizeof(VkPresentTimesInfoGOOGLE);
                    break;
                case VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT:
                    size = sizeof(VkPresentTimingsInfoEXT);
                    break;
                case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT:
                    size = sizeof(VkSwapchainPresentFenceInfoEXT);
                    break;
                case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT:
                    size = sizeof(VkSwapchainPresentModeInfoEXT);
                    break;
                case VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR:
                    size = sizeof(VkPresentId2KHR);
                    break;
                default:
                    continue;  // damage, and what libvulkan ignores anyway
            }
            Node& n = nodes[count++];
            memcpy(&n, p, size);
            n.base.pNext = nullptr;
            if (tail != nullptr) {
                tail->base.pNext = &n.base;
            } else {
                head = &n;
            }
            tail = &n;
        }
        return head;
    }
};

Swapchain* find(VkDevice device, VkSwapchainKHR swapchain) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    if (it == gDevices.end()) return nullptr;
    std::lock_guard swapchains(it->second->swapchainsLock);
    const auto sc = it->second->swapchains.find(swapchain);
    return sc == it->second->swapchains.end() ? nullptr : sc->second.get();
}

// The window's size in the display's orientation, checked about once a second.
void refreshDisplaySize(Swapchain& sc) {
    const int64_t now = Now();
    if (now - sc.sizeCheckedAt <= kSizeCheckNs) return;
    int w = 0;
    int h = 0;
    if (sc.window == nullptr || sc.window->query(sc.window, NATIVE_WINDOW_DEFAULT_WIDTH, &w) != 0 ||
        sc.window->query(sc.window, NATIVE_WINDOW_DEFAULT_HEIGHT, &h) != 0 || w <= 0 || h <= 0) {
        const bool turned = (sc.turns & 1) != 0;
        w = static_cast<int>(turned ? sc.extent.height : sc.extent.width);
        h = static_cast<int>(turned ? sc.extent.width : sc.extent.height);
    }
    sc.displayWidth = static_cast<float>(w);
    sc.displayHeight = static_cast<float>(h);
    sc.sizeCheckedAt = now;
}

VkRenderPass stagePassFor(Device& d, VkFormat format) {
    std::lock_guard lock(d.lock);
    for (const Pass& p : d.stagePasses) {
        if (p.format == format) return p.pass;
    }
    const VkRenderPass pass = makeRenderPass(d, format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (pass != VK_NULL_HANDLE) d.stagePasses.push_back({format, pass});
    return pass;
}

void report(Swapchain& sc, KelirState state, KelirReason reason, uint64_t signature) {
    KelirTargetStatus t;
    t.state = state;
    t.reason = reason;
    t.api = Api::kVulkan;
    t.format = sc.formatClass;
    t.width = static_cast<uint16_t>(sc.extent.width);
    t.height = static_cast<uint16_t>(sc.extent.height);
    t.rotation = static_cast<uint8_t>(sc.turns);
    t.path = state == KelirState::kRunning ? KelirPath::kVulkan : KelirPath::kNone;
    t.signature = signature;
    t.gpuNs = sc.gpuNs;
    if (t == sc.reported) return;
    sc.reported = t;
    Engine::get().reportKelir(t);
}

KelirReason declineReason(const Device& d, const VkSwapchainCreateInfoKHR* info) {
    if (!d.usable) return KelirReason::kResources;
    if (ColourOf(info->imageColorSpace, info->imageFormat) == SwapchainColour::kHdr) {
        return KelirReason::kHdr;
    }
    if (info->presentMode == VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR ||
        info->presentMode == VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR) {
        return KelirReason::kPresentMode;
    }
    if ((info->flags & VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR) != 0) return KelirReason::kProtected;
    switch (info->imageFormat) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            break;
        default:
            return KelirReason::kFormat;  // FP16 is HDR; 565 and the like too shallow to grade
    }
    VkFormatProperties2 props = {};
    props.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    d.instance.get_format_properties2(d.physical, info->imageFormat, &props);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    if ((props.formatProperties.optimalTilingFeatures & need) != need ||
        info->imageArrayLayers != 1) {
        return KelirReason::kFormat;
    }
    return KelirReason::kNone;
}

int turnsOf(VkSurfaceTransformFlagBitsKHR transform) {
    switch (transform) {
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
            return 1;
        case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
            return 2;
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
            return 3;
        default:
            return 0;
    }
}

}  // namespace

void attach(const framefx_vk_host* host) {
    gHost = *host;
    static std::once_flag listening;
    std::call_once(listening, [] { Engine::get().listen(prebuild); });
}

void onDevice(const framefx_vk_instance* in, VkPhysicalDevice physical, VkDevice device) {
    auto d = std::make_shared<Device>();
    d->device = device;
    d->physical = physical;
    d->instance = *in;
    VkPhysicalDeviceMemoryProperties2 memory = {};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    if (in->get_memory_properties2 != nullptr) in->get_memory_properties2(physical, &memory);
    d->memory = memory.memoryProperties;
    if (in->get_properties2 != nullptr) {
        VkPhysicalDeviceProperties2 props = {};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        in->get_properties2(physical, &props);
        if (props.properties.limits.timestampComputeAndGraphics) {
            d->timestampPeriod = props.properties.limits.timestampPeriod;
        }
    }
    d->usable = in->get_memory_properties2 != nullptr && in->get_format_properties2 != nullptr &&
            d->vk.load(device, in->get_device_proc_addr);
    std::lock_guard lock(gLock);
    gDevices[device] = std::move(d);
}

void destroyDevice(VkDevice device) {
    std::shared_ptr<Device> d;
    {
        std::lock_guard lock(gLock);
        const auto it = gDevices.find(device);
        if (it == gDevices.end()) return;
        d = std::move(it->second);
        gDevices.erase(it);
    }
    {
        // Waits out compiles still on it.
        std::unique_lock lock(d->lock);
        d->destroyed = true;
        d->compiled.wait(lock, [&] { return d->building.empty(); });
    }
    if (!d->usable) return;
    d->vk.DeviceWaitIdle(device);
    for (auto& [handle, sc] : d->swapchains) freeSwapchain(*d, *sc);
    destroyDeviceObjects(*d);
}

void getDeviceQueue(VkDevice device, uint32_t family, VkQueue queue) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    if (it == gDevices.end() || queue == VK_NULL_HANDLE) return;
    auto& queues = it->second->queues;
    if (std::none_of(queues.begin(), queues.end(),
                     [queue](const auto& q) { return q.first == queue; })) {
        queues.emplace_back(queue, family);
    }
}

bool adjustSwapchain(VkDevice device, VkSwapchainCreateInfoKHR* info, bool reserved) {
    const Device* d = deviceOf(device);
    if (d == nullptr || declineReason(*d, info) != KelirReason::kNone) return false;
    info->imageUsage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (!reserved) info->minImageCount += 1;
    return true;
}

void onSwapchain(VkDevice device, VkSwapchainKHR swapchain, const VkSwapchainCreateInfoKHR* asked,
                 bool spare) {
    Device* d = deviceOf(device);
    if (d == nullptr) return;
    auto sc = std::make_unique<Swapchain>();
    sc->device = d;
    sc->handle = swapchain;
    sc->window = gHost.window != nullptr ? gHost.window(swapchain) : nullptr;
    sc->format = asked->imageFormat;
    sc->extent = asked->imageExtent;
    sc->turns = turnsOf(asked->preTransform);
    sc->primaries = ColourOf(asked->imageColorSpace, asked->imageFormat) == SwapchainColour::kP3
            ? Primaries::kP3
            : Primaries::kSrgb;
    switch (asked->imageFormat) {
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SRGB:
            sc->formatClass = FormatClass::kSrgb8;
            break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            sc->formatClass = FormatClass::kUnorm10;
            break;
        default:
            break;
    }
    sc->spare = spare;
    sc->declined = declineReason(*d, asked);
    if (sc->declined == KelirReason::kNone && !spare) sc->declined = KelirReason::kResources;
    if (sc->declined != KelirReason::kNone) {
        ALOGI("%s: kelir declines a swapchain: reason %d, format %d, colour space %d, %ux%u",
              Engine::get().package().c_str(), static_cast<int>(sc->declined), asked->imageFormat,
              asked->imageColorSpace, asked->imageExtent.width, asked->imageExtent.height);
    }
    uint32_t count = 0;
    gHost.get_swapchain_images(device, swapchain, &count, nullptr);
    sc->images.resize(count);
    gHost.get_swapchain_images(device, swapchain, &count, sc->images.data());
    std::lock_guard lock(d->swapchainsLock);
    d->swapchains[swapchain] = std::move(sc);
}

void destroySwapchain(VkDevice device, VkSwapchainKHR swapchain) {
    Device* d = deviceOf(device);
    if (d == nullptr) return;
    std::unique_ptr<Swapchain> sc;
    {
        std::lock_guard lock(d->swapchainsLock);
        const auto it = d->swapchains.find(swapchain);
        if (it == d->swapchains.end()) return;
        sc = std::move(it->second);
        d->swapchains.erase(it);
    }
    if (d->usable) freeSwapchain(*d, *sc);
}

bool claim(VkQueue queue, const VkPresentInfoKHR* info, Claim* out, bool fromLuwes) {
    Engine& engine = Engine::get();
    if (!engine.active() || info->swapchainCount != 1) return false;
    Swapchain* sc = nullptr;
    {
        std::lock_guard lock(gLock);
        for (auto& [device, d] : gDevices) {
            if (std::none_of(d->queues.begin(), d->queues.end(),
                             [queue](const auto& q) { return q.first == queue; })) {
                continue;
            }
            std::lock_guard swapchains(d->swapchainsLock);
            const auto it = d->swapchains.find(info->pSwapchains[0]);
            if (it != d->swapchains.end()) sc = it->second.get();
            break;
        }
    }
    if (sc == nullptr) return false;
    sc->watch.refresh(engine);
    const Config& config = sc->watch.config();
    if (config.luwes.on() && !fromLuwes) return false;
    if (!Schedule(config.kelir.get(), false, true).full) return false;
    if (sc->declined != KelirReason::kNone) {
        report(*sc, KelirState::kDeclined, sc->declined, 0);
        return false;
    }
    out->device = sc->device->device;
    out->swapchain = sc->handle;
    out->target = sc;
    return true;
}

VkResult present(VkQueue queue, const VkPresentInfoKHR* info, const Claim& claimed) {
    Swapchain& sc = *static_cast<Swapchain*>(claimed.target);
    Device& d = *sc.device;
    const std::shared_ptr<const Plan>& plan = sc.watch.config().kelir;
    uint32_t family = UINT32_MAX;
    {
        std::lock_guard lock(gLock);
        for (const auto& [q, f] : d.queues) {
            if (q == queue) family = f;
        }
    }
    const auto pass = [&] { return gHost.queue_present(queue, info); };
    bool ok;
    {
        std::lock_guard lock(d.lock);
        ok = build(d);
    }
    if (!ok || !prepare(d, sc, family)) {
        sc.declined = KelirReason::kResources;
        ALOGW("%s: kelir declined on Vulkan: resources", Engine::get().package().c_str());
        freeSwapchain(d, sc);
        report(sc, KelirState::kDeclined, sc.declined, 0);
        return pass();
    }
    refreshDisplaySize(sc);

    Frame& f = sc.frames[sc.frameIndex];
    if (f.pending) {
        // Blocks only when the GPU is kFrames presents behind.
        if (d.vk.WaitForFences(d.device, 1, &f.fence, VK_TRUE, kFenceTimeoutNs) != VK_SUCCESS) {
            report(sc, KelirState::kRunning, KelirReason::kNone, 0);
            return pass();
        }
        d.vk.ResetFences(d.device, 1, &f.fence);
        retire(d, sc, f);
    }
    for (Frame& other : sc.frames) {
        if (other.pending && d.vk.GetFenceStatus(d.device, other.fence) == VK_SUCCESS) {
            d.vk.ResetFences(d.device, 1, &other.fence);
            retire(d, sc, other);
        }
    }
    const int ubo = selectPlan(d, sc, plan);
    if (ubo < 0) return pass();
    const Plan& running = *sc.ubos[ubo].plan;
    if (running.bypass()) return pass();
    const uint64_t signature = Signature(running, targetOf(sc.formatClass));
    const VkPipeline pipeline = pipelineFor(d, sc.format, signature);
    if (pipeline == VK_NULL_HANDLE) {
        std::lock_guard lock(d.lock);
        // Still compiling elsewhere: this frame only goes out as rendered.
        if (d.building.empty()) report(sc, KelirState::kDeclined, KelirReason::kResources, signature);
        return pass();
    }

    uint32_t out = 0;
    const VkResult acquired = gHost.acquire(d.device, sc.handle, f.acquired, &out);
    if (acquired < VK_SUCCESS || out >= sc.images.size()) return pass();
    const uint32_t app = info->pImageIndices[0];
    const bool timed = running.inspect && sc.timer != VK_NULL_HANDLE;
    record(d, sc, f, app, out, pipeline, running, timed);

    VkSemaphore waits[17];
    VkPipelineStageFlags stages[17];
    const uint32_t gameWaits = std::min(info->waitSemaphoreCount, 16u);
    for (uint32_t i = 0; i < gameWaits; i++) {
        waits[i] = info->pWaitSemaphores[i];
        stages[i] = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    waits[gameWaits] = f.acquired;
    stages[gameWaits] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSemaphore drawn = sc.drawn[out];
    const VkSemaphore read = sc.read[app];
    const VkSemaphore signals[2] = {drawn, read};
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = gameWaits + 1;
    submit.pWaitSemaphores = waits;
    submit.pWaitDstStageMask = stages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &f.commands;
    submit.signalSemaphoreCount = 2;
    submit.pSignalSemaphores = signals;
    if (info->waitSemaphoreCount > 16 ||
        d.vk.QueueSubmit(queue, 1, &submit, f.fence) != VK_SUCCESS) {
        // The output image was never drawn: hand it back and present the game's frame.
        gHost.release(queue, sc.handle, out, f.acquired);
        sc.declined = KelirReason::kResources;
        report(sc, KelirState::kDeclined, sc.declined, signature);
        return pass();
    }
    f.pending = true;
    f.timed = timed;
    f.serial = ++sc.serial;
    sc.ubos[ubo].lastUse = f.serial;
    sc.luts[sc.ubos[ubo].lut].lastUse = f.serial;
    sc.frameIndex = (sc.frameIndex + 1) % kFrames;

    Chain chain;
    VkPresentInfoKHR present = *info;
    present.pNext = chain.build(info->pNext);
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &drawn;
    present.pImageIndices = &out;
    const VkResult result = gHost.queue_present(queue, &present);
    // The game's image goes back unpresented once the pass has read it.
    gHost.release(queue, sc.handle, app, read);
    report(sc, KelirState::kRunning, KelirReason::kNone, signature);
    return result;
}

bool beginStage(VkDevice device, VkSwapchainKHR swapchain, uint32_t family,
                const std::shared_ptr<const Plan>& plan, Stage* out) {
    if (!Schedule(plan.get(), true, true).stageB) return false;
    Swapchain* found = find(device, swapchain);
    if (found == nullptr || found->declined != KelirReason::kNone) return false;
    Swapchain& sc = *found;
    Device& d = *sc.device;
    bool ok;
    {
        std::lock_guard lock(d.lock);
        ok = build(d);
    }
    if (!ok || !prepare(d, sc, family)) {
        sc.declined = KelirReason::kResources;
        report(sc, KelirState::kDeclined, sc.declined, 0);
        return false;
    }
    refreshDisplaySize(sc);
    const int ubo = selectPlan(d, sc, plan);
    if (ubo < 0) return false;
    const Plan& running = *sc.ubos[ubo].plan;
    const Stages stages = Schedule(&running, true, true);
    if (!stages.stageB) return false;
    Stage s;
    s.target = &sc;
    s.ubo = ubo;
    s.serial = ++sc.serial;
    sc.ubos[ubo].lastUse = s.serial;
    sc.luts[sc.ubos[ubo].lut].lastUse = s.serial;
    Target a = targetOf(sc.formatClass);
    a.role = Role::kStageA;
    if (stages.stageA) s.stageA = pipelineFor(d, sc.format, Signature(running, a));
    Target b;
    b.role = Role::kStageB;
    s.signatureB = Signature(running, b);
    report(sc, KelirState::kRunning, KelirReason::kNone, Signature(running, targetOf(sc.formatClass)));
    *out = s;
    return true;
}

VkFramebuffer stageTarget(Device& d, Swapchain& sc, VkRenderPass pass, VkImageView view) {
    for (const auto& [v, f] : sc.stageTargets) {
        if (v == view) return f;
    }
    if (sc.stageTargets.size() >= kMaxStageTargets) {
        d.vk.DeviceWaitIdle(d.device);
        d.vk.DestroyFramebuffer(d.device, sc.stageTargets.front().second, nullptr);
        sc.stageTargets.erase(sc.stageTargets.begin());
    }
    const VkFramebuffer fb = makeFramebuffer(d, pass, view, sc.extent.width, sc.extent.height);
    if (fb != VK_NULL_HANDLE) sc.stageTargets.emplace_back(view, fb);
    return fb;
}

void recordStageA(const Stage& stage, VkCommandBuffer cb, uint32_t app, VkImage,
                  VkImageView view) {
    Swapchain& sc = *static_cast<Swapchain*>(stage.target);
    Device& d = *sc.device;
    const VkRenderPass pass = stagePassFor(d, sc.format);
    const Plan& plan = *sc.ubos[stage.ubo].plan;
    recordPass(d, sc, cb, sc.images[app], sc.passSets[app], sc.firstLevel[app], pass,
               stageTarget(d, sc, pass, view), stage.stageA, plan, plan.needsPyramid());
}

namespace {

bool recordFrom(const Stage& stage, VkCommandBuffer cb, VkImageView source, VkImageView view,
                bool edges, bool pyramid) {
    Swapchain& sc = *static_cast<Swapchain*>(stage.target);
    Device& d = *sc.device;
    const Swapchain::Source* found = nullptr;
    for (const Swapchain::Source& s : sc.sources) {
        if (s.view == source) found = &s;
    }
    if (found == nullptr) {
        Swapchain::Source s;
        const auto spare = std::find_if(sc.sources.begin(), sc.sources.end(), [](const auto& o) {
            return o.view == VK_NULL_HANDLE;
        });
        if (spare != sc.sources.end()) {
            s = *spare;
            sc.sources.erase(spare);
        } else if (sc.sources.size() >= kMaxSources) {
            // Luwes makes its images again each time it stops and starts: the oldest view is gone.
            d.vk.DeviceWaitIdle(d.device);
            s = sc.sources.front();
            sc.sources.erase(sc.sources.begin());
        } else {
            for (VkDescriptorSet* set : {&s.pass, &s.first}) {
                VkDescriptorSetAllocateInfo info = {};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                info.descriptorPool = sc.pool;
                info.descriptorSetCount = 1;
                info.pSetLayouts = &d.sources;
                if (d.vk.AllocateDescriptorSets(d.device, &info, set) != VK_SUCCESS) return false;
            }
        }
        s.view = source;
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorImageInfo> infos;
        infos.reserve(4);
        for (VkDescriptorSet set : {s.pass, s.first}) {
            writeImage(writes, infos, set, 0, d.sampler, source);
            writeImage(writes, infos, set, 1, d.sampler, sc.pyramid.view);
        }
        d.vk.UpdateDescriptorSets(d.device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                                  nullptr);
        sc.sources.push_back(s);
        found = &sc.sources.back();
    }
    const VkRenderPass pass = stagePassFor(d, sc.format);
    const VkFramebuffer fb = stageTarget(d, sc, pass, view);
    if (fb == VK_NULL_HANDLE) return false;
    const Plan& plan = *sc.ubos[stage.ubo].plan;
    if (pyramid) {
        // The pyramid was just read for the real frame; this rebuilds it from `source`.
        d.vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0,
                                nullptr, 0, nullptr);
    }
    recordPass(d, sc, cb, VK_NULL_HANDLE, found->pass, found->first, pass, fb, stage.stageA, plan,
               pyramid, edges);
    return true;
}

}  // namespace

bool recordStageAFrom(const Stage& stage, VkCommandBuffer cb, VkImageView source, VkImage,
                      VkImageView view, bool edges) {
    return recordFrom(stage, cb, source, view, edges,
                      static_cast<Swapchain*>(stage.target)->ubos[stage.ubo].plan->needsPyramid());
}

VkDescriptorSet planSet(const Stage& stage) {
    return static_cast<Swapchain*>(stage.target)->ubos[stage.ubo].set;
}

void frameConstants(const Stage& stage, float out[kFrameVec4s * 4]) {
    Swapchain& sc = *static_cast<Swapchain*>(stage.target);
    FrameInfo info;
    info.bufferWidth = static_cast<int>(sc.extent.width);
    info.bufferHeight = static_cast<int>(sc.extent.height);
    info.displayWidth = sc.displayWidth;
    info.displayHeight = sc.displayHeight;
    info.turns = sc.turns;
    info.seed = ++sc.seed;
    info.primaries = sc.primaries;
    info.format = sc.formatClass;
    PackFrame(*sc.ubos[stage.ubo].plan, info, out);
}

VkDescriptorSetLayout planLayout(VkDevice device) {
    Device* d = deviceOf(device);
    if (d == nullptr) return VK_NULL_HANDLE;
    std::lock_guard lock(d->lock);
    return build(*d) ? d->plan : VK_NULL_HANDLE;
}

void onOffscreen(VkDevice device, VkSwapchainKHR key, VkFormat format, VkExtent2D extent,
                 int turns) {
    Device* d = deviceOf(device);
    if (d == nullptr) return;
    auto sc = std::make_unique<Swapchain>();
    sc->device = d;
    sc->handle = key;
    sc->format = format;
    sc->extent = extent;
    sc->turns = turns;
    sc->spare = true;
    sc->declined = d->usable ? KelirReason::kNone : KelirReason::kResources;
    std::lock_guard lock(d->swapchainsLock);
    d->swapchains[key] = std::move(sc);
}

uint64_t offscreenSignature(const Plan& plan, Role role) {
    Target t = targetOf(FormatClass::kUnorm8);
    t.role = role;
    return Signature(plan, t);
}

bool offscreenReady(VkDevice device, VkFormat format, uint64_t signature) {
    Device* d = deviceOf(device);
    if (d == nullptr) return false;
    std::lock_guard lock(d->lock);
    if (!build(*d)) return false;
    size_t index = 0;
    if (passFor(*d, format, &index) == VK_NULL_HANDLE) return false;
    return d->pipelines.find(signature | static_cast<uint64_t>(index) << 48) != nullptr;
}

bool offscreenCompile(VkDevice device, VkFormat format, uint64_t signature) {
    Device* d = deviceOf(device);
    if (d == nullptr) return false;
    {
        std::lock_guard lock(d->lock);
        if (!build(*d)) return false;
    }
    return pipelineFor(*d, format, signature) != VK_NULL_HANDLE;
}

bool beginOffscreen(VkDevice device, VkSwapchainKHR key, uint32_t family,
                    const std::shared_ptr<const Plan>& plan, Role role, Stage* out) {
    Swapchain* found = find(device, key);
    if (found == nullptr || found->declined != KelirReason::kNone) return false;
    Swapchain& sc = *found;
    Device& d = *sc.device;
    bool ok;
    {
        std::lock_guard lock(d.lock);
        ok = build(d);
    }
    if (!ok || !prepare(d, sc, family)) {
        sc.declined = KelirReason::kResources;
        return false;
    }
    refreshDisplaySize(sc);
    const int ubo = selectPlan(d, sc, plan);
    if (ubo < 0) return false;
    const VkPipeline pipeline =
            pipelineFor(d, sc.format, offscreenSignature(*sc.ubos[ubo].plan, role));
    if (pipeline == VK_NULL_HANDLE) return false;
    Stage s;
    s.target = &sc;
    s.ubo = ubo;
    s.serial = ++sc.serial;
    s.stageA = pipeline;
    sc.ubos[ubo].lastUse = s.serial;
    sc.luts[sc.ubos[ubo].lut].lastUse = s.serial;
    *out = s;
    return true;
}

bool recordOffscreen(const Stage& stage, VkCommandBuffer cb, VkImageView source, VkImageView view,
                     Role role) {
    return recordFrom(stage, cb, source, view, false,
                      role != Role::kStageB &&
                              static_cast<Swapchain*>(stage.target)->ubos[stage.ubo].plan->needsPyramid());
}

void forgetViews(VkDevice device, VkSwapchainKHR swapchain) {
    Swapchain* sc = find(device, swapchain);
    if (sc == nullptr) return;
    Device& d = *sc->device;
    for (const auto& [view, fb] : sc->stageTargets) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
    sc->stageTargets.clear();
    // The sets stay: the pool holds no more than these.
    for (Swapchain::Source& s : sc->sources) s.view = VK_NULL_HANDLE;
}

void stageRetired(VkDevice device, VkSwapchainKHR swapchain, uint64_t serial, uint32_t gpuNs) {
    Swapchain* sc = find(device, swapchain);
    if (sc == nullptr) return;
    sc->completed = std::max(sc->completed, serial);
    if (gpuNs != 0) sc->gpuNs = gpuNs;
}

}  // namespace framefx::kelir::vulkan
