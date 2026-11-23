/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "lapis/recorder.h"

#include <android-base/file.h>
#include <android-base/properties.h>
#include <android-base/stringprintf.h>
#include <android-base/strings.h>
#include <framefx/alih.h>
#include <log/log.h>
#include <sys/prctl.h>
#include <sys/system_properties.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <type_traits>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lapis/hud.h"
#include "lapis/hud_probe.h"
#include "lapis/projection.h"
#include "lapis/signals.h"
#include "luwes/capture.h"
#include "luwes/afme.h"
#include "src/engine.h"
#include "src/safety.h"
#include "src/survey.h"

namespace framefx::lapis {
namespace {

using android::base::StringAppendF;
using android::base::StringPrintf;

// A command buffer the game never begins through the loader (a library recording into someone
// else's) stops growing here.
constexpr size_t kMaxEvents = 1 << 12;
// Collected work with no present to close it (an offscreen device) is dropped past this.
constexpr size_t kMaxCollected = 1 << 16;
constexpr int64_t kStatsPeriodNs = 10'000'000'000;
// A float image over a third of the target's area is the scene the tonemap reads, not its bloom.
constexpr uint64_t kUntonedShare = 3;
// Armed frames presented unsplit before the log says so.
constexpr uint32_t kSplitPatience = 120;
// Projection scan, while engaged: bytes per present, fast re-probe interval of cached hit slots,
// retry rest when not yet found, and background full-sweep refresh interval once locked.
constexpr size_t kScanBytesPerPresent = 192 << 10;
constexpr size_t kMaxScannedMapping = 128 << 20;
constexpr int64_t kProbeRestNs = 1'000'000'000;
constexpr int64_t kSweepRestNs = 5'000'000'000;
constexpr int64_t kFullSweepRestNs = 30'000'000'000;

// The entry points under each wrapper: the driver's, or the loader's where it hooks the call. The
// same for every device of the one driver; core and KHR aliases share a slot.
struct Next {
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkBindImageMemory2 BindImageMemory2;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkCreateRenderPass CreateRenderPass;
    PFN_vkCreateRenderPass2 CreateRenderPass2;
    PFN_vkCreateFramebuffer CreateFramebuffer;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateDescriptorUpdateTemplate CreateDescriptorUpdateTemplate;
    PFN_vkUpdateDescriptorSetWithTemplate UpdateDescriptorSetWithTemplate;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkCmdExecuteCommands CmdExecuteCommands;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
    PFN_vkCmdBeginRenderPass2 CmdBeginRenderPass2;
    PFN_vkCmdEndRenderPass CmdEndRenderPass;
    PFN_vkCmdEndRenderPass2 CmdEndRenderPass2;
    PFN_vkCmdBeginRendering CmdBeginRendering;
    PFN_vkCmdEndRendering CmdEndRendering;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdSetDepthWriteEnable CmdSetDepthWriteEnable;
    PFN_vkCmdSetDepthTestEnable CmdSetDepthTestEnable;
    PFN_vkCmdSetDepthCompareOp CmdSetDepthCompareOp;
    PFN_vkCmdClearAttachments CmdClearAttachments;
    PFN_vkCmdSetScissor CmdSetScissor;
    PFN_vkCmdDraw CmdDraw;
    PFN_vkCmdDrawIndexed CmdDrawIndexed;
    PFN_vkCmdDrawIndirect CmdDrawIndirect;
    PFN_vkCmdDrawIndexedIndirect CmdDrawIndexedIndirect;
    PFN_vkCmdDrawIndirectCount CmdDrawIndirectCount;
    PFN_vkCmdDrawIndexedIndirectCount CmdDrawIndexedIndirectCount;
    PFN_vkCmdCopyImage CmdCopyImage;
    PFN_vkCmdCopyImage2 CmdCopyImage2;
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdBlitImage2 CmdBlitImage2;
    PFN_vkCmdResolveImage CmdResolveImage;
    PFN_vkCmdResolveImage2 CmdResolveImage2;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
    PFN_vkCmdClearDepthStencilImage CmdClearDepthStencilImage;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkQueueSubmit2 QueueSubmit2;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdDispatchIndirect CmdDispatchIndirect;
    PFN_vkMapMemory MapMemory;
    PFN_vkMapMemory2 MapMemory2;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkUnmapMemory2 UnmapMemory2;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindOpticalFlowSessionImageNV BindOpticalFlowSessionImageNV;
    PFN_vkCmdOpticalFlowExecuteNV CmdOpticalFlowExecuteNV;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkBindBufferMemory2 BindBufferMemory2;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkCmdSetViewport CmdSetViewport;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
    PFN_vkCmdBindVertexBuffers2 CmdBindVertexBuffers2;
    PFN_vkCmdBindIndexBuffer CmdBindIndexBuffer;
    PFN_vkCmdSetVertexInputEXT CmdSetVertexInputEXT;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkCmdCopyBuffer2 CmdCopyBuffer2;
    PFN_vkCmdUpdateBuffer CmdUpdateBuffer;
    PFN_vkCmdBindIndexBuffer2KHR CmdBindIndexBuffer2;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
    PFN_vkCmdCopyBufferToImage2 CmdCopyBufferToImage2;
};

Next gNext = {};
framefx_vk_host gHost = {};

// Fixed at the first call: what it turns on must see the game's objects from the first.
bool alihOn() {
    static const bool on = [] {
        Engine& engine = Engine::get();
        engine.attach();
        return engine.config().lapis.textTap;
    }();
    return on;
}

bool textTap() {
    return survey::textTap() || alihOn();
}

// debug.framefx.signals: comma-separated packages whose engine signals are gathered. Read once.
bool signalsOn() {
    static const bool on = [] {
        char value[PROP_VALUE_MAX] = {};
        __system_property_get("debug.framefx.signals", value);
        const std::string package = Engine::get().package().substr(0, Engine::get().package().find(':'));
        for (const std::string& p : android::base::Split(value, ",")) {
            if (!p.empty() && p == package) return true;
        }
        return false;
    }();
    return on;
}

// Device functions Lapis calls itself, under every wrapper.
struct Own {
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCreateRenderPass CreateRenderPass;
    PFN_vkCreateRenderPass2 CreateRenderPass2;  // the rest of the 2 family: null on a 1.0 device
    PFN_vkDestroyRenderPass DestroyRenderPass;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
    PFN_vkCmdBeginRenderPass2 CmdBeginRenderPass2;
    PFN_vkCmdEndRenderPass CmdEndRenderPass;
    PFN_vkCmdEndRenderPass2 CmdEndRenderPass2;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdCopyImage CmdCopyImage;

    bool load(VkDevice device, PFN_vkGetDeviceProcAddr get) {
        bool ok = true;
        const auto fetch = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(get(device, name));
            ok = ok && fn != nullptr;
        };
        // Core in 1.2, or the KHR extension's, or none.
        const auto either = [&](auto& fn, const char* core, const char* khr) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(get(device, core));
            if (fn == nullptr) {
                fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(get(device, khr));
            }
        };
        fetch(CreateImage, "vkCreateImage");
        fetch(DestroyImage, "vkDestroyImage");
        fetch(GetImageMemoryRequirements, "vkGetImageMemoryRequirements");
        fetch(AllocateMemory, "vkAllocateMemory");
        fetch(FreeMemory, "vkFreeMemory");
        fetch(BindImageMemory, "vkBindImageMemory");
        fetch(CreateRenderPass, "vkCreateRenderPass");
        fetch(DestroyRenderPass, "vkDestroyRenderPass");
        fetch(CmdBeginRenderPass, "vkCmdBeginRenderPass");
        fetch(CmdEndRenderPass, "vkCmdEndRenderPass");
        either(CreateRenderPass2, "vkCreateRenderPass2", "vkCreateRenderPass2KHR");
        either(CmdBeginRenderPass2, "vkCmdBeginRenderPass2", "vkCmdBeginRenderPass2KHR");
        either(CmdEndRenderPass2, "vkCmdEndRenderPass2", "vkCmdEndRenderPass2KHR");
        fetch(CmdPipelineBarrier, "vkCmdPipelineBarrier");
        fetch(CmdCopyImage, "vkCmdCopyImage");
        return ok;
    }
};

// What replacing text needs under every wrapper. Apart from Own: a miss here must not cost a split.
struct AlihFn {
    PFN_vkCreateImage CreateImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkCreateSampler CreateSampler;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
    PFN_vkCmdBindVertexBuffers2 CmdBindVertexBuffers2;  // null without the feature
    PFN_vkCmdBindIndexBuffer CmdBindIndexBuffer;
    PFN_vkCmdDrawIndexed CmdDrawIndexed;

    bool load(VkDevice device, PFN_vkGetDeviceProcAddr get) {
        bool ok = true;
        const auto fetch = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(get(device, name));
            ok = ok && fn != nullptr;
        };
        fetch(CreateImage, "vkCreateImage");
        fetch(GetImageMemoryRequirements, "vkGetImageMemoryRequirements");
        fetch(AllocateMemory, "vkAllocateMemory");
        fetch(BindImageMemory, "vkBindImageMemory");
        fetch(CreateImageView, "vkCreateImageView");
        fetch(CreateSampler, "vkCreateSampler");
        fetch(CreateBuffer, "vkCreateBuffer");
        fetch(GetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
        fetch(BindBufferMemory, "vkBindBufferMemory");
        fetch(MapMemory, "vkMapMemory");
        fetch(CreateDescriptorPool, "vkCreateDescriptorPool");
        fetch(AllocateDescriptorSets, "vkAllocateDescriptorSets");
        fetch(UpdateDescriptorSets, "vkUpdateDescriptorSets");
        fetch(CmdPipelineBarrier, "vkCmdPipelineBarrier");
        fetch(CmdCopyBufferToImage, "vkCmdCopyBufferToImage");
        fetch(CmdBindDescriptorSets, "vkCmdBindDescriptorSets");
        fetch(CmdBindVertexBuffers, "vkCmdBindVertexBuffers");
        fetch(CmdBindIndexBuffer, "vkCmdBindIndexBuffer");
        fetch(CmdDrawIndexed, "vkCmdDrawIndexed");
        CmdBindVertexBuffers2 = reinterpret_cast<PFN_vkCmdBindVertexBuffers2>(
                get(device, "vkCmdBindVertexBuffers2"));
        if (CmdBindVertexBuffers2 == nullptr) {
            CmdBindVertexBuffers2 = reinterpret_cast<PFN_vkCmdBindVertexBuffers2>(
                    get(device, "vkCmdBindVertexBuffers2EXT"));
        }
        return ok;
    }
};

struct DeviceInfo {
    VkPhysicalDeviceMemoryProperties memory = {};
    Own fn = {};
    AlihFn alih = {};
    bool alihLoaded = false;
    bool owned = false;  // every Own entry point found
};

struct AttachmentDesc {
    VkFormat format;
    VkSampleCountFlagBits samples;
    VkAttachmentLoadOp load;
    VkAttachmentStoreOp store;
    VkAttachmentLoadOp stencilLoad;
    VkAttachmentStoreOp stencilStore;
    VkImageLayout final;
};

// A single-subpass render pass without extension structs, as the game made it: enough to make
// the same pass with every attachment loaded.
struct PassRecipe {
    std::vector<VkAttachmentDescription2> attachments;
    std::vector<VkAttachmentReference2> colors;
    std::vector<VkAttachmentReference2> resolves;
    std::vector<VkAttachmentReference2> inputs;
    std::vector<uint32_t> preserve;
    bool hasDepth = false;
    VkAttachmentReference2 depth = {};
    VkSubpassDescriptionFlags flags = 0;
    std::vector<VkSubpassDependency2> dependencies;
};

struct RenderPassDesc {
    std::vector<AttachmentDesc> attachments;
    std::vector<uint32_t> inputs;  // attachments any subpass reads as input
    uint32_t subpasses = 1;
    bool depthResolve = false;
    std::shared_ptr<PassRecipe> recipe;
};

// Where a buffer's bytes live, for reading what a draw's uniform buffers held.
struct BufferDesc {
    VkDeviceSize size = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkBufferUsageFlags usage = 0;
};

struct BufferBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = 0;
    bool dynamic = false;
};

// A set layout's dynamic buffer descriptors, in the order vkCmdBindDescriptorSets consumes their
// offsets: by binding, then array element.
struct SetLayoutDesc {
    std::vector<uint32_t> dynamicBindings;  // binding << 16 | element
    std::vector<VkDescriptorSetLayoutBinding> bindings;  // kept only to clone a set
};

struct PipelineDesc {
    float viewportHeight = 0;  // static, signed; 0 when dynamic
    float viewportX = 0, viewportY = 0, viewportWidth = 0;
    bool test = false;
    bool write = false;
    bool dynamicTest = false;
    bool dynamicWrite = false;
    VkCompareOp compare = VK_COMPARE_OP_ALWAYS;
    bool dynamicCompare = false;
    bool blend = false;
    bool stencilWrite = false;  // writes a reference value where it draws
    uint32_t fragment = 0;  // the fragment shader's SPIR-V hash
    uint8_t vertex = 0;     // kHudVertex* bits
};

// How a pipeline reads its vertices, for the survey's `atlas` word.
struct VertexLayout {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
};

constexpr int kBindPoints = 2;  // graphics, compute
constexpr int kMaxVertexBindings = 8;
constexpr size_t kMaxDrawTrace = 2000;
constexpr int kMaxSets = 8;

// One command buffer's recording. Kept for the process's life: threads cache a pointer to it.
struct Record {
    std::vector<Step> events;
    // Per event: the descriptor sets its draws or dispatches had bound. Resolved to images at
    // submit, off the draw path.
    std::vector<std::vector<VkDescriptorSet>> sets;
    int open = -1;  // index of the pass being recorded
    Pass loose;     // a secondary's draws inside its caller's pass
    std::vector<VkDescriptorSet> looseSets;
    PipelineDesc pipeline;
    bool dynTest = false;
    bool dynWrite = false;
    VkCompareOp dynCompare = VK_COMPARE_OP_ALWAYS;
    VkExtent2D scissor = {};
    uint32_t drawCount = 0;  // of the draw being counted: indices or vertices
    VkDescriptorSet bound[kBindPoints][kMaxSets] = {};
    std::vector<uint32_t> dynamicOffsets[kMaxSets];  // graphics, per bound set
    uint32_t boundVersion[kBindPoints] = {};
    // The graphics sets bound at the first draw of a pass whose constants Lapis keeps, by event.
    struct FirstDraw {
        int event = -1;
        uint32_t draw = 0;  // in the pass
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSet sets[kMaxSets] = {};
        std::vector<uint32_t> offsets[kMaxSets];
        bool hud = false;  // sampled for Hide HUD: into Pass::hudConstants
        int tile = -1;     // the first into Pass::tiles[tile]
    };
    std::vector<FirstDraw> firstDraws;
    VkPipeline graphics = VK_NULL_HANDLE;
    float viewportHeight = 0;  // the last set dynamically
    float viewportX = 0, viewportY = 0, viewportWidth = 0;
    uint32_t takenVersion = UINT32_MAX;  // the graphics version the open pass last took sets at
    bool ignored = false;  // framefx's own; a game that begins it got the handle reused
    bool pastScene = false;  // the open pass drew its tonemap or reached its HUD
    VkPipelineLayout boundLayout = VK_NULL_HANDLE;  // of the last graphics descriptor bind
    // Hide HUD: what each bound graphics set samples (kHudTex*), the draw being counted, and
    // whether the skip table drops it.
    uint8_t hudTextures[kMaxSets] = {};
    uint32_t hudInstances = 1;
    bool hudIndirect = false;
    bool hudSkip = false;
    // Survey only, word `atlas`: the draw being counted.
    struct {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize stride = 0;  // 0: the pipeline's
    } vertexBuffers[kMaxVertexBindings];
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceSize indexOffset = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    std::shared_ptr<const VertexLayout> dynamicVertex;
    bool indexed = false;
    uint32_t firstElement = 0;
    int32_t vertexOffset = 0;
    // A pass being recorded that framefx splits: the HUD cut, until it is split, and the
    // transparency cut.
    struct Cut {
        bool armed = false;
        bool v2 = false;
        bool toned = false;  // a draw sampled the untoned scene
        VkRenderPassBeginInfo begin = {};
        std::vector<VkClearValue> clears;  // `begin` points here
        std::vector<VkImageView> views;    // and through `attachments`, on an imageless framebuffer
        VkRenderPassAttachmentBeginInfo attachments = {};
        bool atClear = false;
        uint32_t clearIndex = 0;
        uint32_t clearsSeen = 0;
        VkImage target = VK_NULL_HANDLE;
        ImageId targetId = 0;
        Step pass;  // as begun, to open its resumed half
    };
    Cut cut;
    Cut glass;
    // The transparency cut's copies at this pass's end: the opaque scene (its transparency comes
    // in a pass of its own), the lit one.
    bool glassOpaqueAtEnd = false;
    bool glassLitAtEnd = false;
    VkImage glassTarget = VK_NULL_HANDLE;
    VkImageLayout glassFinal = VK_IMAGE_LAYOUT_UNDEFINED;

    void clear() {
        ignored = false;
        pastScene = false;
        std::fill(std::begin(hudTextures), std::end(hudTextures), 0);
        hudSkip = false;
        events.clear();
        sets.clear();
        open = -1;
        loose = Pass();
        looseSets.clear();
        pipeline = PipelineDesc();
        dynTest = false;
        dynWrite = false;
        for (auto& point : bound) std::fill(std::begin(point), std::end(point), VK_NULL_HANDLE);
        for (auto& offsets : dynamicOffsets) offsets.clear();
        firstDraws.clear();
        for (auto& b : vertexBuffers) b = {};
        indexBuffer = VK_NULL_HANDLE;
        dynamicVertex.reset();
        viewportHeight = 0;
        viewportX = viewportY = viewportWidth = 0;
        takenVersion = UINT32_MAX;
        cut.armed = false;
    }

    bool push(Step step) {
        if (ignored || events.size() >= kMaxEvents) return false;
        events.push_back(std::move(step));
        sets.emplace_back();
        return true;
    }
};

struct Swapchain {
    VkDevice device = VK_NULL_HANDLE;
    std::vector<ImageId> images;
    uint32_t width = 0;
    uint32_t height = 0;
};

// The game's presenting pass split at its first HUD draw, where the target holds the scene the
// tonemap drew and no HUD yet: Lapis ends the pass there, copies the target into `scene`, and
// resumes with `resume`, the same pass with every attachment loaded.
struct Split {
    VkDevice device = VK_NULL_HANDLE;
    Own fn = {};
    VkRenderPass pass = VK_NULL_HANDLE;
    VkRenderPass resume = VK_NULL_HANDLE;
    uint32_t attachment = 0;  // the target's
    VkImageLayout final = VK_IMAGE_LAYOUT_UNDEFINED;  // the target's, where the copy finds it
    VkImage scene = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    ImageId sceneId = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    bool logged = false;
    uint32_t missed = 0;  // frames armed but presented unsplit, in a row
    // Split before the depth clear of this index in the pass, not after its tonemap.
    bool atClear = false;
    // The scene's format where the classifier found it unfinished though not float: the draws
    // sampling an image of it are the tonemap too.
    VkFormat untoned = VK_FORMAT_UNDEFINED;
    uint32_t clearIndex = 0;
};
Split gSplit;
// The transparency cut lapis/rust/src/glass.rs finds: the opaque scene copied where it is whole,
// the lit one at the end of the pass its transparency is drawn in. Taps `opaque` and `lit`.
struct Glass {
    uint8_t kind = 0;                       // glass.rs CLASS_*
    VkRenderPass cutPass = VK_NULL_HANDLE;  // in one pass: the pass split; own pass: the opaque one
    VkRenderPass litPass = VK_NULL_HANDLE;  // the pass whose end holds the lit scene
    uint32_t draw = 0;                      // in one pass: the draw split at
    Split opaque;                           // the copy, the resume pass, the target's attachment
    Split lit;                              // its image alone
    ImageId refraction = 0;                 // the game's own copy of the opaque scene
    uint32_t litAttachment = 0;             // the target's attachment in the lit pass
    VkImageLayout litFinal = VK_IMAGE_LAYOUT_UNDEFINED;
    uint64_t refused = 0;
    bool logged = false;
};
Glass gGlass;
// glass.rs GlassTracker, kept here between frames.
alignas(8) uint8_t gGlassTracker[64] = {};
// The depth image the scene draws into, from the last classified frame; 0 while unknown.
std::atomic<ImageId> gSceneDepth{0};
// The scene's depth copied out as its pass ends, where the game stores it there and a later pass
// of its own starts the image over or resolves into it again.
struct Keep {
    VkDevice device = VK_NULL_HANDLE;
    Own fn = {};
    VkRenderPass pass = VK_NULL_HANDLE;  // the pass whose end leaves the depth in `source`
    ImageId source = 0;
    VkImageLayout final = VK_IMAGE_LAYOUT_UNDEFINED;  // the source's, where the copy finds it
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    ImageId id = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    bool logged = false;
};
Keep gKeep;
// Objects a split used, kept until their device goes: a game's command buffer may still hold them.
struct Spent {
    VkDevice device;
    VkRenderPass pass;
    VkImage image;
    VkDeviceMemory memory;
};
std::vector<Spent> gSpent;
// Each split pass's loaded copy, by the game's pass: a game moving between its menu and its
// gameplay passes arms them in turn.
std::unordered_map<VkRenderPass, std::pair<VkDevice, VkRenderPass>> gResumes;

std::mutex gLock;
std::atomic<bool> gOn{false};
std::unordered_map<VkDevice, DeviceInfo> gDevices;
ImageId gNextId = 1;
std::unordered_map<VkImage, ImageId> gImageIds;
std::unordered_map<ImageId, VkImage> gHandles;
Images gImages;
std::unordered_map<VkImageView, VkImage> gViews;
// Single-channel views, and for an image the swizzle of the view its descriptors use: ANGLE keeps
// others of the same image with none, and a GL alpha texture is all in its swizzle.
std::unordered_map<VkImageView, VkComponentMapping> gViewSwizzles;
std::unordered_map<ImageId, VkComponentMapping> gSwizzles;
// A game's descriptor set -> its copy with the translation's atlas in it.
std::unordered_map<VkDescriptorSet, VkDescriptorSet> gAlihClones;
std::unordered_map<VkDeviceMemory, std::pair<VkDevice, uint32_t>> gMemoryTypes;
std::unordered_map<VkRenderPass, RenderPassDesc> gRenderPasses;
std::unordered_map<VkFramebuffer, std::vector<VkImageView>> gFramebuffers;
std::unordered_map<VkPipeline, PipelineDesc> gPipelines;
std::unordered_map<VkDescriptorUpdateTemplate, std::vector<VkDescriptorUpdateTemplateEntry>>
        gTemplates;
std::unordered_map<VkCommandBuffer, std::unique_ptr<Record>> gRecords;
// Images per descriptor set, keyed by binding << 32 | array element.
std::unordered_map<VkDescriptorSet, std::unordered_map<uint64_t, ImageId>> gSetImages;
// The views themselves, keyed like gSetImages: what the rays sample an alpha test's texture from.
std::unordered_map<VkDescriptorSet, std::unordered_map<uint64_t, VkImageView>> gSetViews;
// Uniform buffers per descriptor set, keyed like gSetImages.
std::unordered_map<VkDescriptorSet, std::unordered_map<uint64_t, BufferBinding>> gSetBuffers;
std::unordered_map<VkDescriptorSetLayout, SetLayoutDesc> gSetLayouts;
std::unordered_map<VkDescriptorSet, VkDescriptorSetLayout> gSetLayoutOf;
std::unordered_map<VkBuffer, BufferDesc> gBuffers;
std::unordered_map<VkPipeline, std::shared_ptr<const VertexLayout>> gVertexLayouts;
// The game's pass whose single draw's constants every frame keeps (handle bits), set from the
// frame before; all single-draw passes while the survey runs.
std::atomic<uint64_t> gWatchedPass{0};
// The tonemap pass, while engine signals run: its constants carry exposure.
std::atomic<uint64_t> gWatchedTonemap{0};
// The pass that draws the sun's shadow map: its constants hold the light's matrix.
std::atomic<uint64_t> gWatchedShadow{0};
// Survey only: shader code by module, and each graphics pipeline's fragment shader hash.
std::unordered_map<VkShaderModule, uint32_t> gModuleHashes;
std::unordered_map<VkPipeline, uint32_t> gFragmentOf;
// Images bound to each optical-flow session, by VkOpticalFlowSessionBindingPointNV.
std::unordered_map<VkOpticalFlowSessionNV, std::map<int, ImageId>> gFlowBindings;
std::unordered_map<VkSwapchainKHR, Swapchain> gSwapchains;
// The work submitted since the last present.
std::vector<Step> gCollected;

// Host mappings, for the projection scan. gMapLock keeps a mapping alive while it is read.
struct Mapping {
    const uint8_t* ptr = nullptr;
    VkDeviceSize size = 0;
    uint32_t type = 0;
    VkDeviceSize offset = 0;  // in its memory
};
std::mutex gMapLock;
std::unordered_map<VkDeviceMemory, VkDeviceSize> gMemorySizes;
std::unordered_map<VkDeviceMemory, Mapping> gMappings;

std::atomic<uint32_t> gCalls[7] = {};
enum CallFamily { kRenderPass, kRenderPass2, kRendering, kDraw, kSubmit, kSubmit2, kExecute };

void tally(CallFamily c) {
    gCalls[c].fetch_add(1, std::memory_order_relaxed);
}

thread_local VkCommandBuffer tCb = VK_NULL_HANDLE;
thread_local Record* tRecord = nullptr;

Record& recordOf(VkCommandBuffer cb) {
    if (cb == tCb) return *tRecord;
    std::lock_guard lock(gLock);
    std::unique_ptr<Record>& r = gRecords[cb];
    if (!r) r = std::make_unique<Record>();
    tCb = cb;
    tRecord = r.get();
    return *r;
}

// Callers hold gLock.
ImageId idOfImage(VkImage image) {
    const auto i = gImageIds.find(image);
    return i == gImageIds.end() ? 0 : i->second;
}

ImageId idOfView(VkImageView view) {
    const auto v = gViews.find(view);
    return v == gViews.end() ? 0 : idOfImage(v->second);
}

Image* imageOf(ImageId id) {
    const auto i = gImages.find(id);
    return i == gImages.end() ? nullptr : &i->second;
}

bool readsImage(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT || t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
}

bool isUniform(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
}

bool isDynamic(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
            t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}

// Callers hold gLock.
void noteBuffer(VkDescriptorSet set, uint32_t binding, uint32_t element,
                const VkDescriptorBufferInfo& info, VkDescriptorType type) {
    gSetBuffers[set][static_cast<uint64_t>(binding) << 32 | element] = {
            info.buffer, info.offset, info.range, isDynamic(type)};
}

// Callers hold gLock.
void noteSampled(VkDescriptorSet set, uint32_t binding, uint32_t element, VkImageView view,
                 VkDescriptorType type) {
    gSetViews[set][static_cast<uint64_t>(binding) << 32 | element] = view;
    const ImageId id = idOfView(view);
    if (id == 0) return;
    gSetImages[set][static_cast<uint64_t>(binding) << 32 | element] = id;
    const auto swizzle = gViewSwizzles.find(view);
    if (swizzle != gViewSwizzles.end()) gSwizzles[id] = swizzle->second;
    Image* image = imageOf(id);
    if (image == nullptr || (!IsDepthFormat(image->format) && !IsDepthLikeFormat(image->format))) {
        return;
    }
    image->sampledAs |= 1u << std::min<uint32_t>(type, 31);
    image->descriptorWrites++;
}

ImageId newImage(VkImage handle, Image image) {
    image.id = gNextId++;
    gImageIds[handle] = image.id;
    gHandles[image.id] = handle;
    gImages[image.id] = image;
    return image.id;
}

uint64_t handleBits(VkRenderPass pass) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pass));
}

// Callers hold gLock. Drops the split's scene image, which outlives any command buffer that used
// it, and disarms it.
// Callers hold gLock.
void retireGlassImage(Split& s) {
    if (s.scene != VK_NULL_HANDLE) {
        gImages.erase(s.sceneId);
        gHandles.erase(s.sceneId);
        gImageIds.erase(s.scene);
        gSpent.push_back({s.device, VK_NULL_HANDLE, s.scene, s.memory});
    }
    s = Split();
}

// Callers hold gLock.
void retireGlass() {
    retireGlassImage(gGlass.opaque);
    retireGlassImage(gGlass.lit);
    gGlass = Glass();
}

void retireSplit() {
    if (gSplit.scene != VK_NULL_HANDLE) {
        gImages.erase(gSplit.sceneId);
        gHandles.erase(gSplit.sceneId);
        gImageIds.erase(gSplit.scene);
        gSpent.push_back({gSplit.device, VK_NULL_HANDLE, gSplit.scene, gSplit.memory});
    }
    gSplit = Split();
}

// Callers hold gLock. Drops the kept depth's image, as retireSplit drops the split's.
void retireKeep() {
    if (gKeep.image != VK_NULL_HANDLE) {
        gImages.erase(gKeep.id);
        gHandles.erase(gKeep.id);
        gImageIds.erase(gKeep.image);
        gSpent.push_back({gKeep.device, VK_NULL_HANDLE, gKeep.image, gKeep.memory});
    }
    gKeep = Keep();
}

// Callers hold gLock. The game made a pass with a handle a split knew: it is another pass now.
void forgetPass(VkRenderPass pass) {
    if (gKeep.pass == pass) gKeep.pass = VK_NULL_HANDLE;
    const auto resume = gResumes.find(pass);
    if (resume != gResumes.end()) {
        gSpent.push_back({resume->second.first, resume->second.second, VK_NULL_HANDLE,
                          VK_NULL_HANDLE});
        gResumes.erase(resume);
    }
    if (gSplit.pass == pass) {
        gSplit.pass = VK_NULL_HANDLE;
        gSplit.resume = VK_NULL_HANDLE;
    }
    if (gGlass.cutPass == pass || gGlass.litPass == pass) {
        retireGlass();
    } else if (gGlass.refused == handleBits(pass)) {
        gGlass.refused = 0;
    }
}

VkImageAspectFlags aspectsOf(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT;
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

VkAttachmentReference2 widen(const VkAttachmentReference& r, VkImageAspectFlags aspects) {
    return {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, r.attachment, r.layout, aspects};
}

// ---- Wrappers --------------------------------------------------------------------------------

VKAPI_ATTR VkResult CreateImage(VkDevice device, const VkImageCreateInfo* info,
                                const VkAllocationCallbacks* allocator, VkImage* image) {
    const VkResult r = gNext.CreateImage(device, info, allocator, image);
    if (r != VK_SUCCESS) return r;
    Image i;
    i.format = info->format;
    i.width = info->extent.width;
    i.height = info->extent.height;
    i.mips = info->mipLevels;
    i.layers = info->arrayLayers;
    i.samples = info->samples;
    i.usage = info->usage;
    std::lock_guard lock(gLock);
    newImage(*image, i);
    return r;
}

VKAPI_ATTR void DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(gLock);
        const auto found = gImageIds.find(image);
        if (found != gImageIds.end()) {
            gImages.erase(found->second);
            gHandles.erase(found->second);
            gImageIds.erase(found);
        }
    }
    gNext.DestroyImage(device, image, allocator);
}

VKAPI_ATTR VkResult AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                                   const VkAllocationCallbacks* allocator, VkDeviceMemory* memory) {
    const VkResult r = gNext.AllocateMemory(device, info, allocator, memory);
    if (r == VK_SUCCESS) {
        {
            std::lock_guard lock(gLock);
            gMemoryTypes[*memory] = {device, info->memoryTypeIndex};
        }
        std::lock_guard lock(gMapLock);
        gMemorySizes[*memory] = info->allocationSize;
    }
    return r;
}

void noteMapped(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, void* ptr) {
    uint32_t type = 0;
    {
        std::lock_guard lock(gLock);
        const auto t = gMemoryTypes.find(memory);
        if (t != gMemoryTypes.end()) type = t->second.second;
    }
    std::lock_guard lock(gMapLock);
    if (size == VK_WHOLE_SIZE) {
        const auto total = gMemorySizes.find(memory);
        size = total == gMemorySizes.end() || total->second < offset ? 0 : total->second - offset;
    }
    gMappings[memory] = {static_cast<const uint8_t*>(ptr), size, type, offset};
}

VKAPI_ATTR VkResult MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                              VkDeviceSize size, VkMemoryMapFlags flags, void** data) {
    const VkResult r = gNext.MapMemory(device, memory, offset, size, flags, data);
    if (r == VK_SUCCESS) noteMapped(memory, offset, size, *data);
    return r;
}

VKAPI_ATTR VkResult MapMemory2(VkDevice device, const VkMemoryMapInfo* info, void** data) {
    const VkResult r = gNext.MapMemory2(device, info, data);
    if (r == VK_SUCCESS) noteMapped(info->memory, info->offset, info->size, *data);
    return r;
}

VKAPI_ATTR void UnmapMemory(VkDevice device, VkDeviceMemory memory) {
    {
        std::lock_guard lock(gMapLock);
        gMappings.erase(memory);
    }
    gNext.UnmapMemory(device, memory);
}

VKAPI_ATTR VkResult UnmapMemory2(VkDevice device, const VkMemoryUnmapInfo* info) {
    {
        std::lock_guard lock(gMapLock);
        gMappings.erase(info->memory);
    }
    return gNext.UnmapMemory2(device, info);
}

VKAPI_ATTR void FreeMemory(VkDevice device, VkDeviceMemory memory,
                           const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(gMapLock);
        gMappings.erase(memory);
        gMemorySizes.erase(memory);
    }
    {
        std::lock_guard lock(gLock);
        gMemoryTypes.erase(memory);
    }
    gNext.FreeMemory(device, memory, allocator);
}

void noteBound(VkImage image, VkDeviceMemory memory) {
    std::lock_guard lock(gLock);
    Image* i = imageOf(idOfImage(image));
    const auto m = gMemoryTypes.find(memory);
    if (i == nullptr || m == gMemoryTypes.end()) return;
    i->memoryType = static_cast<int>(m->second.second);
    const auto d = gDevices.find(m->second.first);
    if (d != gDevices.end() && m->second.second < d->second.memory.memoryTypeCount) {
        i->memory = d->second.memory.memoryTypes[m->second.second].propertyFlags;
    }
}

VKAPI_ATTR VkResult BindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory,
                                    VkDeviceSize offset) {
    const VkResult r = gNext.BindImageMemory(device, image, memory, offset);
    if (r == VK_SUCCESS) noteBound(image, memory);
    return r;
}

VKAPI_ATTR VkResult BindImageMemory2(VkDevice device, uint32_t count,
                                     const VkBindImageMemoryInfo* infos) {
    const VkResult r = gNext.BindImageMemory2(device, count, infos);
    if (r == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) noteBound(infos[i].image, infos[i].memory);
    }
    return r;
}

VKAPI_ATTR VkResult CreateBuffer(VkDevice device, const VkBufferCreateInfo* info,
                                 const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
    const VkResult r = gNext.CreateBuffer(device, info, allocator, buffer);
    constexpr VkBufferUsageFlags kRead = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (r == VK_SUCCESS && (info->usage & kRead) != 0) {
        std::lock_guard lock(gLock);
        gBuffers[*buffer] = {info->size, VK_NULL_HANDLE, 0, info->usage};
    }
    return r;
}

VKAPI_ATTR void DestroyBuffer(VkDevice device, VkBuffer buffer,
                              const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(gLock);
        gBuffers.erase(buffer);
    }
    gNext.DestroyBuffer(device, buffer, allocator);
}

void noteBufferBound(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
    std::lock_guard lock(gLock);
    const auto found = gBuffers.find(buffer);
    if (found == gBuffers.end()) return;
    found->second.memory = memory;
    found->second.offset = offset;
}

VKAPI_ATTR VkResult BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory,
                                     VkDeviceSize offset) {
    const VkResult r = gNext.BindBufferMemory(device, buffer, memory, offset);
    if (r == VK_SUCCESS) noteBufferBound(buffer, memory, offset);
    return r;
}

VKAPI_ATTR VkResult BindBufferMemory2(VkDevice device, uint32_t count,
                                      const VkBindBufferMemoryInfo* infos) {
    const VkResult r = gNext.BindBufferMemory2(device, count, infos);
    if (r == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) {
            noteBufferBound(infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
        }
    }
    return r;
}

// Survey only: every shader's SPIR-V goes to the app's cache as framefx-shader-<hash>.spv, so the
// constants a pass reads can be matched with the code that reads them.
VKAPI_ATTR VkResult CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* info,
                                       const VkAllocationCallbacks* allocator,
                                       VkShaderModule* module) {
    const VkResult r = gNext.CreateShaderModule(device, info, allocator, module);
    if (r != VK_SUCCESS) return r;
    // Hide HUD learns per fragment shader, so every module is hashed; the survey also keeps it.
    uint32_t hash = 0x811c9dc5u;
    const auto* bytes = reinterpret_cast<const uint8_t*>(info->pCode);
    for (size_t i = 0; i < info->codeSize; i++) hash = (hash ^ bytes[i]) * 0x01000193u;
    if (!survey::enabled()) {
        std::lock_guard lock(gLock);
        gModuleHashes[*module] = hash;
        return r;
    }
    const std::string path = StringPrintf("/data/user/%u/%s/cache/framefx-shader-%08x.spv",
                                          getuid() / 100000,
                                          Engine::get().package().substr(
                                                  0, Engine::get().package().find(':')).c_str(),
                                          hash);
    if (access(path.c_str(), F_OK) != 0) {
        android::base::WriteStringToFile(std::string(reinterpret_cast<const char*>(bytes),
                                                     info->codeSize),
                                         path);
    }
    std::lock_guard lock(gLock);
    gModuleHashes[*module] = hash;
    return r;
}

VKAPI_ATTR VkResult CreateDescriptorSetLayout(VkDevice device,
                                              const VkDescriptorSetLayoutCreateInfo* info,
                                              const VkAllocationCallbacks* allocator,
                                              VkDescriptorSetLayout* layout) {
    const VkResult r = gNext.CreateDescriptorSetLayout(device, info, allocator, layout);
    if (r != VK_SUCCESS) return r;
    SetLayoutDesc desc;
    for (uint32_t i = 0; i < info->bindingCount; i++) {
        const VkDescriptorSetLayoutBinding& b = info->pBindings[i];
        if (alihOn()) {
            desc.bindings.push_back({b.binding, b.descriptorType, b.descriptorCount, b.stageFlags,
                                     nullptr});
        }
        if (!isDynamic(b.descriptorType)) continue;
        for (uint32_t k = 0; k < b.descriptorCount; k++) {
            desc.dynamicBindings.push_back(b.binding << 16 | k);
        }
    }
    std::sort(desc.dynamicBindings.begin(), desc.dynamicBindings.end());
    std::lock_guard lock(gLock);
    gSetLayouts[*layout] = std::move(desc);
    return r;
}

VKAPI_ATTR VkResult AllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* info,
                                           VkDescriptorSet* sets) {
    const VkResult r = gNext.AllocateDescriptorSets(device, info, sets);
    if (r == VK_SUCCESS) {
        std::lock_guard lock(gLock);
        for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
            gSetLayoutOf[sets[i]] = info->pSetLayouts[i];
            gSetBuffers.erase(sets[i]);
            gAlihClones.erase(sets[i]);
        }
    }
    return r;
}

VKAPI_ATTR VkResult CreateImageView(VkDevice device, const VkImageViewCreateInfo* info,
                                    const VkAllocationCallbacks* allocator, VkImageView* view) {
    const VkResult r = gNext.CreateImageView(device, info, allocator, view);
    if (r == VK_SUCCESS) {
        std::lock_guard lock(gLock);
        gViews[*view] = info->image;
        if (alihOn() && info->format == VK_FORMAT_R8_UNORM) {
            gViewSwizzles[*view] = info->components;
        }
    }
    return r;
}

VKAPI_ATTR void DestroyImageView(VkDevice device, VkImageView view,
                                 const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(gLock);
        gViews.erase(view);
    }
    gNext.DestroyImageView(device, view, allocator);
}

VKAPI_ATTR VkResult CreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* info,
                                     const VkAllocationCallbacks* allocator, VkRenderPass* pass) {
    const VkResult r = gNext.CreateRenderPass(device, info, allocator, pass);
    if (r != VK_SUCCESS) return r;
    RenderPassDesc d;
    for (uint32_t i = 0; i < info->attachmentCount; i++) {
        const VkAttachmentDescription& a = info->pAttachments[i];
        d.attachments.push_back({a.format, a.samples, a.loadOp, a.storeOp, a.stencilLoadOp,
                                 a.stencilStoreOp, a.finalLayout});
    }
    d.subpasses = info->subpassCount;
    for (uint32_t s = 0; s < info->subpassCount; s++) {
        const VkSubpassDescription& sub = info->pSubpasses[s];
        for (uint32_t k = 0; k < sub.inputAttachmentCount; k++) {
            d.inputs.push_back(sub.pInputAttachments[k].attachment);
        }
    }
    if (info->pNext == nullptr && info->subpassCount == 1 &&
        info->pSubpasses[0].pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        auto recipe = std::make_shared<PassRecipe>();
        for (uint32_t i = 0; i < info->attachmentCount; i++) {
            const VkAttachmentDescription& a = info->pAttachments[i];
            recipe->attachments.push_back({VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, nullptr,
                                           a.flags, a.format, a.samples, a.loadOp, a.storeOp,
                                           a.stencilLoadOp, a.stencilStoreOp, a.initialLayout,
                                           a.finalLayout});
        }
        const VkSubpassDescription& sub = info->pSubpasses[0];
        const auto aspects = [&](uint32_t a) {
            return a < info->attachmentCount ? aspectsOf(info->pAttachments[a].format)
                                             : VK_IMAGE_ASPECT_COLOR_BIT;
        };
        recipe->flags = sub.flags;
        for (uint32_t k = 0; k < sub.colorAttachmentCount; k++) {
            recipe->colors.push_back(widen(sub.pColorAttachments[k], VK_IMAGE_ASPECT_COLOR_BIT));
            if (sub.pResolveAttachments != nullptr) {
                recipe->resolves.push_back(
                        widen(sub.pResolveAttachments[k], VK_IMAGE_ASPECT_COLOR_BIT));
            }
        }
        for (uint32_t k = 0; k < sub.inputAttachmentCount; k++) {
            const VkAttachmentReference& in = sub.pInputAttachments[k];
            recipe->inputs.push_back(widen(in, aspects(in.attachment)));
        }
        if (sub.pDepthStencilAttachment != nullptr) {
            recipe->hasDepth = true;
            recipe->depth = widen(*sub.pDepthStencilAttachment,
                                  aspects(sub.pDepthStencilAttachment->attachment));
        }
        recipe->preserve.assign(sub.pPreserveAttachments,
                                sub.pPreserveAttachments + sub.preserveAttachmentCount);
        for (uint32_t k = 0; k < info->dependencyCount; k++) {
            const VkSubpassDependency& dep = info->pDependencies[k];
            recipe->dependencies.push_back({VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2, nullptr,
                                            dep.srcSubpass, dep.dstSubpass, dep.srcStageMask,
                                            dep.dstStageMask, dep.srcAccessMask, dep.dstAccessMask,
                                            dep.dependencyFlags, 0});
        }
        d.recipe = std::move(recipe);
    }
    std::lock_guard lock(gLock);
    forgetPass(*pass);
    gRenderPasses[*pass] = std::move(d);
    return r;
}

VKAPI_ATTR VkResult CreateRenderPass2(VkDevice device, const VkRenderPassCreateInfo2* info,
                                      const VkAllocationCallbacks* allocator, VkRenderPass* pass) {
    const VkResult r = gNext.CreateRenderPass2(device, info, allocator, pass);
    if (r != VK_SUCCESS) return r;
    RenderPassDesc d;
    for (uint32_t i = 0; i < info->attachmentCount; i++) {
        const VkAttachmentDescription2& a = info->pAttachments[i];
        d.attachments.push_back({a.format, a.samples, a.loadOp, a.storeOp, a.stencilLoadOp,
                                 a.stencilStoreOp, a.finalLayout});
    }
    d.subpasses = info->subpassCount;
    for (uint32_t s = 0; s < info->subpassCount; s++) {
        const VkSubpassDescription2& sub = info->pSubpasses[s];
        for (uint32_t k = 0; k < sub.inputAttachmentCount; k++) {
            d.inputs.push_back(sub.pInputAttachments[k].attachment);
        }
        for (auto* p = static_cast<const VkBaseInStructure*>(sub.pNext); p != nullptr; p = p->pNext) {
            if (p->sType == VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE) {
                d.depthResolve = true;
            }
        }
    }
    const VkSubpassDescription2* sub = info->subpassCount == 1 ? info->pSubpasses : nullptr;
    bool plain = info->pNext == nullptr && info->correlatedViewMaskCount == 0 && sub != nullptr &&
            sub->pNext == nullptr && sub->viewMask == 0 &&
            sub->pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS;
    for (uint32_t i = 0; plain && i < info->attachmentCount; i++) {
        plain = info->pAttachments[i].pNext == nullptr;
    }
    for (uint32_t k = 0; plain && k < info->dependencyCount; k++) {
        plain = info->pDependencies[k].pNext == nullptr;
    }
    if (plain) {
        auto recipe = std::make_shared<PassRecipe>();
        recipe->attachments.assign(info->pAttachments, info->pAttachments + info->attachmentCount);
        recipe->flags = sub->flags;
        const auto refs = [](const VkAttachmentReference2* from, uint32_t n,
                             std::vector<VkAttachmentReference2>* into) {
            if (from == nullptr) return true;
            for (uint32_t k = 0; k < n; k++) {
                if (from[k].pNext != nullptr) return false;
                into->push_back(from[k]);
            }
            return true;
        };
        plain = refs(sub->pColorAttachments, sub->colorAttachmentCount, &recipe->colors) &&
                refs(sub->pResolveAttachments, sub->colorAttachmentCount, &recipe->resolves) &&
                refs(sub->pInputAttachments, sub->inputAttachmentCount, &recipe->inputs);
        if (sub->pDepthStencilAttachment != nullptr) {
            recipe->hasDepth = true;
            recipe->depth = *sub->pDepthStencilAttachment;
            plain = plain && recipe->depth.pNext == nullptr;
        }
        recipe->preserve.assign(sub->pPreserveAttachments,
                                sub->pPreserveAttachments + sub->preserveAttachmentCount);
        recipe->dependencies.assign(info->pDependencies,
                                    info->pDependencies + info->dependencyCount);
        if (plain) d.recipe = std::move(recipe);
    }
    std::lock_guard lock(gLock);
    forgetPass(*pass);
    gRenderPasses[*pass] = std::move(d);
    return r;
}

VKAPI_ATTR VkResult CreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* info,
                                      const VkAllocationCallbacks* allocator, VkFramebuffer* fb) {
    const VkResult r = gNext.CreateFramebuffer(device, info, allocator, fb);
    if (r != VK_SUCCESS) return r;
    std::vector<VkImageView> views;
    if (!(info->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT)) {
        views.assign(info->pAttachments, info->pAttachments + info->attachmentCount);
    }
    std::lock_guard lock(gLock);
    gFramebuffers[*fb] = std::move(views);
    return r;
}

VKAPI_ATTR VkResult CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                            const VkGraphicsPipelineCreateInfo* infos,
                                            const VkAllocationCallbacks* allocator,
                                            VkPipeline* pipelines) {
    const VkResult r =
            gNext.CreateGraphicsPipelines(device, cache, count, infos, allocator, pipelines);
    std::lock_guard lock(gLock);
    for (uint32_t i = 0; i < count; i++) {
        if (pipelines[i] == VK_NULL_HANDLE) continue;
        PipelineDesc p;
        if (const VkPipelineVertexInputStateCreateInfo* in = infos[i].pVertexInputState) {
            auto layout = std::make_shared<VertexLayout>();
            layout->bindings.assign(in->pVertexBindingDescriptions,
                                    in->pVertexBindingDescriptions + in->vertexBindingDescriptionCount);
            layout->attributes.assign(in->pVertexAttributeDescriptions,
                                      in->pVertexAttributeDescriptions +
                                              in->vertexAttributeDescriptionCount);
            gVertexLayouts[pipelines[i]] = std::move(layout);
        }
        for (uint32_t k = 0; k < infos[i].stageCount; k++) {
            if (infos[i].pStages[k].stage != VK_SHADER_STAGE_FRAGMENT_BIT) continue;
            const auto hash = gModuleHashes.find(infos[i].pStages[k].module);
            if (hash == gModuleHashes.end()) continue;
            p.fragment = hash->second;
            if (survey::enabled()) gFragmentOf[pipelines[i]] = hash->second;
        }
        // ---- Hide HUD: what the vertex layout says (UI quads: 2D positions, byte colours).
        if (const VkPipelineVertexInputStateCreateInfo* in = infos[i].pVertexInputState) {
            const uint32_t n = in->vertexAttributeDescriptionCount;
            p.vertex = n == 0 ? kHudVertexNone : static_cast<uint8_t>(std::min<uint32_t>(n, kHudVertexAttributes));
            for (uint32_t k = 0; k < n; k++) {
                const VkVertexInputAttributeDescription& a = in->pVertexAttributeDescriptions[k];
                if (a.format == VK_FORMAT_R8G8B8A8_UNORM || a.format == VK_FORMAT_B8G8R8A8_UNORM) {
                    p.vertex |= kHudVertexColour8;
                }
                if (a.location == 0 &&
                    (a.format == VK_FORMAT_R32G32_SFLOAT || a.format == VK_FORMAT_R16G16_SFLOAT)) {
                    p.vertex |= kHudVertex2d;
                }
            }
        }
        const VkPipelineViewportStateCreateInfo* vp = infos[i].pViewportState;
        if (vp != nullptr && vp->viewportCount > 0 && vp->pViewports != nullptr) {
            const VkViewport& v = vp->pViewports[0];
            p.viewportHeight = v.height;
            p.viewportX = v.x;
            p.viewportY = v.y;
            p.viewportWidth = v.width;
        }
        if (const VkPipelineDepthStencilStateCreateInfo* ds = infos[i].pDepthStencilState) {
            p.test = ds->depthTestEnable;
            p.write = ds->depthTestEnable && ds->depthWriteEnable;
            p.compare = ds->depthCompareOp;
            p.stencilWrite = ds->stencilTestEnable && ds->front.writeMask != 0 &&
                    ds->front.passOp == VK_STENCIL_OP_REPLACE;
        }
        if (const VkPipelineDynamicStateCreateInfo* dyn = infos[i].pDynamicState) {
            for (uint32_t k = 0; k < dyn->dynamicStateCount; k++) {
                if (dyn->pDynamicStates[k] == VK_DYNAMIC_STATE_VIEWPORT) p.viewportHeight = 0;
                if (dyn->pDynamicStates[k] == VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE) p.dynamicTest = true;
                if (dyn->pDynamicStates[k] == VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE) p.dynamicWrite = true;
                if (dyn->pDynamicStates[k] == VK_DYNAMIC_STATE_DEPTH_COMPARE_OP) p.dynamicCompare = true;
            }
        }
        if (const VkPipelineColorBlendStateCreateInfo* cb = infos[i].pColorBlendState) {
            p.blend = cb->attachmentCount > 0 && cb->pAttachments != nullptr &&
                    cb->pAttachments[0].blendEnable;
        }
        gPipelines[pipelines[i]] = p;
    }
    return r;
}

VKAPI_ATTR void UpdateDescriptorSets(VkDevice device, uint32_t writeCount,
                                     const VkWriteDescriptorSet* writes, uint32_t copyCount,
                                     const VkCopyDescriptorSet* copies) {
    {
        std::lock_guard lock(gLock);
        for (uint32_t w = 0; w < writeCount; w++) {
            const VkWriteDescriptorSet& write = writes[w];
            gAlihClones.erase(write.dstSet);
            if (isUniform(write.descriptorType) && write.pBufferInfo != nullptr) {
                for (uint32_t k = 0; k < write.descriptorCount; k++) {
                    noteBuffer(write.dstSet, write.dstBinding, write.dstArrayElement + k,
                               write.pBufferInfo[k], write.descriptorType);
                }
                continue;
            }
            if (!readsImage(write.descriptorType) || write.pImageInfo == nullptr) continue;
            for (uint32_t k = 0; k < write.descriptorCount; k++) {
                noteSampled(write.dstSet, write.dstBinding, write.dstArrayElement + k,
                            write.pImageInfo[k].imageView, write.descriptorType);
            }
        }
        for (uint32_t c = 0; c < copyCount; c++) {
            const VkCopyDescriptorSet& copy = copies[c];
            const auto from = gSetImages.find(copy.srcSet);
            if (from == gSetImages.end()) continue;
            for (uint32_t k = 0; k < copy.descriptorCount; k++) {
                const auto id = from->second.find(static_cast<uint64_t>(copy.srcBinding) << 32 |
                                                  (copy.srcArrayElement + k));
                if (id == from->second.end()) continue;
                gSetImages[copy.dstSet][static_cast<uint64_t>(copy.dstBinding) << 32 |
                                        (copy.dstArrayElement + k)] = id->second;
            }
            const auto views = gSetViews.find(copy.srcSet);
            if (views == gSetViews.end()) continue;
            for (uint32_t k = 0; k < copy.descriptorCount; k++) {
                const auto view = views->second.find(static_cast<uint64_t>(copy.srcBinding) << 32 |
                                                     (copy.srcArrayElement + k));
                if (view == views->second.end()) continue;
                gSetViews[copy.dstSet][static_cast<uint64_t>(copy.dstBinding) << 32 |
                                       (copy.dstArrayElement + k)] = view->second;
            }
        }
    }
    gNext.UpdateDescriptorSets(device, writeCount, writes, copyCount, copies);
}

VKAPI_ATTR VkResult CreateDescriptorUpdateTemplate(VkDevice device,
                                                   const VkDescriptorUpdateTemplateCreateInfo* info,
                                                   const VkAllocationCallbacks* allocator,
                                                   VkDescriptorUpdateTemplate* t) {
    const VkResult r = gNext.CreateDescriptorUpdateTemplate(device, info, allocator, t);
    if (r == VK_SUCCESS) {
        std::lock_guard lock(gLock);
        gTemplates[*t].assign(info->pDescriptorUpdateEntries,
                              info->pDescriptorUpdateEntries + info->descriptorUpdateEntryCount);
    }
    return r;
}

VKAPI_ATTR void UpdateDescriptorSetWithTemplate(VkDevice device, VkDescriptorSet set,
                                                VkDescriptorUpdateTemplate t, const void* data) {
    {
        std::lock_guard lock(gLock);
        gAlihClones.erase(set);
        const auto found = gTemplates.find(t);
        if (found != gTemplates.end() && data != nullptr) {
            for (const VkDescriptorUpdateTemplateEntry& e : found->second) {
                if (isUniform(e.descriptorType)) {
                    for (uint32_t k = 0; k < e.descriptorCount; k++) {
                        const auto* info = reinterpret_cast<const VkDescriptorBufferInfo*>(
                                static_cast<const uint8_t*>(data) + e.offset + k * e.stride);
                        noteBuffer(set, e.dstBinding, e.dstArrayElement + k, *info,
                                   e.descriptorType);
                    }
                    continue;
                }
                if (!readsImage(e.descriptorType)) continue;
                for (uint32_t k = 0; k < e.descriptorCount; k++) {
                    const auto* info = reinterpret_cast<const VkDescriptorImageInfo*>(
                            static_cast<const uint8_t*>(data) + e.offset + k * e.stride);
                    noteSampled(set, e.dstBinding, e.dstArrayElement + k, info->imageView,
                                e.descriptorType);
                }
            }
        }
    }
    gNext.UpdateDescriptorSetWithTemplate(device, set, t, data);
}

void sampleSoon();

VKAPI_ATTR VkResult BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* info) {
    recordOf(cb).clear();
    if (signalsOn()) sampleSoon();
    return gNext.BeginCommandBuffer(cb, info);
}

Pass& current(Record& r) {
    return r.open >= 0 ? r.events[r.open].pass : r.loose;
}

std::vector<VkDescriptorSet>& currentSets(Record& r) {
    return r.open >= 0 ? r.sets[r.open] : r.looseSets;
}

void addSets(std::vector<VkDescriptorSet>* into, const VkDescriptorSet* sets, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (sets[i] != VK_NULL_HANDLE &&
            std::find(into->begin(), into->end(), sets[i]) == into->end()) {
            into->push_back(sets[i]);
        }
    }
}

VKAPI_ATTR void CmdExecuteCommands(VkCommandBuffer cb, uint32_t count,
                                   const VkCommandBuffer* secondaries) {
    tally(kExecute);
    Record& r = recordOf(cb);
    for (uint32_t i = 0; i < count && !r.ignored; i++) {
        Record* s;
        {
            std::lock_guard lock(gLock);
            const auto found = gRecords.find(secondaries[i]);
            s = found == gRecords.end() ? nullptr : found->second.get();
        }
        if (s == nullptr) continue;
        Pass& into = current(r);
        into.draws += s->loose.draws;
        into.depthTests += s->loose.depthTests;
        into.depthWrites += s->loose.depthWrites;
        addSets(&currentSets(r), s->looseSets.data(), s->looseSets.size());
        for (size_t e = 0; e < s->events.size(); e++) {
            if (!r.push(s->events[e])) break;
            r.sets.back() = s->sets[e];
        }
    }
    gNext.CmdExecuteCommands(cb, count, secondaries);
}

void openPass(Record& r, Step step) {
    r.pastScene = false;
    if (!r.push(std::move(step))) {
        r.open = -1;
        return;
    }
    r.open = static_cast<int>(r.events.size()) - 1;
    r.takenVersion = UINT32_MAX;
}

// Whether a split can begin the pass again with what `info` chains: nothing, or the attachments
// of an imageless framebuffer.
bool chainFits(const VkRenderPassBeginInfo* info) {
    const auto* n = static_cast<const VkBaseInStructure*>(info->pNext);
    return n == nullptr || (n->sType == VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO &&
                            n->pNext == nullptr);
}

void beginPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* info, bool v2,
               VkSubpassContents contents) {
    Record& r = recordOf(cb);
    Step e;
    e.isPass = true;
    Pass& p = e.pass;
    p.renderPass = handleBits(info->renderPass);
    p.width = info->renderArea.extent.width;
    p.height = info->renderArea.extent.height;
    VkImage cut = VK_NULL_HANDLE;
    ImageId cutId = 0;
    VkImage glassCut = VK_NULL_HANDLE;
    ImageId glassCutId = 0;
    VkImage glassTarget = VK_NULL_HANDLE;
    VkImageLayout glassFinal = VK_IMAGE_LAYOUT_UNDEFINED;
    bool glassOpaqueAtEnd = false;
    bool glassLitAtEnd = false;
    {
        std::lock_guard lock(gLock);
        const auto rp = gRenderPasses.find(info->renderPass);
        if (rp != gRenderPasses.end()) {
            const RenderPassDesc& d = rp->second;
            p.subpasses = d.subpasses;
            p.depthResolve = d.depthResolve;
            std::vector<VkImageView> views;
            const auto fb = gFramebuffers.find(info->framebuffer);
            if (fb != gFramebuffers.end()) views = fb->second;
            for (auto* n = static_cast<const VkBaseInStructure*>(info->pNext); n != nullptr;
                 n = n->pNext) {
                if (n->sType == VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO) {
                    const auto* a = reinterpret_cast<const VkRenderPassAttachmentBeginInfo*>(n);
                    views.assign(a->pAttachments, a->pAttachments + a->attachmentCount);
                }
            }
            for (uint32_t i = 0; i < d.attachments.size(); i++) {
                const AttachmentDesc& a = d.attachments[i];
                Target t;
                t.image = i < views.size() ? idOfView(views[i]) : 0;
                t.attachment = i;
                t.format = a.format;
                t.load = a.load;
                t.store = a.store;
                t.final = a.final;
                t.samples = a.samples;
                if (IsDepthFormat(a.format) && !p.hasDepth) {
                    p.hasDepth = true;
                    p.depth = t;
                    p.stencilLoad = a.stencilLoad;
                    p.stencilStore = a.stencilStore;
                    if (a.load == VK_ATTACHMENT_LOAD_OP_CLEAR && i < info->clearValueCount) {
                        p.clearDepth = info->pClearValues[i].depthStencil.depth;
                    }
                    for (uint32_t in : d.inputs) p.depthInput |= in == i;
                } else {
                    p.colors.push_back(t);
                }
            }
            // The transparency cut's passes, drawn into an image its copies fit.
            const bool glassCutPass = info->renderPass == gGlass.cutPass;
            const bool glassLitPass = info->renderPass == gGlass.litPass;
            const uint32_t glassAt = glassCutPass ? gGlass.opaque.attachment : gGlass.litAttachment;
            if ((glassCutPass || glassLitPass) && glassAt < views.size()) {
                const ImageId id = idOfView(views[glassAt]);
                const Image* image = imageOf(id);
                const auto handle = gHandles.find(id);
                if (image != nullptr && handle != gHandles.end() &&
                    image->format == gGlass.opaque.format && image->width == gGlass.opaque.width &&
                    image->height == gGlass.opaque.height) {
                    const bool inPass = gGlass.kind == 1;
                    glassTarget = handle->second;
                    glassFinal = glassCutPass ? gGlass.opaque.final : gGlass.litFinal;
                    glassOpaqueAtEnd = !inPass && glassCutPass;
                    glassLitAtEnd = glassLitPass;
                    const auto view = gViews.find(views[glassAt]);
                    if (inPass && view != gViews.end() && chainFits(info) &&
                        contents == VK_SUBPASS_CONTENTS_INLINE &&
                        (!v2 || gGlass.opaque.fn.CmdBeginRenderPass2 != nullptr)) {
                        glassCut = view->second;
                        glassCutId = image->id;
                    }
                }
            }
            // The split pass, drawn inline into an image the split's copy fits.
            if (info->renderPass == gSplit.pass && chainFits(info) &&
                contents == VK_SUBPASS_CONTENTS_INLINE && gSplit.attachment < views.size() &&
                (!v2 || gSplit.fn.CmdBeginRenderPass2 != nullptr)) {
                const auto view = gViews.find(views[gSplit.attachment]);
                const Image* image = imageOf(idOfView(views[gSplit.attachment]));
                if (view != gViews.end() && image != nullptr && image->format == gSplit.format &&
                    image->width == gSplit.width && image->height == gSplit.height) {
                    cut = view->second;
                    cutId = image->id;
                }
            }
        }
    }
    openPass(r, std::move(e));
    r.glassTarget = glassTarget;
    r.glassFinal = glassFinal;
    r.glassOpaqueAtEnd = glassOpaqueAtEnd && r.open >= 0;
    r.glassLitAtEnd = glassLitAtEnd && r.open >= 0;
    r.glass.armed = glassCut != VK_NULL_HANDLE && r.open >= 0;
    if (r.glass.armed) {
        // The resumed half begins as the game began it, but loads what the first half stored.
        r.glass.v2 = v2;
        r.glass.begin = *info;
        r.glass.clears.assign(info->pClearValues, info->pClearValues + info->clearValueCount);
        r.glass.begin.clearValueCount = static_cast<uint32_t>(r.glass.clears.size());
        r.glass.begin.pClearValues = r.glass.clears.data();
        r.glass.views.clear();
        if (info->pNext != nullptr) {
            r.glass.attachments = *static_cast<const VkRenderPassAttachmentBeginInfo*>(info->pNext);
            r.glass.views.assign(r.glass.attachments.pAttachments,
                                 r.glass.attachments.pAttachments + r.glass.attachments.attachmentCount);
            r.glass.attachments.pAttachments = r.glass.views.data();
            r.glass.begin.pNext = &r.glass.attachments;
        }
        r.glass.target = glassCut;
        r.glass.targetId = glassCutId;
        r.glass.pass = r.events[r.open];
    }
    r.cut.armed = cut != VK_NULL_HANDLE && r.open >= 0;
    if (r.cut.armed) {
        r.cut.v2 = v2;
        r.cut.toned = false;
        r.cut.begin = *info;
        r.cut.clears.assign(info->pClearValues, info->pClearValues + info->clearValueCount);
        r.cut.begin.clearValueCount = static_cast<uint32_t>(r.cut.clears.size());
        r.cut.begin.pClearValues = r.cut.clears.data();
        r.cut.views.clear();
        if (info->pNext != nullptr) {
            r.cut.attachments = *static_cast<const VkRenderPassAttachmentBeginInfo*>(info->pNext);
            r.cut.views.assign(r.cut.attachments.pAttachments,
                               r.cut.attachments.pAttachments + r.cut.attachments.attachmentCount);
            r.cut.attachments.pAttachments = r.cut.views.data();
            r.cut.begin.pNext = &r.cut.attachments;
        }
        {
            std::lock_guard lock(gLock);
            r.cut.atClear = gSplit.atClear;
            r.cut.clearIndex = gSplit.clearIndex;
        }
        r.cut.clearsSeen = 0;
        r.cut.target = cut;
        r.cut.targetId = cutId;
        r.cut.pass = r.events[r.open];
    }
}

void alihUpload(VkCommandBuffer cb);

VKAPI_ATTR void CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* info,
                                   VkSubpassContents contents) {
    tally(kRenderPass);
    alihUpload(cb);
    beginPass(cb, info, false, contents);
    gNext.CmdBeginRenderPass(cb, info, contents);
}

VKAPI_ATTR void CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* info,
                                    const VkSubpassBeginInfo* begin) {
    tally(kRenderPass2);
    alihUpload(cb);
    beginPass(cb, info, true, begin->contents);
    gNext.CmdBeginRenderPass2(cb, info, begin);
}

// Whether the pass `r` has open is the one whose end leaves the depth to keep; the image then.
VkImage keptAtEnd(const Record& r, Keep* keep) {
    if (r.open < 0 || !r.events[r.open].isPass) return VK_NULL_HANDLE;
    const Pass& p = r.events[r.open].pass;
    std::lock_guard lock(gLock);
    if (gKeep.pass == VK_NULL_HANDLE || p.renderPass != handleBits(gKeep.pass)) return VK_NULL_HANDLE;
    if (!p.writes(gKeep.source)) return VK_NULL_HANDLE;
    const auto handle = gHandles.find(gKeep.source);
    if (handle == gHandles.end()) return VK_NULL_HANDLE;
    *keep = gKeep;
    return handle->second;
}

// Copies the depth the pass just ended left in `source` into the kept image, for the present.
void keepNow(VkCommandBuffer cb, Record& r, const Keep& k, VkImage source) {
    const Own& f = k.fn;
    const VkImageAspectFlags aspects = aspectsOf(k.format);
    VkImageMemoryBarrier b[2] = {};
    for (VkImageMemoryBarrier& one : b) {
        one.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        one.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        one.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        one.subresourceRange = {aspects, 0, 1, 0, 1};
    }
    b[0].image = source;
    b[0].oldLayout = k.final;
    b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    // Whatever read the last frame's copy came before on this queue.
    b[1].image = k.image;
    b[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 2, b);
    VkImageCopy copy = {};
    copy.srcSubresource = {aspects, 0, 0, 1};
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = {k.width, k.height, 1};
    f.CmdCopyImage(cb, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, k.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].newLayout = k.final;
    b[0].srcAccessMask = 0;
    b[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 2, b);
    Step copied;
    copied.op.kind = OpKind::kCopy;
    copied.op.src = k.source;
    copied.op.dst = k.id;
    r.push(std::move(copied));
    Step layout;
    layout.op.kind = OpKind::kLayout;
    layout.op.dst = k.id;
    layout.op.from = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    layout.op.to = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    r.push(std::move(layout));
    std::lock_guard lock(gLock);
    if (gKeep.image == k.image && !gKeep.logged) {
        gKeep.logged = true;
        ALOGI("lapis %s: keeping the scene's depth #%u as its pass ends, in #%u",
              Engine::get().package().c_str(), k.source, k.id);
    }
}

// ---- Hide HUD: the presented frame's low-resolution copy, as its pass ends ---------------------

// The image the open pass leaves for the cross-check's copy, once per frame; null if none.
VkImage hudProbeTarget(const Record& r) {
    if (!HudCapturing() || r.ignored || r.open < 0 || !r.events[r.open].isPass) return VK_NULL_HANDLE;
    const Pass& p = r.events[r.open].pass;
    const uint64_t pass = p.dynamic ? 0 : p.renderPass;
    for (const Target& t : p.colors) {
        if (!hud_probe::Wants(pass, t.format, p.width, p.height)) continue;
        std::lock_guard lock(gLock);
        const Image* i = imageOf(t.image);
        const auto handle = gHandles.find(t.image);
        if (i == nullptr || !(i->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || handle == gHandles.end()) {
            continue;
        }
        return handle->second;
    }
    return VK_NULL_HANDLE;
}

// ------------------------------------------------------------------------------------------------

void glassAtEnd(VkCommandBuffer cb, Record& r);

VKAPI_ATTR void CmdEndRenderPass(VkCommandBuffer cb) {
    Record& r = recordOf(cb);
    Keep keep;
    const VkImage kept = keptAtEnd(r, &keep);
    const VkImage probed = hudProbeTarget(r);
    r.open = -1;
    r.cut.armed = false;
    r.glass.armed = false;
    gNext.CmdEndRenderPass(cb);
    glassAtEnd(cb, r);
    if (kept != VK_NULL_HANDLE) keepNow(cb, r, keep, kept);
    if (probed != VK_NULL_HANDLE) hud_probe::Record(cb, probed);
}

VKAPI_ATTR void CmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo* end) {
    Record& r = recordOf(cb);
    Keep keep;
    const VkImage kept = keptAtEnd(r, &keep);
    const VkImage probed = hudProbeTarget(r);
    r.open = -1;
    r.cut.armed = false;
    r.glass.armed = false;
    gNext.CmdEndRenderPass2(cb, end);
    glassAtEnd(cb, r);
    if (kept != VK_NULL_HANDLE) keepNow(cb, r, keep, kept);
    if (probed != VK_NULL_HANDLE) hud_probe::Record(cb, probed);
}

// Callers hold gLock.
Target targetOf(const VkRenderingAttachmentInfo& a) {
    Target t;
    t.image = idOfView(a.imageView);
    if (const Image* i = imageOf(t.image)) {
        t.format = i->format;
        t.samples = i->samples;
    }
    t.load = a.loadOp;
    t.store = a.storeOp;
    t.final = a.imageLayout;
    return t;
}

VKAPI_ATTR void CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
    tally(kRendering);
    alihUpload(cb);
    Record& r = recordOf(cb);
    Step e;
    e.isPass = true;
    Pass& p = e.pass;
    p.dynamic = true;
    p.width = info->renderArea.extent.width;
    p.height = info->renderArea.extent.height;
    {
        std::lock_guard lock(gLock);
        for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
            if (info->pColorAttachments[i].imageView == VK_NULL_HANDLE) continue;
            p.colors.push_back(targetOf(info->pColorAttachments[i]));
        }
        if (info->pDepthAttachment != nullptr &&
            info->pDepthAttachment->imageView != VK_NULL_HANDLE) {
            p.hasDepth = true;
            p.depth = targetOf(*info->pDepthAttachment);
            if (info->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                p.clearDepth = info->pDepthAttachment->clearValue.depthStencil.depth;
            }
            p.depthResolve = info->pDepthAttachment->resolveMode != VK_RESOLVE_MODE_NONE;
        }
        if (info->pStencilAttachment != nullptr &&
            info->pStencilAttachment->imageView != VK_NULL_HANDLE) {
            p.stencilLoad = info->pStencilAttachment->loadOp;
            p.stencilStore = info->pStencilAttachment->storeOp;
        }
    }
    openPass(r, std::move(e));
    gNext.CmdBeginRendering(cb, info);
}

VKAPI_ATTR void CmdEndRendering(VkCommandBuffer cb) {
    Record& r = recordOf(cb);
    const VkImage probed = hudProbeTarget(r);
    r.open = -1;
    gNext.CmdEndRendering(cb);
    if (probed != VK_NULL_HANDLE) hud_probe::Record(cb, probed);
}

VKAPI_ATTR void CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline) {
    if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        Record& r = recordOf(cb);
        r.graphics = pipeline;
        std::lock_guard lock(gLock);
        const auto found = gPipelines.find(pipeline);
        r.pipeline = found == gPipelines.end() ? PipelineDesc() : found->second;
    }
    gNext.CmdBindPipeline(cb, point, pipeline);
}

VKAPI_ATTR void CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t count,
                               const VkViewport* viewports) {
    if (first == 0 && count > 0) {
        Record& r = recordOf(cb);
        r.viewportHeight = viewports[0].height;
        r.viewportX = viewports[0].x;
        r.viewportY = viewports[0].y;
        r.viewportWidth = viewports[0].width;
    }
    gNext.CmdSetViewport(cb, first, count, viewports);
}

VKAPI_ATTR void CmdSetDepthWriteEnable(VkCommandBuffer cb, VkBool32 enable) {
    recordOf(cb).dynWrite = enable;
    gNext.CmdSetDepthWriteEnable(cb, enable);
}

VKAPI_ATTR void CmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t count,
                              const VkRect2D* scissors) {
    if (first == 0 && count > 0) recordOf(cb).scissor = scissors[0].extent;
    gNext.CmdSetScissor(cb, first, count, scissors);
}

VKAPI_ATTR void CmdSetDepthCompareOp(VkCommandBuffer cb, VkCompareOp op) {
    recordOf(cb).dynCompare = op;
    gNext.CmdSetDepthCompareOp(cb, op);
}

void cutNow(VkCommandBuffer cb, Record& r, const Split& s);

VKAPI_ATTR void CmdClearAttachments(VkCommandBuffer cb, uint32_t count,
                                    const VkClearAttachment* attachments, uint32_t rects,
                                    const VkClearRect* rect) {
    Record& r = recordOf(cb);
    bool depth = false;
    for (uint32_t i = 0; i < count; i++) depth |= attachments[i].aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT;
    if (depth && r.cut.armed && r.cut.atClear && r.open >= 0 &&
        r.cut.clearsSeen++ == r.cut.clearIndex) {
        Split s;
        {
            std::lock_guard lock(gLock);
            s = gSplit;
        }
        if (s.scene != VK_NULL_HANDLE) {
            cutNow(cb, r, s);
        } else {
            r.cut.armed = false;
        }
    }
    if (!r.ignored && r.open >= 0) {
        Pass& p = r.events[r.open].pass;
        if (depth) {
            p.depthClears++;
            p.lastClearDraw = p.draws;
        }
        if (survey::enabled()) p.drawTrace += depth ? '|' : '/';
    }
    gNext.CmdClearAttachments(cb, count, attachments, rects, rect);
}

VKAPI_ATTR void CmdSetDepthTestEnable(VkCommandBuffer cb, VkBool32 enable) {
    recordOf(cb).dynTest = enable;
    gNext.CmdSetDepthTestEnable(cb, enable);
}

// ---- Hide HUD: what a draw samples ----------------------------------------------------------

// Callers hold gLock. What `set` samples, as kHudTex* bits.
uint8_t hudTextureBits(VkDescriptorSet set) {
    const auto found = set == VK_NULL_HANDLE ? gSetImages.end() : gSetImages.find(set);
    if (found == gSetImages.end()) return 0;
    uint8_t bits = 0;
    for (const auto& entry : found->second) {
        const Image* i = imageOf(entry.second);
        if (i == nullptr) continue;
        if (IsDepthFormat(i->format)) {
            bits |= kHudTexDepth;
        } else if (i->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
            bits |= kHudTexRenderTarget | (IsUntoned(i->format) ? kHudTexUntoned : 0);
        } else {
            bits |= kHudTexUploaded;
            const bool single = i->format == VK_FORMAT_R8_UNORM || i->format == VK_FORMAT_A8_UNORM_KHR;
            if (single && i->width >= 64) bits |= kHudTexSingleChannel;
        }
    }
    return bits;
}

// ------------------------------------------------------------------------------------------------

VKAPI_ATTR void CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint point,
                                      VkPipelineLayout layout, uint32_t first, uint32_t count,
                                      const VkDescriptorSet* sets, uint32_t dynamicCount,
                                      const uint32_t* dynamicOffsets) {
    const int at = point == VK_PIPELINE_BIND_POINT_GRAPHICS ? 0
            : point == VK_PIPELINE_BIND_POINT_COMPUTE  ? 1
                                                       : -1;
    if (at >= 0) {
        Record& r = recordOf(cb);
        for (uint32_t i = 0; i < count && first + i < kMaxSets; i++) r.bound[at][first + i] = sets[i];
        r.boundVersion[at]++;
        if (at == 0 && HudCapturing()) {
            std::lock_guard lock(gLock);
            for (uint32_t i = 0; i < count && first + i < kMaxSets; i++) {
                r.hudTextures[first + i] = hudTextureBits(sets[i]);
            }
        }
        if (at == 0) {
            r.boundLayout = layout;
            for (uint32_t i = 0; i < count && first + i < kMaxSets; i++) {
                r.dynamicOffsets[first + i].clear();
            }
            if (dynamicCount > 0) {
                // Offsets go to sets in order, each taking as many as its layout has dynamic
                // descriptors.
                std::lock_guard lock(gLock);
                uint32_t next = 0;
                for (uint32_t i = 0; i < count && next < dynamicCount; i++) {
                    const auto layout = gSetLayoutOf.find(sets[i]);
                    if (layout == gSetLayoutOf.end()) break;
                    const auto desc = gSetLayouts.find(layout->second);
                    if (desc == gSetLayouts.end()) break;
                    const uint32_t n = std::min<uint32_t>(desc->second.dynamicBindings.size(),
                                                          dynamicCount - next);
                    if (first + i < kMaxSets) {
                        r.dynamicOffsets[first + i].assign(dynamicOffsets + next,
                                                           dynamicOffsets + next + n);
                    }
                    next += n;
                }
            }
        }
    }
    gNext.CmdBindDescriptorSets(cb, point, layout, first, count, sets, dynamicCount,
                                dynamicOffsets);
}

void noteDispatch(VkCommandBuffer cb) {
    Record& r = recordOf(cb);
    if (r.events.empty() || r.events.back().isPass ||
        r.events.back().op.kind != OpKind::kDispatch) {
        Step e;
        e.op.kind = OpKind::kDispatch;
        if (!r.push(std::move(e))) return;
    }
    r.events.back().op.count++;
    addSets(&r.sets.back(), r.bound[1], kMaxSets);
}

VKAPI_ATTR void CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    noteDispatch(cb);
    gNext.CmdDispatch(cb, x, y, z);
}

VKAPI_ATTR void CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset) {
    noteDispatch(cb);
    gNext.CmdDispatchIndirect(cb, buffer, offset);
}

VKAPI_ATTR VkResult BindOpticalFlowSessionImageNV(VkDevice device, VkOpticalFlowSessionNV session,
                                                  VkOpticalFlowSessionBindingPointNV point,
                                                  VkImageView view, VkImageLayout layout) {
    {
        std::lock_guard lock(gLock);
        gFlowBindings[session][point] = idOfView(view);
    }
    return gNext.BindOpticalFlowSessionImageNV(device, session, point, view, layout);
}

VKAPI_ATTR void CmdOpticalFlowExecuteNV(VkCommandBuffer cb, VkOpticalFlowSessionNV session,
                                        const VkOpticalFlowExecuteInfoNV* info) {
    Step e;
    e.op.kind = OpKind::kOpticalFlow;
    e.op.count = 1;
    {
        std::lock_guard lock(gLock);
        // Binding point order: input, reference, hint, flow, backward flow, cost, ...
        for (const auto& [point, id] : gFlowBindings[session]) {
            e.op.reads.push_back(static_cast<uint32_t>(point) << 24 | (id & 0xffffff));
        }
    }
    recordOf(cb).push(std::move(e));
    gNext.CmdOpticalFlowExecuteNV(cb, session, info);
}

// In the split pass, before each draw: once the draws that sample the untoned scene are done,
// the target holds the scene and no HUD. The pass ends there, the target is copied out, and the
// same pass resumes with its attachments loaded; the game's bound state carries over.
void cutBefore(VkCommandBuffer cb, Record& r) {
    Split s;
    bool toned = false;
    {
        std::lock_guard lock(gLock);
        if (gSplit.scene == VK_NULL_HANDLE) {
            r.cut.armed = false;
            return;
        }
        // The tonemap: a draw sampling a float scene of about the target's size. Not the image
        // the split was armed with, which the game may have made again since.
        const uint64_t area = static_cast<uint64_t>(gSplit.width) * gSplit.height;
        for (VkDescriptorSet set : r.bound[0]) {
            const auto found = set == VK_NULL_HANDLE ? gSetImages.end() : gSetImages.find(set);
            if (found == gSetImages.end()) continue;
            for (const auto& entry : found->second) {
                const Image* i = imageOf(entry.second);
                toned |= i != nullptr && (IsUntoned(i->format) || i->format == gSplit.untoned) &&
                        static_cast<uint64_t>(i->width) * i->height * kUntonedShare > area;
            }
        }
        s = gSplit;
    }
    if (toned) {
        r.cut.toned = true;
        r.pastScene = true;
        return;
    }
    if (!r.cut.toned) return;
    cutNow(cb, r, s);
}

// Copies all of `src`, in `layout` outside any pass, into `dst`, which ends shader-readable;
// `src` goes back to `layout`.
void copyOut(VkCommandBuffer cb, const Own& f, VkImage src, VkImageLayout layout, VkImage dst,
             uint32_t width, uint32_t height) {
    VkImageMemoryBarrier b[2] = {};
    for (VkImageMemoryBarrier& one : b) {
        one.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        one.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        one.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        one.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    b[0].image = src;
    b[0].oldLayout = layout;
    b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[1].image = dst;
    b[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 2, b);
    VkImageCopy copy = {};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = {width, height, 1};
    f.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].newLayout = layout;
    b[0].srcAccessMask = 0;
    b[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 2, b);
}

// The transparency cut's copies once `r`'s pass has ended.
void glassAtEnd(VkCommandBuffer cb, Record& r) {
    if (!r.glassOpaqueAtEnd && !r.glassLitAtEnd) return;
    Split opaque, lit;
    {
        std::lock_guard lock(gLock);
        opaque = gGlass.opaque;
        lit = gGlass.lit;
    }
    if (r.glassOpaqueAtEnd && opaque.scene != VK_NULL_HANDLE) {
        copyOut(cb, opaque.fn, r.glassTarget, r.glassFinal, opaque.scene, opaque.width, opaque.height);
    }
    if (r.glassLitAtEnd && lit.scene != VK_NULL_HANDLE) {
        copyOut(cb, opaque.fn, r.glassTarget, r.glassFinal, lit.scene, lit.width, lit.height);
    }
    r.glassOpaqueAtEnd = false;
    r.glassLitAtEnd = false;
}

// Ends the pass `c` armed here, copies its target into `s.scene` and resumes the pass with what it
// stored. `noted`: the copy and the resumed half go into the frame's graph, as the HUD cut's do;
// the transparency cut leaves the graph as the game drew it, so its own classes hold still.
void splitPass(VkCommandBuffer cb, Record& r, Record::Cut& c, const Split& s, bool noted) {
    c.armed = false;

    const Own& f = s.fn;
    if (c.v2) {
        const VkSubpassEndInfo end = {VK_STRUCTURE_TYPE_SUBPASS_END_INFO, nullptr};
        f.CmdEndRenderPass2(cb, &end);
    } else {
        f.CmdEndRenderPass(cb);
    }
    VkImageMemoryBarrier b[2] = {};
    for (VkImageMemoryBarrier& one : b) {
        one.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        one.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        one.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        one.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    b[0].image = c.target;
    b[0].oldLayout = s.final;
    b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    // Whatever read the last frame's copy, framefx's or the game's, came before on this queue.
    b[1].image = s.scene;
    b[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 2, b);
    VkImageCopy copy = {};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = {s.width, s.height, 1};
    f.CmdCopyImage(cb, c.target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.scene,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].newLayout = s.final;
    b[0].srcAccessMask = 0;
    b[0].dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 2, b);
    VkRenderPassBeginInfo begin = c.begin;
    begin.renderPass = s.resume;
    if (c.v2) {
        const VkSubpassBeginInfo sub = {VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO, nullptr,
                                        VK_SUBPASS_CONTENTS_INLINE};
        f.CmdBeginRenderPass2(cb, &begin, &sub);
    } else {
        f.CmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
    }

    if (!noted) return;
    r.open = -1;
    Step copied;
    copied.op.kind = OpKind::kCopy;
    copied.op.src = c.targetId;
    copied.op.dst = s.sceneId;
    r.push(std::move(copied));
    Step layout;
    layout.op.kind = OpKind::kLayout;
    layout.op.dst = s.sceneId;
    layout.op.from = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    layout.op.to = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    r.push(std::move(layout));
    Step resumed = c.pass;
    Pass& p = resumed.pass;
    p.renderPass = handleBits(s.resume);
    for (Target& t : p.colors) t.load = VK_ATTACHMENT_LOAD_OP_LOAD;
    if (p.hasDepth) p.depth.load = VK_ATTACHMENT_LOAD_OP_LOAD;
    p.stencilLoad = VK_ATTACHMENT_LOAD_OP_LOAD;
    p.clearDepth = NAN;
    openPass(r, std::move(resumed));
    r.pastScene = true;
}

// Ends the armed pass here, copies its target out and resumes the pass with what it stored.
void cutNow(VkCommandBuffer cb, Record& r, const Split& s) {
    splitPass(cb, r, r.cut, s, true);
    std::lock_guard lock(gLock);
    if (gSplit.scene != s.scene) return;
    gSplit.missed = 0;
    if (!gSplit.logged) {
        gSplit.logged = true;
        ALOGI("lapis %s: split at the first HUD draw; the scene is #%u",
              Engine::get().package().c_str(), s.sceneId);
    }
}

// Whether the constants of `p`'s first draw are kept.
bool watches(const Pass& p) {
    if (p.renderPass == 0) return survey::enabled();
    return survey::enabled() || p.renderPass == gWatchedPass.load(std::memory_order_relaxed) ||
            p.renderPass == gWatchedTonemap.load(std::memory_order_relaxed) ||
            p.renderPass == gWatchedShadow.load(std::memory_order_relaxed);
}

// Survey only, word `textcost`: what a text tap would cost, by doing its work.
struct TextCost {
    std::atomic<bool> sampling{false};
    std::atomic<uint64_t> sampleNs{0};
    std::atomic<uint64_t> hashNs{0};
    std::atomic<uint64_t> hash{0};
    std::atomic<uint32_t> inspected{0};
    std::atomic<uint32_t> atlasDraws{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint32_t> mapped{0};
    std::atomic<uint32_t> copied{0};
    std::atomic<uint32_t> missed{0};
    std::atomic<uint64_t> uploadNs{0};
    std::atomic<uint64_t> uploadHash{0};
    std::atomic<uint32_t> uploads{0};
    std::atomic<uint64_t> copyNs{0};
    std::atomic<uint64_t> copyBytes{0};
    int64_t lastSample = 0;
    int64_t lastLog = 0;
    uint32_t frames = 0;
    uint32_t changes = 0;
    uint64_t lastHash = 0;
    std::vector<uint32_t> samples;
};
TextCost gTextCost;
int64_t nowNs();

uint64_t hashBytes(const uint8_t* p, size_t n, uint64_t h) {
    for (; n >= 8; n -= 8, p += 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        h = (h ^ v) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    for (; n > 0; n--, p++) h = (h ^ *p) * 0x100000001B3ull;
    return h;
}

// Survey only, word `atlas`: text as it goes to the GPU. A single-channel texture is kept here as
// the game uploads it, and a shot keeps the draws that read one, read at record time: the game
// has written them by then, or it could not have told the driver how much to draw.
struct Upload {
    VkImage image;
    VkBuffer source;
    VkDeviceSize offset;
    uint32_t rowLength;
    uint32_t mip;
    uint32_t layer;
    VkOffset3D at;
    VkExtent3D extent;
};

struct Atlas {
    std::string pixels;
    std::vector<uint64_t> sentRows;  // per block of rows: what the text service last got
    uint32_t uploads = 0;
    uint32_t unread = 0;  // from memory the game has not mapped
    std::vector<std::array<uint32_t, 4>> regions;
};

struct TextDraw {
    ImageId atlas = 0;
    int event = 0;
    uint32_t draw = 0;
    uint32_t count = 0;
    bool indexed = false;
    int32_t vertexOffset = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    uint32_t firstVertex = 0;
    uint32_t vertices = 0;
    float viewportHeight = 0;
    VkExtent2D scissor = {};
    std::shared_ptr<const VertexLayout> layout;
    std::string indices;
    std::map<uint32_t, std::string> bindings;
    std::map<uint32_t, uint64_t> strides;
    std::vector<Constants> constants;
    std::string why;
};

constexpr size_t kMaxAtlasBytes = 4096 * 4096;
constexpr size_t kMaxAtlasRegions = 1 << 14;
constexpr size_t kMaxTextDraws = 512;
constexpr VkDeviceSize kMaxTextBytes = 2 << 20;
constexpr VkDeviceSize kMaxShadowBuffer = 8 << 20;
std::unordered_map<ImageId, Atlas> gAtlases;
std::vector<TextDraw> gTextDraws;
// Parts of images the service has not seen yet.
struct AlihRegion {
    ImageId image;
    uint32_t x, y, width, height;
};
std::vector<AlihRegion> gAlihRegions;
constexpr uint32_t kAlihMaxImageSide = 4096;
// What the game copied into a buffer it never maps, as the driver's staging path does.
std::unordered_map<VkBuffer, std::string> gBufferCopies;
uint32_t gTextShot = 0;

bool isSingleChannel(VkFormat f) {
    return f == VK_FORMAT_R8_UNORM || f == VK_FORMAT_R8_SRGB || f == VK_FORMAT_A8_UNORM_KHR;
}

// Callers hold gLock. A buffer's bytes as they stand now, from its mapping or the copies into it.
bool readBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, std::string* out) {
    const auto b = gBuffers.find(buffer);
    if (b == gBuffers.end()) return false;
    {
        std::lock_guard mapLock(gMapLock);
        const auto mapping = gMappings.find(b->second.memory);
        if (mapping != gMappings.end() && mapping->second.ptr != nullptr) {
            const Mapping& m = mapping->second;
            const VkDeviceSize at = b->second.offset + offset;
            if (at < m.offset || at + size > m.offset + m.size) return false;
            out->assign(reinterpret_cast<const char*>(m.ptr + (at - m.offset)), size);
            return true;
        }
    }
    const auto copy = gBufferCopies.find(buffer);
    if (copy == gBufferCopies.end() || offset + size > copy->second.size()) return false;
    out->assign(copy->second, offset, size);
    return true;
}

void noteUpload(const Upload& u) {
    const int64_t start = survey::textCost() ? nowNs() : 0;
    std::lock_guard lock(gLock);
    const ImageId id = idOfImage(u.image);
    const Image* image = imageOf(id);
    if (image == nullptr) return;
    Atlas& atlas = gAtlases[id];
    atlas.uploads++;
    if (!isSingleChannel(image->format) || u.mip != 0 || u.layer != 0) return;
    const size_t bytes = static_cast<size_t>(image->width) * image->height;
    if (bytes > kMaxAtlasBytes) return;
    if (u.at.x < 0 || u.at.y < 0 || u.at.x + u.extent.width > image->width ||
        u.at.y + u.extent.height > image->height) {
        return;
    }
    const uint32_t row = u.rowLength != 0 ? u.rowLength : u.extent.width;
    std::string source;
    if (!readBuffer(u.source, u.offset, static_cast<VkDeviceSize>(row) * u.extent.height, &source)) {
        atlas.unread++;
        return;
    }
    if (alihOn()) {
        gAlihRegions.push_back({id, static_cast<uint32_t>(u.at.x), static_cast<uint32_t>(u.at.y),
                                u.extent.width, u.extent.height});
    }
    if (survey::atlas() || alihOn()) {
        if (atlas.pixels.empty()) atlas.pixels.assign(bytes, '\0');
        for (uint32_t y = 0; y < u.extent.height; y++) {
            memcpy(&atlas.pixels[(u.at.y + y) * size_t{image->width} + u.at.x],
                   &source[y * size_t{row}], u.extent.width);
        }
        if (atlas.regions.size() < kMaxAtlasRegions) {
            atlas.regions.push_back({static_cast<uint32_t>(u.at.x), static_cast<uint32_t>(u.at.y),
                                     u.extent.width, u.extent.height});
        }
    }
    if (start != 0) {
        gTextCost.uploadHash.fetch_xor(
                hashBytes(reinterpret_cast<const uint8_t*>(source.data()), source.size(), 0),
                std::memory_order_relaxed);
        gTextCost.uploads.fetch_add(1, std::memory_order_relaxed);
        gTextCost.uploadNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
    }
}

VKAPI_ATTR void CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer source, VkImage image,
                                     VkImageLayout layout, uint32_t count,
                                     const VkBufferImageCopy* regions) {
    if (textTap()) {
        for (uint32_t i = 0; i < count; i++) {
            const VkBufferImageCopy& c = regions[i];
            noteUpload({image, source, c.bufferOffset, c.bufferRowLength, c.imageSubresource.mipLevel,
                        c.imageSubresource.baseArrayLayer, c.imageOffset, c.imageExtent});
        }
    }
    gNext.CmdCopyBufferToImage(cb, source, image, layout, count, regions);
}

VKAPI_ATTR void CmdCopyBufferToImage2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* info) {
    if (textTap()) {
        for (uint32_t i = 0; i < info->regionCount; i++) {
            const VkBufferImageCopy2& c = info->pRegions[i];
            noteUpload({info->dstImage, info->srcBuffer, c.bufferOffset, c.bufferRowLength,
                        c.imageSubresource.mipLevel, c.imageSubresource.baseArrayLayer,
                        c.imageOffset, c.imageExtent});
        }
    }
    gNext.CmdCopyBufferToImage2(cb, info);
}

// Callers hold gLock. Keeps what is copied into a vertex or index buffer the game never maps.
void shadowCopy(VkBuffer source, VkBuffer target, VkDeviceSize from, VkDeviceSize to,
                VkDeviceSize size) {
    const int64_t start = survey::textCost() ? nowNs() : 0;
    constexpr VkBufferUsageFlags kGeometry =
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    const auto t = gBuffers.find(target);
    bool mapped = true;
    if (t != gBuffers.end() && (t->second.usage & kGeometry) != 0) {
        std::lock_guard mapLock(gMapLock);
        mapped = gMappings.count(t->second.memory) > 0;
    }
    std::string bytes;
    if (!mapped && to + size <= kMaxShadowBuffer && readBuffer(source, from, size, &bytes)) {
        std::string& shadow = gBufferCopies[target];
        if (shadow.size() < to + size) shadow.resize(to + size);
        shadow.replace(to, size, bytes);
        if (start != 0) gTextCost.copyBytes.fetch_add(size, std::memory_order_relaxed);
    }
    if (start != 0) gTextCost.copyNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
}

VKAPI_ATTR void CmdCopyBuffer(VkCommandBuffer cb, VkBuffer source, VkBuffer target, uint32_t count,
                              const VkBufferCopy* regions) {
    if (textTap()) {
        std::lock_guard lock(gLock);
        for (uint32_t i = 0; i < count; i++) {
            shadowCopy(source, target, regions[i].srcOffset, regions[i].dstOffset, regions[i].size);
        }
    }
    gNext.CmdCopyBuffer(cb, source, target, count, regions);
}

VKAPI_ATTR void CmdCopyBuffer2(VkCommandBuffer cb, const VkCopyBufferInfo2* info) {
    if (textTap()) {
        std::lock_guard lock(gLock);
        for (uint32_t i = 0; i < info->regionCount; i++) {
            const VkBufferCopy2& c = info->pRegions[i];
            shadowCopy(info->srcBuffer, info->dstBuffer, c.srcOffset, c.dstOffset, c.size);
        }
    }
    gNext.CmdCopyBuffer2(cb, info);
}

VKAPI_ATTR void CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer target, VkDeviceSize offset,
                                VkDeviceSize size, const void* data) {
    if (textTap() && offset + size <= kMaxShadowBuffer) {
        std::lock_guard lock(gLock);
        const auto t = gBuffers.find(target);
        constexpr VkBufferUsageFlags kGeometry =
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (t != gBuffers.end() && (t->second.usage & kGeometry) != 0) {
            std::string& shadow = gBufferCopies[target];
            if (shadow.size() < offset + size) shadow.resize(offset + size);
            shadow.replace(offset, size, static_cast<const char*>(data), size);
        }
    }
    gNext.CmdUpdateBuffer(cb, target, offset, size, data);
}

VKAPI_ATTR void CmdBindIndexBuffer2(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                    VkDeviceSize size, VkIndexType type) {
    if (textTap()) {
        Record& r = recordOf(cb);
        r.indexBuffer = buffer;
        r.indexOffset = offset;
        r.indexType = type;
    }
    gNext.CmdBindIndexBuffer2(cb, buffer, offset, size, type);
}

VKAPI_ATTR void CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t count,
                                     const VkBuffer* buffers, const VkDeviceSize* offsets) {
    if (textTap()) {
        Record& r = recordOf(cb);
        for (uint32_t i = 0; i < count && first + i < kMaxVertexBindings; i++) {
            r.vertexBuffers[first + i].buffer = buffers[i];
            r.vertexBuffers[first + i].offset = offsets[i];
            r.vertexBuffers[first + i].stride = 0;
        }
    }
    gNext.CmdBindVertexBuffers(cb, first, count, buffers, offsets);
}

VKAPI_ATTR void CmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t count,
                                      const VkBuffer* buffers, const VkDeviceSize* offsets,
                                      const VkDeviceSize* sizes, const VkDeviceSize* strides) {
    if (textTap()) {
        Record& r = recordOf(cb);
        for (uint32_t i = 0; i < count && first + i < kMaxVertexBindings; i++) {
            r.vertexBuffers[first + i].buffer = buffers[i];
            r.vertexBuffers[first + i].offset = offsets[i];
            r.vertexBuffers[first + i].stride = strides != nullptr ? strides[i] : 0;
        }
    }
    gNext.CmdBindVertexBuffers2(cb, first, count, buffers, offsets, sizes, strides);
}

VKAPI_ATTR void CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                   VkIndexType type) {
    if (textTap()) {
        Record& r = recordOf(cb);
        r.indexBuffer = buffer;
        r.indexOffset = offset;
        r.indexType = type;
    }
    gNext.CmdBindIndexBuffer(cb, buffer, offset, type);
}

VKAPI_ATTR void CmdSetVertexInputEXT(VkCommandBuffer cb, uint32_t bindingCount,
                                     const VkVertexInputBindingDescription2EXT* bindings,
                                     uint32_t attributeCount,
                                     const VkVertexInputAttributeDescription2EXT* attributes) {
    if (textTap()) {
        auto layout = std::make_shared<VertexLayout>();
        for (uint32_t i = 0; i < bindingCount; i++) {
            layout->bindings.push_back({bindings[i].binding, bindings[i].stride, bindings[i].inputRate});
        }
        for (uint32_t i = 0; i < attributeCount; i++) {
            layout->attributes.push_back({attributes[i].location, attributes[i].binding,
                                          attributes[i].format, attributes[i].offset});
        }
        recordOf(cb).dynamicVertex = std::move(layout);
    }
    gNext.CmdSetVertexInputEXT(cb, bindingCount, bindings, attributeCount, attributes);
}

// The text tap for org.yasvara.alih. At most 4 times a second a frame's text draws go to the service,
// when their glyphs have changed; the service answers with what to replace, in a table.
//
// Replacing is where the game draws. The original glyph quads are skipped; the translation is drawn
// by the same pipeline, in the same place in the pass, from vertices copied off the quads it replaces,
// so it keeps their colour, fade, clip and z-order. The one thing that differs is a descriptor set
// with a private atlas in it.
constexpr VkDeviceSize kAlihRingBytes = 8 << 20;
constexpr int64_t kAlihSamplePeriodNs = 250'000'000;
constexpr uint32_t kAlihMaxIndices = 12000;
constexpr size_t kAlihMaxSample = 1 << 20;
constexpr uint32_t kAlihChunkRows = 256;
// A line the game is still typing out ends the draw early: this much of it counts as the line.
constexpr uint32_t kAlihMinTyped = 3;
constexpr uint32_t kAlihMaxParts = 4;

struct AlihPart {
    uint32_t source = 0;  // the quad of the text it is drawn as
    std::vector<alih::Glyph> glyphs;
};

struct AlihText {
    ImageId atlas = 0;
    std::vector<uint64_t> keys;
    std::vector<AlihPart> parts;
};

struct AlihTable {
    uint32_t generation = 0;
    uint32_t stableRows = 0;
    uint32_t firstRow = 0;  // pixels holds whole rows from here: what changed since the last upload
    std::string pixels;
    std::vector<AlihText> texts;
    std::unordered_multimap<uint64_t, uint32_t> first;
    std::vector<ImageId> atlases;
};

struct AlihGpu {
    VkDevice device = VK_NULL_HANDLE;
    const AlihFn* fn = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    uint8_t* stagingBytes = nullptr;
    VkBuffer ring = VK_NULL_HANDLE;
    uint8_t* ringBytes = nullptr;
    VkDeviceSize ringAt = 0;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    bool written = false;
    bool failed = false;
};

// What the frame being sampled drew.
struct AlihSample {
    bool on = false;
    uint32_t draws = 0;
    uint64_t hash = 0;
    std::vector<uint8_t> bytes;
};

// What the tap costs the game's render thread, logged every 10 s under debug.framefx.alihcost.
struct AlihCost {
    int64_t drawNs = 0;
    int64_t presentNs = 0;
    int64_t bytes = 0;
    uint32_t calls = 0;
    uint32_t read = 0;
    uint32_t replaced = 0;
    uint32_t frames = 0;
    int64_t since = 0;
};
AlihCost gAlihCost;

bool alihCostOn() {
    static const bool on = [] {
        char value[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.framefx.alihcost", value) > 0 && value[0] == '1';
    }();
    return on;
}

struct AlihMatch {
    uint32_t start;
    uint32_t length;
    const AlihText* text;
};

// A replacement already in the ring, good while the quad it copies from reads the same.
struct AlihBuilt {
    uint32_t indices = 0;
    VkDeviceSize indexAt = 0;
    std::array<VkDeviceSize, kMaxVertexBindings> vertexAt = {};
};

struct AlihAttr {
    uint32_t binding = 0;
    uint32_t offset = 0;
    bool found = false;
};

// Text rarely changes between frames, and these are what makes that cheap. All under gLock.
constexpr size_t kAlihMemoSize = 512;
std::unordered_map<uint64_t, std::vector<AlihMatch>> gAlihMatches;  // by the glyphs a draw holds
std::unordered_map<uint64_t, AlihBuilt> gAlihBuilt;  // by the text and the quad it replaces
std::unordered_map<uint64_t, std::pair<AlihAttr, AlihAttr>> gAlihAttrs;  // uv, position, by layout

uint64_t alihMix(uint64_t h, uint64_t v) {
    h = (h ^ v) * 0x100000001B3ull;
    return h ^ h >> 29;
}

AlihGpu gAlihGpu;
std::shared_ptr<const AlihTable> gAlihNext;
std::shared_ptr<const AlihTable> gAlihTable;
std::atomic<bool> gAlihDirty{false};
std::atomic<bool> gAlihLoading{false};
std::unique_ptr<alih::RingWriter> gAlihWriter;
uint8_t* gAlihTableMemory = nullptr;
uint32_t gAlihEpoch = 0;
uint32_t gAlihTableSeen = 0;
uint64_t gAlihSentHash = 0;
uint32_t gAlihUploadedGeneration = 0;
uint32_t gAlihUploadedStable = 0;
int64_t gAlihSampledAt = 0;
AlihSample gAlihSample;

uint64_t alihSlot(ImageId atlas, uint64_t key) {
    return key ^ (uint64_t{atlas} * 0x9E3779B97F4A7C15ull);
}

bool alihParseTexts(AlihTable* table, const uint8_t* bytes, size_t size, uint32_t count) {
    size_t at = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t head[3];
        if (at + sizeof(head) > size) return false;
        memcpy(head, bytes + at, sizeof(head));
        at += sizeof(head);
        const size_t keys = size_t{head[1]} * sizeof(uint64_t);
        if (head[1] == 0 || head[2] > kAlihMaxParts || at + keys > size) return false;
        AlihText text;
        text.atlas = head[0];
        text.keys.resize(head[1]);
        memcpy(text.keys.data(), bytes + at, keys);
        at += keys;
        text.parts.resize(head[2]);
        for (AlihPart& part : text.parts) {
            uint32_t partHead[2];
            if (at + sizeof(partHead) > size) return false;
            memcpy(partHead, bytes + at, sizeof(partHead));
            at += sizeof(partHead);
            const size_t glyphs = size_t{partHead[1]} * sizeof(alih::Glyph);
            if (partHead[0] >= head[1] || partHead[1] > alih::kMaxGlyphs || at + glyphs > size) return false;
            part.source = partHead[0];
            part.glyphs.resize(partHead[1]);
            if (glyphs > 0) memcpy(part.glyphs.data(), bytes + at, glyphs);
            at += glyphs;
        }
        table->first.emplace(alihSlot(text.atlas, text.keys[0]), table->texts.size());
        if (std::find(table->atlases.begin(), table->atlases.end(), text.atlas) ==
            table->atlases.end()) {
            table->atlases.push_back(text.atlas);
        }
        table->texts.push_back(std::move(text));
    }
    return true;
}

// Off the frame: the copy out from under the service's sequence lock, and the parse.
void alihLoadTable(uint32_t sequence) {
    std::thread([sequence] {
        const auto* header = reinterpret_cast<const alih::TableHeader*>(gAlihTableMemory);
        auto table = std::make_shared<AlihTable>();
        const uint32_t count = header->texts;
        const size_t size = std::min<size_t>(header->bytes, alih::kTableSize - alih::kTableTextsOffset);
        table->generation = header->generation;
        table->stableRows = header->stableRows;
        const uint32_t used = std::min(header->usedRows, alih::kAtlasHeight);
        {
            std::lock_guard lock(gLock);
            table->firstRow = table->generation == gAlihUploadedGeneration ? gAlihUploadedStable : 0;
        }
        if (used > table->firstRow) {
            table->pixels.assign(reinterpret_cast<const char*>(gAlihTableMemory + alih::kTableAtlasOffset +
                                                               size_t{table->firstRow} * alih::kAtlasWidth),
                                 size_t{used - table->firstRow} * alih::kAtlasWidth);
        }
        std::string texts(reinterpret_cast<const char*>(gAlihTableMemory + alih::kTableTextsOffset),
                          size);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (header->sequence.load(std::memory_order_relaxed) == sequence &&
            alihParseTexts(table.get(), reinterpret_cast<const uint8_t*>(texts.data()), texts.size(),
                           count)) {
            ALOGI("alih: %zu texts, %zu KB of atlas", table->texts.size(), table->pixels.size() / 1024);
            std::lock_guard lock(gLock);
            gAlihNext = std::move(table);
            gAlihDirty.store(true);
        }
        gAlihLoading.store(false);
    }).detach();
}

// Callers hold gLock. As many waiting image regions as the ring has room for. Unreal uploads its whole
// atlas for a new glyph, so a big region goes in blocks, and only the blocks that changed.
void alihSendRegions() {
    std::vector<AlihRegion> regions;
    regions.swap(gAlihRegions);
    std::string tight;
    size_t next = 0;
    for (; next < regions.size(); next++) {
        const AlihRegion r = regions[next];
        const Image* image = imageOf(r.image);
        const auto found = gAtlases.find(r.image);
        if (image == nullptr || found == gAtlases.end() || found->second.pixels.empty() ||
            image->width > kAlihMaxImageSide || image->height > kAlihMaxImageSide) {
            continue;
        }
        if (r.height > kAlihChunkRows) {
            // Split up in place: the blocks go through this same loop.
            regions.push_back({r.image, r.x, r.y, r.width, kAlihChunkRows});
            regions.push_back({r.image, r.x, r.y + kAlihChunkRows, r.width, r.height - kAlihChunkRows});
            continue;
        }
        Atlas& atlas = found->second;
        const size_t first = (r.y + 0) * size_t{image->width} + r.x;
        if (r.width == image->width && r.height == kAlihChunkRows && r.y % kAlihChunkRows == 0) {
            const uint64_t hash = hashBytes(reinterpret_cast<const uint8_t*>(&atlas.pixels[first]),
                                            size_t{r.width} * r.height, 0);
            const size_t block = r.y / kAlihChunkRows;
            if (atlas.sentRows.size() <= block) atlas.sentRows.resize(block + 1, 0);
            if (atlas.sentRows[block] == hash) continue;
            atlas.sentRows[block] = hash;
        }
        tight.resize(size_t{r.width} * r.height);
        for (uint32_t y = 0; y < r.height; y++) {
            memcpy(&tight[size_t{y} * r.width],
                   &atlas.pixels[(r.y + y) * size_t{image->width} + r.x], r.width);
        }
        const alih::RegionHead head = {r.image,
                                       static_cast<uint16_t>(image->width),
                                       static_cast<uint16_t>(image->height),
                                       static_cast<uint16_t>(r.x),
                                       static_cast<uint16_t>(r.y),
                                       static_cast<uint16_t>(r.width),
                                       static_cast<uint16_t>(r.height)};
        const alih::RingWriter::Chunk chunks[] = {{&head, sizeof(head)}, {tight.data(), tight.size()}};
        if (!gAlihWriter->write(alih::kRegion, chunks)) break;
    }
    gAlihRegions.assign(regions.begin() + next, regions.end());
}

// Once per present: what the last frame drew goes out, and a new table comes in.
void alihPresentNow();

void alihPresent() {
    if (!alihCostOn()) {
        alihPresentNow();
        return;
    }
    const int64_t start = nowNs();
    alihPresentNow();
    const int64_t end = nowNs();
    std::lock_guard lock(gLock);
    AlihCost& c = gAlihCost;
    c.presentNs += end - start;
    c.frames++;
    if (c.since == 0) c.since = end;
    if (end - c.since < 10'000'000'000) return;
    const double frames = std::max<uint32_t>(c.frames, 1);
    ALOGI("alih cost: %.1f us/frame in draws, %.1f in present, %.0f calls/frame, %.1f read/frame, "
          "%.1f replaced/frame, %.1f KB/frame, %.0f fps",
          c.drawNs / frames / 1000, c.presentNs / frames / 1000, c.calls / frames, c.read / frames,
          c.replaced / frames, c.bytes / frames / 1024, frames * 1e9 / (end - c.since));
    c = AlihCost();
}

void alihPresentNow() {
    static bool opened = false;
    if (!opened) {
        opened = true;
        uint8_t* ring = nullptr;
        uint8_t* table = nullptr;
        if (Engine::get().openAlih(&ring, &table)) {
            std::lock_guard lock(gLock);
            gAlihWriter = std::make_unique<alih::RingWriter>(ring);
            gAlihTableMemory = table;
            ALOGI("alih: text tap open");
        } else {
            ALOGW("alih: the broker gave no memory");
        }
    }
    std::lock_guard lock(gLock);
    if (gAlihWriter == nullptr) return;
    const uint32_t epoch = gAlihWriter->readerEpoch();
    if (epoch != gAlihEpoch) {
        // A new reader: everything it needs, again. Until there is one, nothing is sent.
        gAlihEpoch = epoch;
        gAlihSentHash = 0;
        gAlihRegions.clear();
        for (auto& [id, atlas] : gAtlases) {
            const Image* image = imageOf(id);
            atlas.sentRows.clear();
            if (image == nullptr || atlas.pixels.empty()) continue;
            gAlihRegions.push_back({id, 0, 0, image->width, image->height});
        }
    }
    if (epoch == 0) {
        gAlihRegions.clear();
        return;
    }
    alihSendRegions();
    AlihSample& sample = gAlihSample;
    if (sample.on) {
        sample.on = false;
        // The images must be there first for the sample to mean anything.
        if (sample.hash != gAlihSentHash && gAlihRegions.empty()) {
            const uint32_t draws = sample.draws;
            const alih::RingWriter::Chunk chunks[] = {{&draws, sizeof(draws)},
                                                      {sample.bytes.data(), sample.bytes.size()}};
            if (gAlihWriter->write(alih::kSample, chunks)) gAlihSentHash = sample.hash;
        }
    }
    const int64_t now = nowNs();
    if (now - gAlihSampledAt >= kAlihSamplePeriodNs) {
        gAlihSampledAt = now;
        sample.on = true;
        sample.draws = 0;
        sample.hash = 0;
        sample.bytes.clear();
    }
    const auto* header = reinterpret_cast<const alih::TableHeader*>(gAlihTableMemory);
    const uint32_t sequence = header->sequence.load(std::memory_order_acquire);
    if (sequence != gAlihTableSeen && (sequence & 1) == 0 && !gAlihLoading.exchange(true)) {
        gAlihTableSeen = sequence;
        alihLoadTable(sequence);
    }
}

int alihMemoryType(const VkPhysicalDeviceMemoryProperties& memory, uint32_t bits,
                   VkMemoryPropertyFlags want) {
    for (uint32_t t = 0; t < memory.memoryTypeCount; t++) {
        if ((bits & (1u << t)) && (memory.memoryTypes[t].propertyFlags & want) == want) return t;
    }
    return -1;
}

// Callers hold gLock.
bool alihBuffer(const DeviceInfo& d, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage,
                VkBuffer* buffer, uint8_t** bytes) {
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d.alih.CreateBuffer(device, &info, nullptr, buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements need;
    d.alih.GetBufferMemoryRequirements(device, *buffer, &need);
    const int type = alihMemoryType(d.memory, need.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type < 0) return false;
    VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = need.size;
    alloc.memoryTypeIndex = type;
    VkDeviceMemory memory;
    void* mapped = nullptr;
    if (d.alih.AllocateMemory(device, &alloc, nullptr, &memory) != VK_SUCCESS ||
        d.alih.BindBufferMemory(device, *buffer, memory, 0) != VK_SUCCESS ||
        d.alih.MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        return false;
    }
    *bytes = static_cast<uint8_t*>(mapped);
    return true;
}

// Callers hold gLock. The objects live until the process goes: command buffers hold them.
bool alihCreate() {
    AlihGpu& g = gAlihGpu;
    if (g.device != VK_NULL_HANDLE) return !g.failed;
    for (const auto& [device, info] : gDevices) {
        if (!info.alihLoaded) continue;
        g.device = device;
        g.fn = &info.alih;
        g.failed = true;
        VkImageCreateInfo image = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = VK_FORMAT_R8_UNORM;
        image.extent = {alih::kAtlasWidth, alih::kAtlasHeight, 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (g.fn->CreateImage(device, &image, nullptr, &g.image) != VK_SUCCESS) return false;
        VkMemoryRequirements need;
        g.fn->GetImageMemoryRequirements(device, g.image, &need);
        const int type = alihMemoryType(info.memory, need.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0) return false;
        VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = need.size;
        alloc.memoryTypeIndex = type;
        VkDeviceMemory memory;
        if (g.fn->AllocateMemory(device, &alloc, nullptr, &memory) != VK_SUCCESS ||
            g.fn->BindImageMemory(device, g.image, memory, 0) != VK_SUCCESS) {
            return false;
        }
        VkSamplerCreateInfo sampler = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (g.fn->CreateSampler(device, &sampler, nullptr, &g.sampler) != VK_SUCCESS) return false;
        if (!alihBuffer(info, device, size_t{alih::kAtlasWidth} * alih::kAtlasHeight,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &g.staging, &g.stagingBytes) ||
            !alihBuffer(info, device, kAlihRingBytes,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        &g.ring, &g.ringBytes)) {
            return false;
        }
        g.failed = false;
        return true;
    }
    return false;
}

// Outside any pass: the copy of a new table's atlas goes into the game's own command buffer, ahead
// of the draws that will read it.
void alihUpload(VkCommandBuffer cb) {
    if (!gAlihDirty.load(std::memory_order_relaxed) || recordOf(cb).ignored) return;
    std::lock_guard lock(gLock);
    if (!gAlihDirty.load() || gAlihNext == nullptr) return;
    gAlihDirty.store(false);
    if (!alihCreate()) {
        ALOGW("alih: no resources");
        return;
    }
    AlihGpu& g = gAlihGpu;
    const AlihTable& next = *gAlihNext;
    const auto rows = static_cast<uint32_t>(next.pixels.size() / alih::kAtlasWidth);
    if (rows > 0 || !g.written) {
        const size_t at = size_t{next.firstRow} * alih::kAtlasWidth;
        if (rows > 0) memcpy(g.stagingBytes + at, next.pixels.data(), next.pixels.size());
        VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = g.written ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = g.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g.fn->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        if (rows > 0) {
            VkBufferImageCopy copy = {};
            copy.bufferOffset = at;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageOffset = {0, static_cast<int32_t>(next.firstRow), 0};
            copy.imageExtent = {alih::kAtlasWidth, rows, 1};
            g.fn->CmdCopyBufferToImage(cb, g.staging, g.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        }
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        g.fn->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        g.written = true;
    }
    gAlihUploadedGeneration = next.generation;
    gAlihUploadedStable = next.stableRows;
    gAlihTable = gAlihNext;
    gAlihMatches.clear();
    gAlihBuilt.clear();
}

// Callers hold gLock. The game's set with one image swapped, or null.
VkDescriptorSet alihClone(VkDescriptorSet source, uint64_t slot, ImageId atlas) {
    AlihGpu& g = gAlihGpu;
    const auto known = gAlihClones.find(source);
    if (known != gAlihClones.end()) return known->second;
    const auto layout = gSetLayoutOf.find(source);
    if (layout == gSetLayoutOf.end()) return VK_NULL_HANDLE;
    const auto desc = gSetLayouts.find(layout->second);
    if (desc == gSetLayouts.end() || desc->second.bindings.empty()) return VK_NULL_HANDLE;
    if (g.view == VK_NULL_HANDLE) {
        VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = g.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R8_UNORM;
        const auto swizzle = gSwizzles.find(atlas);
        if (swizzle != gSwizzles.end()) view.components = swizzle->second;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (g.fn->CreateImageView(g.device, &view, nullptr, &g.view) != VK_SUCCESS) {
            g.view = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
    }
    VkDescriptorSet clone = VK_NULL_HANDLE;
    for (int attempt = 0; attempt < 2 && clone == VK_NULL_HANDLE; attempt++) {
        if (g.pool != VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            alloc.descriptorPool = g.pool;
            alloc.descriptorSetCount = 1;
            alloc.pSetLayouts = &layout->second;
            if (g.fn->AllocateDescriptorSets(g.device, &alloc, &clone) == VK_SUCCESS) break;
            clone = VK_NULL_HANDLE;
        }
        const VkDescriptorPoolSize sizes[] = {
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2048},
                {VK_DESCRIPTOR_TYPE_SAMPLER, 512},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 512},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 512},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},
                {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 128},
                {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 128},
                {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 128},
        };
        VkDescriptorPoolCreateInfo pool = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1024;
        pool.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
        pool.pPoolSizes = sizes;
        if (g.fn->CreateDescriptorPool(g.device, &pool, nullptr, &g.pool) != VK_SUCCESS) {
            g.pool = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
    }
    if (clone == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    std::vector<VkCopyDescriptorSet> copies;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    const uint32_t atlasBinding = static_cast<uint32_t>(slot >> 32);
    for (const VkDescriptorSetLayoutBinding& b : desc->second.bindings) {
        if (b.binding == atlasBinding) type = b.descriptorType;
        if (b.descriptorCount == 0) continue;
        VkCopyDescriptorSet c = {VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
        c.srcSet = source;
        c.srcBinding = b.binding;
        c.dstSet = clone;
        c.dstBinding = b.binding;
        c.descriptorCount = b.descriptorCount;
        copies.push_back(c);
    }
    const VkDescriptorImageInfo image = {g.sampler, g.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = clone;
    write.dstBinding = atlasBinding;
    write.dstArrayElement = static_cast<uint32_t>(slot);
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = &image;
    g.fn->UpdateDescriptorSets(g.device, 0, nullptr, copies.size(), copies.data());
    g.fn->UpdateDescriptorSets(g.device, 1, &write, 0, nullptr);
    gAlihClones[source] = clone;
    return clone;
}

bool isFloatPair(VkFormat f) {
    return f == VK_FORMAT_R32G32_SFLOAT || f == VK_FORMAT_R32G32B32_SFLOAT ||
            f == VK_FORMAT_R32G32B32A32_SFLOAT;
}

// Callers hold gLock and gMapLock, which keeps the pointer good. A buffer's bytes where they are:
// a copy of a whole vertex stream per frame would be most of what the tap costs.
const uint8_t* alihBytes(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
    const auto b = gBuffers.find(buffer);
    if (b == gBuffers.end()) return nullptr;
    const auto mapping = gMappings.find(b->second.memory);
    if (mapping != gMappings.end() && mapping->second.ptr != nullptr) {
        const Mapping& m = mapping->second;
        const VkDeviceSize at = b->second.offset + offset;
        if (at < m.offset || at + size > m.offset + m.size) return nullptr;
        return reinterpret_cast<const uint8_t*>(m.ptr) + (at - m.offset);
    }
    const auto copy = gBufferCopies.find(buffer);
    if (copy == gBufferCopies.end() || offset + size > copy->second.size()) return nullptr;
    return reinterpret_cast<const uint8_t*>(copy->second.data()) + offset;
}

struct AlihBlock {
    uint32_t start = 0;  // source glyphs in the draw
    uint32_t length = 0;
    AlihBuilt built;
};

// Returns whether it recorded the draw itself.
bool alihDraw(VkCommandBuffer cb, Record& r, uint32_t indices, uint32_t instances, uint32_t first,
              int32_t offset, uint32_t firstInstance) {
    if (r.open < 0 || r.ignored || instances != 1 || !r.pipeline.blend || indices < 6 ||
        indices % 6 != 0 || indices > kAlihMaxIndices) {
        return false;
    }
    std::unique_lock lock(gLock);
    const bool sampling = gAlihSample.on;
    const std::shared_ptr<const AlihTable> table = gAlihTable;
    if (!sampling && table == nullptr) return false;
    ImageId atlas = 0;
    int setIndex = -1;
    uint64_t slot = 0;
    for (int s = 0; s < kMaxSets && atlas == 0; s++) {
        if (r.bound[0][s] == VK_NULL_HANDLE) continue;
        const auto found = gSetImages.find(r.bound[0][s]);
        if (found == gSetImages.end()) continue;
        for (const auto& entry : found->second) {
            const Image* i = imageOf(entry.second);
            const auto uploaded = gAtlases.find(entry.second);
            // ANGLE makes every texture a possible render target: what counts is the uploads.
            if (i != nullptr && isSingleChannel(i->format) && uploaded != gAtlases.end() &&
                uploaded->second.uploads > 0) {
                atlas = entry.second;
                setIndex = s;
                slot = entry.first;
                break;
            }
        }
    }
    const bool listed = table != nullptr && atlas != 0 &&
            std::find(table->atlases.begin(), table->atlases.end(), atlas) != table->atlases.end();
    if (atlas == 0 || (!sampling && !listed)) return false;
    const Image* image = imageOf(atlas);
    if (image == nullptr) return false;
    std::shared_ptr<const VertexLayout> layout = r.dynamicVertex;
    if (!layout) {
        const auto found = gVertexLayouts.find(r.graphics);
        if (found == gVertexLayouts.end()) return false;
        layout = found->second;
    }
    const uint32_t indexSize = r.indexType == VK_INDEX_TYPE_UINT32 ? 4
            : r.indexType == VK_INDEX_TYPE_UINT8_EXT                ? 1
                                                                    : 2;
    std::unique_lock mapLock(gMapLock);
    const uint8_t* indexBytes = alihBytes(r.indexBuffer, r.indexOffset + VkDeviceSize{first} * indexSize,
                                          VkDeviceSize{indices} * indexSize);
    if (indexBytes == nullptr) return false;
    const auto indexAt = [&](uint32_t i) {
        uint32_t value = 0;
        memcpy(&value, indexBytes + size_t{i} * indexSize, indexSize);
        return value;
    };
    uint32_t low = UINT32_MAX, high = 0;
    for (uint32_t i = 0; i < indices; i++) {
        const uint32_t value = indexAt(i);
        low = std::min(low, value);
        high = std::max(high, value);
    }
    const uint32_t vertexCount = high - low + 1;
    std::array<const uint8_t*, kMaxVertexBindings> vertices = {};
    std::array<VkDeviceSize, kMaxVertexBindings> strides = {};
    for (const VkVertexInputBindingDescription& b : layout->bindings) {
        if (b.binding >= kMaxVertexBindings) return false;
        const auto& bound = r.vertexBuffers[b.binding];
        const VkDeviceSize stride = bound.stride != 0 ? bound.stride : b.stride;
        strides[b.binding] = stride;
        const int64_t at = static_cast<int64_t>(bound.offset) +
                static_cast<int64_t>(stride) * (static_cast<int64_t>(low) + offset);
        if (at < 0 || stride * vertexCount > kMaxTextBytes) return false;
        vertices[b.binding] = alihBytes(bound.buffer, at, stride * vertexCount);
        if (vertices[b.binding] == nullptr) return false;
    }
    const auto pair = [&](const AlihAttr& a, uint32_t vertex, float* out) {
        memcpy(out, vertices[a.binding] + strides[a.binding] * (vertex - low) + a.offset,
               2 * sizeof(float));
    };
    // By their ranges, once per layout: UVs stay inside 0 to 1 and vary, positions leave it.
    uint64_t layoutKey = 0;
    for (const VkVertexInputAttributeDescription& a : layout->attributes) {
        layoutKey = alihMix(layoutKey, uint64_t{a.binding} << 48 | uint64_t{a.format} << 32 | a.offset);
    }
    AlihAttr uv, position;
    const auto knownAttrs = gAlihAttrs.find(layoutKey);
    if (knownAttrs != gAlihAttrs.end()) {
        uv = knownAttrs->second.first;
        position = knownAttrs->second.second;
    } else {
        for (const VkVertexInputAttributeDescription& a : layout->attributes) {
            if (!isFloatPair(a.format) || a.binding >= kMaxVertexBindings ||
                strides[a.binding] < a.offset + 2 * sizeof(float)) {
                continue;
            }
            AlihAttr attr{a.binding, a.offset, true};
            float least = INFINITY, most = -INFINITY;
            for (uint32_t v = low; v <= high; v++) {
                float values[2];
                pair(attr, v, values);
                least = std::min({least, values[0], values[1]});
                most = std::max({most, values[0], values[1]});
            }
            const bool inside = least >= -1e-3f && most <= 1.001f;
            if (inside && most > least && !uv.found) {
                uv = attr;
            } else if (!inside && !position.found) {
                position = attr;
            }
        }
        if (!uv.found || !position.found) return false;
        gAlihAttrs[layoutKey] = {uv, position};
    }
    if (strides[uv.binding] < uv.offset + 2 * sizeof(float) ||
        strides[position.binding] < position.offset + 2 * sizeof(float)) {
        return false;
    }
    if (alihCostOn()) {
        gAlihCost.read++;
        gAlihCost.bytes += indices * indexSize + vertexCount * 2 * sizeof(float);
    }

    const uint32_t quads = indices / 6;
    const auto corners = [&](uint32_t quad, std::array<uint32_t, 4>* out, std::array<uint8_t, 6>* pattern) {
        uint32_t found = 0;
        for (uint32_t i = 0; i < 6; i++) {
            const uint32_t v = indexAt(quad * 6 + i);
            uint32_t at = 0;
            while (at < found && (*out)[at] != v) at++;
            if (at == found) {
                if (found == 4) return false;
                (*out)[found++] = v;
            }
            (*pattern)[i] = static_cast<uint8_t>(at);
        }
        return found == 4;
    };
    std::vector<uint64_t> keys(quads, 0);
    std::vector<alih::Quad> sampled;
    if (sampling) sampled.reserve(quads);
    uint64_t glyphsHash = atlas;
    for (uint32_t q = 0; q < quads; q++) {
        std::array<uint32_t, 4> c;
        std::array<uint8_t, 6> pattern;
        if (!corners(q, &c, &pattern)) continue;
        float umin = INFINITY, umax = -INFINITY, vmin = INFINITY, vmax = -INFINITY;
        for (uint32_t v : c) {
            float values[2];
            pair(uv, v, values);
            umin = std::min(umin, values[0]);
            umax = std::max(umax, values[0]);
            vmin = std::min(vmin, values[1]);
            vmax = std::max(vmax, values[1]);
        }
        const auto px = [](float value, uint32_t size) {
            return static_cast<uint32_t>(std::floor(static_cast<double>(value) * size + 0.5));
        };
        keys[q] = alih::Key(px(umin, image->width), px(vmin, image->height),
                            px(umax, image->width), px(vmax, image->height));
        glyphsHash = alihMix(glyphsHash, keys[q]);
        if (!sampling) continue;
        alih::Quad quad = {};
        quad.key = keys[q];
        quad.xMin = quad.yMin = INFINITY;
        quad.xMax = quad.yMax = -INFINITY;
        float at[4][2];
        for (int i = 0; i < 4; i++) {
            pair(position, c[i], at[i]);
            quad.xMin = std::min(quad.xMin, at[i][0]);
            quad.xMax = std::max(quad.xMax, at[i][0]);
            quad.yMin = std::min(quad.yMin, at[i][1]);
            quad.yMax = std::max(quad.yMax, at[i][1]);
        }
        const float midX = (quad.xMin + quad.xMax) / 2, midY = (quad.yMin + quad.yMax) / 2;
        for (int i = 0; i < 4; i++) {
            float values[2];
            pair(uv, c[i], values);
            const bool left = at[i][0] < midX, low2 = at[i][1] < midY;
            if (left && low2) memcpy(quad.uvLow, values, sizeof(values));
            if (!left && low2) memcpy(quad.uvRight, values, sizeof(values));
            if (left && !low2) memcpy(quad.uvUp, values, sizeof(values));
        }
        sampled.push_back(quad);
    }
    if (sampling && !sampled.empty() &&
        gAlihSample.bytes.size() + sampled.size() * sizeof(alih::Quad) <= kAlihMaxSample) {
        // Only what the glyphs are enters the hash: a text that moves or fades is not new text.
        gAlihSample.hash = alihMix(gAlihSample.hash, glyphsHash);
        const alih::DrawHead head = {atlas, static_cast<uint16_t>(image->width),
                                     static_cast<uint16_t>(image->height),
                                     static_cast<uint32_t>(sampled.size())};
        const auto* headBytes = reinterpret_cast<const uint8_t*>(&head);
        gAlihSample.bytes.insert(gAlihSample.bytes.end(), headBytes, headBytes + sizeof(head));
        const auto* quadBytes = reinterpret_cast<const uint8_t*>(sampled.data());
        gAlihSample.bytes.insert(gAlihSample.bytes.end(), quadBytes,
                                 quadBytes + sampled.size() * sizeof(alih::Quad));
        gAlihSample.draws++;
    }
    if (!listed) return false;

    // The same glyphs as last frame need no looking through the table.
    auto memo = gAlihMatches.find(glyphsHash);
    if (memo == gAlihMatches.end()) {
        std::vector<AlihMatch> found;
        for (uint32_t q = 0; q < quads;) {
            const AlihText* best = nullptr;
            uint32_t bestLength = 0;
            const auto range = table->first.equal_range(alihSlot(atlas, keys[q]));
            for (auto it = range.first; it != range.second; ++it) {
                const AlihText& text = table->texts[it->second];
                if (text.atlas != atlas) continue;
                const auto want = static_cast<uint32_t>(text.keys.size());
                uint32_t have = 0;
                while (have < want && q + have < quads && keys[q + have] == text.keys[have]) have++;
                const bool typed = have >= kAlihMinTyped && q + have == quads;
                if ((have == want || typed) && have > bestLength) {
                    best = &text;
                    bestLength = have;
                }
            }
            if (best == nullptr) {
                q++;
                continue;
            }
            found.push_back({q, bestLength, best});
            q += bestLength;
        }
        if (gAlihMatches.size() >= kAlihMemoSize) gAlihMatches.clear();
        memo = gAlihMatches.emplace(glyphsHash, std::move(found)).first;
    }
    const std::vector<AlihMatch>& matches = memo->second;
    if (matches.empty()) return false;
    const VkDescriptorSet clone = alihClone(r.bound[0][setIndex], slot, atlas);
    if (clone == VK_NULL_HANDLE) return false;

    AlihGpu& g = gAlihGpu;
    const auto reserve = [&](VkDeviceSize bytes) {
        VkDeviceSize at = (g.ringAt + 15) & ~VkDeviceSize{15};
        if (at + bytes > kAlihRingBytes) {
            at = 0;
            gAlihBuilt.clear();
        }
        g.ringAt = at + bytes;
        return at;
    };
    std::vector<AlihBlock> blocks;
    blocks.reserve(matches.size());
    std::vector<uint8_t> source;
    for (const AlihMatch& m : matches) {
        for (const AlihPart& part : m.text->parts) {
            // A text the game is still typing may not have this part's quad yet.
            if (part.source >= m.length) continue;
            std::array<uint32_t, 4> c;
            std::array<uint8_t, 6> pattern;
            if (!corners(m.start + part.source, &c, &pattern)) return false;
            AlihBlock block;
            block.start = m.start;
            block.length = m.length;
            // Position, colour, all the shader reads of the quad it stands in for: if none of that moved
            // since it was built, what is in the ring is still right.
            uint64_t signature = reinterpret_cast<uintptr_t>(&part);
            for (const VkVertexInputAttributeDescription& a : layout->attributes) {
                if (a.binding >= kMaxVertexBindings || strides[a.binding] <= a.offset) continue;
                const size_t size = std::min<size_t>(16, strides[a.binding] - a.offset);
                signature = alihMix(signature, strides[a.binding]);
                for (uint32_t v : c) {
                    uint64_t words[2] = {};
                    memcpy(words, vertices[a.binding] + strides[a.binding] * (v - low) + a.offset, size);
                    signature = alihMix(alihMix(signature, words[0]), words[1]);
                }
            }
            const auto known = gAlihBuilt.find(signature);
            if (known != gAlihBuilt.end()) {
                block.built = known->second;
                blocks.push_back(block);
                continue;
            }
            float xs[4], ys[4];
            float xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
            for (int i = 0; i < 4; i++) {
                float values[2];
                pair(position, c[i], values);
                xs[i] = values[0];
                ys[i] = values[1];
                xmin = std::min(xmin, xs[i]);
                xmax = std::max(xmax, xs[i]);
                ymin = std::min(ymin, ys[i]);
                ymax = std::max(ymax, ys[i]);
            }
            const auto glyphs = static_cast<uint32_t>(part.glyphs.size());
            block.built.indices = glyphs * 6;
            if (glyphs > 0) {
                for (const VkVertexInputBindingDescription& b : layout->bindings) {
                    const VkDeviceSize stride = strides[b.binding];
                    // The game's memory reads slowly: once per corner, not per glyph.
                    source.resize(stride * 4);
                    for (int corner = 0; corner < 4; corner++) {
                        memcpy(source.data() + stride * corner,
                               vertices[b.binding] + stride * (c[corner] - low), stride);
                    }
                    const VkDeviceSize at = reserve(stride * 4 * glyphs);
                    block.built.vertexAt[b.binding] = at;
                    uint8_t* out = g.ringBytes + at;
                    for (const alih::Glyph& glyph : part.glyphs) {
                        const bool upward = glyph.y0 > glyph.y1;
                        for (int corner = 0; corner < 4; corner++) {
                            memcpy(out, source.data() + stride * corner, stride);
                            const bool left = xs[corner] < (xmin + xmax) / 2;
                            const bool bottomOfRaw = ys[corner] < (ymin + ymax) / 2;
                            if (b.binding == position.binding) {
                                const float value[2] = {
                                        xmin + (left ? glyph.x0 : glyph.x1),
                                        ymin + (bottomOfRaw == upward ? glyph.y1 : glyph.y0)};
                                memcpy(out + position.offset, value, sizeof(value));
                            }
                            if (b.binding == uv.binding) {
                                const float value[2] = {
                                        left ? glyph.u0 : glyph.u1,
                                        bottomOfRaw == upward ? glyph.v1 : glyph.v0};
                                memcpy(out + uv.offset, value, sizeof(value));
                            }
                            out += stride;
                        }
                    }
                }
                block.built.indexAt = reserve(block.built.indices * sizeof(uint16_t));
                auto* out = reinterpret_cast<uint16_t*>(g.ringBytes + block.built.indexAt);
                for (uint32_t i = 0; i < glyphs; i++) {
                    for (uint8_t p : pattern) *out++ = static_cast<uint16_t>(i * 4 + p);
                }
            }
            if (gAlihBuilt.size() >= kAlihMemoSize) gAlihBuilt.clear();
            gAlihBuilt[signature] = block.built;
            blocks.push_back(block);
        }
    }
    const AlihFn& fn = *g.fn;
    const VkBuffer ring = g.ring;
    if (alihCostOn()) gAlihCost.replaced += blocks.size();
    mapLock.unlock();
    lock.unlock();

    const std::vector<uint32_t>& dynamicOffsets = r.dynamicOffsets[setIndex];
    const VkDescriptorSet original = r.bound[0][setIndex];
    uint32_t cursor = 0;
    const auto drawGame = [&](uint32_t from, uint32_t to) {
        if (to > from) {
            fn.CmdDrawIndexed(cb, (to - from) * 6, instances, first + from * 6, offset, firstInstance);
        }
    };
    const auto bindVertices = [&](const VkVertexInputBindingDescription& b, VkBuffer buffer,
                                  VkDeviceSize at, VkDeviceSize stride, bool withStride) {
        if (withStride && fn.CmdBindVertexBuffers2 != nullptr) {
            fn.CmdBindVertexBuffers2(cb, b.binding, 1, &buffer, &at, nullptr, &stride);
        } else {
            fn.CmdBindVertexBuffers(cb, b.binding, 1, &buffer, &at);
        }
    };
    for (const AlihBlock& block : blocks) {
        drawGame(cursor, block.start);
        cursor = block.start + block.length;
        if (block.built.indices == 0) continue;
        fn.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r.boundLayout, setIndex, 1,
                                 &clone, dynamicOffsets.size(), dynamicOffsets.data());
        for (const VkVertexInputBindingDescription& b : layout->bindings) {
            bindVertices(b, ring, block.built.vertexAt[b.binding], strides[b.binding],
                         r.vertexBuffers[b.binding].stride != 0);
        }
        fn.CmdBindIndexBuffer(cb, ring, block.built.indexAt, VK_INDEX_TYPE_UINT16);
        fn.CmdDrawIndexed(cb, block.built.indices, instances, 0, 0, firstInstance);
        fn.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r.boundLayout, setIndex, 1,
                                 &original, dynamicOffsets.size(), dynamicOffsets.data());
        for (const VkVertexInputBindingDescription& b : layout->bindings) {
            const auto& bound = r.vertexBuffers[b.binding];
            bindVertices(b, bound.buffer, bound.offset, bound.stride, bound.stride != 0);
        }
        fn.CmdBindIndexBuffer(cb, r.indexBuffer, r.indexOffset, r.indexType);
    }
    drawGame(cursor, quads);
    return true;
}

void resolveConstants(const Record::FirstDraw& draw, std::vector<Constants>* into);

void noteTextDraw(const Record& r, const Pass& p) {
    std::lock_guard lock(gLock);
    if (gTextDraws.size() >= kMaxTextDraws) return;
    ImageId atlas = 0;
    for (VkDescriptorSet set : r.bound[0]) {
        const auto found = set == VK_NULL_HANDLE ? gSetImages.end() : gSetImages.find(set);
        if (found == gSetImages.end()) continue;
        for (const auto& entry : found->second) {
            const Image* i = imageOf(entry.second);
            // ANGLE makes every texture a possible render target: what counts is that it was
            // uploaded to.
            if (i == nullptr || !isSingleChannel(i->format)) continue;
            const auto uploaded = gAtlases.find(entry.second);
            if ((i->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0 ||
                (uploaded != gAtlases.end() && uploaded->second.uploads > 0)) {
                atlas = entry.second;
            }
        }
    }
    if (atlas == 0) return;
    TextDraw d;
    d.atlas = atlas;
    d.event = r.open;
    d.draw = p.draws;
    d.count = r.drawCount;
    d.indexed = r.indexed;
    d.vertexOffset = r.vertexOffset;
    d.indexType = r.indexType;
    d.viewportHeight = r.viewportHeight;
    d.scissor = r.scissor;
    d.layout = r.dynamicVertex;
    if (!d.layout) {
        const auto layout = gVertexLayouts.find(r.graphics);
        if (layout != gVertexLayouts.end()) d.layout = layout->second;
    }
    int64_t low = r.firstElement;
    int64_t high = static_cast<int64_t>(r.firstElement) + r.drawCount - 1;
    if (r.indexed) {
        const uint32_t size = r.indexType == VK_INDEX_TYPE_UINT32 ? 4
                : r.indexType == VK_INDEX_TYPE_UINT8_EXT         ? 1
                                                                 : 2;
        const VkDeviceSize bytes = VkDeviceSize{r.drawCount} * size;
        low = INT64_MAX;
        high = -1;
        if (bytes <= kMaxTextBytes &&
            readBuffer(r.indexBuffer, r.indexOffset + VkDeviceSize{r.firstElement} * size, bytes,
                       &d.indices)) {
            for (uint32_t i = 0; i < r.drawCount; i++) {
                uint32_t value = 0;
                memcpy(&value, &d.indices[VkDeviceSize{i} * size], size);
                low = std::min<int64_t>(low, value);
                high = std::max<int64_t>(high, value);
            }
        } else {
            const auto b = gBuffers.find(r.indexBuffer);
            const auto copy = gBufferCopies.find(r.indexBuffer);
            std::lock_guard mapLock(gMapLock);
            d.why = StringPrintf(
                    "index buffer %p tracked=%d memory_mapped=%d copies=%zu offset=%" PRIu64 " first=%u",
                    r.indexBuffer, b != gBuffers.end(),
                    b != gBuffers.end() && gMappings.count(b->second.memory) > 0,
                    copy == gBufferCopies.end() ? 0 : copy->second.size(), r.indexOffset,
                    r.firstElement);
        }
    }
    if (d.layout && high >= low) {
        d.firstVertex = static_cast<uint32_t>(low);
        d.vertices = static_cast<uint32_t>(high - low + 1);
        for (const VkVertexInputBindingDescription& b : d.layout->bindings) {
            if (b.binding >= kMaxVertexBindings) continue;
            const auto& bound = r.vertexBuffers[b.binding];
            const VkDeviceSize stride = bound.stride != 0 ? bound.stride : b.stride;
            d.strides[b.binding] = stride;
            const VkDeviceSize bytes = stride * d.vertices;
            const VkDeviceSize start = bound.offset + stride * (low + r.vertexOffset);
            std::string data;
            if (bytes <= kMaxTextBytes && readBuffer(bound.buffer, start, bytes, &data)) {
                d.bindings[b.binding] = std::move(data);
            }
        }
    }
    Record::FirstDraw first;
    first.pipeline = r.graphics;
    std::copy(std::begin(r.bound[0]), std::end(r.bound[0]), first.sets);
    for (int i = 0; i < kMaxSets; i++) first.offsets[i] = r.dynamicOffsets[i];
    resolveConstants(first, &d.constants);
    gTextDraws.push_back(std::move(d));
}

// Writes what the shot kept to the app's cache, as framefx-atlas/s<shot>-*.
void flushTextDraws() {
    std::vector<TextDraw> draws;
    std::map<ImageId, Atlas> atlases;
    std::map<ImageId, Image> images;
    {
        std::lock_guard lock(gLock);
        draws.swap(gTextDraws);
        if (draws.empty() && !survey::shooting()) return;
        for (const auto& [id, atlas] : gAtlases) {
            const Image* i = imageOf(id);
            if (i == nullptr) continue;
            images[id] = *i;
            atlases[id] = atlas;
        }
    }
    const std::string dir = StringPrintf("/data/user/%u/%s/cache/framefx-atlas", getuid() / 100000,
                                         Engine::get().package().substr(
                                                 0, Engine::get().package().find(':')).c_str());
    mkdir(dir.c_str(), 0700);
    const std::string base = StringPrintf("%s/s%u-", dir.c_str(), ++gTextShot);
    std::string index;
    for (const auto& [id, atlas] : atlases) {
        const Image& i = images[id];
        if (i.width < 64 || i.height < 64) continue;
        StringAppendF(&index, "image #%u format=%d %ux%u usage=0x%x uploads=%u unread=%u regions=%zu\n",
                      id, i.format, i.width, i.height, i.usage, atlas.uploads, atlas.unread,
                      atlas.regions.size());
        if (atlas.pixels.empty()) continue;
        android::base::WriteStringToFile(
                StringPrintf("P5\n%u %u\n255\n", i.width, i.height) + atlas.pixels,
                StringPrintf("%satlas%u.pgm", base.c_str(), id));
        std::string regions;
        for (const auto& r : atlas.regions) {
            StringAppendF(&regions, "%u %u %u %u\n", r[0], r[1], r[2], r[3]);
        }
        android::base::WriteStringToFile(regions, StringPrintf("%satlas%u.regions", base.c_str(), id));
    }
    uint32_t n = 0;
    for (const TextDraw& d : draws) {
        StringAppendF(&index,
                      "draw %u event=%d in_pass=%u atlas=#%u count=%u indexed=%d index_type=%d "
                      "index_bytes=%zu vertex_offset=%d first_vertex=%u vertices=%u viewport_h=%.0f "
                      "scissor=%ux%u\n",
                      n, d.event, d.draw, d.atlas, d.count, d.indexed, d.indexType, d.indices.size(),
                      d.vertexOffset, d.firstVertex, d.vertices, d.viewportHeight, d.scissor.width,
                      d.scissor.height);
        if (!d.why.empty()) StringAppendF(&index, "  why %s\n", d.why.c_str());
        if (d.layout) {
            for (const auto& a : d.layout->attributes) {
                StringAppendF(&index, "  attr location=%u binding=%u format=%d offset=%u\n", a.location,
                              a.binding, a.format, a.offset);
            }
        }
        for (const auto& [binding, stride] : d.strides) {
            const auto data = d.bindings.find(binding);
            StringAppendF(&index, "  binding %u stride=%" PRIu64 " read=%zu\n", binding, stride,
                          data == d.bindings.end() ? 0 : data->second.size());
            if (data != d.bindings.end()) {
                android::base::WriteStringToFile(
                        data->second, StringPrintf("%sd%u.b%u", base.c_str(), n, binding));
            }
        }
        if (!d.indices.empty()) {
            android::base::WriteStringToFile(d.indices, StringPrintf("%sd%u.idx", base.c_str(), n));
        }
        for (const Constants& c : d.constants) {
            StringAppendF(&index, "  ubo set=%u binding=%u", c.set, c.binding);
            for (float v : c.values) StringAppendF(&index, " %.5g", v);
            index += '\n';
        }
        n++;
    }
    android::base::WriteStringToFile(index, base + "index.txt");
    ALOGI("survey atlas: %u draws in %s", n, base.c_str());
}

// Callers hold gLock and gMapLock. A buffer's bytes in place, from its mapping or the copies into it.
const uint8_t* viewBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, bool* mapped) {
    const auto b = gBuffers.find(buffer);
    if (b == gBuffers.end()) return nullptr;
    const auto mapping = gMappings.find(b->second.memory);
    if (mapping != gMappings.end() && mapping->second.ptr != nullptr) {
        const Mapping& m = mapping->second;
        const VkDeviceSize at = b->second.offset + offset;
        if (at < m.offset || at + size > m.offset + m.size) return nullptr;
        *mapped = true;
        return m.ptr + (at - m.offset);
    }
    const auto copy = gBufferCopies.find(buffer);
    if (copy == gBufferCopies.end() || offset + size > copy->second.size()) return nullptr;
    *mapped = false;
    return reinterpret_cast<const uint8_t*>(copy->second.data()) + offset;
}

// One blended draw that writes no depth, in a frame the tap reads.
void sampleTextDraw(const Record& r) {
    TextCost& c = gTextCost;
    const int64_t start = nowNs();
    c.inspected.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard lock(gLock);
        ImageId atlas = 0;
        for (VkDescriptorSet set : r.bound[0]) {
            const auto found = set == VK_NULL_HANDLE ? gSetImages.end() : gSetImages.find(set);
            if (found == gSetImages.end()) continue;
            for (const auto& entry : found->second) {
                const Image* i = imageOf(entry.second);
                if (i == nullptr || !isSingleChannel(i->format)) continue;
                const auto uploaded = gAtlases.find(entry.second);
                if ((i->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0 ||
                    (uploaded != gAtlases.end() && uploaded->second.uploads > 0)) {
                    atlas = entry.second;
                }
            }
        }
        std::shared_ptr<const VertexLayout> layout = r.dynamicVertex;
        if (!layout) {
            const auto found = gVertexLayouts.find(r.graphics);
            if (found != gVertexLayouts.end()) layout = found->second;
        }
        if (atlas != 0 && layout && r.drawCount % 6 == 0) {
            c.atlasDraws.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard mapLock(gMapLock);
            const int64_t hashing = nowNs();
            uint64_t bytes = 0;
            uint64_t h = 0;
            bool mapped = false;
            bool ok = true;
            int64_t low = r.firstElement;
            int64_t high = static_cast<int64_t>(r.firstElement) + r.drawCount - 1;
            if (r.indexed) {
                const uint32_t size = r.indexType == VK_INDEX_TYPE_UINT32 ? 4
                        : r.indexType == VK_INDEX_TYPE_UINT8_EXT         ? 1
                                                                         : 2;
                const VkDeviceSize span = VkDeviceSize{r.drawCount} * size;
                const uint8_t* index = viewBuffer(
                        r.indexBuffer, r.indexOffset + VkDeviceSize{r.firstElement} * size, span,
                        &mapped);
                low = INT64_MAX;
                high = -1;
                if (index == nullptr) {
                    ok = false;
                } else {
                    for (uint32_t i = 0; i < r.drawCount; i++) {
                        uint32_t value = 0;
                        memcpy(&value, index + VkDeviceSize{i} * size, size);
                        low = std::min<int64_t>(low, value);
                        high = std::max<int64_t>(high, value);
                    }
                    bytes += span;
                }
            }
            if (ok && high >= low) {
                for (const VkVertexInputBindingDescription& b : layout->bindings) {
                    if (b.binding >= kMaxVertexBindings) continue;
                    const auto& bound = r.vertexBuffers[b.binding];
                    const VkDeviceSize stride = bound.stride != 0 ? bound.stride : b.stride;
                    const VkDeviceSize span = stride * (high - low + 1);
                    const uint8_t* vertices = viewBuffer(
                            bound.buffer, bound.offset + stride * (low + r.vertexOffset), span, &mapped);
                    if (vertices == nullptr) {
                        ok = false;
                        continue;
                    }
                    h = hashBytes(vertices, span, h);
                    bytes += span;
                    (mapped ? c.mapped : c.copied).fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!ok) c.missed.fetch_add(1, std::memory_order_relaxed);
            c.hash.fetch_add(h, std::memory_order_relaxed);
            c.bytes.fetch_add(bytes, std::memory_order_relaxed);
            c.hashNs.fetch_add(nowNs() - hashing, std::memory_order_relaxed);
        }
    }
    c.sampleNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
}

void textCostPresent() {
    TextCost& c = gTextCost;
    const int64_t now = nowNs();
    c.frames++;
    if (c.sampling.exchange(false, std::memory_order_relaxed)) {
        if (c.samples.size() < 4096) {
            c.samples.push_back(static_cast<uint32_t>(c.sampleNs.exchange(0, std::memory_order_relaxed)));
        }
        const uint64_t hash = c.hash.exchange(0, std::memory_order_relaxed);
        if (hash != c.lastHash) c.changes++;
        c.lastHash = hash;
    }
    if (now - c.lastSample >= 250'000'000) {
        c.lastSample = now;
        c.sampling.store(true, std::memory_order_relaxed);
    }
    if (c.lastLog == 0) c.lastLog = now;
    if (now - c.lastLog < 10'000'000'000 || c.samples.empty()) return;
    std::sort(c.samples.begin(), c.samples.end());
    uint64_t total = 0;
    for (uint32_t s : c.samples) total += s;
    const size_t n = c.samples.size();
    const uint64_t bytes = c.bytes.exchange(0);
    const uint64_t hashNs = c.hashNs.exchange(0);
    const uint32_t uploads = c.uploads.exchange(0);
    const uint64_t uploadNs = c.uploadNs.exchange(0);
    const uint64_t copyNs = c.copyNs.exchange(0);
    const uint64_t copyBytes = c.copyBytes.exchange(0);
    ALOGI("textcost %s: %u frames %zu samples, sample mean %.1f p95 %.1f max %.1f us, %.2f us per "
          "frame; %u blended %u text draws, %.1f KB per sample, %.0f MB/s; mapped %u copied %u missed "
          "%u; text changed %u; uploads %u %.1f us each; shadow %.1f us per frame %.1f KB per frame",
          Engine::get().package().c_str(), c.frames, n, total / 1e3 / n, c.samples[n * 95 / 100] / 1e3,
          c.samples[n - 1] / 1e3, static_cast<double>(total) / 1e3 / std::max(c.frames, 1u),
          c.inspected.exchange(0), c.atlasDraws.exchange(0), bytes / 1024.0 / n,
          hashNs == 0 ? 0.0 : bytes * 1e3 / hashNs, c.mapped.exchange(0), c.copied.exchange(0),
          c.missed.exchange(0), c.changes, uploads, uploads == 0 ? 0.0 : uploadNs / 1e3 / uploads,
          copyNs / 1e3 / std::max(c.frames, 1u), copyBytes / 1024.0 / std::max(c.frames, 1u));
    c.samples.clear();
    c.frames = 0;
    c.changes = 0;
    c.lastLog = now;
}

// Survey only, word `fontscan`: font files in the process's memory. FreeType needs a font's bytes
// whole, so an engine that loads one from an encrypted bundle still holds it plain. Reads through
// process_vm_readv, which reports an unreadable page instead of faulting.
uint32_t be32(const uint8_t* p) {
    return uint32_t{p[0]} << 24 | uint32_t{p[1]} << 16 | uint32_t{p[2]} << 8 | p[3];
}
uint16_t be16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] << 8 | p[1]);
}

bool readSelf(uintptr_t at, void* into, size_t size) {
    iovec local = {into, size};
    iovec remote = {reinterpret_cast<void*>(at), size};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(size);
}

// The extent of the font whose offset table is at `base + table`, 0 when it is not one.
uint64_t fontExtent(uintptr_t base, uint32_t table) {
    uint8_t head[12];
    if (!readSelf(base + table, head, sizeof(head))) return 0;
    const uint32_t tag = be32(head);
    if (tag != 0x00010000 && tag != 0x4F54544F && tag != 0x74727565) return 0;
    const uint16_t count = be16(head + 4);
    if (count < 4 || count > 64) return 0;
    uint32_t power = 1;
    while (power * 2 <= count) power *= 2;
    if (be16(head + 6) != power * 16 || be16(head + 10) != count * 16 - power * 16) return 0;
    std::vector<uint8_t> records(size_t{count} * 16);
    if (!readSelf(base + table + 12, records.data(), records.size())) return 0;
    uint64_t extent = 0;
    bool hasHead = false;
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t* r = records.data() + size_t{i} * 16;
        const uint64_t offset = be32(r + 8);
        const uint64_t length = be32(r + 12);
        if (offset + length > (uint64_t{256} << 20)) return 0;
        extent = std::max(extent, offset + length);
        if (be32(r) == 0x68656164 && length >= 16) {  // head
            uint8_t magic[4];
            if (!readSelf(base + offset + 12, magic, 4) || be32(magic) != 0x5F0F3CF5) return 0;
            hasHead = true;
        }
    }
    return hasHead ? extent : 0;
}

void scanFontsNow() {
    prctl(PR_SET_NAME, "framefx-fonts");
    setpriority(PRIO_PROCESS, 0, 19);
    const int64_t start = nowNs();
    std::string maps;
    if (!android::base::ReadFileToString("/proc/self/maps", &maps)) return;
    const std::string pkg = Engine::get().package().substr(0, Engine::get().package().find(':'));
    const std::string dir = StringPrintf("/data/user/%u/%s/cache/framefx-fonts", getuid() / 100000,
                                         pkg.c_str());
    mkdir(dir.c_str(), 0700);
    const auto ends = [](const std::string& s, const char* suffix) {
        const size_t n = strlen(suffix);
        return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
    };
    constexpr size_t kChunk = 1 << 20;
    std::vector<uint8_t> chunk(kChunk);
    uint64_t scanned = 0;
    uint32_t regions = 0;
    uint32_t found = 0;
    for (const std::string& line : android::base::Split(maps, "\n")) {
        unsigned long low = 0;
        unsigned long high = 0;
        char perms[8] = {};
        int nameAt = 0;
        if (sscanf(line.c_str(), "%lx-%lx %7s %*s %*s %*s %n", &low, &high, perms, &nameAt) < 3) {
            continue;
        }
        const std::string name = nameAt > 0 ? line.substr(nameAt) : "";
        if (ends(name, ".ttf") || ends(name, ".otf") || ends(name, ".ttc")) {
            ALOGI("fontscan mapped %s", name.c_str());
        }
        if (perms[0] != 'r' || perms[2] == 'x' || high - low < 4096 ||
            high - low > (uint64_t{4} << 30)) {
            continue;
        }
        if (!name.empty()) {
            const bool anon = android::base::StartsWith(name, "[anon:") &&
                    !android::base::StartsWith(name, "[anon:dalvik");
            const bool data = android::base::StartsWith(name, "/data/") && !ends(name, ".so") &&
                    !ends(name, ".apk") && !ends(name, ".dex") && !ends(name, ".oat") &&
                    !ends(name, ".art") && !ends(name, ".vdex") && !ends(name, ".odex") &&
                    !ends(name, ".jar");
            if (!anon && !data) continue;
        }
        regions++;
        for (uintptr_t at = low; at < high;) {
            const size_t size = std::min<uintptr_t>(kChunk, high - at);
            iovec local = {chunk.data(), size};
            iovec remote = {reinterpret_cast<void*>(at), size};
            const ssize_t got = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
            if (got < 16) {
                at += std::max<size_t>(4096, got > 0 ? got : 0);
                continue;
            }
            scanned += got;
            for (ssize_t i = 0; i + 12 <= got; i += 4) {
                const uint8_t* p = chunk.data() + i;
                const bool font = p[0] == 0 ? (p[1] == 1 && p[2] == 0 && p[3] == 0)
                                            : (p[0] == 'O' && p[1] == 'T' && p[2] == 'T' && p[3] == 'O') ||
                                (p[0] == 't' && p[1] == 'r' && p[2] == 'u' && p[3] == 'e') ||
                                (p[0] == 't' && p[1] == 't' && p[2] == 'c' && p[3] == 'f');
                if (!font) continue;
                const uintptr_t base = at + i;
                uint64_t extent = 0;
                if (p[0] == 't' && p[1] == 't') {
                    const uint32_t fonts = be32(p + 8);
                    if (fonts == 0 || fonts > 64) continue;
                    std::vector<uint8_t> offsets(size_t{fonts} * 4);
                    if (!readSelf(base + 12, offsets.data(), offsets.size())) continue;
                    for (uint32_t k = 0; k < fonts; k++) {
                        extent = std::max(extent, fontExtent(base, be32(offsets.data() + size_t{k} * 4)));
                    }
                } else {
                    extent = fontExtent(base, 0);
                }
                if (extent < 1024) continue;
                std::string bytes(extent, '\0');
                if (!readSelf(base, bytes.data(), extent)) {
                    ALOGI("fontscan %zu bytes at %" PRIxPTR ": not all readable", size_t{extent}, base);
                    continue;
                }
                const uint64_t h = hashBytes(reinterpret_cast<const uint8_t*>(bytes.data()), extent, 0);
                const std::string path = StringPrintf("%s/%016" PRIx64 ".ttf", dir.c_str(), h);
                if (access(path.c_str(), F_OK) != 0) {
                    found++;
                    android::base::WriteStringToFile(bytes, path);
                    ALOGI("fontscan font %zu bytes at %" PRIxPTR " in [%s] -> %016" PRIx64,
                          size_t{extent}, base, name.c_str(), h);
                }
            }
            at += got > 12 ? got - 12 : got;
            usleep(500);
        }
    }
    ALOGI("fontscan: %u regions, %" PRIu64 " MB in %" PRId64 " ms, %u new fonts", regions,
          scanned >> 20, (nowNs() - start) / 1000000, found);
}

std::atomic<int64_t> gFontScanAt{0};

void maybeScanFonts() {
    const int64_t now = nowNs();
    int64_t last = gFontScanAt.load(std::memory_order_relaxed);
    if (last != 0 && now - last < 60'000'000'000) return;
    if (!gFontScanAt.compare_exchange_strong(last, now)) return;
    std::thread(scanFontsNow).detach();
}

// ---- Hide HUD: capture a draw and look up whether it is skipped --------------------------------

constexpr size_t kMaxHudDraws = 4096;

void hudDraw(Record& r, Pass& p, uint32_t n) {
    if (p.hudKey == 0) {
        bool swapchain = false;
        if (!p.colors.empty()) {
            std::lock_guard lock(gLock);
            const Image* i = imageOf(p.colors[0].image);
            swapchain = i != nullptr && i->swapchain;
        }
        p.hudKey = HudPassKey(p, swapchain);
    }
    const uint64_t pipeline = reinterpret_cast<uint64_t>(r.graphics);
    const HudLookup look = HudSkipTable().find(HudKey(p.hudKey, pipeline), p.draws);
    r.hudSkip = look.skip;
    if (look.skip) HudNoteSkipped();
    if (p.hudDraws.size() < kMaxHudDraws) {
        const bool test = r.pipeline.dynamicTest ? r.dynTest : r.pipeline.test;
        const bool write = test && (r.pipeline.dynamicWrite ? r.dynWrite : r.pipeline.write);
        const VkCompareOp op = r.pipeline.dynamicCompare ? r.dynCompare : r.pipeline.compare;
        HudDraw d;
        d.pipeline = pipeline;
        d.index = p.draws;
        d.elements = r.hudIndirect ? 0 : r.drawCount;
        d.instances = r.hudIndirect ? n : r.hudInstances;
        d.shader = r.pipeline.fragment;
        d.state = static_cast<uint16_t>(
                (r.pipeline.blend ? kDrawBlend : 0) | (test ? kDrawDepthTest : 0) |
                (write ? kDrawDepthWrite : 0) | (r.pipeline.stencilWrite ? kDrawStencilWrite : 0) |
                (test && op != VK_COMPARE_OP_ALWAYS ? kHudDepthCompare : 0) |
                (r.indexed && !r.hudIndirect ? kHudIndexed : 0) | (r.hudIndirect ? kHudIndirect : 0));
        for (int i = 0; i < kMaxSets; i++) {
            if (r.bound[0][i] != VK_NULL_HANDLE) d.textures |= r.hudTextures[i];
        }
        const uint64_t area = static_cast<uint64_t>(p.width) * p.height;
        const uint64_t scissor = static_cast<uint64_t>(r.scissor.width) * r.scissor.height;
        d.scissor = area == 0 || scissor == 0 || scissor >= area
                ? 255
                : static_cast<uint8_t>(scissor * 255 / area);
        d.vertex = r.pipeline.vertex;
        p.hudDraws.push_back(d);
    }
    // A few draws a frame keep their uniform bytes, until each pair's camera is known.
    if (!look.known && HudTakeSample()) {
        Record::FirstDraw sample;
        sample.event = r.open;
        sample.draw = p.draws;
        sample.pipeline = r.graphics;
        sample.hud = true;
        std::copy(std::begin(r.bound[0]), std::end(r.bound[0]), sample.sets);
        for (int i = 0; i < kMaxSets; i++) sample.offsets[i] = r.dynamicOffsets[i];
        r.firstDraws.push_back(std::move(sample));
    }
}

// Whether the draw just counted is one Hide HUD drops: not forwarded, nothing else changes.
bool hudSkips(VkCommandBuffer cb) {
    return recordOf(cb).hudSkip;
}

// ------------------------------------------------------------------------------------------------

void countDraw(VkCommandBuffer cb, uint32_t n) {
    tally(kDraw);
    Record& r = recordOf(cb);
    r.hudSkip = false;
    if (r.ignored) return;
    if (r.glass.armed && r.open >= 0) {
        Split g;
        uint32_t at = 0;
        {
            std::lock_guard lock(gLock);
            g = gGlass.opaque;
            at = gGlass.draw;
        }
        if (r.events[r.open].pass.draws == at && g.scene != VK_NULL_HANDLE) {
            splitPass(cb, r, r.glass, g, false);
        }
    }
    if (r.cut.armed && r.open >= 0 &&
        static_cast<int>(r.events[r.open].pass.draws) == survey::cutDraw()) {
        Split s;
        {
            std::lock_guard lock(gLock);
            s = gSplit;
        }
        if (s.scene != VK_NULL_HANDLE) cutNow(cb, r, s);
    }
    if (r.cut.armed && !r.cut.atClear && r.open >= 0) cutBefore(cb, r);
    Pass& p = current(r);
    // The survey keeps a few draws of every pass, to find where a game keeps its camera.
    if (r.open >= 0) {
        const bool fixed = r.pipeline.viewportHeight != 0;
        const float x = fixed ? r.pipeline.viewportX : r.viewportX;
        const float y = fixed ? r.pipeline.viewportY : r.viewportY;
        const float w = fixed ? r.pipeline.viewportWidth : r.viewportWidth;
        const float h = fixed ? r.pipeline.viewportHeight : r.viewportHeight;
        if (p.draws == 0) {
            p.viewportHeight = h;
            p.viewportX = x;
            p.viewportY = y;
            p.viewportWidth = w;
        } else if (x != p.lastViewportX || y != p.lastViewportY) {
            p.viewportChanges++;
            const bool known = std::any_of(p.tiles.begin(), p.tiles.end(), [&](const Pass::Tile& t) {
                return t.x == x && t.y == y;
            }) || (x == p.viewportX && y == p.viewportY);
            if (!known && p.tiles.size() < Pass::kMaxTiles && watches(p)) {
                p.tiles.push_back({x, y, w, h, {}});
                Record::FirstDraw first;
                first.event = r.open;
                first.draw = p.draws;
                first.pipeline = r.graphics;
                first.tile = static_cast<int>(p.tiles.size()) - 1;
                std::copy(std::begin(r.bound[0]), std::end(r.bound[0]), first.sets);
                for (int i = 0; i < kMaxSets; i++) first.offsets[i] = r.dynamicOffsets[i];
                r.firstDraws.push_back(std::move(first));
            }
        }
        p.lastViewportX = x;
        p.lastViewportY = y;
    }
    // The scene pass's first draws, to tell its camera from its objects' (FindViewProjection).
    const uint32_t sampled = survey::enabled() ? 3u
            : p.renderPass == gWatchedPass.load(std::memory_order_relaxed) ? 4u : 1u;
    if (p.draws < sampled && r.open >= 0 && watches(p)) {
        Record::FirstDraw first;
        first.event = r.open;
        first.draw = p.draws;
        first.pipeline = r.graphics;
        std::copy(std::begin(r.bound[0]), std::end(r.bound[0]), first.sets);
        for (int i = 0; i < kMaxSets; i++) first.offsets[i] = r.dynamicOffsets[i];
        r.firstDraws.push_back(std::move(first));
    }
    if (r.open >= 0 && HudCapturing()) hudDraw(r, p, n);
    p.draws += n;
    const bool test = r.pipeline.dynamicTest ? r.dynTest : r.pipeline.test;
    const bool write = test && (r.pipeline.dynamicWrite ? r.dynWrite : r.pipeline.write);
    if (survey::enabled() && r.open >= 0 && p.drawTrace.size() < kMaxDrawTrace) {
        const VkCompareOp op = r.pipeline.dynamicCompare ? r.dynCompare : r.pipeline.compare;
        const char kind = write ? 'w' : test && op != VK_COMPARE_OP_ALWAYS ? 't' : 'a';
        p.drawTrace += r.pipeline.blend ? static_cast<char>(kind - 'a' + 'A') : kind;
        if (survey::drawList() && p.draws > 8) {
            std::string line = android::base::StringPrintf(
                    "survey     d%u %c n=%u sc=%ux%u", p.draws, p.drawTrace.back(), r.drawCount,
                    r.scissor.width, r.scissor.height);
            std::lock_guard lock(gLock);
            int shown = 0;
            for (VkDescriptorSet set : r.bound[0]) {
                const auto found = set == VK_NULL_HANDLE ? gSetImages.end() : gSetImages.find(set);
                if (found == gSetImages.end()) continue;
                for (const auto& entry : found->second) {
                    const Image* i = imageOf(entry.second);
                    if (i == nullptr || i->width < 32 || ++shown > 4) continue;
                    android::base::StringAppendF(&line, " #%u f%d %ux%u", i->id, i->format,
                                                 i->width, i->height);
                }
            }
            ALOGI("%s", line.c_str());
        }
    }
    if (test) p.depthTests += n;
    if (write) p.depthWrites += n;
    if (write) {
        p.lastDepthWrite = p.draws - n;
        p.glassFrom = UINT32_MAX;
    } else if (test && r.pipeline.blend && p.glassFrom == UINT32_MAX) {
        p.glassFrom = p.draws - n;
    }
    if (signalsOn()) {
        const uint32_t state = (r.pipeline.blend ? kDrawBlend : 0) | (test ? kDrawDepthTest : 0) |
                (write ? kDrawDepthWrite : 0) | (r.pipeline.stencilWrite ? kDrawStencilWrite : 0);
        if (p.drawsByState[state] == 0) p.firstByState[state] = p.draws - n;
        p.drawsByState[state] += n;
    }
    if (r.takenVersion != r.boundVersion[0]) {
        r.takenVersion = r.boundVersion[0];
        addSets(&currentSets(r), r.bound[0], kMaxSets);
    }
    if (survey::atlas() && survey::shooting() && r.open >= 0) noteTextDraw(r, p);
    if (gTextCost.sampling.load(std::memory_order_relaxed) && r.open >= 0 && r.pipeline.blend && !write) {
        sampleTextDraw(r);
    }
}

VKAPI_ATTR void CmdDraw(VkCommandBuffer cb, uint32_t vertices, uint32_t instances, uint32_t first,
                        uint32_t firstInstance) {
    Record& record = recordOf(cb);
    record.drawCount = vertices;
    record.indexed = false;
    record.firstElement = first;
    record.vertexOffset = 0;
    record.hudInstances = instances;
    record.hudIndirect = false;
    countDraw(cb, 1);
    if (record.hudSkip) return;
    gNext.CmdDraw(cb, vertices, instances, first, firstInstance);
}

VKAPI_ATTR void CmdDrawIndexed(VkCommandBuffer cb, uint32_t indices, uint32_t instances,
                               uint32_t first, int32_t offset, uint32_t firstInstance) {
    Record& record = recordOf(cb);
    record.drawCount = indices;
    record.indexed = true;
    record.firstElement = first;
    record.vertexOffset = offset;
    record.hudInstances = instances;
    record.hudIndirect = false;
    countDraw(cb, 1);
    if (record.hudSkip) return;
    if (alihOn()) {
        const int64_t start = alihCostOn() ? nowNs() : 0;
        const bool drawn = alihDraw(cb, record, indices, instances, first, offset, firstInstance);
        if (start != 0) {
            const int64_t end = nowNs();
            std::lock_guard lock(gLock);
            gAlihCost.drawNs += end - start;
            gAlihCost.calls++;
        }
        if (drawn) return;
    }
    gNext.CmdDrawIndexed(cb, indices, instances, first, offset, firstInstance);
}

VKAPI_ATTR void CmdDrawIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                uint32_t count, uint32_t stride) {
    recordOf(cb).hudIndirect = true;
    countDraw(cb, count);
    if (hudSkips(cb)) return;
    gNext.CmdDrawIndirect(cb, buffer, offset, count, stride);
}

VKAPI_ATTR void CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                       uint32_t count, uint32_t stride) {
    recordOf(cb).hudIndirect = true;
    countDraw(cb, count);
    if (hudSkips(cb)) return;
    gNext.CmdDrawIndexedIndirect(cb, buffer, offset, count, stride);
}

VKAPI_ATTR void CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                     VkBuffer countBuffer, VkDeviceSize countOffset,
                                     uint32_t max, uint32_t stride) {
    recordOf(cb).hudIndirect = true;
    countDraw(cb, 1);
    if (hudSkips(cb)) return;
    gNext.CmdDrawIndirectCount(cb, buffer, offset, countBuffer, countOffset, max, stride);
}

VKAPI_ATTR void CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buffer,
                                            VkDeviceSize offset, VkBuffer countBuffer,
                                            VkDeviceSize countOffset, uint32_t max,
                                            uint32_t stride) {
    recordOf(cb).hudIndirect = true;
    countDraw(cb, 1);
    if (hudSkips(cb)) return;
    gNext.CmdDrawIndexedIndirectCount(cb, buffer, offset, countBuffer, countOffset, max, stride);
}

// Which way a blit region turns the image: a flipped source or destination range.
uint8_t flipsOf(const VkOffset3D* src, const VkOffset3D* dst) {
    uint8_t f = 0;
    if ((src[1].y < src[0].y) != (dst[1].y < dst[0].y)) f |= kFlipY;
    if ((src[1].x < src[0].x) != (dst[1].x < dst[0].x)) f |= kFlipX;
    return f;
}

void noteOp(VkCommandBuffer cb, OpKind kind, VkImage src, VkImage dst, uint8_t flips = 0) {
    Step e;
    e.op.kind = kind;
    e.op.flips = flips;
    {
        std::lock_guard lock(gLock);
        e.op.src = idOfImage(src);
        e.op.dst = idOfImage(dst);
    }
    recordOf(cb).push(std::move(e));
}

VKAPI_ATTR void CmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, VkImage dst,
                             VkImageLayout dstLayout, uint32_t count, const VkImageCopy* regions) {
    noteOp(cb, OpKind::kCopy, src, dst);
    gNext.CmdCopyImage(cb, src, srcLayout, dst, dstLayout, count, regions);
}

VKAPI_ATTR void CmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* info) {
    noteOp(cb, OpKind::kCopy, info->srcImage, info->dstImage);
    gNext.CmdCopyImage2(cb, info);
}

VKAPI_ATTR void CmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, VkImage dst,
                             VkImageLayout dstLayout, uint32_t count, const VkImageBlit* regions,
                             VkFilter filter) {
    noteOp(cb, OpKind::kBlit, src, dst,
           count > 0 ? flipsOf(regions[0].srcOffsets, regions[0].dstOffsets) : 0);
    gNext.CmdBlitImage(cb, src, srcLayout, dst, dstLayout, count, regions, filter);
}

VKAPI_ATTR void CmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* info) {
    noteOp(cb, OpKind::kBlit, info->srcImage, info->dstImage,
           info->regionCount > 0
                   ? flipsOf(info->pRegions[0].srcOffsets, info->pRegions[0].dstOffsets)
                   : 0);
    gNext.CmdBlitImage2(cb, info);
}

VKAPI_ATTR void CmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout,
                                VkImage dst, VkImageLayout dstLayout, uint32_t count,
                                const VkImageResolve* regions) {
    noteOp(cb, OpKind::kResolve, src, dst);
    gNext.CmdResolveImage(cb, src, srcLayout, dst, dstLayout, count, regions);
}

VKAPI_ATTR void CmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2* info) {
    noteOp(cb, OpKind::kResolve, info->srcImage, info->dstImage);
    gNext.CmdResolveImage2(cb, info);
}

VKAPI_ATTR void CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src, VkImageLayout layout,
                                     VkBuffer dst, uint32_t count,
                                     const VkBufferImageCopy* regions) {
    noteOp(cb, OpKind::kToBuffer, src, VK_NULL_HANDLE);
    gNext.CmdCopyImageToBuffer(cb, src, layout, dst, count, regions);
}

VKAPI_ATTR void CmdClearDepthStencilImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
                                          const VkClearDepthStencilValue* value, uint32_t count,
                                          const VkImageSubresourceRange* ranges) {
    Step e;
    e.op.kind = OpKind::kClear;
    {
        std::lock_guard lock(gLock);
        e.op.dst = idOfImage(image);
    }
    e.op.depth = value->depth;
    recordOf(cb).push(std::move(e));
    gNext.CmdClearDepthStencilImage(cb, image, layout, value, count, ranges);
}

void noteBarrier(Record& r, VkImage image, VkImageLayout from, VkImageLayout to) {
    if (from == to) return;
    Step e;
    e.op.kind = OpKind::kLayout;
    {
        std::lock_guard lock(gLock);
        e.op.dst = idOfImage(image);
    }
    if (e.op.dst == 0) return;
    e.op.from = from;
    e.op.to = to;
    r.push(std::move(e));
}

VKAPI_ATTR void CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags srcStages,
                                   VkPipelineStageFlags dstStages, VkDependencyFlags flags,
                                   uint32_t memoryCount, const VkMemoryBarrier* memory,
                                   uint32_t bufferCount, const VkBufferMemoryBarrier* buffers,
                                   uint32_t imageCount, const VkImageMemoryBarrier* images) {
    Record& r = recordOf(cb);
    for (uint32_t i = 0; i < imageCount; i++) {
        noteBarrier(r, images[i].image, images[i].oldLayout, images[i].newLayout);
    }
    gNext.CmdPipelineBarrier(cb, srcStages, dstStages, flags, memoryCount, memory, bufferCount,
                             buffers, imageCount, images);
}

VKAPI_ATTR void CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* info) {
    Record& r = recordOf(cb);
    for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
        const VkImageMemoryBarrier2& b = info->pImageMemoryBarriers[i];
        noteBarrier(r, b.image, b.oldLayout, b.newLayout);
    }
    gNext.CmdPipelineBarrier2(cb, info);
}

// Callers hold gLock.
void resolveReads(const std::vector<VkDescriptorSet>& sets, std::vector<ImageId>* into) {
    for (VkDescriptorSet set : sets) {
        const auto found = gSetImages.find(set);
        if (found == gSetImages.end()) continue;
        for (const auto& [key, id] : found->second) into->push_back(id);
    }
    std::sort(into->begin(), into->end());
    into->erase(std::unique(into->begin(), into->end()), into->end());
}

// Callers hold gLock. The bytes of each uniform buffer the first draw read, as mapped now: the game
// wrote them before submitting.
void resolveConstants(const Record::FirstDraw& draw, std::vector<Constants>* into) {
    constexpr VkDeviceSize kMaxBytes = 4096;
    std::lock_guard mapLock(gMapLock);
    for (uint32_t set = 0; set < kMaxSets; set++) {
        if (draw.sets[set] == VK_NULL_HANDLE) continue;
        const auto bindings = gSetBuffers.find(draw.sets[set]);
        if (bindings == gSetBuffers.end()) continue;
        std::vector<uint32_t> order;  // the set's dynamic bindings, to index its offsets
        if (!draw.offsets[set].empty()) {
            const auto layout = gSetLayoutOf.find(draw.sets[set]);
            const auto desc = layout == gSetLayoutOf.end() ? gSetLayouts.end()
                                                           : gSetLayouts.find(layout->second);
            if (desc != gSetLayouts.end()) order = desc->second.dynamicBindings;
        }
        for (const auto& [key, b] : bindings->second) {
            const auto buffer = gBuffers.find(b.buffer);
            if (buffer == gBuffers.end() || buffer->second.memory == VK_NULL_HANDLE) continue;
            VkDeviceSize offset = b.offset;
            if (b.dynamic) {
                const uint32_t slot = static_cast<uint32_t>(key >> 32) << 16 |
                        static_cast<uint32_t>(key & 0xffff);
                const auto at = std::find(order.begin(), order.end(), slot);
                const size_t k = at - order.begin();
                if (at == order.end() || k >= draw.offsets[set].size()) continue;
                offset += draw.offsets[set][k];
            }
            VkDeviceSize range = b.range == VK_WHOLE_SIZE
                    ? (buffer->second.size > offset ? buffer->second.size - offset : 0)
                    : b.range;
            range = std::min(range, kMaxBytes) & ~VkDeviceSize{3};
            const auto mapping = gMappings.find(buffer->second.memory);
            if (range == 0 || mapping == gMappings.end() || mapping->second.ptr == nullptr) continue;
            const VkDeviceSize at = buffer->second.offset + offset;
            const Mapping& m = mapping->second;
            if (at < m.offset || at + range > m.offset + m.size) continue;
            Constants c;
            c.draw = draw.draw;
            const auto fragment = gFragmentOf.find(draw.pipeline);
            if (fragment != gFragmentOf.end()) c.shader = fragment->second;
            c.set = set;
            c.binding = static_cast<uint32_t>(key >> 32);
            c.memory = reinterpret_cast<uint64_t>(buffer->second.memory);
            c.offset = at;
            c.values.resize(range / sizeof(float));
            memcpy(c.values.data(), m.ptr + (at - m.offset), range);
            into->push_back(std::move(c));
        }
    }
}

// Engine signals: the addresses the next frame's camera may land at, and what they held at each
// submit since the last present. Callers hold gLock.
std::vector<SignalWatch> gSignalWatch;
std::vector<SignalSample> gSignalSamples;
constexpr size_t kMaxSignalSamples = 256;

void sampleWatched(int64_t now) {
    if (gSignalWatch.empty() || gSignalSamples.size() + gSignalWatch.size() > kMaxSignalSamples) return;
    std::lock_guard mapLock(gMapLock);
    for (const SignalWatch& w : gSignalWatch) {
        const auto found = gMappings.find(reinterpret_cast<VkDeviceMemory>(w.memory));
        if (found == gMappings.end() || found->second.ptr == nullptr) continue;
        const Mapping& m = found->second;
        if (w.offset < m.offset || w.offset + 64 > m.offset + m.size) continue;
        SignalSample sample{w.memory, w.offset, now, w.rowMajor, {}};
        memcpy(sample.floats, m.ptr + (w.offset - m.offset), sizeof(sample.floats));
        gSignalSamples.push_back(sample);
    }
}

int64_t nowNs();

// Between submits, at most every 500 us: how early the next camera lands is measured at this grain.
std::atomic<int64_t> gSampledAt{0};

void sampleSoon() {
    const int64_t now = nowNs();
    if (now - gSampledAt.load(std::memory_order_relaxed) < 500'000) return;
    gSampledAt.store(now, std::memory_order_relaxed);
    std::lock_guard lock(gLock);
    sampleWatched(now);
}

void collect(VkQueue queue, const VkCommandBuffer* cbs, uint32_t count) {
    std::lock_guard lock(gLock);
    if (signalsOn()) sampleWatched(nowNs());
    if (gCollected.size() > kMaxCollected) gCollected.clear();
    Step mark;
    mark.op.kind = OpKind::kSubmit;
    mark.op.src = static_cast<ImageId>(reinterpret_cast<uintptr_t>(queue) & 0xffff);
    gCollected.push_back(std::move(mark));
    for (uint32_t i = 0; i < count; i++) {
        const auto found = gRecords.find(cbs[i]);
        if (found == gRecords.end()) continue;
        const Record& r = *found->second;
        size_t next = 0;  // r.firstDraws are in event order
        for (size_t e = 0; e < r.events.size(); e++) {
            gCollected.push_back(r.events[e]);
            Step& s = gCollected.back();
            if (!r.sets[e].empty()) resolveReads(r.sets[e], s.isPass ? &s.pass.reads : &s.op.reads);
            while (next < r.firstDraws.size() && r.firstDraws[next].event < static_cast<int>(e)) {
                next++;
            }
            for (; next < r.firstDraws.size() && r.firstDraws[next].event == static_cast<int>(e);
                 next++) {
                const Record::FirstDraw& first = r.firstDraws[next];
                if (!s.isPass) continue;
                if (first.tile >= 0 && first.tile < static_cast<int>(s.pass.tiles.size())) {
                    resolveConstants(first, &s.pass.tiles[first.tile].constants);
                } else {
                    resolveConstants(first, first.hud ? &s.pass.hudConstants : &s.pass.constants);
                }
            }
        }
    }
}

VKAPI_ATTR VkResult QueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits,
                                VkFence fence) {
    tally(kSubmit);
    for (uint32_t s = 0; s < count; s++) {
        collect(queue, submits[s].pCommandBuffers, submits[s].commandBufferCount);
    }
    return gNext.QueueSubmit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult QueueSubmit2(VkQueue queue, uint32_t count, const VkSubmitInfo2* submits,
                                 VkFence fence) {
    tally(kSubmit2);
    for (uint32_t s = 0; s < count; s++) {
        std::vector<VkCommandBuffer> cbs;
        for (uint32_t i = 0; i < submits[s].commandBufferInfoCount; i++) {
            cbs.push_back(submits[s].pCommandBufferInfos[i].commandBuffer);
        }
        collect(queue, cbs.data(), static_cast<uint32_t>(cbs.size()));
    }
    return gNext.QueueSubmit2(queue, count, submits, fence);
}

// ---- Split ------------------------------------------------------------------------------------

// Callers hold gLock. The game's pass with every attachment loaded where it was left.
VkRenderPass makeResume(const DeviceInfo& d, VkDevice device, const PassRecipe& recipe) {
    std::vector<VkAttachmentDescription2> attachments = recipe.attachments;
    for (VkAttachmentDescription2& a : attachments) {
        // What the first half did not store starts over as it did there.
        if (a.storeOp == VK_ATTACHMENT_STORE_OP_STORE) a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        if (a.stencilStoreOp == VK_ATTACHMENT_STORE_OP_STORE) {
            a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        }
        a.initialLayout = a.finalLayout;
    }
    VkSubpassDescription2 sub = {};
    sub.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
    sub.flags = recipe.flags;
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.inputAttachmentCount = static_cast<uint32_t>(recipe.inputs.size());
    sub.pInputAttachments = recipe.inputs.data();
    sub.colorAttachmentCount = static_cast<uint32_t>(recipe.colors.size());
    sub.pColorAttachments = recipe.colors.data();
    sub.pResolveAttachments = recipe.resolves.empty() ? nullptr : recipe.resolves.data();
    sub.pDepthStencilAttachment = recipe.hasDepth ? &recipe.depth : nullptr;
    sub.preserveAttachmentCount = static_cast<uint32_t>(recipe.preserve.size());
    sub.pPreserveAttachments = recipe.preserve.data();
    VkRenderPass pass = VK_NULL_HANDLE;
    if (d.fn.CreateRenderPass2 != nullptr) {
        VkRenderPassCreateInfo2 info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.subpassCount = 1;
        info.pSubpasses = &sub;
        info.dependencyCount = static_cast<uint32_t>(recipe.dependencies.size());
        info.pDependencies = recipe.dependencies.data();
        return d.fn.CreateRenderPass2(device, &info, nullptr, &pass) == VK_SUCCESS ? pass
                                                                                 : VK_NULL_HANDLE;
    }
    // A 1.0 device: the same pass narrowed back, which a plain recipe loses nothing to.
    std::vector<VkAttachmentDescription> narrow;
    for (const VkAttachmentDescription2& a : attachments) {
        narrow.push_back({a.flags, a.format, a.samples, a.loadOp, a.storeOp, a.stencilLoadOp,
                          a.stencilStoreOp, a.initialLayout, a.finalLayout});
    }
    const auto refs = [](const std::vector<VkAttachmentReference2>& from) {
        std::vector<VkAttachmentReference> out;
        for (const VkAttachmentReference2& r : from) out.push_back({r.attachment, r.layout});
        return out;
    };
    const std::vector<VkAttachmentReference> colors = refs(recipe.colors);
    const std::vector<VkAttachmentReference> resolves = refs(recipe.resolves);
    const std::vector<VkAttachmentReference> inputs = refs(recipe.inputs);
    const VkAttachmentReference depth = {recipe.depth.attachment, recipe.depth.layout};
    VkSubpassDescription one = {};
    one.flags = recipe.flags;
    one.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    one.inputAttachmentCount = static_cast<uint32_t>(inputs.size());
    one.pInputAttachments = inputs.data();
    one.colorAttachmentCount = static_cast<uint32_t>(colors.size());
    one.pColorAttachments = colors.data();
    one.pResolveAttachments = resolves.empty() ? nullptr : resolves.data();
    one.pDepthStencilAttachment = recipe.hasDepth ? &depth : nullptr;
    one.preserveAttachmentCount = static_cast<uint32_t>(recipe.preserve.size());
    one.pPreserveAttachments = recipe.preserve.data();
    std::vector<VkSubpassDependency> dependencies;
    for (const VkSubpassDependency2& dep : recipe.dependencies) {
        dependencies.push_back({dep.srcSubpass, dep.dstSubpass, dep.srcStageMask, dep.dstStageMask,
                                dep.srcAccessMask, dep.dstAccessMask, dep.dependencyFlags});
    }
    VkRenderPassCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = static_cast<uint32_t>(narrow.size());
    info.pAttachments = narrow.data();
    info.subpassCount = 1;
    info.pSubpasses = &one;
    info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    info.pDependencies = dependencies.data();
    return d.fn.CreateRenderPass(device, &info, nullptr, &pass) == VK_SUCCESS ? pass
                                                                            : VK_NULL_HANDLE;
}

// Callers hold gLock. An image the split copies the target into, sampled at present.
bool makeScene(const DeviceInfo& d, VkDevice device, Split* s) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = s->format;
    info.extent = {s->width, s->height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d.fn.CreateImage(device, &info, nullptr, &s->scene) != VK_SUCCESS) return false;
    VkMemoryRequirements need = {};
    d.fn.GetImageMemoryRequirements(device, s->scene, &need);
    uint32_t type = UINT32_MAX;
    for (uint32_t t = 0; t < d.memory.memoryTypeCount && type == UINT32_MAX; t++) {
        const VkMemoryPropertyFlags flags = d.memory.memoryTypes[t].propertyFlags;
        if ((need.memoryTypeBits & (1u << t)) && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)) {
            type = t;
        }
    }
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = need.size;
    alloc.memoryTypeIndex = type;
    if (type == UINT32_MAX ||
        d.fn.AllocateMemory(device, &alloc, nullptr, &s->memory) != VK_SUCCESS ||
        d.fn.BindImageMemory(device, s->scene, s->memory, 0) != VK_SUCCESS) {
        if (s->memory != VK_NULL_HANDLE) d.fn.FreeMemory(device, s->memory, nullptr);
        d.fn.DestroyImage(device, s->scene, nullptr);
        s->scene = VK_NULL_HANDLE;
        s->memory = VK_NULL_HANDLE;
        return false;
    }
    Image image;
    image.format = info.format;
    image.width = s->width;
    image.height = s->height;
    image.usage = info.usage;
    s->sceneId = newImage(s->scene, image);
    return true;
}

// Callers hold gLock. Why the pass that tonemaps into the target cannot be split; null if it can.
const char* splitRefusal(const Pass& p, const RenderPassDesc* d, const Target* target,
                         const Image* image, bool atClear) {
    if (p.dynamic || d == nullptr) return "dynamic rendering";
    if (d->recipe == nullptr) return "not a plain single-subpass render pass";
    if (target == nullptr || image == nullptr) return "the target is not an attachment";
    if (!(image->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return "the target cannot be copied";
    for (const VkAttachmentDescription2& a : d->recipe->attachments) {
        const bool stencil = (aspectsOf(a.format) & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
        if (a.samples != VK_SAMPLE_COUNT_1_BIT) return "the pass is multisampled";
        // Depth the pass never writes, or clears right at the split, starts over the same way.
        const bool lost = a.storeOp != VK_ATTACHMENT_STORE_OP_STORE ||
                (stencil && a.stencilStoreOp != VK_ATTACHMENT_STORE_OP_STORE);
        if (lost && (!IsDepthFormat(a.format) || (p.depthWrites > 0 && !atClear))) {
            return "an attachment is not stored";
        }
    }
    return nullptr;
}

// Callers hold gLock. Arms the split for the pass `roles` found tonemapping into the target.
void armSplit(const Frame& frame, const Roles& roles, VkSwapchainKHR swapchain) {
    static uint64_t refused = 0;
    if ((roles.miss != Miss::kUntoned && !roles.clearSplit) || roles.split < 0) return;
    const Pass& p = frame.steps[roles.split].pass;
    if (p.renderPass == 0 || p.renderPass == refused) return;
    const std::string package = Engine::get().package();
    const Target* target = nullptr;
    for (const Target& t : p.colors) {
        if (t.image == roles.target) target = &t;
    }
    const Image* image = target == nullptr ? nullptr : imageOf(target->image);
    if (p.renderPass == handleBits(gSplit.pass) && image != nullptr &&
        image->format == gSplit.format && image->width == gSplit.width &&
        image->height == gSplit.height) {
        gSplit.clearIndex = roles.clearSplit ? p.depthClears - 1 : 0;
        if (const Image* u = imageOf(roles.untoned)) gSplit.untoned = u->format;
        // Armed for this pass, yet the frame came through whole.
        if (++gSplit.missed == kSplitPatience) {
            ALOGW("lapis %s: the split pass keeps reaching the present unsplit", package.c_str());
        }
        return;
    }
    const auto sc = gSwapchains.find(swapchain);
    const auto dev = sc == gSwapchains.end() ? gDevices.end() : gDevices.find(sc->second.device);
    if (dev == gDevices.end() || !dev->second.owned) {
        refused = p.renderPass;
        ALOGW("lapis %s: no split: %s", package.c_str(),
              dev == gDevices.end() ? "the swapchain's device is unknown"
                                    : "the device lacks an entry point");
        return;
    }
    const VkRenderPass handle = reinterpret_cast<VkRenderPass>(static_cast<uintptr_t>(p.renderPass));
    const auto rp = gRenderPasses.find(handle);
    const RenderPassDesc* d = rp == gRenderPasses.end() ? nullptr : &rp->second;
    if (const char* why = splitRefusal(p, d, target, image, roles.clearSplit)) {
        refused = p.renderPass;
        ALOGI("lapis %s: no split: %s", package.c_str(), why);
        return;
    }
    const VkDevice device = sc->second.device;
    auto resume = gResumes.find(handle);
    if (resume == gResumes.end()) {
        const VkRenderPass made = makeResume(dev->second, device, *d->recipe);
        if (made != VK_NULL_HANDLE) resume = gResumes.emplace(handle, std::pair(device, made)).first;
    }
    if (gSplit.scene != VK_NULL_HANDLE &&
        (gSplit.device != device || gSplit.format != image->format ||
         gSplit.width != image->width || gSplit.height != image->height)) {
        retireSplit();
    }
    gSplit.device = device;
    gSplit.fn = dev->second.fn;
    gSplit.format = image->format;
    gSplit.width = image->width;
    gSplit.height = image->height;
    if (resume == gResumes.end() ||
        (gSplit.scene == VK_NULL_HANDLE && !makeScene(dev->second, device, &gSplit))) {
        gSplit.pass = VK_NULL_HANDLE;
        refused = p.renderPass;
        ALOGW("lapis %s: no split: out of memory", package.c_str());
        return;
    }
    gSplit.pass = handle;
    gSplit.resume = resume->second.second;
    gSplit.attachment = target->attachment;
    gSplit.final = d->recipe->attachments[target->attachment].finalLayout;
    gSplit.logged = false;
    gSplit.missed = 0;
    gSplit.atClear = roles.clearSplit;
    gSplit.clearIndex = roles.clearSplit ? p.depthClears - 1 : 0;
    const Image* untoned = imageOf(roles.untoned);
    gSplit.untoned = untoned != nullptr ? untoned->format : VK_FORMAT_UNDEFINED;
    if (roles.clearSplit) {
        ALOGI("lapis %s: splitting the pass into #%u at its depth clear %u, the switch to the HUD",
              package.c_str(), roles.target, gSplit.clearIndex);
    } else {
        ALOGI("lapis %s: splitting the pass that tonemaps #%u into #%u, at its first HUD draw",
              package.c_str(), roles.untoned, roles.target);
    }
}

// Callers hold gLock. Arms the transparency cut lapis/rust/src/glass.rs finds, once it holds still;
// drops it on a graph change. debug.framefx.lapis_transparency=off turns it off for A/B runs.
void armGlass(const Frame& frame, const Roles& roles, VkSwapchainKHR swapchain) {
    static const bool off = [] {
        char value[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.framefx.lapis_transparency", value) > 0 &&
                strcmp(value, "off") == 0;
    }();
    if (off) return;
    // Asked once the cut is about to run for the first time, never before.
    static int allowed = -1;
    static_assert(sizeof(gGlassTracker) >= 32, "glass.rs GlassTracker");
    if (GlassTrackerSize() > sizeof(gGlassTracker)) return;
    const GlassCut cut = FindGlass(frame, gImages, roles.depth, gGlassTracker);
    const std::string package = Engine::get().package();
    if (!cut.armed) {
        if (gGlass.kind != 0) {
            ALOGI("lapis %s: transparency cut dropped (%d)", package.c_str(), cut.refused);
            retireGlass();
        }
        return;
    }
    // Patience held the previous frame's armed cut across an odd frame: keep gGlass as armed and
    // never index this frame's steps with the previous frame's step index.
    if (cut.held) return;
    if (cut.step < 0 || static_cast<size_t>(cut.step) >= frame.steps.size()) return;
    if (allowed < 0) allowed = safety::allowed(safety::kTransparencyCut) ? 1 : 0;
    if (allowed == 0) return;
    const Step& step = frame.steps[cut.step];
    if (cut.kind == kGlassRefraction) {
        if (gGlass.kind != kGlassRefraction || gGlass.refraction != cut.target) {
            retireGlass();
            gGlass.kind = kGlassRefraction;
            gGlass.refraction = cut.target;
            ALOGI("lapis %s: the game's own copy #%u is the opaque layer", package.c_str(), cut.target);
        }
        return;
    }
    if (!step.isPass) return;
    // In one pass the cut pass is also where the lit scene ends; with transparency in a pass of
    // its own, the opaque pass is the last depth-writing one before it.
    int opaqueStep = cut.step;
    if (cut.kind == kGlassOwnPass) {
        for (int k = cut.step - 1; k >= 0; k--) {
            const Step& s = frame.steps[k];
            if (s.isPass && s.pass.writes(cut.target) && s.pass.depthWrites > 0) {
                opaqueStep = k;
                break;
            }
        }
        if (opaqueStep == cut.step) return;
    }
    const Pass& cutPass = frame.steps[opaqueStep].pass;
    const Pass& litPass = step.pass;
    if (cutPass.renderPass == 0 || litPass.renderPass == 0 || cutPass.renderPass == gGlass.refused) {
        return;  // dynamic rendering: not yet
    }
    const Target* target = nullptr;
    const Target* litTarget = nullptr;
    for (const Target& t : cutPass.colors) {
        if (t.image == cut.target) target = &t;
    }
    for (const Target& t : litPass.colors) {
        if (t.image == cut.target) litTarget = &t;
    }
    const Image* image = target == nullptr ? nullptr : imageOf(target->image);
    if (image == nullptr || litTarget == nullptr) return;
    const VkRenderPass cutHandle = reinterpret_cast<VkRenderPass>(static_cast<uintptr_t>(cutPass.renderPass));
    const VkRenderPass litHandle = reinterpret_cast<VkRenderPass>(static_cast<uintptr_t>(litPass.renderPass));
    if (gGlass.cutPass == cutHandle && gGlass.litPass == litHandle && gGlass.kind == cut.kind &&
        gGlass.draw == cut.draw && gGlass.opaque.format == image->format &&
        gGlass.opaque.width == image->width && gGlass.opaque.height == image->height) {
        return;  // armed as it is
    }
    const auto sc = gSwapchains.find(swapchain);
    const auto dev = sc == gSwapchains.end() ? gDevices.end() : gDevices.find(sc->second.device);
    const auto rp = gRenderPasses.find(cutHandle);
    if (dev == gDevices.end() || !dev->second.owned || rp == gRenderPasses.end()) return;
    const RenderPassDesc& d = rp->second;
    if (const char* why = splitRefusal(cutPass, &d, target, image, false)) {
        gGlass.refused = cutPass.renderPass;
        ALOGI("lapis %s: no transparency cut: %s", package.c_str(), why);
        return;
    }
    const VkDevice device = sc->second.device;
    auto resume = gResumes.find(cutHandle);
    if (resume == gResumes.end() && cut.kind == kGlassInPass) {
        const VkRenderPass made = makeResume(dev->second, device, *d.recipe);
        if (made != VK_NULL_HANDLE) resume = gResumes.emplace(cutHandle, std::pair(device, made)).first;
    }
    retireGlass();
    for (Split* s : {&gGlass.opaque, &gGlass.lit}) {
        s->device = device;
        s->fn = dev->second.fn;
        s->format = image->format;
        s->width = image->width;
        s->height = image->height;
    }
    if ((cut.kind == kGlassInPass && resume == gResumes.end()) ||
        !makeScene(dev->second, device, &gGlass.opaque) ||
        !makeScene(dev->second, device, &gGlass.lit)) {
        retireGlass();
        gGlass.refused = cutPass.renderPass;
        ALOGW("lapis %s: no transparency cut: out of memory", package.c_str());
        return;
    }
    gGlass.kind = cut.kind;
    gGlass.cutPass = cutHandle;
    gGlass.litPass = litHandle;
    gGlass.draw = cut.draw;
    if (resume != gResumes.end()) gGlass.opaque.resume = resume->second.second;
    gGlass.opaque.attachment = target->attachment;
    gGlass.opaque.final = target->final;
    gGlass.litAttachment = litTarget->attachment;
    gGlass.litFinal = litTarget->final;
    ALOGI("lapis %s: transparency cut %s #%u at draw %u", package.c_str(),
          cut.kind == kGlassInPass ? "inside the scene pass into" : "before its own pass, from",
          cut.target, cut.draw);
}

// Callers hold gLock. The transparency layers, while the cut runs.
void glassTaps(Taps* taps) {
    const auto tap = [](const Split& s, Tap* out) {
        if (s.scene == VK_NULL_HANDLE) return;
        out->image = s.scene;
        out->format = s.format;
        out->width = s.width;
        out->height = s.height;
        out->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        out->usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        out->id = s.sceneId;
    };
    if (gGlass.kind == kGlassRefraction) {
        const auto handle = gHandles.find(gGlass.refraction);
        const Image* image = imageOf(gGlass.refraction);
        if (handle != gHandles.end() && image != nullptr) {
            taps->opaque.image = handle->second;
            taps->opaque.format = image->format;
            taps->opaque.width = image->width;
            taps->opaque.height = image->height;
            taps->opaque.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            taps->opaque.usage = image->usage;
            taps->opaque.id = gGlass.refraction;
        }
        return;
    }
    tap(gGlass.opaque, &taps->opaque);
    tap(gGlass.lit, &taps->lit);
}

// Callers hold gLock. Arms the copy of a depth `roles` found stored by the scene's pass and not
// kept to the present; disarms it once the game keeps its depth itself.
void armKeep(const Frame& frame, const Roles& roles, VkSwapchainKHR swapchain) {
    if (roles.depth == 0 || roles.geometry < 0) return;
    if (roles.depth == gKeep.id) return;  // kept, through the copy
    const Pass& p = frame.steps[roles.geometry].pass;
    const Target* source = p.hasDepth && p.depth.image == roles.depth ? &p.depth : nullptr;
    for (const Target& t : p.colors) {
        if (t.image == roles.depth) source = &t;
    }
    const Image* image = imageOf(roles.depth);
    if (roles.depthStored || source == nullptr || image == nullptr || p.dynamic ||
        p.renderPass == 0 || !(image->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        if (gKeep.pass != VK_NULL_HANDLE && roles.depthStored) gKeep.pass = VK_NULL_HANDLE;
        return;
    }
    const VkRenderPass handle = reinterpret_cast<VkRenderPass>(static_cast<uintptr_t>(p.renderPass));
    if (gKeep.pass == handle && gKeep.source == roles.depth) return;
    const auto sc = gSwapchains.find(swapchain);
    const auto dev = sc == gSwapchains.end() ? gDevices.end() : gDevices.find(sc->second.device);
    if (dev == gDevices.end() || !dev->second.owned) return;
    const VkDevice device = sc->second.device;
    if (gKeep.image != VK_NULL_HANDLE &&
        (gKeep.device != device || gKeep.format != image->format ||
         gKeep.width != image->width || gKeep.height != image->height)) {
        retireKeep();
    }
    if (gKeep.image == VK_NULL_HANDLE) {
        Split made;
        made.format = image->format;
        made.width = image->width;
        made.height = image->height;
        if (!makeScene(dev->second, device, &made)) return;
        gKeep.image = made.scene;
        gKeep.memory = made.memory;
        gKeep.id = made.sceneId;
        gKeep.format = made.format;
        gKeep.width = made.width;
        gKeep.height = made.height;
        gKeep.logged = false;
    }
    gKeep.device = device;
    gKeep.fn = dev->second.fn;
    gKeep.pass = handle;
    gKeep.source = roles.depth;
    gKeep.final = source->final;
}

// ---- Classification --------------------------------------------------------------------------

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

// Whether the presented image only shows, through transfers, an image no pass of this frame
// drew: the game drew it before its last present, and the passes are with that frame's steps.
bool ShowsAFrameBefore(const Frame& frame) {
    ImageId target = frame.presented;
    bool moved = false;
    for (auto it = frame.steps.rbegin(); it != frame.steps.rend() && target != 0; ++it) {
        if (it->isPass) {
            if (it->pass.writes(target)) return false;
        } else if (it->op.transfers() && it->op.dst == target) {
            target = it->op.src;
            moved = true;
        }
    }
    return moved;
}

// Whether the game is a GL one drawn through ANGLE: its projections are GL's.
bool GlClip() {
    static const bool angle = luwes::afme::BehindAngle();
    return angle;
}

// What the tracker found, for the log and the consumers.
struct State {
    Engine::Watch watch;
    Tracker tracker;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    bool engaged = false;
    Miss miss = Miss::kNoFrame;
    Roles previous;  // the frame before, for what changed when the tracker lets go
    // The steps of the present before, for a frame that only shows what those drew.
    std::vector<Step> before;
    // The engaged frame's taps, for Luwes and Kelir at the same present.
    VkSwapchainKHR tapped = VK_NULL_HANDLE;
    Taps taps;
    // Projection P, found by sweeping the game's mappings while engaged.
    ProjectionVote vote;
    // Engine signals, while debug.framefx.signals names the game.
    std::unique_ptr<SignalFeed> signals;
    bool signalsValid = false;
    bool signalsProfileRead = false;
    bool signalsProfileWritten = false;
    std::vector<std::pair<VkDeviceMemory, VkDeviceSize>> sweep;  // mappings left, next offset
    std::vector<ProjectionVote::Hit> projHits;                   // cached winning P locations
    int64_t restUntil = 0;
    int64_t nextFullSweep = 0;
    bool havePlanes = false;
    Projection projection;
    float near = 0;
    float far = 0;
    // Stats over the current period.
    int64_t periodStart = 0;
    uint32_t frames = 0;
    uint32_t engagedFrames = 0;
    uint64_t steps = 0;
    uint64_t scanned = 0;
    std::vector<uint32_t> classifyUs;
    std::vector<uint32_t> signalsUs;
    uint32_t depthFrames = 0;  // D found
    uint32_t keptFrames = 0;   // and kept to present
    std::string dropped;       // how the period's first lost D went
};

std::mutex gStateLock;
State gState;

const char* missName(Miss m) {
    switch (m) {
        case Miss::kNone: return "none";
        case Miss::kNoFrame: return "nothing drawn to the presented image";
        case Miss::kNoScene: return "no scene image";
        case Miss::kGameGenerates: return "the game generates frames itself";
        case Miss::kChanged: return "roles changed";
        case Miss::kResized: return "resized";
        case Miss::kUntoned: return "the scene is only sampled before its tonemap";
    }
    return "?";
}

const char* tierName(Tier t) {
    switch (t) {
        case Tier::kA: return "A";
        case Tier::kB: return "B";
        case Tier::kC: return "C";
        default: return "-";
    }
}

// Callers hold gLock.
std::string imageText(ImageId id) {
    const Image* i = imageOf(id);
    if (i == nullptr) return StringPrintf("#%u", id);
    return StringPrintf("#%u fmt%d %ux%u%s", id, i->format, i->width, i->height,
                        i->swapchain ? " swapchain" : "");
}

// Callers hold gLock.
void logRoles(const Roles& r, const Frame& frame) {
    int pass = -1;
    for (int i = 0; i <= r.ui && i < static_cast<int>(frame.steps.size()); i++) {
        if (frame.steps[i].isPass) pass++;
    }
    std::string line = StringPrintf("lapis %s: engaged, tier %s | D %s %s | S %s | U pass %d > %s",
                                    Engine::get().package().c_str(), tierName(r.tier),
                                    imageText(r.depth).c_str(), r.reversed ? "reversed" : "forward",
                                    imageText(r.scene).c_str(), pass, imageText(r.target).c_str());
    StringAppendF(&line, " | V %s", r.velocity != 0 ? imageText(r.velocity).c_str() : "-");
    if (r.jittered != 0) {
        StringAppendF(&line, " | TAA %s > %s", imageText(r.jittered).c_str(),
                      imageText(r.resolved).c_str());
    }
    ALOGI("%s", line.c_str());
}

// The layout `id` is left in at the end of the frame, from attachments' final layouts and
// barriers. UNDEFINED when the frame never says.
VkImageLayout LastLayout(const Frame& frame, ImageId id) {
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    for (const Step& s : frame.steps) {
        if (s.isPass) {
            for (const Target& t : s.pass.colors) {
                if (t.image == id) layout = t.final;
            }
            if (s.pass.hasDepth && s.pass.depth.image == id) layout = s.pass.depth.final;
        } else if (s.op.kind == OpKind::kLayout && s.op.dst == id) {
            layout = s.op.to;
        }
    }
    return layout;
}

// Callers hold gLock. Whether `id` can be read in place: single-sample, one layer and level, and
// copyable.
bool Tappable(const Frame& frame, ImageId id, Tap* out) {
    const Image* i = imageOf(id);
    const auto handle = gHandles.find(id);
    // Level 0 is all a tap reads and all its barriers name, whatever mips the game keeps below.
    if (i == nullptr || handle == gHandles.end() || i->samples != VK_SAMPLE_COUNT_1_BIT ||
        i->layers != 1 || !(i->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        return false;
    }
    const VkImageLayout layout = LastLayout(frame, id);
    if (layout == VK_IMAGE_LAYOUT_UNDEFINED || layout == VK_IMAGE_LAYOUT_PREINITIALIZED) return false;
    *out = {handle->second, i->format, i->width, i->height, layout, i->usage, id};
    return true;
}

// Callers hold gLock. Whether `id` can be sampled in place: single-sample, one layer and level.
// `held`: an image this frame left alone keeps the layout it was last seen in (a cascade the game
// redraws only every few frames).
bool Sampleable(const Frame& frame, ImageId id, Tap* out, bool held = false) {
    static std::unordered_map<ImageId, VkImageLayout> lastSeen;
    const Image* i = imageOf(id);
    const auto handle = gHandles.find(id);
    if (i == nullptr || handle == gHandles.end() || i->samples != VK_SAMPLE_COUNT_1_BIT ||
        i->layers != 1 || i->mips != 1 || !(i->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        if (held) lastSeen.erase(id);
        return false;
    }
    VkImageLayout layout = LastLayout(frame, id);
    if (held) {
        if (layout != VK_IMAGE_LAYOUT_UNDEFINED) {
            lastSeen[id] = layout;
        } else if (const auto seen = lastSeen.find(id); seen != lastSeen.end()) {
            layout = seen->second;
        }
    }
    if (layout == VK_IMAGE_LAYOUT_UNDEFINED || layout == VK_IMAGE_LAYOUT_PREINITIALIZED) return false;
    *out = {handle->second, i->format, i->width, i->height, layout, i->usage, id};
    return true;
}

// Sweeps the game's mappings for P a slice at a time; once found, cheaply re-probes the cached
// hit slots and only runs full sweeps when hits disappear or on a slower refresh interval.
void scanProjection(State& s, const Roles& roles, int64_t now) {
    if (now < s.restUntil) return;
    float aspect = 0;
    {
        std::lock_guard lock(gLock);
        const Image* d = imageOf(roles.depth);
        if (d == nullptr || d->height == 0) return;
        aspect = static_cast<float>(d->width) / d->height;
    }
    const DepthRange range = RangeOf(roles.reversed, GlClip());
    std::lock_guard lock(gMapLock);
    if (s.sweep.empty() && s.havePlanes && !s.projHits.empty() && now < s.nextFullSweep) {
        const float want = aspect >= 1.0f ? aspect : 1.0f / aspect;
        bool valid = false;
        for (const ProjectionVote::Hit& hit : s.projHits) {
            const auto found = gMappings.find(reinterpret_cast<VkDeviceMemory>(hit.tag));
            if (found == gMappings.end() || found->second.ptr == nullptr ||
                hit.offset + 64 > found->second.size) {
                continue;
            }
            const auto* m = reinterpret_cast<const float*>(found->second.ptr + hit.offset);
            Projection p;
            if (!AsProjection(m, false, &p) && !AsProjection(m, true, &p)) continue;
            const float pa = p.aspect();
            const float got = pa >= 1.0f ? pa : 1.0f / pa;
            if (std::fabs(got / want - 1.0f) > 0.01f) continue;
            float near, far;
            if (!p.planes(range, &near, &far)) continue;
            s.projection = p;
            s.near = near;
            s.far = far;
            valid = true;
        }
        if (valid) {
            s.restUntil = now + kProbeRestNs;
            return;
        }
        s.projHits.clear();
    }
    if (s.sweep.empty()) {
        for (const auto& [memory, m] : gMappings) {
            if (m.ptr != nullptr && m.size >= 64 && m.size <= kMaxScannedMapping) {
                s.sweep.push_back({memory, 0});
            }
        }
        // Popped from the back: scan smaller (more likely UBO) mappings first.
        std::sort(s.sweep.begin(), s.sweep.end(), [](const auto& a, const auto& b) {
            const auto fa = gMappings.find(a.first);
            const auto fb = gMappings.find(b.first);
            const VkDeviceSize sa = fa == gMappings.end() ? 0 : fa->second.size;
            const VkDeviceSize sb = fb == gMappings.end() ? 0 : fb->second.size;
            return sa > sb;
        });
        s.vote.clear();
        if (s.sweep.empty()) {
            s.restUntil = now + kSweepRestNs;
            return;
        }
    }
    size_t budget = kScanBytesPerPresent;
    while (budget > 0 && !s.sweep.empty()) {
        auto& [memory, offset] = s.sweep.back();
        const auto found = gMappings.find(memory);
        if (found == gMappings.end() || offset >= found->second.size) {
            s.sweep.pop_back();
            continue;
        }
        const Mapping& m = found->second;
        const size_t bytes = std::min<size_t>(budget, m.size - offset) & ~size_t{15};
        if (bytes < 64) {
            s.sweep.pop_back();
            continue;
        }
        s.vote.scan(reinterpret_cast<const float*>(m.ptr + offset), bytes / sizeof(float), aspect,
                    range, reinterpret_cast<uintptr_t>(memory), offset);
        // Overlap by a matrix so one straddling the slice edge is still seen.
        offset += bytes > 64 ? bytes - 48 : bytes;
        budget -= std::min(budget, bytes);
        s.scanned += bytes;
    }
    if (s.sweep.empty()) {
        const bool had = s.havePlanes;
        s.havePlanes = s.vote.best(&s.projection, &s.near, &s.far, &s.projHits);
        if (s.havePlanes && !had) {
            ALOGI("lapis %s: P fovY %.1f aspect %.3f near %.3f far %.1f %s jitter (%.5f, %.5f)",
                  Engine::get().package().c_str(), s.projection.fovY(), s.projection.aspect(),
                  s.near, s.far, s.projection.rowMajor ? "row" : "col", s.projection.jx,
                  s.projection.jy);
        }
        if (s.havePlanes) {
            s.restUntil = now + kProbeRestNs;
            s.nextFullSweep = now + kFullSweepRestNs;
        } else {
            s.projHits.clear();
            s.restUntil = now + kSweepRestNs;
        }
    }
}

// One frame into the engine signals; the profile is read once and written once learned.
void updateSignals(State& s, const Frame& frame, const Roles& roles, bool tappable, int64_t now) {
    if (s.signals == nullptr) s.signals = std::make_unique<SignalFeed>();
    const std::string path = luwes::AppCacheDir() + "/framefx-signals.profile";
    if (!s.signalsProfileRead) {
        s.signalsProfileRead = true;
        std::string line;
        if (android::base::ReadFileToString(path, &line)) s.signals->warm(line);
    }
    const int64_t start = nowNs();
    SignalFrame f;
    f.frame = &frame;
    f.roles = roles;
    f.presentNs = now;
    if (roles.geometry >= 0 && static_cast<size_t>(roles.geometry) < frame.steps.size()) {
        f.cameraBlocks = &frame.steps[roles.geometry].pass.constants;
    }
    const int tonemap = s.signals->last().tonemapStep;
    if (tonemap >= 0 && static_cast<size_t>(tonemap) < frame.steps.size() &&
        frame.steps[tonemap].isPass) {
        f.tonemapBlocks = &frame.steps[tonemap].pass.constants;
    }
    {
        std::lock_guard lock(gLock);
        sampleWatched(now);
        f.images = &gImages;
        f.samples = &gSignalSamples;
        const Image* d = imageOf(roles.depth != 0 ? roles.depth : roles.scene);
        if (d != nullptr) {
            f.sceneWidth = d->width;
            f.sceneHeight = d->height;
        }
        s.signals->prepare(f);
        gSignalSamples.clear();
    }
    const EngineSignals& out = s.signals->run();
    s.signalsValid = tappable;
    if (out.tonemapStep >= 0 && static_cast<size_t>(out.tonemapStep) < frame.steps.size() &&
        !frame.steps[out.tonemapStep].pass.dynamic) {
        gWatchedTonemap.store(frame.steps[out.tonemapStep].pass.renderPass, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(gLock);
        gSignalWatch = s.signals->watch();
    }
    s.signalsUs.push_back(static_cast<uint32_t>((nowNs() - start) / 1000));
    if (!s.signalsProfileWritten && out.cameraConfidence >= 0.75f) {
        const std::string line = s.signals->profile();
        if (!line.empty()) s.signalsProfileWritten = android::base::WriteStringToFile(line, path);
    }
}

// Callers hold gLock. While not engaged: what the last frame's presenting pass looked like.
std::string missText(const Frame& frame, const Roles& r) {
    std::string s = StringPrintf("tier %s D %s%s S #%u target %s", tierName(r.tier),
                                 imageText(r.depth).c_str(), r.depthStored ? " stored" : "",
                                 r.scene, imageText(r.target).c_str());
    StringAppendF(&s, " presented %s", imageText(frame.presented).c_str());
    for (int i = static_cast<int>(frame.steps.size()) - 1; i >= 0; i--) {
        const Step& st = frame.steps[i];
        if (!st.isPass) continue;
        bool writes = false;
        for (const Target& t : st.pass.colors) writes = writes || t.image == r.target;
        if (!writes) continue;
        StringAppendF(&s, " | U %ux%u draws %u reads %zu:", st.pass.width, st.pass.height,
                      st.pass.draws, st.pass.reads.size());
        for (size_t k = 0; k < st.pass.reads.size() && k < 8; k++) {
            s += " " + imageText(st.pass.reads[k]);
        }
        break;
    }
    return s;
}

// How D was lost this frame: its last depth-writing pass and every later pass that attaches it.
std::string depthDrop(const Frame& frame, ImageId depth) {
    std::string out;
    int lastWrite = -1;
    for (int i = 0; i < static_cast<int>(frame.steps.size()); i++) {
        const Step& st = frame.steps[i];
        if (st.isPass && st.pass.hasDepth && st.pass.depth.image == depth && st.pass.depthWrites > 0) {
            lastWrite = i;
        }
    }
    for (int i = std::max(lastWrite, 0); i < static_cast<int>(frame.steps.size()); i++) {
        const Step& st = frame.steps[i];
        if (st.isPass && st.pass.hasDepth && st.pass.depth.image == depth) {
            out += android::base::StringPrintf(
                    " [%d rp 0x%" PRIx64 " sub %u load %d store %d stencil %d/%d writes %u%s]", i,
                    st.pass.renderPass, st.pass.subpasses, st.pass.depth.load, st.pass.depth.store,
                    st.pass.stencilLoad, st.pass.stencilStore, st.pass.depthWrites,
                    i == lastWrite ? " last write" : "");
        } else if (!st.isPass && st.op.dst == depth) {
            out += android::base::StringPrintf(" [%d op %d]", i, static_cast<int>(st.op.kind));
        }
    }
    return out.empty() ? " never written" : out;
}

void logStats(State& s, int64_t now, const Frame& frame, const Roles& roles) {
    if (s.periodStart == 0) s.periodStart = now;
    if (now - s.periodStart < kStatsPeriodNs || s.frames == 0) return;
    if (!s.engaged) {
        std::lock_guard lock(gLock);
        ALOGI("lapis %s: last frame %s", Engine::get().package().c_str(),
              missText(frame, roles).c_str());
    }
    std::vector<uint32_t>& us = s.classifyUs;
    std::sort(us.begin(), us.end());
    const auto at = [&](double q) { return us.empty() ? 0u : us[static_cast<size_t>(q * (us.size() - 1))]; };
    const Calls c = calls();
    ALOGI("lapis %s: %u frames, tapped %.0f%%, %s, %.0f steps per frame, classify p50 %u us p95 "
          "%u us, scanned %.1f MB, draws so far %u",
          Engine::get().package().c_str(), s.frames, 100.0 * s.engagedFrames / s.frames,
          s.engaged ? "engaged" : missName(s.miss), static_cast<double>(s.steps) / s.frames,
          at(0.5), at(0.95), s.scanned / 1048576.0, c.draw);
    if (s.depthFrames > 0) {
        ALOGI("lapis %s: D kept %u of %u frames%s%s", Engine::get().package().c_str(),
              s.keptFrames, s.depthFrames, s.dropped.empty() ? "" : "; lost:", s.dropped.c_str());
    }
    if (s.signals != nullptr) {
        std::vector<uint32_t>& su = s.signalsUs;
        std::sort(su.begin(), su.end());
        const auto sat = [&](double q) { return su.empty() ? 0u : su[static_cast<size_t>(q * (su.size() - 1))]; };
        const EngineSignals& g = s.signals->last();
        ALOGI("lapis %s: signals p50 %u us p95 %u us; camera %.2f flags %#x turn %.2f; next %.2f "
              "hit %.2f early %.2f lead %.2f ms own %.2f ms; jitter %.2f pattern %u period %u "
              "(%.3f, %.3f) px; cut %u state %u since %u; post %#x pre-post #%u step %d flags %#x "
              "tonemap %d; transparent %u of %u at step %d draw %u; scale %.3fx%.3f drawn %.3fx%.3f; "
              "stencil ids %u (%u); clock %.2f flags %#x step %.2f ms interval %.2f ms; exposure "
              "%.2f gain %.4f",
              Engine::get().package().c_str(), sat(0.5), sat(0.95), g.cameraConfidence,
              g.cameraFlags, g.turnDegrees, g.nextConfidence, g.nextHitRate, g.nextEarlyRate,
              g.nextLeadMedianNs / 1e6, g.nextOwnLeadMedianNs / 1e6, g.jitterConfidence,
              g.jitterPattern, g.jitterPeriod, g.jitterPixels[0], g.jitterPixels[1], g.cut,
              g.sceneState, g.framesSinceCut, g.postEffects, g.prePostImage, g.prePostStep,
              g.prePostFlags, g.tonemapStep, g.transparentDraws, g.sceneDraws, g.transparentStep,
              g.transparentFrom, g.renderScale[0], g.renderScale[1], g.drawn[0], g.drawn[1],
              g.stencilIds, g.stencilWrites, g.clockConfidence, g.clockFlags, g.simStep * 1e3,
              g.presentInterval * 1e3, g.exposureConfidence, g.exposureGain);
        su.clear();
    }
    s.depthFrames = 0;
    s.keptFrames = 0;
    s.dropped.clear();
    s.periodStart = now;
    s.frames = 0;
    s.engagedFrames = 0;
    s.steps = 0;
    s.scanned = 0;
    us.clear();
}

// Callers hold gLock. What Pentas's status line shows.
LapisStatus statusOf(const State& s, const Roles& roles, bool on) {
    LapisStatus st;
    if (!on) return st;
    st.state = s.engaged ? LapisState::kEngaged : LapisState::kLearning;
    st.tier = static_cast<uint8_t>(roles.tier);
    st.miss = static_cast<uint8_t>(s.tracker.lastMiss());
    if (roles.depth != 0) st.roles |= kLapisDepth;
    if (roles.depthStored) st.roles |= kLapisKept;
    if (roles.velocity != 0) st.roles |= kLapisMotion;
    if (s.havePlanes) st.roles |= kLapisCamera;
    if (const Image* scene = imageOf(roles.scene)) {
        st.roles |= kLapisScene;
        st.sceneWidth = static_cast<uint16_t>(scene->width);
        st.sceneHeight = static_cast<uint16_t>(scene->height);
    }
    return st;
}

// One slot per call for every device: a slot already holding another function (a second device,
// or one asking by the KHR alias, whose libvulkan entry calls a driver function the first device
// never enabled) keeps it, and that query goes unwrapped.
#define WRAP(fn, slot)                                                                   \
    if (strcmp(name, "vk" #fn) == 0) {                                                   \
        const auto want = reinterpret_cast<decltype(gNext.slot)>(next);                  \
        auto* const held = &gNext.slot;                                                  \
        decltype(gNext.slot) none = nullptr;                                             \
        if (!__atomic_compare_exchange_n(held, &none, want, false, __ATOMIC_ACQ_REL,     \
                                         __ATOMIC_ACQUIRE) &&                            \
            none != want) {                                                              \
            return nullptr;                                                              \
        }                                                                                \
        return reinterpret_cast<PFN_vkVoidFunction>(slot);                               \
    }

// Callers hold gLock. A pass that writes a square depth image alone, other than the scene's: a
// light's shadow map. Deferred renderers draw them after the scene's geometry, forward ones
// before.
bool IsShadowPass(const Frame& frame, int k, const Roles& roles) {
    if (k == roles.geometry) return false;
    const Step& s = frame.steps[k];
    if (!s.isPass || s.pass.dynamic || s.pass.renderPass == 0 || !s.pass.hasDepth ||
        !s.pass.colors.empty() || s.pass.depthWrites == 0 || s.pass.depth.image == roles.depth) {
        return false;
    }
    const Image* image = imageOf(s.pass.depth.image);
    return image != nullptr && image->width == image->height;
}

// Callers hold gLock. The busiest shadow pass, whose constants the next frames keep, or -1.
int ShadowPass(const Frame& frame, const Roles& roles) {
    int best = -1;
    for (int k = 0; k < static_cast<int>(frame.steps.size()); k++) {
        if (!IsShadowPass(frame, k, roles)) continue;
        if (best < 0 || frame.steps[k].pass.draws > frame.steps[best].pass.draws) best = k;
    }
    return best;
}

LightVote gLightVote;  // under gLock

// Where a shadow atlas's draws read their cascade's matrix: a uniform buffer and float offset.
struct CascadeSlot {
    uint32_t set = 0;
    uint32_t binding = 0;
    size_t offset = 0;
    bool operator<(const CascadeSlot& o) const {
        return std::tie(set, binding, offset) < std::tie(o.set, o.binding, o.offset);
    }
};
// By shadow image: the slots seen holding a different light matrix in two tiles' draws, and
// how many frames each held one in the pass's first draw since.
std::unordered_map<ImageId, std::map<CascadeSlot, int>> gCascadeSlots;  // under gLock

// The slots holding a light matrix in both draws' blocks, a different one in each.
std::vector<CascadeSlot> CascadeSlotsOf(const std::vector<Constants>& first,
                                        const std::vector<Constants>& tile) {
    std::vector<CascadeSlot> out;
    const std::vector<LightMatrix> a = LightMatrices(first);
    const std::vector<LightMatrix> b = LightMatrices(tile);
    for (const LightMatrix& x : a) {
        const Constants& bx = first[x.block];
        for (const LightMatrix& y : b) {
            const Constants& by = tile[y.block];
            if (bx.set != by.set || bx.binding != by.binding || x.offset != y.offset) continue;
            if (std::fabs(x.scale - y.scale) <= 1e-3f * std::max(x.scale, y.scale)) continue;
            out.push_back({bx.set, bx.binding, x.offset});
        }
    }
    return out;
}

// The 16 floats at `slot` of the first block holding them.
std::vector<Constants> AtSlot(const std::vector<Constants>& blocks, const CascadeSlot& slot) {
    for (const Constants& c : blocks) {
        if (c.set != slot.set || c.binding != slot.binding || slot.offset + 16 > c.values.size()) {
            continue;
        }
        Constants only = c;
        only.values.assign(c.values.begin() + slot.offset, c.values.begin() + slot.offset + 16);
        return {only};
    }
    return {};
}

}  // namespace

int sceneState(VkCommandBuffer cb) {
    const ImageId depth = gSceneDepth.load(std::memory_order_relaxed);
    if (depth == 0 || !gOn.load(std::memory_order_relaxed)) return -1;
    const Record& r = recordOf(cb);
    if (r.ignored || r.open < 0 || r.pastScene) return 0;
    const Step& e = r.events[r.open];
    return e.isPass && e.pass.hasDepth && e.pass.depth.image == depth ? 1 : 0;
}

VkImageView sampledView(VkCommandBuffer cb, uint32_t set, uint32_t binding) {
    if (!gOn.load(std::memory_order_relaxed) || set >= kMaxSets) return VK_NULL_HANDLE;
    const VkDescriptorSet bound = recordOf(cb).bound[0][set];
    if (bound == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    std::lock_guard lock(gLock);
    const auto views = gSetViews.find(bound);
    if (views == gSetViews.end()) return VK_NULL_HANDLE;
    const auto view = views->second.find(static_cast<uint64_t>(binding) << 32);
    // A view since destroyed is no longer known.
    if (view == views->second.end() || !gViews.contains(view->second)) return VK_NULL_HANDLE;
    return view->second;
}

size_t uniformBytes(VkCommandBuffer cb, uint32_t set, uint32_t binding, size_t size, uint8_t* out) {
    if (!gOn.load(std::memory_order_relaxed) || set >= kMaxSets || size == 0) return 0;
    const Record& r = recordOf(cb);
    const VkDescriptorSet bound = r.bound[0][set];
    if (bound == VK_NULL_HANDLE) return 0;
    std::lock_guard lock(gLock);
    const auto bindings = gSetBuffers.find(bound);
    if (bindings == gSetBuffers.end()) return 0;
    const auto found = bindings->second.find(static_cast<uint64_t>(binding) << 32);
    if (found == bindings->second.end()) return 0;
    const BufferBinding& b = found->second;
    const auto buffer = gBuffers.find(b.buffer);
    if (buffer == gBuffers.end() || buffer->second.memory == VK_NULL_HANDLE) return 0;
    VkDeviceSize offset = b.offset;
    if (b.dynamic) {
        // The set's dynamic offsets go by its layout's dynamic bindings, in order.
        const auto layout = gSetLayoutOf.find(bound);
        const auto desc = layout == gSetLayoutOf.end() ? gSetLayouts.end()
                                                       : gSetLayouts.find(layout->second);
        if (desc == gSetLayouts.end()) return 0;
        const auto& order = desc->second.dynamicBindings;
        const auto at = std::find(order.begin(), order.end(), binding << 16);
        const size_t k = at - order.begin();
        if (at == order.end() || k >= r.dynamicOffsets[set].size()) return 0;
        offset += r.dynamicOffsets[set][k];
    }
    const VkDeviceSize range = b.range == VK_WHOLE_SIZE
            ? (buffer->second.size > offset ? buffer->second.size - offset : 0)
            : b.range;
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(range, size));
    std::string bytes;
    if (n == 0 || !readBuffer(b.buffer, offset, n, &bytes)) return 0;
    memcpy(out, bytes.data(), n);
    return n;
}

bool viewAlive(VkImageView view) {
    std::lock_guard lock(gLock);
    return gViews.contains(view);
}

bool recording() {
    static const bool on = [] {
        Engine& engine = Engine::get();
        engine.attach();
        return engine.config().lapis.on() || survey::enabled() || alihOn();
    }();
    return on;
}

void attach(const framefx_vk_host* host) {
    gHost = *host;
}

void onDevice(const framefx_vk_instance* instance, VkPhysicalDevice physical, VkDevice device) {
    if (!recording()) return;
    VkPhysicalDeviceMemoryProperties2 memory = {};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    if (instance->get_memory_properties2 != nullptr) {
        instance->get_memory_properties2(physical, &memory);
    }
    DeviceInfo info;
    info.memory = memory.memoryProperties;
    info.owned = instance->get_device_proc_addr != nullptr &&
            info.fn.load(device, instance->get_device_proc_addr);
    info.alihLoaded = alihOn() && instance->get_device_proc_addr != nullptr &&
            info.alih.load(device, instance->get_device_proc_addr);
    hud_probe::OnDevice(device, instance->get_device_proc_addr, info.memory);
    std::lock_guard lock(gLock);
    gDevices[device] = info;
    gOn.store(true);
    ALOGI("lapis %s: recording device %p (%zu)", Engine::get().package().c_str(), device,
          gDevices.size());
}

void onDestroyDevice(VkDevice device) {
    hud_probe::OnDestroyDevice(device);
    std::lock_guard lock(gLock);
    const auto found = gDevices.find(device);
    if (found == gDevices.end()) return;
    // The game waits for its device to idle before destroying it.
    if (gSplit.device == device) retireSplit();
    if (gGlass.opaque.device == device) retireGlass();
    if (gKeep.device == device) retireKeep();
    for (auto i = gResumes.begin(); i != gResumes.end();) {
        if (i->second.first != device) {
            ++i;
            continue;
        }
        gSpent.push_back({device, i->second.second, VK_NULL_HANDLE, VK_NULL_HANDLE});
        i = gResumes.erase(i);
    }
    const Own& f = found->second.fn;
    for (auto i = gSpent.begin(); i != gSpent.end();) {
        if (i->device != device) {
            ++i;
            continue;
        }
        if (i->pass != VK_NULL_HANDLE) f.DestroyRenderPass(device, i->pass, nullptr);
        if (i->image != VK_NULL_HANDLE) f.DestroyImage(device, i->image, nullptr);
        if (i->memory != VK_NULL_HANDLE) f.FreeMemory(device, i->memory, nullptr);
        i = gSpent.erase(i);
    }
    gDevices.erase(found);
    if (gDevices.empty()) gOn.store(false);
}

void onSwapchain(VkDevice device, VkSwapchainKHR swapchain, const VkSwapchainCreateInfoKHR* info,
                 const std::vector<VkImage>* own) {
    if (!gOn.load(std::memory_order_relaxed) || gHost.get_swapchain_images == nullptr) return;
    std::vector<VkImage> images;
    if (own != nullptr) {
        images = *own;
    } else {
        uint32_t count = 0;
        if (gHost.get_swapchain_images(device, swapchain, &count, nullptr) != VK_SUCCESS) return;
        images.resize(count);
        if (gHost.get_swapchain_images(device, swapchain, &count, images.data()) != VK_SUCCESS) {
            return;
        }
    }
    Swapchain sc;
    sc.device = device;
    sc.width = info->imageExtent.width;
    sc.height = info->imageExtent.height;
    std::lock_guard lock(gLock);
    for (VkImage handle : images) {
        Image i;
        i.format = info->imageFormat;
        i.width = sc.width;
        i.height = sc.height;
        i.layers = info->imageArrayLayers;
        i.usage = info->imageUsage;
        i.swapchain = true;
        sc.images.push_back(newImage(handle, i));
    }
    gSwapchains[swapchain] = std::move(sc);
}

void onDestroySwapchain(VkSwapchainKHR swapchain) {
    std::lock_guard lock(gLock);
    const auto found = gSwapchains.find(swapchain);
    if (found == gSwapchains.end()) return;
    for (ImageId id : found->second.images) {
        gImages.erase(id);
        gHandles.erase(id);
    }
    for (auto i = gImageIds.begin(); i != gImageIds.end();) {
        const bool gone = std::find(found->second.images.begin(), found->second.images.end(),
                                    i->second) != found->second.images.end();
        i = gone ? gImageIds.erase(i) : std::next(i);
    }
    gSwapchains.erase(found);
}

PFN_vkVoidFunction intercept(VkDevice, const char* name, PFN_vkVoidFunction next) {
    if (next == nullptr || !recording()) return nullptr;
    WRAP(CreateImage, CreateImage)
    WRAP(DestroyImage, DestroyImage)
    WRAP(AllocateMemory, AllocateMemory)
    WRAP(BindImageMemory, BindImageMemory)
    WRAP(BindImageMemory2, BindImageMemory2)
    WRAP(BindImageMemory2KHR, BindImageMemory2)
    WRAP(CreateImageView, CreateImageView)
    WRAP(DestroyImageView, DestroyImageView)
    WRAP(CreateRenderPass, CreateRenderPass)
    WRAP(CreateRenderPass2, CreateRenderPass2)
    WRAP(CreateRenderPass2KHR, CreateRenderPass2)
    WRAP(CreateFramebuffer, CreateFramebuffer)
    WRAP(CreateGraphicsPipelines, CreateGraphicsPipelines)
    WRAP(UpdateDescriptorSets, UpdateDescriptorSets)
    WRAP(CreateDescriptorUpdateTemplate, CreateDescriptorUpdateTemplate)
    WRAP(CreateDescriptorUpdateTemplateKHR, CreateDescriptorUpdateTemplate)
    WRAP(UpdateDescriptorSetWithTemplate, UpdateDescriptorSetWithTemplate)
    WRAP(UpdateDescriptorSetWithTemplateKHR, UpdateDescriptorSetWithTemplate)
    WRAP(BeginCommandBuffer, BeginCommandBuffer)
    WRAP(CmdExecuteCommands, CmdExecuteCommands)
    WRAP(CmdBeginRenderPass, CmdBeginRenderPass)
    WRAP(CmdBeginRenderPass2, CmdBeginRenderPass2)
    WRAP(CmdBeginRenderPass2KHR, CmdBeginRenderPass2)
    WRAP(CmdEndRenderPass, CmdEndRenderPass)
    WRAP(CmdEndRenderPass2, CmdEndRenderPass2)
    WRAP(CmdEndRenderPass2KHR, CmdEndRenderPass2)
    WRAP(CmdBeginRendering, CmdBeginRendering)
    WRAP(CmdBeginRenderingKHR, CmdBeginRendering)
    WRAP(CmdEndRendering, CmdEndRendering)
    WRAP(CmdEndRenderingKHR, CmdEndRendering)
    WRAP(CmdBindPipeline, CmdBindPipeline)
    WRAP(CmdSetDepthWriteEnable, CmdSetDepthWriteEnable)
    WRAP(CmdSetDepthWriteEnableEXT, CmdSetDepthWriteEnable)
    WRAP(CmdSetDepthTestEnable, CmdSetDepthTestEnable)
    WRAP(CmdSetDepthCompareOp, CmdSetDepthCompareOp)
    WRAP(CmdSetDepthCompareOpEXT, CmdSetDepthCompareOp)
    WRAP(CmdClearAttachments, CmdClearAttachments)
    WRAP(CmdSetScissor, CmdSetScissor)
    WRAP(CmdSetDepthTestEnableEXT, CmdSetDepthTestEnable)
    WRAP(CmdDraw, CmdDraw)
    WRAP(CmdDrawIndexed, CmdDrawIndexed)
    WRAP(CmdDrawIndirect, CmdDrawIndirect)
    WRAP(CmdDrawIndexedIndirect, CmdDrawIndexedIndirect)
    WRAP(CmdDrawIndirectCount, CmdDrawIndirectCount)
    WRAP(CmdDrawIndirectCountKHR, CmdDrawIndirectCount)
    WRAP(CmdDrawIndexedIndirectCount, CmdDrawIndexedIndirectCount)
    WRAP(CmdDrawIndexedIndirectCountKHR, CmdDrawIndexedIndirectCount)
    WRAP(CmdCopyImage, CmdCopyImage)
    WRAP(CmdCopyImage2, CmdCopyImage2)
    WRAP(CmdCopyImage2KHR, CmdCopyImage2)
    WRAP(CmdBlitImage, CmdBlitImage)
    WRAP(CmdBlitImage2, CmdBlitImage2)
    WRAP(CmdBlitImage2KHR, CmdBlitImage2)
    WRAP(CmdResolveImage, CmdResolveImage)
    WRAP(CmdResolveImage2, CmdResolveImage2)
    WRAP(CmdResolveImage2KHR, CmdResolveImage2)
    WRAP(CmdCopyImageToBuffer, CmdCopyImageToBuffer)
    WRAP(CmdClearDepthStencilImage, CmdClearDepthStencilImage)
    WRAP(CmdPipelineBarrier, CmdPipelineBarrier)
    WRAP(CmdPipelineBarrier2, CmdPipelineBarrier2)
    WRAP(CmdPipelineBarrier2KHR, CmdPipelineBarrier2)
    WRAP(QueueSubmit, QueueSubmit)
    WRAP(QueueSubmit2, QueueSubmit2)
    WRAP(QueueSubmit2KHR, QueueSubmit2)
    WRAP(CmdBindDescriptorSets, CmdBindDescriptorSets)
    WRAP(CmdDispatch, CmdDispatch)
    WRAP(CmdDispatchIndirect, CmdDispatchIndirect)
    WRAP(MapMemory, MapMemory)
    WRAP(MapMemory2, MapMemory2)
    WRAP(MapMemory2KHR, MapMemory2)
    WRAP(UnmapMemory, UnmapMemory)
    WRAP(UnmapMemory2, UnmapMemory2)
    WRAP(UnmapMemory2KHR, UnmapMemory2)
    WRAP(FreeMemory, FreeMemory)
    WRAP(CreateBuffer, CreateBuffer)
    WRAP(DestroyBuffer, DestroyBuffer)
    WRAP(BindBufferMemory, BindBufferMemory)
    WRAP(BindBufferMemory2, BindBufferMemory2)
    WRAP(BindBufferMemory2KHR, BindBufferMemory2)
    WRAP(CreateDescriptorSetLayout, CreateDescriptorSetLayout)
    WRAP(AllocateDescriptorSets, AllocateDescriptorSets)
    WRAP(CreateShaderModule, CreateShaderModule)
    WRAP(CmdSetViewport, CmdSetViewport)
    WRAP(CmdBindVertexBuffers, CmdBindVertexBuffers)
    WRAP(CmdBindVertexBuffers2, CmdBindVertexBuffers2)
    WRAP(CmdBindVertexBuffers2EXT, CmdBindVertexBuffers2)
    WRAP(CmdBindIndexBuffer, CmdBindIndexBuffer)
    WRAP(CmdSetVertexInputEXT, CmdSetVertexInputEXT)
    WRAP(CmdCopyBuffer, CmdCopyBuffer)
    WRAP(CmdCopyBuffer2, CmdCopyBuffer2)
    WRAP(CmdCopyBuffer2KHR, CmdCopyBuffer2)
    WRAP(CmdUpdateBuffer, CmdUpdateBuffer)
    WRAP(CmdBindIndexBuffer2KHR, CmdBindIndexBuffer2)
    WRAP(CmdBindIndexBuffer2, CmdBindIndexBuffer2)
    WRAP(CmdCopyBufferToImage, CmdCopyBufferToImage)
    WRAP(CmdCopyBufferToImage2, CmdCopyBufferToImage2)
    WRAP(CmdCopyBufferToImage2KHR, CmdCopyBufferToImage2)
    WRAP(BindOpticalFlowSessionImageNV, BindOpticalFlowSessionImageNV)
    WRAP(CmdOpticalFlowExecuteNV, CmdOpticalFlowExecuteNV)
    return nullptr;
}

// ---- Hide HUD: labels, the skip table and the cross-check, once per present ------------------

// Under gStateLock.
struct HudPresent {
    std::unique_ptr<HudLabeler> labeler;
    const Frame* labelled = nullptr;  // the frame the labeler last ran on, for the survey
    size_t labelledSteps = 0;
    bool profileRead = false;
    bool skipping = false;
    HudCross cross = HudCross::kNone;
};
HudPresent gHud;

// Callers hold gLock. The cross-check copies what the pass that draws the presented image leaves.
void hudArmProbe(const Frame& frame, VkSwapchainKHR swapchain) {
    const auto sc = gSwapchains.find(swapchain);
    for (auto it = frame.steps.rbegin(); sc != gSwapchains.end() && it != frame.steps.rend(); ++it) {
        if (!it->isPass || !it->pass.writes(frame.presented)) continue;
        const Pass& p = it->pass;
        for (const Target& t : p.colors) {
            if (t.image != frame.presented) continue;
            hud_probe::Arm(sc->second.device, p.dynamic ? 0 : p.renderPass, t.format, p.width,
                           p.height, t.final);
            return;
        }
    }
    hud_probe::Disarm();
}

void hudPresent(State& s, const Frame& frame, const Roles& roles, VkSwapchainKHR swapchain) {
    const std::string& package = Engine::get().package();
    const bool hide = s.watch.config().lapis.hideHud;
    const uint32_t skipped = HudFrameDone();
    gHud.labelled = nullptr;
    if (!hide && !survey::hudDraws()) {
        if (HudCapturing()) {
            SetHudCapturing(false);
            hud_probe::Disarm();
            if (gHud.labeler != nullptr) gHud.labeler->reset();
            if (gHud.skipping) ALOGI("lapis %s: HUD draws no longer skipped", package.c_str());
            gHud.skipping = false;
        }
        return;
    }
    SetHudCapturing(true);
    if (gHud.labeler == nullptr) gHud.labeler = std::make_unique<HudLabeler>(&HudSkipTable());
    HudLabeler& labeler = *gHud.labeler;
    const std::string profile = luwes::AppCacheDir() + "/framefx-hud.profile";
    if (!gHud.profileRead) {
        gHud.profileRead = true;
        std::string line;
        if (android::base::ReadFileToString(profile, &line)) labeler.warm(line);
        if (android::base::ReadFileToString(luwes::AppCacheDir() + "/framefx-hud.weights", &line)) {
            ALOGI("lapis %s: HUD weights from the game's cache %s", package.c_str(),
                  labeler.setWeights(line) ? "taken" : "refused: not this build's shape");
        }
    }
    labeler.keepFeatures(survey::hudDraws());
    const bool camera = s.tapped == swapchain && s.taps.camera.matrices;
    hud_probe::Presented(camera, camera ? s.taps.camera.vp : nullptr);
    HudGrid grid;
    const bool haveGrid = hud_probe::Take(&grid);
    {
        std::lock_guard lock(gLock);
        hudArmProbe(frame, swapchain);
        HudFrameIn in;
        in.frame = &frame;
        in.images = &gImages;
        in.roles = roles;
        in.hideHud = hide;
        in.skipped = skipped;
        in.grid = haveGrid ? &grid : nullptr;
        labeler.prepare(in);
    }
    const HudOutput& out = labeler.run();
    gHud.labelled = &frame;
    gHud.labelledSteps = frame.steps.size();
    if (s.tapped == swapchain) s.taps.hudSkipped = out.skipping && out.applied;
    if (out.skipping != gHud.skipping) {
        ALOGI("lapis %s: %s; HUD draws covered %.2f, label confidence %.2f, %u table entries",
              package.c_str(),
              out.skipping ? "hiding the HUD by skipping its draws" : "HUD draws no longer skipped",
              out.coverage, out.confidence, out.entries);
        gHud.skipping = out.skipping;
    }
    if (out.cross != gHud.cross) {
        static const char* kCross[] = {"no verdict", "agrees with the labels",
                                       "still shows HUD while skipping",
                                       "shows no HUD the labels find"};
        ALOGI("lapis %s: the screen %s (HUD %.3f of it, %.3f left while skipping)",
              package.c_str(), kCross[static_cast<int>(out.cross) & 3], out.hudShare,
              out.residue);
        gHud.cross = out.cross;
    }
    if (out.dirty) {
        const std::string line = labeler.profile();
        if (!line.empty() && android::base::WriteStringToFile(line + "\n", profile)) {
            labeler.profileWritten();
        }
    }
}

// ------------------------------------------------------------------------------------------------

void onPresent(VkQueue, const VkPresentInfoKHR* info) {
    if (survey::atlas()) flushTextDraws();
    if (alihOn()) alihPresent();
    if (survey::textCost()) textCostPresent();
    if (survey::fontScan() && survey::shooting()) maybeScanFonts();
    if (!gOn.load(std::memory_order_relaxed) || info == nullptr) return;
    std::lock_guard stateLock(gStateLock);
    State& s = gState;
    const int64_t start = nowNs();
    Frame frame;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    Roles roles;
    {
        std::lock_guard lock(gLock);
        frame.steps.swap(gCollected);
        for (uint32_t i = 0; i < info->swapchainCount && frame.presented == 0; i++) {
            const auto found = gSwapchains.find(info->pSwapchains[i]);
            if (found == gSwapchains.end()) continue;
            const uint32_t index = info->pImageIndices[i];
            if (index >= found->second.images.size()) continue;
            frame.presented = found->second.images[index];
            frame.width = found->second.width;
            frame.height = found->second.height;
            swapchain = info->pSwapchains[i];
        }
    }
    // A frame is what the presented image came from, not what was submitted since the last
    // present: a game may draw two presents' worth at once and only copy the second out later.
    const bool late = !s.before.empty() && ShowsAFrameBefore(frame);
    if (late) frame.steps.insert(frame.steps.begin(), s.before.begin(), s.before.end());
    s.watch.refresh(Engine::get());
    const bool on = s.watch.config().lapis.on();
    const bool surveyed = survey::wantsFrame();
    if (!on) {
        // Pentas turned Lapis off while the game runs: nothing tapped, nothing learned.
        if (s.engaged) ALOGI("lapis %s: off", Engine::get().package().c_str());
        s.tracker.reset();
        s.engaged = false;
        s.tapped = VK_NULL_HANDLE;
        s.taps = Taps();
        Engine::get().reportLapis(LapisStatus());
        if (!surveyed) return;
    }
    int shadow = -1;
    {
        std::lock_guard lock(gLock);
        roles = Classify(frame, gImages);
        if (on && roles.geometry >= 0) shadow = ShadowPass(frame, roles);
        gSceneDepth.store(on && roles.geometry >= 0 ? roles.depth : 0, std::memory_order_relaxed);
        if (on) armSplit(frame, roles, swapchain);
        if (on) armKeep(frame, roles, swapchain);
        if (on) armGlass(frame, roles, swapchain);
    }
    // The next frames keep the constants of the pass that draws the scene: its camera.
    if (on && roles.geometry >= 0 && !frame.steps[roles.geometry].pass.dynamic) {
        gWatchedPass.store(frame.steps[roles.geometry].pass.renderPass, std::memory_order_relaxed);
    }
    // And of the sun's shadow map: the busiest depth-only pass before the scene into a square.
    if (shadow >= 0) {
        gWatchedShadow.store(frame.steps[shadow].pass.renderPass, std::memory_order_relaxed);
    }
    if (!on) {
        Images images;
        {
            std::lock_guard lock(gLock);
            images = gImages;
        }
        survey::onFrame(frame, images, roles);
        return;
    }
    if (swapchain != s.swapchain) {
        s.tracker.reset();
        s.swapchain = swapchain;
        s.before.clear();
        s.sweep.clear();
        s.projHits.clear();
        s.nextFullSweep = 0;
        s.havePlanes = false;
    }
    const bool tappable = s.tracker.update(roles, frame.width, frame.height);
    const bool engaged = s.tracker.engaged();
    s.tapped = VK_NULL_HANDLE;
    if (tappable) {
        std::lock_guard lock(gLock);
        Taps taps;
        taps.reversed = roles.reversed;
        taps.gameGenerates = roles.gameGenerates;
        if (s.havePlanes) {
            const Projection& p = s.projection;
            const bool gl = RangeOf(roles.reversed, GlClip()) == DepthRange::kForwardGl;
            taps.camera = {true, gl, p.w < 0 ? -p.a : p.a, p.b, p.sx, p.sy, p.jx, p.jy};
        }
        // This frame's view-projection, from the constants of the pass that draws the scene.
        const Image* d = imageOf(roles.depth);
        ViewProjection vp;
        if (roles.geometry >= 0 && d != nullptr && d->height != 0 &&
            FindViewProjection(frame.steps[roles.geometry].pass.constants,
                               static_cast<float>(d->width) / d->height, &vp)) {
            taps.camera.matrices = true;
            std::copy(std::begin(vp.m), std::end(vp.m), taps.camera.vp);
            taps.camera.ndcJitterX = vp.jx;
            taps.camera.ndcJitterY = vp.jy;
            taps.camera.flipsY = frame.steps[roles.geometry].pass.viewportHeight < 0;
            taps.camera.world = vp.world;
        }
        // debug.framefx.constants_dump: the scene pass's and shadow passes' uniform blocks, 8
        // frames 30 apart, into the game's cache (constants-<n>.txt), to find a camera's position.
        static int constantsDumps = -1;
        static uint32_t constantsPoll = 0;
        if (constantsPoll++ % 30 == 0) {
            if (constantsDumps < 0 &&
                android::base::GetBoolProperty("debug.framefx.constants_dump", false)) {
                constantsDumps = 0;
            }
            if (constantsDumps >= 0 && constantsDumps < 8 && roles.geometry >= 0) {
                std::string text = StringPrintf("vp world %d\n", vp.world ? 1 : 0);
                const auto dump = [&](const char* what, int k) {
                    for (const Constants& c : frame.steps[k].pass.constants) {
                        text += StringPrintf("%s %d set %u binding %u offset %llu floats %zu\n", what,
                                             k, c.set, c.binding,
                                             static_cast<unsigned long long>(c.offset),
                                             c.values.size());
                        for (size_t i = 0; i + 4 <= c.values.size(); i += 4) {
                            text += StringPrintf("  %zu: %.6g %.6g %.6g %.6g\n", i, c.values[i],
                                                 c.values[i + 1], c.values[i + 2], c.values[i + 3]);
                        }
                    }
                };
                dump("scene", roles.geometry);
                for (int k = 0; k < static_cast<int>(frame.steps.size()); k++) {
                    if (IsShadowPass(frame, k, roles)) dump("shadow", k);
                }
                const std::string package =
                        Engine::get().package().substr(0, Engine::get().package().find(':'));
                android::base::WriteStringToFile(
                        text, StringPrintf("/data/user/%u/%s/cache/constants-%d.txt",
                                           getuid() / 100000, package.c_str(), constantsDumps));
                constantsDumps++;
            }
        }
        // debug.framefx.shadow_log: 40 frames of the shadow passes and the maps the rays get.
        static int shadowLog = 0;
        static bool shadowLogged = false;
        static uint32_t shadowPoll = 0;
        if (shadowLog == 0 && shadowPoll++ % 60 == 0) {
            const bool asked = android::base::GetBoolProperty("debug.framefx.shadow_log", false);
            if (asked && !shadowLogged) shadowLog = 40;
            shadowLogged = asked;
        }
        std::string logged;
        std::string slotLog;
        // Toward the sun: every shadow map this frame drew, voted on (LightVote).
        if (shadow >= 0) {
            float eye[3];
            const bool placed = taps.camera.matrices && EyeOf(taps.camera.vp, eye);
            for (int k = 0; k < static_cast<int>(frame.steps.size()); k++) {
                if (!IsShadowPass(frame, k, roles)) continue;
                const Pass& sp = frame.steps[k].pass;
                if (sp.constants.empty()) continue;
                // A cascade drawn onto another's clear follows the scene's depth convention.
                const bool reversed =
                        std::isnan(sp.clearDepth) ? roles.reversed : sp.clearDepth == 0.0f;
                const Image* map = imageOf(sp.depth.image);
                // Where in the map a viewport drew, in texture coordinates.
                const auto regionOf = [&](float x, float y, float w, float h, float out[4]) {
                    out[0] = out[1] = 0;
                    out[2] = out[3] = 1;
                    if (map == nullptr) return;
                    const VkRect2D r = DrawnRegion(x, y, w, h, map->width, map->height);
                    if (r.extent.width == 0) return;
                    out[0] = static_cast<float>(r.offset.x) / map->width;
                    out[1] = static_cast<float>(r.offset.y) / map->height;
                    out[2] = static_cast<float>(r.extent.width) / map->width;
                    out[3] = static_cast<float>(r.extent.height) / map->height;
                };
                // An atlas's cascade matrix is where a tile's first draw reads another light
                // matrix than the pass's first: there only, the other light-like matrices being
                // the objects'. Kept for the frames that redraw no other tile.
                // Object blocks hold light-like matrices too, some frames: the slot that
                // holds one most steadily is the cascade's.
                if (gCascadeSlots.size() > 64) gCascadeSlots.clear();  // images long gone
                std::map<CascadeSlot, int>& slots = gCascadeSlots[sp.depth.image];
                for (const Pass::Tile& t : sp.tiles) {
                    for (const CascadeSlot& c : CascadeSlotsOf(sp.constants, t.constants)) {
                        slots.emplace(c, 0);
                    }
                }
                const CascadeSlot* slot = nullptr;
                for (auto& [c, held] : slots) {
                    held = std::min(held + (LightMatrices(AtSlot(sp.constants, c)).empty() ? -1 : 1),
                                    1000);
                    if (slot == nullptr || held > slots[*slot]) slot = &c;
                }
                if (shadowLog > 0 && slot != nullptr) {
                    slotLog += StringPrintf(" slot%u.%u+%zu held %d", slot->set, slot->binding,
                                           slot->offset, slots[*slot]);
                }
                float region[4];
                regionOf(sp.viewportX, sp.viewportY, sp.viewportWidth, sp.viewportHeight, region);
                gLightVote.add(slot != nullptr ? AtSlot(sp.constants, *slot) : sp.constants,
                               reversed, placed ? eye : nullptr, sp.draws, sp.depth.image,
                               sp.viewportHeight < 0, region);
                for (const Pass::Tile& t : sp.tiles) {
                    if (slot == nullptr) continue;
                    regionOf(t.x, t.y, t.width, t.height, region);
                    gLightVote.add(AtSlot(t.constants, *slot), reversed, placed ? eye : nullptr, 0,
                                   sp.depth.image, t.height < 0, region);
                }
            }
        }
        float scales[3] = {0, 0, 0};
        taps.camera.light = gLightVote.settle(taps.camera.toLight, scales);
        if (shadowLog > 0) {
            for (int k = 0; k < static_cast<int>(frame.steps.size()); k++) {
                if (!IsShadowPass(frame, k, roles)) continue;
                const Pass& sp = frame.steps[k].pass;
                const Image* mapImage = imageOf(sp.depth.image);
                logged += StringPrintf(
                        " pass#%d img%llu %ux%u d%u vp %.0f,%.0f %.0fx%.0f moves %u tiles %zu", k,
                        static_cast<unsigned long long>(sp.depth.image),
                        mapImage ? mapImage->width : 0, mapImage ? mapImage->height : 0, sp.draws,
                        sp.viewportX, sp.viewportY, sp.viewportWidth, sp.viewportHeight,
                        sp.viewportChanges, sp.tiles.size());
                const auto matrices = [&](const std::vector<Constants>& blocks) {
                    for (const LightMatrix& m : LightMatrices(blocks)) {
                        logged += StringPrintf(" [b%zu+%zu s%.4g t%.3g,%.3g,%.3g set%u.%u]", m.block,
                                               m.offset, m.scale, m.tx, m.ty, m.tz,
                                               blocks[m.block].set, blocks[m.block].binding);
                    }
                };
                matrices(sp.constants);
                for (const Pass::Tile& t : sp.tiles) {
                    logged += StringPrintf(" tile %.0f,%.0f:", t.x, t.y);
                    matrices(t.constants);
                }
            }
            logged += slotLog + " |";
        }
        if (taps.camera.light) {
            for (const LightVote::Cascade& c : gLightVote.cascades()) {
                if (taps.shadowCount == kShadowMaps) break;
                ShadowMap& m = taps.shadows[taps.shadowCount];
                const bool sampled = Sampleable(frame, c.image, &m.map, true);
                if (shadowLog > 0) {
                    logged += StringPrintf(" img%llu@%.2f,%.2f s%.4g age%llu%s",
                                           static_cast<unsigned long long>(c.image), c.region[0],
                                           c.region[1], c.scale,
                                           static_cast<unsigned long long>(
                                                   gLightVote.frame() - c.frame),
                                           sampled ? "" : " unsampled");
                }
                if (!sampled) continue;
                std::copy(c.matrix, c.matrix + 16, m.matrix);
                m.reversed = c.reversed;
                m.flipsY = c.flipsY;
                m.scale = c.scale;
                std::copy(c.region, c.region + 4, m.region);
                taps.shadowCount++;
            }
        }
        if (shadowLog > 0) {
            ALOGI("lapis: shadows%s light %d", logged.c_str(), taps.camera.light ? 1 : 0);
            shadowLog--;
        }
        // The light turning more than 10 degrees, for the log.
        static float last[3] = {0, 0, 0};
        const float* l = taps.camera.toLight;
        if (taps.camera.light && l[0] * last[0] + l[1] * last[1] + l[2] * last[2] < 0.985f) {
            ALOGI("lapis: light (%.2f %.2f %.2f), its nearest map's rows %.4g %.4g %.4g", l[0],
                  l[1], l[2], scales[0], scales[1], scales[2]);
            std::copy(l, l + 3, last);
        }
        if (roles.geometry >= 0) {
            for (const Target& t : frame.steps[roles.geometry].pass.colors) {
                if (taps.gbufferCount == kGBufferTargets) break;
                if (t.image == roles.scene || t.store != VK_ATTACHMENT_STORE_OP_STORE) continue;
                if (Sampleable(frame, t.image, &taps.gbuffer[taps.gbufferCount])) {
                    taps.gbufferCount++;
                }
            }
        }
        if (Tappable(frame, roles.scene, &taps.scene)) {
            taps.scene.upright = roles.scene == gSplit.sceneId && roles.direct;
            if (!taps.scene.upright) {
                taps.scene.orientation =
                        SceneOrientation(frame, gImages, roles, &taps.scene.orientationExact);
            }
            if (roles.geometry >= 0 && !taps.scene.upright) {
                const Pass& g = frame.steps[roles.geometry].pass;
                taps.scene.region = DrawnRegion(g.viewportX, g.viewportY, g.viewportWidth,
                                                g.viewportHeight, taps.scene.width,
                                                taps.scene.height);
            }
            if (roles.depthStored) Tappable(frame, roles.depth, &taps.depth);
            glassTaps(&taps);
            if (roles.jittered != 0 && Sampleable(frame, roles.jittered, &taps.jittered) &&
                Sampleable(frame, roles.resolved, &taps.resolved)) {
                Sampleable(frame, roles.velocity, &taps.velocity);
            }
            s.taps = taps;
            s.tapped = swapchain;
        }
    }
    if (s.tapped == VK_NULL_HANDLE && engaged && s.taps.scene.image != VK_NULL_HANDLE) {
        // Between the frames with S: a game's generated ones, or S out of reach in this frame.
        lapis::Taps held;
        held.scene = s.taps.scene;
        held.gameGenerates = s.taps.gameGenerates;
        held.held = true;
        s.taps = held;
        s.tapped = swapchain;
    }
    s.classifyUs.push_back(static_cast<uint32_t>((nowNs() - start) / 1000));
    hudPresent(s, frame, roles, swapchain);
    if (engaged && !s.engaged) {
        std::lock_guard lock(gLock);
        logRoles(roles, frame);
    } else if (!engaged && s.engaged) {
        std::lock_guard lock(gLock);
        const Roles& p = s.previous;
        ALOGI("lapis %s: released: %s; %zu steps, %s; before: direct %d target #%u D #%u S #%u "
              "V #%u, now: direct %d V #%u",
              Engine::get().package().c_str(), missName(s.tracker.lastMiss()), frame.steps.size(),
              missText(frame, roles).c_str(), p.direct, p.target, p.depth, p.scene, p.velocity,
              roles.direct, roles.velocity);
        survey::onMiss(frame, gImages, roles);
    }
    s.engaged = engaged;
    s.previous = roles;
    s.miss = s.tracker.lastMiss();
    s.frames++;
    s.engagedFrames += tappable;
    s.steps += frame.steps.size();
    if (roles.depth != 0) {
        s.depthFrames++;
        s.keptFrames += roles.depthStored;
        if (!roles.depthStored && s.dropped.empty()) s.dropped = depthDrop(frame, roles.depth);
    }
    const int64_t now = nowNs();
    if (tappable) scanProjection(s, roles, now);
    logStats(s, now, frame, roles);
    {
        std::lock_guard lock(gLock);
        Engine::get().reportLapis(statusOf(s, engaged ? s.tracker.roles() : roles, true));
    }
    if (surveyed) {
        Images images;
        {
            std::lock_guard lock(gLock);
            images = gImages;
        }
        survey::onFrame(frame, images, roles);
    }
    if (signalsOn()) updateSignals(s, frame, roles, tappable, now);
    // A late frame's own steps add nothing to draw from: the passes stay for the next one.
    if (!late) s.before = std::move(frame.steps);
}

bool signals(VkSwapchainKHR swapchain, EngineSignals* out) {
    if (!gOn.load(std::memory_order_relaxed)) return false;
    std::lock_guard lock(gStateLock);
    if (gState.signals == nullptr || !gState.signalsValid || gState.tapped != swapchain) return false;
    *out = gState.signals->last();
    return true;
}

bool taps(VkSwapchainKHR swapchain, Taps* out) {
    if (!gOn.load(std::memory_order_relaxed)) return false;
    std::lock_guard lock(gStateLock);
    if (gState.tapped == VK_NULL_HANDLE || gState.tapped != swapchain) return false;
    *out = gState.taps;
    return true;
}

void ignore(const VkCommandBuffer* buffers, uint32_t count) {
    if (!recording()) return;
    for (uint32_t i = 0; i < count; i++) recordOf(buffers[i]).ignored = true;
}

Calls calls() {
    Calls c;
    c.renderPass = gCalls[kRenderPass].load(std::memory_order_relaxed);
    c.renderPass2 = gCalls[kRenderPass2].load(std::memory_order_relaxed);
    c.rendering = gCalls[kRendering].load(std::memory_order_relaxed);
    c.draw = gCalls[kDraw].load(std::memory_order_relaxed);
    c.submit = gCalls[kSubmit].load(std::memory_order_relaxed);
    c.submit2 = gCalls[kSubmit2].load(std::memory_order_relaxed);
    c.execute = gCalls[kExecute].load(std::memory_order_relaxed);
    return c;
}

std::vector<std::string> hudSurveyLines(const Frame& frame, size_t step) {
    std::vector<std::string> lines;
    if (gHud.labeler == nullptr || gHud.labelled != &frame ||
        gHud.labelledSteps != frame.steps.size()) {
        return lines;
    }
    const HudLabeler& l = *gHud.labeler;
    const size_t features = HudLabeler::featureCount();
    const uint32_t end = l.firstDraw(step + 1);
    static const char kLabel[] = {'S', 'P', 'H', 'W'};
    for (uint32_t i = l.firstDraw(step); i < end && i < l.draws().size(); i++) {
        const HudDraw& d = l.draws()[i];
        std::string line = StringPrintf(
                "survey     hud d%u %c%02d w%c p=%" PRIx64 " n=%u i=%u s=%08x st=%x t=%x v=%x sc=%u",
                d.index, kLabel[l.labels()[i] & 3],
                std::min(99, static_cast<int>(l.confidences()[i] * 100)),
                kLabel[l.weakLabels()[i] & 3], d.pipeline, d.elements, d.instances, d.shader,
                d.state, d.textures, d.vertex, d.scissor);
        if (l.features().size() >= (i + 1) * features) {
            line += " f=";
            for (size_t f = 0; f < features; f++) {
                StringAppendF(&line, f == 0 ? "%.3g" : ",%.3g", l.features()[i * features + f]);
            }
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

std::string hudSurveySummary(const Frame& frame) {
    if (gHud.labeler == nullptr || gHud.labelled != &frame ||
        gHud.labelledSteps != frame.steps.size()) {
        return "";
    }
    const HudOutput& o = gHud.labeler->last();
    std::string names;
    for (size_t f = 0; f < HudLabeler::featureCount(); f++) {
        names += (f == 0 ? "" : ",") + HudLabeler::featureName(f);
    }
    return StringPrintf("survey   hud: skipping %d applied %d coverage %.2f confidence %.2f labels "
                        "S%u P%u H%u W%u cross %d share %.3f residue %.3f entries %u unsampled "
                        "%u features %s",
                        o.skipping, o.applied, o.coverage, o.confidence, o.labelled[0],
                        o.labelled[1], o.labelled[2], o.labelled[3], static_cast<int>(o.cross),
                        o.hudShare, o.residue, o.entries, o.unsampled, names.c_str());
}

void forEachMapping(const std::function<void(const uint8_t*, size_t, uint32_t)>& fn) {
    std::lock_guard lock(gMapLock);
    for (const auto& [memory, m] : gMappings) {
        if (m.ptr != nullptr) fn(m.ptr, m.size, m.type);
    }
}

}  // namespace framefx::lapis
