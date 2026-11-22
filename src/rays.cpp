/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "src/rays.h"

#include <android-base/properties.h>
#include <log/log.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "lapis/recorder.h"
#include "luwes/shaders.h"
#include "src/engine.h"
#include "src/pipeline_set.h"

extern "C" {
bool framefx_lapis_xfb_is_vertex(const uint32_t* code, size_t words);
size_t framefx_lapis_xfb_patch(const uint32_t* code, size_t words, uint32_t* out, size_t capacity,
                               uint32_t varying, uint8_t* why);
bool framefx_lapis_xfb_discards(const uint32_t* code, size_t words);
// lapis/rust/src/alpha.rs AlphaTest, Program and Block.
struct AlphaTest {
    uint32_t set, binding, samplerSet, samplerBinding;
    uint32_t location, component, channel, switched;
};
struct AlphaProgram;
struct AlphaBlock {
    uint32_t set, binding, size;
};
uint8_t framefx_lapis_alpha_read(const uint32_t* code, size_t words, AlphaTest* out,
                                 AlphaProgram** program);
void framefx_lapis_alpha_free(AlphaProgram* p);
size_t framefx_lapis_alpha_blocks(const AlphaProgram* p, AlphaBlock* out, size_t capacity);
size_t framefx_lapis_alpha_parts(const AlphaProgram* p, AlphaTest* out, size_t capacity);
uint8_t framefx_lapis_alpha_eval(const AlphaProgram* p, const uint8_t* const* data,
                                 const size_t* sizes, size_t n, float* ab, uint32_t* part);
struct SceneState;
SceneState* framefx_lapis_scene_new(uint64_t treeBytes, uint64_t pointBytes);
void framefx_lapis_scene_free(SceneState* s);
void framefx_lapis_scene_begin(SceneState* s);
uint8_t framefx_lapis_scene_place(SceneState* s, uint64_t key, uint32_t vertices, uint32_t* entry);
void framefx_lapis_scene_moved(SceneState* s, uint32_t entry, uint8_t far);
void framefx_lapis_scene_drop(SceneState* s, uint32_t entry);
size_t framefx_lapis_scene_evict(SceneState* s, uint64_t age, uint32_t* out, size_t capacity);
size_t framefx_lapis_scene_retained(SceneState* s, uint64_t age, uint32_t* out, size_t capacity);
uint64_t framefx_lapis_scene_take_moving_out_of_sight(SceneState* s);
uint64_t framefx_lapis_scene_alloc(SceneState* s, uint32_t which, uint64_t bytes);
void framefx_lapis_scene_release(SceneState* s, uint32_t which, uint64_t offset);
uint64_t framefx_lapis_scene_in_use(SceneState* s, uint32_t which);
}

namespace framefx::rays {
namespace {

constexpr uint64_t kStride = 16;           // a captured vertex: its clip position
constexpr uint64_t kStrideWithVarying = 32;  // and an alpha test's coordinate after it
constexpr uint32_t kNoVarying = UINT32_MAX;
constexpr uint32_t kSolid = UINT32_MAX;      // a surface with no alpha test
constexpr size_t kMaxAlphaBlocks = 4;        // uniform blocks an alpha test reads, at most
constexpr size_t kMaxAlphaBytes = 4096;      // of each
constexpr VkDeviceSize kSurfaceBytes = 32;   // rays.h Scene::surfaces, per instance
constexpr uint32_t kEntryBits = (1u << 22) - 1;  // an instance's custom index: its draw's entry
constexpr uint32_t kChangedBit = 1u << 22;       // and whether its tree changed this frame,
constexpr uint32_t kRetainedBit = 1u << 23;      // or it is held out of sight
constexpr uint64_t kChunk = 1ull << 20;    // ring space a command buffer takes at once
constexpr uint64_t kFramesInRing = 3;      // frames the ring holds before it wraps
// Frames whose instances, draw table and motion flags are kept apart: more than are in flight.
constexpr uint32_t kSlots = 4;
constexpr uint32_t kMaxInstances = 8192;   // draws a frame's top level holds
constexpr uint64_t kEvictAge = 120;        // frames a draw not seen keeps its tree
constexpr uint64_t kRetainAge = 90;        // frames a still draw not seen stays in the scene
constexpr uint64_t kPressedAge = 2 * kSlots;  // the same when memory runs short
constexpr uint8_t kKeep = 0, kBuild = 1, kRefit = 2;  // lapis/rust/src/scene.rs
constexpr uint64_t kNoSpace = UINT64_MAX;
constexpr uint32_t kTrees = 0, kPoints = 1;
constexpr VkDeviceSize kConstantBytes = 4096;
constexpr VkDeviceSize kStatBytes = 1024;
constexpr size_t kLabelBytes = 64;

// Triangles a frame may capture: past it, the frame's later draws stay out.
uint32_t maxTriangles() {
    static const uint32_t n = static_cast<uint32_t>(std::clamp<int64_t>(
            android::base::GetIntProperty<int64_t>("debug.framefx.rays_triangles", 600000), 10000,
            4000000));
    return n;
}

#define FRAMEFX_RAYS_VK(X)                                                                   \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory)        \
    X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(CreateShaderModule)                     \
    X(DestroyShaderModule) X(CreateComputePipelines) X(DestroyPipeline)                      \
    X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateDescriptorSetLayout)            \
    X(DestroyDescriptorSetLayout) X(CreateDescriptorPool) X(DestroyDescriptorPool)           \
    X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CmdBindPipeline)                     \
    X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDispatch) X(CmdPipelineBarrier)        \
    X(DeviceWaitIdle) X(CreateAccelerationStructureKHR) X(DestroyAccelerationStructureKHR)   \
    X(GetAccelerationStructureBuildSizesKHR) X(CmdBuildAccelerationStructuresKHR)            \
    X(GetAccelerationStructureDeviceAddressKHR) X(CmdBindTransformFeedbackBuffersEXT)        \
    X(CmdBeginTransformFeedbackEXT) X(CmdEndTransformFeedbackEXT) X(CreateSampler)             \
    X(DestroySampler)

struct Fns {
#define DECLARE(name) PFN_vk##name name = nullptr;
    FRAMEFX_RAYS_VK(DECLARE)
#undef DECLARE
    PFN_vkGetBufferDeviceAddress GetBufferDeviceAddress = nullptr;

    bool load(VkDevice device, PFN_vkGetDeviceProcAddr proc) {
        bool ok = true;
#define LOAD(name)                                                       \
    name = reinterpret_cast<PFN_vk##name>(proc(device, "vk" #name)); \
    ok = ok && name != nullptr;
        FRAMEFX_RAYS_VK(LOAD)
#undef LOAD
        GetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
                proc(device, "vkGetBufferDeviceAddress"));
        if (GetBufferDeviceAddress == nullptr) {
            GetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
                    proc(device, "vkGetBufferDeviceAddressKHR"));
        }
        return ok && GetBufferDeviceAddress != nullptr;
    }
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* mapped = nullptr;
};

// One captured draw: whole triangles in the ring, and what it is known by from frame to frame.
struct DrawRecord {
    uint64_t key = 0;     // its pipeline and draw parameters
    uint64_t offset = 0;  // ring bytes
    uint32_t vertices = 0;
    uint32_t step = 1;    // vec4s a captured vertex takes
    // An alpha-tested draw's: the texture it had bound, and its test as rays.h Scene::surfaces.
    VkImageView texture = VK_NULL_HANDLE;
    uint32_t test = 0;
    float cut[2] = {};
};

// A draw's own bottom-level tree, in the device's arenas.
struct Tree {
    VkAccelerationStructureKHR structure = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    uint64_t treeOffset = kNoSpace;
    uint64_t pointOffset = kNoSpace;
    uint32_t vertices = 0;
    double eye[3] = {};  // the camera its points are around
    VkImageView texture = VK_NULL_HANDLE;  // its draw's last, null solid
    uint32_t test = 0;
    float cut[2] = {};
};

// positions.comp's row for a draw.
struct DrawRow {
    uint32_t first;     // the draw's first vertex among the frame's
    uint32_t count;
    uint32_t captured;  // its first vertex in the ring
    uint32_t point;     // its first float in the points arena
    float offset[3];    // the camera its points are around, less this frame's
    float write;        // 1 write the points, 0 compare and flag motion
    uint32_t uv;        // its first vertex's coordinate in the UV arena, kNoVarying none
    uint32_t step;      // vec4s a captured vertex takes
    uint32_t component; // the coordinate's first in the captured varying
    uint32_t pad;
};

// Sizes of a tree of so many triangles.
struct Sizes {
    VkDeviceSize tree = 0;
    VkDeviceSize build = 0;
    VkDeviceSize update = 0;
};

// An alpha-tested pipeline's test: where it samples, and how it cuts with a draw's uniforms in.
struct AlphaPipeline {
    AlphaTest test = {};  // the first part's: the varying all parts sample at
    std::shared_ptr<AlphaProgram> program;
    std::vector<AlphaBlock> blocks;
    std::vector<AlphaTest> parts;
};

// What the rays hit: the device's ring, and the trees of the draws it captured, made at the first
// build.
struct Device {
    VkDevice device = VK_NULL_HANDLE;
    Fns vk;
    VkPhysicalDeviceMemoryProperties memory = {};
    VkDeviceSize scratchAlign = 256;
    Buffer ring;
    uint64_t ringBytes = 0;
    std::atomic<uint64_t> next{0};          // ring bytes handed out, ever
    std::atomic<uint64_t> frameVertices{0};  // captured since the last present

    std::mutex lock;  // the draws below
    std::vector<DrawRecord> pending;  // submitted since the last present
    std::vector<DrawRecord> frame;    // the frame the last present closed

    bool made = false;
    bool broken = false;
    SceneState* scene = nullptr;
    Buffer treeArena;
    Buffer pointArena;
    Buffer uvArena;  // two floats a vertex, beside the points arena's three
    VkSampler sampler = VK_NULL_HANDLE;
    Buffer scratch;
    Buffer topBuffer;
    VkAccelerationStructureKHR top = VK_NULL_HANDLE;
    VkDeviceSize topScratch = 0;  // where the top level's scratch starts in `scratch`
    std::vector<Tree> trees;      // by scene entry
    std::unordered_map<uint32_t, Sizes> sizes;  // by triangles
    // Per slot: the top level's instances, the draw table and the motion flags it fills.
    Buffer instances[kSlots];
    Buffer rows[kSlots];
    Buffer moved[kSlots];
    Buffer constants[kSlots];  // frameConstants()
    Buffer stats[kSlots];      // frameStats()
    Buffer surfaces[kSlots];   // rays.h Scene::surfaces
    uint8_t statLabels[kSlots][kLabelBytes] = {};
    uint32_t slot = 0;         // the last build's
    std::vector<uint32_t> rowEntries[kSlots];
    uint64_t frameCount = 0;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kSlots] = {};
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline convert = VK_NULL_HANDLE;
    // Its capturing pipelines, by how a draw's vertices become triangles.
    PipelineSet lists;
    PipelineSet strips;
    // Those of them alpha-tested, and their tests.
    PipelineSet alphas;
    std::mutex alphaLock;
    std::unordered_map<uint64_t, AlphaPipeline> alphaOf;
    uint32_t built = 0;     // trees, since the last log
    uint32_t refitted = 0;
    uint32_t kept = 0;
    uint32_t skipped = 0;
    uint32_t retained = 0;
    uint32_t compared = 0;   // draws whose points the GPU compared, and those it saw move
    uint32_t movedNear = 0;
    uint32_t movedFar = 0;
    uint32_t alphaDraws = 0;   // in the trees since the last log, and those left out
    std::atomic<uint32_t> untextured{0};
    std::atomic<uint32_t> whole{0};    // alpha-tested draws whose switches are off
    std::atomic<uint32_t> unknown{0};  // and those whose test could not be evaluated
    uint32_t overflowed = 0;
};

std::mutex gLock;  // the maps below
std::unordered_map<VkDevice, std::shared_ptr<Device>> gDevices;
std::unordered_map<VkQueue, VkDevice> gQueues;
// The device whose draws are captured: the one the game presents with. Read at every draw.
std::atomic<Device*> gActive{nullptr};
std::shared_ptr<Device> gActiveHold;

// A device made with capture, or null.
Device* deviceOf(VkDevice device) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    return it == gDevices.end() ? nullptr : it->second.get();
}

// --- Modules and pipelines ----------------------------------------------------------------------

struct Module {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule capturing = VK_NULL_HANDLE;  // the copy with transform feedback
    // A vertex module's code, and its copies that also capture the output at a location.
    std::vector<uint32_t> code;
    std::unordered_map<uint32_t, VkShaderModule> withVarying;
    bool discards = false;
    bool alphaRead = false;  // the discard is an alpha test the rays can repeat
    AlphaPipeline alpha;
};
// Discarding fragment shaders by what lapis/rust/src/alpha.rs made of them: 0 read, else why not.
std::atomic<uint32_t> gAlphaReads[8];
std::mutex gModuleLock;
std::unordered_map<VkShaderModule, Module> gModules;

std::atomic<uint32_t> gVertexModules{0};
std::atomic<uint32_t> gRefusedModules{0};
std::atomic<uint32_t> gCapturing{0};
std::atomic<uint32_t> gAlphaTested{0};
std::atomic<uint32_t> gAlphaCapturing{0};
std::atomic<uint32_t> gOther{0};

uint64_t bits(VkPipeline p) {
    return reinterpret_cast<uint64_t>(p);
}

std::atomic<PFN_vkCreateShaderModule> gCreateModule{nullptr};
std::atomic<PFN_vkDestroyShaderModule> gDestroyModule{nullptr};
std::atomic<PFN_vkCreateGraphicsPipelines> gCreateGraphics{nullptr};
std::atomic<PFN_vkDestroyPipeline> gDestroyPipeline{nullptr};

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice device,
                                                  const VkShaderModuleCreateInfo* info,
                                                  const VkAllocationCallbacks* allocator,
                                                  VkShaderModule* module) {
    const PFN_vkCreateShaderModule next = gCreateModule.load(std::memory_order_relaxed);
    const VkResult r = next(device, info, allocator, module);
    if (r != VK_SUCCESS || info->codeSize % 4 != 0) return r;
    if (deviceOf(device) == nullptr) return r;
    const size_t words = info->codeSize / 4;
    Module m;
    m.device = device;
    if (framefx_lapis_xfb_is_vertex(info->pCode, words)) {
        gVertexModules.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint32_t> patched(words + 64);
        uint8_t why = 0;
        const size_t n = framefx_lapis_xfb_patch(info->pCode, words, patched.data(),
                                                 patched.size(), kNoVarying, &why);
        static const bool dump = android::base::GetBoolProperty("debug.framefx.rays_dump", false);
        static std::atomic<int> dumped{0};
        if (dump && n != 0 && dumped.fetch_add(1) < 6) {
            const int k = dumped.load();
            const std::string dir = "/data/user/0/" + Engine::get().package() + "/cache/";
            for (const auto& [name, data, size] :
                 {std::tuple{"in", info->pCode, words}, std::tuple{"out", static_cast<const uint32_t*>(patched.data()), n}}) {
                const std::string path = dir + "rays-" + std::to_string(k) + "-" + name + ".spv";
                if (FILE* f = fopen(path.c_str(), "wb")) {
                    fwrite(data, 4, size, f);
                    fclose(f);
                }
            }
        }
        if (n != 0) {
            VkShaderModuleCreateInfo copy = *info;
            copy.codeSize = n * 4;
            copy.pCode = patched.data();
            if (next(device, &copy, allocator, &m.capturing) != VK_SUCCESS) {
                m.capturing = VK_NULL_HANDLE;
            }
        }
        if (m.capturing == VK_NULL_HANDLE) {
            const uint32_t refused = gRefusedModules.fetch_add(1, std::memory_order_relaxed);
            if (refused < 4) ALOGI("rays: a vertex shader keeps its code (%u)", why);
            return r;
        }
        m.code.assign(info->pCode, info->pCode + words);
    } else {
        m.discards = framefx_lapis_xfb_discards(info->pCode, words);
        if (!m.discards) return r;
        AlphaProgram* program = nullptr;
        const uint8_t why = framefx_lapis_alpha_read(info->pCode, words, &m.alpha.test, &program);
        m.alphaRead = why == 0;
        if (m.alphaRead) {
            // A pipeline keeps its test past the module, which the game may destroy first.
            m.alpha.program.reset(program, framefx_lapis_alpha_free);
            m.alpha.blocks.resize(framefx_lapis_alpha_blocks(program, nullptr, 0));
            framefx_lapis_alpha_blocks(program, m.alpha.blocks.data(), m.alpha.blocks.size());
            m.alpha.parts.resize(framefx_lapis_alpha_parts(program, nullptr, 0));
            framefx_lapis_alpha_parts(program, m.alpha.parts.data(), m.alpha.parts.size());
            m.alphaRead = m.alpha.blocks.size() <= kMaxAlphaBlocks;
        }
        const uint32_t seen = gAlphaReads[std::min<uint8_t>(why, 7)].fetch_add(1, std::memory_order_relaxed);
        static const bool dump = android::base::GetBoolProperty("debug.framefx.rays_dump", false);
        if (dump && why != 0 && seen < 40) {
            const std::string path = "/data/user/0/" + Engine::get().package() + "/cache/alpha-" +
                                     std::to_string(why) + "-" + std::to_string(seen) + ".spv";
            if (FILE* f = fopen(path.c_str(), "wb")) {
                fwrite(info->pCode, 4, words, f);
                fclose(f);
            }
        }
    }
    std::lock_guard lock(gModuleLock);
    gModules[*module] = m;
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice device, VkShaderModule module,
                                               const VkAllocationCallbacks* allocator) {
    std::vector<VkShaderModule> copies;
    if (module != VK_NULL_HANDLE) {
        std::lock_guard lock(gModuleLock);
        const auto it = gModules.find(module);
        if (it != gModules.end()) {
            if (it->second.capturing != VK_NULL_HANDLE) copies.push_back(it->second.capturing);
            for (const auto& [location, copy] : it->second.withVarying) copies.push_back(copy);
            gModules.erase(it);
        }
    }
    const PFN_vkDestroyShaderModule destroy = gDestroyModule.load(std::memory_order_relaxed);
    for (VkShaderModule copy : copies) destroy(device, copy, allocator);
    destroy(device, module, allocator);
}

bool hasDynamic(const VkPipelineDynamicStateCreateInfo* d, VkDynamicState state) {
    if (d == nullptr) return false;
    for (uint32_t i = 0; i < d->dynamicStateCount; i++) {
        if (d->pDynamicStates[i] == state) return true;
    }
    return false;
}

enum Shape { kNone = 0, kList = 1, kStrip = 2 };

// How a pipeline's draws would be captured, and the module its vertex stage takes for it.
// The copy of vertex module `m` that also captures its output at `location`, made the first time
// a pipeline asks. Under gModuleLock.
VkShaderModule withVarying(Module& m, uint32_t location) {
    const auto found = m.withVarying.find(location);
    if (found != m.withVarying.end()) return found->second;
    VkShaderModule copy = VK_NULL_HANDLE;
    std::vector<uint32_t> patched(m.code.size() + 96);
    uint8_t why = 0;
    const size_t n = framefx_lapis_xfb_patch(m.code.data(), m.code.size(), patched.data(),
                                             patched.size(), location, &why);
    if (n != 0) {
        VkShaderModuleCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = n * 4;
        info.pCode = patched.data();
        const PFN_vkCreateShaderModule create = gCreateModule.load(std::memory_order_relaxed);
        if (create(m.device, &info, nullptr, &copy) != VK_SUCCESS) copy = VK_NULL_HANDLE;
    }
    m.withVarying[location] = copy;  // a refusal too: asked once
    return copy;
}

// `alphaTested`: the pipeline discards; `test` filled when its discard is an alpha test the rays
// repeat, and then its draws are captured with the test's coordinate.
Shape shapeOf(const VkGraphicsPipelineCreateInfo& info, uint32_t* vertexStage,
              VkShaderModule* capturing, bool* alphaTested, AlphaPipeline* test) {
    if ((info.flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) != 0 || info.pStages == nullptr) {
        return kNone;
    }
    // Solid geometry the scene's depth holds: tested and written, rasterized.
    const VkPipelineDepthStencilStateCreateInfo* ds = info.pDepthStencilState;
    const VkPipelineRasterizationStateCreateInfo* rs = info.pRasterizationState;
    if (ds == nullptr || rs == nullptr || rs->rasterizerDiscardEnable) return kNone;
    const bool writes = (ds->depthTestEnable || hasDynamic(info.pDynamicState,
                                                           VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE)) &&
            (ds->depthWriteEnable ||
             hasDynamic(info.pDynamicState, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE));
    if (!writes) return kNone;
    const VkPipelineInputAssemblyStateCreateInfo* ia = info.pInputAssemblyState;
    if (ia == nullptr || hasDynamic(info.pDynamicState, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY) ||
        hasDynamic(info.pDynamicState, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE)) {
        return kNone;
    }
    Shape shape;
    switch (ia->topology) {
        case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: shape = kList; break;
        case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
        case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
            // A restart index splits a strip where the count cannot tell.
            if (ia->primitiveRestartEnable) return kNone;
            shape = kStrip;
            break;
        default: return kNone;
    }
    int vertex = -1;
    const Module* fragment = nullptr;
    std::lock_guard lock(gModuleLock);
    for (uint32_t i = 0; i < info.stageCount; i++) {
        const VkPipelineShaderStageCreateInfo& s = info.pStages[i];
        switch (s.stage) {
            case VK_SHADER_STAGE_VERTEX_BIT: vertex = static_cast<int>(i); break;
            case VK_SHADER_STAGE_FRAGMENT_BIT: {
                const auto it = gModules.find(s.module);
                if (it != gModules.end() && it->second.discards) {
                    *alphaTested = true;
                    if (!it->second.alphaRead) return kNone;
                    fragment = &it->second;
                }
                break;
            }
            default: return kNone;  // tessellation, geometry: not what the vertex shader wrote
        }
    }
    if (vertex < 0) return kNone;
    const auto it = gModules.find(info.pStages[vertex].module);
    if (it == gModules.end() || it->second.capturing == VK_NULL_HANDLE) return kNone;
    *vertexStage = static_cast<uint32_t>(vertex);
    if (fragment != nullptr) {
        *capturing = withVarying(it->second, fragment->alpha.test.location);
        if (*capturing == VK_NULL_HANDLE) return kNone;
        *test = fragment->alpha;
    } else {
        *capturing = it->second.capturing;
    }
    return shape;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache,
                                                       uint32_t n,
                                                       const VkGraphicsPipelineCreateInfo* infos,
                                                       const VkAllocationCallbacks* allocator,
                                                       VkPipeline* out) {
    const PFN_vkCreateGraphicsPipelines next = gCreateGraphics.load(std::memory_order_relaxed);
    Device* d = deviceOf(device);
    if (d == nullptr) return next(device, cache, n, infos, allocator, out);
    std::vector<Shape> shapes(n, kNone);
    std::vector<AlphaPipeline> tests(n);
    std::vector<VkGraphicsPipelineCreateInfo> changed(infos, infos + n);
    std::vector<std::vector<VkPipelineShaderStageCreateInfo>> stages(n);
    size_t count = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t vertex = 0;
        VkShaderModule capturing = VK_NULL_HANDLE;
        bool alpha = false;
        shapes[i] = shapeOf(infos[i], &vertex, &capturing, &alpha, &tests[i]);
        if (alpha && shapes[i] == kNone) gAlphaTested.fetch_add(1, std::memory_order_relaxed);
        if (shapes[i] == kNone) {
            if (!alpha) gOther.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        stages[i].assign(infos[i].pStages, infos[i].pStages + infos[i].stageCount);
        stages[i][vertex].module = capturing;
        changed[i].pStages = stages[i].data();
        count++;
    }
    if (count == 0 || !d->lists.reserve(count)) {
        return next(device, cache, n, infos, allocator, out);
    }
    VkResult r = next(device, cache, n, changed.data(), allocator, out);
    if (r != VK_SUCCESS && r != VK_PIPELINE_COMPILE_REQUIRED) {
        // The driver refused the copies: the game gets its pipelines as asked.
        ALOGW("rays: capturing pipelines refused (%d); %zu made as asked", r, count);
        for (uint32_t i = 0; i < n; i++) shapes[i] = kNone;
        r = next(device, cache, n, infos, allocator, out);
    }
    for (uint32_t i = 0; i < n; i++) {
        if (shapes[i] == kNone || out[i] == VK_NULL_HANDLE) continue;
        (shapes[i] == kList ? d->lists : d->strips).insert(bits(out[i]));
        gCapturing.fetch_add(1, std::memory_order_relaxed);
        if (tests[i].program != nullptr && d->alphas.reserve(1)) {
            {
                std::lock_guard lock(d->alphaLock);
                d->alphaOf[bits(out[i])] = tests[i];
            }
            d->alphas.insert(bits(out[i]));
            gAlphaCapturing.fetch_add(1, std::memory_order_relaxed);
        }
    }
    d->lists.unreserve(count);
    return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline,
                                           const VkAllocationCallbacks* allocator) {
    if (pipeline != VK_NULL_HANDLE) {
        if (Device* d = deviceOf(device)) {
            d->lists.erase(bits(pipeline));
            d->strips.erase(bits(pipeline));
            if (d->alphas.contains(bits(pipeline))) {
                d->alphas.erase(bits(pipeline));
                std::lock_guard lock(d->alphaLock);
                d->alphaOf.erase(bits(pipeline));
            }
        }
    }
    gDestroyPipeline.load(std::memory_order_relaxed)(device, pipeline, allocator);
}

// --- Recording ----------------------------------------------------------------------------------

// The capture in one command buffer: the ring space it holds, and the draws written into it.
struct Recording {
    Shape bound = kNone;
    uint64_t pipeline = 0;
    bool alpha = false;  // the bound pipeline's alpha test, repeated by the rays
    AlphaPipeline test;
    bool active = false;
    uint64_t chunkEnd = 0;  // ring bytes, absolute offsets
    uint64_t cursor = 0;
    std::vector<DrawRecord> draws;
};

std::mutex gRecordingLock;
std::unordered_map<VkCommandBuffer, std::unique_ptr<Recording>> gRecordings;
thread_local VkCommandBuffer tCb = VK_NULL_HANDLE;
thread_local Recording* tRecording = nullptr;

Recording& recordingOf(VkCommandBuffer cb) {
    if (tCb == cb && tRecording != nullptr) return *tRecording;
    std::lock_guard lock(gRecordingLock);
    auto& slot = gRecordings[cb];
    if (!slot) slot = std::make_unique<Recording>();
    tCb = cb;
    tRecording = slot.get();
    return *slot;
}

// The one already made for `cb`, or none: commands outside a capture need no state.
Recording* existingOf(VkCommandBuffer cb) {
    if (tCb == cb) return tRecording;
    std::lock_guard lock(gRecordingLock);
    const auto it = gRecordings.find(cb);
    return it == gRecordings.end() ? nullptr : it->second.get();
}

void endCapture(Device& d, VkCommandBuffer cb, Recording& r) {
    if (!r.active) return;
    d.vk.CmdEndTransformFeedbackEXT(cb, 0, 0, nullptr, nullptr);
    r.active = false;
}

uint64_t mixKey(uint64_t h, uint64_t v) {
    return (h ^ v) * 0x100000001b3ull;
}

// Ring space for `bytes` more in `r`, starting a new chunk when the one held is short.
bool reserve(Device& d, Recording& r, uint64_t bytes) {
    if (r.cursor != 0 && r.cursor + bytes <= r.chunkEnd) return true;
    const uint64_t want = (std::max(bytes, kChunk) + kChunk - 1) / kChunk * kChunk;
    if (want > d.ringBytes / kFramesInRing) return false;
    for (int attempt = 0; attempt < 2; attempt++) {
        const uint64_t at = d.next.fetch_add(want, std::memory_order_relaxed) % d.ringBytes;
        if (at + want <= d.ringBytes) {
            r.cursor = at;
            r.chunkEnd = at + want;
            return true;
        }
    }
    return false;
}

// Vertices transform feedback writes for one draw of `count` vertices or indices.
uint64_t capturedVertices(Shape shape, uint32_t count, uint32_t instances) {
    const uint64_t perInstance = shape == kList ? count - count % 3 : count < 3 ? 0 : 3ull * (count - 2);
    return perInstance * instances;
}

// The alpha test of the draw about to be recorded in `cb`, its uniforms as they stand: 0 it cuts
// where cut[0]·t + cut[1] < 0, t sampled as p.parts[*part] says; 1 its surface is whole; 2
// unknown.
uint8_t alphaVerdict(VkCommandBuffer cb, const AlphaPipeline& p, float cut[2], uint32_t* part) {
    static thread_local uint8_t bytes[kMaxAlphaBlocks][kMaxAlphaBytes];
    const uint8_t* data[kMaxAlphaBlocks] = {};
    size_t sizes[kMaxAlphaBlocks] = {};
    const size_t n = std::min(p.blocks.size(), kMaxAlphaBlocks);
    for (size_t i = 0; i < n; i++) {
        const AlphaBlock& b = p.blocks[i];
        sizes[i] = lapis::uniformBytes(cb, b.set, b.binding, std::min<size_t>(b.size, kMaxAlphaBytes),
                                       bytes[i]);
        data[i] = sizes[i] != 0 ? bytes[i] : nullptr;
    }
    const uint8_t verdict = framefx_lapis_alpha_eval(p.program.get(), data, sizes, n, cut, part);
    return verdict == 0 && *part >= p.parts.size() ? 2 : verdict;
}

// Before a draw in `cb`: transform feedback on when it captures, off when it does not. A draw is
// known from frame to frame by its pipeline, counts and where in its buffers it starts (`start`):
// the game sorts its draws by distance, so draws of one mesh alike are told apart by order alone
// only where they share all of these (lapis/rust/src/scene.rs). Streaming moves a draw's start
// now and then; its tree is built again.
void beforeDraw(VkCommandBuffer cb, uint32_t count, uint32_t instances, bool indexed,
                uint64_t start) {
    Device* d = gActive.load(std::memory_order_acquire);
    if (d == nullptr) return;
    Recording* existing = existingOf(cb);
    const Shape shape = existing != nullptr ? existing->bound : kNone;
    if (shape == kNone) return;
    Recording& r = *existing;
    const uint64_t vertices = capturedVertices(shape, count, instances);
    const uint64_t limit = 3ull * maxTriangles();
    const bool captures = vertices != 0 && lapis::sceneState(cb) == 1 &&
            d->frameVertices.load(std::memory_order_relaxed) + vertices <= limit;
    // An alpha-tested draw goes with the texture its test samples and its test made numbers by
    // its uniforms; whole where its switches are off; out when either is unknown.
    VkImageView texture = VK_NULL_HANDLE;
    uint8_t verdict = 1;
    float cut[2] = {};
    uint32_t part = 0;
    if (captures && r.alpha) {
        verdict = alphaVerdict(cb, r.test, cut, &part);
        if (verdict == 0) {
            const AlphaTest& t = r.test.parts[part];
            texture = lapis::sampledView(cb, t.set, t.binding);
        }
    }
    const bool out = captures && r.alpha && (verdict == 2 || (verdict == 0 && texture == VK_NULL_HANDLE));
    if (!captures || out) {
        if (out) (verdict == 2 ? d->unknown : d->untextured)++;
        endCapture(*d, cb, r);
        return;
    }
    if (r.alpha && verdict == 1) d->whole++;
    const uint64_t bytes = vertices * (r.alpha ? kStrideWithVarying : kStride);
    if (!r.active || r.cursor + bytes > r.chunkEnd) {
        endCapture(*d, cb, r);
        if (!reserve(*d, r, bytes)) return;
        const VkDeviceSize offset = r.cursor;
        const VkDeviceSize size = r.chunkEnd - r.cursor;
        d->vk.CmdBindTransformFeedbackBuffersEXT(cb, 0, 1, &d->ring.buffer, &offset, &size);
        d->vk.CmdBeginTransformFeedbackEXT(cb, 0, 0, nullptr, nullptr);
        r.active = true;
    }
    uint64_t key = 0xcbf29ce484222325ull;
    for (uint64_t v : {r.pipeline, uint64_t{count}, uint64_t{instances}, uint64_t{indexed}, start}) {
        key = mixKey(key, v);
    }
    DrawRecord draw = {key, r.cursor, static_cast<uint32_t>(vertices)};
    draw.step = r.alpha ? kStrideWithVarying / 16 : 1;
    if (texture != VK_NULL_HANDLE) {
        draw.texture = texture;
        draw.test = (r.test.parts[part].channel & 3) | (r.test.test.component & 3) << 3;
        std::copy(cut, cut + 2, draw.cut);
    }
    r.draws.push_back(draw);
    r.cursor += bytes;
    d->frameVertices.fetch_add(vertices, std::memory_order_relaxed);
}

// Transform feedback must be off before most state changes and at a pass's end.
void stop(VkCommandBuffer cb) {
    Device* d = gActive.load(std::memory_order_acquire);
    if (d == nullptr) return;
    Recording* r = existingOf(cb);
    if (r != nullptr) endCapture(*d, cb, *r);
}

std::atomic<PFN_vkCmdBindPipeline> gBindPipeline{nullptr};
std::atomic<PFN_vkCmdDraw> gDraw{nullptr};
std::atomic<PFN_vkCmdDrawIndexed> gDrawIndexed{nullptr};
std::atomic<PFN_vkCmdDrawIndirect> gDrawIndirect{nullptr};
std::atomic<PFN_vkCmdDrawIndexedIndirect> gDrawIndexedIndirect{nullptr};
std::atomic<PFN_vkCmdDrawIndirectCount> gDrawIndirectCount{nullptr};
std::atomic<PFN_vkCmdDrawIndexedIndirectCount> gDrawIndexedIndirectCount{nullptr};
std::atomic<PFN_vkCmdEndRenderPass> gEndRenderPass{nullptr};
std::atomic<PFN_vkCmdEndRenderPass2> gEndRenderPass2{nullptr};
std::atomic<PFN_vkCmdNextSubpass> gNextSubpass{nullptr};
std::atomic<PFN_vkCmdNextSubpass2> gNextSubpass2{nullptr};
std::atomic<PFN_vkCmdEndRendering> gEndRendering{nullptr};
std::atomic<PFN_vkCmdExecuteCommands> gExecuteCommands{nullptr};
std::atomic<PFN_vkBeginCommandBuffer> gBeginCommandBuffer{nullptr};
std::atomic<PFN_vkEndCommandBuffer> gEndCommandBuffer{nullptr};
std::atomic<PFN_vkQueueSubmit> gQueueSubmit{nullptr};
std::atomic<PFN_vkQueueSubmit2> gQueueSubmit2{nullptr};

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point,
                                           VkPipeline pipeline) {
    // No bind of any kind while transform feedback is active.
    if (Device* d = gActive.load(std::memory_order_acquire)) {
        const Shape shape = point != VK_PIPELINE_BIND_POINT_GRAPHICS     ? kNone
                : d->lists.contains(bits(pipeline))  ? kList
                : d->strips.contains(bits(pipeline)) ? kStrip
                                                            : kNone;
        Recording* r = shape != kNone ? &recordingOf(cb) : existingOf(cb);
        if (r != nullptr) {
            endCapture(*d, cb, *r);
            if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
                r->bound = shape;
                r->pipeline = bits(pipeline);
                r->alpha = shape != kNone && d->alphas.contains(bits(pipeline));
                if (r->alpha) {
                    std::lock_guard lock(d->alphaLock);
                    const auto it = d->alphaOf.find(bits(pipeline));
                    r->alpha = it != d->alphaOf.end();
                    if (r->alpha) r->test = it->second;
                }
            }
        }
    }
    gBindPipeline.load(std::memory_order_relaxed)(cb, point, pipeline);
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(VkCommandBuffer cb, uint32_t vertices, uint32_t instances,
                                   uint32_t firstVertex, uint32_t firstInstance) {
    beforeDraw(cb, vertices, instances, false, uint64_t{firstVertex} | uint64_t{firstInstance} << 32);
    gDraw.load(std::memory_order_relaxed)(cb, vertices, instances, firstVertex, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(VkCommandBuffer cb, uint32_t indices, uint32_t instances,
                                          uint32_t firstIndex, int32_t vertexOffset,
                                          uint32_t firstInstance) {
    beforeDraw(cb, indices, instances, true,
               uint64_t{firstIndex} | uint64_t{static_cast<uint32_t>(vertexOffset)} << 32);
    gDrawIndexed.load(std::memory_order_relaxed)(cb, indices, instances, firstIndex, vertexOffset,
                                                 firstInstance);
}

// Counts the CPU cannot see: never captured.
VKAPI_ATTR void VKAPI_CALL CmdDrawIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,
                                           uint32_t count, uint32_t stride) {
    stop(cb);
    gDrawIndirect.load(std::memory_order_relaxed)(cb, buffer, offset, count, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buffer,
                                                  VkDeviceSize offset, uint32_t count,
                                                  uint32_t stride) {
    stop(cb);
    gDrawIndexedIndirect.load(std::memory_order_relaxed)(cb, buffer, offset, count, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buffer,
                                                VkDeviceSize offset, VkBuffer countBuffer,
                                                VkDeviceSize countOffset, uint32_t maxCount,
                                                uint32_t stride) {
    stop(cb);
    gDrawIndirectCount.load(std::memory_order_relaxed)(cb, buffer, offset, countBuffer, countOffset,
                                                       maxCount, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buffer,
                                                       VkDeviceSize offset, VkBuffer countBuffer,
                                                       VkDeviceSize countOffset, uint32_t maxCount,
                                                       uint32_t stride) {
    stop(cb);
    gDrawIndexedIndirectCount.load(std::memory_order_relaxed)(cb, buffer, offset, countBuffer,
                                                              countOffset, maxCount, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(VkCommandBuffer cb) {
    stop(cb);
    gEndRenderPass.load(std::memory_order_relaxed)(cb);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo* info) {
    stop(cb);
    gEndRenderPass2.load(std::memory_order_relaxed)(cb, info);
}

VKAPI_ATTR void VKAPI_CALL CmdNextSubpass(VkCommandBuffer cb, VkSubpassContents contents) {
    stop(cb);
    gNextSubpass.load(std::memory_order_relaxed)(cb, contents);
}

VKAPI_ATTR void VKAPI_CALL CmdNextSubpass2(VkCommandBuffer cb, const VkSubpassBeginInfo* begin,
                                           const VkSubpassEndInfo* end) {
    stop(cb);
    gNextSubpass2.load(std::memory_order_relaxed)(cb, begin, end);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRendering(VkCommandBuffer cb) {
    stop(cb);
    gEndRendering.load(std::memory_order_relaxed)(cb);
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(VkCommandBuffer cb, uint32_t n,
                                              const VkCommandBuffer* buffers) {
    stop(cb);
    gExecuteCommands.load(std::memory_order_relaxed)(cb, n, buffers);
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer cb,
                                                  const VkCommandBufferBeginInfo* info) {
    if (gActive.load(std::memory_order_acquire) != nullptr) {
        if (Recording* r = existingOf(cb)) {
            r->bound = kNone;
            r->active = false;
            r->cursor = 0;
            r->chunkEnd = 0;
            r->draws.clear();
        }
    }
    return gBeginCommandBuffer.load(std::memory_order_relaxed)(cb, info);
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer cb) {
    stop(cb);
    return gEndCommandBuffer.load(std::memory_order_relaxed)(cb);
}

// The draws of `buffers` go to the frame being submitted.
void submitted(VkQueue queue, const VkCommandBuffer* buffers, uint32_t n) {
    Device* d = gActive.load(std::memory_order_acquire);
    if (d == nullptr || n == 0) return;
    {
        std::lock_guard lock(gLock);
        const auto q = gQueues.find(queue);
        if (q == gQueues.end() || q->second != d->device) return;
    }
    std::vector<DrawRecord> draws;
    {
        std::lock_guard lock(gRecordingLock);
        for (uint32_t i = 0; i < n; i++) {
            const auto it = gRecordings.find(buffers[i]);
            if (it == gRecordings.end()) continue;
            const auto& s = it->second->draws;
            draws.insert(draws.end(), s.begin(), s.end());
        }
    }
    if (draws.empty()) return;
    std::lock_guard lock(d->lock);
    d->pending.insert(d->pending.end(), draws.begin(), draws.end());
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo* submits,
                                           VkFence fence) {
    for (uint32_t i = 0; i < n; i++) {
        submitted(queue, submits[i].pCommandBuffers, submits[i].commandBufferCount);
    }
    return gQueueSubmit.load(std::memory_order_relaxed)(queue, n, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2(VkQueue queue, uint32_t n, const VkSubmitInfo2* submits,
                                            VkFence fence) {
    for (uint32_t i = 0; i < n; i++) {
        std::vector<VkCommandBuffer> buffers;
        for (uint32_t k = 0; k < submits[i].commandBufferInfoCount; k++) {
            buffers.push_back(submits[i].pCommandBufferInfos[k].commandBuffer);
        }
        submitted(queue, buffers.data(), static_cast<uint32_t>(buffers.size()));
    }
    return gQueueSubmit2.load(std::memory_order_relaxed)(queue, n, submits, fence);
}

template <typename Fn>
PFN_vkVoidFunction wrap(std::atomic<Fn>* slot, PFN_vkVoidFunction next, Fn own) {
    Fn none = nullptr;
    const auto want = reinterpret_cast<Fn>(next);
    if (!slot->compare_exchange_strong(none, want) && none != want) return nullptr;
    return reinterpret_cast<PFN_vkVoidFunction>(own);
}

// --- Device memory ------------------------------------------------------------------------------

bool makeBuffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage, bool host, Buffer* out) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d.vk.CreateBuffer(d.device, &info, nullptr, &out->buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d.vk.GetBufferMemoryRequirements(d.device, out->buffer, &req);
    const VkMemoryPropertyFlags want = host
            ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; i++) {
        if ((req.memoryTypeBits >> i & 1) != 0 &&
            (d.memory.memoryTypes[i].propertyFlags & want) == want) {
            type = i;
            break;
        }
    }
    const bool address = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    VkMemoryAllocateFlagsInfo flags = {};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = address ? &flags : nullptr;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (type == UINT32_MAX ||
        d.vk.AllocateMemory(d.device, &alloc, nullptr, &out->memory) != VK_SUCCESS ||
        d.vk.BindBufferMemory(d.device, out->buffer, out->memory, 0) != VK_SUCCESS) {
        return false;
    }
    if (host && d.vk.MapMemory(d.device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->mapped) != VK_SUCCESS) {
        return false;
    }
    out->size = size;
    if (address) {
        VkBufferDeviceAddressInfo a = {};
        a.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        a.buffer = out->buffer;
        out->address = d.vk.GetBufferDeviceAddress(d.device, &a);
    }
    return true;
}

void freeBuffer(Device& d, Buffer& b) {
    if (b.buffer != VK_NULL_HANDLE) d.vk.DestroyBuffer(d.device, b.buffer, nullptr);
    if (b.memory != VK_NULL_HANDLE) d.vk.FreeMemory(d.device, b.memory, nullptr);
    b = Buffer();
}

VkAccelerationStructureGeometryKHR triangles(VkDeviceAddress points, uint32_t maxVertex) {
    VkAccelerationStructureGeometryKHR g = {};
    g.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& t = g.geometry.triangles;
    t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    t.vertexData.deviceAddress = points;
    t.vertexStride = 12;
    t.maxVertex = maxVertex;
    t.indexType = VK_INDEX_TYPE_NONE_KHR;
    return g;
}

VkAccelerationStructureGeometryKHR instancesOf(VkDeviceAddress data) {
    VkAccelerationStructureGeometryKHR g = {};
    g.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    g.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    g.geometry.instances.data.deviceAddress = data;
    return g;
}

// Built once and kept, so built for tracing; refitted while its draw moves.
constexpr VkBuildAccelerationStructureFlagsKHR kBottomFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_BIT_KHR;
constexpr VkBuildAccelerationStructureFlagsKHR kTopFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) {
    return (v + a - 1) / a * a;
}

VkDeviceSize powerOfTwoAtLeast(VkDeviceSize v) {
    VkDeviceSize p = 1;
    while (p < v) p <<= 1;
    return p;
}

Sizes sizesFor(Device& d, uint32_t triangleCount) {
    const auto it = d.sizes.find(triangleCount);
    if (it != d.sizes.end()) return it->second;
    VkAccelerationStructureGeometryKHR geometry = triangles(0, 3 * triangleCount - 1);
    VkAccelerationStructureBuildGeometryInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = kBottomFlags;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    VkAccelerationStructureBuildSizesInfoKHR got = {};
    got.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    d.vk.GetAccelerationStructureBuildSizesKHR(d.device,
                                               VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                               &info, &triangleCount, &got);
    const Sizes out = {got.accelerationStructureSize, got.buildScratchSize, got.updateScratchSize};
    if (d.sizes.size() > 8192) d.sizes.clear();
    d.sizes[triangleCount] = out;
    return out;
}

VkAccelerationStructureKHR makeStructure(Device& d, VkBuffer backing, VkDeviceSize offset,
                                         VkDeviceSize size, VkAccelerationStructureTypeKHR type) {
    VkAccelerationStructureCreateInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    info.buffer = backing;
    info.offset = offset;
    info.size = size;
    info.type = type;
    VkAccelerationStructureKHR out = VK_NULL_HANDLE;
    d.vk.CreateAccelerationStructureKHR(d.device, &info, nullptr, &out);
    return out;
}

VkDeviceAddress addressOf(Device& d, VkAccelerationStructureKHR structure) {
    VkAccelerationStructureDeviceAddressInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    info.accelerationStructure = structure;
    return d.vk.GetAccelerationStructureDeviceAddressKHR(d.device, &info);
}

void freeTree(Device& d, Tree& t) {
    if (t.structure != VK_NULL_HANDLE) {
        d.vk.DestroyAccelerationStructureKHR(d.device, t.structure, nullptr);
    }
    if (t.treeOffset != kNoSpace) framefx_lapis_scene_release(d.scene, kTrees, t.treeOffset);
    if (t.pointOffset != kNoSpace) framefx_lapis_scene_release(d.scene, kPoints, t.pointOffset);
    t = Tree();
}

// Trees of draws not drawn for more than `age` frames, freed.
void evictTrees(Device& d, uint64_t age) {
    uint32_t gone[256];
    size_t n;
    while ((n = framefx_lapis_scene_evict(d.scene, age, gone, 256)) != 0) {
        for (size_t i = 0; i < n; i++) {
            if (gone[i] < d.trees.size()) freeTree(d, d.trees[gone[i]]);
        }
    }
}

// Space and a tree for a draw of `vertices`; when the arenas are full, trees of draws gone for a
// while make room first.
bool makeTree(Device& d, Tree& t, uint32_t vertices, const Sizes& sz, bool* pressed) {
    for (int attempt = 0; attempt < 2; attempt++) {
        t.treeOffset = framefx_lapis_scene_alloc(d.scene, kTrees, sz.tree);
        t.pointOffset = framefx_lapis_scene_alloc(d.scene, kPoints, VkDeviceSize{vertices} * 12);
        if (t.treeOffset != kNoSpace && t.pointOffset != kNoSpace) break;
        freeTree(d, t);
        if (*pressed) return false;
        *pressed = true;
        evictTrees(d, kPressedAge);
    }
    if (t.treeOffset == kNoSpace) return false;
    t.structure = makeStructure(d, d.treeArena.buffer, t.treeOffset, sz.tree,
                                VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    if (t.structure == VK_NULL_HANDLE) {
        freeTree(d, t);
        return false;
    }
    t.address = addressOf(d, t.structure);
    t.vertices = vertices;
    return true;
}

// Everything a build needs, sized for the most triangles a frame captures.
bool make(Device& d) {
    if (d.made || d.broken) return d.made;
    d.broken = true;
    const uint32_t maxTris = maxTriangles();
    // The arenas hold twice a frame's worst: draws out of sight a while, and the buddy blocks'
    // rounding up.
    const Sizes whole = sizesFor(d, maxTris);
    const VkDeviceSize treeBytes = powerOfTwoAtLeast(2 * whole.tree);
    const VkDeviceSize pointBytes = powerOfTwoAtLeast(VkDeviceSize{maxTris} * 36 * 2);
    if (!makeBuffer(d, treeBytes,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    false, &d.treeArena) ||
        !makeBuffer(d, pointBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                    false, &d.pointArena) ||
        !makeBuffer(d, (pointBytes / 12 + 1) * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                    &d.uvArena)) {
        return false;
    }
    VkSamplerCreateInfo sampler = {};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.maxLod = VK_LOD_CLAMP_NONE;
    if (d.vk.CreateSampler(d.device, &sampler, nullptr, &d.sampler) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < kSlots; i++) {
        if (!makeBuffer(d, kMaxInstances * sizeof(VkAccelerationStructureInstanceKHR),
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        true, &d.instances[i]) ||
            !makeBuffer(d, kMaxInstances * sizeof(DrawRow), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        true, &d.rows[i]) ||
            !makeBuffer(d, kMaxInstances * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        true, &d.moved[i]) ||
            !makeBuffer(d, kConstantBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true,
                        &d.constants[i]) ||
            !makeBuffer(d, kStatBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &d.stats[i]) ||
            !makeBuffer(d, kMaxInstances * kSurfaceBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true,
                        &d.surfaces[i])) {
            return false;
        }
        memset(d.moved[i].mapped, 0, kMaxInstances * sizeof(uint32_t));
        memset(d.stats[i].mapped, 0, kStatBytes);
    }

    VkAccelerationStructureGeometryKHR top = instancesOf(0);
    VkAccelerationStructureBuildGeometryInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info.flags = kTopFlags;
    info.geometryCount = 1;
    info.pGeometries = &top;
    const uint32_t most = kMaxInstances;
    VkAccelerationStructureBuildSizesInfoKHR topSizes = {};
    topSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    d.vk.GetAccelerationStructureBuildSizesKHR(d.device,
                                               VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                               &info, &most, &topSizes);
    if (!makeBuffer(d, topSizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    false, &d.topBuffer)) {
        return false;
    }
    d.top = makeStructure(d, d.topBuffer.buffer, 0, topSizes.accelerationStructureSize,
                          VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    if (d.top == VK_NULL_HANDLE) return false;

    // Scratch: the top level's, then the trees built or refitted in a frame, up to half the most
    // triangles a frame captures; the rest wait for the next frames.
    const VkDeviceSize align = d.scratchAlign;
    d.topScratch = alignUp(topSizes.buildScratchSize, align);
    const VkDeviceSize trees = alignUp(sizesFor(d, std::max(maxTris / 2, 1u)).build, align);
    if (!makeBuffer(d, d.topScratch + trees + align,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    false, &d.scratch)) {
        return false;
    }
    d.scene = framefx_lapis_scene_new(treeBytes, pointBytes);

    // positions.comp: ring, points, the draw table, its motion flags and the coordinates.
    // positions.comp's buffers: captured, points, rows, moved, coordinates and the instances.
    constexpr uint32_t kPositionBuffers = 6;
    VkDescriptorSetLayoutBinding bindings[kPositionBuffers] = {};
    for (uint32_t i = 0; i < kPositionBuffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout = {};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = kPositionBuffers;
    layout.pBindings = bindings;
    if (d.vk.CreateDescriptorSetLayout(d.device, &layout, nullptr, &d.setLayout) != VK_SUCCESS) {
        return false;
    }
    const VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 80};
    VkPipelineLayoutCreateInfo pipe = {};
    pipe.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe.setLayoutCount = 1;
    pipe.pSetLayouts = &d.setLayout;
    pipe.pushConstantRangeCount = 1;
    pipe.pPushConstantRanges = &push;
    if (d.vk.CreatePipelineLayout(d.device, &pipe, nullptr, &d.pipelineLayout) != VK_SUCCESS) {
        return false;
    }
    const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kPositionBuffers * kSlots};
    VkDescriptorPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = kSlots;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    if (d.vk.CreateDescriptorPool(d.device, &pool, nullptr, &d.pool) != VK_SUCCESS) return false;
    for (uint32_t slot = 0; slot < kSlots; slot++) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = d.pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &d.setLayout;
        if (d.vk.AllocateDescriptorSets(d.device, &alloc, &d.sets[slot]) != VK_SUCCESS) {
            return false;
        }
        const VkDescriptorBufferInfo buffers[kPositionBuffers] = {
                {d.ring.buffer, 0, VK_WHOLE_SIZE},       {d.pointArena.buffer, 0, VK_WHOLE_SIZE},
                {d.rows[slot].buffer, 0, VK_WHOLE_SIZE}, {d.moved[slot].buffer, 0, VK_WHOLE_SIZE},
                {d.uvArena.buffer, 0, VK_WHOLE_SIZE},    {d.instances[slot].buffer, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[kPositionBuffers] = {};
        for (uint32_t i = 0; i < kPositionBuffers; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = d.sets[slot];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        d.vk.UpdateDescriptorSets(d.device, kPositionBuffers, writes, 0, nullptr);
    }
    const auto& spirv = luwes::shaders::kPositionsCompSpirv;
    VkShaderModuleCreateInfo module = {};
    module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module.codeSize = spirv.size;
    module.pCode = spirv.code;
    if (d.vk.CreateShaderModule(d.device, &module, nullptr, &d.module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo compute = {};
    compute.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute.stage.module = d.module;
    compute.stage.pName = "main";
    compute.layout = d.pipelineLayout;
    if (d.vk.CreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &compute, nullptr, &d.convert) !=
        VK_SUCCESS) {
        return false;
    }
    ALOGI("rays: scene of up to %u triangles a frame: trees %.0f MB, points %.0f MB, scratch "
          "%.1f MB, ring %.1f MB",
          maxTris, treeBytes / 1048576.0, pointBytes / 1048576.0, d.scratch.size / 1048576.0,
          d.ringBytes / 1048576.0);
    d.made = true;
    d.broken = false;
    return true;
}

void freeDevice(Device& d) {
    if (d.convert != VK_NULL_HANDLE) d.vk.DestroyPipeline(d.device, d.convert, nullptr);
    if (d.module != VK_NULL_HANDLE) d.vk.DestroyShaderModule(d.device, d.module, nullptr);
    if (d.pool != VK_NULL_HANDLE) d.vk.DestroyDescriptorPool(d.device, d.pool, nullptr);
    if (d.pipelineLayout != VK_NULL_HANDLE) {
        d.vk.DestroyPipelineLayout(d.device, d.pipelineLayout, nullptr);
    }
    if (d.setLayout != VK_NULL_HANDLE) {
        d.vk.DestroyDescriptorSetLayout(d.device, d.setLayout, nullptr);
    }
    if (d.top != VK_NULL_HANDLE) d.vk.DestroyAccelerationStructureKHR(d.device, d.top, nullptr);
    if (d.scene != nullptr) {
        for (Tree& t : d.trees) freeTree(d, t);
        framefx_lapis_scene_free(d.scene);
        d.scene = nullptr;
    }
    for (Buffer* b : {&d.ring, &d.treeArena, &d.pointArena, &d.uvArena, &d.topBuffer, &d.scratch}) {
        freeBuffer(d, *b);
    }
    if (d.sampler != VK_NULL_HANDLE) d.vk.DestroySampler(d.device, d.sampler, nullptr);
    for (uint32_t i = 0; i < kSlots; i++) {
        freeBuffer(d, d.instances[i]);
        freeBuffer(d, d.rows[i]);
        freeBuffer(d, d.moved[i]);
        freeBuffer(d, d.constants[i]);
        freeBuffer(d, d.stats[i]);
        freeBuffer(d, d.surfaces[i]);
    }
}

void barrier(Device& d, VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags srcAccess,
             VkPipelineStageFlags dst, VkAccessFlags dstAccess) {
    VkMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    d.vk.CmdPipelineBarrier(cb, src, dst, 0, 1, &b, 0, nullptr, 0, nullptr);
}

}  // namespace

void onDevice(const framefx_vk_instance& in, VkPhysicalDevice physical, VkDevice device,
              bool granted) {
    if (!granted || in.get_device_proc_addr == nullptr || in.get_memory_properties2 == nullptr) {
        return;
    }
    auto d = std::make_shared<Device>();
    d->device = device;
    if (!d->vk.load(device, in.get_device_proc_addr)) {
        ALOGW("rays: the driver lacks an entry point capture needs; no rays");
        return;
    }
    VkPhysicalDeviceMemoryProperties2 memory = {};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    in.get_memory_properties2(physical, &memory);
    d->memory = memory.memoryProperties;
    if (in.get_properties2 != nullptr) {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR structures = {};
        structures.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 properties = {};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &structures;
        in.get_properties2(physical, &properties);
        d->scratchAlign = std::max<VkDeviceSize>(
                256, structures.minAccelerationStructureScratchOffsetAlignment);
    }
    d->ringBytes = alignUp(kFramesInRing * 3ull * maxTriangles() * kStride, kChunk);
    if (!makeBuffer(*d, d->ringBytes,
                    VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    false, &d->ring)) {
        ALOGW("rays: no memory for the capture ring");
        freeBuffer(*d, d->ring);
        return;
    }
    std::lock_guard lock(gLock);
    gDevices[device] = d;
    ALOGI("rays: device %p can capture its scene", device);
}

void onDestroyDevice(VkDevice device) {
    std::shared_ptr<Device> d;
    {
        std::lock_guard lock(gLock);
        const auto it = gDevices.find(device);
        if (it == gDevices.end()) return;
        d = it->second;
        gDevices.erase(it);
        for (auto q = gQueues.begin(); q != gQueues.end();) {
            q = q->second == device ? gQueues.erase(q) : std::next(q);
        }
        if (gActive.load() == d.get()) gActive.store(nullptr, std::memory_order_release);
    }
    freeDevice(*d);
    // Recordings and the hold keep no device objects: the ring is gone with freeDevice.
}

void onQueue(VkDevice device, VkQueue queue) {
    if (queue == VK_NULL_HANDLE) return;
    std::lock_guard lock(gLock);
    gQueues[queue] = device;
}

PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next) {
    if (next == nullptr) return nullptr;
#define WRAP(slot, fn, ...)                                                        \
    for (const char* n : {__VA_ARGS__}) {                                          \
        if (strcmp(name, n) == 0) return wrap(&slot, next, &fn);                   \
    }
    WRAP(gCreateModule, CreateShaderModule, "vkCreateShaderModule")
    WRAP(gDestroyModule, DestroyShaderModule, "vkDestroyShaderModule")
    WRAP(gCreateGraphics, CreateGraphicsPipelines, "vkCreateGraphicsPipelines")
    WRAP(gDestroyPipeline, DestroyPipeline, "vkDestroyPipeline")
    WRAP(gBindPipeline, CmdBindPipeline, "vkCmdBindPipeline")
    WRAP(gDraw, CmdDraw, "vkCmdDraw")
    WRAP(gDrawIndexed, CmdDrawIndexed, "vkCmdDrawIndexed")
    WRAP(gDrawIndirect, CmdDrawIndirect, "vkCmdDrawIndirect")
    WRAP(gDrawIndexedIndirect, CmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")
    WRAP(gDrawIndirectCount, CmdDrawIndirectCount, "vkCmdDrawIndirectCount",
         "vkCmdDrawIndirectCountKHR")
    WRAP(gDrawIndexedIndirectCount, CmdDrawIndexedIndirectCount, "vkCmdDrawIndexedIndirectCount",
         "vkCmdDrawIndexedIndirectCountKHR")
    WRAP(gEndRenderPass, CmdEndRenderPass, "vkCmdEndRenderPass")
    WRAP(gEndRenderPass2, CmdEndRenderPass2, "vkCmdEndRenderPass2", "vkCmdEndRenderPass2KHR")
    WRAP(gNextSubpass, CmdNextSubpass, "vkCmdNextSubpass")
    WRAP(gNextSubpass2, CmdNextSubpass2, "vkCmdNextSubpass2", "vkCmdNextSubpass2KHR")
    WRAP(gEndRendering, CmdEndRendering, "vkCmdEndRendering", "vkCmdEndRenderingKHR")
    WRAP(gExecuteCommands, CmdExecuteCommands, "vkCmdExecuteCommands")
    WRAP(gBeginCommandBuffer, BeginCommandBuffer, "vkBeginCommandBuffer")
    WRAP(gEndCommandBuffer, EndCommandBuffer, "vkEndCommandBuffer")
    WRAP(gQueueSubmit, QueueSubmit, "vkQueueSubmit")
    WRAP(gQueueSubmit2, QueueSubmit2, "vkQueueSubmit2", "vkQueueSubmit2KHR")
#undef WRAP
    return nullptr;
}

void onPresent(VkQueue queue) {
    Device* d = nullptr;
    {
        std::lock_guard lock(gLock);
        const auto q = gQueues.find(queue);
        if (q == gQueues.end()) return;
        const auto it = gDevices.find(q->second);
        if (it == gDevices.end()) return;
        d = it->second.get();
        // The device that presents is the one whose draws are the game's frames.
        if (gActive.load(std::memory_order_relaxed) != d) {
            gActiveHold = it->second;
            gActive.store(d, std::memory_order_release);
            ALOGI("rays: capturing the scene of device %p", d->device);
        }
    }
    std::lock_guard lock(d->lock);
    d->frame.swap(d->pending);
    d->pending.clear();
    d->frameVertices.store(0, std::memory_order_relaxed);
}

// A tree's first vertex in the UV arena: its points' first, as the arena's whole two-float slots
// only (a tree's points start on any word, its coordinates' slots never overlap another's).
uint32_t uvFirstOf(const Tree& t) {
    return static_cast<uint32_t>((t.pointOffset + 11) / 12);
}

// The instance's surface (rays.h Scene::surfaces): solid, or its test over a texture of the
// frame's, which it adds; false when the frame's textures are full.
bool surfaceOf(const Tree& t, std::unordered_map<VkImageView, uint32_t>& textures,
               Scene* out, uint32_t surface[8]) {
    surface[0] = uvFirstOf(t);
    surface[1] = kSolid;
    surface[2] = t.test;
    surface[3] = 0;
    memcpy(&surface[4], t.cut, 8);
    surface[6] = surface[7] = 0;
    if (t.texture == VK_NULL_HANDLE) return true;
    const auto it = textures.find(t.texture);
    if (it != textures.end()) {
        surface[1] = it->second;
        return true;
    }
    // The game streams its textures: a view it has since destroyed is never bound.
    if (textures.size() >= kTextures || !lapis::viewAlive(t.texture)) return false;
    surface[1] = static_cast<uint32_t>(textures.size());
    out->textures[surface[1]] = t.texture;
    textures.emplace(t.texture, surface[1]);
    return true;
}

bool build(VkDevice device, VkCommandBuffer cb, const float inverse[16], const double eye[3],
           Scene* out) {
    Device* d = gActive.load(std::memory_order_acquire);
    if (d == nullptr || d->device != device || !make(*d)) return false;
    std::vector<DrawRecord> draws;
    {
        std::lock_guard lock(d->lock);
        draws = d->frame;
    }
    if (draws.empty()) return false;
    const uint32_t slot = static_cast<uint32_t>(d->frameCount++ % kSlots);
    d->slot = slot;
    // The motion positions.comp saw when this slot was last used, frames ago and done.
    auto* moved = static_cast<uint32_t*>(d->moved[slot].mapped);
    auto& entries = d->rowEntries[slot];
    for (size_t i = 0; i < entries.size(); i++) {
        d->compared++;
        if (moved[i] != 0) (moved[i] > 1 ? d->movedFar : d->movedNear)++;
        if (moved[i] == 0) continue;
        framefx_lapis_scene_moved(d->scene, entries[i], moved[i] > 1 ? 1 : 0);
        moved[i] = 0;
    }
    entries.clear();
    framefx_lapis_scene_begin(d->scene);
    evictTrees(*d, kEvictAge);

    auto* rows = static_cast<DrawRow*>(d->rows[slot].mapped);
    auto* instances = static_cast<VkAccelerationStructureInstanceKHR*>(d->instances[slot].mapped);
    auto* surfaces = static_cast<uint32_t(*)[8]>(d->surfaces[slot].mapped);
    std::unordered_map<VkImageView, uint32_t> textures;
    out->textureCount = 0;
    const size_t most = std::min<size_t>(draws.size(), kMaxInstances);
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    geometries.reserve(most);
    infos.reserve(most);
    ranges.reserve(most);
    const VkDeviceSize align = d->scratchAlign;
    const VkDeviceAddress scratchBase = alignUp(d->scratch.address, align);
    const VkDeviceSize scratchEnd = d->scratch.size - (scratchBase - d->scratch.address);
    VkDeviceSize scratchAt = d->topScratch;
    bool pressed = false;
    uint32_t n = 0;
    uint32_t vertices = 0;
    uint32_t triangleCount = 0;
    for (const DrawRecord& r : draws) {
        if (n == kMaxInstances) break;
        if (r.vertices < 3) continue;
        uint32_t entry = 0;
        uint8_t action = framefx_lapis_scene_place(d->scene, r.key, r.vertices, &entry);
        if (entry >= d->trees.size()) d->trees.resize(entry + 1);
        Tree& t = d->trees[entry];
        const uint32_t tris = r.vertices / 3;
        const Sizes sz = sizesFor(*d, tris);
        if (action == kBuild && t.vertices != r.vertices) {
            freeTree(*d, t);
            if (!makeTree(*d, t, r.vertices, sz, &pressed)) {
                framefx_lapis_scene_drop(d->scene, entry);
                d->skipped++;
                continue;
            }
        }
        if (t.structure == VK_NULL_HANDLE || t.address == 0) {
            // No tree to trace: built again when it is next drawn.
            freeTree(*d, t);
            framefx_lapis_scene_drop(d->scene, entry);
            d->skipped++;
            continue;
        }
        if (action != kKeep) {
            const VkDeviceSize need = action == kBuild ? sz.build : sz.update;
            const VkDeviceSize at = alignUp(scratchAt, align);
            if (at + need > scratchEnd) {
                // Past this frame's scratch: a new tree waits for a later frame, a moving one
                // stays where it was for this one.
                if (action == kBuild) {
                    freeTree(*d, t);
                    framefx_lapis_scene_drop(d->scene, entry);
                    d->skipped++;
                    continue;
                }
                action = kKeep;
            } else {
                scratchAt = at + need;
                geometries.push_back(
                        triangles(d->pointArena.address + t.pointOffset, r.vertices - 1));
                VkAccelerationStructureBuildGeometryInfoKHR info = {};
                info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                info.flags = kBottomFlags;
                info.mode = action == kBuild ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                                             : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                info.srcAccelerationStructure = action == kBuild ? VK_NULL_HANDLE : t.structure;
                info.dstAccelerationStructure = t.structure;
                info.geometryCount = 1;
                info.pGeometries = &geometries.back();
                info.scratchData.deviceAddress = scratchBase + at;
                infos.push_back(info);
                VkAccelerationStructureBuildRangeInfoKHR range = {};
                range.primitiveCount = tris;
                ranges.push_back(range);
            }
        }
        // Points kept from an earlier frame sit around the camera of that frame.
        float offset[3];
        for (int k = 0; k < 3; k++) offset[k] = static_cast<float>(t.eye[k] - eye[k]);
        DrawRow& row = rows[n];
        row.first = vertices;
        row.count = r.vertices;
        row.captured = static_cast<uint32_t>(r.offset / kStride);
        row.point = static_cast<uint32_t>(t.pointOffset / 4);
        std::copy(offset, offset + 3, row.offset);
        // A new tree's points are written; a refitted one's compared, then written; a kept one's
        // only compared.
        row.write = action == kBuild ? 2.0f : action == kRefit ? 1.0f : 0.0f;
        const bool alpha = r.texture != VK_NULL_HANDLE;
        row.uv = alpha ? uvFirstOf(t) : kNoVarying;
        row.step = r.step;
        row.component = (r.test >> 3) & 3;
        row.pad = 0;
        t.texture = r.texture;
        t.test = r.test;
        std::copy(r.cut, r.cut + 2, t.cut);
        VkAccelerationStructureInstanceKHR& instance = instances[n];
        instance = {};
        instance.transform.matrix[0][0] = instance.transform.matrix[1][1] =
                instance.transform.matrix[2][2] = 1.0f;
        if (action == kKeep) {
            instance.transform.matrix[0][3] = offset[0];
            instance.transform.matrix[1][3] = offset[1];
            instance.transform.matrix[2][3] = offset[2];
        } else {
            std::copy(eye, eye + 3, t.eye);
        }
        // Bit 22 the tree changed this frame, 23 a draw out of sight (rays.comp's debug views).
        instance.instanceCustomIndex = (entry & kEntryBits) | (action != kKeep ? kChangedBit : 0);
        // Past the frame's textures an alpha-tested draw is drawn into no ray.
        const bool placed = surfaceOf(t, textures, out, surfaces[n]);
        instance.mask = placed ? 0xff : 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
                (alpha ? VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR
                       : VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR);
        instance.accelerationStructureReference = t.address;
        if (alpha) (placed ? d->alphaDraws : d->overflowed)++;
        entries.push_back(entry);
        (action == kBuild ? d->built : action == kRefit ? d->refitted : d->kept)++;
        vertices += r.vertices;
        triangleCount += tris;
        n++;
    }
    if (n == 0) return false;
    // Still draws the game did not draw this frame stay, where they were.
    uint32_t instanceCount = n;
    {
        std::vector<uint32_t> kept(kMaxInstances - n);
        kept.resize(framefx_lapis_scene_retained(d->scene, kRetainAge, kept.data(), kept.size()));
        for (uint32_t entry : kept) {
            if (entry >= d->trees.size()) continue;
            const Tree& t = d->trees[entry];
            if (t.structure == VK_NULL_HANDLE || t.address == 0) continue;
            VkAccelerationStructureInstanceKHR& instance = instances[instanceCount++];
            instance = {};
            instance.transform.matrix[0][0] = instance.transform.matrix[1][1] =
                    instance.transform.matrix[2][2] = 1.0f;
            for (int k = 0; k < 3; k++) {
                instance.transform.matrix[k][3] = static_cast<float>(t.eye[k] - eye[k]);
            }
            instance.instanceCustomIndex = (entry & kEntryBits) | kRetainedBit;
            const bool placed = surfaceOf(t, textures, out, surfaces[instanceCount - 1]);
            instance.mask = placed ? 0xff : 0;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
                    (t.texture != VK_NULL_HANDLE ? VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR
                                                 : VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR);
            instance.accelerationStructureReference = t.address;
        }
        d->retained += instanceCount - n;
    }

    // The game's transform feedback, and last frame's rays and builds, before this frame's points.
    barrier(*d, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                    VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    d->vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->convert);
    d->vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipelineLayout, 0, 1,
                                &d->sets[slot], 0, nullptr);
    struct {
        float inverse[16];
        uint32_t count[4];
    } push;
    memcpy(push.inverse, inverse, sizeof(push.inverse));
    push.count[0] = vertices;
    push.count[1] = n;
    push.count[2] = push.count[3] = 0;
    d->vk.CmdPushConstants(cb, d->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
    d->vk.CmdDispatch(cb, (vertices + 63) / 64, 1, 1);
    barrier(*d, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_SHADER_READ_BIT);

    if (!infos.empty()) {
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers(ranges.size());
        for (size_t i = 0; i < ranges.size(); i++) rangePointers[i] = &ranges[i];
        d->vk.CmdBuildAccelerationStructuresKHR(cb, static_cast<uint32_t>(infos.size()),
                                                infos.data(), rangePointers.data());
        barrier(*d, cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }

    VkAccelerationStructureGeometryKHR top = instancesOf(d->instances[slot].address);
    VkAccelerationStructureBuildGeometryInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info.flags = kTopFlags;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.dstAccelerationStructure = d->top;
    info.geometryCount = 1;
    info.pGeometries = &top;
    info.scratchData.deviceAddress = scratchBase;
    VkAccelerationStructureBuildRangeInfoKHR range = {};
    range.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* ranged = &range;
    d->vk.CmdBuildAccelerationStructuresKHR(cb, 1, &info, &ranged);
    barrier(*d, cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    if (d->frameCount % 600 == 1) {
        ALOGI("rays: discarding fragment shaders: %u alpha tests read; not: %u many discards, %u "
              "no condition, %u not a texel, %u not a varying, %u no binding, %u malformed",
              gAlphaReads[0].load(), gAlphaReads[3].load(), gAlphaReads[4].load(),
              gAlphaReads[5].load(), gAlphaReads[6].load(), gAlphaReads[7].load(),
              gAlphaReads[1].load());
        ALOGI("rays: %u triangles in %u draws, %u more out of sight; trees since the last report: "
              "%u built, %u refitted, %u kept, %u waiting, %u out of sight; arenas %.0f/%.0f MB "
              "trees, %.0f/%.0f MB points; %u capturing pipelines (%u alpha-tested), %u "
              "alpha-tested and %u others left out, %u of %u vertex shaders refused; alpha-tested "
              "draws %u traced, %u past the textures, %u with no texture known, %u whole, %u "
              "unknown; %zu textures",
              triangleCount, n, instanceCount - n, d->built, d->refitted, d->kept, d->skipped,
              d->retained,
              framefx_lapis_scene_in_use(d->scene, kTrees) / 1048576.0,
              d->treeArena.size / 1048576.0,
              framefx_lapis_scene_in_use(d->scene, kPoints) / 1048576.0,
              d->pointArena.size / 1048576.0, gCapturing.load(), gAlphaCapturing.load(),
              gAlphaTested.load(), gOther.load(), gRefusedModules.load(), gVertexModules.load(),
              d->alphaDraws, d->overflowed, d->untextured.load(), d->whole.load(),
              d->unknown.load(), textures.size());
        ALOGI("rays: of %u draws compared, %u moved and %u moved far; %llu out of sight left out "
              "as moving",
              d->compared, d->movedNear, d->movedFar,
              static_cast<unsigned long long>(
                      framefx_lapis_scene_take_moving_out_of_sight(d->scene)));
        d->compared = d->movedNear = d->movedFar = 0;
        d->built = d->refitted = d->kept = d->skipped = d->retained = 0;
        d->alphaDraws = d->overflowed = 0;
        d->untextured = 0;
        d->whole = 0;
        d->unknown = 0;
    }
    out->top = d->top;
    out->triangles = triangleCount;
    out->refitted = !infos.empty();
    out->surfaces = {d->surfaces[slot].buffer, 0, VkDeviceSize{instanceCount} * kSurfaceBytes};
    out->uvs = {d->uvArena.buffer, 0, VK_WHOLE_SIZE};
    out->sampler = d->sampler;
    out->textureCount = static_cast<uint32_t>(textures.size());
    return true;
}

bool frameConstants(VkDevice device, const void* data, size_t size, VkDescriptorBufferInfo* out) {
    Device* d = gActive.load(std::memory_order_acquire);
    if (d == nullptr || d->device != device || !d->made || size > kConstantBytes) return false;
    Buffer& b = d->constants[d->slot];
    memcpy(b.mapped, data, size);
    *out = {b.buffer, 0, size};
    return true;
}

bool frameStats(VkDevice device, size_t size, const void* label, size_t labelSize, void* earlier,
                void* earlierLabel, VkDescriptorBufferInfo* out) {
    Device* d = gActive.load(std::memory_order_acquire);
    if (d == nullptr || d->device != device || !d->made || size > kStatBytes ||
        labelSize > kLabelBytes) {
        return false;
    }
    Buffer& b = d->stats[d->slot];
    memcpy(earlier, b.mapped, size);
    memset(b.mapped, 0, size);
    memcpy(earlierLabel, d->statLabels[d->slot], labelSize);
    memcpy(d->statLabels[d->slot], label, labelSize);
    *out = {b.buffer, 0, size};
    return true;
}

Stats stats() {
    return {gVertexModules.load(), gRefusedModules.load(), gCapturing.load(), gAlphaTested.load(),
            gOther.load()};
}

}  // namespace framefx::rays
