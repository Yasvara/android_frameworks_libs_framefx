/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "video/motion.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "luwes/shaders.h"

namespace framefx::video {
namespace {

constexpr uint32_t kAlign = 128;   // the coarsest level holds whole blocks
constexpr int kSearchRadius = 12;  // coarsest level, its pixels: +-192 frame pixels
constexpr float kSmoothness = 0.004f;
constexpr float kFieldScale = 4.0f;  // the finest level's pixels to the frame's
constexpr int kInterpolate = 1;      // output.frag's mode

// motion.comp's passes.
constexpr int kSearch = 0;
constexpr int kCandidates = 1;
constexpr int kPropagate = 2;
constexpr int kRefine = 3;
constexpr int kMedian = 4;
constexpr int kGlobal = 5;

// A block's mean absolute luma difference from its best match at 1/16 of the frame, past which it
// has no match, and the share of such blocks that makes a cut. A pan or a crowd keeps most blocks
// well under it even where the vectors are wrong.
constexpr float kNoMatch = 0.07f;
constexpr float kCutShare = 0.7f;

float halfToFloat(uint16_t h) {
    const int exponent = (h >> 10) & 0x1f;
    const float mantissa = static_cast<float>(h & 0x3ff) / 1024.0f;
    const float value = exponent == 0 ? std::ldexp(mantissa, -14)
                                      : std::ldexp(1.0f + mantissa, exponent - 15);
    return (h & 0x8000) != 0 ? -value : value;
}

constexpr VkFormat kFieldFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

}  // namespace

Motion::~Motion() {
    freeSized();
    for (VkPipeline p : {pyramid_, motion_, output_}) {
        if (p != VK_NULL_HANDLE) g_.DestroyPipeline(g_.device, p, nullptr);
    }
    for (VkPipelineLayout l : {pyramidPipe_, motionPipe_, outputPipe_}) {
        if (l != VK_NULL_HANDLE) g_.DestroyPipelineLayout(g_.device, l, nullptr);
    }
    for (VkDescriptorSetLayout l : {pyramidLayout_, motionLayout_, outputLayout_}) {
        if (l != VK_NULL_HANDLE) g_.DestroyDescriptorSetLayout(g_.device, l, nullptr);
    }
    for (VkShaderModule m : {vertex_, pyramidModule_, motionModule_, outputModule_}) {
        if (m != VK_NULL_HANDLE) g_.DestroyShaderModule(g_.device, m, nullptr);
    }
    for (VkSampler s : {linear_, nearest_}) {
        if (s != VK_NULL_HANDLE) g_.DestroySampler(g_.device, s, nullptr);
    }
    if (r8Pass_ != VK_NULL_HANDLE) g_.DestroyRenderPass(g_.device, r8Pass_, nullptr);
}

bool Motion::warm(VkRenderPass pass) {
    namespace shaders = luwes::shaders;
    pass_ = pass;
    linear_ = MakeSampler(g_, VK_FILTER_LINEAR);
    nearest_ = MakeSampler(g_, VK_FILTER_NEAREST);
    r8Pass_ = MakePass(g_, VK_FORMAT_R8_UNORM);
    vertex_ = MakeModule(g_, shaders::kFullscreenVertSpirv.code, shaders::kFullscreenVertSpirv.size);
    pyramidModule_ = MakeModule(g_, shaders::kPyramidFragSpirv.code, shaders::kPyramidFragSpirv.size);
    motionModule_ = MakeModule(g_, shaders::kMotionCompSpirv.code, shaders::kMotionCompSpirv.size);
    outputModule_ = MakeModule(g_, shaders::kOutputFragSpirv.code, shaders::kOutputFragSpirv.size);
    pyramidLayout_ = MakeSetLayout(g_, 1, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    motionLayout_ = MakeSetLayout(g_, 5, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    outputLayout_ = MakeSetLayout(g_, 5, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    pyramidPipe_ = MakePipelineLayout(g_, pyramidLayout_);
    motionPipe_ = MakePipelineLayout(g_, motionLayout_);
    outputPipe_ = MakePipelineLayout(g_, outputLayout_);
    if (linear_ == VK_NULL_HANDLE || nearest_ == VK_NULL_HANDLE || r8Pass_ == VK_NULL_HANDLE ||
        vertex_ == VK_NULL_HANDLE || pyramidModule_ == VK_NULL_HANDLE ||
        motionModule_ == VK_NULL_HANDLE || outputModule_ == VK_NULL_HANDLE ||
        pyramidPipe_ == VK_NULL_HANDLE || motionPipe_ == VK_NULL_HANDLE ||
        outputPipe_ == VK_NULL_HANDLE) {
        return false;
    }
    pyramid_ = MakeFullscreen(g_, vertex_, pyramidModule_, pyramidPipe_, r8Pass_);
    output_ = MakeFullscreen(g_, vertex_, outputModule_, outputPipe_, pass_);
    VkComputePipelineCreateInfo compute = {};
    compute.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute.stage.module = motionModule_;
    compute.stage.pName = "main";
    compute.layout = motionPipe_;
    g_.CreateComputePipelines(g_.device, VK_NULL_HANDLE, 1, &compute, nullptr, &motion_);
    return pyramid_ != VK_NULL_HANDLE && output_ != VK_NULL_HANDLE && motion_ != VK_NULL_HANDLE;
}

void Motion::freeSized() {
    for (auto& parity : lumaTarget_) {
        for (VkFramebuffer& f : parity) {
            if (f != VK_NULL_HANDLE) g_.DestroyFramebuffer(g_.device, f, nullptr);
            f = VK_NULL_HANDLE;
        }
    }
    for (auto& parity : luma_) {
        for (Image& image : parity) FreeImage(g_, image);
    }
    for (auto& level : field_) {
        for (Image& image : level) FreeImage(g_, image);
    }
    FreeImage(g_, global_);
    if (pool_ != VK_NULL_HANDLE) g_.DestroyDescriptorPool(g_.device, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    if (errorsMap_ != nullptr) g_.UnmapMemory(g_.device, errorsMemory_);
    errorsMap_ = nullptr;
    if (errors_ != VK_NULL_HANDLE) g_.DestroyBuffer(g_.device, errors_, nullptr);
    errors_ = VK_NULL_HANDLE;
    if (errorsMemory_ != VK_NULL_HANDLE) g_.FreeMemory(g_.device, errorsMemory_, nullptr);
    errorsMemory_ = VK_NULL_HANDLE;
    primed_ = false;
    searched_ = false;
}

bool Motion::resize(uint32_t width, uint32_t height, VkImageView source, const VkImageView pair[2],
                    uint32_t pairWidth, uint32_t pairHeight) {
    freeSized();
    width_ = width;
    height_ = height;
    pairWidth_ = pairWidth;
    pairHeight_ = pairHeight;
    alignedWidth_ = (width + kAlign - 1) / kAlign * kAlign;
    alignedHeight_ = (height + kAlign - 1) / kAlign * kAlign;
    const VkImageUsageFlags r8 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags vectors = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    bool ok = true;
    for (int p = 0; p < 2 && ok; p++) {
        for (int k = 0; k < kLevels && ok; k++) {
            ok = MakeImage(g_, &luma_[p][k], VK_FORMAT_R8_UNORM, levelWidth(k), levelHeight(k), r8);
            if (ok) {
                lumaTarget_[p][k] = MakeFramebuffer(g_, r8Pass_, luma_[p][k].view, levelWidth(k),
                                                    levelHeight(k));
                ok = lumaTarget_[p][k] != VK_NULL_HANDLE;
            }
        }
    }
    for (int k = kFinest; k < kLevels && ok; k++) {
        for (int j = 0; j < 2 && ok; j++) {
            ok = MakeImage(g_, &field_[k][j], kFieldFormat, levelWidth(k) / 8, levelHeight(k) / 8,
                           vectors);
        }
    }
    ok = ok && MakeImage(g_, &global_, kFieldFormat, 1, 1, vectors);
    if (!ok) return false;

    const uint32_t sets = 2 * (kLevels + kMotionSets + 1);
    const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             2 * (kLevels + 5 * kMotionSets + 5)},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kMotionSets},
    };
    VkDescriptorPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = sets;
    pool.poolSizeCount = 2;
    pool.pPoolSizes = sizes;
    if (g_.CreateDescriptorPool(g_.device, &pool, nullptr, &pool_) != VK_SUCCESS) return false;

    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImageLayout general = VK_IMAGE_LAYOUT_GENERAL;
    const auto field = [&](int k, int j) { return Binding{field_[k][j].view, general, nearest_}; };
    const Binding global = {global_.view, general, nearest_};
    const int top = kLevels - 1;
    for (int p = 0; p < 2; p++) {
        const int q = 1 - p;
        for (int k = 0; k < kLevels; k++) {
            pyramidSet_[p][k] = AllocateSet(g_, pool_, pyramidLayout_);
            const Binding in = k == 0 ? Binding{source, read, linear_}
                                      : Binding{luma_[p][k - 1].view, read, linear_};
            WriteSet(g_, pyramidSet_[p][k], {in}, nullptr);
        }
        int n = 0;
        const auto motion = [&](int pass, int k, Binding parent, Binding input, Binding globalIn,
                                Binding out) {
            const VkDescriptorSet set = AllocateSet(g_, pool_, motionLayout_);
            WriteSet(g_, set,
                     {Binding{luma_[p][k].view, read, nearest_},
                      Binding{luma_[q][k].view, read, nearest_}, parent, input, globalIn},
                     &out);
            dispatch_[p][n++] = {set, pass, k};
        };
        // Coarsest level: search into A, median into B (its result), dominant vector from B.
        motion(kSearch, top, field(top, 1), field(top, 1), global, field(top, 0));
        motion(kMedian, top, field(top, 0), field(top, 0), global, field(top, 1));
        motion(kGlobal, top, field(top, 0), field(top, 1), field(top, 0), global);
        // Finer levels: candidates into A, propagate into B, refine twice, median into A (the
        // level's result).
        for (int k = top - 1; k >= kFinest; k--) {
            const Binding parent = k + 1 == top ? field(top, 1) : field(k + 1, 0);
            motion(kCandidates, k, parent, field(k, 1), global, field(k, 0));
            motion(kPropagate, k, parent, field(k, 0), global, field(k, 1));
            motion(kRefine, k, parent, field(k, 1), global, field(k, 0));
            motion(kRefine, k, parent, field(k, 0), global, field(k, 1));
            motion(kMedian, k, parent, field(k, 1), global, field(k, 0));
        }
        const Binding current = {pair[p], read, linear_};
        outputSet_[p] = AllocateSet(g_, pool_, outputLayout_);
        WriteSet(g_, outputSet_[p],
                 {current, Binding{pair[q], read, linear_},
                  Binding{field_[kFinest][0].view, general, linear_}, current, current},
                 nullptr);
    }

    const VkDeviceSize bytes = (levelWidth(top) / 8) * (levelHeight(top) / 8) * 8;
    VkBufferCreateInfo buffer = {};
    buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer.size = bytes;
    buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (g_.CreateBuffer(g_.device, &buffer, nullptr, &errors_) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements;
    g_.GetBufferMemoryRequirements(g_.device, errors_, &requirements);
    VkMemoryAllocateInfo allocate = {};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex =
            MemoryType(g_, requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    return allocate.memoryTypeIndex != UINT32_MAX &&
            g_.AllocateMemory(g_.device, &allocate, nullptr, &errorsMemory_) == VK_SUCCESS &&
            g_.BindBufferMemory(g_.device, errors_, errorsMemory_, 0) == VK_SUCCESS &&
            g_.MapMemory(g_.device, errorsMemory_, 0, VK_WHOLE_SIZE, 0, &errorsMap_) == VK_SUCCESS;
}

void Motion::record(VkCommandBuffer cb, int cur, bool search) {
    if (!primed_) {
        for (int k = kFinest; k < kLevels; k++) {
            for (Image& image : field_[k]) {
                Barrier(g_, cb, image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT);
            }
        }
        Barrier(g_, cb, global_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT);
        primed_ = true;
    }
    for (int k = 0; k < kLevels; k++) {
        const float sw = static_cast<float>(k == 0 ? width_ : levelWidth(k - 1));
        const float sh = static_cast<float>(k == 0 ? height_ : levelHeight(k - 1));
        const float params[16] = {1.0f / sw, 1.0f / sh, k == 0 ? 1.0f : 0.0f};
        Fullscreen(g_, cb, r8Pass_, lumaTarget_[cur][k], levelWidth(k), levelHeight(k), pyramid_,
                   pyramidPipe_, pyramidSet_[cur][k], params);
    }
    searched_ = search;
    if (!search) return;

    // Last frame's output pass still reads the field this frame rewrites.
    g_.CmdPipelineBarrier(cb,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
    g_.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, motion_);
    VkMemoryBarrier written = {};
    written.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    written.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    written.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    for (const Dispatch& m : dispatch_[cur]) {
        const float params[16] = {static_cast<float>(m.pass), static_cast<float>(kSearchRadius),
                                  kSmoothness, static_cast<float>(1 << (kLevels - 1 - m.level))};
        g_.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, motionPipe_, 0, 1, &m.set, 0,
                                 nullptr);
        g_.CmdPushConstants(cb, motionPipe_, VK_SHADER_STAGE_ALL, 0, 64, params);
        if (m.pass == kGlobal) {
            g_.CmdDispatch(cb, 1, 1, 1);
        } else {
            g_.CmdDispatch(cb, levelWidth(m.level) / 8, levelHeight(m.level) / 8, 1);
        }
        g_.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                              0, 1, &written, 0, nullptr, 0, nullptr);
    }
    // The coarsest level's result, for cut().
    VkMemoryBarrier copy = {};
    copy.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    g_.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 1, &copy, 0, nullptr, 0, nullptr);
    const int top = kLevels - 1;
    VkBufferImageCopy region = {};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {levelWidth(top) / 8, levelHeight(top) / 8, 1};
    g_.CmdCopyImageToBuffer(cb, field_[top][1].image, VK_IMAGE_LAYOUT_GENERAL, errors_, 1, &region);
}

bool Motion::cut() const {
    if (!searched_ || errorsMap_ == nullptr) return false;
    const int top = kLevels - 1;
    const uint32_t blocks = (levelWidth(top) / 8) * (levelHeight(top) / 8);
    const auto* texels = static_cast<const uint16_t*>(errorsMap_);
    uint32_t lost = 0;
    for (uint32_t i = 0; i < blocks; i++) {
        if (halfToFloat(texels[4 * i + 2]) > kNoMatch) lost++;
    }
    return static_cast<float>(lost) >= kCutShare * static_cast<float>(blocks);
}

void Motion::recordOutput(VkCommandBuffer cb, VkFramebuffer target, int cur, float phase) {
    // The field is in the decoded frame's pixels; the pair may be larger.
    const float scaleX = static_cast<float>(pairWidth_) / static_cast<float>(width_);
    const float scaleY = static_cast<float>(pairHeight_) / static_cast<float>(height_);
    const float params[16] = {
            static_cast<float>(pairWidth_),
            static_cast<float>(pairHeight_),
            static_cast<float>(alignedWidth_) * scaleX,
            static_cast<float>(alignedHeight_) * scaleY,
            phase,
            static_cast<float>(kInterpolate),
            kFieldScale * 0.5f * (scaleX + scaleY),
            0.0f,
    };
    Fullscreen(g_, cb, pass_, target, pairWidth_, pairHeight_, output_, outputPipe_, outputSet_[cur],
               params);
}

}  // namespace framefx::video
