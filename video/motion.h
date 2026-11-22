/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include "video/gpu.h"

// Luwes's motion on a video's frames: the luma pyramid and block search over the decoded frame, the
// output pass over the pair it is about to show. The search stays at the decoded size whatever the
// pair was upscaled to, so an upscale never costs it anything.
namespace framefx::video {

class Motion {
public:
    explicit Motion(const Gpu& gpu) : g_(gpu) {}
    ~Motion();

    // Pipelines. `pass` is the render pass the output pass draws in.
    bool warm(VkRenderPass pass);

    // The pyramid and field of a `width`x`height` frame read from `source`, and the pair of
    // `pairWidth`x`pairHeight` frames the output pass warps.
    bool resize(uint32_t width, uint32_t height, VkImageView source, const VkImageView pair[2],
                uint32_t pairWidth, uint32_t pairHeight);

    // The pyramid of the frame in `source` as frame `cur` of the pair, then, with `search`, its
    // motion against the other one.
    void record(VkCommandBuffer cb, int cur, bool search);

    // Whether what `record` just searched is a cut: most of the frame matches nothing in the frame
    // before. Valid once its submission is done.
    bool cut() const;

    // One frame at `phase` between the pair's other frame and `cur`, into `target`.
    void recordOutput(VkCommandBuffer cb, VkFramebuffer target, int cur, float phase);

private:
    static constexpr int kLevels = 4;  // 1/2 to 1/16 of the frame
    static constexpr int kFinest = 1;  // the search stops at 1/4: 32-pixel blocks
    static constexpr int kMotionSets = 3 + 5 * (kLevels - 1 - kFinest);

    struct Dispatch {
        VkDescriptorSet set = VK_NULL_HANDLE;
        int pass = 0;
        int level = 0;
    };

    uint32_t levelWidth(int level) const { return alignedWidth_ >> (level + 1); }
    uint32_t levelHeight(int level) const { return alignedHeight_ >> (level + 1); }
    void freeSized();

    const Gpu& g_;
    VkRenderPass pass_ = VK_NULL_HANDLE;
    VkRenderPass r8Pass_ = VK_NULL_HANDLE;
    VkSampler linear_ = VK_NULL_HANDLE;
    VkSampler nearest_ = VK_NULL_HANDLE;
    VkShaderModule vertex_ = VK_NULL_HANDLE;
    VkShaderModule pyramidModule_ = VK_NULL_HANDLE;
    VkShaderModule motionModule_ = VK_NULL_HANDLE;
    VkShaderModule outputModule_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout pyramidLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout motionLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout outputLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pyramidPipe_ = VK_NULL_HANDLE;
    VkPipelineLayout motionPipe_ = VK_NULL_HANDLE;
    VkPipelineLayout outputPipe_ = VK_NULL_HANDLE;
    VkPipeline pyramid_ = VK_NULL_HANDLE;
    VkPipeline motion_ = VK_NULL_HANDLE;
    VkPipeline output_ = VK_NULL_HANDLE;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t alignedWidth_ = 0;
    uint32_t alignedHeight_ = 0;
    uint32_t pairWidth_ = 0;
    uint32_t pairHeight_ = 0;
    Image luma_[2][kLevels];
    VkFramebuffer lumaTarget_[2][kLevels] = {};
    Image field_[kLevels][2];  // kFinest and coarser
    Image global_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet pyramidSet_[2][kLevels] = {};
    Dispatch dispatch_[2][kMotionSets];
    VkDescriptorSet outputSet_[2] = {};
    bool primed_ = false;
    bool searched_ = false;

    VkBuffer errors_ = VK_NULL_HANDLE;
    VkDeviceMemory errorsMemory_ = VK_NULL_HANDLE;
    void* errorsMap_ = nullptr;
};

}  // namespace framefx::video
