/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "video/afme_engine.h"

#include <log/log.h>

#include <mutex>

namespace framefx::video {
namespace {

constexpr float kHalfway = -0.5f;  // AFME counts phase from the current frame

}  // namespace

AfmeEngine::~AfmeEngine() { freeSized(); }

void AfmeEngine::freeSized() {
    if (engine_ != nullptr) api_->destroy(engine_);
    engine_ = nullptr;
    if (depthView_ != VK_NULL_HANDLE) g_.DestroyImageView(g_.device, depthView_, nullptr);
    depthView_ = VK_NULL_HANDLE;
    FreeImage(g_, depth_);
    FreeImage(g_, motion_);
    FreeImage(g_, output_);
    primed_ = false;
    width_ = 0;
    height_ = 0;
}

bool AfmeEngine::resize(uint32_t width, uint32_t height) {
    if (sized(width, height)) return true;
    freeSized();
    if (!g_.opticalFlow) return false;
    api_ = luwes::afme::Load();
    if (api_ == nullptr) return false;
    luwes::afme::Context context = {g_.instance, g_.device, g_.physical};
    if (!api_->isSupported(&context)) {
        ALOGW("video: AFME does not support this GPU or driver");
        return false;
    }
    const VkImageUsageFlags fill = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // The view MakeImage takes of a depth format is a colour one: this image needs its own.
    VkImageCreateInfo image = {};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = VK_FORMAT_D24_UNORM_S8_UINT;
    image.extent = {width, height, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = fill | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (g_.CreateImage(g_.device, &image, nullptr, &depth_.image) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements;
    g_.GetImageMemoryRequirements(g_.device, depth_.image, &requirements);
    VkMemoryAllocateInfo allocate = {};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex =
            MemoryType(g_, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocate.memoryTypeIndex == UINT32_MAX ||
        g_.AllocateMemory(g_.device, &allocate, nullptr, &depth_.memory) != VK_SUCCESS ||
        g_.BindImageMemory(g_.device, depth_.image, depth_.memory, 0) != VK_SUCCESS) {
        return false;
    }
    VkImageViewCreateInfo view = {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = depth_.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_D24_UNORM_S8_UINT;
    view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (g_.CreateImageView(g_.device, &view, nullptr, &depthView_) != VK_SUCCESS) return false;
    if (!MakeImage(g_, &motion_, VK_FORMAT_R32G32_SFLOAT, width, height,
                   fill | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
        !MakeImage(g_, &output_, VK_FORMAT_R8G8B8A8_UNORM, width, height,
                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_STORAGE_BIT)) {
        return false;
    }
    luwes::afme::InitInfo init = {width, height, width, height, luwes::afme::kFlags, kHalfway};
    engine_ = api_->init(&context, &init);
    if (engine_ == nullptr) {
        ALOGW("video: AFME refused %ux%u", width, height);
        return false;
    }
    width_ = width;
    height_ = height;
    return true;
}

bool AfmeEngine::record(VkCommandBuffer cb, VkImageView previous, VkImageView current) {
    if (engine_ == nullptr) return false;
    if (!primed_) {
        const VkImageLayout dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        const VkClearDepthStencilValue far = {0.0f, 0};
        const VkImageSubresourceRange depth = {
                VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        VkImageMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = dst;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = depth_.image;
        b.subresourceRange = depth;
        g_.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &b);
        g_.CmdClearDepthStencilImage(cb, depth_.image, dst, &far, 1, &depth);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = dst;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        g_.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        Barrier(g_, cb, motion_.image, VK_IMAGE_LAYOUT_UNDEFINED, dst,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearColorValue zero = {};
        const VkImageSubresourceRange color = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g_.CmdClearColorImage(cb, motion_.image, dst, &zero, 1, &color);
        Barrier(g_, cb, motion_.image, dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        Barrier(g_, cb, output_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        primed_ = true;
    } else {
        Barrier(g_, cb, output_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    }
    luwes::afme::FEInfo in = {};
    in.src1Color = previous;
    in.src2Color = current;
    in.src1Depth = depthView_;
    in.src2Depth = depthView_;
    in.src1Motion = motion_.view;
    in.src2Motion = motion_.view;
    in.output = output_.view;
    for (int i = 0; i < 4; i++) in.matrix[i * 5] = 1.0f;
    in.cmd = cb;
    bool ok;
    {
        // The library can submit from here, to the queue it finds for itself: this one.
        std::lock_guard lock(g_.queueLock);
        ok = api_->estimate(engine_, &in);
    }
    ALOGW_IF(!ok, "video: AFME estimate failed");
    Barrier(g_, cb, output_.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    return ok;
}

}  // namespace framefx::video
