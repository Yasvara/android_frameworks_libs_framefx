/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include "luwes/afme.h"
#include "video/gpu.h"

// Qualcomm's AFME on a video's frames: one frame between every two, from the colour alone. A video
// has no depth or velocity to give it, so both stay at their still values.
namespace framefx::video {

class AfmeEngine {
public:
    explicit AfmeEngine(const Gpu& gpu) : g_(gpu) {}
    ~AfmeEngine();

    // For frames of `width`x`height`. False where the library or the GPU turns them down. Takes about
    // 270 ms in the library's pipelines, none of it on the queue.
    bool resize(uint32_t width, uint32_t height);
    bool sized(uint32_t width, uint32_t height) const {
        return engine_ != nullptr && width_ == width && height_ == height;
    }

    // The frame halfway between the two, into output(), left shader-read. The next call takes it
    // back. Both inputs are shader-read.
    bool record(VkCommandBuffer cb, VkImageView previous, VkImageView current);

    VkImageView output() const { return output_.view; }

private:
    void freeSized();

    const Gpu& g_;
    const luwes::afme::Api* api_ = nullptr;
    void* engine_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    Image depth_;
    VkImageView depthView_ = VK_NULL_HANDLE;
    Image motion_;
    Image output_;
    bool primed_ = false;
};

}  // namespace framefx::video
