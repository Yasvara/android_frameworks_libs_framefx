/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lapis/model.h"

// A recorded frame flattened into the arrays lapis/rust reads (lapis/rust/src/model.rs Frame).
namespace framefx::lapis {

struct FfiImage {
    ImageId id;
    uint32_t format;
    uint32_t width;
    uint32_t height;
};

struct FfiTarget {
    ImageId image;
    uint32_t format;
    uint32_t load;
    uint32_t store;
    uint32_t samples;
};

struct FfiSpan {
    uint32_t start;
    uint32_t len;
};

struct FfiStep {
    bool isPass;
    bool hasDepth;
    uint8_t opKind;
    uint8_t flips;  // kFlipY, kFlipX: a blit's offsets, or a pass's first viewport
    FfiTarget depth;
    float clearDepth;
    uint32_t draws;
    uint32_t depthWrites;
    uint32_t depthClears;
    uint32_t lastClearDraw;
    FfiSpan colors;
    FfiSpan reads;
    ImageId src;
    ImageId dst;
};

struct FfiFrame {
    const FfiStep* steps;
    uint32_t stepCount;
    const FfiTarget* targets;
    uint32_t targetCount;
    const ImageId* reads;
    uint32_t readCount;
    const FfiImage* images;  // sorted by id
    uint32_t imageCount;
    ImageId presented;
    uint32_t width;
    uint32_t height;
};

static_assert(sizeof(FfiImage) == 16 && sizeof(FfiTarget) == 20);
static_assert(sizeof(FfiStep) == 68 && offsetof(FfiStep, depth) == 4 && offsetof(FfiStep, dst) == 64);

// Reused across frames on the recorder's thread, so a frame costs no allocations once warm.
struct FlatScratch {
    std::vector<FfiStep> steps;
    std::vector<FfiTarget> targets;
    std::vector<ImageId> reads;
    std::vector<ImageId> ids;
    std::vector<FfiImage> images;
};

// Points into `scratch`, valid until its next use.
FfiFrame Flatten(const Frame& frame, const Images& images, FlatScratch* scratch);

}  // namespace framefx::lapis
