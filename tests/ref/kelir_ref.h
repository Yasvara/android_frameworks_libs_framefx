/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include "kelir/colour.h"
#include "kelir/plan.h"

// The Kelir pass on the CPU, in float: the same semantics as kelir/shaders, written apart from
// them so either can catch the other. Tests only.
namespace framefx::kelir::ref {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> rgb;  // row 0 is the texture's first row

    Image() = default;
    Image(int w, int h) : width(w), height(h), rgb(static_cast<size_t>(w) * h * 3, 0.0f) {}

    Rgb at(int x, int y) const;  // clamped to the edges
    void set(int x, int y, Rgb c);
};

struct Frame {
    float displayWidth = 0.0f;
    float displayHeight = 0.0f;
    int turns = 0;       // quarter turns from buffer to display
    bool flipY = false;  // GL: buffer row 0 is the bottom
    float split = -1.0f;
    uint32_t seed = 0;
    Primaries primaries = Primaries::kSrgb;
    float halfLsb = 0.5f / 255.0f;
    bool readsLinear = false;   // the source holds what an sRGB view or fetch returns
    bool writesLinear = false;  // the target encodes what the pass writes
    bool dither = true;
};

// The quarter-resolution pyramid, fp16, levels 0..4, as the pyramid passes build it.
struct Pyramid {
    std::vector<Image> levels;
};

Pyramid BuildPyramid(const Image& source, bool readsLinear);

// The pass's output at every pixel, in what it writes (before the target's quantisation).
Image Render(const Plan& plan, const Image& source, const Frame& frame);

// One pixel of it, for tests that probe a few.
Rgb RenderPixel(const Plan& plan, const Image& source, const Pyramid& pyramid, const Frame& frame,
                int x, int y);

// The shaders' PCG3D and its uniform values, for statistics on per-present noise.
void Pcg(uint32_t v[3]);

}  // namespace framefx::kelir::ref
