/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstdint>

#include "kelir/params.h"

// The colour band evaluated exactly on the CPU: what the LUT bake samples, and the reference the
// tests hold the LUT and the passes to.
//
// Values are the target's encoded RGB (sRGB transfer, sRGB or Display P3 primaries) and are never
// clamped between steps; linear-light steps decode and re-encode exactly.
namespace framefx::kelir {

struct Rgb {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

enum class Primaries : uint8_t { kSrgb = 0, kP3 = 1 };

// One node of a plan, decoded.
struct Step {
    uint8_t kind = 0;
    float intensity = 1.0f;
    float v[kMaxParams] = {};

    bool operator==(const Step& o) const;
};

// Luma weights of the primaries, for encoded values.
const float* LumaWeights(Primaries p);

// Exact piecewise sRGB transfer, odd-extended below 0 and continued above 1.
float Decode(float encoded);
float Encode(float linear);

// The step's function of one colour, without its intensity. Colour-band kinds only.
Rgb ApplyColour(const Step& step, Rgb x, Primaries p);

// The step with its intensity: mixed with its input in encoded space.
Rgb Blend(const Step& step, Rgb x, Primaries p);

// True when the step leaves every colour unchanged.
bool IsIdentityColour(const Step& step);

}  // namespace framefx::kelir
