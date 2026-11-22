/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "tests/parity/kelir_legacy/params.h"

#include <cmath>

namespace framefx::kelir_legacy {
namespace {

constexpr Param C(const char* name, float lo, float def, float hi, int8_t needs = -1) {
    return {name, 0, lo, def, hi, needs};
}

constexpr Param D(const char* name, uint8_t choices) {
    return {name, choices, 0.0f, 0.0f, 0.0f, -1};
}

// Mirrored by tests/golden/kelir_params.json, which Pentas pins too.
constexpr KindInfo kKinds[kKindCount] = {
        {"Brightness/Contrast", Band::kColour, false, 5,
         {C("Exposure", -2.0f, 0.0f, 2.0f), C("Contrast", -1.0f, 0.0f, 1.0f),
          C("Highlights", -1.0f, 0.0f, 1.0f), C("Shadows", -1.0f, 0.0f, 1.0f),
          C("Gamma", 0.5f, 1.0f, 2.0f)}},
        {"Color", Band::kColour, false, 7,
         {C("Temperature", -1.0f, 0.0f, 1.0f), C("Tint", -1.0f, 0.0f, 1.0f),
          C("Hue", -180.0f, 0.0f, 180.0f), C("Saturation", -1.0f, 0.0f, 1.0f),
          C("Vibrance", -1.0f, 0.0f, 1.0f), C("Tint colour", 0.0f, 0.0f, 360.0f, p::kTintIntensity),
          C("Tint intensity", 0.0f, 0.0f, 1.0f)}},
        {"Levels", Band::kColour, false, 4,
         {C("Black", 0.0f, 0.0f, 0.5f), C("White", 0.5f, 1.0f, 1.0f),
          C("Midtones", 0.5f, 1.0f, 2.0f), C("Brightness", -0.5f, 0.0f, 0.5f)}},
        {"Black & White", Band::kColour, false, 0, {}},
        {"Sepia", Band::kColour, false, 0, {}},
        {"Night Mode", Band::kColour, false, 0, {}},
        {"Colorblind", Band::kColour, false, 1, {D("Type", 3)}},
        {"Sharpen", Band::kDetail, false, 2,
         {C("Ignore film grain", 0.0f, 0.0f, 1.0f), D("Quality", 2)}},
        {"Details", Band::kDetail, false, 4,
         {C("Clarity", -1.0f, 0.0f, 1.0f), C("HDR toning", 0.0f, 0.0f, 1.0f),
          C("Bloom", 0.0f, 0.0f, 1.0f), C("Bloom threshold", 0.2f, 0.75f, 1.0f, p::kBloom)}},
        {"Tilt-Shift", Band::kDetail, false, 6,
         {D("Shape", 2), C("Position", 0.0f, 0.5f, 1.0f), C("Size", 0.05f, 0.3f, 1.0f),
          C("Softness", 0.0f, 0.5f, 1.0f), C("Angle", -90.0f, 0.0f, 90.0f),
          C("Blur", 0.0f, 0.5f, 1.0f)}},
        {"Painterly", Band::kDetail, true, 2,
         {C("Brush", 0.0f, 0.5f, 1.0f), C("Detail", 0.0f, 0.5f, 1.0f)}},
        {"Watercolor", Band::kDetail, true, 4,
         {C("Blurriness", 0.0f, 0.5f, 1.0f), C("Fuzziness", 0.0f, 0.5f, 1.0f),
          C("Pencil detail", 0.0f, 0.5f, 1.0f), C("Pencil intensity", 0.0f, 0.5f, 1.0f)}},
        {"Vignette", Band::kScreen, false, 2,
         {C("Size", 0.0f, 0.5f, 1.0f), C("Roundness", 0.0f, 0.5f, 1.0f)}},
        {"Film Grain", Band::kScreen, false, 2,
         {C("Size", 0.5f, 1.0f, 3.0f), C("Colour", 0.0f, 0.0f, 1.0f)}},
        {"Letterbox", Band::kScreen, false, 1, {C("Aspect", 1.33f, 2.39f, 3.0f)}},
        {"Old Film", Band::kScreen, false, 3,
         {C("Dust", 0.0f, 0.5f, 1.0f), C("Scratches", 0.0f, 0.5f, 1.0f),
          C("Flicker", 0.0f, 0.3f, 1.0f)}},
        {"Retro", Band::kScreen, false, 3,
         {C("Pixel size", 1.0f, 4.0f, 16.0f), C("Colour levels", 2.0f, 8.0f, 32.0f),
          C("Scanlines", 0.0f, 0.5f, 1.0f)}},
        {"Halftone", Band::kScreen, false, 2, {C("Dot size", 2.0f, 6.0f, 16.0f), D("Ink", 2)}},
        {"Ambient occlusion", Band::kScene, false, 1, {C("Radius", 0.1f, 0.6f, 3.0f)}},
        {"Depth fog", Band::kScene, false, 5,
         {C("Start", 0.0f, 30.0f, 500.0f), C("Density", 0.0f, 0.3f, 1.0f),
          C("Hue", 0.0f, 210.0f, 360.0f), C("Saturation", 0.0f, 0.15f, 1.0f),
          C("Brightness", 0.0f, 0.85f, 1.0f)}},
        {"Depth of field", Band::kScene, false, 4,
         {D("Focus", 2), C("Distance", 0.3f, 10.0f, 300.0f, p::kFocus),
          C("Blur", 0.0f, 0.5f, 1.0f), C("Range", 0.0f, 0.3f, 1.0f)}},
        {"Global illumination", Band::kScene, false, 2,
         {C("Reach", 0.5f, 4.0f, 20.0f), C("Bounce", 0.0f, 1.0f, 3.0f)}},
};

}  // namespace

const KindInfo& Info(int kind) {
    return kKinds[kind];
}

bool CodeValid(const Param& p, int16_t q) {
    if (p.choices > 0) return q >= 0 && q < p.choices;
    if (q == INT16_MIN) return false;
    if (q > 0 && p.hi == p.def) return false;
    if (q < 0 && p.lo == p.def) return false;
    return true;
}

float Decode(const Param& p, int16_t q) {
    if (p.choices > 0) return static_cast<float>(q);
    // The ends exactly: rounding must not step outside the range.
    if (q == kQMax) return p.hi;
    if (q == -kQMax) return p.lo;
    const float t = static_cast<float>(q) / kQMax;
    const float v = q >= 0 ? p.def + t * (p.hi - p.def) : p.def + t * (p.def - p.lo);
    return std::fmin(std::fmax(v, p.lo), p.hi);
}

int16_t Encode(const Param& p, float value) {
    if (p.choices > 0) {
        const long v = std::lround(value);
        return static_cast<int16_t>(v < 0 ? 0 : v >= p.choices ? p.choices - 1 : v);
    }
    if (value >= p.def) {
        if (p.hi == p.def) return 0;
        const float t = std::fmin((value - p.def) / (p.hi - p.def), 1.0f);
        return static_cast<int16_t>(std::lround(t * kQMax));
    }
    if (p.lo == p.def) return 0;
    const float t = std::fmin((p.def - value) / (p.def - p.lo), 1.0f);
    return static_cast<int16_t>(-std::lround(t * kQMax));
}

}  // namespace framefx::kelir_legacy
