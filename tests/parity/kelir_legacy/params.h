/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstdint>

// The Kelir kinds and their parameters: the one numeric table the codec, the plan, the bake and
// the shaders read. Ordinals and parameter order are the wire format; kinds are append-only.
namespace framefx::kelir_legacy {

enum class Band : uint8_t {
    kDetail = 0,  // reads the rendered frame's neighbourhood
    kColour = 1,  // a function of the pixel alone, folded into the LUT
    kScreen = 2,  // a function of the pixel and its place on the display
    kScene = 3,   // the game's 3D before its HUD, with its depth: only where Lapis taps them
};

enum Kind : uint8_t {
    kBrightnessContrast = 0,
    kColor = 1,
    kLevels = 2,
    kBlackWhite = 3,
    kSepia = 4,
    kNightMode = 5,
    kColorblind = 6,
    kSharpen = 7,
    kDetails = 8,
    kTiltShift = 9,
    kPainterly = 10,
    kWatercolor = 11,
    kVignette = 12,
    kFilmGrain = 13,
    kLetterbox = 14,
    kOldFilm = 15,
    kRetro = 16,
    kHalftone = 17,
    kAmbientOcclusion = 18,
    kDepthFog = 19,
    kDepthOfField = 20,
    kGlobalIllumination = 21,
    kKindCount = 22,
};

inline constexpr int kMaxParams = 8;

struct Param {
    const char* name;
    // Discrete: the number of choices, coded 0..choices-1. Continuous: 0, and lo <= def <= hi.
    uint8_t choices;
    float lo;
    float def;
    float hi;
    // Index of a parameter that must be away from its default for this one to matter, or -1.
    int8_t needs;
};

struct KindInfo {
    const char* name;
    Band band;
    bool style;  // the heavy detail kinds, at most one per stack
    uint8_t paramCount;
    Param params[kMaxParams];
};

const KindInfo& Info(int kind);

inline bool Known(int kind) { return kind >= 0 && kind < kKindCount; }

// Codes are s16, piecewise around the default: q >= 0 spans def..hi, q < 0 spans lo..def, so the
// default is exactly 0 and every code decodes inside [lo, hi]. -32768 is never valid.
inline constexpr int16_t kQMax = 32767;

bool CodeValid(const Param& p, int16_t q);
float Decode(const Param& p, int16_t q);
// Nearest code for a value, clamped into range. Exact for the default and the ends.
int16_t Encode(const Param& p, float value);

// Parameter indices, named where the bake and the passes read them.
namespace p {
// Brightness/Contrast
inline constexpr int kExposure = 0, kContrast = 1, kHighlights = 2, kShadows = 3, kGamma = 4;
// Color
inline constexpr int kTemperature = 0, kTint = 1, kHue = 2, kSaturation = 3, kVibrance = 4,
                     kTintColour = 5, kTintIntensity = 6;
// Levels
inline constexpr int kBlack = 0, kWhite = 1, kMidtones = 2, kBrightness = 3;
// Colorblind
inline constexpr int kType = 0;
// Sharpen
inline constexpr int kIgnoreGrain = 0, kQuality = 1;
// Details
inline constexpr int kClarity = 0, kHdrToning = 1, kBloom = 2, kBloomThreshold = 3;
// Tilt-Shift
inline constexpr int kShape = 0, kPosition = 1, kSize = 2, kSoftness = 3, kAngle = 4, kBlur = 5;
// Painterly
inline constexpr int kBrush = 0, kDetail = 1;
// Watercolor
inline constexpr int kBlurriness = 0, kFuzziness = 1, kPencilDetail = 2, kPencilIntensity = 3;
// Vignette
inline constexpr int kVignetteSize = 0, kRoundness = 1;
// Film Grain
inline constexpr int kGrainSize = 0, kGrainColour = 1;
// Letterbox
inline constexpr int kAspect = 0;
// Old Film
inline constexpr int kDust = 0, kScratches = 1, kFlicker = 2;
// Retro
inline constexpr int kPixelSize = 0, kColourLevels = 1, kScanlines = 2;
// Halftone
inline constexpr int kDotSize = 0, kInk = 1;
// Ambient occlusion
inline constexpr int kRadius = 0;
// Depth fog
inline constexpr int kFogStart = 0, kDensity = 1, kFogHue = 2, kFogSaturation = 3,
                     kFogBrightness = 4;
// Depth of field
inline constexpr int kFocus = 0, kDistance = 1, kDofBlur = 2, kRange = 3;
// Global illumination
inline constexpr int kReach = 0, kBounce = 1;
}  // namespace p

enum ColorblindType : uint8_t { kProtan = 0, kDeutan = 1, kTritan = 2 };
enum SharpenQuality : uint8_t { kStandard = 0, kPlus = 1 };
enum TiltShape : uint8_t { kBand = 0, kRadial = 1 };
enum Ink : uint8_t { kMono = 0, kCmyk = 1 };
enum Focus : uint8_t { kAutoFocus = 0, kFixedFocus = 1 };

}  // namespace framefx::kelir_legacy
