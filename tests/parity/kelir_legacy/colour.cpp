/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "tests/parity/kelir_legacy/colour.h"

#include <algorithm>
#include <cmath>

namespace framefx::kelir_legacy {
namespace {

constexpr float kLumaSrgb[3] = {0.2126f, 0.7152f, 0.0722f};
constexpr float kLumaP3[3] = {0.2289746f, 0.6917385f, 0.0792869f};

// Linear Display P3 <-> linear sRGB (both D65).
constexpr float kP3ToSrgb[9] = {1.2249401f,  -0.2249404f, 0.0f,        -0.0420569f, 1.0420571f,
                                0.0f,        -0.0196376f, -0.0786361f, 1.0982735f};
constexpr float kSrgbToP3[9] = {0.8224621f, 0.1775380f, 0.0f,       0.0331941f, 0.9668058f,
                                0.0f,       0.0170827f, 0.0723974f, 0.9105199f};

// Machado, Oliveira and Fernandes 2009, severity 1.0, linear sRGB.
constexpr float kSimulate[3][9] = {
        {0.152286f, 1.052583f, -0.204868f, 0.114503f, 0.786281f, 0.099216f, -0.003882f, -0.048116f,
         1.051998f},
        {0.367322f, 0.860646f, -0.227968f, 0.280085f, 0.672501f, 0.047413f, -0.011820f, 0.042940f,
         0.968881f},
        {1.255528f, -0.076749f, -0.178779f, -0.078411f, 0.930809f, 0.147602f, 0.004733f, 0.691367f,
         0.303900f},
};

// Where the colours a deficiency loses go back in (Fidaner, Lin and Ozguven): red-green losses
// into green and blue, blue-yellow losses into red and green.
constexpr float kShiftRedGreen[9] = {0.0f, 0.0f, 0.0f, 0.7f, 1.0f, 0.0f, 0.7f, 0.0f, 1.0f};
constexpr float kShiftBlueYellow[9] = {1.0f, 0.0f, 0.7f, 0.0f, 1.0f, 0.7f, 0.0f, 0.0f, 0.0f};

// A warm white near 2850 K against D65, linear: blue light drops to 42 %.
constexpr Rgb kNightWhite = {1.0f, 0.70f, 0.42f};

// Sepia duotone: brown in the shadows, cream at white.
constexpr Rgb kSepiaShadow = {1.0f, 0.78f, 0.55f};
constexpr Rgb kSepiaLight = {1.0f, 0.95f, 0.85f};

Rgb operator+(Rgb a, Rgb b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
Rgb operator-(Rgb a, Rgb b) { return {a.r - b.r, a.g - b.g, a.b - b.b}; }
Rgb operator*(Rgb a, float s) { return {a.r * s, a.g * s, a.b * s}; }
Rgb operator*(Rgb a, Rgb b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }
Rgb grey(float v) { return {v, v, v}; }
Rgb mix(Rgb a, Rgb b, float t) { return a + (b - a) * t; }

float clamp01(float v) { return std::clamp(v, 0.0f, 1.0f); }

float dot(Rgb a, const float* w) { return a.r * w[0] + a.g * w[1] + a.b * w[2]; }

Rgb mul(const float* m, Rgb v) {
    return {m[0] * v.r + m[1] * v.g + m[2] * v.b, m[3] * v.r + m[4] * v.g + m[5] * v.b,
            m[6] * v.r + m[7] * v.g + m[8] * v.b};
}

float spow(float v, float e) { return std::copysign(std::pow(std::fabs(v), e), v); }

Rgb decode(Rgb x) { return {Decode(x.r), Decode(x.g), Decode(x.b)}; }
Rgb encode(Rgb x) { return {Encode(x.r), Encode(x.g), Encode(x.b)}; }

// Linear sRGB primaries for the matrices defined on them.
Rgb toSrgb(Rgb lin, Primaries p) { return p == Primaries::kP3 ? mul(kP3ToSrgb, lin) : lin; }
Rgb fromSrgb(Rgb lin, Primaries p) { return p == Primaries::kP3 ? mul(kSrgbToP3, lin) : lin; }

// Linear luminance of an encoded colour, re-encoded.
float encodedLuminance(Rgb x, Primaries p) { return Encode(dot(decode(x), LumaWeights(p))); }

Rgb brightnessContrast(const Step& s, Rgb x, Primaries p) {
    const float exposure = s.v[p::kExposure];
    if (exposure != 0.0f) x = encode(decode(x) * std::exp2(exposure));
    const float gamma = s.v[p::kGamma];
    if (gamma != 1.0f) x = {spow(x.r, 1.0f / gamma), spow(x.g, 1.0f / gamma), spow(x.b, 1.0f / gamma)};
    const float contrast = std::exp2(s.v[p::kContrast]);
    x = grey(0.5f) + (x - grey(0.5f)) * contrast;
    // Tone curve on luma, 0 and 1 fixed: shadows peak at a third, highlights at two thirds.
    // Outside [0, 1] each weight continues with its slope at the end, so the curve has no kink
    // for the LUT to miss.
    const float l = dot(x, LumaWeights(p));
    const float shadows = l < 0.0f ? l : l > 1.0f ? 0.0f : l * (1.0f - l) * (1.0f - l);
    const float highlights = l < 0.0f ? 0.0f : l > 1.0f ? 1.0f - l : l * l * (1.0f - l);
    return x + grey(0.8f * (s.v[p::kShadows] * shadows + s.v[p::kHighlights] * highlights));
}

// Hue in turns of a colour's own HSV hexagon, for the skin test.
float hue(Rgb c) {
    const float mx = std::max({c.r, c.g, c.b});
    const float mn = std::min({c.r, c.g, c.b});
    const float d = mx - mn;
    if (d <= 0.0f) return 0.0f;
    float h;
    if (mx == c.r) {
        h = (c.g - c.b) / d;
    } else if (mx == c.g) {
        h = 2.0f + (c.b - c.r) / d;
    } else {
        h = 4.0f + (c.r - c.g) / d;
    }
    h /= 6.0f;
    return h < 0.0f ? h + 1.0f : h;
}

Rgb fromHue(float degrees) {
    const float h = degrees / 60.0f;
    const auto channel = [h](float n) {
        const float k = std::fmod(n + h, 6.0f);
        return clamp01(1.0f - std::max(0.0f, std::min({k, 4.0f - k, 1.0f})));
    };
    return {channel(5.0f), channel(3.0f), channel(1.0f)};
}

Rgb color(const Step& s, Rgb x, Primaries p) {
    const float* w = LumaWeights(p);
    // White balance in linear light, luminance kept: warmer raises red over blue, a positive tint
    // takes green out (magenta).
    const float t = s.v[p::kTemperature];
    const float tint = s.v[p::kTint];
    if (t != 0.0f || tint != 0.0f) {
        Rgb gain = {std::exp2(0.35f * t), std::exp2(-0.25f * tint), std::exp2(-0.35f * t)};
        gain = gain * (1.0f / dot(gain, w));
        x = encode(decode(x) * gain);
    }
    const float h = s.v[p::kHue];
    if (h != 0.0f) {
        // Rotation about the grey axis.
        const float a = h * static_cast<float>(M_PI) / 180.0f;
        const float c = std::cos(a);
        const float k = 1.0f / std::sqrt(3.0f);
        const float sn = std::sin(a) * k;
        const float along = (x.r + x.g + x.b) * k * k * (1.0f - c);
        x = {x.r * c + (x.b - x.g) * sn + along, x.g * c + (x.r - x.b) * sn + along,
             x.b * c + (x.g - x.r) * sn + along};
    }
    const float vibrance = s.v[p::kVibrance];
    if (vibrance != 0.0f) {
        // Muted colours move most, greys and skin barely: saturation-weighted with grey and
        // skin protection, the non-learned form of a dynamic vibrance.
        // Weighted by 4 s (1 - s) of the saturation s: nothing for greys, most for muted
        // colours, nothing for colours already saturated, so they never crush. Skin tones move
        // a third as much. Every weight is smooth over the cube, so the LUT holds it.
        const Rgb c = {clamp01(x.r), clamp01(x.g), clamp01(x.b)};
        const float sat = std::max({c.r, c.g, c.b}) - std::min({c.r, c.g, c.b});
        const float d = std::fabs(hue(c) - 0.075f);
        const float skin = std::exp(-std::pow(std::min(d, 1.0f - d) / 0.07f, 2.0f));
        const float k = 1.0f + vibrance * 4.0f * sat * (1.0f - sat) * (1.0f - 0.7f * skin);
        const float l = dot(x, w);
        x = grey(l) + (x - grey(l)) * k;
    }
    const float saturation = s.v[p::kSaturation];
    if (saturation != 0.0f) {
        const float l = dot(x, w);
        x = grey(l) + (x - grey(l)) * (1.0f + saturation);
    }
    const float tintIntensity = s.v[p::kTintIntensity];
    if (tintIntensity != 0.0f) {
        // Colourise toward the hue at the pixel's own luma, as far as black and white allow.
        const Rgb tone = fromHue(s.v[p::kTintColour]);
        const Rgb chroma = tone - grey(dot(tone, w));
        const float l = dot(x, w);
        const Rgb target = grey(l) + chroma * std::max(0.0f, std::min(l, 1.0f - l));
        x = mix(x, target, tintIntensity);
    }
    return x;
}

Rgb levels(const Step& s, Rgb x) {
    const float white = s.v[p::kWhite];
    const float black = std::min(s.v[p::kBlack], white - 1.0f / 255.0f);
    const float range = 1.0f / (white - black);
    const float e = 1.0f / s.v[p::kMidtones];
    const auto f = [&](float c) { return spow((c - black) * range, e) + s.v[p::kBrightness]; };
    return {f(x.r), f(x.g), f(x.b)};
}

Rgb sepia(Rgb x, Primaries p) {
    const float y = encodedLuminance(x, p);
    return mix(kSepiaShadow, kSepiaLight, clamp01(y)) * y;
}

Rgb colorblind(const Step& s, Rgb x, Primaries p) {
    const int type = static_cast<int>(s.v[p::kType]);
    const Rgb lin = toSrgb(decode(x), p);
    const Rgb error = lin - mul(kSimulate[type], lin);
    const Rgb shifted = mul(type == kTritan ? kShiftBlueYellow : kShiftRedGreen, error);
    return encode(fromSrgb(lin + shifted, p));
}

}  // namespace

bool Step::operator==(const Step& o) const {
    return kind == o.kind && intensity == o.intensity && std::equal(v, v + kMaxParams, o.v);
}

const float* LumaWeights(Primaries p) {
    return p == Primaries::kP3 ? kLumaP3 : kLumaSrgb;
}

namespace {

float decodeExact(float a) {
    return a <= 0.04045f ? a / 12.92f : std::pow((a + 0.055f) / 1.055f, 2.4f);
}

float encodeExact(float a) {
    return a <= 0.0031308f ? a * 12.92f : 1.055f * std::pow(a, 1.0f / 2.4f) - 0.055f;
}

// The bake evaluates the transfer pair millions of times. Inside [0, 1] tables interpolate it to
// within 1e-7, far under half-float precision: decode over the encoded value, encode over the
// square root of the linear one, where it is smooth.
constexpr int kTableSize = 4096;

struct Tables {
    float decode[kTableSize + 1];
    float encode[kTableSize + 1];

    Tables() {
        for (int i = 0; i <= kTableSize; i++) {
            const float t = static_cast<float>(i) / kTableSize;
            decode[i] = decodeExact(t);
            encode[i] = encodeExact(t * t);
        }
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

float lookup(const float* table, float t) {
    const float x = t * kTableSize;
    const int i = std::min(static_cast<int>(x), kTableSize - 1);
    return table[i] + (table[i + 1] - table[i]) * (x - static_cast<float>(i));
}

}  // namespace

float Decode(float e) {
    const float a = std::fabs(e);
    return std::copysign(a <= 1.0f ? lookup(tables().decode, a) : decodeExact(a), e);
}

float Encode(float l) {
    const float a = std::fabs(l);
    return std::copysign(a <= 1.0f ? lookup(tables().encode, std::sqrt(a)) : encodeExact(a), l);
}

Rgb ApplyColour(const Step& s, Rgb x, Primaries p) {
    switch (s.kind) {
        case kBrightnessContrast:
            return brightnessContrast(s, x, p);
        case kColor:
            return color(s, x, p);
        case kLevels:
            return levels(s, x);
        case kBlackWhite:
            return grey(encodedLuminance(x, p));
        case kSepia:
            return sepia(x, p);
        case kNightMode:
            return encode(fromSrgb(toSrgb(decode(x), p) * kNightWhite, p));
        case kColorblind:
            return colorblind(s, x, p);
        default:
            return x;
    }
}

Rgb Blend(const Step& s, Rgb x, Primaries p) {
    return mix(x, ApplyColour(s, x, p), s.intensity);
}

bool IsIdentityColour(const Step& s) {
    const Param* params = Info(s.kind).params;
    const auto atDefault = [&](int i) { return s.v[i] == params[i].def; };
    switch (s.kind) {
        case kBrightnessContrast:
            return atDefault(p::kExposure) && atDefault(p::kContrast) &&
                    atDefault(p::kHighlights) && atDefault(p::kShadows) && atDefault(p::kGamma);
        case kColor:
            return atDefault(p::kTemperature) && atDefault(p::kTint) && atDefault(p::kHue) &&
                    atDefault(p::kSaturation) && atDefault(p::kVibrance) &&
                    atDefault(p::kTintIntensity);
        case kLevels:
            return atDefault(p::kBlack) && atDefault(p::kWhite) && atDefault(p::kMidtones) &&
                    atDefault(p::kBrightness);
        default:
            return false;
    }
}

}  // namespace framefx::kelir_legacy
