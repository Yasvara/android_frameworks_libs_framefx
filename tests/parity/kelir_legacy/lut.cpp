/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "tests/parity/kelir_legacy/lut.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace framefx::kelir_legacy {
namespace {

// Black, 25/50/75 % grey, white, then skin, sky and foliage as sRGB-encoded references.
constexpr Rgb kReferences[kKelirSwatchColours] = {
        {0.0f, 0.0f, 0.0f},       {0.25f, 0.25f, 0.25f}, {0.5f, 0.5f, 0.5f},
        {0.75f, 0.75f, 0.75f},    {1.0f, 1.0f, 1.0f},    {0.84f, 0.62f, 0.50f},
        {0.42f, 0.62f, 0.85f},    {0.33f, 0.47f, 0.22f},
};

uint8_t unorm8(float v) {
    return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
}

Rgb texel(const Lut& lut, int r, int g, int b) {
    const uint16_t* t = &lut.texels[((static_cast<size_t>(b) * kLutSize + g) * kLutSize + r) * 4];
    return {FromHalf(t[0]), FromHalf(t[1]), FromHalf(t[2])};
}

Rgb lerp(Rgb a, Rgb b, float t) {
    return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t};
}

}  // namespace

uint16_t ToHalf(float v) {
    uint32_t f;
    memcpy(&f, &v, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((f >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = f & 0x7fffffu;
    if (((f >> 23) & 0xffu) == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0u));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7bffu);  // largest finite
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (half & 1u) != 0)) half++;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = static_cast<uint32_t>(exponent) << 10 | mantissa >> 13;
    const uint32_t rest = mantissa & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u) != 0)) half++;  // nearest even
    return static_cast<uint16_t>(sign | std::min(half, 0x7bffu));
}

float FromHalf(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign;
        } else {
            int e = -1;
            do {
                e++;
                mantissa <<= 1;
            } while ((mantissa & 0x400u) == 0);
            f = sign | static_cast<uint32_t>(127 - 15 - e) << 23 | (mantissa & 0x3ffu) << 13;
        }
    } else if (exponent == 31) {
        f = sign | 0x7f800000u | mantissa << 13;
    } else {
        f = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
    }
    float v;
    memcpy(&v, &f, sizeof(v));
    return v;
}

Rgb EvaluateColour(const Step* steps, int count, Primaries p, Rgb x) {
    for (int i = 0; i < count; i++) x = Blend(steps[i], x, p);
    return x;
}

void BakeLut(const Step* steps, int count, Primaries p, Lut* out) {
    out->texels.resize(kLutTexels * 4);
    out->linear = count > 0 && (steps[count - 1].kind == kColorblind ||
                                steps[count - 1].kind == kNightMode);
    float grid[kLutSize];
    for (int i = 0; i < kLutSize; i++) {
        grid[i] = std::pow(static_cast<float>(i) / (kLutSize - 1), 1.0f / kLutShaper);
    }
    uint16_t* t = out->texels.data();
    const uint16_t one = ToHalf(1.0f);
    for (int b = 0; b < kLutSize; b++) {
        for (int g = 0; g < kLutSize; g++) {
            for (int r = 0; r < kLutSize; r++) {
                Rgb y = EvaluateColour(steps, count, p, {grid[r], grid[g], grid[b]});
                if (out->linear) y = {Decode(y.r), Decode(y.g), Decode(y.b)};
                t[0] = ToHalf(y.r);
                t[1] = ToHalf(y.g);
                t[2] = ToHalf(y.b);
                t[3] = one;
                t += 4;
            }
        }
    }
}

Luts::Luts(const Step* steps, int count) : count_(std::min(count, kMaxKelirNodes)) {
    std::copy(steps, steps + count_, steps_);
    BakeLut(steps_, count_, Primaries::kSrgb, &srgb_);
}

const Lut& Luts::get(Primaries p) const {
    if (p == Primaries::kSrgb) return srgb_;
    std::call_once(p3Once_, [this] { BakeLut(steps_, count_, Primaries::kP3, &p3_); });
    return p3_;
}

Rgb SampleLut(const Lut& lut, Rgb x) {
    const auto axis = [](float v, int* i, float* f) {
        const float s = std::pow(std::clamp(v, 0.0f, 1.0f), kLutShaper) * (kLutSize - 1);
        *i = std::min(static_cast<int>(s), kLutSize - 2);
        *f = s - static_cast<float>(*i);
    };
    int r, g, b;
    float fr, fg, fb;
    axis(x.r, &r, &fr);
    axis(x.g, &g, &fg);
    axis(x.b, &b, &fb);
    const Rgb c00 = lerp(texel(lut, r, g, b), texel(lut, r + 1, g, b), fr);
    const Rgb c10 = lerp(texel(lut, r, g + 1, b), texel(lut, r + 1, g + 1, b), fr);
    const Rgb c01 = lerp(texel(lut, r, g, b + 1), texel(lut, r + 1, g, b + 1), fr);
    const Rgb c11 = lerp(texel(lut, r, g + 1, b + 1), texel(lut, r + 1, g + 1, b + 1), fr);
    const Rgb y = lerp(lerp(c00, c10, fg), lerp(c01, c11, fg), fb);
    return lut.linear ? Rgb{Encode(y.r), Encode(y.g), Encode(y.b)} : y;
}

void Swatch(const Step* steps, int count, Primaries p, uint8_t out[kKelirSwatchColours * 3]) {
    for (int i = 0; i < kKelirSwatchColours; i++) {
        const Rgb y = EvaluateColour(steps, count, p, kReferences[i]);
        out[3 * i] = unorm8(y.r);
        out[3 * i + 1] = unorm8(y.g);
        out[3 * i + 2] = unorm8(y.b);
    }
}

void Curve(const Step& step, Primaries p, uint8_t out[kKelirCurvePoints]) {
    const float* w = LumaWeights(p);
    for (int i = 0; i < kKelirCurvePoints; i++) {
        const float v = static_cast<float>(i) / (kKelirCurvePoints - 1);
        const Rgb y = Blend(step, {v, v, v}, p);
        out[i] = unorm8(y.r * w[0] + y.g * w[1] + y.b * w[2]);
    }
}

}  // namespace framefx::kelir_legacy
