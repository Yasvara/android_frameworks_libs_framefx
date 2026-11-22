/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cmath>
#include <cstdint>

// How the game's scene, as stored, lines up with the frame shown: the two compared as RGBA8
// thumbnails of one size. Each is reduced to luma and normalised to zero mean and unit spread
// first, so a scene kept before its tonemap (brighter, flatter, another curve) still matches the
// shown frame by its shapes. No Vulkan call.
namespace framefx::luwes {

// A pixel counts as showing the scene within this many spreads of it.
inline constexpr float kShownSpread = 0.35f;
// A thumbnail whose luma spreads less than this (of 255) says nothing: a flat or black frame.
inline constexpr float kFlatSpread = 2.0f;

struct ThumbVote {
    uint64_t err[4] = {};  // per flip, bit 0 flips y, bit 1 flips x; in thousandths of a spread
    float shown = -1.0f;   // share of the frame showing the scene at `current`; -1 unknown
    bool flat = false;
};

namespace detail {
inline void NormalisedLuma(const uint8_t* rgba, int n, float* out, bool* flat) {
    double sum = 0, sq = 0;
    for (int i = 0; i < n; i++) {
        const uint8_t* p = rgba + i * 4;
        out[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
        sum += out[i];
        sq += static_cast<double>(out[i]) * out[i];
    }
    const double mean = sum / n;
    const double spread = std::sqrt(std::fmax(0.0, sq / n - mean * mean));
    *flat = spread < kFlatSpread;
    const float scale = *flat ? 0.0f : static_cast<float>(1.0 / spread);
    for (int i = 0; i < n; i++) out[i] = static_cast<float>(out[i] - mean) * scale;
}
}  // namespace detail

// `current` is the decided flip, or -1. W and H are the thumbnails' size.
template <int W, int H>
ThumbVote CompareThumbs(const uint8_t* real, const uint8_t* scene, int current) {
    float a[W * H], b[W * H];
    bool flatA = false, flatB = false;
    detail::NormalisedLuma(real, W * H, a, &flatA);
    detail::NormalisedLuma(scene, W * H, b, &flatB);
    ThumbVote v;
    v.flat = flatA || flatB;
    // A flat screen over a scene with shapes: a menu or a fade covers it.
    if (flatA && !flatB && current >= 0) v.shown = 0.0f;
    if (v.flat) return v;
    double err[4] = {};
    int shown = 0;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const float pa = a[y * W + x];
            for (int o = 0; o < 4; o++) {
                const int sy = (o & 1) ? H - 1 - y : y;
                const int sx = (o & 2) ? W - 1 - x : x;
                const float d = std::fabs(pa - b[sy * W + sx]);
                err[o] += d;
                if (o == current && d <= kShownSpread) shown++;
            }
        }
    }
    for (int o = 0; o < 4; o++) v.err[o] = static_cast<uint64_t>(err[o] * 1000.0);
    if (current >= 0) v.shown = static_cast<float>(shown) / (W * H);
    return v;
}

// The flip one vote is for, or -1 when no flip wins clearly.
inline int VoteOf(const ThumbVote& v) {
    if (v.flat) return -1;
    int best = 0;
    for (int o = 1; o < 4; o++) {
        if (v.err[o] < v.err[best]) best = o;
    }
    uint64_t second = UINT64_MAX;
    for (int o = 0; o < 4; o++) {
        if (o != best && v.err[o] < second) second = v.err[o];
    }
    return v.err[best] * 10 < second * 6 ? best : -1;
}

}  // namespace framefx::luwes
