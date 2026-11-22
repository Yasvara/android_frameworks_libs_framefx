/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "tests/ref/kelir_ref.h"

#include <algorithm>
#include <cmath>

#include "kelir/lut.h"

namespace framefx::kelir::ref {
namespace {

constexpr int kLevels = 5;
constexpr float kPi = 3.14159265f;

struct V2 {
    float x = 0.0f;
    float y = 0.0f;
};

V2 operator+(V2 a, V2 b) { return {a.x + b.x, a.y + b.y}; }
V2 operator-(V2 a, V2 b) { return {a.x - b.x, a.y - b.y}; }
V2 operator*(V2 a, float s) { return {a.x * s, a.y * s}; }
V2 operator*(V2 a, V2 b) { return {a.x * b.x, a.y * b.y}; }
V2 operator/(V2 a, V2 b) { return {a.x / b.x, a.y / b.y}; }
float length(V2 a) { return std::sqrt(a.x * a.x + a.y * a.y); }
V2 floor2(V2 a) { return {std::floor(a.x), std::floor(a.y)}; }

Rgb operator+(Rgb a, Rgb b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
Rgb operator-(Rgb a, Rgb b) { return {a.r - b.r, a.g - b.g, a.b - b.b}; }
Rgb operator*(Rgb a, float s) { return {a.r * s, a.g * s, a.b * s}; }
Rgb operator*(Rgb a, Rgb b) { return {a.r * b.r, a.g * b.g, a.b * b.b}; }
Rgb grey(float v) { return {v, v, v}; }
Rgb mix(Rgb a, Rgb b, float t) { return a + (b - a) * t; }
Rgb min3(Rgb a, Rgb b) { return {std::min(a.r, b.r), std::min(a.g, b.g), std::min(a.b, b.b)}; }
Rgb max3(Rgb a, Rgb b) { return {std::max(a.r, b.r), std::max(a.g, b.g), std::max(a.b, b.b)}; }
Rgb clamp01(Rgb a) {
    return {std::clamp(a.r, 0.0f, 1.0f), std::clamp(a.g, 0.0f, 1.0f), std::clamp(a.b, 0.0f, 1.0f)};
}
float clampf(float v, float lo, float hi) { return std::clamp(v, lo, hi); }
float fract(float v) { return v - std::floor(v); }

float smoothstep(float e0, float e1, float x) {
    const float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

Rgb decode(Rgb c) { return {Decode(c.r), Decode(c.g), Decode(c.b)}; }
Rgb encode(Rgb c) { return {Encode(c.r), Encode(c.g), Encode(c.b)}; }
float half(float v) { return FromHalf(ToHalf(v)); }

// Filtering weights as the GPU's texture unit takes them: 8 fractional bits, the
// subTexelPrecisionBits and mipmapPrecisionBits Adreno reports.
float weight(float f) { return std::round(f * 256.0f) / 256.0f; }

Rgb bilinear(const Image& img, V2 uv) {
    const float x = uv.x * img.width - 0.5f;
    const float y = uv.y * img.height - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float fx = weight(x - x0);
    const float fy = weight(y - y0);
    const Rgb top = mix(img.at(x0, y0), img.at(x0 + 1, y0), fx);
    const Rgb bottom = mix(img.at(x0, y0 + 1), img.at(x0 + 1, y0 + 1), fx);
    return mix(top, bottom, fy);
}

struct Context {
    const Plan& plan;
    const Image& source;
    const Pyramid& pyramid;
    const Frame& frame;
    const float* luma;
    V2 bufferSize;
    V2 displaySize;

    float lumaOf(Rgb c) const { return c.r * luma[0] + c.g * luma[1] + c.b * luma[2]; }

    Rgb tap(int x, int y) const {
        const Rgb c = source.at(x, y);
        return frame.readsLinear ? encode(c) : c;
    }

    Rgb blur(V2 uv, float level) const {
        level = clampf(level, 0.0f, kLevels - 1.0f);
        const int l0 = static_cast<int>(std::floor(level));
        const int l1 = std::min(l0 + 1, kLevels - 1);
        return mix(bilinear(pyramid.levels[l0], uv), bilinear(pyramid.levels[l1], uv),
                   weight(level - l0));
    }

    Rgb colour(Rgb c) const {
        const Lut& lut = plan.luts->get(frame.primaries);
        return SampleLut(lut, c);
    }

    V2 displayUv(V2 px) const {
        V2 b = px / bufferSize;
        if (frame.flipY) b.y = 1.0f - b.y;
        switch (frame.turns) {
            case 1:
                return {b.y, 1.0f - b.x};
            case 2:
                return {1.0f - b.x, 1.0f - b.y};
            case 3:
                return {1.0f - b.y, b.x};
            default:
                return b;
        }
    }

    V2 bufferUv(V2 d) const {
        V2 b = d;
        if (frame.turns == 1) b = {1.0f - d.y, d.x};
        if (frame.turns == 2) b = {1.0f - d.x, 1.0f - d.y};
        if (frame.turns == 3) b = {d.y, 1.0f - d.x};
        if (frame.flipY) b.y = 1.0f - b.y;
        return b;
    }
};

Rgb random3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t v[3] = {a, b, c};
    Pcg(v);
    constexpr float k = 1.0f / 16777216.0f;
    return {static_cast<float>(v[0] >> 8) * k, static_cast<float>(v[1] >> 8) * k,
            static_cast<float>(v[2] >> 8) * k};
}

Rgb randomAt(V2 cell, uint32_t salt) {
    return random3(static_cast<uint32_t>(static_cast<int>(cell.x) + 32768),
                   static_cast<uint32_t>(static_cast<int>(cell.y) + 32768), salt);
}

// ---- Detail band ----------------------------------------------------------------------------

Rgb sharpen(const Context& ctx, const Step& s, Rgb c, Rgb src, int x, int y) {
    const Rgb n = ctx.tap(x, y - 1);
    const Rgb w = ctx.tap(x - 1, y);
    const Rgb e = ctx.tap(x + 1, y);
    const Rgb so = ctx.tap(x, y + 1);
    Rgb mn = min3(src, min3(min3(n, w), min3(e, so)));
    Rgb mx = max3(src, max3(max3(n, w), max3(e, so)));
    Rgb ring = n + w + e + so;
    float count = 4.0f;
    const auto headroomOf = [](Rgb lo, Rgb hi) {
        return Rgb{std::min(lo.r, 1.0f - hi.r), std::min(lo.g, 1.0f - hi.g),
                   std::min(lo.b, 1.0f - hi.b)};
    };
    Rgb headroom = headroomOf(mn, mx);
    if (static_cast<int>(s.v[p::kQuality]) == kPlus) {
        const Rgb nw = ctx.tap(x - 1, y - 1);
        const Rgb ne = ctx.tap(x + 1, y - 1);
        const Rgb sw = ctx.tap(x - 1, y + 1);
        const Rgb se = ctx.tap(x + 1, y + 1);
        const Rgb mn2 = min3(mn, min3(min3(nw, ne), min3(sw, se)));
        const Rgb mx2 = max3(mx, max3(max3(nw, ne), max3(sw, se)));
        headroom = (headroomOf(mn, mx) + headroomOf(mn2, mx2)) * 0.5f;
        mx = mx2;
        mn = mn2;
        ring = ring + (nw + ne + sw + se) * 0.5f;
        count = 6.0f;
    }
    const auto amp = [](float h, float m) {
        return std::sqrt(clampf(h / std::max(m, 1.0f / 1024.0f), 0.0f, 1.0f));
    };
    const float scale = (-1.0f / 5.0f) * (4.0f / count);
    const Rgb weight = {amp(headroom.r, mx.r) * scale, amp(headroom.g, mx.g) * scale,
                        amp(headroom.b, mx.b) * scale};
    const Rgb sharp = {(src.r + ring.r * weight.r) / (1.0f + count * weight.r),
                       (src.g + ring.g * weight.g) / (1.0f + count * weight.g),
                       (src.b + ring.b * weight.b) / (1.0f + count * weight.b)};
    const Rgb delta = sharp - src;
    const float range = std::max({mx.r - mn.r, mx.g - mn.g, mx.b - mn.b});
    const float floorAt = s.v[p::kIgnoreGrain] * 0.08f;
    const float keep = s.v[p::kIgnoreGrain] > 0.0f ? smoothstep(0.5f * floorAt, floorAt, range) : 1.0f;
    return c + delta * (keep * s.intensity);
}

Rgb bright(Rgb c, float threshold) {
    const float peak = std::max({c.r, c.g, c.b});
    const float knee = 0.05f + 0.25f * (1.0f - threshold);
    const float w = clampf((peak - threshold + knee) / (2.0f * knee), 0.0f, 1.0f);
    return c * (w * w);
}

Rgb details(const Context& ctx, const Step& s, Rgb c, Rgb src, V2 px) {
    const V2 uv = px / ctx.bufferSize;
    const float l = ctx.lumaOf(src);
    float lift = 0.0f;
    if (s.v[p::kClarity] != 0.0f) {
        const float base = ctx.lumaOf(ctx.blur(uv, 2.0f));
        const float mid = clampf(4.0f * l * (1.0f - l), 0.0f, 1.0f);
        lift += s.v[p::kClarity] * (l - base) * mid;
    }
    if (s.v[p::kHdrToning] != 0.0f) {
        const float mean = ctx.lumaOf(ctx.blur(uv, 3.0f));
        lift += s.v[p::kHdrToning] * (0.5f * (0.5f - mean) + 0.35f * (l - mean));
    }
    Rgb glow;
    if (s.v[p::kBloom] != 0.0f) {
        const float t = s.v[p::kBloomThreshold];
        glow = bright(ctx.blur(uv, 1.0f), t) * 0.30f + bright(ctx.blur(uv, 2.0f), t) * 0.30f +
                bright(ctx.blur(uv, 3.0f), t) * 0.25f + bright(ctx.blur(uv, 4.0f), t) * 0.15f;
        glow = glow * s.v[p::kBloom];
    }
    return c + (grey(lift) + glow) * s.intensity;
}

Rgb tiltShift(const Context& ctx, const Step& s, Rgb c, V2 px) {
    const V2 d = ctx.displayUv(px);
    const V2 q = (d - V2{0.5f, s.v[p::kPosition]}) * V2{ctx.displaySize.x / ctx.displaySize.y, 1.0f};
    float dist;
    if (s.v[p::kShape] < 0.5f) {
        const float t = s.v[p::kAngle] * kPi / 180.0f;
        dist = std::fabs(q.x * -std::sin(t) + q.y * std::cos(t));
    } else {
        dist = length(q);
    }
    const float edge = 0.5f * s.v[p::kSize];
    const float away = smoothstep(edge, edge + 0.5f * std::max(s.v[p::kSoftness], 0.01f), dist);
    const float level = away * s.v[p::kBlur] * 5.0f;
    const Rgb blurred = ctx.blur(px / ctx.bufferSize, std::max(level - 1.0f, 0.0f));
    return mix(c, blurred, clampf(level, 0.0f, 1.0f) * s.intensity);
}

Rgb painterly(const Context& ctx, const Step& s, Rgb c, Rgb src, V2 px) {
    const V2 uv = px / ctx.bufferSize;
    const Image& level0 = ctx.pyramid.levels[0];
    const V2 texel = {1.0f / level0.width, 1.0f / level0.height};
    const float reach = 0.75f + 1.75f * s.v[p::kBrush];
    Rgb sum;
    float total = 0.0f;
    for (int i = 0; i < 4; i++) {
        const V2 dir = V2{i == 1 || i == 3 ? 1.0f : -1.0f, i >= 2 ? 1.0f : -1.0f} * texel * reach;
        Rgb mean;
        float m1 = 0.0f;
        float m2 = 0.0f;
        for (int t = 0; t < 4; t++) {
            const V2 at = {t == 1 || t == 3 ? 1.5f : 0.5f, t >= 2 ? 1.5f : 0.5f};
            const Rgb v = ctx.blur(uv + dir * at, 0.0f);
            const float y = ctx.lumaOf(v);
            mean = mean + v;
            m1 += y;
            m2 += y * y;
        }
        mean = mean * 0.25f;
        const float variance = std::max(m2 * 0.25f - m1 * m1 * 0.0625f, 0.0f);
        const float w = 1.0f / (1e-4f + variance * variance * 1e4f);
        sum = sum + mean * w;
        total += w;
    }
    Rgb paint = sum * (1.0f / total);
    paint = paint + (src - ctx.blur(uv, 0.0f)) * (s.v[p::kDetail] * 0.6f);
    return mix(c, paint, s.intensity);
}

Rgb watercolor(const Context& ctx, const Step& s, Rgb c, V2 px, int x, int y) {
    const V2 uv = px / ctx.bufferSize;
    const V2 dpx = ctx.displayUv(px) * ctx.displaySize;
    const Rgb r = randomAt(floor2(dpx * (1.0f / 3.0f)), 0x57a7u);
    const V2 drift = V2{r.r - 0.5f, r.g - 0.5f} * (12.0f * s.v[p::kFuzziness]);
    const float toFrame = std::sqrt((ctx.bufferSize.x * ctx.bufferSize.y) /
                                    (ctx.displaySize.x * ctx.displaySize.y));
    Rgb wash = ctx.blur(uv + drift * toFrame / ctx.bufferSize, s.v[p::kBlurriness] * 2.0f);
    wash = mix(wash, grey(1.0f), 0.12f);
    const auto l = [&](int dx, int dy) { return ctx.lumaOf(ctx.tap(x + dx, y + dy)); };
    const float gx = (l(1, -1) + 2.0f * l(1, 0) + l(1, 1)) - (l(-1, -1) + 2.0f * l(-1, 0) + l(-1, 1));
    const float gy = (l(-1, 1) + 2.0f * l(0, 1) + l(1, 1)) - (l(-1, -1) + 2.0f * l(0, -1) + l(1, -1));
    const float strength = std::sqrt(gx * gx + gy * gy);
    const float onset = 0.9f + (0.12f - 0.9f) * s.v[p::kPencilDetail];
    const float line = smoothstep(onset, onset + 0.3f, strength) * s.v[p::kPencilIntensity];
    const float paper = 1.0f - 0.05f * randomAt(floor2(dpx * 0.5f), 0x9a9eu).r;
    const Rgb painting = wash * ((1.0f - 0.85f * line) * paper);
    return mix(c, painting, s.intensity);
}

// ---- Screen band ----------------------------------------------------------------------------

Rgb vignette(const Context& ctx, const Step& s, Rgb c, V2 duv) {
    const V2 d = (duv - V2{0.5f, 0.5f}) * 2.0f;
    const V2 circle = d * ctx.displaySize * (1.0f / std::min(ctx.displaySize.x, ctx.displaySize.y));
    const V2 q = d + (circle - d) * s.v[p::kRoundness];
    const float inner = 1.25f + (0.15f - 1.25f) * s.v[p::kVignetteSize];
    const float v = smoothstep(inner, inner + 0.9f, length(q));
    return c * (1.0f - v * s.intensity);
}

Rgb filmGrain(const Context& ctx, const Step& s, Rgb c, V2 duv) {
    const V2 cell = floor2(duv * ctx.displaySize * (1.0f / s.v[p::kGrainSize]));
    const Rgb u = randomAt(cell, ctx.frame.seed);
    const Rgb v = randomAt(cell, ctx.frame.seed ^ 0x6a09e667u);
    const Rgb noise = mix(grey(u.r + v.r - 1.0f), u + v - grey(1.0f), s.v[p::kGrainColour]);
    const float l = clampf(ctx.lumaOf(c), 0.0f, 1.0f);
    const float amplitude = 0.10f * (0.25f + 3.0f * l * (1.0f - l));
    return c + noise * (amplitude * s.intensity);
}

Rgb letterbox(const Context& ctx, const Step& s, Rgb c, V2 duv) {
    const float target = s.v[p::kAspect];
    const bool landscape = ctx.displaySize.x >= ctx.displaySize.y;
    const float longSide = std::max(ctx.displaySize.x, ctx.displaySize.y);
    const float shortSide = std::min(ctx.displaySize.x, ctx.displaySize.y);
    const float aspect = longSide / shortSide;
    float t;
    float span;
    float keep;
    if (target >= aspect) {
        t = landscape ? duv.y : duv.x;
        span = shortSide;
        keep = aspect / target;
    } else {
        t = landscape ? duv.x : duv.y;
        span = longSide;
        keep = target / aspect;
    }
    const float inside = clampf((keep - std::fabs(t - 0.5f) * 2.0f) * 0.5f * span + 0.5f, 0.0f, 1.0f);
    return c * (1.0f - (1.0f - inside) * s.intensity);
}

Rgb oldFilm(const Context& ctx, const Step& s, Rgb c, V2 duv) {
    const uint32_t seed = ctx.frame.seed;
    const V2 dpx = duv * ctx.displaySize;
    Rgb film = grey(ctx.lumaOf(c)) * Rgb{1.0f, 0.96f, 0.88f};
    film = film * (1.0f + (random3(seed, 0xf11cu, 0u).r - 0.5f) * 0.16f * s.v[p::kFlicker]);
    const V2 cell = floor2(dpx * (1.0f / 8.0f));
    const Rgb h = randomAt(cell, seed ^ 0xd057u);
    const float there =
            randomAt(cell, seed ^ 0xd058u).r >= 1.0f - 0.03f * s.v[p::kDust] ? 1.0f : 0.0f;
    const float radius = 0.6f + 1.6f * h.b;
    const float speck = there * (1.0f - smoothstep(radius - 0.5f, radius + 0.5f,
                                                   length(dpx - (cell + V2{h.r, h.g}) * 8.0f)));
    film = mix(film, grey(0.06f), speck);
    for (uint32_t i = 0; i < 3; i++) {
        const Rgb r = random3(seed, 0x5c7au, i);
        const float x = r.r * ctx.displaySize.x;
        const float on = r.g <= 0.6f * s.v[p::kScratches] ? 1.0f : 0.0f;
        const float line = on * (1.0f - smoothstep(0.5f, 1.5f, std::fabs(dpx.x - x)));
        film = mix(film, grey(0.9f), line * (0.35f + 0.4f * r.b));
    }
    return mix(c, film, s.intensity);
}

Rgb retro(const Context& ctx, const Step& s, Rgb c, V2 duv) {
    const float size = s.v[p::kPixelSize];
    const V2 cell = floor2(duv * ctx.displaySize * (1.0f / size));
    const V2 centre = ctx.bufferUv((cell + V2{0.5f, 0.5f}) * size / ctx.displaySize);
    const V2 at = centre * ctx.bufferSize;
    Rgb v = ctx.tap(static_cast<int>(at.x), static_cast<int>(at.y));
    if (ctx.plan.colourCount > 0) v = ctx.colour(v);
    const float steps = std::floor(s.v[p::kColourLevels] + 0.5f) - 1.0f;
    const auto q = [steps](float x) { return std::floor(clampf(x, 0.0f, 1.0f) * steps + 0.5f) / steps; };
    const float within = fract(duv.y * ctx.displaySize.y / size);
    const float scan = 1.0f - s.v[p::kScanlines] * 0.6f * smoothstep(0.45f, 0.95f, within) *
                    (size >= 2.0f ? 1.0f : 0.0f);
    return mix(c, Rgb{q(v.r), q(v.g), q(v.b)} * scan, s.intensity);
}

float dot(V2 p, float size, float degrees, float coverage) {
    const float t = degrees * kPi / 180.0f;
    const V2 r = V2{std::cos(t) * p.x + std::sin(t) * p.y, -std::sin(t) * p.x + std::cos(t) * p.y} *
            (1.0f / size);
    const float d = length(V2{fract(r.x) - 0.5f, fract(r.y) - 0.5f});
    const float radius = std::sqrt(clampf(coverage, 0.0f, 1.0f) / kPi);
    const float aa = 0.5f / size;
    return 1.0f - smoothstep(radius - aa, radius + aa, d);
}

Rgb halftone(const Context& ctx, const Step& s, Rgb c, V2 duv) {
    const V2 p = duv * ctx.displaySize;
    const float size = s.v[p::kDotSize];
    const Rgb cc = clamp01(c);
    Rgb printed;
    if (s.v[p::kInk] < 0.5f) {
        printed = grey(1.0f - dot(p, size, 45.0f, 1.0f - ctx.lumaOf(cc)));
    } else {
        const float k = 1.0f - std::max({cc.r, cc.g, cc.b});
        const float inv = 1.0f / std::max(1.0f - k, 1e-4f);
        const Rgb cmy = {(1.0f - cc.r - k) * inv, (1.0f - cc.g - k) * inv, (1.0f - cc.b - k) * inv};
        const Rgb inks = {dot(p, size, 15.0f, cmy.r), dot(p, size, 75.0f, cmy.g),
                          dot(p, size, 0.0f, cmy.b)};
        printed = (grey(1.0f) - inks) * (1.0f - dot(p, size, 45.0f, k));
    }
    return mix(c, printed, s.intensity);
}

float dither(V2 dpx) {
    const V2 f = floor2(dpx);
    return fract(52.9829189f * fract(f.x * 0.06711056f + f.y * 0.00583715f)) - 0.5f;
}

}  // namespace

Rgb Image::at(int x, int y) const {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    const float* p = &rgb[(static_cast<size_t>(y) * width + x) * 3];
    return {p[0], p[1], p[2]};
}

void Image::set(int x, int y, Rgb c) {
    float* p = &rgb[(static_cast<size_t>(y) * width + x) * 3];
    p[0] = c.r;
    p[1] = c.g;
    p[2] = c.b;
}

void Pcg(uint32_t v[3]) {
    for (int i = 0; i < 3; i++) v[i] = v[i] * 1664525u + 1013904223u;
    v[0] += v[1] * v[2];
    v[1] += v[2] * v[0];
    v[2] += v[0] * v[1];
    for (int i = 0; i < 3; i++) v[i] ^= v[i] >> 16u;
    v[0] += v[1] * v[2];
    v[1] += v[2] * v[0];
    v[2] += v[0] * v[1];
}

Pyramid BuildPyramid(const Image& source, bool readsLinear) {
    Pyramid p;
    Image level((source.width + 3) / 4, (source.height + 3) / 4);
    const V2 size = {static_cast<float>(source.width), static_cast<float>(source.height)};
    for (int y = 0; y < level.height; y++) {
        for (int x = 0; x < level.width; x++) {
            const V2 base = {x * 4.0f, y * 4.0f};
            Rgb c = (bilinear(source, (base + V2{1.0f, 1.0f}) / size) +
                     bilinear(source, (base + V2{3.0f, 1.0f}) / size) +
                     bilinear(source, (base + V2{1.0f, 3.0f}) / size) +
                     bilinear(source, (base + V2{3.0f, 3.0f}) / size)) *
                    0.25f;
            if (readsLinear) c = encode(c);
            level.set(x, y, {half(c.r), half(c.g), half(c.b)});
        }
    }
    p.levels.push_back(std::move(level));
    for (int k = 1; k < kLevels; k++) {
        const Image& above = p.levels.back();
        Image next(std::max(1, above.width / 2), std::max(1, above.height / 2));
        const V2 aboveSize = {static_cast<float>(above.width), static_cast<float>(above.height)};
        for (int y = 0; y < next.height; y++) {
            for (int x = 0; x < next.width; x++) {
                const Rgb c = bilinear(above, V2{x * 2.0f + 1.0f, y * 2.0f + 1.0f} / aboveSize);
                next.set(x, y, {half(c.r), half(c.g), half(c.b)});
            }
        }
        p.levels.push_back(std::move(next));
    }
    return p;
}

Rgb RenderPixel(const Plan& plan, const Image& source, const Pyramid& pyramid, const Frame& frame,
                int x, int y) {
    const Context ctx = {plan,
                         source,
                         pyramid,
                         frame,
                         LumaWeights(frame.primaries),
                         {static_cast<float>(source.width), static_cast<float>(source.height)},
                         {frame.displayWidth, frame.displayHeight}};
    const V2 px = {x + 0.5f, y + 0.5f};
    const Rgb src = ctx.tap(x, y);
    const V2 duv = ctx.displayUv(px);
    const V2 dpx = duv * ctx.displaySize;
    Rgb c = src;
    if (duv.x >= frame.split) {
        for (int i = 0; i < plan.detailCount; i++) {
            const Step& s = plan.detail[i];
            switch (s.kind) {
                case kSharpen:
                    c = sharpen(ctx, s, c, src, x, y);
                    break;
                case kDetails:
                    c = details(ctx, s, c, src, px);
                    break;
                case kTiltShift:
                    c = tiltShift(ctx, s, c, px);
                    break;
                case kPainterly:
                    c = painterly(ctx, s, c, src, px);
                    break;
                case kWatercolor:
                    c = watercolor(ctx, s, c, px, x, y);
                    break;
            }
        }
        if (plan.colourCount > 0) c = ctx.colour(c);
        for (int i = 0; i < plan.screenCount; i++) {
            const Step& s = plan.screen[i];
            switch (s.kind) {
                case kVignette:
                    c = vignette(ctx, s, c, duv);
                    break;
                case kFilmGrain:
                    c = filmGrain(ctx, s, c, duv);
                    break;
                case kLetterbox:
                    c = letterbox(ctx, s, c, duv);
                    break;
                case kOldFilm:
                    c = oldFilm(ctx, s, c, duv);
                    break;
                case kRetro:
                    c = retro(ctx, s, c, duv);
                    break;
                case kHalftone:
                    c = halftone(ctx, s, c, duv);
                    break;
            }
        }
        if (frame.dither) c = c + grey(dither(dpx) * frame.halfLsb * 2.0f);
    }
    if (frame.split >= 0.0f && std::fabs(dpx.x - frame.split * ctx.displaySize.x) < 0.5f) {
        c = mix(c, grey(1.0f), 0.75f);
    }
    return frame.writesLinear ? decode(c) : c;
}

Image Render(const Plan& plan, const Image& source, const Frame& frame) {
    const Pyramid pyramid = plan.needsPyramid() ? BuildPyramid(source, frame.readsLinear) : Pyramid();
    Image out(source.width, source.height);
    for (int y = 0; y < source.height; y++) {
        for (int x = 0; x < source.width; x++) {
            out.set(x, y, RenderPixel(plan, source, pyramid, frame, x, y));
        }
    }
    return out;
}

}  // namespace framefx::kelir::ref
