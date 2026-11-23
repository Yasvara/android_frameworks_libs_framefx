/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "lapis/projection.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <tuple>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace framefx::lapis {
namespace {

constexpr float kAspectTolerance = 0.01f;
// Exact IEEE-754 binary32 representations for branchless magnitude comparisons.
// Masking the sign bit (& 0x7fffffff) preserves ordering over [0, +inf]; NaNs lie above kInfBits.
constexpr uint32_t kAbsMask = 0x7fffffffu;
constexpr uint32_t kZeroBits = 0x358637bdu;      // 1e-6f
constexpr uint32_t kInfBits = 0x7f800000u;       // +inf
constexpr uint32_t kLightMaxBits = 0x49742400u;  // 1e6f

inline uint32_t AbsBits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u & kAbsMask;
}

inline bool Zero(float v) {
    return AbsBits(v) < kZeroBits;
}

float LongOverShort(float aspect) {
    return aspect >= 1.0f ? aspect : 1.0f / aspect;
}

int LogKey(float v) {
    return std::isinf(v) ? INT_MAX : static_cast<int>(std::lround(std::log2(v) * 1000.0f));
}

template <bool RowMajor>
inline bool AsProjectionImpl(const float* m, Projection* out) {
    // Reject on w = ±1 first: in real GPU memory (mostly zeros and non-perspective data) this
    // eliminates >99.9% of candidates in one load before touching the rest of the matrix.
    const float w = RowMajor ? m[14] : m[11];
    if (!(std::fabs(std::fabs(w) - 1.0f) <= 1e-4f)) return false;

    const float sx = m[0];
    const float sy = m[5];
    const float asx = std::fabs(sx);
    const float asy = std::fabs(sy);
    if (!(asx >= 0.05f && asx <= 50.0f && asy >= 0.05f && asy <= 50.0f)) return false;

    const float b = RowMajor ? m[11] : m[14];
    if (AbsBits(b) < kZeroBits) return false;

    const float jx = RowMajor ? m[2] : m[8];
    const float jy = RowMajor ? m[6] : m[9];
    if (!(std::fabs(jx) <= 0.5f && std::fabs(jy) <= 0.5f)) return false;

    // All nine structurally-zero entries must have |v| < 1e-6f (and be non-NaN/Inf).
    // Shared across both layouts: m[1], m[3], m[4], m[7], m[12], m[13], m[15].
    // Layout-specific: row-major m[8], m[9]; column-major m[2], m[6].
    uint32_t maxZ = std::max(AbsBits(m[1]), AbsBits(m[3]));
    maxZ = std::max(maxZ, std::max(AbsBits(m[4]), AbsBits(m[7])));
    maxZ = std::max(maxZ, std::max(AbsBits(m[12]), AbsBits(m[13])));
    maxZ = std::max(maxZ, AbsBits(m[15]));
    if constexpr (RowMajor) {
        maxZ = std::max(maxZ, std::max(AbsBits(m[8]), AbsBits(m[9])));
    } else {
        maxZ = std::max(maxZ, std::max(AbsBits(m[2]), AbsBits(m[6])));
    }
    if (maxZ >= kZeroBits) return false;

    const float a = m[10];
    if (!std::isfinite(sx + sy + jx + jy + a + b)) return false;

    out->sx = sx;
    out->sy = sy;
    out->jx = jx;
    out->jy = jy;
    out->a = a;
    out->b = b;
    out->w = w;
    out->rowMajor = RowMajor;
    return true;
}

}  // namespace

float Projection::fovY() const {
    return 2.0f * std::atan(1.0f / std::fabs(sy)) * 180.0f / static_cast<float>(M_PI);
}

float Projection::aspect() const {
    return std::fabs(sy / sx);
}

bool Projection::planes(DepthRange range, float* near, float* far) const {
    const float c = w < 0 ? -a : a;
    const float inf = INFINITY;
    float n;
    float f;
    switch (range) {
        case DepthRange::kReversed:
            n = b / (1.0f - c);
            f = Zero(c) ? inf : -b / c;
            break;
        case DepthRange::kForward:
            if (Zero(c)) return false;
            n = -b / c;
            f = Zero(1.0f - c) ? inf : b / (1.0f - c);
            break;
        case DepthRange::kForwardGl:
            n = b / (-1.0f - c);
            f = Zero(1.0f - c) ? inf : b / (1.0f - c);
            break;
        default:
            return false;
    }
    if (!std::isfinite(n) || n <= 0.0f || std::isnan(f) || f <= n) return false;
    *near = n;
    *far = f;
    return true;
}

bool AsProjection(const float* m, bool rowMajor, Projection* out) {
    return rowMajor ? AsProjectionImpl<true>(m, out) : AsProjectionImpl<false>(m, out);
}

namespace {

inline float Dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Whether the 16 floats at `f`, read with `RowMajor`, are P V: the w row a unit direction (the
// view axis), x and y rows perpendicular to each other with the focal scales `aspect` apart.
template <bool RowMajor>
bool AsViewProjection(const float* f, float aspect, ViewProjection* out) {
    // Test the w row's unit length first: rejects almost all non-camera windows before transposing
    // or validating the other 12 floats.
    const float w0 = RowMajor ? f[12] : f[3];
    const float w1 = RowMajor ? f[13] : f[7];
    const float w2 = RowMajor ? f[14] : f[11];
    const float ww = w0 * w0 + w1 * w1 + w2 * w2;
    if (!(std::fabs(ww - 1.0f) <= 0.02f)) return false;

    ViewProjection v;
    if constexpr (RowMajor) {
        std::memcpy(v.m, f, sizeof(v.m));
    } else {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) v.m[r * 4 + c] = f[c * 4 + r];
        }
    }
    for (float val : v.m) {
        if (!std::isfinite(val)) return false;
    }
    const float* x = v.m;
    const float* y = v.m + 4;
    const float* w = v.m + 12;
    // Jitter tilts x and y toward the view axis; take it out before the shape tests.
    const float invWw = 1.0f / ww;
    v.jx = Dot3(x, w) * invWw;
    v.jy = Dot3(y, w) * invWw;
    if (std::fabs(v.jx) > 0.05f || std::fabs(v.jy) > 0.05f) return false;
    float xs[3];
    float ys[3];
    for (int k = 0; k < 3; k++) {
        xs[k] = x[k] - v.jx * w[k];
        ys[k] = y[k] - v.jy * w[k];
    }
    const float sx = std::sqrt(Dot3(xs, xs));
    const float sy = std::sqrt(Dot3(ys, ys));
    if (sx < 0.05f || sy < 0.05f) return false;
    if (std::fabs(sy / sx / aspect - 1.0f) > 0.02f) return false;
    if (std::fabs(Dot3(xs, ys)) > 0.01f * sx * sy) return false;
    v.world = std::fabs(w[3]) > 1e-3f || std::fabs(x[3]) > 1e-3f || std::fabs(y[3]) > 1e-3f;
    *out = v;
    return true;
}

}  // namespace

bool FindViewProjection(const std::vector<Constants>& blocks, float aspect, ViewProjection* out) {
    // Draws' blocks hold their objects' model-view-projections too, and a block read past its
    // struct (its size unknown, the read capped) holds stale frames' cameras. The camera's own is
    // one every draw reads, in a block of its own: in more than one block, and among those, the
    // smallest block's; then the most blocks.
    struct Candidate {
        ViewProjection v;
        int blocks = 0;
        size_t smallest = SIZE_MAX;  // floats of the smallest block holding it
    };
    std::vector<Candidate> found;
    const auto alike = [](const ViewProjection& a, const ViewProjection& b) {
        for (int k = 0; k < 16; k++) {
            if (std::fabs(a.m[k] - b.m[k]) > 1e-4f * std::max(1.0f, std::fabs(a.m[k]))) return false;
        }
        return true;
    };
    for (size_t b = 0; b < blocks.size(); b++) {
        const Constants& c = blocks[b];
        std::vector<size_t> counted;  // candidates this block already counted for
        for (size_t i = 0; i + 16 <= c.values.size(); i += 4) {
            ViewProjection v;
            if (!AsViewProjection<false>(c.values.data() + i, aspect, &v) &&
                !AsViewProjection<true>(c.values.data() + i, aspect, &v)) {
                continue;
            }
            size_t at = 0;
            while (at < found.size() && !alike(found[at].v, v)) at++;
            if (at == found.size()) found.push_back({v, 0});
            if (std::find(counted.begin(), counted.end(), at) == counted.end()) {
                found[at].blocks++;
                found[at].smallest = std::min(found[at].smallest, c.values.size());
                counted.push_back(at);
            }
        }
    }
    if (found.empty()) return false;
    const auto rank = [](const Candidate& c) {
        return std::make_tuple(c.v.world, c.blocks > 1, -static_cast<int64_t>(c.smallest),
                               c.blocks);
    };
    const Candidate* best = &found[0];
    for (const Candidate& c : found) {
        if (rank(c) > rank(*best)) best = &c;
    }
    *out = best->v;
    return true;
}

namespace {

inline bool LightW3Ok(float w3) {
    const float aw3 = std::fabs(w3);
    return aw3 >= 1e-3f && aw3 <= 1e6f;
}

// The 16 floats at `f` as an orthographic light matrix: w row (0, 0, 0, 1) up to scale, the three
// axes perpendicular and not the identity's (a UI's), a square map whose depth reaches well past
// its width (a rigid or scaled matrix, a camera's view or an object's, scales its axes alike or
// near), and `eye`, when given, inside its map. The depth axis, normalized, into `axis`; the
// rows' lengths into `scales`; the matrix, row-major, into `m`.
template <bool RowMajor>
bool AsLightMatrix(const float* f, const float* eye, float axis[3], float scales[3], float m[16]) {
    // Reject on the w row first (~94% of calls fail here) before touching the other 12 floats.
    const float w3 = f[15];
    if (!LightW3Ok(w3)) return false;
    const float w0 = RowMajor ? f[12] : f[3];
    const float w1 = RowMajor ? f[13] : f[7];
    const float w2 = RowMajor ? f[14] : f[11];
    if (std::max(AbsBits(w0), std::max(AbsBits(w1), AbsBits(w2))) >= kZeroBits) return false;

    float x[4], y[4], z[4];
    if constexpr (RowMajor) {
        std::memcpy(x, f, sizeof(x));
        std::memcpy(y, f + 4, sizeof(y));
        std::memcpy(z, f + 8, sizeof(z));
    } else {
        x[0] = f[0]; x[1] = f[4]; x[2] = f[8];  x[3] = f[12];
        y[0] = f[1]; y[1] = f[5]; y[2] = f[9];  y[3] = f[13];
        z[0] = f[2]; z[1] = f[6]; z[2] = f[10]; z[3] = f[14];
    }
    uint32_t maxMag = 0;
    for (int k = 0; k < 4; k++) {
        maxMag = std::max(maxMag, std::max(AbsBits(x[k]), std::max(AbsBits(y[k]), AbsBits(z[k]))));
    }
    if (maxMag > kLightMaxBits) return false;

    const float xx = Dot3(x, x);
    const float yy = Dot3(y, y);
    const float zz = Dot3(z, z);
    if (xx < 1e-12f || yy < 1e-12f || zz < 1e-12f) return false;
    // Scale ratio rejects ~90% of remaining candidates before computing cross dot products.
    if (std::fabs(xx - yy) > 0.04f * xx || zz > 0.25f * xx) return false;
    const auto alongAxis = [](const float* v, int k, float len2) {
        return v[k] * v[k] > 0.9999f * len2;
    };
    if (alongAxis(x, 0, xx) && alongAxis(y, 1, yy) && alongAxis(z, 2, zz)) return false;
    // Squared orthogonality avoids three sqrts: 1e-12 <= xx,yy,zz <= 3e12 keeps products in range.
    const float dxy = Dot3(x, y);
    const float dxz = Dot3(x, z);
    const float dyz = Dot3(y, z);
    if (dxy * dxy > 1e-4f * xx * yy || dxz * dxz > 1e-4f * xx * zz ||
        dyz * dyz > 1e-4f * yy * zz) {
        return false;
    }
    if (eye != nullptr) {
        // A cascade is fitted around the camera's view: the camera falls inside its square, and
        // within its depth, with a margin.
        const float cx = (Dot3(x, eye) + x[3]) / w3;
        const float cy = (Dot3(y, eye) + y[3]) / w3;
        const float cz = (Dot3(z, eye) + z[3]) / w3;
        if (!(std::fabs(cx) <= 1.5f && std::fabs(cy) <= 1.5f && std::fabs(cz) <= 1.5f)) {
            return false;
        }
    }
    const float sz = std::sqrt(zz);
    const float inv = 1.0f / sz;
    for (int k = 0; k < 3; k++) axis[k] = z[k] * inv;
    scales[0] = std::sqrt(xx);
    scales[1] = std::sqrt(yy);
    scales[2] = sz;
    std::memcpy(m, x, sizeof(x));
    std::memcpy(m + 4, y, sizeof(y));
    std::memcpy(m + 8, z, sizeof(z));
    m[12] = w0;
    m[13] = w1;
    m[14] = w2;
    m[15] = w3;
    return true;
}

}  // namespace

bool EyeOf(const float vp[16], float out[3]) {
    // Rows x, y and w at zero: three planes through the eye.
    const float* r0 = vp;
    const float* r1 = vp + 4;
    const float* r3 = vp + 12;
    const double a[3][3] = {{r0[0], r0[1], r0[2]}, {r1[0], r1[1], r1[2]}, {r3[0], r3[1], r3[2]}};
    const double b[3] = {-r0[3], -r1[3], -r3[3]};
    const auto det3 = [](const double m[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };
    const double det = det3(a);
    if (!std::isfinite(det) || std::fabs(det) < 1e-20) return false;
    for (int k = 0; k < 3; k++) {
        double m[3][3];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) m[r][c] = c == k ? b[r] : a[r][c];
        }
        out[k] = static_cast<float>(det3(m) / det);
    }
    return true;
}

std::vector<LightMatrix> LightMatrices(const std::vector<Constants>& blocks) {
    std::vector<LightMatrix> out;
    for (size_t b = 0; b < blocks.size(); b++) {
        const std::vector<float>& v = blocks[b].values;
        for (size_t i = 0; i + 16 <= v.size(); i += 4) {
            const float* p = v.data() + i;
            if (!LightW3Ok(p[15])) continue;
            float axis[3];
            float scales[3];
            float m[16];
            if (!AsLightMatrix<false>(p, nullptr, axis, scales, m) &&
                !AsLightMatrix<true>(p, nullptr, axis, scales, m)) {
                continue;
            }
            out.push_back({b, i, scales[0], m[3] / m[15], m[7] / m[15], m[11] / m[15]});
        }
    }
    return out;
}

bool FindLightDirection(const std::vector<Constants>& blocks, bool reversed, const float* eye,
                        float out[3], float* scales, float* matrix) {
    struct Seen {
        float axis[3];
        float scales[3];
        float m[16];
        int count;
    };
    std::vector<Seen> seen;
    for (const Constants& c : blocks) {
        for (size_t i = 0; i + 16 <= c.values.size(); i += 4) {
            const float* p = c.values.data() + i;
            if (!LightW3Ok(p[15])) continue;
            Seen found = {};
            found.count = 1;
            if (!AsLightMatrix<false>(p, eye, found.axis, found.scales, found.m) &&
                !AsLightMatrix<true>(p, eye, found.axis, found.scales, found.m)) {
                continue;
            }
            const float* axis = found.axis;
            auto it = std::find_if(seen.begin(), seen.end(),
                                   [&](const Seen& s) { return Dot3(s.axis, axis) > 0.999f; });
            if (it == seen.end()) {
                seen.push_back(found);
            } else {
                it->count++;
            }
        }
    }
    if (seen.empty()) return false;
    // The nearest cascade, the one that shades what the camera sees up close: the narrowest map.
    const auto best = std::max_element(seen.begin(), seen.end(), [](const Seen& a, const Seen& b) {
        return a.scales[0] < b.scales[0];
    });
    // Depth grows away from the light, or toward it when reversed.
    const float sign = reversed ? 1.0f : -1.0f;
    for (int k = 0; k < 3; k++) out[k] = best->axis[k] * sign;
    if (scales != nullptr) std::copy(best->scales, best->scales + 3, scales);
    if (matrix != nullptr) std::copy(best->m, best->m + 16, matrix);
    return true;
}

namespace {

// A cascade's place: its image, and its region's corner to a 1/64 of the image.
std::tuple<ImageId, int, int> keyOf(const LightVote::Cascade& m) {
    return {m.image, static_cast<int>(std::lround(m.region[0] * 64)),
            static_cast<int>(std::lround(m.region[1] * 64))};
}

}  // namespace

void LightVote::add(const std::vector<Constants>& blocks, bool reversed, const float* eye,
                     uint32_t draws, ImageId image, bool flipsY, const float region[4]) {
    Light l = {};
    Cascade map;
    map.image = image;
    map.reversed = reversed;
    map.flipsY = flipsY;
    if (region != nullptr) std::copy(region, region + 4, map.region);
    if (!FindLightDirection(blocks, reversed, eye, l.axis, l.scales, map.matrix)) {
        // A far cascade, fitted to a slice of the view past the camera: it does not hold the
        // camera. It counts once a nearer one shows its light.
        if (eye != nullptr && image != 0 &&
            FindLightDirection(blocks, reversed, nullptr, l.axis, l.scales, map.matrix)) {
            map.scale = l.scales[0];
            far_.push_back({{l.axis[0], l.axis[1], l.axis[2]}, {}, 0, {map}});
        }
        return;
    }
    l.draws = draws;
    map.scale = l.scales[0];
    if (image != 0) l.maps.push_back(map);
    for (Light& seen : frame_) {
        if (Dot3(seen.axis, l.axis) > 0.99f) {
            seen.draws += draws;
            seen.maps.insert(seen.maps.end(), l.maps.begin(), l.maps.end());
            if (l.scales[0] > seen.scales[0]) {
                std::copy(l.axis, l.axis + 3, seen.axis);
                std::copy(l.scales, l.scales + 3, seen.scales);
            }
            return;
        }
    }
    frame_.push_back(l);
}

bool LightVote::settle(float out[3], float scales[3]) {
    frames_++;
    if (frame_.empty()) {
        if (++missing_ > kHold) has_ = false;
    } else {
        missing_ = 0;
        const Light best = *std::max_element(
                frame_.begin(), frame_.end(),
                [](const Light& a, const Light& b) { return a.draws < b.draws; });
        frame_.clear();
        bool keep = false;
        if (!has_ || Dot3(best.axis, current_.axis) > 0.99f) {
            // The same light, followed as it moves.
            keep = true;
            has_ = true;
            wins_ = 0;
        } else {
            wins_ = Dot3(best.axis, challenger_.axis) > 0.99f ? wins_ + 1 : 1;
            challenger_ = best;
            if (wins_ >= kTakeOver) {
                keep = true;
                maps_.clear();
                wins_ = 0;
            }
        }
        if (keep) {
            current_ = best;
            for (Cascade m : best.maps) {
                m.frame = frames_;
                maps_[keyOf(m)] = m;
            }
            for (const Light& f : far_) {
                if (Dot3(f.axis, current_.axis) < 0.999f) continue;
                Cascade m = f.maps.front();
                m.frame = frames_;
                maps_[keyOf(m)] = m;
            }
        }
    }
    far_.clear();
    for (auto it = maps_.begin(); it != maps_.end();) {
        it = frames_ - it->second.frame > static_cast<uint64_t>(kHold) ? maps_.erase(it)
                                                                         : std::next(it);
    }
    if (!has_) {
        maps_.clear();
        return false;
    }
    std::copy(current_.axis, current_.axis + 3, out);
    std::copy(current_.scales, current_.scales + 3, scales);
    return true;
}

std::vector<LightVote::Cascade> LightVote::cascades() const {
    std::vector<Cascade> out;
    for (const auto& [key, m] : maps_) out.push_back(m);
    std::sort(out.begin(), out.end(),
              [](const Cascade& a, const Cascade& b) { return a.scale > b.scale; });
    return out;
}

__attribute__((noinline)) void ProjectionVote::scan(const float* f, size_t count, float aspect,
                                                    DepthRange range, uint64_t tag,
                                                    size_t baseOffset) {
    if (range != range_) {
        seen_.clear();
        range_ = range;
    }
    if (count < 16) return;
    const float want = LongOverShort(aspect);
    const size_t numVecs = count / 4;

    const auto checkAt = [&](size_t i) __attribute__((noinline, cold)) {
        Projection p;
        if (!AsProjectionImpl<false>(f + i, &p) && !AsProjectionImpl<true>(f + i, &p)) return;
        if (std::fabs(LongOverShort(p.aspect()) / want - 1.0f) > kAspectTolerance) return;
        float near;
        float far;
        if (!p.planes(range, &near, &far)) return;
        const Key key = {static_cast<int>(std::lround(p.fovY() * 10.0f)),
                         static_cast<int>(std::lround(p.aspect() * 1000.0f)), LogKey(near),
                         LogKey(far)};
        Seen& s = seen_[key];
        s.p = p;
        s.near = near;
        s.far = far;
        s.count++;
        if (tag != 0) {
            const size_t off = baseOffset + i * sizeof(float);
            for (uint8_t h = 0; h < s.hitCount; h++) {
                if (s.hits[h].tag == tag && s.hits[h].offset == off) return;
            }
            if (s.hitCount < Seen::kMaxHits) s.hits[s.hitCount++] = {tag, off};
        }
    };

    // Every perspective matrix in either layout has its last vec4 [m[12], m[13], m[14], m[15]]
    // shaped as [0, 0, non-zero finite, 0] (m[14] is `b` in col-major, `w = ±1` in row-major).
    // Filtering on that vec4 rejects zero-filled buffers, affine/rigid matrices (m[15] = 1),
    // general float data, and NaN/Inf in a single pass before touching the rest of the matrix.
    const auto* u = reinterpret_cast<const uint32_t*>(f);
    const auto vec3MayMatch = [](const uint32_t* v) -> bool {
        const uint32_t a0 = v[0] & kAbsMask;
        const uint32_t a1 = v[1] & kAbsMask;
        const uint32_t a2 = v[2] & kAbsMask;
        const uint32_t a3 = v[3] & kAbsMask;
        return std::max(a0, std::max(a1, a3)) < kZeroBits &&
                (a2 - kZeroBits) < (kInfBits - kZeroBits);
    };

    size_t j = 3;
#if defined(__ARM_NEON)
    // Align the 64-byte (4-vec4) NEON stream to cache lines when possible.
    while (j < numVecs && (reinterpret_cast<uintptr_t>(u + j * 4) & 63u) != 0) {
        if (__builtin_expect(vec3MayMatch(u + j * 4), 0)) checkAt((j - 3) * 4);
        j++;
    }
    const uint32x4_t vAbsMask = vdupq_n_u32(kAbsMask);
    const uint32_t subArr[4] = {0u, 0u, kZeroBits, 0u};
    const uint32_t limArr[4] = {kZeroBits, kZeroBits, kInfBits - kZeroBits, kZeroBits};
    const uint32x4_t vSub = vld1q_u32(subArr);
    const uint32x4_t vLim = vld1q_u32(limArr);
    for (; j + 4 <= numVecs; j += 4) {
        const uint32_t* p = u + j * 4;
        const uint32x4_t a0 = vandq_u32(vld1q_u32(p), vAbsMask);
        const uint32x4_t a1 = vandq_u32(vld1q_u32(p + 4), vAbsMask);
        const uint32x4_t a2 = vandq_u32(vld1q_u32(p + 8), vAbsMask);
        const uint32x4_t a3 = vandq_u32(vld1q_u32(p + 12), vAbsMask);

        const uint32x4_t ok0 = vcltq_u32(vsubq_u32(a0, vSub), vLim);
        const uint32x4_t ok1 = vcltq_u32(vsubq_u32(a1, vSub), vLim);
        const uint32x4_t ok2 = vcltq_u32(vsubq_u32(a2, vSub), vLim);
        const uint32x4_t ok3 = vcltq_u32(vsubq_u32(a3, vSub), vLim);

        const uint32x4_t hits = vpminq_u32(vpminq_u32(ok0, ok1), vpminq_u32(ok2, ok3));
        const uint64_t mask = vget_lane_u64(vreinterpret_u64_u16(vmovn_u32(hits)), 0);
        if (__builtin_expect(mask != 0, 0)) {
            if (mask & 0xffffu) checkAt((j - 3) * 4);
            if ((mask >> 16) & 0xffffu) checkAt((j - 2) * 4);
            if ((mask >> 32) & 0xffffu) checkAt((j - 1) * 4);
            if (mask >> 48) checkAt(j * 4);
        }
    }
#endif
    for (; j < numVecs; j++) {
        if (__builtin_expect(vec3MayMatch(u + j * 4), 0)) checkAt((j - 3) * 4);
    }
}

bool ProjectionVote::best(Projection* out, float* near, float* far,
                          std::vector<Hit>* hits) const {
    const Seen* top = nullptr;
    for (const auto& [key, s] : seen_) {
        if (top == nullptr || s.count > top->count) top = &s;
    }
    if (top == nullptr) return false;
    *out = top->p;
    *near = top->near;
    *far = top->far;
    if (hits != nullptr) hits->assign(top->hits, top->hits + top->hitCount);
    return true;
}

}  // namespace framefx::lapis
