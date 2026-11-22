/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "lapis/projection.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace framefx::lapis {
namespace {

constexpr float kAspectTolerance = 0.01f;

bool Zero(float v) {
    return std::fabs(v) < 1e-6f;
}

float LongOverShort(float aspect) {
    return aspect >= 1.0f ? aspect : 1.0f / aspect;
}

int LogKey(float v) {
    return std::isinf(v) ? INT_MAX : static_cast<int>(std::lround(std::log2(v) * 1000.0f));
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
    const auto at = [&](int r, int c) { return rowMajor ? m[r * 4 + c] : m[c * 4 + r]; };
    if (!Zero(at(0, 1)) || !Zero(at(0, 3)) || !Zero(at(1, 0)) || !Zero(at(1, 3)) ||
        !Zero(at(2, 0)) || !Zero(at(2, 1)) || !Zero(at(3, 0)) || !Zero(at(3, 1)) ||
        !Zero(at(3, 3))) {
        return false;
    }
    const float w = at(3, 2);
    if (std::fabs(std::fabs(w) - 1.0f) > 1e-4f) return false;
    Projection p;
    p.sx = at(0, 0);
    p.sy = at(1, 1);
    p.jx = at(0, 2);
    p.jy = at(1, 2);
    p.a = at(2, 2);
    p.b = at(2, 3);
    p.w = w;
    p.rowMajor = rowMajor;
    if (!std::isfinite(p.sx + p.sy + p.jx + p.jy + p.a + p.b)) return false;
    if (std::fabs(p.sx) < 0.05f || std::fabs(p.sx) > 50.0f || std::fabs(p.sy) < 0.05f ||
        std::fabs(p.sy) > 50.0f || Zero(p.b) || std::fabs(p.jx) > 0.5f || std::fabs(p.jy) > 0.5f) {
        return false;
    }
    *out = p;
    return true;
}

namespace {

float Dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Whether the 16 floats at `f`, read with `rowMajor`, are P V: the w row a unit direction (the
// view axis), x and y rows perpendicular to each other with the focal scales `aspect` apart.
bool AsViewProjection(const float* f, bool rowMajor, float aspect, ViewProjection* out) {
    ViewProjection v;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) v.m[r * 4 + c] = rowMajor ? f[r * 4 + c] : f[c * 4 + r];
    }
    for (float x : v.m) {
        if (!std::isfinite(x)) return false;
    }
    const float* x = v.m;
    const float* y = v.m + 4;
    const float* w = v.m + 12;
    const float ww = Dot3(w, w);
    if (std::fabs(ww - 1.0f) > 0.02f) return false;
    // Jitter tilts x and y toward the view axis; take it out before the shape tests.
    v.jx = Dot3(x, w) / ww;
    v.jy = Dot3(y, w) / ww;
    float xs[3];
    float ys[3];
    for (int k = 0; k < 3; k++) {
        xs[k] = x[k] - v.jx * w[k];
        ys[k] = y[k] - v.jy * w[k];
    }
    const float sx = std::sqrt(Dot3(xs, xs));
    const float sy = std::sqrt(Dot3(ys, ys));
    if (sx < 0.05f || sy < 0.05f || std::fabs(v.jx) > 0.05f || std::fabs(v.jy) > 0.05f) return false;
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
            if (!AsViewProjection(c.values.data() + i, false, aspect, &v) &&
                !AsViewProjection(c.values.data() + i, true, aspect, &v)) {
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

// The 16 floats at `f` as an orthographic light matrix: w row (0, 0, 0, 1) up to scale, the three
// axes perpendicular and not the identity's (a UI's), a square map whose depth reaches well past
// its width (a rigid or scaled matrix, a camera's view or an object's, scales its axes alike or
// near), and `eye`, when given, inside its map. The depth axis, normalized, into `axis`; the
// rows' lengths into `scales`; the matrix, row-major, into `m`.
bool AsLightMatrix(const float* f, bool rowMajor, const float* eye, float axis[3],
                   float scales[3], float m[16]) {
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) m[r * 4 + c] = rowMajor ? f[r * 4 + c] : f[c * 4 + r];
    }
    for (int i = 0; i < 16; i++) {
        if (!std::isfinite(m[i]) || std::fabs(m[i]) > 1e6f) return false;
    }
    const float* x = m;
    const float* y = m + 4;
    const float* z = m + 8;
    const float* w = m + 12;
    if (!Zero(w[0]) || !Zero(w[1]) || !Zero(w[2]) || std::fabs(w[3]) < 1e-3f) return false;
    const float xx = Dot3(x, x);
    const float yy = Dot3(y, y);
    const float zz = Dot3(z, z);
    if (xx < 1e-12f || yy < 1e-12f || zz < 1e-12f) return false;
    if (std::fabs(Dot3(x, y)) > 0.01f * std::sqrt(xx * yy) ||
        std::fabs(Dot3(x, z)) > 0.01f * std::sqrt(xx * zz) ||
        std::fabs(Dot3(y, z)) > 0.01f * std::sqrt(yy * zz)) {
        return false;
    }
    const auto alongAxis = [](const float* v, int k, float len2) {
        return v[k] * v[k] > 0.9999f * len2;
    };
    if (alongAxis(x, 0, xx) && alongAxis(y, 1, yy) && alongAxis(z, 2, zz)) return false;
    if (std::fabs(xx - yy) > 0.04f * xx || zz > 0.25f * xx) return false;
    if (eye != nullptr) {
        // A cascade is fitted around the camera's view: the camera falls inside its square, and
        // within its depth, with a margin.
        const float cx = (Dot3(x, eye) + x[3]) / w[3];
        const float cy = (Dot3(y, eye) + y[3]) / w[3];
        const float cz = (Dot3(z, eye) + z[3]) / w[3];
        if (!(std::fabs(cx) <= 1.5f && std::fabs(cy) <= 1.5f && std::fabs(cz) <= 1.5f)) {
            return false;
        }
    }
    const float inv = 1.0f / std::sqrt(zz);
    for (int k = 0; k < 3; k++) axis[k] = z[k] * inv;
    scales[0] = std::sqrt(xx);
    scales[1] = std::sqrt(yy);
    scales[2] = std::sqrt(zz);
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
            float axis[3];
            float scales[3];
            float m[16];
            if (!AsLightMatrix(v.data() + i, false, nullptr, axis, scales, m) &&
                !AsLightMatrix(v.data() + i, true, nullptr, axis, scales, m)) {
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
            Seen found = {};
            found.count = 1;
            if (!AsLightMatrix(c.values.data() + i, false, eye, found.axis, found.scales, found.m) &&
                !AsLightMatrix(c.values.data() + i, true, eye, found.axis, found.scales, found.m)) {
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

void ProjectionVote::scan(const float* f, size_t count, float aspect, DepthRange range) {
    if (range != range_) {
        seen_.clear();
        range_ = range;
    }
    const float want = LongOverShort(aspect);
    for (size_t i = 0; i + 16 <= count; i += 4) {
        // Cheap reject first: one of the two layouts keeps these zero.
        if (!Zero(f[i + 1]) && !Zero(f[i + 4])) continue;
        Projection p;
        if (!AsProjection(f + i, false, &p) && !AsProjection(f + i, true, &p)) continue;
        if (std::fabs(LongOverShort(p.aspect()) / want - 1.0f) > kAspectTolerance) continue;
        float near;
        float far;
        if (!p.planes(range, &near, &far)) continue;
        const Key key = {static_cast<int>(std::lround(p.fovY() * 10.0f)),
                         static_cast<int>(std::lround(p.aspect() * 1000.0f)), LogKey(near),
                         LogKey(far)};
        Seen& s = seen_[key];
        s.p = p;
        s.near = near;
        s.far = far;
        s.count++;
    }
}

bool ProjectionVote::best(Projection* out, float* near, float* far) const {
    const Seen* top = nullptr;
    for (const auto& [key, s] : seen_) {
        if (top == nullptr || s.count > top->count) top = &s;
    }
    if (top == nullptr) return false;
    *out = top->p;
    *near = top->near;
    *far = top->far;
    return true;
}

}  // namespace framefx::lapis
