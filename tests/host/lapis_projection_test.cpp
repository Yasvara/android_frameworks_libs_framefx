/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "lapis/projection.h"

namespace framefx::lapis {
namespace {

// A right-handed perspective (w = -1), column-major unless asked, for `range`.
std::vector<float> Perspective(float fovY, float aspect, float near, float far, DepthRange range,
                               bool rowMajor = false, float jx = 0, float jy = 0) {
    const float sy = 1.0f / std::tan(fovY * static_cast<float>(M_PI) / 360.0f);
    float a;
    float b;
    switch (range) {
        case DepthRange::kReversed:
            a = std::isinf(far) ? 0.0f : near / (far - near);
            b = std::isinf(far) ? near : near * far / (far - near);
            break;
        case DepthRange::kForward:
            a = -far / (far - near);
            b = near * far / (near - far);
            break;
        case DepthRange::kForwardGl:
            a = -(far + near) / (far - near);
            b = 2.0f * near * far / (near - far);
            break;
    }
    const float rows[4][4] = {{sy / aspect, 0, jx, 0}, {0, sy, jy, 0}, {0, 0, a, b}, {0, 0, -1, 0}};
    std::vector<float> m(16);
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) m[rowMajor ? r * 4 + c : c * 4 + r] = rows[r][c];
    }
    return m;
}

void Put(std::vector<float>* memory, size_t at, const std::vector<float>& m) {
    std::copy(m.begin(), m.end(), memory->begin() + at);
}

TEST(LapisProjection, ReadsGenshinsCamera) {
    Projection p;
    ASSERT_TRUE(AsProjection(Perspective(45, 2.166f, 0.25f, 6000, DepthRange::kReversed).data(),
                             false, &p));
    EXPECT_NEAR(p.fovY(), 45.0f, 0.01f);
    EXPECT_NEAR(p.aspect(), 2.166f, 0.001f);
    float near;
    float far;
    ASSERT_TRUE(p.planes(DepthRange::kReversed, &near, &far));
    EXPECT_NEAR(near, 0.25f, 1e-3f);
    EXPECT_NEAR(far, 6000.0f, 30.0f);
    // The same matrix is no frustum as forward depth.
    EXPECT_FALSE(p.planes(DepthRange::kForward, &near, &far));
}

TEST(LapisProjection, ReadsAnInfiniteReversedCamera) {
    Projection p;
    ASSERT_TRUE(AsProjection(
            Perspective(46.7f, 2.163f, 10, INFINITY, DepthRange::kReversed, true).data(), true, &p));
    float near;
    float far;
    ASSERT_TRUE(p.planes(DepthRange::kReversed, &near, &far));
    EXPECT_NEAR(near, 10.0f, 1e-3f);
    EXPECT_TRUE(std::isinf(far));
}

TEST(LapisProjection, ReadsAGlCamera) {
    Projection p;
    ASSERT_TRUE(
            AsProjection(Perspective(30, 2.167f, 0.3f, 250, DepthRange::kForwardGl).data(), false, &p));
    float near;
    float far;
    ASSERT_TRUE(p.planes(DepthRange::kForwardGl, &near, &far));
    EXPECT_NEAR(near, 0.3f, 1e-3f);
    EXPECT_NEAR(far, 250.0f, 1.0f);
}

// A GL game drawn through ANGLE keeps GL's clip depth: read in Vulkan's range, its near plane
// doubles and every stored depth lands far away.
TEST(LapisProjection, AGlGameOnAngleReadsItsDepthInGlsRange) {
    EXPECT_EQ(RangeOf(false, true), DepthRange::kForwardGl);
    EXPECT_EQ(RangeOf(false, false), DepthRange::kForward);
    EXPECT_EQ(RangeOf(true, true), DepthRange::kReversed);
    EXPECT_EQ(RangeOf(true, false), DepthRange::kReversed);

    // Uma's camera: 33 degrees, portrait, 0.3 to 200.
    const float n = 0.3f, f = 200.0f;
    Projection p;
    ASSERT_TRUE(AsProjection(Perspective(33, 0.463f, n, f, DepthRange::kForwardGl).data(), false, &p));
    float near = 0, far = 0;
    ASSERT_TRUE(p.planes(DepthRange::kForwardGl, &near, &far));
    EXPECT_NEAR(near, n, 1e-3f);
    EXPECT_NEAR(far, f, 0.5f);
    for (const float distance : {0.5f, 3.0f, 25.0f, 150.0f}) {
        const float ndc = (f + n) / (f - n) - 2.0f * f * n / ((f - n) * distance);
        const float stored = 0.5f * ndc + 0.5f;
        EXPECT_NEAR(ViewDistance(p, DepthRange::kForwardGl, stored), distance, distance * 0.01f);
        EXPECT_GT(ViewDistance(p, DepthRange::kForward, stored), distance * 1.1f);
    }
}

TEST(LapisProjection, RejectsOrthographicAndIdentity) {
    const float ortho[16] = {0.01f, 0, 0, 0, 0, 0.02f, 0, 0, 0, 0, -0.001f, 0, -1, 1, 0.5f, 1};
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    Projection p;
    EXPECT_FALSE(AsProjection(ortho, false, &p));
    EXPECT_FALSE(AsProjection(ortho, true, &p));
    EXPECT_FALSE(AsProjection(identity, false, &p));
}

TEST(LapisProjection, VotePicksTheSceneCameraByAspectAndRange) {
    std::vector<float> memory(4096, 0.5f);
    // Genshin keeps a reversed scene camera and a forward one at the same field of view.
    for (int k = 0; k < 6; k++) {
        Put(&memory, 64 + k * 64,
            Perspective(45, 2.166f, 0.25f, 6000, DepthRange::kReversed, false, 0.0004f * k));
    }
    for (int k = 0; k < 8; k++) {
        Put(&memory, 1024 + k * 64, Perspective(45, 2.166f, 0.5f, 6000, DepthRange::kForward));
    }
    // A square shadow-ish camera that appears most often.
    for (int k = 0; k < 20; k++) {
        Put(&memory, 2048 + k * 64, Perspective(90, 1.0f, 0.1f, 100, DepthRange::kReversed));
    }

    ProjectionVote vote;
    vote.scan(memory.data(), memory.size(), 2118.0f / 978.0f, DepthRange::kReversed);
    Projection p;
    float near;
    float far;
    ASSERT_TRUE(vote.best(&p, &near, &far));
    EXPECT_NEAR(p.fovY(), 45.0f, 0.01f);
    EXPECT_NEAR(near, 0.25f, 1e-3f);
    EXPECT_NEAR(p.jx, 0.0020f, 1e-6f);  // the newest jitter

    vote.scan(memory.data(), memory.size(), 2118.0f / 978.0f, DepthRange::kForward);
    ASSERT_TRUE(vote.best(&p, &near, &far));
    EXPECT_NEAR(near, 0.5f, 1e-3f);
}

TEST(LapisProjection, VoteMatchesAPortraitDepthBuffer) {
    std::vector<float> memory(256, 0.0f);
    Put(&memory, 16, Perspective(30, 2.167f, 0.3f, 250, DepthRange::kForwardGl));
    ProjectionVote vote;
    vote.scan(memory.data(), memory.size(), 908.0f / 1968.0f, DepthRange::kForwardGl);
    Projection p;
    float near;
    float far;
    EXPECT_TRUE(vote.best(&p, &near, &far));
}

TEST(LapisProjection, VoteFindsNothingInNoise) {
    std::vector<float> memory(4096);
    for (size_t i = 0; i < memory.size(); i++) memory[i] = std::sin(static_cast<float>(i) * 0.37f);
    ProjectionVote vote;
    vote.scan(memory.data(), memory.size(), 2.0f, DepthRange::kReversed);
    Projection p;
    float near;
    float far;
    EXPECT_FALSE(vote.best(&p, &near, &far));
}

TEST(LapisProjection, VoteRecordsHitOffsetsAndEdges) {
    // Test start=0, unaligned-to-64 offsets, tail edge, both layouts, and NaN/zero resilience.
    for (bool rowMajor : {false, true}) {
        const auto mat = Perspective(45, 2.166f, 0.25f, 6000, DepthRange::kReversed, rowMajor);
        for (size_t at : {size_t{0}, size_t{4}, size_t{8}, size_t{12}, size_t{16}, size_t{84}}) {
            std::vector<float> mem(100, 0.0f);
            mem[3] = NAN;
            mem[7] = INFINITY;
            mem[11] = -0.0f;
            Put(&mem, at, mat);
            ProjectionVote vote;
            vote.scan(mem.data(), mem.size(), 2118.0f / 978.0f, DepthRange::kReversed, 0x1234u, 64u);
            Projection p;
            float near, far;
            std::vector<ProjectionVote::Hit> hits;
            ASSERT_TRUE(vote.best(&p, &near, &far, &hits)) << "rowMajor=" << rowMajor << " at=" << at;
            EXPECT_EQ(p.rowMajor, rowMajor);
            ASSERT_EQ(hits.size(), 1u);
            EXPECT_EQ(hits[0].tag, 0x1234u);
            EXPECT_EQ(hits[0].offset, 64u + at * sizeof(float));
        }
    }
}

// Genshin's per-camera constants as its scene pass read them on device (shot 2026-10-21): a block
// with the camera position, the world-space view-projection with its jitter, then the same
// camera-relative.
std::vector<Constants> GenshinCamera() {
    Constants c;
    c.values = {1.27242f,     211.5791f,   -3.470809f,   31.76604f,  -1.0f,       0.25f,
                6000.0f,      1.666667e-4f, 0.160728f,    0.9156845f, 0.3683591f,  0.0f,
                0.8882637f,   0.02850848f, -0.4584482f,   0.0f,       -0.4302953f, 0.4008856f,
                -0.8087871f,  0.0f,        0.07504319f,  211.6942f,  -0.4969115f, 1.0f,
                1.04669f,     0.09383692f, 1.424864e-05f, -0.3419922f, 1.001917e-05f, -2.398772f,
                4.716459e-06f, -0.1132032f, 0.3836296f,  -0.2559609f, -3.886632e-05f, 0.9328592f,
                -0.002444658f, 506.5221f,  0.2488595f,   27.62437f};
    return {c};
}

TEST(LapisViewProjection, FindsGenshinsJitteredCamera) {
    ViewProjection vp;
    ASSERT_TRUE(FindViewProjection(GenshinCamera(), 2118.0f / 978.0f, &vp));
    EXPECT_TRUE(vp.world);
    // Jitter to within float noise of what its anti-aliasing read the same frame.
    EXPECT_NEAR(vp.jx, -8.87e-5f, 2e-6f);
    EXPECT_NEAR(vp.jy, 6.81e-4f, 2e-6f);
    // Rows: x, y, depth, w; w the view axis.
    EXPECT_NEAR(vp.m[12], -0.3419922f, 1e-6f);
    EXPECT_NEAR(vp.m[15], 27.62437f, 1e-4f);
}

TEST(LapisViewProjection, PrefersTheCameraEveryDrawReads) {
    const std::vector<Constants> camera = GenshinCamera();
    const std::vector<float>& m = camera[0].values;  // the matrix row-major from float 24
    Constants object = camera[0];
    // The camera turned a quarter about y: its x and z columns swap.
    for (int r = 0; r < 4; r++) {
        object.values[24 + r * 4] = m[24 + r * 4 + 2];
        object.values[24 + r * 4 + 2] = -m[24 + r * 4];
    }
    // Both draws' large blocks, read past their structs, hold a stale camera alike; each draw's
    // small block holds the camera.
    std::vector<float> large = object.values;
    for (int i = 0; i < 64; i++) large.push_back(std::sin(static_cast<float>(i) * 0.37f));
    Constants first;
    first.values = large;
    Constants second = first;
    second.binding = 2;
    Constants own = camera[0];
    own.binding = 1;
    Constants again = own;
    again.set = 1;
    ViewProjection vp;
    ASSERT_TRUE(FindViewProjection({first, own, second, again}, 2118.0f / 978.0f, &vp));
    EXPECT_NEAR(vp.m[15], 27.62437f, 1e-4f);
    ViewProjection alone;
    ASSERT_TRUE(FindViewProjection(camera, 2118.0f / 978.0f, &alone));
    for (int k = 0; k < 16; k++) EXPECT_FLOAT_EQ(vp.m[k], alone.m[k]);
}

TEST(LapisViewProjection, WantsTheScenesAspect) {
    ViewProjection vp;
    EXPECT_FALSE(FindViewProjection(GenshinCamera(), 16.0f / 9.0f, &vp));
}

TEST(LapisViewProjection, IgnoresObjectMatricesAndNoise) {
    Constants c;
    // An object's world matrix: no view axis in its last row.
    c.values = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4.0f, 221.0f, 3.4f, 1};
    for (int i = 0; i < 64; i++) c.values.push_back(std::sin(static_cast<float>(i) * 0.37f));
    ViewProjection vp;
    EXPECT_FALSE(FindViewProjection({c}, 2.0f, &vp));
}

// A light's cascade around `eye` looking along `f` (by default a sun's), row-major.
std::vector<float> SunCascade(const float eye[3], std::vector<float> f = {-0.29f, -0.77f, 0.56f}) {
    float len = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    const float z[3] = {-f[0] / len, -f[1] / len, -f[2] / len};  // depth grows away from the sun
    float x[3] = {z[1], -z[0], 0};
    len = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    x[0] /= len;
    x[1] /= len;
    const float y[3] = {z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2],
                        z[0] * x[1] - z[1] * x[0]};
    const float half = 40.0f;
    const float range = 400.0f;
    std::vector<float> m(16, 0.0f);
    for (int k = 0; k < 3; k++) {
        m[k] = x[k] / half;
        m[4 + k] = y[k] / half;
        m[8 + k] = -z[k] / range;
    }
    m[3] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]) / half;
    m[7] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]) / half;
    m[11] = 0.5f + (z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]) / range;
    m[15] = 1.0f;
    return m;
}

TEST(LapisLight, FindsTheSunAmongObjectRotations) {
    const float eye[3] = {100.0f, 20.0f, -40.0f};
    Constants c;
    // Draws' own rotations about the vertical, more of them than the one light matrix.
    for (int i = 1; i <= 5; i++) {
        const float a = 0.7f * static_cast<float>(i);
        const std::vector<float> object = {std::cos(a), 0, std::sin(a), 90.0f + i, 0, 1, 0, 2.0f,
                                           -std::sin(a), 0, std::cos(a), -30.0f, 0, 0, 0, 1};
        c.values.insert(c.values.end(), object.begin(), object.end());
        c.values.insert(c.values.end(), object.begin(), object.end());
    }
    const std::vector<float> sun = SunCascade(eye);
    c.values.insert(c.values.end(), sun.begin(), sun.end());
    float toLight[3];
    ASSERT_TRUE(FindLightDirection({c}, false, eye, toLight));
    EXPECT_NEAR(toLight[0], 0.29f / 1.0012f, 0.01f);
    EXPECT_NEAR(toLight[1], 0.77f / 1.0012f, 0.01f);
    EXPECT_NEAR(toLight[2], -0.56f / 1.0012f, 0.01f);
    // Rigid matrices scale their axes alike: never a light's, camera or not.
    ASSERT_TRUE(FindLightDirection({c}, false, nullptr, toLight));
    EXPECT_GT(toLight[1], 0.7f);
}

TEST(LapisLight, RefusesTheCamerasOwnView) {
    // A camera's view matrix maps the camera to the origin, well inside any map: it is rigid.
    const float eye[3] = {100.0f, 20.0f, -40.0f};
    Constants c;
    c.values = {0.8f, 0, -0.6f, -56.0f, 0, 1, 0, -20.0f, 0.6f, 0, 0.8f, -28.0f, 0, 0, 0, 1};
    float toLight[3];
    EXPECT_FALSE(FindLightDirection({c}, false, eye, toLight));
}

TEST(LapisLight, VoteKeepsTheSunOverACharactersLight) {
    const float eye[3] = {100.0f, 20.0f, -40.0f};
    Constants sun;
    sun.values = SunCascade(eye);
    // A character's own light: another direction, its map of a few draws.
    Constants own;
    own.values = SunCascade(eye, {0.6f, -0.5f, -0.62f});
    float toSun[3];
    float scales[3];
    LightVote vote;
    vote.add({sun}, false, eye, 90);
    vote.add({own}, false, eye, 20);
    ASSERT_TRUE(vote.settle(toSun, scales));
    EXPECT_GT(toSun[1], 0.7f);
    const float first[3] = {toSun[0], toSun[1], toSun[2]};
    // Frames where only the character's map has constants do not flip it at once...
    for (int i = 1; i < LightVote::kTakeOver; i++) {
        vote.add({own}, false, eye, 20);
        ASSERT_TRUE(vote.settle(toSun, scales));
        EXPECT_FLOAT_EQ(toSun[0], first[0]);
    }
    // ...nor do frames with none, for a while.
    for (int i = 0; i < LightVote::kHold; i++) {
        ASSERT_TRUE(vote.settle(toSun, scales));
        EXPECT_FLOAT_EQ(toSun[2], first[2]);
    }
    EXPECT_FALSE(vote.settle(toSun, scales));
}

TEST(LapisLight, VoteKeepsFarCascadesOfTheSameSun) {
    const float eye[3] = {100.0f, 20.0f, -40.0f};
    const float ahead[3] = {300.0f, 20.0f, -40.0f};  // a slice of the view past the camera
    Constants nearMap;
    nearMap.values = SunCascade(eye);
    Constants farMap;
    farMap.values = SunCascade(ahead);
    Constants otherFar;
    otherFar.values = SunCascade(ahead, {0.6f, -0.5f, -0.62f});
    LightVote vote;
    vote.add({nearMap}, false, eye, 50, 7);
    vote.add({farMap}, false, eye, 80, 8);
    vote.add({otherFar}, false, eye, 80, 9);
    float toSun[3];
    float scales[3];
    ASSERT_TRUE(vote.settle(toSun, scales));
    const auto maps = vote.cascades();
    ASSERT_EQ(maps.size(), 2u);
    EXPECT_TRUE((maps[0].image == 7 && maps[1].image == 8) ||
                (maps[0].image == 8 && maps[1].image == 7));
}

TEST(LapisLight, VoteKeepsCascadesSharingAnImage) {
    // Genshin redraws one cascade a frame into its own quarter of one image.
    const float eye[3] = {100.0f, 20.0f, -40.0f};
    const float ahead[3] = {300.0f, 20.0f, -40.0f};
    Constants nearMap;
    nearMap.values = SunCascade(eye);
    Constants farMap;
    farMap.values = SunCascade(ahead);
    const float left[4] = {0.0f, 0.0f, 0.5f, 0.5f};
    const float right[4] = {0.5f, 0.0f, 0.5f, 0.5f};
    LightVote vote;
    float toSun[3];
    float scales[3];
    vote.add({nearMap}, false, eye, 50, 7, false, left);
    ASSERT_TRUE(vote.settle(toSun, scales));
    vote.add({nearMap}, false, eye, 50, 3, false, nullptr);
    vote.add({farMap}, false, eye, 80, 7, false, right);
    ASSERT_TRUE(vote.settle(toSun, scales));
    const auto maps = vote.cascades();
    int inSeven = 0;
    for (const auto& m : maps) inSeven += m.image == 7;
    EXPECT_EQ(inSeven, 2) << "a quarter redrawn leaves the other quarters' cascades";
}

TEST(LapisLight, PlacesTheCamera) {
    // Genshin's frame camera stands where its view-projection's x, y and w planes meet.
    const std::vector<float> m = {1, 0, 0, -3, 0, 2, 0, 5, 0, 0, 0.1f, 1, 0, 0, -1, 7};
    float eye[3];
    ASSERT_TRUE(EyeOf(m.data(), eye));
    EXPECT_NEAR(eye[0], 3.0f, 1e-5f);
    EXPECT_NEAR(eye[1], -2.5f, 1e-5f);
    EXPECT_NEAR(eye[2], 7.0f, 1e-5f);
}

}  // namespace
}  // namespace framefx::lapis
