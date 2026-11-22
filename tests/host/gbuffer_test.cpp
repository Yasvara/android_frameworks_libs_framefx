/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <cmath>

#include "luwes/gbuffer.h"

namespace framefx::luwes {
namespace {

// Reversed infinite perspective, row-major, looking down -z.
void Perspective(float fovY, float aspect, float near, float* m) {
    std::fill(m, m + 16, 0.0f);
    const float sy = 1.0f / std::tan(fovY * static_cast<float>(M_PI) / 360.0f);
    m[0] = sy / aspect;
    m[5] = sy;
    m[11] = near;
    m[14] = -1.0f;
}

GBufferInput Genshin() {
    GBufferInput in;
    in.sceneWidth = in.depthWidth = in.velocityWidth = 1662;
    in.sceneHeight = in.depthHeight = in.velocityHeight = 768;
    in.depthFormat = 129;    // D24_UNORM_S8_UINT
    in.velocityFormat = 64;  // A2B10G10R10
    in.hasScene = in.hasDepth = in.depthSampled = in.reversed = true;
    in.hasVelocity = in.temporal = in.cameraKnown = in.cameraMatrices = in.cameraWorld = true;
    in.clearDepth = 0.0f;
    Perspective(45.0f, 1662.0f / 768.0f, 0.25f, in.vp);
    in.sx = in.vp[0];
    in.sy = in.vp[5];
    in.signature = GBufferSignature(in);
    return in;
}

TEST(GBuffer, SettledChannelsAreTrustedAndAStillCameraReprojectsToIdentity) {
    GBufferState state;
    const GBufferInput in = Genshin();
    GBufferOutput out;
    for (int i = 0; i < 40; i++) out = GBufferUpdate(&state, in, false);
    EXPECT_FLOAT_EQ(out.depth, 1.0f);
    EXPECT_FLOAT_EQ(out.velocity, 1.0f);
    EXPECT_FLOAT_EQ(out.camera, 1.0f);
    ASSERT_TRUE(out.reprojectValid);
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) EXPECT_NEAR(out.reproject[r * 4 + c], r == c ? 1.0f : 0.0f, 1e-5);
    }
    EXPECT_EQ(out.range, DepthRangeKind::kReversed);
}

TEST(GBuffer, ACameraRelativeViewProjectionIsOnlyPartlyTrusted) {
    GBufferState state;
    GBufferInput in = Genshin();
    in.cameraWorld = false;
    GBufferOutput out;
    for (int i = 0; i < 40; i++) out = GBufferUpdate(&state, in, false);
    EXPECT_TRUE(out.reprojectValid);
    EXPECT_LT(out.camera, 0.8f);  // never fed to AFME
}

TEST(GBuffer, SignatureFollowsShapesNotHandles) {
    GBufferInput a = Genshin();
    GBufferInput b = a;
    b.jitter[0] = 0.001f;
    EXPECT_EQ(GBufferSignature(a), GBufferSignature(b));
    b.depthWidth = 1124;
    EXPECT_NE(GBufferSignature(a), GBufferSignature(b));
    EXPECT_NE(GBufferSignature(GBufferInput()), 0u);
}

TEST(GBuffer, ProfileRoundTripsAndWarms) {
    const GBufferInput in = Genshin();
    const std::string line = GBufferProfileLine(in);
    EXPECT_EQ(line.rfind("gbuffer1 ", 0), 0u);
    GBufferState warm;
    ASSERT_TRUE(GBufferWarm(&warm, line));
    EXPECT_GT(GBufferUpdate(&warm, in, false).depth, 0.75f);
    GBufferState cold;
    EXPECT_LT(GBufferUpdate(&cold, in, false).depth, 0.6f);
    EXPECT_FALSE(GBufferWarm(&cold, "nonsense"));
}

TEST(GBuffer, FeedFollowsThePropertyWords) {
    GBufferFeed f;
    f.configure("com.miHoYo.GenshinImpact,afme", "com.miHoYo.GenshinImpact");
    EXPECT_TRUE(f.on() && f.feedsAfme() && !f.inverse());
    f.configure("com.mobile.legends,inverse", "com.mobile.legends:UnityKillsMe");
    EXPECT_TRUE(f.on() && !f.feedsAfme() && f.inverse());
    f.configure("", "com.mobile.legends");
    EXPECT_FALSE(f.on());
}

TEST(GBuffer, InvertsAndRefusesSingular) {
    float m[16];
    Perspective(60.0f, 2.0f, 0.1f, m);
    float i[16];
    ASSERT_TRUE(Invert4x4(m, i));
    EXPECT_NEAR(i[0] * m[0], 1.0f, 1e-5);
    const float zero[16] = {};
    EXPECT_FALSE(Invert4x4(zero, i));
}

}  // namespace
}  // namespace framefx::luwes
