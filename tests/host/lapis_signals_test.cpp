/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include "lapis/classifier.h"
#include "lapis/signals.h"
#include "tests/host/lapis_fixture.h"

// The engine signals over the frame graphs the survey recorded in games, through the same bridge
// the recorder calls. The detectors' own cases are lapis/rust's tests.
namespace framefx::lapis {
namespace {

using test::LoadSurvey;
using test::PassStep;

SignalFrame FrameOf(const test::Fixture& f, const Roles& roles, int64_t now) {
    SignalFrame s;
    s.frame = &f.frame;
    s.images = &f.images;
    s.roles = roles;
    s.presentNs = now;
    const auto d = f.images.find(roles.depth);
    if (d != f.images.end()) {
        s.sceneWidth = d->second.width;
        s.sceneHeight = d->second.height;
    }
    return s;
}

TEST(LapisSignals, GenshinsPostChainAndTheColourBeforeIt) {
    const std::vector<test::Fixture> frames = LoadSurvey("genshin.log");
    ASSERT_FALSE(frames.empty());
    SignalFeed feed;
    int64_t now = 1'000'000'000;
    for (const test::Fixture& f : frames) {
        const Roles r = Classify(f.frame, f.images);
        const EngineSignals& g = feed.update(FrameOf(f, r, now += 16'666'667));
        EXPECT_EQ(g.tonemapStep, PassStep(f.frame, 27));
        // The TAA output, before bloom; the resolve ping-pongs between two images.
        const int taa = PassStep(f.frame, 19);
        EXPECT_EQ(g.prePostStep, taa);
        EXPECT_EQ(g.prePostImage, f.frame.steps[taa].pass.colors[0].image);
        EXPECT_EQ(g.prePostFlags & kPrePostUntoned, kPrePostUntoned);
        EXPECT_EQ(g.postEffects & (kPostTemporal | kPostBloom | kPostTonemap),
                  kPostTemporal | kPostBloom | kPostTonemap);
        EXPECT_EQ(g.sceneState, 1u);
        EXPECT_NEAR(g.renderScale[0], 2118.0f / 2119.0f, 1e-3f);
        // The survey keeps no constants: no camera, so nothing camera-made is claimed.
        EXPECT_EQ(g.cameraFlags, 0u);
        EXPECT_EQ(g.nextValid, 0u);
    }
    EXPECT_TRUE(feed.watch().empty());
    EXPECT_TRUE(feed.profile().empty());
}

TEST(LapisSignals, AFrameWithoutASceneIsFlatAndItsReturnIsACut) {
    const std::vector<test::Fixture> frames = LoadSurvey("genshin.log");
    ASSERT_FALSE(frames.empty());
    const test::Fixture& f = frames[0];
    const Roles r = Classify(f.frame, f.images);
    SignalFeed feed;
    for (int i = 0; i < 3; i++) EXPECT_EQ(feed.update(FrameOf(f, r, 1 + i)).cut, 0u);
    test::Fixture menu;
    menu.frame.presented = f.frame.presented;
    menu.frame.width = f.frame.width;
    menu.frame.height = f.frame.height;
    const EngineSignals& gone = feed.update(FrameOf(menu, Roles(), 10));
    EXPECT_EQ(gone.sceneState, 2u);
    EXPECT_EQ(gone.cut, 1u);
    EXPECT_EQ(feed.update(FrameOf(f, r, 11)).cut, 1u);
}

TEST(LapisSignals, ProfilesRefuseJunk) {
    SignalFeed feed;
    EXPECT_FALSE(feed.warm("junk"));
    EXPECT_TRUE(feed.warm("signals1 0000beef 0.1.16.0 9"));
}

}  // namespace
}  // namespace framefx::lapis
