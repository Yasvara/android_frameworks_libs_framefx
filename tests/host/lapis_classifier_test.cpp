/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <algorithm>

#include "lapis/classifier.h"
#include "tests/host/lapis_fixture.h"

namespace framefx::lapis {
namespace {

using test::CopyLastTargetToSwapchain;
using test::kSwapchain;
using test::LoadSurvey;
using test::PassStep;

std::vector<test::Fixture> Load(const std::string& name) {
    std::vector<test::Fixture> frames = LoadSurvey(name);
    EXPECT_FALSE(frames.empty()) << name << " in " << test::FixtureDir();
    return frames;
}

TEST(LapisClassifier, GenshinIsTierA) {
    for (const test::Fixture& f : Load("genshin.log")) {
        ASSERT_EQ(f.frame.presented, kSwapchain);
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.tier, Tier::kA);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_EQ(r.depth, 3000u);
        EXPECT_TRUE(r.depthStored);
        EXPECT_TRUE(r.reversed);
        EXPECT_EQ(r.scene, 3123u);
        EXPECT_EQ(r.ui, PassStep(f.frame, 28));
        EXPECT_EQ(r.velocity, 3113u);
        EXPECT_EQ(r.target, kSwapchain);
    }
}

TEST(LapisClassifier, WuwaIsTierAThroughItsBackBufferCopy) {
    for (test::Fixture& f : Load("wuwa.log")) {
        // Without the copy the presented image is never drawn.
        EXPECT_EQ(Classify(f.frame, f.images).miss, Miss::kNoFrame);
        CopyLastTargetToSwapchain(&f);
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.tier, Tier::kA);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_EQ(r.depth, 1534u);
        EXPECT_TRUE(r.depthStored);
        EXPECT_TRUE(r.reversed);
        // After the game's upscale, not the G-buffer colour the UI pass also samples.
        EXPECT_EQ(r.scene, 6098u);
        EXPECT_EQ(r.ui, PassStep(f.frame, 47));
        EXPECT_EQ(r.velocity, 1539u);
        EXPECT_EQ(r.target, 6114u);
    }
}

TEST(LapisClassifier, MlbbNeedsTierB) {
    for (const test::Fixture& f : Load("mlbb.log")) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.tier, Tier::kB);
        EXPECT_EQ(r.miss, Miss::kNoScene);
        EXPECT_EQ(r.depth, 128u);
        EXPECT_FALSE(r.depthStored);  // DONT_CARE: never leaves the tile memory
        EXPECT_FALSE(r.reversed);
        EXPECT_EQ(r.scene, 0u);
        EXPECT_EQ(r.velocity, 0u);
    }
}

// The pass that draws into the swapchain, which Delta Force's tonemap and HUD share.
int SwapchainPass(const Frame& frame) {
    for (int i = 0; i < static_cast<int>(frame.steps.size()); i++) {
        if (frame.steps[i].isPass && frame.steps[i].pass.writes(kSwapchain)) return i;
    }
    return -1;
}

TEST(LapisClassifier, DeltaForceTonemapsStraightIntoTheSwapchain) {
    for (const test::Fixture& f : Load("deltaforce.log")) {
        ASSERT_EQ(f.frame.presented, kSwapchain);
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kUntoned);
        EXPECT_EQ(r.scene, 0u);
        EXPECT_EQ(r.untoned, 1543u);
        EXPECT_EQ(r.split, SwapchainPass(f.frame));
        EXPECT_EQ(r.depth, 1544u);
        EXPECT_FALSE(r.depthStored);  // MSAA, transient, never leaves the tile memory
        EXPECT_EQ(r.tier, Tier::kB);
    }
}

TEST(LapisClassifier, RobloxConvertsItsDeeperSceneInThePresentingPass) {
    // Its 3D in 10 bits, drawn into the 8-bit swapchain by the presenting pass's first draw with
    // the game's last colour step, then its HUD: the split goes after that draw.
    const auto frames = Load("roblox.log");
    for (const test::Fixture& f : frames) {
        ASSERT_EQ(f.frame.presented, kSwapchain);
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kUntoned);
        EXPECT_EQ(r.scene, 0u);
        EXPECT_EQ(r.untoned, 507u);
        EXPECT_EQ(r.split, SwapchainPass(f.frame));
        EXPECT_EQ(r.depth, 505u);
        EXPECT_FALSE(r.depthStored);  // MSAA, transient
    }
}

// Delta Force's frame as Lapis splits it: the tonemap's half, the copy of the target, and the
// HUD's half loading it.
test::Fixture SplitDeltaForce(test::Fixture f, ImageId scene) {
    const int at = SwapchainPass(f.frame);
    Step copy;
    copy.op.kind = OpKind::kCopy;
    copy.op.src = kSwapchain;
    copy.op.dst = scene;
    Step hud = f.frame.steps[at];
    for (Target& t : hud.pass.colors) t.load = VK_ATTACHMENT_LOAD_OP_LOAD;
    hud.pass.reads.erase(std::remove(hud.pass.reads.begin(), hud.pass.reads.end(), 1543u),
                         hud.pass.reads.end());
    f.frame.steps.insert(f.frame.steps.begin() + at + 1, {copy, hud});
    Image& image = f.images[scene];
    image.id = scene;
    image.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    image.width = f.frame.width;
    image.height = f.frame.height;
    return f;
}

TEST(LapisClassifier, DeltaForceSplitOffersTheSceneBeforeItsHud) {
    for (const test::Fixture& surveyed : Load("deltaforce.log")) {
        const test::Fixture f = SplitDeltaForce(surveyed, 9000);
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_EQ(r.scene, 9000u);
        EXPECT_EQ(r.ui, SwapchainPass(surveyed.frame) + 2);
        EXPECT_EQ(r.split, -1);
        EXPECT_EQ(r.target, kSwapchain);
    }
}

TEST(LapisClassifier, ACopyOfAHalfSizeTargetIsNotTheScene) {
    test::Fixture f = SplitDeltaForce(Load("deltaforce.log")[0], 9000);
    f.images[9000].width /= 2;
    EXPECT_EQ(Classify(f.frame, f.images).scene, 0u);
}

TEST(LapisClassifier, WuwaWithItsOwnFrameGenerationIsNamed) {
    std::vector<test::Fixture> frames = Load("wuwa-afme.log");
    ASSERT_EQ(frames.size(), 2u);
    test::Fixture& f = frames[1];  // frame 0 is the short generated present
    CopyLastTargetToSwapchain(&f);
    const Roles r = Classify(f.frame, f.images);
    EXPECT_TRUE(r.gameGenerates);
    EXPECT_NE(r.scene, 0u);
    EXPECT_EQ(r.miss, Miss::kNone);
}

// MH Outlander draws upright, then turns the finished frame into a pre-rotated swapchain with one
// more pass. That frame has the HUD: the scene is what the pass before the blit started from.
TEST(LapisClassifier, MhoSeesThroughTheBlitToItsSwapchain) {
    std::vector<test::Fixture> frames = Load("mho.log");
    ASSERT_EQ(frames.size(), 2u);
    for (const test::Fixture& f : frames) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_FALSE(r.direct);
        ASSERT_NE(r.target, 0u);
        const Image& target = f.images.at(r.target);
        EXPECT_EQ(target.width, 2336u);
        EXPECT_EQ(target.height, 1080u);
        ASSERT_NE(r.scene, 0u);
        EXPECT_LT(f.images.at(r.scene).width, 1600u);
    }
}

// Uma draws its HUD over the 3D scene, then blits that through a multisampled pass resolved into
// the image a last pass blits to the swapchain: S is the scene the HUD pass samples. Its scene keeps
// 720 wide at any render scale; only the HUD follows (uma-small: 291x629).
TEST(LapisClassifier, UmaSeesThroughItsMultisampledBlit) {
    for (const char* name : {"uma.log", "uma-small.log"}) {
    for (const test::Fixture& f : Load(name)) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_TRUE(r.behindBlit);
        ASSERT_NE(r.scene, 0u);
        EXPECT_EQ(f.images.at(r.scene).width, 720u) << name;
        EXPECT_GE(f.images.at(r.scene).height, 1556u) << name;
    }
    }
}

// In a career Uma's HUD pass loads its target instead of starting it over: the scene it samples
// is still the scene, being drawn from the 3D.
TEST(LapisClassifier, UmaHudPassThatLoadsItsTargetStillShowsTheScene) {
    const std::vector<test::Fixture> frames = Load("uma-career.log");
    ASSERT_FALSE(frames.empty());
    for (const test::Fixture& f : frames) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_TRUE(r.behindBlit);
        EXPECT_FALSE(r.finished);
        ASSERT_NE(r.scene, 0u);
        EXPECT_EQ(f.images.at(r.scene).width, 720u);
    }
}

// Started at a quarter of the window and grown since, Uma keeps its HUD at the size it started
// at and stretches the frame in its last pass: the scene is still behind that.
TEST(LapisClassifier, UmaStretchingItsFrameStillShowsTheScene) {
    const std::vector<test::Fixture> frames = Load("uma-grown.log");
    ASSERT_FALSE(frames.empty());
    for (const test::Fixture& f : frames) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_TRUE(r.behindBlit);
        ASSERT_NE(r.scene, 0u);
        EXPECT_EQ(f.images.at(r.scene).width, 720u);
    }
}

// Endfield draws its scene upright and its HUD turned, into a pre-rotated swapchain or into an
// image copied to it, a frame late: the depth is the one the scene's own passes wrote.
TEST(LapisClassifier, EndfieldFindsTheDepthItsSceneWasDrawnWith) {
    const std::vector<test::Fixture> frames = Load("endfield.log");
    ASSERT_EQ(frames.size(), 2u);
    // The second present only copies what the first drew: the recorder hands both over as one.
    test::Fixture late = frames[1];
    late.frame.steps.insert(late.frame.steps.begin(), frames[0].frame.steps.begin(),
                            frames[0].frame.steps.end());
    late.images.insert(frames[0].images.begin(), frames[0].images.end());
    late.frame.presented = kSwapchain;
    for (const test::Fixture& f : {frames[0], late}) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        ASSERT_NE(r.scene, 0u);
        EXPECT_EQ(f.images.at(r.scene).width, 2244u);
        ASSERT_NE(r.depth, 0u);
        EXPECT_TRUE(r.depthStored);
        EXPECT_EQ(f.images.at(r.depth).width, 2244u);
        EXPECT_EQ(f.images.at(r.depth).height, 1036u);
    }
}

// Endfield's HUD pass at times also samples a G-buffer image left bound: the scene is the image
// furthest down the frame, its last pass only loading it.
TEST(LapisClassifier, EndfieldTakesTheSceneOverItsGBuffer) {
    const std::vector<test::Fixture> frames = Load("endfield-stale.log");
    ASSERT_EQ(frames.size(), 1u);
    const Roles r = Classify(frames[0].frame, frames[0].images);
    EXPECT_EQ(r.miss, Miss::kNone);
    ASSERT_NE(r.scene, 0u);
    EXPECT_EQ(frames[0].images.at(r.scene).format, VK_FORMAT_R8G8B8A8_UNORM);
    EXPECT_NE(r.scene, r.velocity);
    EXPECT_TRUE(r.depthStored);
}

// A copy of `image` taken right after step `after`, into a new image of its shape.
ImageId CopyAfter(test::Fixture* f, ImageId image, int after) {
    const ImageId copy = 9100;
    f->images[copy] = f->images.at(image);
    f->images[copy].id = copy;
    Step step;
    step.op.kind = OpKind::kCopy;
    step.op.src = image;
    step.op.dst = copy;
    f->frame.steps.insert(f->frame.steps.begin() + after + 1, step);
    return copy;
}

// ANGLE renders a GL game's multisampled scene into attachments it never stores and resolves
// colour and depth out of them. Uma's later pass brings both back in and resolves again, which
// leaves the depth image in pieces: D is the resolved image, and not kept.
TEST(LapisClassifier, AngleDepthIsTheOneItsScenePassResolvesInto) {
    const std::vector<test::Fixture> frames = Load("uma-msaa.log");
    ASSERT_FALSE(frames.empty());
    for (const test::Fixture& f : frames) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        ASSERT_NE(r.depth, 0u);
        EXPECT_FALSE(r.depthStored);
        EXPECT_EQ(r.tier, Tier::kB);
        EXPECT_FALSE(r.reversed);
        EXPECT_EQ(f.images.at(r.depth).width, 720u);
        EXPECT_EQ(f.images.at(r.depth).format, VK_FORMAT_D24_UNORM_S8_UINT);
        ASSERT_GE(r.geometry, 0);
        const Pass& scene = f.frame.steps[r.geometry].pass;
        EXPECT_GT(scene.depthWrites, 0u);
        EXPECT_GT(scene.depth.samples, VK_SAMPLE_COUNT_1_BIT);
        // The image the scene's pass resolves into, never the multisampled attachment.
        EXPECT_NE(scene.depth.image, r.depth);
        EXPECT_TRUE(scene.writes(r.depth));
    }
}

// Copied out as the scene's pass ends, that depth is this frame's whatever the game does after.
TEST(LapisClassifier, AngleDepthCopiedAsItsPassEndsIsKept) {
    std::vector<test::Fixture> frames = Load("uma-msaa.log");
    ASSERT_FALSE(frames.empty());
    for (test::Fixture& f : frames) {
        const Roles before = Classify(f.frame, f.images);
        ASSERT_GE(before.geometry, 0);
        const ImageId copy = CopyAfter(&f, before.depth, before.geometry);
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.depth, copy);
        EXPECT_TRUE(r.depthStored);
        EXPECT_EQ(r.tier, Tier::kA);
        EXPECT_EQ(r.geometry, before.geometry);
        EXPECT_EQ(r.scene, before.scene);
    }
}

// A copy taken only after the game resolved into the image again holds what that left.
TEST(LapisClassifier, AngleDepthCopiedTooLateIsNotKept) {
    std::vector<test::Fixture> frames = Load("uma-msaa.log");
    ASSERT_FALSE(frames.empty());
    test::Fixture f = frames[0];
    const Roles before = Classify(f.frame, f.images);
    CopyAfter(&f, before.depth, static_cast<int>(f.frame.steps.size()) - 1);
    const Roles r = Classify(f.frame, f.images);
    EXPECT_EQ(r.depth, before.depth);
    EXPECT_FALSE(r.depthStored);
}

// Without a later pass over it, the resolved depth is the scene's as it stands.
TEST(LapisClassifier, AngleDepthNothingResolvesIntoAgainIsKept) {
    std::vector<test::Fixture> frames = Load("uma-msaa.log");
    ASSERT_FALSE(frames.empty());
    test::Fixture f = frames[0];
    const Roles before = Classify(f.frame, f.images);
    ASSERT_GE(before.geometry, 0);
    auto& steps = f.frame.steps;
    for (size_t i = steps.size(); i-- > static_cast<size_t>(before.geometry) + 1;) {
        if (steps[i].isPass && steps[i].pass.writes(before.depth)) steps.erase(steps.begin() + i);
    }
    const Roles r = Classify(f.frame, f.images);
    EXPECT_EQ(r.depth, before.depth);
    EXPECT_TRUE(r.depthStored);
}

// A depth the scene's pass stores and a later pass clears for its own draws is kept the same way.
TEST(LapisClassifier, DepthALaterPassStartsOverIsKeptThroughACopy) {
    std::vector<test::Fixture> frames = Load("uma-msaa.log");
    ASSERT_FALSE(frames.empty());
    test::Fixture f = frames[0];
    const Roles first = Classify(f.frame, f.images);
    ASSERT_GE(first.geometry, 0);
    // The scene's pass, single-sampled and storing its depth; the game's last pass clears it.
    Pass& scene = f.frame.steps[first.geometry].pass;
    const ImageId depth = scene.depth.image;
    scene.depth.samples = VK_SAMPLE_COUNT_1_BIT;
    scene.depth.store = VK_ATTACHMENT_STORE_OP_STORE;
    for (Target& c : scene.colors) c.samples = VK_SAMPLE_COUNT_1_BIT;
    Step over;
    over.isPass = true;
    over.pass.hasDepth = true;
    over.pass.depth = scene.depth;
    over.pass.depth.load = VK_ATTACHMENT_LOAD_OP_CLEAR;
    over.pass.width = scene.width;
    over.pass.height = scene.height;
    f.frame.steps.push_back(over);
    const Roles lost = Classify(f.frame, f.images);
    EXPECT_EQ(lost.depth, depth);
    EXPECT_FALSE(lost.depthStored);
    const ImageId copy = CopyAfter(&f, depth, lost.geometry);
    const Roles r = Classify(f.frame, f.images);
    EXPECT_EQ(r.depth, copy);
    EXPECT_TRUE(r.depthStored);
}

// Uma as it runs with the copy armed: D is the copy, taken between the scene's pass and the one
// that resolves into its depth again.
TEST(LapisClassifier, UmaWithItsDepthKeptIsTierA) {
    const std::vector<test::Fixture> frames = Load("uma-kept.log");
    ASSERT_FALSE(frames.empty());
    for (const test::Fixture& f : frames) {
        const Roles r = Classify(f.frame, f.images);
        EXPECT_EQ(r.miss, Miss::kNone);
        EXPECT_EQ(r.tier, Tier::kA);
        EXPECT_TRUE(r.depthStored);
        ASSERT_NE(r.depth, 0u);
        EXPECT_EQ(f.images.at(r.depth).format, VK_FORMAT_D24_UNORM_S8_UINT);
        for (const Step& s : f.frame.steps) {
            EXPECT_FALSE(s.isPass && s.pass.writes(r.depth));  // no pass of the game's draws it
        }
    }
}

// Every frame surveyed from a game on ANGLE, at any render scale: a scene, and one run of frames.
TEST(LapisTracker, AngleFramesEngageAtEveryScale) {
    for (const char* name :
         {"uma.log", "uma-small.log", "uma-career.log", "uma-grown.log", "uma-msaa.log",
          "uma-kept.log"}) {
        const std::vector<test::Fixture> frames = Load(name);
        ASSERT_FALSE(frames.empty()) << name;
        Tracker t;
        for (int i = 0; i < Tracker::kEngageFrames; i++) {
            const test::Fixture& f = frames[i % frames.size()];
            const Roles r = Classify(f.frame, f.images);
            EXPECT_EQ(r.miss, Miss::kNone) << name;
            ASSERT_NE(r.scene, 0u) << name;
            EXPECT_EQ(f.images.at(r.scene).width, 720u) << name;
            t.update(r, f.frame.width, f.frame.height);
        }
        EXPECT_TRUE(t.engaged()) << name;
    }
}

// Star Rail tonemaps into a pre-rotated swapchain and draws the HUD in the same pass, its depth
// cleared and thrown away: the split's case.
TEST(LapisClassifier, HsrTonemapsStraightIntoItsSwapchain) {
    const std::vector<test::Fixture> frames = Load("hsr.log");
    ASSERT_FALSE(frames.empty());
    const Roles r = Classify(frames[0].frame, frames[0].images);
    EXPECT_EQ(r.miss, Miss::kUntoned);
    EXPECT_TRUE(r.direct);
    EXPECT_GE(r.split, 0);
    EXPECT_NE(r.untoned, 0u);
}

TEST(LapisClassifier, ABlitWithNothingBehindItIsTheHudlessFrame) {
    // Genshin without a HUD: its last pass only stretches S, and what builds S samples its scene
    // before the tonemap. S stays.
    test::Fixture f = Load("genshin.log")[0];
    Roles before = Classify(f.frame, f.images);
    ASSERT_NE(before.scene, 0u);
    f.frame.steps[before.ui].pass.draws = 1;
    const std::vector<ImageId> reads = {before.scene};
    f.frame.steps[before.ui].pass.reads = reads;
    EXPECT_EQ(Classify(f.frame, f.images).scene, before.scene);
}

// Wuthering Waves can draw its HUD into a layer of its own and put it over the finished scene with
// one more pass: the layer is the newest image the target's passes sample, and not the scene.
TEST(LapisClassifier, WuwaHudLayerIsNotTheScene) {
    for (test::Fixture& f : Load("wuwa-uilayer.log")) {
        if (f.frame.presented == 0) CopyLastTargetToSwapchain(&f);
        // The short frames between are the game's own, with no scene at all.
        const bool real = f.frame.steps.size() > 100;
        EXPECT_EQ(Classify(f.frame, f.images).scene, real ? 1636u : 0u);
    }
}

// MLBB draws everything in one pass and its engine clears depth when it switches to the HUD's
// camera.
TEST(LapisClassifier, MlbbSplitsAtItsDepthClear) {
    test::Fixture f = Load("mlbb.log")[0];
    Pass& p = f.frame.steps[PassStep(f.frame, 0)].pass;
    EXPECT_FALSE(Classify(f.frame, f.images).clearSplit);
    p.depthClears = 1;
    p.lastClearDraw = p.draws / 2;
    const Roles r = Classify(f.frame, f.images);
    EXPECT_TRUE(r.clearSplit);
    EXPECT_EQ(r.split, PassStep(f.frame, 0));
    EXPECT_EQ(r.miss, Miss::kNoScene);
    p.lastClearDraw = 0;  // a clear before any draw splits nothing
    EXPECT_FALSE(Classify(f.frame, f.images).clearSplit);
}

TEST(LapisClassifier, MenuFrameHasNoScene) {
    test::Fixture f = Load("genshin.log")[0];
    // Only the UI pass is left, drawing over textures.
    std::vector<Step> kept;
    for (const Step& s : f.frame.steps) {
        if (s.isPass && s.pass.writes(kSwapchain)) kept.push_back(s);
    }
    ASSERT_EQ(kept.size(), 1u);
    f.frame.steps = kept;
    const Roles r = Classify(f.frame, f.images);
    EXPECT_EQ(r.scene, 0u);
    EXPECT_EQ(r.depth, 0u);
    EXPECT_EQ(r.tier, Tier::kC);
    EXPECT_EQ(r.miss, Miss::kNoScene);
}

TEST(LapisClassifier, DepthClearedAfterTheSceneIsNotKept) {
    test::Fixture f = Load("genshin.log")[0];
    Step clear;
    clear.op.kind = OpKind::kClear;
    clear.op.dst = 3000;
    clear.op.depth = 0.0f;
    const int ui = PassStep(f.frame, 28);
    f.frame.steps.insert(f.frame.steps.begin() + ui, clear);
    const Roles r = Classify(f.frame, f.images);
    EXPECT_EQ(r.depth, 3000u);
    EXPECT_FALSE(r.depthStored);
    EXPECT_EQ(r.tier, Tier::kB);
}

TEST(LapisClassifier, DepthDiscardedIsNotKept) {
    test::Fixture f = Load("genshin.log")[0];
    f.frame.steps[PassStep(f.frame, 4)].pass.depth.store = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    const Roles r = Classify(f.frame, f.images);
    EXPECT_FALSE(r.depthStored);
    EXPECT_EQ(r.tier, Tier::kB);
}

TEST(LapisClassifier, SceneMustCoverTheSceneDepth) {
    test::Fixture f = Load("genshin.log")[0];
    f.images[3123].width = 1059;
    f.images[3123].height = 489;
    EXPECT_EQ(Classify(f.frame, f.images).scene, 0u);
}

TEST(LapisClassifier, SceneBeforeItsTonemapIsNotTheScene) {
    test::Fixture f = Load("genshin.log")[0];
    // As in Unreal's mobile renderer: the tonemap pass samples the float scene and draws into the
    // swapchain, which the HUD then loads.
    f.images[3123].format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    const Roles r = Classify(f.frame, f.images);
    EXPECT_EQ(r.scene, 0u);
    EXPECT_EQ(r.miss, Miss::kUntoned);
    EXPECT_EQ(r.depth, 3000u);
}

TEST(LapisClassifier, ShadowMapsAreNotSceneDepth) {
    test::Fixture f = Load("genshin.log")[0];
    // Drop the G-buffer's depth writes: what is left is shadows and half-res passes.
    f.frame.steps[PassStep(f.frame, 4)].pass.depthWrites = 0;
    const Roles r = Classify(f.frame, f.images);
    EXPECT_NE(r.depth, 3098u);  // D16 1024x1024 cascade
    EXPECT_NE(r.depth, 4525u);  // D16 512x512
}

TEST(LapisClassifier, NothingPresentedIsNoFrame) {
    Frame frame;
    frame.presented = kSwapchain;
    EXPECT_EQ(Classify(frame, {}).miss, Miss::kNoFrame);
    EXPECT_EQ(Classify(frame, {}).tier, Tier::kNone);
}

Roles TierA(ImageId depth, ImageId scene) {
    Roles r;
    r.depth = depth;
    r.scene = scene;
    r.target = kSwapchain;
    r.depthStored = true;
    r.tier = Tier::kA;
    r.miss = Miss::kNone;
    return r;
}

TEST(LapisTracker, EngagesAfterAStableRun) {
    Tracker t;
    for (int i = 1; i < Tracker::kEngageFrames; i++) {
        EXPECT_FALSE(t.update(TierA(1, 2), 100, 50)) << i;
    }
    EXPECT_TRUE(t.update(TierA(1, 2), 100, 50));
    EXPECT_EQ(t.lastMiss(), Miss::kNone);
    EXPECT_EQ(t.roles().scene, 2u);
}

TEST(LapisTracker, LetsGoAtTheFirstChange) {
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(TierA(1, 2), 100, 50);
    ASSERT_TRUE(t.engaged());
    EXPECT_FALSE(t.update(TierA(1, 4), 100, 50));
    EXPECT_EQ(t.lastMiss(), Miss::kChanged);
    // A game that was read needs its new roles to agree once.
    EXPECT_TRUE(t.update(TierA(1, 4), 100, 50));
}

TEST(LapisTracker, LearnsFromScratchWhenItWasNotEngaged) {
    Tracker t;
    for (int i = 0; i < 3; i++) t.update(TierA(1, 2), 100, 50);
    for (int i = 1; i < Tracker::kEngageFrames; i++) EXPECT_FALSE(t.update(TierA(1, 4), 100, 50));
    EXPECT_TRUE(t.update(TierA(1, 4), 100, 50));
}

TEST(LapisTracker, FramesOfOneStructureAreOneRun) {
    Tracker t;
    // Images rotate freely under one signature.
    for (int i = 0; i < 3 * Tracker::kEngageFrames; i++) {
        Roles r = TierA(10 + i % 3, 20 + i % 5);
        r.signature = 7;
        t.update(r, 100, 50);
    }
    EXPECT_TRUE(t.engaged());
    Roles other = TierA(1, 2);
    other.signature = 9;
    EXPECT_FALSE(t.update(other, 100, 50));
    EXPECT_EQ(t.lastMiss(), Miss::kChanged);
}

TEST(LapisTracker, ToleratesPingPongedImages) {
    Tracker t;
    for (int i = 0; i < 2 * Tracker::kEngageFrames; i++) {
        t.update(TierA(1, i % 2 == 0 ? 2 : 3), 100, 50);
    }
    EXPECT_TRUE(t.engaged());
    // A third image breaks the pattern.
    EXPECT_FALSE(t.update(TierA(1, 9), 100, 50));
}

Roles Menu() {
    Roles menu;
    menu.tier = Tier::kC;
    menu.miss = Miss::kNoScene;
    return menu;
}

TEST(LapisTracker, RidesOutAFewFramesWithoutTheScene) {
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(TierA(1, 2), 100, 50);
    for (int i = 0; i < Tracker::kGraceFrames; i++) {
        EXPECT_FALSE(t.update(Menu(), 100, 50)) << "never tapped";
        EXPECT_TRUE(t.engaged());
    }
    EXPECT_TRUE(t.update(TierA(1, 2), 100, 50));
    EXPECT_EQ(t.lastMiss(), Miss::kNone);
}

TEST(LapisTracker, RidesOutAFewFramesThatOnlyShowTheirBlit) {
    Roles r = TierA(1, 2);
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(r, 100, 50);
    ASSERT_TRUE(t.engaged());
    Roles blit = r;
    blit.scene = 99;
    blit.finished = true;
    EXPECT_FALSE(t.update(blit, 100, 50));
    EXPECT_TRUE(t.engaged());
    EXPECT_TRUE(t.update(r, 100, 50));
    for (int i = 0; i <= Tracker::kGraceFrames; i++) t.update(blit, 100, 50);
    EXPECT_FALSE(t.engaged());
}

TEST(LapisTracker, ScenesFoundBehindABlitMayCycle) {
    Tracker t;
    Roles r = TierA(1, 2);
    r.behindBlit = true;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(r, 100, 50);
    ASSERT_TRUE(t.engaged());
    for (ImageId scene : {3u, 4u, 5u, 2u}) {
        r.scene = scene;
        EXPECT_TRUE(t.update(r, 100, 50));
    }
}

TEST(LapisTracker, LearnsAcrossFramesWithoutTheScene) {
    Tracker t;
    for (int i = 0; i < 2 * Tracker::kEngageFrames; i++) {
        t.update(i % 2 == 0 ? TierA(1, 2) : Menu(), 100, 50);
    }
    EXPECT_TRUE(t.engaged());
}

TEST(LapisTracker, MenuDisengagesAfterTheGrace) {
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(TierA(1, 2), 100, 50);
    for (int i = 0; i <= Tracker::kGraceFrames; i++) EXPECT_FALSE(t.update(Menu(), 100, 50));
    EXPECT_FALSE(t.engaged());
    EXPECT_EQ(t.lastMiss(), Miss::kNoScene);
    EXPECT_FALSE(t.update(TierA(1, 2), 100, 50));
}

TEST(LapisTracker, DepthComingAndGoingKeepsTheScene) {
    // Genshin: some frames end with D discarded, which makes them tier B.
    Tracker t;
    for (int i = 0; i < 3 * Tracker::kEngageFrames; i++) {
        Roles r = TierA(1, 2);
        if (i % 5 == 4) {
            r.depthStored = false;
            r.tier = Tier::kB;
        }
        if (i % 7 == 6) r.velocity = 0;
        const bool tapped = t.update(r, 100, 50);
        if (i >= Tracker::kEngageFrames) EXPECT_TRUE(tapped) << i;
    }
}

TEST(LapisTracker, VelocityMovingKeepsTheRun) {
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(TierA(1, 2), 100, 50);
    Roles r = TierA(1, 2);
    r.velocity = 7;
    EXPECT_TRUE(t.update(r, 100, 50));
    r.velocity = 8;
    EXPECT_TRUE(t.update(r, 100, 50));
}

TEST(LapisTracker, ADifferentDepthIsAChange) {
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(TierA(1, 2), 100, 50);
    t.update(TierA(1, 2), 100, 50);
    EXPECT_FALSE(t.update(TierA(7, 2), 100, 50));
    EXPECT_EQ(t.lastMiss(), Miss::kChanged);
}

TEST(LapisTracker, TierBNeverEngagesYet) {
    Tracker t;
    Roles r = TierA(1, 0);
    r.tier = Tier::kB;
    r.miss = Miss::kNoScene;
    for (int i = 0; i < 2 * Tracker::kEngageFrames; i++) EXPECT_FALSE(t.update(r, 100, 50));
}

TEST(LapisTracker, GameFrameGenerationNeverEngages) {
    Tracker t;
    Roles r = TierA(1, 2);
    r.miss = Miss::kGameGenerates;
    for (int i = 0; i < 2 * Tracker::kEngageFrames; i++) EXPECT_FALSE(t.update(r, 100, 50));
    EXPECT_EQ(t.lastMiss(), Miss::kGameGenerates);
}

TEST(LapisTracker, ResizeRelearns) {
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) t.update(TierA(1, 2), 100, 50);
    ASSERT_TRUE(t.engaged());
    EXPECT_FALSE(t.update(TierA(1, 2), 200, 100));
    EXPECT_EQ(t.lastMiss(), Miss::kResized);
    for (int i = 2; i < Tracker::kEngageFrames; i++) EXPECT_FALSE(t.update(TierA(1, 2), 200, 100));
    EXPECT_TRUE(t.update(TierA(1, 2), 200, 100));
}

TEST(LapisTracker, EngagesWhileTheSwapchainImageRotates) {
    std::vector<test::Fixture> frames = Load("genshin.log");
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) {
        test::Fixture f = frames[i % frames.size()];
        // Each present shows the next of three swapchain images.
        const ImageId shown = kSwapchain - 1 - i % 3;
        for (Step& s : f.frame.steps) {
            if (!s.isPass) continue;
            for (Target& c : s.pass.colors) {
                if (c.image == kSwapchain) c.image = shown;
            }
        }
        f.images[shown] = f.images[kSwapchain];
        f.images[shown].id = shown;
        f.frame.presented = shown;
        const Roles r = Classify(f.frame, f.images);
        EXPECT_TRUE(r.direct);
        t.update(r, f.frame.width, f.frame.height);
    }
    EXPECT_TRUE(t.engaged());
}

TEST(LapisTracker, DeltaForceEngagesOnceSplit) {
    std::vector<test::Fixture> frames = Load("deltaforce.log");
    Tracker t;
    for (int i = 0; i < Tracker::kEngageFrames; i++) {
        const test::Fixture f = frames[i % frames.size()];
        EXPECT_FALSE(t.update(Classify(f.frame, f.images), f.frame.width, f.frame.height));
    }
    EXPECT_EQ(t.lastMiss(), Miss::kUntoned);
    for (int i = 0; i < Tracker::kEngageFrames; i++) {
        const test::Fixture f = SplitDeltaForce(frames[i % frames.size()], 9000);
        t.update(Classify(f.frame, f.images), f.frame.width, f.frame.height);
    }
    EXPECT_TRUE(t.engaged());
    EXPECT_EQ(t.roles().scene, 9000u);
}

TEST(LapisTracker, SurveyedGamesEngage) {
    for (const char* name : {"genshin.log", "wuwa.log", "mho.log"}) {
        std::vector<test::Fixture> frames = Load(name);
        Tracker t;
        for (int i = 0; i < Tracker::kEngageFrames; i++) {
            test::Fixture f = frames[i % frames.size()];
            if (f.frame.presented == 0) CopyLastTargetToSwapchain(&f);
            t.update(Classify(f.frame, f.images), f.frame.width, f.frame.height);
        }
        EXPECT_TRUE(t.engaged()) << name;
    }
}

}  // namespace
}  // namespace framefx::lapis
