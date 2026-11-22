/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

// Luwes on this device's GPU, through the same entry points the graphics loaders call. Frames go
// to an in-process image queue, so every presented frame, real or generated, is read back.
//
// The game's frames are a block-noise texture panning right by kStep pixels per frame under a
// fixed HUD band. A generated frame at phase t must show the texture panned by kStep * t from
// its real frame, and the HUD band where it was.

#include <gtest/gtest.h>
#include <system/window.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "luwes/pacer.h"
#include "src/engine.h"
#include "tests/device/harness.h"

namespace framefx {
namespace {

using test::Apply;
using test::Frame;
using test::GlGame;
using test::GlOptions;
using test::OpenFds;
using test::RssKb;
using test::VkGame;
using test::VkOptions;

constexpr int kWidth = 256;
constexpr int kHeight = 128;
constexpr int kHud = 16;   // rows of fixed HUD at the frame's top edge
constexpr int kStep = 8;   // pan per real frame, pixels
constexpr int kEngageFrames = 60;
constexpr uint8_t kHudColor[4] = {255, 0, 255, 255};

LuwesSettings Luwes(int multiplier, Mode mode) {
    LuwesSettings s;
    s.multiplier = multiplier;
    s.mode = mode;
    return s;
}

// ---- The game's frames ------------------------------------------------------------------------

uint32_t Hash(int x, int y) {
    uint32_t h = static_cast<uint32_t>(x) * 0x9e3779b1u ^ static_cast<uint32_t>(y) * 0x85ebca77u;
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    return h ^ (h >> 13);
}

// Row 0 is the top of the frame.
void Pattern(int shift, uint8_t* rgba) {
    for (int y = 0; y < kHeight; y++) {
        for (int x = 0; x < kWidth; x++) {
            uint8_t* p = rgba + (y * kWidth + x) * 4;
            if (y < kHud) {
                memcpy(p, kHudColor, 4);
                continue;
            }
            const uint32_t h = Hash((x - shift) >> 2, y >> 2);
            p[0] = static_cast<uint8_t>(h);
            p[1] = static_cast<uint8_t>(h >> 8);
            p[2] = static_cast<uint8_t>(h >> 16);
            p[3] = 255;
        }
    }
}

test::Source Panning() {
    return [](int frame, int, int, std::vector<uint8_t>* pixels) {
        Pattern(kStep * frame, pixels->data());
    };
}

GlGame Gl() {
    return GlGame(Panning(), GlOptions{.width = kWidth, .height = kHeight});
}

// Mean absolute difference over the moving interior, against the pattern panned by `shift`.
double Error(const Frame& f, double shift) {
    std::vector<uint8_t> expected(kWidth * kHeight * 4);
    Pattern(static_cast<int>(std::lround(shift)), expected.data());
    double sum = 0;
    int n = 0;
    for (int y = kHud + 8; y < kHeight - 8; y++) {
        for (int x = 32; x < kWidth - 32; x++) {
            for (int c = 0; c < 3; c++) {
                const size_t i = (y * kWidth + x) * 4 + c;
                sum += std::abs(static_cast<int>(f.rgba[i]) - static_cast<int>(expected[i]));
                n++;
            }
        }
    }
    return sum / n;
}

double HudError(const Frame& f) {
    double sum = 0;
    int n = 0;
    for (int y = 2; y < kHud - 2; y++) {
        for (int x = 0; x < kWidth; x++) {
            for (int c = 0; c < 3; c++) {
                sum += std::abs(f.rgba[(y * kWidth + x) * 4 + c] - kHudColor[c]);
                n++;
            }
        }
    }
    return sum / n;
}

// Presents per real frame the engine runs for `multiplier` on a `refreshNs` panel.
int Presents(int multiplier, int64_t refreshNs) {
    return luwes::EffectiveMultiplier(multiplier, 1e9 / static_cast<double>(refreshNs));
}

// Checks the presents one real frame made in steady state: the real one, and every generated
// one at its phase. `shift` is the real frame's pan.
void CheckPresents(const std::vector<Frame>& frames, int multiplier, Mode mode, int shift) {
    ASSERT_EQ(static_cast<size_t>(multiplier), frames.size());
    const int real = mode == Mode::kExtrapolate ? 0 : multiplier - 1;
    EXPECT_LT(Error(frames[real], shift), 1.0) << "the real frame is presented untouched";
    for (int i = 0; i < multiplier; i++) {
        if (i == real) continue;
        // Extrapolation projects past the real frame, interpolation fills toward it.
        const double phase = mode == Mode::kExtrapolate
                ? static_cast<double>(i) / multiplier
                : static_cast<double>(i + 1) / multiplier - 1.0;
        const double expected = Error(frames[i], shift + kStep * phase);
        const double unmoved = Error(frames[i], shift);
        EXPECT_LT(expected, unmoved * 0.5)
                << "generated frame " << i << " is not at phase " << phase << " (error "
                << expected << " against " << unmoved << " unmoved)";
        EXPECT_LT(HudError(frames[i]), 8.0) << "the HUD band moved";
    }
}

LuwesStatus ReadStatus() {
    const uint8_t* b = test::TheChannel().status() + kLuwesStatusOffset;
    LuwesStatus s;
    s.state = static_cast<LuwesState>(b[0]);
    s.reason = static_cast<LuwesReason>(b[1]);
    s.api = static_cast<Api>(b[2]);
    s.source = static_cast<MotionSource>(b[3]);
    s.width = ReadU16(b + 4);
    s.height = ReadU16(b + 6);
    return s;
}

// ---- GLES -------------------------------------------------------------------------------------

class LuwesGles : public testing::TestWithParam<std::tuple<int, Mode>> {};

TEST_P(LuwesGles, GeneratesFramesAtTheirPhase) {
    const auto [multiplier, mode] = GetParam();
    Apply(Luwes(multiplier, mode));
    GlGame game = Gl();
    int64_t refreshNs = 1'000'000'000 / 60;
    native_window_get_refresh_cycle_duration(game.queue.window(), &refreshNs);
    game.periodNs = refreshNs * Presents(multiplier, refreshNs);
    for (int i = 0; i < kEngageFrames; i++) game.present();
    for (int i = 0; i < 10; i++) {
        const int shift = kStep * game.frame;
        CheckPresents(game.present(), Presents(multiplier, refreshNs), mode, shift);
    }
    const LuwesStatus status = ReadStatus();
    EXPECT_EQ(Api::kGles, status.api);
    EXPECT_EQ(MotionSource::kBlockSearch, status.source);
    EXPECT_EQ(kWidth, status.width);
    EXPECT_EQ(kHeight, status.height);
}

INSTANTIATE_TEST_SUITE_P(MultipliersAndModes, LuwesGles,
                         testing::Combine(testing::Values(2, 3, 4),
                                          testing::Values(Mode::kExtrapolate, Mode::kInterpolate)));

TEST(LuwesGlesLifecycle, OffPresentsOnlyTheRealFrame) {
    Apply(LuwesSettings());
    GlGame game = Gl();
    for (int i = 0; i < 40; i++) {
        const int shift = kStep * game.frame;
        const std::vector<Frame> frames = game.present();
        ASSERT_EQ(1u, frames.size());
        EXPECT_LT(Error(frames[0], shift), 1.0);
    }
    EXPECT_EQ(nullptr, game.state) << "an idle present keeps no state";
}

TEST(LuwesGlesLifecycle, LiveChangesApplyWithoutRestart) {
    Apply(Luwes(2, Mode::kExtrapolate));
    GlGame game = Gl();
    int64_t refreshNs = 1'000'000'000 / 60;
    native_window_get_refresh_cycle_duration(game.queue.window(), &refreshNs);
    game.periodNs = refreshNs * Presents(2, refreshNs);
    for (int i = 0; i < kEngageFrames; i++) game.present();
    EXPECT_EQ(2u, game.present().size());
    Apply(Luwes(3, Mode::kInterpolate));
    game.periodNs = refreshNs * Presents(3, refreshNs);
    for (int i = 0; i < 20; i++) game.present();
    const int shift = kStep * game.frame;
    CheckPresents(game.present(), Presents(3, refreshNs), Mode::kInterpolate, shift);
    Apply(LuwesSettings());
    game.present();
    EXPECT_EQ(1u, game.present().size());
}

TEST(LuwesGlesLifecycle, NoLeaksOverContextCycles) {
    Apply(Luwes(2, Mode::kExtrapolate));
    long rss = 0;
    int fds = 0;
    for (int cycle = 0; cycle < 12; cycle++) {
        {
            GlGame game = Gl();
            for (int i = 0; i < 40; i++) game.present();
        }
        if (cycle == 1) {
            rss = RssKb();
            fds = OpenFds();
        }
    }
    EXPECT_LE(OpenFds(), fds + 2);
    EXPECT_LT(RssKb() - rss, 16 * 1024) << "kB grown over 10 cycles";
}

// ---- Vulkan -----------------------------------------------------------------------------------

class LuwesVulkan : public testing::TestWithParam<std::tuple<int, Mode>> {};

TEST_P(LuwesVulkan, GeneratesFramesAtTheirPhase) {
    const auto [multiplier, mode] = GetParam();
    Apply(Luwes(multiplier, mode));
    VkGame game(Panning(), VkOptions{.width = kWidth, .height = kHeight});
    game.periodNs = VkGame::kRefreshNs * Presents(multiplier, VkGame::kRefreshNs);
    for (int i = 0; i < kEngageFrames; i++) game.present();
    for (int i = 0; i < 10; i++) {
        const int shift = kStep * game.frame;
        CheckPresents(game.present(), Presents(multiplier, VkGame::kRefreshNs), mode, shift);
    }
    const LuwesStatus status = ReadStatus();
    EXPECT_EQ(Api::kVulkan, status.api);
    EXPECT_EQ(MotionSource::kBlockSearch, status.source);
    EXPECT_EQ(kWidth, status.width);
    EXPECT_EQ(kHeight, status.height);
}

INSTANTIATE_TEST_SUITE_P(MultipliersAndModes, LuwesVulkan,
                         testing::Combine(testing::Values(2, 3, 4),
                                          testing::Values(Mode::kExtrapolate, Mode::kInterpolate)));

TEST(LuwesVulkanLifecycle, OffPresentsOnlyTheRealFrame) {
    Apply(LuwesSettings());
    VkGame game(Panning(), VkOptions{.width = kWidth, .height = kHeight});
    for (int i = 0; i < 40; i++) {
        const int shift = kStep * game.frame;
        const std::vector<Frame> frames = game.present();
        ASSERT_EQ(1u, frames.size());
        EXPECT_LT(Error(frames[0], shift), 1.0);
    }
}

TEST(LuwesVulkanLifecycle, NoLeaksOverDeviceCycles) {
    Apply(Luwes(3, Mode::kInterpolate));
    long rss = 0;
    int fds = 0;
    for (int cycle = 0; cycle < 12; cycle++) {
        {
            VkGame game(Panning(), VkOptions{.width = kWidth, .height = kHeight});
            for (int i = 0; i < 40; i++) game.present();
        }
        if (cycle == 1) {
            rss = RssKb();
            fds = OpenFds();
        }
    }
    EXPECT_LE(OpenFds(), fds + 2);
    EXPECT_LT(RssKb() - rss, 16 * 1024) << "kB grown over 10 cycles";
}

}  // namespace
}  // namespace framefx
