/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "luwes/orientation.h"

namespace framefx::luwes {
namespace {

constexpr int W = 64;
constexpr int H = 32;

ThumbVote Compare(const uint8_t* real, const uint8_t* scene, int current) {
    return CompareThumbs<W, H>(real, scene, current);
}

// A frame with structure: a gradient, a bright block off centre, a dark bar.
std::vector<uint8_t> Frame() {
    std::vector<uint8_t> f(W * H * 4);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int v = 30 + x * 2 + y;
            if (x > 8 && x < 22 && y > 4 && y < 14) v = 240;
            if (y > 24 && x > 30) v = 10;
            uint8_t* p = &f[(y * W + x) * 4];
            p[0] = p[1] = p[2] = static_cast<uint8_t>(v);
            p[3] = 255;
        }
    }
    return f;
}

// The frame as a scene kept before its tonemap: another curve, brighter, flatter; then flipped.
std::vector<uint8_t> Scene(const std::vector<uint8_t>& shown, int flip) {
    std::vector<uint8_t> s(shown.size());
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const int sy = (flip & 1) ? H - 1 - y : y;
            const int sx = (flip & 2) ? W - 1 - x : x;
            const uint8_t* a = &shown[(y * W + x) * 4];
            uint8_t* b = &s[(sy * W + sx) * 4];
            for (int c = 0; c < 3; c++) {
                b[c] = static_cast<uint8_t>(120 + 0.4 * 255 * std::pow(a[c] / 255.0, 2.2));
            }
            b[3] = 255;
        }
    }
    return s;
}

TEST(Orientation, ASceneBeforeItsTonemapVotesForItsFlip) {
    const auto shown = Frame();
    for (int flip = 0; flip < 4; flip++) {
        const auto scene = Scene(shown, flip);
        EXPECT_EQ(flip, VoteOf(Compare(shown.data(), scene.data(), -1))) << flip;
    }
}

TEST(Orientation, AFlatFrameSaysNothingOfTheFlip) {
    std::vector<uint8_t> black(W * H * 4, 0);
    const auto shown = Frame();
    const ThumbVote v = Compare(black.data(), shown.data(), -1);
    EXPECT_TRUE(v.flat);
    EXPECT_EQ(-1, VoteOf(v));
    EXPECT_LT(v.shown, 0.0f);
    // A flat scene says nothing of coverage either.
    EXPECT_LT(Compare(shown.data(), black.data(), 0).shown, 0.0f);
}

TEST(Orientation, TheSceneShowsUntilAMenuCoversIt) {
    const auto shown = Frame();
    const auto scene = Scene(shown, 1);
    EXPECT_GT(Compare(shown.data(), scene.data(), 1).shown, 0.6f);
    // A fade or a plain menu: the screen goes flat over the scene.
    std::vector<uint8_t> fade(W * H * 4, 18);
    EXPECT_EQ(0.0f, Compare(fade.data(), scene.data(), 1).shown);
    // A menu with shapes of its own: well under kShownFrom.
    std::vector<uint8_t> menu(W * H * 4, 255);
    for (int i = 0; i < W * H; i++) menu[i * 4] = menu[i * 4 + 1] = menu[i * 4 + 2] = (i * 37) % 251;
    EXPECT_LT(Compare(menu.data(), scene.data(), 1).shown, 0.25f);
}

}  // namespace
}  // namespace framefx::luwes
