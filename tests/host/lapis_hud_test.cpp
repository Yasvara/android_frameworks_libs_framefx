/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>

#include "lapis/classifier.h"
#include "lapis/hud.h"
#include "tests/host/lapis_fixture.h"

extern "C" uint64_t framefx_hud_key(uint64_t pass, uint64_t pipeline, uint32_t* slot);
extern "C" size_t framefx_hud_table_size();
extern "C" size_t framefx_hud_max_probe();

// Hide HUD over the frame graphs the survey recorded in games, through the bridge the recorder
// calls. Survey logs carry each draw's depth and blend state (the `draws` letters), not its
// camera, mesh, textures or shader: those features read 0 here, and the labels lean on the frame's
// structure. Set FRAMEFX_HUD_DUMP to a path to write every draw's features and labels as CSV for
// lab/hud-detect/train_hud.py.
namespace framefx::lapis {
namespace {

using test::LoadSurvey;

// The draws of one pass as its survey letters give them; a pass logged without letters (eight
// draws or fewer) by its depth counts. Pipelines stand for the letter: one per kind of draw.
void CaptureFromTrace(Pass* p, bool swapchain) {
    p->hudDraws.clear();
    p->hudKey = HudPassKey(*p, swapchain);
    std::string trace = p->drawTrace;
    if (trace.empty()) {
        const char kind = p->depthWrites == p->draws && p->draws > 0 ? 'w'
                : p->depthTests > 0                                ? 't'
                                                                   : 'a';
        trace.assign(p->draws, kind);
    }
    uint32_t index = 0;
    p->depthClears = 0;
    for (char c : trace) {
        if (c == '|') {
            p->depthClears++;
            p->lastClearDraw = index;
            continue;
        }
        if (c == '/') continue;
        const bool blend = c >= 'A' && c <= 'Z';
        const char k = blend ? static_cast<char>(c - 'A' + 'a') : c;
        HudDraw d;
        d.index = index++;
        d.pipeline = 0x1000 + static_cast<uint64_t>(c);
        d.state = static_cast<uint16_t>((blend ? kDrawBlend : 0) |
                                        (k != 'a' ? kDrawDepthTest | kHudDepthCompare : 0) |
                                        (k == 'w' ? kDrawDepthWrite : 0));
        d.instances = 1;
        p->hudDraws.push_back(d);
    }
}

void Capture(test::Fixture* f) {
    for (Step& s : f->frame.steps) {
        if (!s.isPass) continue;
        const bool swapchain = !s.pass.colors.empty() && s.pass.colors[0].image == test::kSwapchain;
        CaptureFromTrace(&s.pass, swapchain);
    }
}

struct Result {
    size_t draws = 0;
    size_t agree = 0;
    size_t hud = 0;
    size_t weakHud = 0;
    size_t sceneSkipped = 0;  // draws at or before the geometry pass the table skips
    HudOutput out = {};
};

// Runs the fixture's frames `rounds` times over, Hide HUD on; the table as the next frame sees it.
Result RunGame(const std::string& name, int rounds, FILE* dump) {
    std::vector<test::Fixture> frames = LoadSurvey(name);
    Result r;
    if (frames.empty()) return r;
    for (test::Fixture& f : frames) Capture(&f);
    HudTable table;
    HudLabeler labeler(&table);
    labeler.keepFeatures(dump != nullptr);
    for (int round = 0; round < rounds; round++) {
        for (size_t k = 0; k < frames.size(); k++) {
            const test::Fixture& f = frames[k];
            const Roles roles = Classify(f.frame, f.images);
            HudFrameIn in;
            in.frame = &f.frame;
            in.images = &f.images;
            in.roles = roles;
            in.hideHud = true;
            r.out = labeler.update(in);
            if (round != rounds - 1) continue;
            const std::vector<HudDraw>& draws = labeler.draws();
            for (size_t i = 0; i < draws.size(); i++) {
                r.draws++;
                r.agree += labeler.labels()[i] == labeler.weakLabels()[i];
                r.hud += labeler.labels()[i] == static_cast<uint8_t>(HudLabel::kHud);
                r.weakHud += labeler.weakLabels()[i] == static_cast<uint8_t>(HudLabel::kHud);
            }
            for (size_t step = 0; step < f.frame.steps.size(); step++) {
                const Step& s = f.frame.steps[step];
                if (!s.isPass || roles.geometry < 0 || static_cast<int>(step) > roles.geometry) continue;
                for (const HudDraw& d : s.pass.hudDraws) {
                    r.sceneSkipped += table.find(HudKey(s.pass.hudKey, d.pipeline), d.index).skip;
                }
            }
            if (dump == nullptr) continue;
            const size_t n = HudLabeler::featureCount();
            for (size_t step = 0; step < f.frame.steps.size(); step++) {
                for (uint32_t i = labeler.firstDraw(step); i < labeler.firstDraw(step + 1); i++) {
                    fprintf(dump, "%s,%zu,%zu,%u,%u,%.3f,%u", name.c_str(), k, step,
                            draws[i].index, labeler.labels()[i], labeler.confidences()[i],
                            labeler.weakLabels()[i]);
                    for (size_t c = 0; c < n; c++) fprintf(dump, ",%g", labeler.features()[i * n + c]);
                    fprintf(dump, "\n");
                }
            }
        }
    }
    return r;
}

FILE* Dump() {
    static FILE* file = [] {
        const char* path = getenv("FRAMEFX_HUD_DUMP");
        if (path == nullptr) return static_cast<FILE*>(nullptr);
        FILE* f = fopen(path, "w");
        if (f == nullptr) return f;
        fprintf(f, "game,frame,step,draw,label,confidence,weak");
        for (size_t c = 0; c < HudLabeler::featureCount(); c++) {
            fprintf(f, ",%s", HudLabeler::featureName(c).c_str());
        }
        fprintf(f, "\n");
        return f;
    }();
    return file;
}

TEST(LapisHud, KeysAndSlotsMatchTheRustTable) {
    ASSERT_EQ(framefx_hud_table_size(), kHudTableSize);
    ASSERT_EQ(framefx_hud_max_probe(), kHudMaxProbe);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 1000; i++) {
        const uint64_t pass = rng();
        const uint64_t pipeline = rng();
        uint32_t slot = 0;
        EXPECT_EQ(framefx_hud_key(pass, pipeline, &slot), HudKey(pass, pipeline));
        EXPECT_EQ(slot, HudSlot(HudKey(pass, pipeline)));
    }
}

TEST(LapisHud, TheTableLooksUpRangesAndForgetsOnClear) {
    std::vector<HudEntry> entries(kHudTableSize, HudEntry{0, 0});
    const uint64_t key = HudKey(0x10, 0x20);
    const uint64_t other = HudKey(0x10, 0x21);
    entries[HudSlot(key)] = {key, kHudSkip | kHudKnown | 3 | (kHudOpen << 24)};
    // A neighbour on the same slot probes on.
    entries[(HudSlot(key) + 1) & (kHudTableSize - 1)] = {other, kHudKnown};
    HudTable table;
    table.publish(entries.data());
    EXPECT_FALSE(table.find(key, 2).skip);
    EXPECT_TRUE(table.find(key, 3).skip);
    EXPECT_TRUE(table.find(key, 900).skip);
    EXPECT_TRUE(table.find(key, 0).known);
    EXPECT_FALSE(table.find(HudKey(0x11, 0x20), 5).skip);
    table.clear();
    EXPECT_FALSE(table.find(key, 3).skip);
    EXPECT_FALSE(table.find(key, 3).known);
}

TEST(LapisHud, PassKeysIgnoreWhichImagesThePassDrawsInto) {
    Pass a;
    a.renderPass = 0xabc;
    a.width = 2400;
    a.height = 1080;
    Target t;
    t.image = 5;
    t.format = VK_FORMAT_R8G8B8A8_UNORM;
    a.colors.push_back(t);
    Pass b = a;
    b.colors[0].image = 9;
    EXPECT_EQ(HudPassKey(a, true), HudPassKey(b, true));
    EXPECT_NE(HudPassKey(a, true), HudPassKey(a, false));
    b.width = 1200;
    EXPECT_NE(HudPassKey(a, true), HudPassKey(b, true));
}

// The structure-only labels hold to the frame's roles across the survey's games, and the table
// never skips a draw of a pass up to the scene's geometry pass.
TEST(LapisHud, SurveyGamesAreLabelledLikeTheirFrameStructureAndTheSceneIsNeverSkipped) {
    for (const char* game : {"genshin.log", "hsr.log", "mho.log", "mlbb.log", "roblox.log",
                             "uma.log", "uma-career.log", "uma-small.log", "endfield.log",
                             "deltaforce.log", "wuwa.log", "mlbb-draws.log"}) {
        SCOPED_TRACE(game);
        const Result r = RunGame(game, 3, Dump());
        ASSERT_GT(r.draws, 0u);
        EXPECT_GE(static_cast<double>(r.agree) / r.draws, 0.9)
                << r.agree << " of " << r.draws << " agree; HUD " << r.hud << " vs " << r.weakHud;
        EXPECT_EQ(r.sceneSkipped, 0u);
    }
    if (FILE* dump = Dump()) fflush(dump);
}

// Games whose HUD the frame structure shows, draw by draw, get it skipped from the labels alone.
// Star Rail's log has no letters for its presenting pass: its HUD is found, not sure enough to skip.
TEST(LapisHud, AHudThePresentingPassDrawsIsSkipped) {
    const Result hsr = RunGame("hsr.log", 4, nullptr);
    EXPECT_GT(hsr.hud, 0u);
    EXPECT_FALSE(hsr.out.skipping);
    for (const char* game : {"mlbb-draws.log", "uma.log", "uma-career.log"}) {
        SCOPED_TRACE(game);
        const Result r = RunGame(game, 4, nullptr);
        EXPECT_GT(r.hud, 0u);
        EXPECT_TRUE(r.out.skipping) << "coverage " << r.out.coverage;
        EXPECT_GE(r.out.coverage, 0.9f);
    }
}

TEST(LapisHud, NothingIsSkippedWithHideHudOff) {
    std::vector<test::Fixture> frames = LoadSurvey("mlbb-draws.log");
    ASSERT_FALSE(frames.empty());
    for (test::Fixture& f : frames) Capture(&f);
    HudTable table;
    HudLabeler labeler(&table);
    for (int round = 0; round < 6; round++) {
        for (const test::Fixture& f : frames) {
            HudFrameIn in;
            in.frame = &f.frame;
            in.images = &f.images;
            in.roles = Classify(f.frame, f.images);
            EXPECT_FALSE(labeler.update(in).skipping);
        }
    }
    for (const Step& s : frames.back().frame.steps) {
        for (const HudDraw& d : s.isPass ? s.pass.hudDraws : std::vector<HudDraw>()) {
            EXPECT_FALSE(table.find(HudKey(s.pass.hudKey, d.pipeline), d.index).skip);
        }
    }
}

TEST(LapisHud, FeatureNamesAndProfilesCrossTheBridge) {
    const size_t n = HudLabeler::featureCount();
    ASSERT_GT(n, 30u);
    EXPECT_EQ(HudLabeler::featureName(0), "bias");
    EXPECT_EQ(HudLabeler::featureName(n), "");
    HudTable table;
    HudLabeler labeler(&table);
    EXPECT_TRUE(labeler.warm("hudpriors 1 0000abcd:20:1"));
    EXPECT_NE(labeler.profile().find("0000abcd:20:1"), std::string::npos);
    EXPECT_FALSE(labeler.warm("nonsense"));
    EXPECT_FALSE(labeler.setWeights("hudweights 1 4 " + std::to_string(n) + " 1 2"));
}

}  // namespace
}  // namespace framefx::lapis
