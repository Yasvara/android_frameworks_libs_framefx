/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

// Kelir on this device's GPU, through the entry points the graphics loaders call: every kind on
// every target format, held to the CPU reference renderer (tests/ref), and how the present paths
// behave around the game: what they decline, what they leave untouched, the GL state they restore
// and the memory they keep.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "kelir/colour.h"
#include "kelir/plan.h"
#include "kelir/section.h"
#include "src/engine.h"
#include "tests/device/harness.h"
#include "tests/ref/kelir_ref.h"

namespace framefx::kelir {
namespace {

using test::Apply;
using test::Frame;
using test::GlGame;
using test::GlOptions;
using test::VkGame;
using test::VkOptions;
using test::Entry;

constexpr int kWidth = 256;
constexpr int kHeight = 128;

// ---- Stacks ----------------------------------------------------------------------------------

Node MakeNode(uint8_t kind, float intensity = 1.0f) {
    Node n;
    n.kind = kind;
    n.enabled = true;
    n.intensity = static_cast<uint16_t>(std::lround(intensity * 65535.0f));
    return n;
}

void Set(Node& n, int param, float value) {
    n.q[param] = Encode(Info(n.kind).params[param], value);
}

std::vector<uint8_t> Section(const Stack& s) {
    std::vector<uint8_t> out(kMaxKelirSectionSize);
    out.resize(EncodeSection(s, out.data()));
    return out;
}

Stack StackOf(std::initializer_list<Node> nodes) {
    static uint32_t revision = 1;
    Stack s;
    s.revision = revision++;
    for (const Node& n : nodes) s.nodes[s.count++] = n;
    return s;
}

// Publishes the stack and waits until the engine compiled it; the plan it compiled.
std::shared_ptr<const Plan> Use(const Stack& s, const LuwesSettings& luwes = LuwesSettings()) {
    Apply(luwes, Section(s));
    for (int i = 0; i < 2000 && Engine::get().config().kelir->revision != s.revision; i++) {
        usleep(1000);
    }
    const auto plan = Engine::get().config().kelir;
    EXPECT_EQ(s.revision, plan->revision);
    return plan;
}

// A kind at a setting: "mid" halfway to each parameter's far end, "extreme" all the way, with
// whatever the kind's defaults leave at identity moved too.
Node Setting(uint8_t kind, float toward) {
    Node n = MakeNode(kind);
    const KindInfo& info = Info(kind);
    for (int i = 0; i < info.paramCount; i++) {
        const Param& p = info.params[i];
        if (p.choices > 0) {
            Set(n, i, toward >= 1.0f ? static_cast<float>(p.choices - 1) : 0.0f);
            continue;
        }
        const float end = p.hi != p.def ? p.hi : p.lo;
        // Tint colour wraps: its far end is its default.
        const float t = kind == kColor && i == p::kTintColour ? std::min(toward, 0.5f) : toward;
        Set(n, i, p.def + t * (end - p.def));
    }
    return n;
}

// ---- The game's frame ------------------------------------------------------------------------

// Grey ramp, RGB cube patches, a zone plate, checker over a gradient, flat patches, highlights.
float TestValue(int x, int y, int c, int w, int h) {
    const float u = (x + 0.5f) / w;
    const float v = (y + 0.5f) / h;
    if (y < h / 8) return u;  // grey ramp
    if (y < h / 2) {
        // 16 x 4 patches of the colour cube
        const int px = x * 16 / w;
        const int py = (y - h / 8) * 4 / (h / 2 - h / 8);
        const int idx = py * 16 + px;
        const int r = idx % 4;
        const int g = (idx / 4) % 4;
        const int b = idx / 16;
        return (c == 0 ? r : c == 1 ? g : b) / 3.0f;
    }
    if (x < w / 3) {
        // zone plate
        const float dx = (x - w / 6.0f) / w;
        const float dy = (y - 3.0f * h / 4) / h;
        return 0.5f + 0.5f * std::cos(220.0f * (dx * dx + dy * dy) + c);
    }
    if (x < 2 * w / 3) {
        const bool checker = ((x / 6) + (y / 6)) % 2 == 0;
        return checker ? 0.95f : 0.1f + 0.6f * v;
    }
    return c == 0 ? 0.85f : c == 1 ? 0.62f - 0.3f * u : 0.5f;
}

std::vector<float> TestImage(int w, int h) {
    std::vector<float> img(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) img[(static_cast<size_t>(y) * w + x) * 3 + c] = TestValue(x, y, c, w, h);
        }
    }
    return img;
}

// The image as the game uploads it: RGBA8 bytes, or RGB10A2 words; top row first. Made once
// per size.
test::Source Upload(bool tenBit) {
    return [tenBit](int, int w, int h, std::vector<uint8_t>* pixels) {
        static std::vector<uint8_t> made[2];
        static int madeW[2] = {};
        static int madeH[2] = {};
        std::vector<uint8_t>& bytes = made[tenBit];
        if (madeW[tenBit] != w || madeH[tenBit] != h) {
            const std::vector<float> img = TestImage(w, h);
            bytes.assign(static_cast<size_t>(w) * h * 4, 0);
            for (size_t i = 0; i < static_cast<size_t>(w) * h; i++) {
                const float* p = &img[i * 3];
                if (tenBit) {
                    const auto q = [](float v) { return static_cast<uint32_t>(std::lround(v * 1023.0f)); };
                    const uint32_t word = q(p[0]) | q(p[1]) << 10 | q(p[2]) << 20 | 3u << 30;
                    memcpy(&bytes[i * 4], &word, 4);
                } else {
                    for (int c = 0; c < 3; c++) bytes[i * 4 + c] = static_cast<uint8_t>(std::lround(p[c] * 255.0f));
                    bytes[i * 4 + 3] = 255;
                }
            }
            madeW[tenBit] = w;
            madeH[tenBit] = h;
        }
        memcpy(pixels->data(), bytes.data(), bytes.size());
    };
}

enum class Format { kRgba8, kSrgb, kRgb10a2 };

const char* Name(Format f) {
    return f == Format::kRgba8 ? "rgba8" : f == Format::kSrgb ? "srgb" : "rgb10a2";
}

// What the reference renders for the game's frame: the stored values, as the pass reads them.
ref::Image Source(Format format, bool bottomFirst) {
    const float scale = format == Format::kRgb10a2 ? 1023.0f : 255.0f;
    const std::vector<float> img = TestImage(kWidth, kHeight);
    ref::Image out(kWidth, kHeight);
    for (int y = 0; y < kHeight; y++) {
        const int row = bottomFirst ? kHeight - 1 - y : y;
        for (int x = 0; x < kWidth; x++) {
            const float* p = &img[(static_cast<size_t>(row) * kWidth + x) * 3];
            Rgb c = {std::round(p[0] * scale) / scale, std::round(p[1] * scale) / scale,
                     std::round(p[2] * scale) / scale};
            if (format == Format::kSrgb) c = {Decode(c.r), Decode(c.g), Decode(c.b)};
            out.set(x, y, c);
        }
    }
    return out;
}

ref::Frame RefFrame(Format format, uint32_t seed, bool gl) {
    ref::Frame f;
    f.displayWidth = kWidth;
    f.displayHeight = kHeight;
    f.flipY = gl;
    f.seed = seed;
    f.halfLsb = format == Format::kRgb10a2 ? 0.5f / 1023.0f : 0.5f / 255.0f;
    f.readsLinear = format == Format::kSrgb;
    f.writesLinear = format == Format::kSrgb;
    return f;
}

struct Diff {
    float max = 0.0f;   // LSB
    double mean = 0.0;  // LSB
    int x = 0, y = 0, c = 0;  // where the worst one is
    float got = 0.0f, want = 0.0f;
};

std::ostream& operator<<(std::ostream& os, const Diff& d) {
    return os << "worst at (" << d.x << ", " << d.y << ") channel " << d.c << ": got " << d.got
              << " want " << d.want;
}

// The frame against the reference, in LSB of the target's depth, `border` px left out.
Diff Compare(const Frame& got, const ref::Image& want, Format format, bool bottomFirst, int border) {
    const float lsb = format == Format::kRgb10a2 ? 1023.0f : 255.0f;
    Diff d;
    size_t n = 0;
    for (int y = border; y < got.height - border; y++) {
        const int row = bottomFirst ? got.height - 1 - y : y;
        for (int x = border; x < got.width - border; x++) {
            Rgb w = want.at(x, row);
            if (format == Format::kSrgb) w = {Encode(w.r), Encode(w.g), Encode(w.b)};
            const float wc[3] = {w.r, w.g, w.b};
            for (int c = 0; c < 3; c++) {
                const float w = std::clamp(wc[c], 0.0f, 1.0f);
                const float e = std::fabs(got.at(x, y, c) - w) * lsb;
                if (e > d.max) d = {e, d.mean, x, y, c, got.at(x, y, c), w};
                d.mean += e;
                n++;
            }
        }
    }
    d.mean /= static_cast<double>(n);
    return d;
}

// The engine's Kelir status, for failure messages.
std::string StatusText() {
    const uint8_t* b = test::TheChannel().status() + kKelirStatusOffset;
    return "status state " + std::to_string(b[0]) + " reason " + std::to_string(b[1]) + " path " +
            std::to_string(b[9]);
}

bool Neighbourhood(const Plan& plan) {
    return plan.needsSource();
}

// ---- KelirGles -------------------------------------------------------------------------------

GlGame Gl(Format format) {
    return GlGame(Upload(format == Format::kRgb10a2),
                  GlOptions{.width = kWidth,
                            .height = kHeight,
                            .srgb = format == Format::kSrgb,
                            .tenBit = format == Format::kRgb10a2});
}

// Presents twice (the first compiles) and checks the second against the reference.
void CheckGles(const Stack& stack, Format format, const std::string& what) {
    const auto plan = Use(stack);
    GlGame game = Gl(format);
    game.present();
    const std::vector<Frame> frames = game.present();
    ASSERT_EQ(1u, frames.size()) << what;
    const ref::Image want =
            ref::Render(*plan, Source(format, /*bottomFirst=*/true), RefFrame(format, 2, true));
    const Diff d = Compare(frames[0], want, format, /*bottomFirst=*/true, Neighbourhood(*plan) ? 2 : 0);
    EXPECT_LE(d.max, 2.0f) << what << " " << Name(format) << ", " << d;
    EXPECT_LE(d.mean, 0.5) << what << " " << Name(format);
}

class KelirGles : public testing::TestWithParam<std::tuple<int, float, Format>> {};

TEST_P(KelirGles, EveryKindMatchesReference) {
    const auto [kind, toward, format] = GetParam();
    CheckGles(StackOf({Setting(static_cast<uint8_t>(kind), toward)}), format,
              std::string(Info(kind).name) + " at " + std::to_string(toward));
}

INSTANTIATE_TEST_SUITE_P(KindsSettingsFormats, KelirGles,
                         testing::Combine(testing::Range(0, static_cast<int>(kKindCount)),
                                          testing::Values(0.05f, 0.5f, 1.0f),
                                          testing::Values(Format::kRgba8, Format::kSrgb,
                                                          Format::kRgb10a2)));

TEST(KelirGlesPresent, AllDefaultIsBitExactPassthrough) {
    Use(StackOf({MakeNode(kBrightnessContrast), MakeNode(kColor), MakeNode(kDetails)}));
    GlGame game = Gl(Format::kRgba8);
    const std::vector<Frame> frames = game.present();
    ASSERT_EQ(1u, frames.size());
    const ref::Image src = Source(Format::kRgba8, false);
    for (int y = 0; y < kHeight; y++) {
        for (int x = 0; x < kWidth; x++) {
            for (int c = 0; c < 3; c++) {
                const float want = c == 0 ? src.at(x, y).r : c == 1 ? src.at(x, y).g : src.at(x, y).b;
                ASSERT_EQ(std::lround(want * 255.0f), frames[0].rgba[(y * kWidth + x) * 4 + c]);
            }
        }
    }
    EXPECT_EQ(0, Entry()->gl_redrawn(game.surface)) << "an untouched frame keeps its buffer age";
}

TEST(KelirGlesPresent, Stacks1_3_5MatchReference) {
    Node color = Setting(kColor, 0.5f);
    Node bc = Setting(kBrightnessContrast, 0.4f);
    Node vignette = Setting(kVignette, 0.6f);
    Node sharpen = Setting(kSharpen, 0.7f);
    Node details = MakeNode(kDetails);
    Set(details, p::kClarity, 0.6f);
    for (Format f : {Format::kRgba8, Format::kSrgb}) {
        CheckGles(StackOf({color}), f, "1 node");
        CheckGles(StackOf({bc, color, vignette}), f, "3 nodes");
        CheckGles(StackOf({bc, color, vignette, sharpen, details}), f, "5 nodes");
    }
}

TEST(KelirGlesPresent, FetchPathEqualsTexturePath) {
    // A pointwise stack takes the fetch path; with a Sharpen at zero strength in front it takes
    // the copy, and must give the same picture.
    const Node color = Setting(kColor, 0.6f);
    Use(StackOf({color, Setting(kVignette, 0.5f)}));
    GlGame fetch = Gl(Format::kRgba8);
    fetch.present();
    const Frame a = fetch.present()[0];
    Node idle = MakeNode(kTiltShift);
    Set(idle, p::kSize, 1.0f);
    Set(idle, p::kSoftness, 0.0f);
    Set(idle, p::kBlur, 0.001f);
    Use(StackOf({idle, color, Setting(kVignette, 0.5f)}));
    GlGame texture = Gl(Format::kRgba8);
    texture.present();
    const Frame b = texture.present()[0];
    int worst = 0;
    for (int y = 0; y < kHeight; y++) {
        for (int x = kWidth / 4; x < 3 * kWidth / 4; x++) {  // inside the tilt-shift's sharp band
            for (int c = 0; c < 3; c++) {
                const size_t i = (static_cast<size_t>(y) * kWidth + x) * 4 + c;
                worst = std::max(worst, std::abs(a.rgba[i] - b.rgba[i]));
            }
        }
    }
    EXPECT_LE(worst, 1);
}

// Every piece of state the pass touches, as a game might leave it.
struct GameState {
    GLint program, drawFb, readFb, vao, active, viewport[4], texture2d, texture3d, sampler;
    GLboolean mask[4];
    bool blend, depth, stencil, scissor, cull, discard;
};

GameState ReadState(GLint unit) {
    GameState s = {};
    glGetIntegerv(GL_CURRENT_PROGRAM, &s.program);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s.drawFb);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.readFb);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &s.active);
    glGetIntegerv(GL_VIEWPORT, s.viewport);
    glGetBooleanv(GL_COLOR_WRITEMASK, s.mask);
    glActiveTexture(GL_TEXTURE0 + unit);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.texture2d);
    glGetIntegerv(GL_TEXTURE_BINDING_3D, &s.texture3d);
    glGetIntegerv(GL_SAMPLER_BINDING, &s.sampler);
    glActiveTexture(static_cast<GLenum>(s.active));
    s.blend = glIsEnabled(GL_BLEND);
    s.depth = glIsEnabled(GL_DEPTH_TEST);
    s.stencil = glIsEnabled(GL_STENCIL_TEST);
    s.scissor = glIsEnabled(GL_SCISSOR_TEST);
    s.cull = glIsEnabled(GL_CULL_FACE);
    s.discard = glIsEnabled(GL_RASTERIZER_DISCARD);
    return s;
}

TEST(KelirGlesPresent, StateRestoredExactly) {
    Use(StackOf({Setting(kSharpen, 0.5f), Setting(kColor, 0.5f), Setting(kVignette, 0.5f)}));
    GlGame game = Gl(Format::kRgba8);
    GLint units = 0;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &units);
    std::mt19937 rng(9);
    GLuint textures[4];
    glGenTextures(4, textures);
    glBindTexture(GL_TEXTURE_3D, textures[3]);  // a 3D texture from the start
    GLuint vaos[2];
    glGenVertexArrays(2, vaos);
    GLuint samplers[2];
    glGenSamplers(2, samplers);
    for (int round = 0; round < 200; round++) {
        // The harness's own frame uses the default framebuffer and its texture first.
        game.present();
        const GLint unit = units - 1 - static_cast<GLint>(rng() % 3);
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, textures[rng() % 3]);
        glBindTexture(GL_TEXTURE_3D, rng() % 2 ? textures[3] : 0);
        glBindSampler(static_cast<GLuint>(unit), rng() % 2 ? samplers[rng() % 2] : 0);
        glActiveTexture(GL_TEXTURE0 + static_cast<GLint>(rng() % 8));
        glBindVertexArray(vaos[rng() % 2]);
        glViewport(static_cast<GLint>(rng() % 20), static_cast<GLint>(rng() % 20), 10 + rng() % 200,
                   10 + rng() % 100);
        glColorMask(rng() % 2, rng() % 2, rng() % 2, rng() % 2);
        for (GLenum cap : {GL_BLEND, GL_DEPTH_TEST, GL_STENCIL_TEST, GL_SCISSOR_TEST, GL_CULL_FACE}) {
            if (rng() % 2) glEnable(cap); else glDisable(cap);
        }
        const GameState before = ReadState(unit);
        // An error the game has not read yet stays its own across the pass.
        glEnable(GL_TEXTURE_2D);
        const framefx_gl_frame f = {&game.state, game.display, game.surface, game.config,
                                    game.queue.window(), GlGame::Swap, &game};
        ASSERT_NE(FRAMEFX_PASS, Entry()->gl_present(&f)) << "round " << round << ", " << StatusText();
        const GameState after = ReadState(unit);
        ASSERT_EQ(0, memcmp(&before, &after, sizeof(before))) << "round " << round;
        ASSERT_EQ(static_cast<GLenum>(GL_INVALID_ENUM), glGetError());
        ASSERT_EQ(static_cast<GLenum>(GL_NO_ERROR), glGetError());
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        for (GLenum cap : {GL_BLEND, GL_DEPTH_TEST, GL_STENCIL_TEST, GL_SCISSOR_TEST, GL_CULL_FACE}) {
            glDisable(cap);
        }
        glViewport(0, 0, kWidth, kHeight);
        game.queue.drain();
    }
}

TEST(KelirGlesPresent, BufferAgeZeroWhileFiltered) {
    Use(StackOf({Setting(kSepia, 1.0f)}));
    GlGame game = Gl(Format::kRgba8);
    game.present();
    EXPECT_EQ(1, Entry()->gl_redrawn(game.surface));
    Use(StackOf({}));
    game.present();
    EXPECT_EQ(0, Entry()->gl_redrawn(game.surface));
}

TEST(KelirGlesPresent, PeekPresentsUntouched) {
    Stack s = StackOf({Setting(kSepia, 1.0f)});
    s.peek = true;
    Use(s);
    GlGame game = Gl(Format::kRgba8);
    const Frame f = game.present()[0];
    const ref::Image src = Source(Format::kRgba8, false);
    EXPECT_EQ(std::lround(src.at(10, 2).r * 255.0f), f.rgba[(2 * kWidth + 10) * 4]);
}

TEST(KelirGlesPresent, SplitShowsTheSourceLeftOfTheDivider) {
    Stack s = StackOf({Setting(kBlackWhite, 1.0f)});
    s.split = true;
    s.splitX = 32768;
    const auto plan = Use(s);
    GlGame game = Gl(Format::kRgba8);
    const Frame f = game.present()[0];
    ref::Frame frame = RefFrame(Format::kRgba8, 1, true);
    frame.split = plan->splitX;
    const ref::Image want = ref::Render(*plan, Source(Format::kRgba8, true), frame);
    const Diff d = Compare(f, want, Format::kRgba8, true, 0);
    EXPECT_LE(d.max, 2.0f) << d;
    // Left: the colour cube as it was; right: grey.
    const size_t left = (static_cast<size_t>(40) * kWidth + 20) * 4;
    const size_t right = (static_cast<size_t>(40) * kWidth + 200) * 4;
    EXPECT_FALSE(f.rgba[right] != f.rgba[right + 1] || f.rgba[right] != f.rgba[right + 2]);
    EXPECT_TRUE(f.rgba[left] != f.rgba[left + 1] || f.rgba[left] != f.rgba[left + 2]);
}

TEST(KelirGlesPresent, LiveSwapNoMixedFrame) {
    const Stack a = StackOf({Setting(kSepia, 1.0f)});
    const Stack b = StackOf({Setting(kNightMode, 1.0f)});
    const auto planA = Use(a);
    GlGame game = Gl(Format::kRgba8);
    game.present();
    const ref::Image wantA = ref::Render(*planA, Source(Format::kRgba8, true), RefFrame(Format::kRgba8, 2, true));
    const auto planB = Use(b);
    const Frame f = game.present()[0];
    const ref::Image wantB = ref::Render(*planB, Source(Format::kRgba8, true), RefFrame(Format::kRgba8, 2, true));
    EXPECT_LE(Compare(f, wantB, Format::kRgba8, true, 0).max, 2.0f);
    EXPECT_GT(Compare(f, wantA, Format::kRgba8, true, 0).max, 8.0f);
}

TEST(KelirGlesPresent, LeakFree1000Cycles) {
    const Stack stacks[] = {
            StackOf({Setting(kColor, 0.5f)}),
            StackOf({Setting(kSharpen, 0.5f), Setting(kVignette, 0.5f)}),
            StackOf({Setting(kDetails, 0.5f), Setting(kFilmGrain, 0.5f)}),
            StackOf({Setting(kWatercolor, 0.5f)}),
    };
    GlGame game = Gl(Format::kRgba8);
    long rss = 0;
    for (int i = 0; i < 1000; i++) {
        if (i % 50 == 0) Use(stacks[(i / 50) % 4]);
        game.present();
        if (i == 200) rss = test::RssKb();
    }
    EXPECT_LT(test::RssKb() - rss, 8 * 1024) << "kB grown";
}

TEST(KelirGlesPresent, StatusReportsTheTarget) {
    Use(StackOf({Setting(kVignette, 0.5f)}));
    GlGame game = Gl(Format::kRgba8);
    game.present();
    const uint8_t* b = test::TheChannel().status() + kKelirStatusOffset;
    EXPECT_EQ(static_cast<uint8_t>(KelirState::kRunning), b[0]);
    EXPECT_EQ(static_cast<uint8_t>(Api::kGles), b[2]);
    EXPECT_EQ(kWidth, ReadU16(b + 4));
    EXPECT_EQ(kHeight, ReadU16(b + 6));
    EXPECT_EQ(static_cast<uint8_t>(KelirPath::kFetch), b[9]);
}

TEST(KelirGlesPresent, NoGlErrors) {
    GlGame game = Gl(Format::kRgba8);
    for (int kind = 0; kind < static_cast<int>(kKindCount); kind++) {
        Use(StackOf({Setting(static_cast<uint8_t>(kind), 0.5f)}));
        game.present();
        game.present();
        EXPECT_EQ(static_cast<GLenum>(GL_NO_ERROR), glGetError()) << Info(kind).name;
    }
}

// Fetch reads one sample of a multisampled framebuffer: those take the copy.
TEST(KelirGlesPresent, MsaaFb0UsesTexturePath) {
    const auto plan = Use(StackOf({Setting(kSepia, 1.0f), Setting(kVignette, 0.5f)}));
    GlGame game(Upload(false), GlOptions{.width = kWidth, .height = kHeight, .samples = 4});
    game.present();
    const Frame f = game.present()[0];
    const uint8_t* b = test::TheChannel().status() + kKelirStatusOffset;
    EXPECT_EQ(static_cast<uint8_t>(KelirState::kRunning), b[0]) << StatusText();
    EXPECT_EQ(static_cast<uint8_t>(KelirPath::kTexture), b[9]);
    const ref::Image want = ref::Render(*plan, Source(Format::kRgba8, true), RefFrame(Format::kRgba8, 2, true));
    EXPECT_LE(Compare(f, want, Format::kRgba8, true, 0).max, 2.0f);
}

// A game that keeps its last frame to draw over it must get it back unfiltered.
TEST(KelirGlesPresent, DeclinesPreservedSwap) {
    Use(StackOf({Setting(kSepia, 1.0f)}));
    GlGame game(Upload(false), GlOptions{.width = kWidth, .height = kHeight, .preserved = true});
    const Frame f = game.present()[0];
    const ref::Image src = Source(Format::kRgba8, false);
    EXPECT_EQ(std::lround(src.at(40, 40).g * 255.0f), f.rgba[(40 * kWidth + 40) * 4 + 1]);
    const uint8_t* b = test::TheChannel().status() + kKelirStatusOffset;
    EXPECT_EQ(static_cast<uint8_t>(KelirState::kDeclined), b[0]);
    EXPECT_EQ(static_cast<uint8_t>(KelirReason::kPreserved), b[1]);
}

// ---- KelirVulkan -----------------------------------------------------------------------------

VkFormat VkFormatOf(Format f) {
    return f == Format::kRgba8 ? VK_FORMAT_R8G8B8A8_UNORM
            : f == Format::kSrgb ? VK_FORMAT_R8G8B8A8_SRGB
                                 : VK_FORMAT_A2B10G10R10_UNORM_PACK32;
}

VkGame Vk(Format format, VkSurfaceTransformFlagBitsKHR transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
    return VkGame(Upload(format == Format::kRgb10a2),
                  VkOptions{.width = kWidth, .height = kHeight, .format = VkFormatOf(format),
                            .transform = transform});
}

void CheckVulkan(const Stack& stack, Format format, const std::string& what) {
    const auto plan = Use(stack);
    VkGame game = Vk(format);
    game.present();
    const std::vector<Frame> frames = game.present();
    ASSERT_EQ(1u, frames.size()) << what;
    const ref::Image want =
            ref::Render(*plan, Source(format, false), RefFrame(format, 2, false));
    const Diff d = Compare(frames[0], want, format, false, Neighbourhood(*plan) ? 2 : 0);
    EXPECT_LE(d.max, 2.0f) << what << " " << Name(format) << ", " << d;
    EXPECT_LE(d.mean, 0.5) << what << " " << Name(format);
}

class KelirVulkan : public testing::TestWithParam<std::tuple<int, float, Format>> {};

TEST_P(KelirVulkan, EveryKindMatchesReference) {
    const auto [kind, toward, format] = GetParam();
    CheckVulkan(StackOf({Setting(static_cast<uint8_t>(kind), toward)}), format,
                std::string(Info(kind).name) + " at " + std::to_string(toward));
}

INSTANTIATE_TEST_SUITE_P(KindsSettingsFormats, KelirVulkan,
                         testing::Combine(testing::Range(0, static_cast<int>(kKindCount)),
                                          testing::Values(0.05f, 0.5f, 1.0f),
                                          testing::Values(Format::kRgba8, Format::kSrgb,
                                                          Format::kRgb10a2)));

// The reference renderer filters with 8-bit weights, as this GPU does.
TEST(KelirVulkanPresent, FilteringPrecisionMatchesReference) {
    VkGame game = Vk(Format::kRgba8);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(game.physical, &props);
    EXPECT_EQ(8u, props.limits.subTexelPrecisionBits);
    EXPECT_EQ(8u, props.limits.mipmapPrecisionBits);
}

TEST(KelirVulkanPresent, EmptyPlanPresentsAppBufferUntouched) {
    Use(StackOf({}));
    VkGame game = Vk(Format::kRgba8);
    const Frame f = game.present()[0];
    const ref::Image src = Source(Format::kRgba8, false);
    for (int x = 0; x < kWidth; x++) {
        ASSERT_EQ(std::lround(src.at(x, 3).r * 255.0f), f.rgba[(3 * kWidth + x) * 4]);
    }
}

TEST(KelirVulkanPresent, Stacks1_3_5MatchReference) {
    Node details = MakeNode(kDetails);
    Set(details, p::kClarity, 0.6f);
    for (Format f : {Format::kRgba8, Format::kSrgb, Format::kRgb10a2}) {
        CheckVulkan(StackOf({Setting(kColor, 0.5f)}), f, "1 node");
        CheckVulkan(StackOf({Setting(kBrightnessContrast, 0.4f), Setting(kColor, 0.5f),
                             Setting(kVignette, 0.6f)}),
                    f, "3 nodes");
        CheckVulkan(StackOf({Setting(kBrightnessContrast, 0.4f), Setting(kColor, 0.5f),
                             Setting(kVignette, 0.6f), Setting(kSharpen, 0.7f), details}),
                    f, "5 nodes");
    }
}

// A pre-rotated swapchain: screen nodes follow the display, not the buffer.
TEST(KelirVulkanPresent, PreTransform0_90_180_270ScreenNodesInDisplaySpace) {
    Node letterbox = MakeNode(kLetterbox);
    Set(letterbox, p::kAspect, 3.0f);  // wider than 2:1: bars across the display's long edges
    Stack s = StackOf({letterbox});
    s.split = true;
    s.splitX = 16384;  // a divider at a quarter of the display's width
    Use(s);
    for (VkSurfaceTransformFlagBitsKHR t :
         {VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR,
          VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR, VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR}) {
        VkGame game = Vk(Format::kRgba8, t);
        const Frame f = game.present()[0];
        // Back to display orientation, by the rotation the transform names.
        const auto display = [&](int dx, int dy) -> const uint16_t* {
            int bx = dx;
            int by = dy;
            if (t == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR) {
                bx = f.width - 1 - dy;
                by = dx;
            } else if (t == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR) {
                bx = f.width - 1 - dx;
                by = f.height - 1 - dy;
            } else if (t == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) {
                bx = dy;
                by = f.height - 1 - dx;
            }
            return &f.rgba[(static_cast<size_t>(by) * f.width + bx) * 4];
        };
        // Top bar: display row 2 is black left and right of the divider.
        EXPECT_EQ(0, display(kWidth - 10, 2)[0]) << "transform " << t;
        // Divider at display x = 64.
        EXPECT_GT(display(64, kHeight / 2)[1], 180) << "transform " << t;
    }
}

TEST(KelirVulkanPresent, DeclinesHdrSwapchain) {
    Use(StackOf({Setting(kSepia, 1.0f)}));
    // PQ in 10 bits; extended sRGB in 8 bits cannot leave [0, 1], so it is filtered as sRGB.
    VkGame game(Upload(true), VkOptions{.width = kWidth,
                                        .height = kHeight,
                                        .format = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                        .colorSpace = VK_COLOR_SPACE_HDR10_ST2084_EXT});
    const Frame f = game.present()[0];
    const ref::Image src = Source(Format::kRgb10a2, false);
    EXPECT_EQ(std::lround(src.at(40, 40).r * 1023.0f), f.rgba[(40 * kWidth + 40) * 4]);
    const uint8_t* b = test::TheChannel().status() + kKelirStatusOffset;
    EXPECT_EQ(static_cast<uint8_t>(KelirState::kDeclined), b[0]);
    EXPECT_EQ(static_cast<uint8_t>(KelirReason::kHdr), b[1]);
}

// A new stack's pipelines are built on the watcher thread: no present waits on a compile.
TEST(KelirVulkanPresent, PipelinesBuiltOffPresentThread) {
    const uint8_t detail[] = {kSharpen, kDetails, kTiltShift, kPainterly, kWatercolor};
    const uint8_t screen[] = {kVignette, kFilmGrain, kLetterbox, kOldFilm, kRetro, kHalftone};
    VkGame game = Vk(Format::kRgba8);
    game.present();
    std::vector<double> us;
    for (int i = 0; i < 100; i++) {
        Stack s = StackOf({Setting(detail[i % 5], 0.5f), Setting(kColor, 0.5f),
                           Setting(screen[(i / 5) % 6], 0.5f), Setting(screen[(i / 30 + 1 + (i / 5) % 6) % 6], 0.5f)});
        Use(s);
        usleep(30'000);  // the watcher's build, which runs after the stack is published
        game.present();
        us.push_back((game.presentedNs - game.enteredNs) / 1e3);
    }
    std::sort(us.begin(), us.end());
    printf("kelir vulkan present after a stack change: p50 %.0f us, p99 %.0f us\n", us[50], us[98]);
    // A compile costs milliseconds; a new plan's LUT upload (290 KB) stays well under one.
    EXPECT_LE(us[98], 2000.0);
}

// Pass-through on UNORM is what Android shows as sRGB (Unity games ask for it).
TEST(KelirVulkanPresent, PassThroughUnormIsFiltered) {
    const auto plan = Use(StackOf({Setting(kSepia, 1.0f)}));
    VkGame game(Upload(false), VkOptions{.width = kWidth,
                                         .height = kHeight,
                                         .colorSpace = VK_COLOR_SPACE_PASS_THROUGH_EXT});
    game.present();
    const Frame f = game.present()[0];
    const ref::Image want = ref::Render(*plan, Source(Format::kRgba8, false), RefFrame(Format::kRgba8, 2, false));
    EXPECT_LE(Compare(f, want, Format::kRgba8, false, 0).max, 2.0f) << StatusText();
}

// Unity asks for images bound only at their first acquire; Kelir views them all up front.
TEST(KelirVulkanPresent, DeferredMemorySwapchainIsFiltered) {
    const auto plan = Use(StackOf({Setting(kSepia, 1.0f)}));
    VkGame game(Upload(false),
                VkOptions{.width = kWidth,
                          .height = kHeight,
                          .flags = VK_SWAPCHAIN_CREATE_DEFERRED_MEMORY_ALLOCATION_BIT_EXT});
    game.present();
    const Frame f = game.present()[0];
    const ref::Image want = ref::Render(*plan, Source(Format::kRgba8, false), RefFrame(Format::kRgba8, 2, false));
    EXPECT_LE(Compare(f, want, Format::kRgba8, false, 0).max, 2.0f) << StatusText();
}

// Every pass Kelir records, through the Khronos validation layer: not one message.
TEST(KelirVulkanPresent, ValidationClean) {
    Node details = MakeNode(kDetails);
    Set(details, p::kClarity, 0.6f);
    const Stack stacks[] = {
            StackOf({Setting(kColor, 0.5f)}),
            StackOf({Setting(kSharpen, 0.5f), Setting(kColor, 0.5f), Setting(kVignette, 0.5f)}),
            StackOf({details, Setting(kTiltShift, 0.5f), Setting(kFilmGrain, 0.5f), Setting(kOldFilm, 0.5f)}),
            StackOf({Setting(kPainterly, 0.5f), Setting(kRetro, 0.5f)}),
            StackOf({Setting(kWatercolor, 0.5f), Setting(kHalftone, 0.5f), Setting(kLetterbox, 0.5f)}),
            StackOf({}),
    };
    // With Luwes on, Kelir's finish rides in Luwes's output pass.
    LuwesSettings luwes;
    luwes.multiplier = 2;
    test::ResetValidationMessages();
    for (const LuwesSettings& l : {LuwesSettings(), luwes}) {
        for (Format format : {Format::kRgba8, Format::kSrgb, Format::kRgb10a2}) {
            VkGame game(Upload(format == Format::kRgb10a2),
                        VkOptions{.width = kWidth, .height = kHeight, .format = VkFormatOf(format),
                                  .validation = true});
            // Luwes engages after a steady run of frames.
            Use(stacks[0], l);
            for (int i = 0; i < 30; i++) game.present();
            for (const Stack& s : stacks) {
                Use(s, l);
                for (int i = 0; i < 8; i++) game.present();
            }
        }
    }
    EXPECT_EQ(0, test::ValidationMessages());
}

TEST(KelirVulkanPresent, LeakFree1000Cycles) {
    const Stack stacks[] = {
            StackOf({Setting(kColor, 0.5f)}),
            StackOf({Setting(kSharpen, 0.5f), Setting(kVignette, 0.5f)}),
            StackOf({Setting(kDetails, 0.5f), Setting(kFilmGrain, 0.5f)}),
            StackOf({Setting(kWatercolor, 0.5f)}),
    };
    VkGame game = Vk(Format::kRgba8);
    long rss = 0;
    for (int i = 0; i < 1000; i++) {
        if (i % 50 == 0) Use(stacks[(i / 50) % 4]);
        game.present();
        if (i == 200) rss = test::RssKb();
    }
    EXPECT_LT(test::RssKb() - rss, 8 * 1024) << "kB grown";
}

// ---- KelirPerf -------------------------------------------------------------------------------

// GPU time of the pass at the panel's full size, from the inspect timers, with the CPU time of
// the present around it.
struct Cost {
    double gpuMs = 0.0;
    double cpuP95Us = 0.0;
};

template <typename Game>
Cost Measure(Game& game, const Stack& stack) {
    Stack s = stack;
    s.inspect = true;
    Use(s);
    std::vector<double> cpu;
    for (int i = 0; i < 60; i++) {
        const auto start = std::chrono::steady_clock::now();
        game.present();
        cpu.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(cpu.begin(), cpu.end());
    const uint8_t* b = test::TheChannel().status() + kKelirStatusOffset;
    return {ReadU32(b + 20) / 1e6, cpu[cpu.size() * 95 / 100]};
}

TEST(KelirPerf, Budgets1_3_5) {
    constexpr int kPanelW = 1280;
    constexpr int kPanelH = 2772;
    Node details = MakeNode(kDetails);
    Set(details, p::kClarity, 0.6f);
    const Stack stacks[] = {
            StackOf({Setting(kColor, 0.5f)}),
            StackOf({Setting(kBrightnessContrast, 0.4f), Setting(kColor, 0.5f), Setting(kVignette, 0.6f)}),
            StackOf({Setting(kBrightnessContrast, 0.4f), Setting(kColor, 0.5f), Setting(kVignette, 0.6f),
                     Setting(kSharpen, 0.7f), details}),
    };
    const double glesBudget[] = {0.35, 0.4, 2.0};
    const double vkBudget[] = {0.8, 0.85, 1.3};
    {
        GlGame game(Upload(false), GlOptions{.width = kPanelW, .height = kPanelH});
        for (int i = 0; i < 3; i++) {
            const Cost c = Measure(game, stacks[i]);
            printf("kelir perf gles %d nodes: gpu %.3f ms\n", i == 0 ? 1 : i == 1 ? 3 : 5, c.gpuMs);
            EXPECT_GT(c.gpuMs, 0.0);
            EXPECT_LE(c.gpuMs, glesBudget[i]);
        }
    }
    {
        VkGame game(Upload(false), VkOptions{.width = kPanelW, .height = kPanelH});
        for (int i = 0; i < 3; i++) {
            const Cost c = Measure(game, stacks[i]);
            printf("kelir perf vulkan %d nodes: gpu %.3f ms\n", i == 0 ? 1 : i == 1 ? 3 : 5, c.gpuMs);
            EXPECT_GT(c.gpuMs, 0.0);
            EXPECT_LE(c.gpuMs, vkBudget[i]);
        }
    }
}

TEST(KelirPerf, EmptyStackZeroGpu) {
    Use(StackOf({}));
    GlGame game = Gl(Format::kRgba8);
    game.present();
    EXPECT_EQ(nullptr, game.state) << "nothing armed: no engine state at all";
}

// The engine's own CPU per present: from entering it to the driver's swap or present.
template <typename Game>
double EngineP95Us(Game& game, const Stack& stack, int64_t Game::*handedOff) {
    Use(stack);
    game.present();
    std::vector<double> us;
    for (int i = 0; i < 200; i++) {
        game.present();
        us.push_back((game.*handedOff - game.enteredNs) / 1e3);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() * 95 / 100];
}

TEST(KelirPerf, CpuPerPresent) {
    Node details = MakeNode(kDetails);
    Set(details, p::kClarity, 0.6f);
    const Stack stacks[] = {
            StackOf({Setting(kColor, 0.5f)}),
            StackOf({Setting(kBrightnessContrast, 0.4f), Setting(kColor, 0.5f), Setting(kVignette, 0.6f)}),
            StackOf({Setting(kBrightnessContrast, 0.4f), Setting(kColor, 0.5f), Setting(kVignette, 0.6f),
                     Setting(kSharpen, 0.7f), details}),
    };
    {
        GlGame game = Gl(Format::kRgba8);
        for (const Stack& s : stacks) {
            const double p95 = EngineP95Us(game, s, &GlGame::swappedNs);
            printf("kelir cpu gles %d nodes: p95 %.1f us\n", s.count, p95);
            EXPECT_LE(p95, 30.0);
        }
    }
    {
        VkGame game = Vk(Format::kRgba8);
        for (const Stack& s : stacks) {
            const double p95 = EngineP95Us(game, s, &VkGame::presentedNs);
            printf("kelir cpu vulkan %d nodes: p95 %.1f us\n", s.count, p95);
            EXPECT_LE(p95, 60.0);
        }
    }
}

}  // namespace
}  // namespace framefx::kelir
