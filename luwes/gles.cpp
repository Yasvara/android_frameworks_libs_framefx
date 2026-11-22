/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "gles.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <log/log.h>
#include <system/window.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

#include "kelir/shaders.h"
#include "luwes/pacer.h"
#include "luwes/shaders.h"
#include "luwes/timing.h"
#include "src/engine.h"
#include "src/target.h"

namespace framefx::luwes::gles {
namespace {

constexpr int kMaxSurfaces = 4;
constexpr int kLevels = 4;         // luma pyramid, 1/2 to 1/16 of the frame
constexpr int kAlign = 128;        // frame size aligned so the coarsest level holds whole blocks
constexpr int kSearchRadius = 12;  // coarsest level, its pixels
constexpr float kSmoothness = 0.004f;
constexpr int kFinest = 1;          // motion stops at 1/4 of the frame: 32-pixel blocks
constexpr float kFieldScale = 2 << kFinest;  // finest motion level to frame pixels
constexpr int64_t kRefreshCheckNs = 1'000'000'000;
constexpr int64_t kDefaultRefreshNs = 1'000'000'000 / 60;
// Presents wait in SurfaceFlinger for their refresh, a lead and a window ahead: a window gets as
// many buffers as a swapchain under Luwes, one of them the game's to render into. Without, the
// pacer plans for the usual three.
constexpr int kWindowBuffers = kMaxPresents + 3;
constexpr int kUsualBuffers = 3;

// motion.comp passes.
enum MotionPass { kSearch = 0, kCandidates, kPropagate, kRefine, kMedian, kGlobal };

#define FRAMEFX_GL(X)                                                                       \
    X(glActiveTexture) X(glAttachShader) X(glBindFramebuffer) X(glBindImageTexture)         \
    X(glBindTexture) X(glBlitFramebuffer) X(glCompileShader) X(glCreateProgram)             \
    X(glCreateShader) X(glDeleteFramebuffers) X(glDeleteProgram) X(glDeleteShader)         \
    X(glDeleteTextures) X(glDisable) X(glDispatchCompute) X(glDrawArrays) X(glFlush)        \
    X(glFramebufferTexture2D) X(glGenFramebuffers) X(glGenTextures) X(glGetError)           \
    X(glGetProgramInfoLog) X(glGetProgramiv) X(glGetShaderInfoLog) X(glGetShaderiv)         \
    X(glGetUniformLocation) X(glInvalidateFramebuffer) X(glLinkProgram) X(glMemoryBarrier)  \
    X(glShaderSource) X(glTexParameteri) X(glTexStorage2D) X(glUniform4fv) X(glUseProgram)  \
    X(glViewport)

#define FRAMEFX_EGL(X)                                                                      \
    X(eglCreateContext) X(eglDestroyContext) X(eglMakeCurrent) X(eglGetCurrentContext)      \
    X(eglGetCurrentSurface) X(eglGetCurrentDisplay) X(eglQuerySurface)

// Driver entry points, below libEGL's tables: the private context never goes through them.
struct Driver {
#define DECLARE(name) decltype(&::name) name = nullptr;
    FRAMEFX_GL(DECLARE)
    FRAMEFX_EGL(DECLARE)
#undef DECLARE
    PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR = nullptr;
    PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR = nullptr;
    PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR = nullptr;
    bool complete = false;
};

Driver gl;
std::atomic<int32_t> gRequestedInterval{1};
std::atomic<int> gRunningSurfaces{0};

struct Program {
    GLuint id = 0;
    GLint params = -1;
};

// The output pass with Kelir's finish, for one stage-B signature.
struct Finish {
    Program program;
    GLint plan = -1;
    GLint frame = -1;
    uint64_t signature = 0;
    std::shared_ptr<const kelir::Plan> uploaded;
    uint32_t seed = 0;
    bool broken = false;
};

// One window surface the game context presents to.
struct Surface {
    EGLSurface surface = EGL_NO_SURFACE;
    ANativeWindow* window = nullptr;
    int64_t lastUse = 0;
    Target target;
    Meter meter;
    LuwesStatus status;
    WindowTiming timing;
    int64_t refreshNs = kDefaultRefreshNs;
    int64_t refreshCheckedAt = 0;
    bool running = false;
    float displayWidth = 0.0f;  // the window's size, for Kelir's screen-space finish
    float displayHeight = 0.0f;

    int width = 0;
    int height = 0;
    int alignedWidth = 0;
    int alignedHeight = 0;
    GLuint color[2] = {};
    GLuint colorFb[2] = {};
    GLuint luma[2][kLevels] = {};
    GLuint lumaFb[2][kLevels] = {};
    GLuint field[kLevels][2] = {};
    GLuint global = 0;
    int current = 0;

    int levelWidth(int level) const { return alignedWidth >> (level + 1); }
    int levelHeight(int level) const { return alignedHeight >> (level + 1); }
};

// Engine state of one game context.
struct Context {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLConfig config = nullptr;
    EGLContext own = EGL_NO_CONTEXT;
    LuwesReason declined = LuwesReason::kNone;
    Program pyramid;
    Program motion;
    Program output;
    Finish finish;
    std::vector<std::unique_ptr<Surface>> surfaces;
    Engine::Watch watch;

    ~Context() {
        for (auto& s : surfaces) {
            if (s->running) gRunningSurfaces.fetch_sub(1, std::memory_order_relaxed);
        }
        // Destroying the context frees every object in it.
        if (own != EGL_NO_CONTEXT) gl.eglDestroyContext(display, own);
    }
};

GLuint compile(GLenum stage, const char* body) {
    const GLuint shader = gl.glCreateShader(stage);
    const char* parts[] = {shaders::kGlesPrelude, body};
    gl.glShaderSource(shader, 2, parts, nullptr);
    gl.glCompileShader(shader);
    GLint ok = GL_FALSE;
    gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok != GL_TRUE) {
        char log[512] = {};
        gl.glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        ALOGE("shader compile failed: %s", log);
        gl.glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool link(Program* p, GLuint a, GLuint b) {
    p->id = gl.glCreateProgram();
    gl.glAttachShader(p->id, a);
    if (b != 0) gl.glAttachShader(p->id, b);
    gl.glLinkProgram(p->id);
    GLint ok = GL_FALSE;
    gl.glGetProgramiv(p->id, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        char log[512] = {};
        gl.glGetProgramInfoLog(p->id, sizeof(log), nullptr, log);
        ALOGE("program link failed: %s", log);
        return false;
    }
    p->params = gl.glGetUniformLocation(p->id, "uParams");
    return true;
}

// With the private context current for the first time: the programs.
LuwesReason setUp(Context& c) {
    gl.glDisable(GL_DITHER);
    const GLuint vertex = compile(GL_VERTEX_SHADER, shaders::kFullscreenVert);
    const GLuint pyramid = compile(GL_FRAGMENT_SHADER, shaders::kPyramidFrag);
    const GLuint motion = compile(GL_COMPUTE_SHADER, shaders::kMotionComp);
    const GLuint output = compile(GL_FRAGMENT_SHADER, shaders::kOutputFrag);
    const bool ok = vertex != 0 && pyramid != 0 && motion != 0 && output != 0 &&
            link(&c.pyramid, vertex, pyramid) && link(&c.motion, motion, 0) &&
            link(&c.output, vertex, output);
    for (GLuint s : {vertex, pyramid, motion, output}) {
        if (s != 0) gl.glDeleteShader(s);
    }
    return ok ? LuwesReason::kNone : LuwesReason::kResources;
}

GLuint texture(GLenum format, int width, int height, GLenum filter) {
    GLuint t = 0;
    gl.glGenTextures(1, &t);
    gl.glBindTexture(GL_TEXTURE_2D, t);
    gl.glTexStorage2D(GL_TEXTURE_2D, 1, format, width, height);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(filter));
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(filter));
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

GLuint framebuffer(GLuint tex) {
    GLuint fb = 0;
    gl.glGenFramebuffers(1, &fb);
    gl.glBindFramebuffer(GL_FRAMEBUFFER, fb);
    gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return fb;
}

void freeImages(Surface& s) {
    std::vector<GLuint> textures = {s.color[0], s.color[1], s.global};
    std::vector<GLuint> framebuffers = {s.colorFb[0], s.colorFb[1]};
    for (int p = 0; p < 2; p++) {
        for (int k = 0; k < kLevels; k++) {
            textures.push_back(s.luma[p][k]);
            framebuffers.push_back(s.lumaFb[p][k]);
            textures.push_back(s.field[k][p]);
        }
    }
    gl.glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
    gl.glDeleteFramebuffers(static_cast<GLsizei>(framebuffers.size()), framebuffers.data());
    s.color[0] = s.color[1] = s.colorFb[0] = s.colorFb[1] = s.global = 0;
    std::fill(&s.luma[0][0], &s.luma[0][0] + 2 * kLevels, 0u);
    std::fill(&s.lumaFb[0][0], &s.lumaFb[0][0] + 2 * kLevels, 0u);
    std::fill(&s.field[0][0], &s.field[0][0] + 2 * kLevels, 0u);
    s.width = s.height = 0;
}

bool allocate(Surface& s, int width, int height) {
    freeImages(s);
    s.width = width;
    s.height = height;
    s.alignedWidth = (width + kAlign - 1) / kAlign * kAlign;
    s.alignedHeight = (height + kAlign - 1) / kAlign * kAlign;
    s.current = 0;
    for (int p = 0; p < 2; p++) {
        s.color[p] = texture(GL_RGBA8, width, height, GL_LINEAR);
        s.colorFb[p] = framebuffer(s.color[p]);
        for (int k = 0; k < kLevels; k++) {
            s.luma[p][k] = texture(GL_R8, s.levelWidth(k), s.levelHeight(k), GL_LINEAR);
            s.lumaFb[p][k] = framebuffer(s.luma[p][k]);
        }
    }
    for (int k = kFinest; k < kLevels; k++) {
        for (int j = 0; j < 2; j++) {
            s.field[k][j] =
                    texture(GL_RGBA16F, s.levelWidth(k) / 8, s.levelHeight(k) / 8, GL_LINEAR);
        }
    }
    s.global = texture(GL_RGBA16F, 1, 1, GL_NEAREST);
    return gl.glGetError() == GL_NO_ERROR;
}

void invalidateBound(GLenum attachment) {
    gl.glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &attachment);
}

void bind(GLenum unit, GLuint tex) {
    gl.glActiveTexture(GL_TEXTURE0 + unit);
    gl.glBindTexture(GL_TEXTURE_2D, tex);
}

void motionPass(const Context& c, const Surface& s, MotionPass pass, int level, GLuint parent,
                GLuint input, GLuint global, GLuint out) {
    const int cur = s.current;
    const float params[16] = {static_cast<float>(pass), static_cast<float>(kSearchRadius),
                              kSmoothness, static_cast<float>(1 << (kLevels - 1 - level))};
    gl.glUniform4fv(c.motion.params, 4, params);
    bind(0, s.luma[cur][level]);
    bind(1, s.luma[1 - cur][level]);
    bind(2, parent);
    bind(3, input);
    bind(4, global);
    gl.glBindImageTexture(0, out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    const GLuint bw = static_cast<GLuint>(s.levelWidth(level) / 8);
    const GLuint bh = static_cast<GLuint>(s.levelHeight(level) / 8);
    if (pass == kGlobal) {
        gl.glDispatchCompute(1, 1, 1);
    } else {
        gl.glDispatchCompute(bw, bh, 1);  // a workgroup per block
    }
    gl.glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

// The luma pyramid of the captured frame and, with history, the motion field against the
// previous one. Same passes and order as the Vulkan backend.
void analyse(const Context& c, Surface& s, bool history) {
    const int cur = s.current;
    gl.glUseProgram(c.pyramid.id);
    for (int k = 0; k < kLevels; k++) {
        gl.glBindFramebuffer(GL_FRAMEBUFFER, s.lumaFb[cur][k]);
        invalidateBound(GL_COLOR_ATTACHMENT0);
        gl.glViewport(0, 0, s.levelWidth(k), s.levelHeight(k));
        const float sw = static_cast<float>(k == 0 ? s.width : s.levelWidth(k - 1));
        const float sh = static_cast<float>(k == 0 ? s.height : s.levelHeight(k - 1));
        const float params[16] = {1.0f / sw, 1.0f / sh, k == 0 ? 1.0f : 0.0f};
        gl.glUniform4fv(c.pyramid.params, 4, params);
        bind(0, k == 0 ? s.color[cur] : s.luma[cur][k - 1]);
        gl.glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    if (!history) return;

    gl.glUseProgram(c.motion.id);
    const int top = kLevels - 1;
    const auto& f = s.field;
    motionPass(c, s, kSearch, top, f[top][1], f[top][1], s.global, f[top][0]);
    motionPass(c, s, kMedian, top, f[top][0], f[top][0], s.global, f[top][1]);
    motionPass(c, s, kGlobal, top, f[top][0], f[top][1], f[top][0], s.global);
    for (int k = top - 1; k >= kFinest; k--) {
        const GLuint parent = k + 1 == top ? f[top][1] : f[k + 1][0];
        motionPass(c, s, kCandidates, k, parent, f[k][1], s.global, f[k][0]);
        motionPass(c, s, kPropagate, k, parent, f[k][0], s.global, f[k][1]);
        motionPass(c, s, kRefine, k, parent, f[k][1], s.global, f[k][0]);
        motionPass(c, s, kRefine, k, parent, f[k][0], s.global, f[k][1]);
        motionPass(c, s, kMedian, k, parent, f[k][1], s.global, f[k][0]);
    }
}

// The output pass fused with Kelir's finish for the plan's stage B; nullptr when it cannot be
// built, and Kelir's finish is then left out.
Finish* finishFor(Context& c, const std::shared_ptr<const kelir::Plan>& plan) {
    kelir::Target target;
    target.role = kelir::Role::kStageB;
    const uint64_t signature = kelir::Signature(*plan, target);
    Finish& f = c.finish;
    if (f.program.id != 0 && f.signature == signature) return &f;
    if (f.broken && f.signature == signature) return nullptr;
    if (f.program.id != 0) gl.glDeleteProgram(f.program.id);
    f = Finish();
    f.signature = signature;
    const GLuint vertex = compile(GL_VERTEX_SHADER, shaders::kFullscreenVert);
    const std::string body = kelir::shaders::GlesFinish(signature) + shaders::kOutputFrag;
    const GLuint fragment = compile(GL_FRAGMENT_SHADER, body.c_str());
    const bool ok = vertex != 0 && fragment != 0 && link(&f.program, vertex, fragment);
    for (GLuint sh : {vertex, fragment}) {
        if (sh != 0) gl.glDeleteShader(sh);
    }
    if (!ok) {
        f.broken = true;
        ALOGW("%s: kelir finish for frame generation did not build; generated frames go out without "
              "it", Engine::get().package().c_str());
        return nullptr;
    }
    f.plan = gl.glGetUniformLocation(f.program.id, "uKelirPlan");
    f.frame = gl.glGetUniformLocation(f.program.id, "uKelirFrame");
    return &f;
}

// Draws the frame at phase t into the default framebuffer: generated, or the held real one at
// its own phase; with Kelir's finish when `finish` is given.
void draw(const Context& c, const Surface& s, Mode mode, float t, Finish* finish,
          const std::shared_ptr<const kelir::Plan>& plan) {
    const int cur = s.current;
    gl.glBindFramebuffer(GL_FRAMEBUFFER, 0);
    invalidateBound(GL_COLOR);
    gl.glViewport(0, 0, s.width, s.height);
    const float params[16] = {
            static_cast<float>(s.width),
            static_cast<float>(s.height),
            static_cast<float>(s.alignedWidth),
            static_cast<float>(s.alignedHeight),
            t,
            static_cast<float>(mode),
            kFieldScale,
    };
    const Program& program = finish != nullptr ? finish->program : c.output;
    gl.glUseProgram(program.id);
    gl.glUniform4fv(program.params, 4, params);
    if (finish != nullptr) {
        if (finish->uploaded != plan) {
            float packed[kelir::kPlanVec4s * 4];
            kelir::PackPlan(*plan, packed);
            gl.glUniform4fv(finish->plan, kelir::kPlanVec4s, packed);
            finish->uploaded = plan;
        }
        kelir::FrameInfo info;
        info.bufferWidth = s.width;
        info.bufferHeight = s.height;
        info.displayWidth = s.displayWidth;
        info.displayHeight = s.displayHeight;
        info.flipY = true;
        info.seed = ++finish->seed;
        float frame[kelir::kFrameVec4s * 4];
        kelir::PackFrame(*plan, info, frame);
        gl.glUniform4fv(finish->frame, kelir::kFrameVec4s, frame);
    }
    bind(0, s.color[cur]);
    bind(1, s.color[1 - cur]);
    bind(2, s.field[kFinest][0]);
    gl.glDrawArrays(GL_TRIANGLES, 0, 3);
}

// Redraws the held real frame into the default framebuffer.
void copyReal(const Surface& s) {
    gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, s.colorFb[s.current]);
    gl.glBlitFramebuffer(0, 0, s.width, s.height, 0, 0, s.width, s.height, GL_COLOR_BUFFER_BIT,
                         GL_NEAREST);
}

Surface& surfaceFor(Context& c, const framefx_gl_frame* frame, int64_t now) {
    for (auto& s : c.surfaces) {
        if (s->surface == frame->surface && s->window == frame->window) {
            s->lastUse = now;
            return *s;
        }
    }
    if (c.surfaces.size() >= kMaxSurfaces) {
        // Surfaces are destroyed without telling us, and their window with them: take over the
        // one idle longest without touching its window. Its images go with the next resize.
        Surface& s = **std::min_element(c.surfaces.begin(), c.surfaces.end(),
                                        [](const auto& a, const auto& b) {
                                            return a->lastUse < b->lastUse;
                                        });
        if (s.running) gRunningSurfaces.fetch_sub(1, std::memory_order_relaxed);
        s.surface = frame->surface;
        s.window = frame->window;
        s.lastUse = now;
        s.target = Target();
        s.meter = Meter();
        s.status = LuwesStatus();
        s.timing = WindowTiming();
        s.refreshCheckedAt = 0;
        s.running = false;
        return s;
    }
    auto s = std::make_unique<Surface>();
    s->surface = frame->surface;
    s->window = frame->window;
    s->lastUse = now;
    c.surfaces.push_back(std::move(s));
    return *c.surfaces.back();
}

void setRunning(Surface& s, bool running) {
    if (s.running == running) return;
    s.running = running;
    gRunningSurfaces.fetch_add(running ? 1 : -1, std::memory_order_relaxed);
    // Generated frames must not be dropped by an unthrottled queue.
    const int32_t requested = gRequestedInterval.load(std::memory_order_relaxed);
    if (requested < 1) s.window->setSwapInterval(s.window, running ? 1 : requested);
    if (running) {
        const bool more = native_window_set_buffer_count(s.window, kWindowBuffers) == 0;
        s.target.pacer().buffers((more ? kWindowBuffers : kUsualBuffers) - 1);
    } else {
        s.timing.reset(s.window);
    }
}

void report(Surface& s, const Context& c, const LuwesSettings& luwes, LuwesState state,
            int generated, bool paced) {
    LuwesStatus& st = s.status;
    st.state = state;
    st.reason = c.declined;
    st.api = Api::kGles;
    st.source = MotionSource::kBlockSearch;
    st.width = static_cast<uint16_t>(std::max(s.width, 0));
    st.height = static_cast<uint16_t>(std::max(s.height, 0));
    st.multiplier = static_cast<uint8_t>(luwes.multiplier);
    st.generated = static_cast<uint8_t>(generated);
    st.mode = luwes.mode;
    st.paced = paced;
    st.realFps = s.meter.realFps();
    st.presentedFps = s.meter.presentedFps();
    st.panelHz = 1e9f / static_cast<float>(s.refreshNs);
    Engine::get().status().writeLuwes(st);
}

}  // namespace

void attach(const framefx_gl_host* host) {
    Engine::get().attach();
#define RESOLVE(name) gl.name = reinterpret_cast<decltype(gl.name)>(host->proc(#name));
    FRAMEFX_GL(RESOLVE)
    FRAMEFX_EGL(RESOLVE)
    RESOLVE(eglCreateSyncKHR)
    RESOLVE(eglDestroySyncKHR)
    RESOLVE(eglWaitSyncKHR)
#undef RESOLVE
    bool complete = gl.eglCreateSyncKHR != nullptr && gl.eglDestroySyncKHR != nullptr &&
            gl.eglWaitSyncKHR != nullptr;
#define CHECK(name) complete = complete && gl.name != nullptr;
    FRAMEFX_GL(CHECK)
    FRAMEFX_EGL(CHECK)
#undef CHECK
    gl.complete = complete;
    if (!complete) ALOGW("GLES entry points missing; Luwes stays off on GLES");
}

int32_t present(const framefx_gl_frame* frame, const std::shared_ptr<const kelir::Plan>& kelir) {
    Engine& engine = Engine::get();
    auto* c = static_cast<Context*>(*frame->state);
    if (c == nullptr) {
        if (!engine.active() || !gl.complete) return FRAMEFX_PASS;
        c = new Context();
        c->display = frame->display;
        c->config = frame->config;
        *frame->state = c;
    }
    c->watch.refresh(engine);
    const LuwesSettings& luwes = c->watch.config().luwes;
    // Off, with nothing left to release: the swap is the game's own.
    if (!luwes.on() && std::none_of(c->surfaces.begin(), c->surfaces.end(),
                                    [](const auto& s) { return s->running; })) {
        return FRAMEFX_PASS;
    }
    const int64_t now = Now();
    Surface& s = surfaceFor(*c, frame, now);

    EGLint width = 0;
    EGLint height = 0;
    gl.eglQuerySurface(frame->display, frame->surface, EGL_WIDTH, &width);
    gl.eglQuerySurface(frame->display, frame->surface, EGL_HEIGHT, &height);
    if (now - s.refreshCheckedAt > kRefreshCheckNs) {
        int64_t ns = 0;
        if (native_window_get_refresh_cycle_duration(frame->window, &ns) == 0 && ns > 0) {
            s.refreshNs = ns;
        }
        int w = 0;
        int h = 0;
        if (frame->window->query(frame->window, NATIVE_WINDOW_DEFAULT_WIDTH, &w) != 0 ||
            frame->window->query(frame->window, NATIVE_WINDOW_DEFAULT_HEIGHT, &h) != 0 || w <= 0 ||
            h <= 0) {
            w = width;
            h = height;
        }
        s.displayWidth = static_cast<float>(w);
        s.displayHeight = static_cast<float>(h);
        s.refreshCheckedAt = now;
    }

    const Target::Decision d =
            s.target.begin({now, width, height, 0, s.refreshNs}, luwes);
    if (d.releaseLuwes) {
        setRunning(s, false);
        report(s, *c, luwes, LuwesState::kOff, 0, false);
    }
    if (!d.process || c->declined != LuwesReason::kNone) {
        s.timing.stamp(frame->window, 0);
        if (luwes.on() && s.meter.frame(now, 1)) {
            report(s, *c, luwes,
                   c->declined != LuwesReason::kNone ? LuwesState::kDeclined : LuwesState::kWaiting,
                   0, false);
        }
        return FRAMEFX_PASS;
    }

    // Order the game's rendering before ours on the GPU, then switch to the private context.
    const EGLDisplay gameDisplay = gl.eglGetCurrentDisplay();
    const EGLContext gameContext = gl.eglGetCurrentContext();
    const EGLSurface gameDraw = gl.eglGetCurrentSurface(EGL_DRAW);
    const EGLSurface gameRead = gl.eglGetCurrentSurface(EGL_READ);
    const EGLSyncKHR fence = gl.eglCreateSyncKHR(frame->display, EGL_SYNC_FENCE_KHR, nullptr);
    gl.glFlush();

    bool fresh = false;
    if (c->own == EGL_NO_CONTEXT) {
        const EGLint attribs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2,
                                  EGL_NONE};
        c->own = gl.eglCreateContext(frame->display, frame->config, EGL_NO_CONTEXT, attribs);
        fresh = true;
    }
    if (c->own == EGL_NO_CONTEXT ||
        gl.eglMakeCurrent(frame->display, frame->surface, frame->surface, c->own) != EGL_TRUE) {
        if (fence != EGL_NO_SYNC_KHR) gl.eglDestroySyncKHR(frame->display, fence);
        c->declined = LuwesReason::kResources;
        ALOGW("%s: no private GLES context; Luwes declined", engine.package().c_str());
        return FRAMEFX_PASS;
    }
    if (fence != EGL_NO_SYNC_KHR) {
        gl.eglWaitSyncKHR(frame->display, fence, 0);
        gl.eglDestroySyncKHR(frame->display, fence);
    }
    if (fresh) {
        c->declined = setUp(*c);
        if (c->declined == LuwesReason::kNone) {
            ALOGI("%s: luwes on GLES, pyramid block search and warp", engine.package().c_str());
        } else {
            ALOGW("%s: luwes declined on GLES (reason %d)", engine.package().c_str(),
                  static_cast<int>(c->declined));
        }
    }

    EGLBoolean result = EGL_TRUE;
    bool presentedReal = false;
    int presents = 0;
    int generated = 0;
    if (c->declined == LuwesReason::kNone && (d.resize || s.width != width || s.height != height) &&
        !allocate(s, width, height)) {
        c->declined = LuwesReason::kResources;
        freeImages(s);
    }
    luwes::Pacer& pacer = s.target.pacer();
    if (c->declined == LuwesReason::kNone) {
        if (!s.running) {
            ALOGI("%s: luwes running on GLES %dx%d, %dx %s", engine.package().c_str(), width,
                  height, luwes.multiplier,
                  luwes.mode == Mode::kInterpolate ? "interpolate" : "extrapolate");
            setRunning(s, true);
        }
        s.timing.take(frame->window);
        s.timing.learn(frame->window, pacer, s.refreshNs);
        const int cur = s.current;
        gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, s.colorFb[cur]);
        gl.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT,
                             GL_NEAREST);
        analyse(*c, s, d.generate);

        Present order[kMaxPresents];
        const int count =
                pacer.schedule(Now(), d.plan, luwes.mode, d.generate, s.refreshNs, order);
        Finish* finish =
                kelir::Schedule(kelir.get(), true, false).stageB ? finishFor(*c, kelir) : nullptr;
        // The real frame's own phase: where the output pass reproduces it exactly.
        const float realPhase = luwes.mode == Mode::kInterpolate ? 1.0f : 0.0f;
        for (int i = 0; i < count; i++) {
            const Present& p = order[i];
            if (p.generated) {
                draw(*c, s, luwes.mode, p.phase, finish, kelir);
                generated++;
            } else if (finish != nullptr) {
                draw(*c, s, luwes.mode, realPhase, finish, kelir);
            } else if (i > 0) {
                // Earlier presents took the default framebuffer.
                copyReal(s);
            }
            s.timing.expect(frame->window, p.desired);
            s.timing.stamp(frame->window, p.desired);
            if (p.own) {
                result = frame->swap(frame->cookie);
                presentedReal = true;
                presents++;
            } else if (frame->swap(frame->cookie) == EGL_TRUE) {
                presents++;
            }
        }
        s.target.captured();
        s.current = 1 - cur;
    }
    gl.eglMakeCurrent(gameDisplay, gameDraw, gameRead, gameContext);
    if (!presentedReal) return FRAMEFX_PASS;
    s.status.framesGenerated += static_cast<uint32_t>(generated);

    const int64_t wake = pacer.limit(Now());
    if (wake > 0) SleepUntil(wake);
    pacer.resumed(Now());
    if (s.meter.frame(now, presents)) {
        report(s, *c, luwes, LuwesState::kRunning, d.generate ? d.plan.multiplier - 1 : 0,
               d.plan.paced);
    }
    return result;
}

void release(void* state) {
    delete static_cast<Context*>(state);
}

int32_t swapInterval(int32_t requested) {
    gRequestedInterval.store(requested, std::memory_order_relaxed);
    return gRunningSurfaces.load(std::memory_order_relaxed) > 0 ? std::max(requested, 1)
                                                                 : requested;
}

}  // namespace framefx::luwes::gles
