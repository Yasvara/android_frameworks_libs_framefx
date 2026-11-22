/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "kelir/gles.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <GLES2/gl2ext.h>
#include <log/log.h>
#include <system/window.h>

#include <algorithm>
#include <cstring>
#include <string>

#include "kelir/cache.h"
#include "kelir/shaders.h"
#include "src/engine.h"

namespace framefx::kelir::gles {
namespace {

// QCOM_shading_rate and EXT_fragment_shading_rate; not in this tree's headers.
constexpr GLenum kShadingRateQcom = 0x96A4;
constexpr GLenum kShadingRateExt = 0x96D0;
constexpr GLenum kShadingRate1x1 = 0x96A6;

constexpr int kPyramidLevels = 5;
constexpr int kTimers = 4;

#define KELIR_GL(X)                                                                           \
    X(glActiveTexture) X(glAttachShader) X(glBeginQuery) X(glBindBuffer) X(glBindFramebuffer) \
    X(glBindSampler) X(glBindTexture) X(glBindVertexArray) X(glBlitFramebuffer)              \
    X(glColorMask) X(glCompileShader) X(glCreateProgram) X(glCreateShader)                   \
    X(glDeleteProgram) X(glDeleteShader) X(glDisable) X(glDrawArrays) X(glDrawBuffers)        \
    X(glEnable) X(glEndQuery) X(glFramebufferTexture2D) X(glGenFramebuffers) X(glGenQueries)  \
    X(glGenTextures) X(glGenVertexArrays) X(glGetBooleanv) X(glCheckFramebufferStatus)       \
    X(glGetFramebufferAttachmentParameteriv)                                                  \
    X(glGetIntegerv) X(glGetProgramInfoLog) X(glGetProgramiv) X(glGetQueryObjectuiv)          \
    X(glGetShaderInfoLog) X(glGetShaderiv) X(glGetString) X(glGetUniformLocation)             \
    X(glInvalidateFramebuffer) X(glIsEnabled) X(glLinkProgram) X(glPixelStorei)               \
    X(glReadBuffer) X(glShaderSource) X(glTexParameteri) X(glTexStorage2D) X(glTexStorage3D)  \
    X(glTexSubImage3D) X(glUniform1i) X(glUniform4fv) X(glUseProgram) X(glViewport)

#define KELIR_EGL(X) X(eglGetConfigAttrib) X(eglQuerySurface)

using ShadingRateFn = void (*)(GLenum rate);

// Driver entry points, below libEGL's tables: Kelir draws at the buffer's real size, under any
// render scale the game's own table applies.
struct Driver {
#define DECLARE(name) decltype(&::name) name = nullptr;
    KELIR_GL(DECLARE)
    KELIR_EGL(DECLARE)
#undef DECLARE
    ShadingRateFn glShadingRateQCOM = nullptr;
    ShadingRateFn glShadingRateEXT = nullptr;
    bool complete = false;
};

Driver gl;

struct Program {
    GLuint id = 0;
    GLint plan = -1;
    GLint frame = -1;
    std::shared_ptr<const Plan> uploaded;  // whose constants it holds
};

// One window surface the context presents to: what it is, and its copies when a stack reads
// neighbours.
struct Surface {
    EGLSurface handle = EGL_NO_SURFACE;
    KelirReason declined = KelirReason::kNone;
    FormatClass format = FormatClass::kUnorm8;
    Primaries primaries = Primaries::kSrgb;
    bool msaa = false;
    bool alpha = true;
    int width = 0;
    int height = 0;
    float displayWidth = 0.0f;
    float displayHeight = 0.0f;

    GLuint scratch = 0;  // the frame, copied
    GLuint scratchFb = 0;
    GLuint pyramid = 0;  // its quarter-resolution levels
    GLuint pyramidFb[kPyramidLevels] = {};
    bool copiesComplete = false;
    bool pyramidComplete = false;
};

}  // namespace

struct Context {
    bool ready = false;
    KelirReason declined = KelirReason::kNone;
    bool fetch = false;
    bool timer = false;
    GLint units[3] = {};  // source, pyramid, LUT: the context's top texture units
    GLuint vao = 0;
    GLuint vertex = 0;
    GLuint pyramidProgram = 0;
    GLint pyramidParams = -1;
    GLuint lut[2] = {};
    int lutFront = -1;
    std::shared_ptr<const Luts> lutsHeld;  // what lut[lutFront] holds
    Primaries lutPrimaries = Primaries::kSrgb;
    Lru<Program> programs;
    Surface surface;
    uint32_t seed = 0;
    GLuint timers[kTimers] = {};
    int timerNext = 0;
    bool timerPending[kTimers] = {};
    uint32_t gpuNs = 0;
    KelirTargetStatus reported;
};

namespace {

enum Unit { kSource = 0, kPyramid = 1, kLut = 2 };

bool has(const char* extensions, const char* name) {
    const size_t n = strlen(name);
    for (const char* p = extensions; (p = strstr(p, name)) != nullptr; p += n) {
        if ((p == extensions || p[-1] == ' ') && (p[n] == ' ' || p[n] == '\0')) return true;
    }
    return false;
}

GLuint compile(GLenum stage, const std::string& source) {
    const GLuint shader = gl.glCreateShader(stage);
    const char* text = source.c_str();
    gl.glShaderSource(shader, 1, &text, nullptr);
    gl.glCompileShader(shader);
    GLint ok = GL_FALSE;
    gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok != GL_TRUE) {
        char log[1024] = {};
        gl.glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        ALOGE("kelir: shader compile failed: %s", log);
        gl.glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint link(GLuint vertex, GLuint fragment) {
    const GLuint program = gl.glCreateProgram();
    gl.glAttachShader(program, vertex);
    gl.glAttachShader(program, fragment);
    gl.glLinkProgram(program);
    gl.glDeleteShader(fragment);
    GLint ok = GL_FALSE;
    gl.glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        char log[1024] = {};
        gl.glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        ALOGE("kelir: program link failed: %s", log);
        gl.glDeleteProgram(program);
        return 0;
    }
    return program;
}

void bindSamplers(GLuint program, const Context& c) {
    const char* names[] = {"kelirSource", "kelirPyramid", "kelirLut", "kelirLevelSource"};
    const int units[] = {c.units[kSource], c.units[kPyramid], c.units[kLut], c.units[kSource]};
    for (int i = 0; i < 4; i++) {
        const GLint at = gl.glGetUniformLocation(program, names[i]);
        if (at >= 0) gl.glUniform1i(at, units[i]);
    }
}

// Before anything is saved: whether the context can run Kelir, its extensions and the units
// Kelir takes. Changes no state.
KelirReason probe(Context& c) {
    const char* version = reinterpret_cast<const char*>(gl.glGetString(GL_VERSION));
    int major = 0;
    if (version == nullptr || sscanf(version, "OpenGL ES %d", &major) != 1 || major < 3) {
        return KelirReason::kGlesVersion;
    }
    const char* ext = reinterpret_cast<const char*>(gl.glGetString(GL_EXTENSIONS));
    if (ext == nullptr) ext = "";
    c.fetch = has(ext, "GL_EXT_shader_framebuffer_fetch");
    c.timer = has(ext, "GL_EXT_disjoint_timer_query");
    if (!has(ext, "GL_QCOM_shading_rate")) gl.glShadingRateQCOM = nullptr;
    if (!has(ext, "GL_EXT_fragment_shading_rate")) gl.glShadingRateEXT = nullptr;
    GLint units = 0;
    gl.glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &units);
    if (units < 8) return KelirReason::kResources;
    for (int i = 0; i < 3; i++) c.units[i] = units - 1 - i;
    return KelirReason::kNone;
}

// With the game context current, inside the state snapshot: everything every surface shares.
KelirReason setUp(Context& c) {
    gl.glGenVertexArrays(1, &c.vao);
    c.vertex = compile(GL_VERTEX_SHADER, shaders::GlesVertex());
    const GLuint pyramid = compile(GL_FRAGMENT_SHADER, shaders::GlesPyramid());
    if (c.vertex == 0 || pyramid == 0) return KelirReason::kResources;
    c.pyramidProgram = link(c.vertex, pyramid);
    if (c.pyramidProgram == 0) return KelirReason::kResources;
    gl.glUseProgram(c.pyramidProgram);
    bindSamplers(c.pyramidProgram, c);
    c.pyramidParams = gl.glGetUniformLocation(c.pyramidProgram, "uParams");
    gl.glGenTextures(2, c.lut);
    for (GLuint t : c.lut) {
        gl.glActiveTexture(GL_TEXTURE0 + c.units[kLut]);
        gl.glBindTexture(GL_TEXTURE_3D, t);
        gl.glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA16F, kLutSize, kLutSize, kLutSize);
        gl.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        gl.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        for (GLenum wrap : {GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R}) {
            gl.glTexParameteri(GL_TEXTURE_3D, wrap, GL_CLAMP_TO_EDGE);
        }
    }
    if (c.timer) gl.glGenQueries(kTimers, c.timers);
    return KelirReason::kNone;
}

// The surface as EGL created it: what Kelir cannot take, and how it reads and writes.
void inspectSurface(Surface& s, EGLDisplay display, EGLSurface surface, EGLConfig config) {
    s = Surface{.handle = surface};
    EGLint value = 0;
    if (gl.eglQuerySurface(display, surface, EGL_RENDER_BUFFER, &value) &&
        value == EGL_SINGLE_BUFFER) {
        s.declined = KelirReason::kPreserved;
        return;
    }
    value = EGL_FALSE;
    if (gl.eglQuerySurface(display, surface, EGL_PROTECTED_CONTENT_EXT, &value) && value) {
        s.declined = KelirReason::kProtected;
        return;
    }
    EGLint space = EGL_GL_COLORSPACE_LINEAR_KHR;
    gl.eglQuerySurface(display, surface, EGL_GL_COLORSPACE_KHR, &space);
    switch (space) {
        case EGL_GL_COLORSPACE_LINEAR_KHR:
            break;
        case EGL_GL_COLORSPACE_SRGB_KHR:
            s.format = FormatClass::kSrgb8;
            break;
        case EGL_GL_COLORSPACE_DISPLAY_P3_EXT:
            s.format = FormatClass::kSrgb8;
            s.primaries = Primaries::kP3;
            break;
        case EGL_GL_COLORSPACE_DISPLAY_P3_PASSTHROUGH_EXT:
            s.primaries = Primaries::kP3;
            break;
        default:
            s.declined = KelirReason::kHdr;  // PQ, HLG, scRGB, linear P3
            return;
    }
    EGLint red = 8;
    EGLint alpha = 8;
    EGLint samples = 0;
    EGLint type = EGL_COLOR_COMPONENT_TYPE_FIXED_EXT;
    gl.eglGetConfigAttrib(display, config, EGL_RED_SIZE, &red);
    gl.eglGetConfigAttrib(display, config, EGL_ALPHA_SIZE, &alpha);
    gl.eglGetConfigAttrib(display, config, EGL_SAMPLES, &samples);
    gl.eglGetConfigAttrib(display, config, EGL_COLOR_COMPONENT_TYPE_EXT, &type);
    if (type == EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT) {
        s.declined = KelirReason::kHdr;
        return;
    }
    if (red == 10) {
        if (s.format == FormatClass::kSrgb8) {
            s.declined = KelirReason::kFormat;
            return;
        }
        s.format = FormatClass::kUnorm10;
    } else if (red != 8) {
        s.declined = KelirReason::kFormat;  // RGB565 and the like
        return;
    }
    s.msaa = samples > 1;
    s.alpha = alpha > 0;
}

GLenum scratchFormat(const Surface& s) {
    switch (s.format) {
        case FormatClass::kSrgb8:
            return GL_SRGB8_ALPHA8;
        case FormatClass::kUnorm10:
            return GL_RGB10_A2;
        case FormatClass::kUnorm8:
            break;
    }
    return s.alpha ? GL_RGBA8 : GL_RGB8;
}

void texture2d(GLuint* t, GLenum format, int width, int height, int levels) {
    gl.glGenTextures(1, t);
    gl.glBindTexture(GL_TEXTURE_2D, *t);
    gl.glTexStorage2D(GL_TEXTURE_2D, levels, format, width, height);
    // Linear even on the copy: the pyramid's first level takes bilinear taps between its texels.
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                       levels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

GLuint framebuffer(GLuint texture, int level) {
    GLuint fb = 0;
    gl.glGenFramebuffers(1, &fb);
    gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb);
    gl.glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture,
                              level);
    return fb;
}

int levelSize(int size, int level) {
    return std::max(1, ((size + 3) / 4) >> level);
}

bool complete(GLuint fb) {
    gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb);
    return gl.glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
}

// The frame's copy and pyramid, sized to the surface; made on first need, kept until it resizes.
// Their names die with the context, so a resize just leaves the old ones to it. They are checked
// as objects: a pending GL error is the game's, left for it to read.
bool prepareCopies(Context& c, Surface& s, bool pyramid) {
    gl.glActiveTexture(GL_TEXTURE0 + c.units[kSource]);
    if (s.scratch == 0) {
        texture2d(&s.scratch, scratchFormat(s), s.width, s.height, 1);
        s.scratchFb = framebuffer(s.scratch, 0);
        s.copiesComplete = complete(s.scratchFb);
    }
    if (pyramid && s.pyramid == 0) {
        texture2d(&s.pyramid, GL_RGBA16F, levelSize(s.width, 0), levelSize(s.height, 0),
                  kPyramidLevels);
        s.pyramidComplete = true;
        for (int k = 0; k < kPyramidLevels; k++) {
            s.pyramidFb[k] = framebuffer(s.pyramid, k);
            s.pyramidComplete = s.pyramidComplete && complete(s.pyramidFb[k]);
        }
    }
    return s.copiesComplete && (!pyramid || s.pyramidComplete);
}

void drawTriangle(const Context& c) {
    gl.glBindVertexArray(c.vao);
    gl.glDrawArrays(GL_TRIANGLES, 0, 3);
}

void buildPyramid(const Context& c, const Surface& s, bool readsLinear) {
    gl.glUseProgram(c.pyramidProgram);
    for (int k = 0; k < kPyramidLevels; k++) {
        const int sw = k == 0 ? s.width : levelSize(s.width, k - 1);
        const int sh = k == 0 ? s.height : levelSize(s.height, k - 1);
        const float params[16] = {1.0f / sw, 1.0f / sh, k == 0 ? 1.0f : 0.0f,
                                  k == 0 && readsLinear ? 1.0f : 0.0f};
        gl.glUniform4fv(c.pyramidParams, 4, params);
        gl.glActiveTexture(GL_TEXTURE0 + c.units[kSource]);
        if (k == 0) {
            gl.glBindTexture(GL_TEXTURE_2D, s.scratch);
        } else {
            // Reads only the level above the one it writes.
            gl.glBindTexture(GL_TEXTURE_2D, s.pyramid);
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, k - 1);
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, k - 1);
        }
        gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, s.pyramidFb[k]);
        const GLenum color = GL_COLOR_ATTACHMENT0;
        gl.glInvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 1, &color);
        gl.glViewport(0, 0, levelSize(s.width, k), levelSize(s.height, k));
        drawTriangle(c);
    }
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, kPyramidLevels - 1);
}

// Uploads the plan's LUT into the texture not in use, when it changed.
void prepareLut(Context& c, const Plan& plan, Primaries primaries) {
    if (plan.luts == nullptr) return;
    if (plan.luts == c.lutsHeld && primaries == c.lutPrimaries) return;
    const int back = c.lutFront < 0 ? 0 : 1 - c.lutFront;
    GLint unpack[6];
    const GLenum names[] = {GL_PIXEL_UNPACK_BUFFER_BINDING, GL_UNPACK_ALIGNMENT,
                            GL_UNPACK_ROW_LENGTH,          GL_UNPACK_IMAGE_HEIGHT,
                            GL_UNPACK_SKIP_PIXELS,         GL_UNPACK_SKIP_ROWS};
    for (int i = 0; i < 6; i++) gl.glGetIntegerv(names[i], &unpack[i]);
    GLint skipImages = 0;
    gl.glGetIntegerv(GL_UNPACK_SKIP_IMAGES, &skipImages);
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    for (int i = 1; i < 6; i++) gl.glPixelStorei(names[i], i == 1 ? 4 : 0);
    gl.glPixelStorei(GL_UNPACK_SKIP_IMAGES, 0);
    gl.glActiveTexture(GL_TEXTURE0 + c.units[kLut]);
    gl.glBindTexture(GL_TEXTURE_3D, c.lut[back]);
    gl.glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kLutSize, kLutSize, kLutSize, GL_RGBA,
                       GL_HALF_FLOAT, plan.luts->get(primaries).texels.data());
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(unpack[0]));
    for (int i = 1; i < 6; i++) gl.glPixelStorei(names[i], unpack[i]);
    gl.glPixelStorei(GL_UNPACK_SKIP_IMAGES, skipImages);
    c.lutFront = back;
    c.lutsHeld = plan.luts;
    c.lutPrimaries = primaries;
}

// The pass for a signature, from the cache or compiled now (no parallel compile on this GPU:
// it costs a frame only while the editor changes the stack's shape).
Program* programFor(Context& c, uint64_t signature) {
    if (Program* p = c.programs.find(signature)) return p;
    Program p;
    const GLuint fragment = compile(GL_FRAGMENT_SHADER, shaders::GlesKelir(signature));
    if (fragment != 0) p.id = link(c.vertex, fragment);
    if (p.id == 0) return nullptr;
    gl.glUseProgram(p.id);
    bindSamplers(p.id, c);
    p.plan = gl.glGetUniformLocation(p.id, "uKelirPlan");
    p.frame = gl.glGetUniformLocation(p.id, "uKelirFrame");
    const Program evicted = c.programs.insert(signature, p);
    if (evicted.id != 0) gl.glDeleteProgram(evicted.id);
    return c.programs.find(signature);
}

// Every piece of state the passes touch, as the game left it.
struct Saved {
    GLint program = 0;
    GLint drawFb = 0;
    GLint readFb = 0;
    GLint vao = 0;
    GLint active = 0;
    GLint viewport[4] = {};
    GLboolean mask[4] = {};
    GLint texture[3] = {};  // 2D on source and pyramid units, 3D on the LUT unit
    GLint sampler[3] = {};
    GLint drawBuffer = GL_BACK;
    GLint readBuffer = GL_BACK;
    GLint shadingRate = 0;
    bool enabled[10] = {};
};

constexpr GLenum kCaps[10] = {GL_BLEND,
                              GL_DEPTH_TEST,
                              GL_STENCIL_TEST,
                              GL_SCISSOR_TEST,
                              GL_CULL_FACE,
                              GL_RASTERIZER_DISCARD,
                              GL_SAMPLE_ALPHA_TO_COVERAGE,
                              GL_SAMPLE_COVERAGE,
                              GL_SAMPLE_MASK,
                              GL_POLYGON_OFFSET_FILL};

void save(const Context& c, Saved& s) {
    gl.glGetIntegerv(GL_CURRENT_PROGRAM, &s.program);
    gl.glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s.drawFb);
    gl.glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.readFb);
    gl.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
    gl.glGetIntegerv(GL_ACTIVE_TEXTURE, &s.active);
    gl.glGetIntegerv(GL_VIEWPORT, s.viewport);
    gl.glGetBooleanv(GL_COLOR_WRITEMASK, s.mask);
    for (int u = 0; u < 3; u++) {
        gl.glActiveTexture(GL_TEXTURE0 + c.units[u]);
        gl.glGetIntegerv(u == kLut ? GL_TEXTURE_BINDING_3D : GL_TEXTURE_BINDING_2D, &s.texture[u]);
        gl.glGetIntegerv(GL_SAMPLER_BINDING, &s.sampler[u]);
    }
    for (int i = 0; i < 10; i++) s.enabled[i] = gl.glIsEnabled(kCaps[i]) == GL_TRUE;
    if (gl.glShadingRateQCOM != nullptr) gl.glGetIntegerv(kShadingRateQcom, &s.shadingRate);
    if (gl.glShadingRateEXT != nullptr) gl.glGetIntegerv(kShadingRateExt, &s.shadingRate);
}

void neutral(const Context& c) {
    for (GLenum cap : kCaps) gl.glDisable(cap);
    gl.glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    for (int u = 0; u < 3; u++) gl.glBindSampler(static_cast<GLuint>(c.units[u]), 0);
    if (gl.glShadingRateQCOM != nullptr) gl.glShadingRateQCOM(kShadingRate1x1);
    if (gl.glShadingRateEXT != nullptr) gl.glShadingRateEXT(kShadingRate1x1);
}

void restore(const Context& c, const Saved& s) {
    for (int i = 0; i < 10; i++) {
        if (s.enabled[i]) gl.glEnable(kCaps[i]);
    }
    gl.glColorMask(s.mask[0], s.mask[1], s.mask[2], s.mask[3]);
    for (int u = 0; u < 3; u++) {
        gl.glActiveTexture(GL_TEXTURE0 + c.units[u]);
        gl.glBindTexture(u == kLut ? GL_TEXTURE_3D : GL_TEXTURE_2D, static_cast<GLuint>(s.texture[u]));
        gl.glBindSampler(static_cast<GLuint>(c.units[u]), static_cast<GLuint>(s.sampler[u]));
    }
    gl.glActiveTexture(static_cast<GLenum>(s.active));
    gl.glViewport(s.viewport[0], s.viewport[1], s.viewport[2], s.viewport[3]);
    gl.glBindVertexArray(static_cast<GLuint>(s.vao));
    gl.glUseProgram(static_cast<GLuint>(s.program));
    gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(s.drawFb));
    gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(s.readFb));
    if (gl.glShadingRateQCOM != nullptr) gl.glShadingRateQCOM(static_cast<GLenum>(s.shadingRate));
    if (gl.glShadingRateEXT != nullptr) gl.glShadingRateEXT(static_cast<GLenum>(s.shadingRate));
}

// Inspect only: GPU time of Kelir's work, from timer queries read back when ready.
void collectTimers(Context& c) {
    for (int i = 0; i < kTimers; i++) {
        if (!c.timerPending[i]) continue;
        GLuint available = GL_FALSE;
        gl.glGetQueryObjectuiv(c.timers[i], GL_QUERY_RESULT_AVAILABLE, &available);
        if (!available) continue;
        GLuint ns = 0;
        gl.glGetQueryObjectuiv(c.timers[i], GL_QUERY_RESULT, &ns);
        c.timerPending[i] = false;
        GLint disjoint = 0;
        gl.glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
        if (!disjoint) c.gpuNs = c.gpuNs == 0 ? ns : (c.gpuNs * 7 + ns) / 8;
    }
}

bool beginTimer(Context& c) {
    GLint active = 0;
    gl.glGetIntegerv(GL_TIME_ELAPSED_EXT, &active);
    if (active != 0 || c.timerPending[c.timerNext]) return false;  // the game times, or busy
    gl.glBeginQuery(GL_TIME_ELAPSED_EXT, c.timers[c.timerNext]);
    return true;
}

void endTimer(Context& c) {
    gl.glEndQuery(GL_TIME_ELAPSED_EXT);
    c.timerPending[c.timerNext] = true;
    c.timerNext = (c.timerNext + 1) % kTimers;
}

void report(Context& c, const Surface& s, KelirState state, KelirReason reason, KelirPath path,
            uint64_t signature) {
    KelirTargetStatus t;
    t.state = state;
    t.reason = reason;
    t.api = Api::kGles;
    t.format = s.format;
    t.width = static_cast<uint16_t>(s.width);
    t.height = static_cast<uint16_t>(s.height);
    t.path = path;
    t.signature = signature;
    t.gpuNs = c.gpuNs;
    if (t == c.reported) return;
    c.reported = t;
    Engine::get().reportKelir(t);
}

void displaySize(Surface& s, ANativeWindow* window) {
    int w = 0;
    int h = 0;
    if (window->query(window, NATIVE_WINDOW_DEFAULT_WIDTH, &w) != 0 ||
        window->query(window, NATIVE_WINDOW_DEFAULT_HEIGHT, &h) != 0 || w <= 0 || h <= 0) {
        w = s.width;
        h = s.height;
    }
    s.displayWidth = static_cast<float>(w);
    s.displayHeight = static_cast<float>(h);
}

}  // namespace

void attach(const framefx_gl_host* host) {
#define RESOLVE(name) gl.name = reinterpret_cast<decltype(gl.name)>(host->proc(#name));
    KELIR_GL(RESOLVE)
    KELIR_EGL(RESOLVE)
    RESOLVE(glShadingRateQCOM)
    RESOLVE(glShadingRateEXT)
#undef RESOLVE
    bool complete = true;
#define CHECK(name) complete = complete && gl.name != nullptr;
    KELIR_GL(CHECK)
    KELIR_EGL(CHECK)
#undef CHECK
    gl.complete = complete;
    if (!complete) ALOGW("kelir: GLES entry points missing; Kelir stays off on GLES");
}

namespace {

// Whether the pass for `role` reads pixels around its own, which rules out framebuffer fetch.
bool needsSource(const Plan& plan, Role role) {
    switch (role) {
        case Role::kStageA:
            return plan.detailCount > 0;
        case Role::kStageB:
            return plan.has(kRetro);
        case Role::kFull:
            break;
    }
    return plan.needsSource();
}

// The Kelir pass for `role` over the default framebuffer: FRAMEFX_PASS when nothing was drawn;
// otherwise, for stage A, 1, and for the rest the result of the swap that presents it.
int32_t run(Context** slot, const framefx_gl_frame* frame, const std::shared_ptr<const Plan>& planRef,
            Role role) {
    const Plan& plan = *planRef;
    const Stages stages = Schedule(&plan, role != Role::kFull, true);
    const bool wanted = role == Role::kFull     ? stages.full
                        : role == Role::kStageA ? stages.stageA
                                                : stages.stageB;
    if (!wanted || !gl.complete) return FRAMEFX_PASS;
    if (*slot == nullptr) *slot = new Context();
    Context& c = **slot;
    if (c.declined != KelirReason::kNone) return FRAMEFX_PASS;

    Surface& s = c.surface;
    EGLint width = 0;
    EGLint height = 0;
    gl.eglQuerySurface(frame->display, frame->surface, EGL_WIDTH, &width);
    gl.eglQuerySurface(frame->display, frame->surface, EGL_HEIGHT, &height);
    if (s.handle != frame->surface || s.width != width || s.height != height) {
        inspectSurface(s, frame->display, frame->surface, frame->config);
        s.width = width;
        s.height = height;
        displaySize(s, frame->window);
    }
    EGLint behavior = EGL_BUFFER_DESTROYED;
    gl.eglQuerySurface(frame->display, frame->surface, EGL_SWAP_BEHAVIOR, &behavior);
    const KelirReason declined =
            behavior == EGL_BUFFER_PRESERVED ? KelirReason::kPreserved : s.declined;
    if (declined != KelirReason::kNone || width <= 0 || height <= 0) {
        report(c, s, KelirState::kDeclined, declined, KelirPath::kNone, 0);
        return FRAMEFX_PASS;
    }
    GLboolean feedback[2] = {};
    gl.glGetBooleanv(GL_TRANSFORM_FEEDBACK_ACTIVE, &feedback[0]);
    gl.glGetBooleanv(GL_TRANSFORM_FEEDBACK_PAUSED, &feedback[1]);
    if (feedback[0] && !feedback[1]) return FRAMEFX_PASS;  // a draw would feed the game's buffer

    if (!c.ready) {
        c.declined = probe(c);
        if (c.declined != KelirReason::kNone) {
            report(c, s, KelirState::kDeclined, c.declined, KelirPath::kNone, 0);
            return FRAMEFX_PASS;
        }
    }
    Saved saved;
    save(c, saved);
    neutral(c);
    if (!c.ready) {
        c.declined = setUp(c);
        c.ready = c.declined == KelirReason::kNone;
        if (!c.ready) {
            restore(c, saved);
            ALOGW("%s: kelir declined on GLES (reason %d)", Engine::get().package().c_str(),
                  static_cast<int>(c.declined));
            report(c, s, KelirState::kDeclined, c.declined, KelirPath::kNone, 0);
            return FRAMEFX_PASS;
        }
    }

    gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    GLint encoding = GL_LINEAR;
    gl.glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_BACK,
                                             GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, &encoding);
    // sRGB conversion follows the surface and, where the game may turn it off, its switch.
    const bool srgbOn = encoding == GL_SRGB && gl.glIsEnabled(GL_FRAMEBUFFER_SRGB_EXT) != GL_FALSE;
    Target target;
    target.format = s.format == FormatClass::kSrgb8 && !srgbOn ? FormatClass::kUnorm8 : s.format;
    target.msaa = s.msaa;
    target.fetch = c.fetch && !s.msaa && !needsSource(plan, role);
    target.readsLinear = target.format == FormatClass::kSrgb8;
    target.role = role;
    const uint64_t signature = Signature(plan, target);
    const bool pyramid = role != Role::kStageB && plan.needsPyramid();

    const bool timing = plan.inspect && c.timer;
    if (timing) collectTimers(c);
    const bool timed = timing && beginTimer(c);
    bool ok = true;
    if (!target.fetch) {
        ok = prepareCopies(c, s, pyramid);
        if (ok) {
            gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            gl.glGetIntegerv(GL_READ_BUFFER, &saved.readBuffer);
            gl.glReadBuffer(GL_BACK);
            gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, s.scratchFb);
            gl.glBlitFramebuffer(0, 0, s.width, s.height, 0, 0, s.width, s.height,
                                 GL_COLOR_BUFFER_BIT, GL_NEAREST);
            gl.glReadBuffer(static_cast<GLenum>(saved.readBuffer));
            if (pyramid) buildPyramid(c, s, target.readsLinear);
        }
    }
    Program* program = ok ? programFor(c, signature) : nullptr;
    if (program == nullptr) {
        if (timed) endTimer(c);
        restore(c, saved);
        report(c, s, KelirState::kDeclined, KelirReason::kResources, KelirPath::kNone, signature);
        return FRAMEFX_PASS;
    }
    prepareLut(c, plan, s.primaries);

    gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    gl.glGetIntegerv(GL_DRAW_BUFFER0, &saved.drawBuffer);
    if (saved.drawBuffer != GL_BACK) {
        const GLenum back = GL_BACK;
        gl.glDrawBuffers(1, &back);
    }
    if (!target.fetch) {
        const GLenum color = GL_COLOR;
        gl.glInvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 1, &color);
        gl.glActiveTexture(GL_TEXTURE0 + c.units[kSource]);
        gl.glBindTexture(GL_TEXTURE_2D, s.scratch);
        gl.glActiveTexture(GL_TEXTURE0 + c.units[kPyramid]);
        gl.glBindTexture(GL_TEXTURE_2D, s.pyramid);
    }
    gl.glActiveTexture(GL_TEXTURE0 + c.units[kLut]);
    gl.glBindTexture(GL_TEXTURE_3D, c.lutFront >= 0 ? c.lut[c.lutFront] : 0);
    gl.glUseProgram(program->id);
    if (program->uploaded != planRef) {
        float packed[kPlanVec4s * 4];
        PackPlan(plan, packed);
        gl.glUniform4fv(program->plan, kPlanVec4s, packed);
        program->uploaded = planRef;
    }
    FrameInfo info;
    info.bufferWidth = s.width;
    info.bufferHeight = s.height;
    info.displayWidth = s.displayWidth;
    info.displayHeight = s.displayHeight;
    info.flipY = true;
    info.seed = ++c.seed;
    info.primaries = s.primaries;
    info.format = s.format;
    float packedFrame[kFrameVec4s * 4];
    PackFrame(plan, info, packedFrame);
    gl.glUniform4fv(program->frame, kFrameVec4s, packedFrame);
    gl.glViewport(0, 0, s.width, s.height);
    drawTriangle(c);
    if (saved.drawBuffer != GL_BACK) {
        const GLenum previous = static_cast<GLenum>(saved.drawBuffer);
        gl.glDrawBuffers(1, &previous);
    }
    if (timed) endTimer(c);
    restore(c, saved);

    report(c, s, KelirState::kRunning, KelirReason::kNone,
           target.fetch ? KelirPath::kFetch : KelirPath::kTexture, signature);
    return role == Role::kStageA ? 1 : frame->swap(frame->cookie);
}

}  // namespace

int32_t present(Context** slot, const framefx_gl_frame* frame,
                const std::shared_ptr<const Plan>& plan) {
    return run(slot, frame, plan, Role::kFull);
}

bool stageA(Context** slot, const framefx_gl_frame* frame, const std::shared_ptr<const Plan>& plan) {
    return run(slot, frame, plan, Role::kStageA) != FRAMEFX_PASS;
}

int32_t finish(Context** slot, const framefx_gl_frame* frame,
               const std::shared_ptr<const Plan>& plan) {
    // Nothing on screen to add: the frame goes out as stage A left it.
    if (plan->screenCount == 0 && !plan->split) return frame->swap(frame->cookie);
    const int32_t result = run(slot, frame, plan, Role::kStageB);
    return result != FRAMEFX_PASS ? result : frame->swap(frame->cookie);
}

void release(Context* context) {
    delete context;
}

}  // namespace framefx::kelir::gles
