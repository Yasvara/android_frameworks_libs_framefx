/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3.h>
#include <log/log.h>

#include <algorithm>
#include <atomic>
#include <cstring>

#include "src/engine.h"
#include "src/extensions.h"
#include "src/gles.h"
#include "src/shading.h"
#include "src/texture.h"

#ifndef GL_TEXTURE_LOD_BIAS_QCOM
#define GL_TEXTURE_LOD_BIAS_QCOM 0x8C96
#endif

// What GLES games get beyond the present path: the texture overrides Vulkan games have, and the
// context attributes of the extension settings.
namespace framefx::gles {
namespace {

// Samplers and filters are set at load, so a change in Pentas waits for the next start.
const TextureSettings& textureSettings() {
    static const TextureSettings settings = [] {
        Engine& engine = Engine::get();
        engine.attach();
        return engine.config().texture;
    }();
    return settings;
}

using TexI = void (*)(GLenum, GLenum, GLint);
using TexIv = void (*)(GLenum, GLenum, const GLint*);
using TexF = void (*)(GLenum, GLenum, GLfloat);
using TexFv = void (*)(GLenum, GLenum, const GLfloat*);
using SamI = void (*)(GLuint, GLenum, GLint);
using SamIv = void (*)(GLuint, GLenum, const GLint*);
using SamF = void (*)(GLuint, GLenum, GLfloat);
using SamFv = void (*)(GLuint, GLenum, const GLfloat*);
using GetFv = void (*)(GLenum, GLfloat*);
using GetString = const GLubyte* (*)(GLenum);

struct Next {
    std::atomic<TexI> texI{nullptr};
    std::atomic<TexIv> texIv{nullptr};
    std::atomic<TexF> texF{nullptr};
    std::atomic<TexFv> texFv{nullptr};
    std::atomic<SamI> samI{nullptr};
    std::atomic<SamIv> samIv{nullptr};
    std::atomic<SamF> samF{nullptr};
    std::atomic<SamFv> samFv{nullptr};
    std::atomic<GetFv> getFv{nullptr};
    std::atomic<GetString> getString{nullptr};
} gNext;

template <typename T>
T load(const std::atomic<T>& slot) {
    return slot.load(std::memory_order_relaxed);
}

// glGetString works in every ES version without raising an error the game would read.
bool hasExtension(const char* name) {
    const GetString getString = load(gNext.getString);
    const auto* all = getString != nullptr
            ? reinterpret_cast<const char*>(getString(GL_EXTENSIONS))
            : nullptr;
    if (all == nullptr) return false;
    const size_t n = strlen(name);
    for (const char* p = strstr(all, name); p != nullptr; p = strstr(p + n, name)) {
        if ((p == all || p[-1] == ' ') && (p[n] == ' ' || p[n] == '\0')) return true;
    }
    return false;
}

// Asked from inside the game's GL call, so its context is current. Read once: every context of
// one process sits on the same driver.
struct Caps {
    float maxAnisotropy = 0.0f;
    bool lodBias = false;
};
const Caps& caps() {
    static const Caps c = [] {
        Caps out;
        const GetFv getFv = load(gNext.getFv);
        if (getFv != nullptr && hasExtension("GL_EXT_texture_filter_anisotropic")) {
            getFv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &out.maxAnisotropy);
        }
        out.lodBias = hasExtension("GL_QCOM_texture_lod_bias");
        const TextureSettings& s = textureSettings();
        ALOGI("texture: GL AF %dx (driver max %.0f), LOD bias %+.3f%s", s.anisotropy,
              out.maxAnisotropy, s.lodBiasEighths / 8.0, out.lodBias ? "" : " (no QCOM bias)");
        return out;
    }();
    return c;
}

bool filterFor(GLint asked, texture::GlFilter* out) {
    return texture::OverrideGl(textureSettings(), caps().maxAnisotropy, asked, out);
}

// The game's own anisotropy, raised to the override's.
float anisotropyFor(float asked) {
    const TextureSettings& s = textureSettings();
    const float max = caps().maxAnisotropy;
    if (s.anisotropy <= 1 || max <= 1.0f) return asked;
    return std::max(asked, std::min(static_cast<float>(s.anisotropy), max));
}

void applyTexture(GLenum target, const texture::GlFilter& f) {
    const TexF texF = load(gNext.texF);
    if (texF == nullptr) return;
    if (f.anisotropy > 0.0f) texF(target, GL_TEXTURE_MAX_ANISOTROPY_EXT, f.anisotropy);
    if (f.lodBias != 0.0f && caps().lodBias) texF(target, GL_TEXTURE_LOD_BIAS_QCOM, f.lodBias);
}

void applySampler(GLuint sampler, const texture::GlFilter& f) {
    const SamF samF = load(gNext.samF);
    if (samF != nullptr && f.anisotropy > 0.0f) {
        samF(sampler, GL_TEXTURE_MAX_ANISOTROPY_EXT, f.anisotropy);
    }
}

void TexParameteri(GLenum target, GLenum pname, GLint param) {
    texture::GlFilter f;
    if (pname == GL_TEXTURE_MIN_FILTER && filterFor(param, &f)) {
        load(gNext.texI)(target, pname, f.minFilter);
        applyTexture(target, f);
        return;
    }
    if (pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) {
        param = static_cast<GLint>(anisotropyFor(static_cast<float>(param)));
    }
    load(gNext.texI)(target, pname, param);
}

void TexParameteriv(GLenum target, GLenum pname, const GLint* params) {
    if (pname == GL_TEXTURE_MIN_FILTER || pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) {
        TexParameteri(target, pname, params[0]);
        return;
    }
    load(gNext.texIv)(target, pname, params);
}

void TexParameterf(GLenum target, GLenum pname, GLfloat param) {
    texture::GlFilter f;
    if (pname == GL_TEXTURE_MIN_FILTER && filterFor(static_cast<GLint>(param), &f)) {
        load(gNext.texF)(target, pname, static_cast<GLfloat>(f.minFilter));
        applyTexture(target, f);
        return;
    }
    if (pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) param = anisotropyFor(param);
    load(gNext.texF)(target, pname, param);
}

void TexParameterfv(GLenum target, GLenum pname, const GLfloat* params) {
    if (pname == GL_TEXTURE_MIN_FILTER || pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) {
        TexParameterf(target, pname, params[0]);
        return;
    }
    load(gNext.texFv)(target, pname, params);
}

void SamplerParameteri(GLuint sampler, GLenum pname, GLint param) {
    texture::GlFilter f;
    if (pname == GL_TEXTURE_MIN_FILTER && filterFor(param, &f)) {
        load(gNext.samI)(sampler, pname, f.minFilter);
        applySampler(sampler, f);
        return;
    }
    if (pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) {
        param = static_cast<GLint>(anisotropyFor(static_cast<float>(param)));
    }
    load(gNext.samI)(sampler, pname, param);
}

void SamplerParameteriv(GLuint sampler, GLenum pname, const GLint* params) {
    if (pname == GL_TEXTURE_MIN_FILTER || pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) {
        SamplerParameteri(sampler, pname, params[0]);
        return;
    }
    load(gNext.samIv)(sampler, pname, params);
}

void SamplerParameterf(GLuint sampler, GLenum pname, GLfloat param) {
    texture::GlFilter f;
    if (pname == GL_TEXTURE_MIN_FILTER && filterFor(static_cast<GLint>(param), &f)) {
        load(gNext.samF)(sampler, pname, static_cast<GLfloat>(f.minFilter));
        applySampler(sampler, f);
        return;
    }
    if (pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) param = anisotropyFor(param);
    load(gNext.samF)(sampler, pname, param);
}

void SamplerParameterfv(GLuint sampler, GLenum pname, const GLfloat* params) {
    if (pname == GL_TEXTURE_MIN_FILTER || pname == GL_TEXTURE_MAX_ANISOTROPY_EXT) {
        SamplerParameterf(sampler, pname, params[0]);
        return;
    }
    load(gNext.samFv)(sampler, pname, params);
}

template <typename Fn>
void* keep(std::atomic<Fn>* slot, void* next, Fn own, bool wrap) {
    slot->store(reinterpret_cast<Fn>(next), std::memory_order_relaxed);
    return wrap ? reinterpret_cast<void*>(own) : nullptr;
}

// The value of `name` in an EGL attribute list, or `none` when absent.
EGLint valueOf(const EGLint* attribs, EGLint name, EGLint none) {
    for (const EGLint* a = attribs; a != nullptr && a[0] != EGL_NONE; a += 2) {
        if (a[0] == name) return a[1];
    }
    return none;
}

}  // namespace

const EGLint* contextAttribs(const EGLint* asked, EGLint* out, int32_t capacity) {
    const extensions::Settings& s = extensions::settings();
    // No error checking cannot pair with robustness, a debug context or ES 1.
    constexpr EGLint kAbsent = -1;
    const bool noError = s.glNoError &&
            valueOf(asked, EGL_CONTEXT_CLIENT_VERSION, 1) >= 2 &&
            valueOf(asked, EGL_CONTEXT_OPENGL_NO_ERROR_KHR, kAbsent) == kAbsent &&
            valueOf(asked, EGL_CONTEXT_OPENGL_ROBUST_ACCESS, EGL_FALSE) == EGL_FALSE &&
            valueOf(asked, EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT, EGL_FALSE) == EGL_FALSE &&
            valueOf(asked, EGL_CONTEXT_OPENGL_DEBUG, EGL_FALSE) == EGL_FALSE &&
            valueOf(asked, EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY, kAbsent) == kAbsent &&
            valueOf(asked, EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT, kAbsent) == kAbsent;
    const bool priority = s.glPriority != 0 &&
            valueOf(asked, EGL_CONTEXT_PRIORITY_LEVEL_IMG, kAbsent) == kAbsent;
    if (!noError && !priority) return asked;
    int32_t n = 0;
    for (const EGLint* a = asked; a != nullptr && a[0] != EGL_NONE; a += 2) n += 2;
    if (n + 5 > capacity) return asked;
    if (n > 0) memcpy(out, asked, n * sizeof(EGLint));
    if (priority) {
        out[n++] = EGL_CONTEXT_PRIORITY_LEVEL_IMG;
        out[n++] = s.glPriority;
    }
    if (noError) {
        out[n++] = EGL_CONTEXT_OPENGL_NO_ERROR_KHR;
        out[n++] = EGL_TRUE;
    }
    out[n] = EGL_NONE;
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        ALOGI("extensions: GL contexts with%s%s", priority ? " priority" : "",
              noError ? " no error checking" : "");
    }
    return out;
}

void* intercept(const char* name, void* next) {
    if (next == nullptr || strncmp(name, "gl", 2) != 0) return nullptr;
    const bool on = textureSettings().on();
    if (strcmp(name, "glGetFloatv") == 0) return keep(&gNext.getFv, next, GetFv{}, false);
    if (strcmp(name, "glGetString") == 0) return keep(&gNext.getString, next, GetString{}, false);
    if (strcmp(name, "glTexParameteri") == 0) return keep(&gNext.texI, next, &TexParameteri, on);
    if (strcmp(name, "glTexParameteriv") == 0) return keep(&gNext.texIv, next, &TexParameteriv, on);
    if (strcmp(name, "glTexParameterf") == 0) return keep(&gNext.texF, next, &TexParameterf, on);
    if (strcmp(name, "glTexParameterfv") == 0) return keep(&gNext.texFv, next, &TexParameterfv, on);
    if (strcmp(name, "glSamplerParameteri") == 0) {
        return keep(&gNext.samI, next, &SamplerParameteri, on);
    }
    if (strcmp(name, "glSamplerParameteriv") == 0) {
        return keep(&gNext.samIv, next, &SamplerParameteriv, on);
    }
    if (strcmp(name, "glSamplerParameterf") == 0) {
        return keep(&gNext.samF, next, &SamplerParameterf, on);
    }
    if (strcmp(name, "glSamplerParameterfv") == 0) {
        return keep(&gNext.samFv, next, &SamplerParameterfv, on);
    }
    return shading::interceptGl(name, next, textureSettings().shading());
}

}  // namespace framefx::gles
