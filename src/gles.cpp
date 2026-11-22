/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "src/gles.h"

#include <atomic>

#include "kelir/gles.h"
#include "luwes/gles.h"
#include "src/engine.h"
#include "src/shading.h"

#include <system/window.h>
#include <utils/Timers.h>

namespace framefx::gles {
namespace {

// Engine state of one game context.
struct State {
    void* luwes = nullptr;
    kelir::gles::Context* kelir = nullptr;
    Engine::Watch watch;
};

// Window surfaces whose last frame went out redrawn by framefx. A game that redraws only what
// changed would build on that frame, so their buffer age reads 0.
constexpr int kTracked = 8;
std::atomic<EGLSurface> gRedrawn[kTracked];

void mark(EGLSurface surface, bool redrawn) {
    for (auto& slot : gRedrawn) {
        if (slot.load(std::memory_order_relaxed) == surface) {
            if (!redrawn) slot.store(EGL_NO_SURFACE, std::memory_order_relaxed);
            return;
        }
    }
    if (!redrawn) return;
    for (auto& slot : gRedrawn) {
        EGLSurface empty = EGL_NO_SURFACE;
        if (slot.compare_exchange_strong(empty, surface, std::memory_order_relaxed)) return;
    }
    gRedrawn[0].store(surface, std::memory_order_relaxed);
}

}  // namespace

void attach(const framefx_gl_host* host) {
    shading::attachGl(host);
    luwes::gles::attach(host);
    kelir::gles::attach(host);
}

int32_t present(const framefx_gl_frame* frame) {
    Engine& engine = Engine::get();
    engine.status().countPresent();
    auto* s = static_cast<State*>(*frame->state);
    if (s == nullptr) {
        if (!engine.active()) return FRAMEFX_PASS;
        s = new State();
        *frame->state = s;
    }
    s->watch.refresh(engine);
    const Config& config = s->watch.config();
    if (config.texture.shading()) {
        shading::onGlPresent();
        int64_t refresh = 0;
        if (frame->window != nullptr) native_window_get_refresh_cycle_duration(frame->window, &refresh);
        shading::onPresent(config.texture, systemTime(SYSTEM_TIME_MONOTONIC), refresh);
    }
    const std::shared_ptr<const kelir::Plan>& kelir = config.kelir;
    framefx_gl_frame luwes = *frame;
    luwes.state = &s->luwes;
    int32_t result;
    if (!config.luwes.on()) {
        // Luwes off, or releasing what it held; then Kelir filters the frame on its own.
        result = luwes::gles::present(&luwes, nullptr);
        if (result == FRAMEFX_PASS) result = kelir::gles::present(&s->kelir, frame, kelir);
    } else {
        // Stage A goes into the frame in the game's own pass, so generation works from it; Luwes
        // finishes every frame it presents. When Luwes passes this one, Kelir finishes it.
        const bool staged = kelir::gles::stageA(&s->kelir, frame, kelir);
        result = luwes::gles::present(&luwes, kelir);
        if (result == FRAMEFX_PASS) {
            result = staged ? kelir::gles::finish(&s->kelir, frame, kelir)
                            : kelir::gles::present(&s->kelir, frame, kelir);
        }
    }
    mark(frame->surface, result != FRAMEFX_PASS);
    return result;
}

void release(void* state) {
    auto* s = static_cast<State*>(state);
    luwes::gles::release(s->luwes);
    kelir::gles::release(s->kelir);
    delete s;
}

int32_t swapInterval(int32_t requested) {
    return luwes::gles::swapInterval(requested);
}

int32_t redrawn(EGLSurface surface) {
    for (const auto& slot : gRedrawn) {
        if (slot.load(std::memory_order_relaxed) == surface) return 1;
    }
    return 0;
}

}  // namespace framefx::gles
