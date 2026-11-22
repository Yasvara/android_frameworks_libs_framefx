/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <memory>

#include "kelir/plan.h"

// Kelir on GLES: one draw over the game's default framebuffer, in the game's own context, right
// before its swap. A stack that only needs the pixel itself reads it with framebuffer fetch inside
// the game's still open pass; one that reads neighbours copies the frame first. Every piece of GL
// state the draw touches is read before and restored after it.
namespace framefx::kelir::gles {

struct Context;

void attach(const framefx_gl_host* host);

// Filters the current context's default framebuffer with `plan` and presents it with full damage.
// FRAMEFX_PASS when the surface is not one Kelir takes, or this frame goes out untouched.
// *context is Kelir's state of the game context, made on first use.
int32_t present(Context** context, const framefx_gl_frame* frame,
                const std::shared_ptr<const Plan>& plan);

// With frame generation: stage A over the default framebuffer in place, before generation
// captures the frame. Nothing is presented; true when something was drawn.
bool stageA(Context** context, const framefx_gl_frame* frame,
            const std::shared_ptr<const Plan>& plan);

// A frame generation passed on after stage A: the finish (stage B, divider, dither) in place,
// then the swap.
int32_t finish(Context** context, const framefx_gl_frame* frame,
               const std::shared_ptr<const Plan>& plan);

// The game context is going away, and its GL objects with it.
void release(Context* context);

}  // namespace framefx::kelir::gles
