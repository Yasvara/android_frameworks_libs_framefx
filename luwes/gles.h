/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

#include <memory>

#include "kelir/plan.h"

// Luwes on GLES. Generation runs in a private context per game context, current on the game's
// own surface only inside its swap, so no game GL state is read, saved or touched.
namespace framefx::luwes::gles {

void attach(const framefx_gl_host* host);
// `kelir`, when it is to be applied: stage A is already in the frame; every present Luwes makes
// gets Kelir's finish.
int32_t present(const framefx_gl_frame* frame, const std::shared_ptr<const kelir::Plan>& kelir);
void release(void* state);
int32_t swapInterval(int32_t requested);

}  // namespace framefx::luwes::gles
