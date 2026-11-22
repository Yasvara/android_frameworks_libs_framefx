/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/hook.h>

// The GLES present path of an armed process: Luwes presents while it generates frames; otherwise
// Kelir filters the frame the game presents.
namespace framefx::gles {

void attach(const framefx_gl_host* host);
int32_t present(const framefx_gl_frame* frame);
void release(void* state);
int32_t swapInterval(int32_t requested);
int32_t redrawn(EGLSurface surface);
const EGLint* contextAttribs(const EGLint* asked, EGLint* out, int32_t capacity);
void* intercept(const char* name, void* next);

}  // namespace framefx::gles
