/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/video.h>

// A decoder's output through the GPU: its own BufferQueue in front of the app's window. Each frame
// is converted from Y'CbCr, drawn through Kelir and queued to the window under its own timestamp,
// transform and scaling mode. One worker thread per decoder, one submission per frame, ordered by
// sync fds; the worker only waits on the GPU when it is a whole ring of frames behind.
namespace framefx::video {

const framefx_video_api* api();

}  // namespace framefx::video
