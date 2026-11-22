/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <stdint.h>

// Video hook between MediaCodec and libframefx.
//
// A decoder queues straight to its window, so the present hook never sees its frames. In an app
// armed for its video (kVideoMarkerDir), MediaCodec starts a decoder on the proxy's window, or moves
// it there with the same switch setOutputSurface makes when its filters come on later, and the
// proxy queues what it draws to the app's window, under the frame's own timestamp. With nothing to
// do, MediaCodec switches back, so a decoder with its filters off costs what it costs on any other
// phone.

#define FRAMEFX_VIDEO_ABI 2
#define FRAMEFX_VIDEO_ENTRY "framefx_video_entry"

// framefx_video_api.state
#define FRAMEFX_VIDEO_WARMING 0
#define FRAMEFX_VIDEO_READY 1
#define FRAMEFX_VIDEO_FAILED (-1)  // close it: frames are being dropped, or never went through

#ifdef __cplusplus
extern "C" {
#endif

struct ANativeWindow;
typedef struct framefx_video framefx_video;

typedef struct framefx_video_api {
    uint32_t abi;
    // Nonzero while Pentas wants this app's video filtered: one atomic load.
    int32_t (*wanted)(void);
    // A proxy warm or warming up on its own thread. Frames queued to it wait for that, and for the
    // first attach.
    framefx_video* (*open)(void);
    int32_t (*state)(framefx_video* video);
    // The window the decoder queues to.
    struct ANativeWindow* (*input)(framefx_video* video);
    // Connects to `target`, replacing the window it had. 0 when it is connected. The decoder must
    // have let go of `target`.
    int32_t (*attach)(framefx_video* video, struct ANativeWindow* target);
    // Disconnects from the window, on return free for a decoder again; later frames are dropped.
    void (*detach)(framefx_video* video);
    // Detaches and destroys it. Blocks until nothing runs on it.
    void (*close)(framefx_video* video);
    // Nonzero while the frames are shown large enough to be worth the decoder's own SR filter.
    int32_t (*superres)(framefx_video* video);
} framefx_video_api;

typedef const framefx_video_api* (*framefx_video_entry_fn)(uint32_t abi);

// The table for `abi`, or NULL when it speaks another. Opens the process's channel.
const framefx_video_api* framefx_video_entry(uint32_t abi);

#ifdef __cplusplus
}  // extern "C"

#include <dlfcn.h>
#include <framefx/channel.h>
#include <unistd.h>

namespace framefx {

// The video engine for a process Pentas armed for its video, nullptr otherwise. *error is set when
// it is armed but the engine could not be loaded.
inline const framefx_video_api* LoadVideoEngine(const char** error) {
    *error = nullptr;
    if (!Marked(kVideoMarkerDir, getuid())) return nullptr;
    void* lib = dlopen("libframefx.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        *error = dlerror();
        return nullptr;
    }
    const auto entry = reinterpret_cast<framefx_video_entry_fn>(dlsym(lib, FRAMEFX_VIDEO_ENTRY));
    const framefx_video_api* api = entry != nullptr ? entry(FRAMEFX_VIDEO_ABI) : nullptr;
    if (api == nullptr) *error = "ABI mismatch";
    return api;
}

}  // namespace framefx
#endif
