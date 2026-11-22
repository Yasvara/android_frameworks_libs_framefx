/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace framefx::luwes {

// Frame-by-frame capture of what Luwes presents.
//
// Writing the trigger file /data/user/<user>/<package>/cache/framefx-capture asks for the next
// real frames: its size in bytes is how many (0 means 16). Every real frame, the frames generated
// with it, its motion field and HUD mask are then written losslessly, with an index of what each
// file holds and when each present went out, to cache/framefx-capture-<time>/ of the same app.
// Only root or the app itself can write there, so a capture shows nothing the app could not read
// back on its own.
class Capture {
public:
    // Checks the trigger at most once a second. Returns how many real frames to capture now, 0
    // for none.
    int poll(int64_t now);

    bool active() const { return session_ != nullptr; }

    // One file of the active session; written on a background thread.
    void write(const std::string& name, std::vector<uint8_t> bytes);
    // One line of the session's index.
    void note(const std::string& line);
    // Closes the session once everything queued is written.
    void end();

    struct Session;

private:
    int64_t checkedAt_ = 0;
    int64_t seenMtime_ = -1;
    std::shared_ptr<Session> session_;
};

// The app's cache directory, whichever of its processes this is.
std::string AppCacheDir();

// Bytes per pixel of a capture format name, 0 when unknown.
uint32_t CaptureBytesPerPixel(const char* format);

}  // namespace framefx::luwes
