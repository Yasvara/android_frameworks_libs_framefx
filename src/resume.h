/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

namespace framefx {

// Replays the game's resumed activity's pause and resume in its own process
// (ActivityThread.replayPauseResume), from the main thread. After framefx made the game rebuild
// its swapchain, engines that take their render size on resume (Unity) re-read the window, which
// a scaled window needs. A no-op in a process without a Java VM.
void ReplayResume();

}  // namespace framefx
