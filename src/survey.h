/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include "lapis/classifier.h"
#include "lapis/model.h"

// Logs what Lapis's recorder sees of a game's frames: the images it makes and how their memory is
// backed, each render pass with its attachments' load and store ops, depth clear and sampled
// images, draws per pass with and without depth writes, dispatches, where the game copies or
// transitions depth itself, and every perspective projection in its mapped memory. Diagnostic.
//
// `debug.framefx.survey` holds comma separated words, at most 91 characters:
//   <package>  a process of this package surveys, from its next frame on
//   full       a shot adds the projections in mapped memory and the watched constants
//   draws      a shot adds a line per draw of a pass with many: depth use, count, scissor, images
//   misses     the next 4 frames at which Lapis lets go are logged as a shot's are
//   hud        Hide HUD labels every draw even with the HUD shown, and a shot adds a line per
//              draw: its label, structural label, raw fields and features, for
//              lab/hud-detect/train_hud.py
//   atlas      read once at process start: keep the game's single-channel texture uploads, and a
//              shot writes the draws that read them to cache/framefx-atlas: their vertices,
//              indices and constants, with the textures as uploaded
//   textcost   read once at process start: reads the text draws at most 4 times a second as a text
//              tap would, and logs what that costs every 10 s
//   fontscan   read once at process start: a shot also looks through the process's memory for font
//              files, and writes them to cache/framefx-fonts
//   cut=N      a split pass splits before draw N, to find by eye where a game's HUD starts. What
//              the pass draws after that loses its depth. Reset with a value without it
// Any other word, such as a counter, only changes the value: every change asks for a shot of the
// next frames, in submission order. A shot also logs how Luwes's scene layer fared.
namespace framefx::survey {

bool enabled();
// Whether a shot is under way, for what logs its own part of it.
bool shooting();
// Whether a shot lists its draws.
bool drawList();
// Whether Hide HUD labels draws for the survey, while the `hud` word is set.
bool hudDraws();
// What this process does with text as it goes to the GPU. Fixed at the first call of any of them.
bool atlas();
bool textCost();
bool fontScan();
// Whether any of those needs the draws' vertex state.
inline bool textTap() {
    return atlas() || textCost();
}
// The draw a split pass splits before, or -1.
int cutDraw();
// Whether the frame being closed is to be logged: a shot is under way or was just asked for.
bool wantsFrame();
void onFrame(const lapis::Frame& frame, const lapis::Images& images, const lapis::Roles& roles);
// The frame at which the tracker let go, logged when `misses` asks for it.
void onMiss(const lapis::Frame& frame, const lapis::Images& images, const lapis::Roles& roles);

}  // namespace framefx::survey
