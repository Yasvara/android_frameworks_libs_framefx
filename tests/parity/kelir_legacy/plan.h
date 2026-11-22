/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/channel.h>

#include <cstdint>
#include <memory>

#include "tests/parity/kelir_legacy/colour.h"
#include "tests/parity/kelir_legacy/lut.h"
#include "tests/parity/kelir_legacy/section.h"

// A stack compiled for the passes: the bands in their fixed order Detail -> Colour -> Screen,
// stack order inside each, every node that would change nothing dropped, and the colour band
// baked into one LUT.
namespace framefx::kelir_legacy {

inline constexpr int kMaxDetail = 3;
inline constexpr int kMaxScreen = 4;
inline constexpr int kMaxScene = 4;  // each scene kind at most once

struct Plan {
    uint32_t revision = 0;
    bool peek = false;
    bool split = false;
    bool inspect = false;
    float splitX = 0.0f;  // display space

    int detailCount = 0;
    Step detail[kMaxDetail];
    int colourCount = 0;
    Step colour[kMaxKelirNodes];
    int screenCount = 0;
    Step screen[kMaxScreen];
    // Run by Luwes on the game's scene before its HUD, never by Kelir's own passes.
    int sceneCount = 0;
    Step scene[kMaxScene];

    // The colour band baked, when colourCount > 0; shared with the plan before when the band did
    // not change.
    std::shared_ptr<const Luts> luts;

    // Stack index of each colour step, for the per-node curves.
    int colourNode[kMaxKelirNodes] = {};

    // Nothing for Kelir's passes to draw: the frame goes out as the game rendered it.
    bool empty() const { return detailCount + colourCount + screenCount == 0; }
    // The present goes out untouched: nothing to draw, or the editor holds peek.
    bool bypass() const { return empty() || peek; }

    bool has(uint8_t kind) const;
    const Step* sceneStep(uint8_t kind) const;
    // The shared quarter-resolution pyramid: Details, Tilt-Shift, Painterly, Watercolor.
    bool needsPyramid() const;
    // Neighbouring source pixels: Sharpen, Watercolor, Retro. On GLES this rules out fetch.
    bool needsSource() const;
};

// Which stages run on one present. Without frame generation one pass does both. With it, stage A
// (detail and colour) runs once per real frame, before generation reads that frame, and stage B
// (screen nodes, divider, dither) on every present Luwes makes, real or generated.
struct Stages {
    bool full = false;
    bool stageA = false;
    bool stageB = false;
};
// `generation`: Luwes presents this swapchain. `real`: the present carries the game's own frame.
Stages Schedule(const Plan* plan, bool generation, bool real);

// Compiles a validated stack. `previous` lends its LUTs when the colour band is unchanged.
std::shared_ptr<const Plan> Compile(const Stack& stack, const Plan* previous);

// What a pass is compiled for besides the plan.
enum class Role : uint8_t {
    kFull = 0,         // stage A and B into the presented image
    kStageA = 1,       // with Luwes: the presented real frame (A + B) and generation's source (A)
    kStageB = 2,       // with Luwes: stage B alone over a generated frame
};

struct Target {
    FormatClass format = FormatClass::kUnorm8;  // kSrgb8: the pass writes linear values
    bool readsLinear = false;  // the frame reads back decoded: an sRGB view or sRGB fetch
    bool fetch = false;        // GLES: framebuffer fetch inside the game's pass
    bool msaa = false;         // GLES: multisampled default framebuffer
    Role role = Role::kFull;
};

// Everything that changes the code of the pass and nothing else: parameters, intensities, the
// split position and rotation are uniforms, so a colour edit never changes it.
uint64_t Signature(const Plan& plan, const Target& target);

// A signature taken apart, as the shader sources need it.
struct Program {
    Target target;
    bool lut = false;
    int detail[kMaxDetail] = {-1, -1, -1};  // kinds, -1 for an empty slot
    bool sharpenPlus = false;
    int screen[kMaxScreen] = {-1, -1, -1, -1};

    bool has(int kind) const;
};
Program Unpack(uint64_t signature);

// Plan constants the pass reads, kPlanVec4s vec4s: per detail slot two vec4s of parameters
// (0..5), per screen slot two (6..13), detail intensities (14.xyz), 1 in 14.w when the LUT holds
// linear light, screen intensities (15).
inline constexpr int kPlanVec4s = 16;
void PackPlan(const Plan& plan, float out[kPlanVec4s * 4]);

// The scene band's constants for Luwes's scene passes (luwes/shaders/scene.frag), kSceneVec4s
// vec4s; a kind that is not in the plan has intensity 0:
//   0: ambient occlusion  x radius m, y intensity
//   1: depth fog          x start m, y extinction per m, z intensity
//   2: fog colour         rgb encoded
//   3: depth of field     x 1 for a fixed focus, y focus distance m, z blur radius as a fraction of
//                         the scene's height, w in-focus range
//   4: depth of field     x intensity
//   5: global illumination x reach m, y bounce, z intensity
inline constexpr int kSceneVec4s = 6;
void PackScene(const Plan& plan, float out[kSceneVec4s * 4]);

// One present's target, for the frame constants (kelir/shaders/common.glsl, FRAME).
struct FrameInfo {
    int bufferWidth = 0;
    int bufferHeight = 0;
    float displayWidth = 0.0f;  // display px, in the display's orientation
    float displayHeight = 0.0f;
    int turns = 0;       // quarter turns from the buffer to the display
    bool flipY = false;  // GL: row 0 of the buffer is the bottom
    uint32_t seed = 0;   // per present
    Primaries primaries = Primaries::kSrgb;
    FormatClass format = FormatClass::kUnorm8;
    bool edges = false;  // the source's alpha marks depth edges
};

inline constexpr int kFrameVec4s = 4;
void PackFrame(const Plan& plan, const FrameInfo& frame, float out[kFrameVec4s * 4]);

}  // namespace framefx::kelir_legacy
