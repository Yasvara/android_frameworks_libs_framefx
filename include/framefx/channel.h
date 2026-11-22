/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

// Layouts shared by libframefx, system_server's channel broker and Pentas. Every multi-byte
// field is little-endian and decoded byte-wise.
//
// Channel: one sealed shared-memory region per armed package, delivered by the "yasvara_framefx"
// binder service to that package's processes only, mapped read-only. Pentas writes sections
// through the broker; the broker rewrites the region under a sequence lock (odd while a write is
// in flight) and wakes the futex on the sequence word.
//
//   0   u32  magic "FFXC"
//   4   u16  version
//   6   u16  size, kChannelSize
//   8   u32  sequence
//   12  u8   section count, <= kMaxSections
//   13  u8   0   14 u16 0
//   16  section table, kMaxSections x 8 bytes: u8 id, u8 0, u16 offset, u16 length, u16 0
//   80  section data
//
// Status page: one shared-memory region per armed process, written by the game under its own
// sequence lock and read by the broker when Pentas asks.
//
//   0   u32  magic "FFXS"   4 u16 version   6 u16 size, kStatusSize   8 u32 sequence
//   12  u32  0
//   16  Luwes block, kLuwesStatusSize bytes
//   64  Kelir block
//   264 Lapis block
//   280 Upscale block
//   304 Video block
//   352 u32  presents: the game adds one per present, relaxed and outside the sequence lock, for
//        lepasd to time the game from outside (it reads the page from the game's memory)

namespace framefx {

inline constexpr char kServiceName[] = "yasvara_framefx";
inline constexpr char kInterface[] = "yasvara.framefx.IBroker";

// Broker transactions. kOpen is for the armed game itself; the others take the system uid.
enum BrokerCode : uint32_t {
    // -> i32 status, then on 0: channel fd (read-only), status fd (read-write)
    kOpen = 1,
    // String package, i32 section id, byte[] bytes (empty removes) -> i32 status
    kSetSection = 2,
    // String package, i32 section id -> i32 status, byte[] bytes
    kGetSection = 3,
    // String package -> i32 count, then count x (i32 pid, byte[] status page)
    kStatus = 4,
    // IBinder token, held until the process goes -> i32 status, then on 0: ring fd, table fd, both
    // read-write; layouts in framefx/alih.h. For an armed game whose Lapis section taps text.
    kAlihOpen = 5,
};

// A zero-length file per armed uid. The loaders test it once per process, before anything else.
inline constexpr char kMarkerDir[] = "/data/misc/framefx/armed";
// The same per uid whose video is filtered. Only MediaCodec tests it, so the graphics loaders of an
// app armed for its video alone stay untouched: its UI never runs through the present hook.
inline constexpr char kVideoMarkerDir[] = "/data/misc/framefx/video";

inline constexpr uint32_t kChannelMagic = 0x43584646u;  // "FFXC"
inline constexpr uint16_t kChannelVersion = 1;
inline constexpr size_t kChannelSize = 4096;
inline constexpr int kMaxSections = 8;
inline constexpr size_t kSectionTableOffset = 16;
inline constexpr size_t kSectionDataOffset = 80;
inline constexpr size_t kSequenceOffset = 8;

enum SectionId : uint8_t {
    kSectionLuwes = 1,
    kSectionKelir = 2,
    kSectionLapis = 3,
    kSectionUpscale = 4,
    kSectionTexture = 5,
    kFirstVideoSection = 6,  // 6..7 arm the video path, not the loaders
    kSectionVideoKelir = 6,  // the Kelir section's layout, for the frames of a decoder
    kSectionVideo = 7,
    kLastVideoSection = 7,
    kSectionLepas = 8,  // read by system_server alone; arms the loaders for the present count
};

// Lepas section v1, 4 bytes; none means the game's own cap:
//   0 u8 major (1)  1 u8 target fps  2 u8 0  3 u8 0
// The game never reads it. The broker starts lepasd for each armed process and moves the panel to
// the target; lepasd finds the game's cap from outside and holds it at the target.
inline constexpr uint8_t kLepasMajor = 1;

// Luwes section v1, 8 bytes:
//   0 u8 major (1)  1 u8 multiplier: 0 off, else presents per real frame
//   2 u8 mode       3 u8 flags: bit0 HUD protection, bit1 Luwes's own motion over Qualcomm AFME
//   4 u32 0
inline constexpr size_t kLuwesSectionSize = 8;
inline constexpr uint8_t kLuwesMajor = 1;
inline constexpr uint8_t kLuwesHudProtection = 1u << 0;
inline constexpr uint8_t kLuwesOwnEngine = 1u << 1;
inline constexpr int kMinMultiplier = 2;
inline constexpr int kMaxMultiplier = 4;

enum class Mode : uint8_t {
    kExtrapolate = 0,  // real frame first, synthetic frames after it
    kInterpolate = 1,  // synthetic frames between the previous and the held real frame
};

// Video section v1, 8 bytes; none means neither:
//   0 u8 major (1)  1 u8 motion: 0 off, else the display rate to generate up to, Hz
//   2 u8 flags: bit0 upscale to the app's surface, bit1 Qualcomm AFME over Luwes's own motion: one
//   frame between every two  3 u8 smoothness (unorm: 255 follows the motion all the way, less
//   keeps some of the film's cadence)
//   4 u16 display's long side  6 u16 short side: what a surface fixed at the video's size is shown
//   at, which its producer cannot see
inline constexpr size_t kVideoSectionSize = 8;
inline constexpr uint8_t kVideoMajor = 1;
inline constexpr uint8_t kVideoUpscale = 1u << 0;
inline constexpr uint8_t kVideoAfme = 1u << 1;
inline constexpr int kMinVideoHz = 48;

// Lapis section v1, 4 bytes; none means auto:
//   0 u8 major (1)  1 u8 mode  2 u8 flags: bit0 hide the HUD (photo mode), bit1 tap the game's
//   text for org.yasvara.alih, read once at process start and arming the game, bit2 Kelir grades
//   the whole screen at the last stage instead of the game's 3D under its HUD  3 u8 0
inline constexpr size_t kLapisSectionSize = 4;
inline constexpr uint8_t kLapisMajor = 1;
inline constexpr uint8_t kLapisHideHud = 1u << 0;
inline constexpr uint8_t kLapisTextTap = 1u << 1;
inline constexpr uint8_t kLapisWholeScreen = 1u << 2;

enum class LapisMode : uint8_t {
    kOff = 0,
    kAuto = 1,  // read the game's own frame wherever it offers a scene before its HUD
};

// Upscale section v1, 4 bytes; none means auto in a game armed for something else, while auto
// written out arms the game itself:
//   0 u8 major (1)  1 u8 mode  2 u8 method  3 u8 sharpness: 0 the filter's own, else 1 soft to
//   255 crisp
inline constexpr size_t kUpscaleSectionSize = 4;
inline constexpr uint8_t kUpscaleMajor = 1;

enum class UpscaleMode : uint8_t {
    kOff = 0,
    kAuto = 1,  // wherever the game draws its 3D smaller than its HUD, or its frame smaller
};

enum class UpscaleFilter : uint8_t {
    kBest = 0,     // SGSR 2 where the game's anti-aliasing offers what it reads, SGSR 1 otherwise
    kSpatial = 1,  // SGSR 1 always
};

// Texture section v1, 4 bytes; none leaves the game's texture sampling and shading alone. Read as
// the game makes its device and samplers, so a change waits for its restart, except the shading
// rate of a game that started with shading on, which follows live:
//   0 u8 major (1)  1 u8 anisotropy (0 the game's, else 2, 4, 8 or 16)
//   2 s8 LOD bias in eighths (-16..0)
//   3 u8 shading: bits 0-1 the coarsest rate for depth-tested work (0 full, 1 2x1, 2 2x2, 3 4x4),
//        bit 7 adaptive (full rate while the game keeps its pace); other bits 0
inline constexpr size_t kTextureSectionSize = 4;
inline constexpr uint8_t kShadingRateMask = 0x03;
inline constexpr uint8_t kShadingAdaptive = 0x80;
inline constexpr uint8_t kTextureMajor = 1;
inline constexpr int kMinLodBiasEighths = -16;

// Kelir section v1: a 16-byte header and up to kMaxKelirNodes 20-byte nodes; its length is
// exactly 16 + 20 * count.
//   0 u8 major (1)  1 u8 minor  2 u8 flags  3 u8 count  4 u16 split (unorm)  6 u16 0
//   8 u32 revision  12 u32 0
// Node: 0 u8 kind  1 u8 flags (bit0 enabled)  2 u16 intensity (unorm)  4 s16 p[8]
inline constexpr size_t kKelirHeaderSize = 16;
inline constexpr size_t kKelirNodeSize = 20;
inline constexpr int kMaxKelirNodes = 8;
inline constexpr size_t kMaxKelirSectionSize = 256;
inline constexpr uint8_t kKelirMajor = 1;
inline constexpr uint8_t kKelirPeek = 1u << 0;
inline constexpr uint8_t kKelirSplit = 1u << 1;
inline constexpr uint8_t kKelirInspect = 1u << 2;

inline constexpr uint32_t kStatusMagic = 0x53584646u;  // "FFXS"
inline constexpr uint16_t kStatusVersion = 1;
inline constexpr size_t kStatusSize = 4096;
inline constexpr size_t kLuwesStatusOffset = 16;
inline constexpr size_t kKelirStatusOffset = 64;
inline constexpr size_t kLapisStatusOffset = 264;
inline constexpr size_t kUpscaleStatusOffset = 280;
inline constexpr size_t kVideoStatusOffset = 304;
inline constexpr size_t kPresentCountOffset = 352;

// Luwes status block, 48 bytes:
//   0 u8 state   1 u8 reason   2 u8 api   3 u8 motion source
//   4 u16 width  6 u16 height  8 u8 multiplier  9 u8 generated  10 u8 mode  11 u8 paced
//   12 f32 real fps  16 f32 presented fps  20 f32 panel Hz  24 u32 frames generated (total)
//   28 u8 flags: bit0 generating on the game's scene before its HUD   29..47 0
inline constexpr size_t kLuwesStatusSize = 48;
inline constexpr uint8_t kLuwesStatusScene = 1u << 0;

enum class LuwesState : uint8_t {
    kOff = 0,
    kWaiting = 1,   // armed, no game loop presenting yet
    kRunning = 2,
    kDeclined = 3,  // see the reason
};

enum class LuwesReason : uint8_t {
    kNone = 0,
    kNoMotionSource = 1,  // the GPU offers no motion estimation or frame extrapolation
    kFormat = 2,          // HDR, wide or unsupported presentable format
    kPresentMode = 3,     // shared presentable images
    kResources = 4,       // a GPU object could not be created
};

enum class Api : uint8_t {
    kNone = 0,
    kGles = 1,
    kVulkan = 2,
};

enum class MotionSource : uint8_t {
    kNone = 0,
    kGlesMotionEstimation = 1,  // GL_QCOM_motion_estimation and own warp
    kGlesExtrapolation = 2,     // GL_QCOM_frame_extrapolation
    kVulkanBlockMatch = 3,      // VK_QCOM_image_processing2 window block matching and own warp
    kBlockSearch = 4,           // compute block search on a luma pyramid and own warp
    kAfme = 5,                  // Qualcomm AFME on the optical-flow hardware
};

// Lapis status block, kLapisStatusSize bytes:
//   0 u8 state   1 u8 tier   2 u8 roles found   3 u8 why not engaged
//   4 u16 scene width   6 u16 scene height   8..15 0
inline constexpr size_t kLapisStatusSize = 16;
inline constexpr uint8_t kLapisDepth = 1u << 0;     // D, the scene depth
inline constexpr uint8_t kLapisKept = 1u << 1;      // D still holds the scene at present
inline constexpr uint8_t kLapisScene = 1u << 2;     // S, the frame before the HUD
inline constexpr uint8_t kLapisMotion = 1u << 3;    // V, the game's velocity buffer
inline constexpr uint8_t kLapisCamera = 1u << 4;    // P, the camera projection

enum class LapisState : uint8_t {
    kOff = 0,
    kLearning = 1,  // recording, the roles not yet stable
    kEngaged = 2,
};

// Upscale status block, kUpscaleStatusSize bytes:
//   0 u8 state   1 u8 reason   2 u16 real frames per 10 s   4 u16 scene width   6 u16 scene height
//   8 u16 output width   10 u16 output height   12 u32 GPU ns per real frame   16 u8 method
//   17 u8 0   18 u16 drawn width   20 u16 drawn height (the game's frame, when it is not the
//   output's size: Luwes resamples it)   22..23 0
inline constexpr size_t kUpscaleStatusSize = 24;

enum class UpscaleMethod : uint8_t {
    kNone = 0,
    kSgsr1 = 1,     // the scene, spatial
    kSgsr2 = 2,     // the scene, temporal, from the game's jittered frames
    kResample = 3,  // the whole frame: SGSR 1 up, Lanczos down
};

enum class UpscaleState : uint8_t {
    kOff = 0,
    kWaiting = 1,  // on, but nothing to upscale: see the reason
    kRunning = 2,
};

enum class UpscaleReason : uint8_t {
    kNone = 0,
    kNoScene = 1,    // Lapis has not found the game's scene before its HUD
    kFullSize = 2,   // the game draws its 3D at its full size
    kLapisOff = 3,   // Lapis is off for this game
    kResources = 4,  // a GPU object could not be created
};

// Kelir status block, kKelirStatusSize bytes:
//   0 u8 state   1 u8 reason   2 u8 api   3 u8 format class   4 u16 width   6 u16 height
//   8 u8 rotation (quarter turns)   9 u8 path   10 u8 reject reason   11 u8 0
//   12 u32 applied revision   16 u32 rejected revision   20 u32 GPU ns per real frame (inspect)
//   24 u64 signature   32 swatch: 8 x RGB8   56 curves: 8 nodes x 17 luma u8, stack order;
//   a node outside the colour band leaves its curve 0   192..199 0
// Swatch, curves and GPU time are written only while the editor has the inspect flag set.
inline constexpr size_t kKelirStatusSize = 200;
inline constexpr int kKelirSwatchColours = 8;
inline constexpr int kKelirCurvePoints = 17;

enum class KelirState : uint8_t {
    kOff = 0,
    kWaiting = 1,   // armed with a stack, nothing presented through it yet
    kRunning = 2,
    kDeclined = 3,  // see the reason
};

enum class KelirReason : uint8_t {
    kNone = 0,
    kHdr = 1,           // PQ, HLG, scRGB or FP16 target
    kFormat = 2,        // a target format the passes cannot read or write
    kPreserved = 3,     // EGL_BUFFER_PRESERVED swap or single-buffered surface
    kPresentMode = 4,   // shared presentable images
    kProtected = 5,     // protected content
    kResources = 6,     // a GPU object could not be created
    kGlesVersion = 7,   // the game's context is below GLES 3.0
};

enum class KelirPath : uint8_t {
    kNone = 0,
    kFetch = 1,    // GLES, framebuffer fetch inside the game's last pass
    kTexture = 2,  // GLES, through a copy of the frame
    kVulkan = 3,   // Vulkan, into a redirected output image
};

enum class FormatClass : uint8_t {
    kUnorm8 = 0,
    kSrgb8 = 1,  // reads decode and writes encode sRGB
    kUnorm10 = 2,
};

// Why a Kelir section was rejected; the last good stack stays.
enum class KelirReject : uint8_t {
    kNone = 0,
    kSize = 1,          // shorter than its header, larger than kMaxKelirSectionSize, or not 16 + 20n
    kMajor = 2,
    kFlags = 3,         // unknown header or node flag bits
    kCount = 4,         // more than kMaxKelirNodes
    kReserved = 5,      // a reserved field is not 0
    kKind = 6,          // a kind this library does not know
    kUnusedParam = 7,   // a parameter slot the kind does not use is not 0
    kParamRange = 8,    // a code outside its parameter's range
    kRepeat = 9,        // a kind outside the colour band appears twice
    kBandLimit = 10,    // more detail, style, screen or Retro nodes than allowed
};

// Video status block, kVideoStatusSize bytes, of the decoder last through the video path:
//   0 u8 state   1 u8 reason   2 u16 0   4 u16 source width   6 u16 source height
//   8 u16 output width   10 u16 output height   12 f32 frames per second
//   16 u32 GPU ns per real frame   20 u32 real frames through (total)   24 u32 dropped (total)
//   28 u32 applied Kelir revision   32 f32 frames presented per second   36 u32 frames generated
//   (total)   40 u16 delay added to the video, us/100   42 u8 flags: bit0 upscaled, bit1 by the
//   decoder's SR filter   43..47 0
inline constexpr size_t kVideoStatusSize = 48;
inline constexpr uint8_t kVideoStatusUpscaled = 1u << 0;
inline constexpr uint8_t kVideoStatusSuperres = 1u << 1;

enum class VideoState : uint8_t {
    kOff = 0,
    kWaiting = 1,   // on, no decoder rendering through the path yet
    kRunning = 2,
    kDeclined = 3,  // see the reason; that decoder's frames go out as decoded
};

enum class VideoReason : uint8_t {
    kNone = 0,
    kHdr = 1,        // PQ or HLG frames
    kGpu = 2,        // no usable device or a GPU object could not be created
    kTarget = 3,     // the app's surface refused the path
};

// Apps only: system, isolated and SDK sandbox uids are never armed and never look.
inline bool IsAppUid(uid_t uid) {
    const uid_t appId = uid % 100000;
    return appId >= 10000 && appId <= 19999;
}

// Whether `uid` has its marker in `dir`: one access().
inline bool Marked(const char* dir, uid_t uid) {
    if (!IsAppUid(uid)) return false;
    char path[64];
    const int n = snprintf(path, sizeof(path), "%s/%u", dir, static_cast<unsigned>(uid));
    return n > 0 && static_cast<size_t>(n) < sizeof(path) && access(path, F_OK) == 0;
}

inline uint16_t ReadU16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | p[1] << 8); }
inline uint32_t ReadU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
            static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
}

static_assert(kKelirStatusOffset + kKelirStatusSize <= kLapisStatusOffset &&
                      kLapisStatusOffset + kLapisStatusSize <= kUpscaleStatusOffset &&
                      kUpscaleStatusOffset + kUpscaleStatusSize <= kVideoStatusOffset &&
                      kVideoStatusOffset + kVideoStatusSize <= kPresentCountOffset &&
                      kPresentCountOffset + 4 <= kStatusSize,
              "status blocks overlap");

}  // namespace framefx
