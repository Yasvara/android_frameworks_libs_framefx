/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/channel.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace framefx {

// Section views into a validated channel snapshot.
struct Sections {
    uint32_t sequence = 0;
    const uint8_t* luwes = nullptr;
    size_t luwesLength = 0;
    const uint8_t* kelir = nullptr;
    size_t kelirLength = 0;
    const uint8_t* lapis = nullptr;
    size_t lapisLength = 0;
    const uint8_t* upscale = nullptr;
    size_t upscaleLength = 0;
    const uint8_t* texture = nullptr;
    size_t textureLength = 0;
    const uint8_t* videoKelir = nullptr;
    size_t videoKelirLength = 0;
    const uint8_t* video = nullptr;
    size_t videoLength = 0;

    // Armed for sections the game skips (Lepas) alone: nothing here asks the engine for work.
    bool none() const {
        return luwes == nullptr && kelir == nullptr && lapis == nullptr && upscale == nullptr &&
                texture == nullptr && videoKelir == nullptr && video == nullptr;
    }
};

// Validates the container of a consistent snapshot of kChannelSize bytes. Unknown section ids
// are skipped, so an older library ignores a newer feature; a malformed table rejects it all.
bool DecodeChannel(const uint8_t* bytes, Sections* out);

// Copies a consistent snapshot out of a channel the broker may be rewriting. False when every
// attempt overlapped a write; the caller keeps what it had.
bool SnapshotChannel(const uint8_t* shared, uint8_t* out);

struct LuwesSettings {
    int multiplier = 0;
    Mode mode = Mode::kExtrapolate;
    bool hudProtection = true;
    bool ownEngine = false;  // Luwes's own motion even where Qualcomm AFME runs

    bool on() const { return multiplier >= kMinMultiplier; }
    bool operator==(const LuwesSettings& o) const {
        return multiplier == o.multiplier && mode == o.mode && hudProtection == o.hudProtection &&
                ownEngine == o.ownEngine;
    }
    bool operator!=(const LuwesSettings& o) const { return !(*this == o); }
};

// Strict: anything but a well-formed v1 section reads as false and Luwes stays off.
bool DecodeLuwes(const uint8_t* data, size_t length, LuwesSettings* out);

struct LapisSettings {
    LapisMode mode = LapisMode::kAuto;
    bool hideHud = false;  // photo mode: the game's 3D alone, where Lapis has it
    bool textTap = false;
    bool wholeScreen = false;  // Kelir on the finished frame, HUD included, never the scene layer

    bool on() const { return mode != LapisMode::kOff; }
    bool operator==(const LapisSettings& o) const {
        return mode == o.mode && hideHud == o.hideHud && textTap == o.textTap &&
                wholeScreen == o.wholeScreen;
    }
    bool operator!=(const LapisSettings& o) const { return !(*this == o); }
};

// Strict: anything but a well-formed v1 section reads as false, and Lapis stays off.
bool DecodeLapis(const uint8_t* data, size_t length, LapisSettings* out);

struct UpscaleSettings {
    UpscaleMode mode = UpscaleMode::kAuto;
    UpscaleFilter method = UpscaleFilter::kBest;
    uint8_t sharpness = 0;
    bool written = false;  // a section, not the default: it arms the game itself

    // SGSR 1's edge sharpness, 1 to 3; 0 leaves the filter its own.
    float edgeSharpness() const {
        return sharpness == 0 ? 0.0f : 1.0f + static_cast<float>(sharpness - 1) / 127.0f;
    }

    bool on() const { return mode != UpscaleMode::kOff; }
    bool operator==(const UpscaleSettings& o) const {
        return mode == o.mode && method == o.method && sharpness == o.sharpness &&
                written == o.written;
    }
    bool operator!=(const UpscaleSettings& o) const { return !(*this == o); }
};

// Strict: anything but a well-formed v1 section reads as false, and upscaling stays off.
bool DecodeUpscale(const uint8_t* data, size_t length, UpscaleSettings* out);

struct TextureSettings {
    int anisotropy = 0;      // 0: the game's
    int lodBiasEighths = 0;  // <= 0: sharper mip levels
    int shadingRate = 0;     // coarsest rate of depth-tested work: 0 full, 1 2x1, 2 2x2, 3 4x4
    bool shadingAdaptive = false;

    // The texture overrides; shading is apart.
    bool on() const { return anisotropy != 0 || lodBiasEighths != 0; }
    bool shading() const { return shadingRate != 0; }
    bool operator==(const TextureSettings& o) const {
        return anisotropy == o.anisotropy && lodBiasEighths == o.lodBiasEighths &&
                shadingRate == o.shadingRate && shadingAdaptive == o.shadingAdaptive;
    }
    bool operator!=(const TextureSettings& o) const { return !(*this == o); }
};

// Strict: anything but a well-formed v1 section reads as false, and the game's sampling stays.
bool DecodeTexture(const uint8_t* data, size_t length, TextureSettings* out);

struct VideoSettings {
    int motionHz = 0;  // 0: no generated frames
    bool upscale = false;
    bool afme = false;
    float smoothness = 1.0f;
    int displayLong = 0;
    int displayShort = 0;

    bool on() const { return motionHz != 0 || upscale; }
    bool operator==(const VideoSettings& o) const {
        return motionHz == o.motionHz && upscale == o.upscale && afme == o.afme &&
                smoothness == o.smoothness &&
                displayLong == o.displayLong && displayShort == o.displayShort;
    }
    bool operator!=(const VideoSettings& o) const { return !(*this == o); }
};

// Strict: anything but a well-formed v1 section reads as false, and both stay off.
bool DecodeVideo(const uint8_t* data, size_t length, VideoSettings* out);

struct UpscaleStatus {
    UpscaleState state = UpscaleState::kOff;
    UpscaleReason reason = UpscaleReason::kNone;
    uint16_t realFramesPer10s = 0;
    uint16_t sceneWidth = 0;
    uint16_t sceneHeight = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    uint32_t gpuNs = 0;
    UpscaleMethod method = UpscaleMethod::kNone;
    uint16_t drawnWidth = 0;
    uint16_t drawnHeight = 0;

    bool operator==(const UpscaleStatus& o) const {
        return state == o.state && reason == o.reason && realFramesPer10s == o.realFramesPer10s &&
                sceneWidth == o.sceneWidth && sceneHeight == o.sceneHeight && width == o.width &&
                height == o.height && gpuNs == o.gpuNs && method == o.method &&
                drawnWidth == o.drawnWidth && drawnHeight == o.drawnHeight;
    }
    bool operator!=(const UpscaleStatus& o) const { return !(*this == o); }
};

// Encodes the Upscale status block (kUpscaleStatusSize bytes).
void EncodeUpscaleStatus(const UpscaleStatus& status, uint8_t* out);

struct LapisStatus {
    LapisState state = LapisState::kOff;
    uint8_t tier = 0;   // lapis::Tier
    uint8_t roles = 0;  // kLapis* bits
    uint8_t miss = 0;   // lapis::Miss
    uint16_t sceneWidth = 0;
    uint16_t sceneHeight = 0;

    bool operator==(const LapisStatus& o) const {
        return state == o.state && tier == o.tier && roles == o.roles && miss == o.miss &&
                sceneWidth == o.sceneWidth && sceneHeight == o.sceneHeight;
    }
    bool operator!=(const LapisStatus& o) const { return !(*this == o); }
};

// Encodes the Lapis status block (kLapisStatusSize bytes).
void EncodeLapisStatus(const LapisStatus& status, uint8_t* out);

struct LuwesStatus {
    LuwesState state = LuwesState::kOff;
    LuwesReason reason = LuwesReason::kNone;
    Api api = Api::kNone;
    MotionSource source = MotionSource::kNone;
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t multiplier = 0;
    uint8_t generated = 0;
    Mode mode = Mode::kExtrapolate;
    bool paced = false;
    float realFps = 0.0f;
    float presentedFps = 0.0f;
    float panelHz = 0.0f;
    uint32_t framesGenerated = 0;
    bool scene = false;  // generating on the game's scene before its HUD
};

// Encodes the Luwes status block (kLuwesStatusSize bytes).
void EncodeLuwesStatus(const LuwesStatus& status, uint8_t* out);

struct KelirStatus {
    KelirState state = KelirState::kOff;
    KelirReason reason = KelirReason::kNone;
    Api api = Api::kNone;
    FormatClass format = FormatClass::kUnorm8;
    uint16_t width = 0;
    uint16_t height = 0;
    uint8_t rotation = 0;
    KelirPath path = KelirPath::kNone;
    KelirReject reject = KelirReject::kNone;
    uint32_t appliedRevision = 0;
    uint32_t rejectedRevision = 0;
    uint32_t gpuNs = 0;
    uint64_t signature = 0;
    // Inspect only; zero otherwise.
    uint8_t swatch[kKelirSwatchColours * 3] = {};
    uint8_t curves[kMaxKelirNodes][kKelirCurvePoints] = {};
};

// Encodes the Kelir status block (kKelirStatusSize bytes).
void EncodeKelirStatus(const KelirStatus& status, uint8_t* out);

struct VideoStatus {
    VideoState state = VideoState::kOff;
    VideoReason reason = VideoReason::kNone;
    uint16_t sourceWidth = 0;
    uint16_t sourceHeight = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    float fps = 0.0f;
    uint32_t gpuNs = 0;
    uint32_t frames = 0;
    uint32_t dropped = 0;
    uint32_t appliedRevision = 0;
    float presentedFps = 0.0f;
    uint32_t generated = 0;
    uint32_t delayUs = 0;
    bool upscaled = false;
    bool superres = false;
};

// Encodes the Video status block (kVideoStatusSize bytes).
void EncodeVideoStatus(const VideoStatus& status, uint8_t* out);

// The game's side of its status page: a sequence-locked writer over shared memory. Writes are
// rare (about once a second per target) and may come from any render thread.
class StatusPage {
public:
    // `page` is kStatusSize bytes of read-write shared memory.
    void attach(uint8_t* page);
    void writeLuwes(const LuwesStatus& status);
    void writeKelir(const KelirStatus& status);
    void writeLapis(const LapisStatus& status);
    void writeUpscale(const UpscaleStatus& status);
    void writeVideo(const VideoStatus& status);
    // One present: a relaxed add on the page, no lock.
    void countPresent() {
        if (uint8_t* page = __atomic_load_n(&page_, __ATOMIC_ACQUIRE)) {
            __atomic_fetch_add(reinterpret_cast<uint32_t*>(page + kPresentCountOffset), 1u,
                               __ATOMIC_RELAXED);
        }
    }

private:
    void write(size_t offset, const uint8_t* block, size_t size);

    std::mutex lock_;
    uint8_t* page_ = nullptr;
};

}  // namespace framefx
