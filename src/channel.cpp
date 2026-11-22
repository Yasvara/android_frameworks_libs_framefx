/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "channel.h"

#include <algorithm>
#include <atomic>
#include <cstring>

namespace framefx {
namespace {

void putU16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}

void putU32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

void putF32(uint8_t* p, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    putU32(p, bits);
}

uint32_t* sequenceWord(uint8_t* base) {
    return reinterpret_cast<uint32_t*>(base + kSequenceOffset);
}

}  // namespace

bool DecodeChannel(const uint8_t* r, Sections* out) {
    if (ReadU32(r) != kChannelMagic || ReadU16(r + 4) != kChannelVersion ||
        ReadU16(r + 6) != kChannelSize) {
        return false;
    }
    const uint32_t sequence = ReadU32(r + kSequenceOffset);
    const uint8_t count = r[12];
    if ((sequence & 1u) != 0 || count > kMaxSections || r[13] != 0 || ReadU16(r + 14) != 0) {
        return false;
    }
    Sections s;
    s.sequence = sequence;
    uint32_t seen = 0;
    size_t starts[kMaxSections] = {};
    size_t ends[kMaxSections] = {};
    for (int i = 0; i < count; i++) {
        const uint8_t* e = r + kSectionTableOffset + 8 * i;
        const uint8_t id = e[0];
        const size_t offset = ReadU16(e + 2);
        const size_t length = ReadU16(e + 4);
        if (id == 0 || id > 31 || e[1] != 0 || ReadU16(e + 6) != 0) return false;
        if ((seen & (1u << id)) != 0) return false;
        if (offset < kSectionDataOffset || length == 0 || offset + length > kChannelSize) {
            return false;
        }
        for (int j = 0; j < i; j++) {
            if (offset < ends[j] && starts[j] < offset + length) return false;
        }
        seen |= 1u << id;
        starts[i] = offset;
        ends[i] = offset + length;
        if (id == kSectionLuwes) {
            s.luwes = r + offset;
            s.luwesLength = length;
        } else if (id == kSectionKelir) {
            s.kelir = r + offset;
            s.kelirLength = length;
        } else if (id == kSectionLapis) {
            s.lapis = r + offset;
            s.lapisLength = length;
        } else if (id == kSectionUpscale) {
            s.upscale = r + offset;
            s.upscaleLength = length;
        } else if (id == kSectionTexture) {
            s.texture = r + offset;
            s.textureLength = length;
        } else if (id == kSectionVideoKelir) {
            s.videoKelir = r + offset;
            s.videoKelirLength = length;
        } else if (id == kSectionVideo) {
            s.video = r + offset;
            s.videoLength = length;
        }
    }
    *out = s;
    return true;
}

bool SnapshotChannel(const uint8_t* shared, uint8_t* out) {
    const auto* word = reinterpret_cast<const uint32_t*>(shared + kSequenceOffset);
    for (int attempt = 0; attempt < 4; attempt++) {
        const uint32_t before = __atomic_load_n(word, __ATOMIC_ACQUIRE);
        if ((before & 1u) != 0) continue;
        memcpy(out, shared, kChannelSize);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (__atomic_load_n(word, __ATOMIC_RELAXED) == before) {
            putU32(out + kSequenceOffset, before);
            return true;
        }
    }
    return false;
}

bool DecodeLuwes(const uint8_t* d, size_t length, LuwesSettings* out) {
    if (d == nullptr || length != kLuwesSectionSize) return false;
    const uint8_t multiplier = d[1];
    if (d[0] != kLuwesMajor ||
        (multiplier != 0 && (multiplier < kMinMultiplier || multiplier > kMaxMultiplier)) ||
        d[2] > static_cast<uint8_t>(Mode::kInterpolate) ||
        (d[3] & ~(kLuwesHudProtection | kLuwesOwnEngine)) != 0 ||
        ReadU32(d + 4) != 0) {
        return false;
    }
    out->multiplier = multiplier;
    out->mode = static_cast<Mode>(d[2]);
    out->hudProtection = (d[3] & kLuwesHudProtection) != 0;
    out->ownEngine = (d[3] & kLuwesOwnEngine) != 0;
    return true;
}

bool DecodeLapis(const uint8_t* d, size_t length, LapisSettings* out) {
    if (d == nullptr || length != kLapisSectionSize || d[0] != kLapisMajor ||
        d[1] > static_cast<uint8_t>(LapisMode::kAuto) || (d[2] & ~(kLapisHideHud | kLapisTextTap | kLapisWholeScreen)) != 0 ||
        d[3] != 0) {
        return false;
    }
    out->mode = static_cast<LapisMode>(d[1]);
    out->hideHud = (d[2] & kLapisHideHud) != 0;
    out->textTap = (d[2] & kLapisTextTap) != 0;
    out->wholeScreen = (d[2] & kLapisWholeScreen) != 0;
    return true;
}

bool DecodeUpscale(const uint8_t* d, size_t length, UpscaleSettings* out) {
    if (d == nullptr || length != kUpscaleSectionSize || d[0] != kUpscaleMajor ||
        d[1] > static_cast<uint8_t>(UpscaleMode::kAuto) ||
        d[2] > static_cast<uint8_t>(UpscaleFilter::kSpatial)) {
        return false;
    }
    out->mode = static_cast<UpscaleMode>(d[1]);
    out->method = static_cast<UpscaleFilter>(d[2]);
    out->sharpness = d[3];
    out->written = true;
    return true;
}

bool DecodeTexture(const uint8_t* d, size_t length, TextureSettings* out) {
    if (d == nullptr || length != kTextureSectionSize || d[0] != kTextureMajor ||
        (d[3] & ~(kShadingRateMask | kShadingAdaptive)) != 0 ||
        ((d[3] & kShadingAdaptive) != 0 && (d[3] & kShadingRateMask) == 0)) {
        return false;
    }
    const int anisotropy = d[1];
    const int bias = static_cast<int8_t>(d[2]);
    if ((anisotropy != 0 && anisotropy != 2 && anisotropy != 4 && anisotropy != 8 &&
         anisotropy != 16) ||
        bias < kMinLodBiasEighths || bias > 0) {
        return false;
    }
    out->anisotropy = anisotropy;
    out->lodBiasEighths = bias;
    out->shadingRate = d[3] & kShadingRateMask;
    out->shadingAdaptive = (d[3] & kShadingAdaptive) != 0;
    return true;
}

bool DecodeVideo(const uint8_t* d, size_t length, VideoSettings* out) {
    if (d == nullptr || length != kVideoSectionSize || d[0] != kVideoMajor ||
        (d[1] != 0 && d[1] < kMinVideoHz) || (d[2] & ~(kVideoUpscale | kVideoAfme)) != 0) {
        return false;
    }
    out->motionHz = d[1];
    out->upscale = (d[2] & kVideoUpscale) != 0;
    out->afme = (d[2] & kVideoAfme) != 0;
    out->smoothness = static_cast<float>(d[3]) / 255.0f;
    out->displayLong = ReadU16(d + 4);
    out->displayShort = ReadU16(d + 6);
    return true;
}

void EncodeUpscaleStatus(const UpscaleStatus& s, uint8_t* out) {
    memset(out, 0, kUpscaleStatusSize);
    out[0] = static_cast<uint8_t>(s.state);
    out[1] = static_cast<uint8_t>(s.reason);
    putU16(out + 2, s.realFramesPer10s);
    putU16(out + 4, s.sceneWidth);
    putU16(out + 6, s.sceneHeight);
    putU16(out + 8, s.width);
    putU16(out + 10, s.height);
    putU32(out + 12, s.gpuNs);
    out[16] = static_cast<uint8_t>(s.method);
    putU16(out + 18, s.drawnWidth);
    putU16(out + 20, s.drawnHeight);
}

void EncodeLapisStatus(const LapisStatus& s, uint8_t* out) {
    memset(out, 0, kLapisStatusSize);
    out[0] = static_cast<uint8_t>(s.state);
    out[1] = s.tier;
    out[2] = s.roles;
    out[3] = s.miss;
    putU16(out + 4, s.sceneWidth);
    putU16(out + 6, s.sceneHeight);
}

void EncodeLuwesStatus(const LuwesStatus& s, uint8_t* out) {
    memset(out, 0, kLuwesStatusSize);
    out[0] = static_cast<uint8_t>(s.state);
    out[1] = static_cast<uint8_t>(s.reason);
    out[2] = static_cast<uint8_t>(s.api);
    out[3] = static_cast<uint8_t>(s.source);
    putU16(out + 4, s.width);
    putU16(out + 6, s.height);
    out[8] = s.multiplier;
    out[9] = s.generated;
    out[10] = static_cast<uint8_t>(s.mode);
    out[11] = s.paced ? 1 : 0;
    putF32(out + 12, s.realFps);
    putF32(out + 16, s.presentedFps);
    putF32(out + 20, s.panelHz);
    putU32(out + 24, s.framesGenerated);
    out[28] = s.scene ? kLuwesStatusScene : 0;
}

void EncodeKelirStatus(const KelirStatus& s, uint8_t* out) {
    memset(out, 0, kKelirStatusSize);
    out[0] = static_cast<uint8_t>(s.state);
    out[1] = static_cast<uint8_t>(s.reason);
    out[2] = static_cast<uint8_t>(s.api);
    out[3] = static_cast<uint8_t>(s.format);
    putU16(out + 4, s.width);
    putU16(out + 6, s.height);
    out[8] = s.rotation;
    out[9] = static_cast<uint8_t>(s.path);
    out[10] = static_cast<uint8_t>(s.reject);
    putU32(out + 12, s.appliedRevision);
    putU32(out + 16, s.rejectedRevision);
    putU32(out + 20, s.gpuNs);
    putU32(out + 24, static_cast<uint32_t>(s.signature));
    putU32(out + 28, static_cast<uint32_t>(s.signature >> 32));
    memcpy(out + 32, s.swatch, sizeof(s.swatch));
    memcpy(out + 56, s.curves, sizeof(s.curves));
}

void EncodeVideoStatus(const VideoStatus& s, uint8_t* out) {
    memset(out, 0, kVideoStatusSize);
    out[0] = static_cast<uint8_t>(s.state);
    out[1] = static_cast<uint8_t>(s.reason);
    putU16(out + 4, s.sourceWidth);
    putU16(out + 6, s.sourceHeight);
    putU16(out + 8, s.width);
    putU16(out + 10, s.height);
    putF32(out + 12, s.fps);
    putU32(out + 16, s.gpuNs);
    putU32(out + 20, s.frames);
    putU32(out + 24, s.dropped);
    putU32(out + 28, s.appliedRevision);
    putF32(out + 32, s.presentedFps);
    putU32(out + 36, s.generated);
    putU16(out + 40, static_cast<uint16_t>(std::min<uint32_t>(s.delayUs / 100, 0xffff)));
    out[42] = (s.upscaled ? kVideoStatusUpscaled : 0) | (s.superres ? kVideoStatusSuperres : 0);
}

void StatusPage::attach(uint8_t* page) {
    std::lock_guard lock(lock_);
    __atomic_store_n(sequenceWord(page), 1u, __ATOMIC_RELAXED);
    std::atomic_thread_fence(std::memory_order_release);
    memset(page + 12, 0, kStatusSize - 12);
    putU32(page, kStatusMagic);
    putU16(page + 4, kStatusVersion);
    putU16(page + 6, kStatusSize);
    __atomic_store_n(sequenceWord(page), 2u, __ATOMIC_RELEASE);
    // Published last: presents count on a page whose header is already in place.
    __atomic_store_n(&page_, page, __ATOMIC_RELEASE);
}

void StatusPage::writeLuwes(const LuwesStatus& status) {
    uint8_t block[kLuwesStatusSize];
    EncodeLuwesStatus(status, block);
    write(kLuwesStatusOffset, block, sizeof(block));
}

void StatusPage::writeKelir(const KelirStatus& status) {
    uint8_t block[kKelirStatusSize];
    EncodeKelirStatus(status, block);
    write(kKelirStatusOffset, block, sizeof(block));
}

void StatusPage::writeUpscale(const UpscaleStatus& status) {
    uint8_t block[kUpscaleStatusSize];
    EncodeUpscaleStatus(status, block);
    write(kUpscaleStatusOffset, block, sizeof(block));
}

void StatusPage::writeLapis(const LapisStatus& status) {
    uint8_t block[kLapisStatusSize];
    EncodeLapisStatus(status, block);
    write(kLapisStatusOffset, block, sizeof(block));
}

void StatusPage::writeVideo(const VideoStatus& status) {
    uint8_t block[kVideoStatusSize];
    EncodeVideoStatus(status, block);
    write(kVideoStatusOffset, block, sizeof(block));
}

void StatusPage::write(size_t offset, const uint8_t* block, size_t size) {
    std::lock_guard lock(lock_);
    if (page_ == nullptr) return;
    uint32_t* word = sequenceWord(page_);
    const uint32_t sequence = __atomic_load_n(word, __ATOMIC_RELAXED);
    __atomic_store_n(word, sequence + 1, __ATOMIC_RELAXED);
    std::atomic_thread_fence(std::memory_order_release);
    memcpy(page_ + offset, block, size);
    __atomic_store_n(word, sequence + 2, __ATOMIC_RELEASE);
}

}  // namespace framefx
