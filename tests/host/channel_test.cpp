/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <array>
#include <framefx/hook.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "src/channel.h"

namespace framefx {
namespace {

void put16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}

void put32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

struct Entry {
    uint8_t id;
    uint16_t offset;
    std::vector<uint8_t> bytes;
};

// Lays a channel out the way the broker does: sections packed from kSectionDataOffset in table
// order unless an entry names its own offset.
std::vector<uint8_t> Channel(const std::vector<Entry>& entries, uint32_t sequence = 2) {
    std::vector<uint8_t> c(kChannelSize, 0);
    put32(c.data(), kChannelMagic);
    put16(c.data() + 4, kChannelVersion);
    put16(c.data() + 6, kChannelSize);
    put32(c.data() + kSequenceOffset, sequence);
    c[12] = static_cast<uint8_t>(entries.size());
    size_t next = kSectionDataOffset;
    for (size_t i = 0; i < entries.size(); i++) {
        const Entry& e = entries[i];
        const size_t offset = e.offset != 0 ? e.offset : next;
        uint8_t* t = c.data() + kSectionTableOffset + 8 * i;
        t[0] = e.id;
        put16(t + 2, static_cast<uint16_t>(offset));
        put16(t + 4, static_cast<uint16_t>(e.bytes.size()));
        if (offset + e.bytes.size() <= kChannelSize) {
            memcpy(c.data() + offset, e.bytes.data(), e.bytes.size());
        }
        next = offset + e.bytes.size();
    }
    return c;
}

// Pentas encodes these exact bytes for Luwes 3x, Interpolate, HUD protection on
// (LuwesSettingsTest in Pentas pins the same array).
const std::vector<uint8_t> kLuwes3xInterpolateHud = {1, 3, 1, 1, 0, 0, 0, 0};

TEST(Channel, DecodesBothSections) {
    const std::vector<uint8_t> kelir(kKelirHeaderSize + kKelirNodeSize, 7);
    const auto c = Channel({{kSectionLuwes, 0, kLuwes3xInterpolateHud}, {kSectionKelir, 0, kelir}},
                           /*sequence=*/42);
    Sections s;
    ASSERT_TRUE(DecodeChannel(c.data(), &s));
    EXPECT_EQ(42u, s.sequence);
    ASSERT_EQ(kLuwesSectionSize, s.luwesLength);
    EXPECT_EQ(c.data() + kSectionDataOffset, s.luwes);
    ASSERT_EQ(kelir.size(), s.kelirLength);
    EXPECT_EQ(0, memcmp(kelir.data(), s.kelir, kelir.size()));

    LuwesSettings luwes;
    ASSERT_TRUE(DecodeLuwes(s.luwes, s.luwesLength, &luwes));
    EXPECT_EQ(3, luwes.multiplier);
    EXPECT_EQ(Mode::kInterpolate, luwes.mode);
    EXPECT_TRUE(luwes.hudProtection);
    EXPECT_TRUE(luwes.on());
}

TEST(Channel, EmptyChannelHasNoSections) {
    const auto c = Channel({});
    Sections s;
    ASSERT_TRUE(DecodeChannel(c.data(), &s));
    EXPECT_EQ(nullptr, s.luwes);
    EXPECT_EQ(nullptr, s.kelir);
    LuwesSettings luwes;
    EXPECT_FALSE(DecodeLuwes(s.luwes, s.luwesLength, &luwes));
}

TEST(Channel, UnknownSectionsAreSkipped) {
    const auto c = Channel({{9, 0, {1, 2, 3}}, {kSectionLuwes, 0, kLuwes3xInterpolateHud}});
    Sections s;
    ASSERT_TRUE(DecodeChannel(c.data(), &s));
    EXPECT_EQ(kLuwesSectionSize, s.luwesLength);
    EXPECT_EQ(nullptr, s.kelir);
}

TEST(Channel, RejectsBadHeaders) {
    const auto good = Channel({{kSectionLuwes, 0, kLuwes3xInterpolateHud}});
    Sections s;
    ASSERT_TRUE(DecodeChannel(good.data(), &s));
    struct Case {
        const char* what;
        size_t at;
        uint8_t value;
    };
    for (const Case& k : {Case{"magic", 0, 'X'}, Case{"version", 4, 2}, Case{"size", 7, 0},
                          Case{"write in flight", kSequenceOffset, 3},
                          Case{"too many sections", 12, kMaxSections + 1},
                          Case{"reserved 13", 13, 1}, Case{"reserved 14", 14, 1}}) {
        auto c = good;
        c[k.at] = k.value;
        EXPECT_FALSE(DecodeChannel(c.data(), &s)) << k.what;
    }
}

TEST(Channel, RejectsBadTables) {
    const std::vector<uint8_t> l = kLuwes3xInterpolateHud;
    Sections s;
    EXPECT_FALSE(DecodeChannel(Channel({{0, 0, l}}).data(), &s)) << "id 0";
    EXPECT_FALSE(DecodeChannel(Channel({{32, 0, l}}).data(), &s)) << "id out of range";
    EXPECT_FALSE(DecodeChannel(Channel({{kSectionLuwes, 0, l}, {kSectionLuwes, 0, l}}).data(), &s))
            << "duplicate";
    EXPECT_FALSE(DecodeChannel(Channel({{kSectionLuwes, 0, {}}}).data(), &s)) << "empty";
    EXPECT_FALSE(DecodeChannel(Channel({{kSectionLuwes, 16, l}}).data(), &s)) << "inside table";
    EXPECT_FALSE(DecodeChannel(Channel({{kSectionLuwes, kChannelSize - 4, l}}).data(), &s))
            << "past the end";
    EXPECT_FALSE(DecodeChannel(Channel({{kSectionLuwes, 100, l}, {kSectionKelir, 104, l}}).data(),
                               &s))
            << "overlap";
    EXPECT_TRUE(DecodeChannel(Channel({{kSectionLuwes, 100, l}, {kSectionKelir, 108, l}}).data(),
                              &s))
            << "adjacent";

    auto reserved = Channel({{kSectionLuwes, 0, l}});
    reserved[kSectionTableOffset + 1] = 1;
    EXPECT_FALSE(DecodeChannel(reserved.data(), &s));
    reserved = Channel({{kSectionLuwes, 0, l}});
    reserved[kSectionTableOffset + 6] = 1;
    EXPECT_FALSE(DecodeChannel(reserved.data(), &s));
}

TEST(Channel, FailedDecodeLeavesTheOutputAlone) {
    auto c = Channel({{kSectionLuwes, 0, kLuwes3xInterpolateHud}}, 8);
    Sections s;
    ASSERT_TRUE(DecodeChannel(c.data(), &s));
    c[0] = 0;
    EXPECT_FALSE(DecodeChannel(c.data(), &s));
    EXPECT_EQ(8u, s.sequence);
    EXPECT_NE(nullptr, s.luwes);
}

TEST(Luwes, AcceptsEveryValidSetting) {
    for (int multiplier : {0, 2, 3, 4}) {
        for (int mode : {0, 1}) {
            for (int flags : {0, 1, 2, 3}) {
                const uint8_t d[] = {1, static_cast<uint8_t>(multiplier),
                                     static_cast<uint8_t>(mode), static_cast<uint8_t>(flags),
                                     0, 0, 0, 0};
                LuwesSettings s;
                ASSERT_TRUE(DecodeLuwes(d, sizeof(d), &s));
                EXPECT_EQ(multiplier, s.multiplier);
                EXPECT_EQ(static_cast<Mode>(mode), s.mode);
                EXPECT_EQ((flags & 1) != 0, s.hudProtection);
                EXPECT_EQ((flags & 2) != 0, s.ownEngine);
                EXPECT_EQ(multiplier != 0, s.on());
            }
        }
    }
}

TEST(Luwes, RejectsAnythingElse) {
    const uint8_t good[] = {1, 2, 0, 1, 0, 0, 0, 0};
    LuwesSettings s;
    ASSERT_TRUE(DecodeLuwes(good, sizeof(good), &s));
    EXPECT_FALSE(DecodeLuwes(good, sizeof(good) - 1, &s)) << "short";
    const uint8_t longer[] = {1, 2, 0, 1, 0, 0, 0, 0, 0};
    EXPECT_FALSE(DecodeLuwes(longer, sizeof(longer), &s)) << "long";
    struct Case {
        const char* what;
        size_t at;
        uint8_t value;
    };
    for (const Case& k : {Case{"major", 0, 2}, Case{"multiplier 1", 1, 1},
                          Case{"multiplier 5", 1, 5}, Case{"mode", 2, 2}, Case{"flags", 3, 4},
                          Case{"reserved", 7, 1}}) {
        uint8_t d[sizeof(good)];
        memcpy(d, good, sizeof(d));
        d[k.at] = k.value;
        LuwesSettings untouched;
        untouched.multiplier = 4;
        EXPECT_FALSE(DecodeLuwes(d, sizeof(d), &untouched)) << k.what;
        EXPECT_EQ(4, untouched.multiplier) << k.what;
    }
}

TEST(Lapis, AbsentIsAuto) {
    LapisSettings s;
    EXPECT_TRUE(s.on());
    Sections sections;
    ASSERT_TRUE(DecodeChannel(Channel({{kSectionLuwes, 0, kLuwes3xInterpolateHud}}).data(),
                              &sections));
    EXPECT_EQ(nullptr, sections.lapis);
}

TEST(Lapis, DecodesOffAndAuto) {
    Sections sections;
    ASSERT_TRUE(DecodeChannel(Channel({{kSectionLapis, 0, {1, 0, 0, 0}}}).data(), &sections));
    LapisSettings s;
    ASSERT_TRUE(DecodeLapis(sections.lapis, sections.lapisLength, &s));
    EXPECT_FALSE(s.on());
    const uint8_t on[] = {1, 1, 0, 0};
    ASSERT_TRUE(DecodeLapis(on, sizeof(on), &s));
    EXPECT_EQ(LapisMode::kAuto, s.mode);
}

TEST(Lapis, DecodesHideHud) {
    const uint8_t hidden[] = {1, 1, kLapisHideHud, 0};
    LapisSettings s;
    ASSERT_TRUE(DecodeLapis(hidden, sizeof(hidden), &s));
    EXPECT_TRUE(s.hideHud);
    EXPECT_TRUE(s.on());
    const uint8_t shown[] = {1, 1, 0, 0};
    ASSERT_TRUE(DecodeLapis(shown, sizeof(shown), &s));
    EXPECT_FALSE(s.hideHud);
}

TEST(Lapis, DecodesTextTap) {
    const uint8_t tapped[] = {1, 1, kLapisTextTap, 0};
    LapisSettings s;
    ASSERT_TRUE(DecodeLapis(tapped, sizeof(tapped), &s));
    EXPECT_TRUE(s.textTap);
    EXPECT_FALSE(s.hideHud);
    const uint8_t plain[] = {1, 1, 0, 0};
    ASSERT_TRUE(DecodeLapis(plain, sizeof(plain), &s));
    EXPECT_FALSE(s.textTap);
}

TEST(Lapis, DecodesWholeScreen) {
    const uint8_t whole[] = {1, 1, kLapisWholeScreen, 0};
    LapisSettings s;
    ASSERT_TRUE(DecodeLapis(whole, sizeof(whole), &s));
    EXPECT_TRUE(s.wholeScreen);
    EXPECT_FALSE(s.hideHud);
    EXPECT_TRUE(s.on());
    LapisSettings under;
    ASSERT_TRUE(DecodeLapis(std::array<uint8_t, 4>{1, 1, 0, 0}.data(), 4, &under));
    EXPECT_FALSE(under.wholeScreen);
    EXPECT_NE(s, under);
}

TEST(Lapis, RejectsAnythingElse) {
    for (const std::vector<uint8_t>& d :
         std::vector<std::vector<uint8_t>>{{1, 1, 0}, {1, 1, 0, 0, 0}, {2, 1, 0, 0}, {1, 2, 0, 0},
                                           {1, 1, 8, 0}, {1, 1, 0x80, 0}}) {
        LapisSettings s;
        EXPECT_FALSE(DecodeLapis(d.data(), d.size(), &s));
        EXPECT_EQ(LapisMode::kAuto, s.mode);
    }
    LapisSettings s;
    EXPECT_FALSE(DecodeLapis(nullptr, 0, &s));
}

TEST(Upscale, AbsentIsAutoButDoesNotArm) {
    UpscaleSettings s;
    EXPECT_TRUE(s.on());
    EXPECT_FALSE(s.written);
}

TEST(Upscale, DecodesOffAndAuto) {
    Sections sections;
    ASSERT_TRUE(DecodeChannel(Channel({{kSectionUpscale, 0, {1, 1, 0, 0}}}).data(), &sections));
    UpscaleSettings s;
    ASSERT_TRUE(DecodeUpscale(sections.upscale, sections.upscaleLength, &s));
    EXPECT_TRUE(s.on());
    EXPECT_TRUE(s.written);
    const uint8_t off[] = {1, 0, 0, 0};
    ASSERT_TRUE(DecodeUpscale(off, sizeof(off), &s));
    EXPECT_FALSE(s.on());
}

TEST(Upscale, DecodesMethodAndSharpness) {
    const uint8_t d[] = {1, 1, 1, 255};
    UpscaleSettings s;
    ASSERT_TRUE(DecodeUpscale(d, sizeof(d), &s));
    EXPECT_EQ(s.method, UpscaleFilter::kSpatial);
    EXPECT_FLOAT_EQ(s.edgeSharpness(), 3.0f);
    const uint8_t soft[] = {1, 1, 0, 1};
    ASSERT_TRUE(DecodeUpscale(soft, sizeof(soft), &s));
    EXPECT_FLOAT_EQ(s.edgeSharpness(), 1.0f);
    EXPECT_FLOAT_EQ(UpscaleSettings().edgeSharpness(), 0.0f);
}

TEST(Upscale, RejectsAnythingElse) {
    for (const std::vector<uint8_t>& d :
         std::vector<std::vector<uint8_t>>{{1, 1, 0}, {1, 1, 0, 0, 0}, {2, 1, 0, 0}, {1, 2, 0, 0},
                                           {1, 1, 4, 0}, {1, 1, 0x80, 0}}) {
        UpscaleSettings s;
        EXPECT_FALSE(DecodeUpscale(d.data(), d.size(), &s));
        EXPECT_FALSE(s.written);
    }
}

// Whatever lands in the region, decoding returns and every view stays inside it.
TEST(Channel, RandomBytesNeverEscapeTheRegion) {
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_int_distribution<int> small(0, 9);
    int accepted = 0;
    for (int round = 0; round < 20000; round++) {
        auto c = Channel({{kSectionLuwes, 0, kLuwes3xInterpolateHud}, {kSectionKelir, 0, {1, 0}}});
        // Mostly table damage, sometimes anywhere in the header.
        const int flips = 1 + small(rng);
        for (int i = 0; i < flips; i++) {
            const size_t at = small(rng) < 8 ? 12 + byte(rng) % 68 : byte(rng) % 16;
            c[at] = static_cast<uint8_t>(byte(rng));
        }
        Sections s;
        if (!DecodeChannel(c.data(), &s)) continue;
        accepted++;
        const uint8_t* begin = c.data() + kSectionDataOffset;
        const uint8_t* end = c.data() + kChannelSize;
        if (s.luwes != nullptr) {
            EXPECT_GE(s.luwes, begin);
            EXPECT_LE(s.luwes + s.luwesLength, end);
            LuwesSettings luwes;
            if (DecodeLuwes(s.luwes, s.luwesLength, &luwes)) {
                EXPECT_TRUE(luwes.multiplier == 0 || (luwes.multiplier >= kMinMultiplier &&
                                                      luwes.multiplier <= kMaxMultiplier));
            }
        }
        if (s.kelir != nullptr) {
            EXPECT_GE(s.kelir, begin);
            EXPECT_LE(s.kelir + s.kelirLength, end);
        }
    }
    EXPECT_GT(accepted, 0);
}

TEST(Channel, FullyRandomRegionsAreRejectedOrBounded) {
    std::mt19937 rng(5);
    std::vector<uint8_t> c(kChannelSize);
    for (int round = 0; round < 2000; round++) {
        for (auto& b : c) b = static_cast<uint8_t>(rng());
        if (round % 2 == 0) {  // a valid header over a random table
            put32(c.data(), kChannelMagic);
            put16(c.data() + 4, kChannelVersion);
            put16(c.data() + 6, kChannelSize);
            c[kSequenceOffset] &= 0xfe;
        }
        Sections s;
        if (DecodeChannel(c.data(), &s) && s.luwes != nullptr) {
            EXPECT_LE(s.luwes + s.luwesLength, c.data() + kChannelSize);
        }
    }
}

// The broker's write: odd sequence, rewrite, even sequence. A reader racing it gets an older or
// a newer consistent region, never a mix.
TEST(Channel, SnapshotsAreNeverTorn) {
    std::vector<uint8_t> shared = Channel({{kSectionLuwes, 0, kLuwes3xInterpolateHud}});
    auto* word = reinterpret_cast<uint32_t*>(shared.data() + kSequenceOffset);
    std::atomic<bool> stop{false};
    std::thread writer([&] {
        uint32_t sequence = 2;
        for (int round = 0; round < 200000 && !stop.load(std::memory_order_relaxed); round++) {
            __atomic_store_n(word, sequence + 1, __ATOMIC_RELAXED);
            std::atomic_thread_fence(std::memory_order_release);
            const uint8_t fill = static_cast<uint8_t>(sequence / 2);
            for (size_t i = kSectionDataOffset + kLuwesSectionSize; i < kChannelSize; i++) {
                __atomic_store_n(&shared[i], fill, __ATOMIC_RELAXED);
            }
            sequence += 2;
            __atomic_store_n(word, sequence, __ATOMIC_RELEASE);
            // Pentas writes now and then; a reader must also get to see the page at rest.
            std::this_thread::yield();
        }
    });
    std::vector<uint8_t> copy(kChannelSize);
    int consistent = 0;
    for (int i = 0; i < 20000; i++) {
        if (!SnapshotChannel(shared.data(), copy.data())) continue;
        consistent++;
        Sections s;
        ASSERT_TRUE(DecodeChannel(copy.data(), &s));
        const uint8_t fill = copy[kSectionDataOffset + kLuwesSectionSize];
        for (size_t j = kSectionDataOffset + kLuwesSectionSize; j < kChannelSize; j++) {
            ASSERT_EQ(fill, copy[j]) << "torn at " << j << " sequence " << s.sequence;
        }
        if (s.sequence > 2) {
            ASSERT_EQ(static_cast<uint8_t>((s.sequence - 2) / 2), fill);
        }
    }
    stop = true;
    writer.join();
    EXPECT_GT(consistent, 0);
}

TEST(Channel, SnapshotGivesUpOnAWriteThatNeverEnds) {
    auto shared = Channel({});
    put32(shared.data() + kSequenceOffset, 7);
    std::vector<uint8_t> copy(kChannelSize, 0xee);
    EXPECT_FALSE(SnapshotChannel(shared.data(), copy.data()));
    EXPECT_EQ(0xee, copy[0]);
}

TEST(Status, PageHeaderAndLuwesBlockAreByteExact) {
    std::vector<uint8_t> page(kStatusSize, 0xaa);
    StatusPage writer;
    writer.attach(page.data());
    EXPECT_EQ(kStatusMagic, ReadU32(page.data()));
    EXPECT_EQ(kStatusVersion, ReadU16(page.data() + 4));
    EXPECT_EQ(kStatusSize, ReadU16(page.data() + 6));
    EXPECT_EQ(2u, ReadU32(page.data() + kSequenceOffset));
    EXPECT_EQ(0, page[kKelirStatusOffset]);
    EXPECT_EQ(0, page[kStatusSize - 1]);

    LuwesStatus s;
    s.state = LuwesState::kRunning;
    s.api = Api::kVulkan;
    s.source = MotionSource::kVulkanBlockMatch;
    s.width = 1968;
    s.height = 908;
    s.multiplier = 3;
    s.generated = 2;
    s.mode = Mode::kInterpolate;
    s.paced = true;
    s.realFps = 45.0f;
    s.presentedFps = 135.0f;
    s.panelHz = 135.0f;
    s.framesGenerated = 0x01020304;
    writer.writeLuwes(s);
    EXPECT_EQ(4u, ReadU32(page.data() + kSequenceOffset));
    const uint8_t expected[28] = {2,    0,    2,    3,    0xb0, 0x07, 0x8c, 0x03, 3,    2,
                                  1,    1,    0,    0,    0x34, 0x42, 0,    0,    0x07, 0x43,
                                  0,    0,    0x07, 0x43, 4,    3,    2,    1};
    EXPECT_EQ(0, memcmp(expected, page.data() + kLuwesStatusOffset, sizeof(expected)));
    for (size_t i = kLuwesStatusOffset + sizeof(expected); i < kLuwesStatusOffset + kLuwesStatusSize;
         i++) {
        EXPECT_EQ(0, page[i]) << i;
    }
}

TEST(Status, LuwesNamesItsEngineAndScene) {
    LuwesStatus s;
    s.source = MotionSource::kAfme;
    s.scene = true;
    uint8_t block[kLuwesStatusSize];
    EncodeLuwesStatus(s, block);
    EXPECT_EQ(5, block[3]);
    EXPECT_EQ(kLuwesStatusScene, block[28]);
}

TEST(Status, LapisBlockIsByteExact) {
    std::vector<uint8_t> page(kStatusSize, 0xaa);
    StatusPage writer;
    writer.attach(page.data());
    LapisStatus s;
    s.state = LapisState::kEngaged;
    s.tier = 1;
    s.roles = kLapisDepth | kLapisKept | kLapisScene;
    s.miss = 0;
    s.sceneWidth = 2400;
    s.sceneHeight = 1080;
    writer.writeLapis(s);
    const uint8_t expected[kLapisStatusSize] = {2, 1, 7, 0, 0x60, 0x09, 0x38, 0x04};
    EXPECT_EQ(0, memcmp(expected, page.data() + kLapisStatusOffset, sizeof(expected)));
    EXPECT_EQ(0, page[kLuwesStatusOffset]);  // other blocks untouched by this write
}

TEST(Status, UpscaleBlockIsByteExact) {
    std::vector<uint8_t> page(kStatusSize, 0xaa);
    StatusPage writer;
    writer.attach(page.data());
    UpscaleStatus s;
    s.state = UpscaleState::kRunning;
    s.realFramesPer10s = 600;
    s.sceneWidth = 1124;
    s.sceneHeight = 518;
    s.width = 2119;
    s.height = 979;
    s.gpuNs = 610000;
    s.method = UpscaleMethod::kSgsr2;
    s.drawnWidth = 2119;
    s.drawnHeight = 979;
    writer.writeUpscale(s);
    const uint8_t expected[kUpscaleStatusSize] = {2,    0,    0x58, 0x02, 0x64, 0x04, 0x06, 0x02,
                                                  0x47, 0x08, 0xd3, 0x03, 0xd0, 0x4e, 0x09, 0x00,
                                                  2,    0,    0x47, 0x08, 0xd3, 0x03, 0,    0};
    EXPECT_EQ(0, memcmp(expected, page.data() + kUpscaleStatusOffset, sizeof(expected)));
    EXPECT_EQ(0, page[kLapisStatusOffset]);
}

TEST(Status, VideoBlockIsByteExact) {
    std::vector<uint8_t> page(kStatusSize, 0xaa);
    StatusPage writer;
    writer.attach(page.data());
    VideoStatus s;
    s.state = VideoState::kRunning;
    s.sourceWidth = 1920;
    s.sourceHeight = 1088;
    s.width = 1920;
    s.height = 1080;
    s.fps = 24.0f;
    s.gpuNs = 2500000;
    s.frames = 1000;
    s.dropped = 3;
    s.appliedRevision = 7;
    s.presentedFps = 120.0f;
    s.generated = 3000;
    s.delayUs = 12300;
    s.upscaled = true;
    s.superres = true;
    writer.writeVideo(s);
    const uint8_t expected[kVideoStatusSize] = {
            2,    0,    0,    0,    0x80, 0x07, 0x40, 0x04, 0x80, 0x07, 0x38, 0x04,
            0,    0,    0xc0, 0x41, 0xa0, 0x25, 0x26, 0x00, 0xe8, 0x03, 0,    0,
            3,    0,    0,    0,    7,    0,    0,    0,    0,    0,    0xf0, 0x42,
            0xb8, 0x0b, 0,    0,    0x7b, 0x00, 3,    0,    0,    0,    0,    0};
    EXPECT_EQ(0, memcmp(expected, page.data() + kVideoStatusOffset, sizeof(expected)));
    EXPECT_EQ(0, page[kUpscaleStatusOffset]);
}

TEST(Video, DecodesMotionUpscaleAndSmoothness) {
    Sections sections;
    ASSERT_TRUE(DecodeChannel(Channel({{kSectionVideo, 0, {1, 120, 1, 255, 0xd4, 0x0a, 0x00, 0x05}}}).data(),
                              &sections));
    VideoSettings s;
    ASSERT_TRUE(DecodeVideo(sections.video, sections.videoLength, &s));
    EXPECT_EQ(120, s.motionHz);
    EXPECT_TRUE(s.upscale);
    EXPECT_FALSE(s.afme);
    EXPECT_FLOAT_EQ(1.0f, s.smoothness);
    EXPECT_TRUE(s.on());
    EXPECT_EQ(2772, s.displayLong);
    EXPECT_EQ(1280, s.displayShort);
    const uint8_t off[] = {1, 0, 0, 128, 0, 0, 0, 0};
    ASSERT_TRUE(DecodeVideo(off, sizeof(off), &s));
    EXPECT_FALSE(s.on());
    EXPECT_NEAR(0.5f, s.smoothness, 0.01f);
}

TEST(Video, RejectsAnythingElse) {
    for (const std::vector<uint8_t>& d :
         std::vector<std::vector<uint8_t>>{{1, 120, 0, 255}, {1, 120, 0, 255, 0, 0, 0, 0, 0},
                                           {2, 120, 0, 255, 0, 0, 0, 0}, {1, 47, 0, 255, 0, 0, 0, 0},
                                           {1, 120, 4, 255, 0, 0, 0, 0}}) {
        VideoSettings s;
        EXPECT_FALSE(DecodeVideo(d.data(), d.size(), &s));
        EXPECT_FALSE(s.on());
    }
}

TEST(Video, ItsSectionIsApartFromTheGamesKelir) {
    Sections sections;
    const std::vector<uint8_t> stack(kKelirHeaderSize, 1);
    ASSERT_TRUE(DecodeChannel(Channel({{kSectionVideoKelir, 0, stack}}).data(), &sections));
    EXPECT_EQ(nullptr, sections.kelir);
    ASSERT_NE(nullptr, sections.videoKelir);
    EXPECT_EQ(stack.size(), sections.videoKelirLength);
    EXPECT_GE(kSectionVideoKelir, kFirstVideoSection);
    EXPECT_LT(kSectionTexture, kFirstVideoSection);
}

TEST(Status, PresentsCountWithoutTouchingTheBlocks) {
    std::vector<uint8_t> page(kStatusSize, 0xaa);
    StatusPage writer;
    writer.countPresent();  // before attach: nowhere to count, no crash
    writer.attach(page.data());
    EXPECT_EQ(0u, ReadU32(page.data() + kPresentCountOffset));
    const std::vector<uint8_t> before(page);
    for (int i = 0; i < 1000; i++) writer.countPresent();
    EXPECT_EQ(1000u, ReadU32(page.data() + kPresentCountOffset));
    for (size_t i = 0; i < kStatusSize; i++) {
        if (i >= kPresentCountOffset && i < kPresentCountOffset + 4) continue;
        ASSERT_EQ(before[i], page[i]) << "byte " << i;
    }
}

TEST(Status, PresentsCountFromManyThreads) {
    std::vector<uint8_t> page(kStatusSize, 0);
    StatusPage writer;
    writer.attach(page.data());
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&] {
            for (int i = 0; i < 10000; i++) writer.countPresent();
        });
    }
    for (std::thread& t : threads) t.join();
    EXPECT_EQ(40000u, ReadU32(page.data() + kPresentCountOffset));
}

TEST(Status, PresentCountSitsAfterTheBlocks) {
    EXPECT_GE(kPresentCountOffset, kVideoStatusOffset + kVideoStatusSize);
    EXPECT_EQ(0u, kPresentCountOffset % 4);
    EXPECT_EQ(352u, kPresentCountOffset);  // lepasd reads it at this offset
}

TEST(Lepas, ItsSectionArmsTheLoadersNotTheVideoPath) {
    EXPECT_GT(kSectionLepas, kLastVideoSection);
    EXPECT_LE(kSectionVideo, kLastVideoSection);
    Sections sections;
    const std::vector<uint8_t> lepas = {kLepasMajor, 120, 0, 0};
    // The game skips the section: its decode sees nothing it acts on.
    ASSERT_TRUE(DecodeChannel(Channel({{kSectionLepas, 0, lepas}}).data(), &sections));
    EXPECT_EQ(nullptr, sections.kelir);
    EXPECT_EQ(nullptr, sections.videoKelir);
    // So the engine stays idle but for the present count: Lapis records nothing.
    EXPECT_TRUE(sections.none());
    ASSERT_TRUE(DecodeChannel(
            Channel({{kSectionLepas, 0, lepas}, {kSectionLapis, 0, {1, 1, 0, 0}}}).data(),
            &sections));
    EXPECT_FALSE(sections.none());
}

TEST(Status, DeclinedCarriesItsReason) {
    LuwesStatus s;
    s.state = LuwesState::kDeclined;
    s.reason = LuwesReason::kFormat;
    uint8_t block[kLuwesStatusSize];
    EncodeLuwesStatus(s, block);
    EXPECT_EQ(3, block[0]);
    EXPECT_EQ(2, block[1]);
}

TEST(Status, WritesBeforeAttachAreDropped) {
    StatusPage writer;
    writer.writeLuwes(LuwesStatus{});  // no page yet: nothing to write, nothing to crash
}

TEST(Arming, SystemAndIsolatedUidsNeverLook) {
    EXPECT_FALSE(Armed(0));
    EXPECT_FALSE(Armed(1000));
    EXPECT_FALSE(Armed(9999));
    EXPECT_FALSE(Armed(90000));      // isolated
    EXPECT_FALSE(Armed(20000));      // SDK sandbox
    EXPECT_FALSE(Armed(1010000));    // system uid of user 10
    EXPECT_FALSE(Armed(10123));      // an app, but no marker on this host
}

}  // namespace
}  // namespace framefx
