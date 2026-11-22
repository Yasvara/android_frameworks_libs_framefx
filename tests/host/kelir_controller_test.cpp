/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "kelir/controller.h"
#include "src/channel.h"
#include "tests/host/kelir_test_util.h"

namespace framefx::kelir {
namespace {

using test::Bytes;
using test::MakeNode;
using test::MakeStack;
using test::Set;

std::vector<uint8_t> Bad(uint32_t revision) {
    auto b = Bytes(MakeStack({MakeNode(kSepia)}, revision));
    b[0] = 2;  // a major this library does not speak
    return b;
}

// A channel holding one Kelir section, as the broker lays it out.
std::vector<uint8_t> Channel(const std::vector<uint8_t>& kelir, uint32_t sequence) {
    std::vector<uint8_t> c(kChannelSize, 0);
    const uint32_t magic = kChannelMagic;
    memcpy(c.data(), &magic, 4);
    c[4] = kChannelVersion & 0xff;
    c[5] = kChannelVersion >> 8;
    c[6] = kChannelSize & 0xff;
    c[7] = kChannelSize >> 8;
    memcpy(c.data() + kSequenceOffset, &sequence, 4);
    c[12] = 1;
    uint8_t* t = c.data() + kSectionTableOffset;
    t[0] = kSectionKelir;
    t[2] = kSectionDataOffset & 0xff;
    t[3] = kSectionDataOffset >> 8;
    t[4] = static_cast<uint8_t>(kelir.size());
    t[5] = static_cast<uint8_t>(kelir.size() >> 8);
    memcpy(c.data() + kSectionDataOffset, kelir.data(), kelir.size());
    return c;
}

bool Feed(Controller& c, const std::vector<uint8_t>& section) {
    return c.update(section.data(), section.size());
}

// ---- KelirController ------------------------------------------------------------------------

TEST(KelirController, StartsOff) {
    Controller c;
    ASSERT_NE(nullptr, c.plan());
    EXPECT_TRUE(c.plan()->empty());
}

TEST(KelirController, InitRejectMeansOff) {
    Controller c;
    EXPECT_FALSE(Feed(c, Bad(3)));
    EXPECT_TRUE(c.plan()->empty());
    EXPECT_EQ(3u, c.rejectedRevision());
    EXPECT_EQ(KelirReject::kMajor, c.rejectReason());
    EXPECT_EQ(0u, c.appliedRevision());
}

TEST(KelirController, KeepsLastGoodOnReject) {
    Controller c;
    ASSERT_TRUE(Feed(c, Bytes(MakeStack({MakeNode(kSepia)}, 1))));
    const auto good = c.plan();
    EXPECT_FALSE(Feed(c, Bad(2)));
    EXPECT_EQ(good, c.plan());
    EXPECT_EQ(1u, c.appliedRevision());
    EXPECT_EQ(2u, c.rejectedRevision());
}

TEST(KelirController, RemovedSectionTurnsOff) {
    Controller c;
    ASSERT_TRUE(Feed(c, Bytes(MakeStack({MakeNode(kSepia)}))));
    EXPECT_TRUE(c.update(nullptr, 0));
    EXPECT_TRUE(c.plan()->empty());
    EXPECT_FALSE(c.update(nullptr, 0));
}

TEST(KelirController, UnchangedStackKeepsThePlan) {
    Controller c;
    const auto bytes = Bytes(MakeStack({MakeNode(kSepia)}, 4));
    ASSERT_TRUE(Feed(c, bytes));
    const auto plan = c.plan();
    EXPECT_FALSE(Feed(c, bytes));
    EXPECT_EQ(plan, c.plan());
}

// A drag publishes faster than anyone reads: whatever is read is compiled, and the newest wins.
TEST(KelirController, CoalescesToNewestRevision) {
    Controller c;
    Node n = MakeNode(kBrightnessContrast);
    std::vector<std::vector<uint8_t>> published;
    for (uint32_t r = 1; r <= 50; r++) {
        Set(n, p::kExposure, 0.02f * r);
        published.push_back(Bytes(MakeStack({n}, r)));
    }
    // The watcher wakes for a few of them only.
    for (size_t i : {4u, 17u, 49u}) EXPECT_TRUE(Feed(c, published[i]));
    EXPECT_EQ(50u, c.appliedRevision());
    EXPECT_EQ(50u, c.plan()->revision);
    EXPECT_NEAR(1.0f, c.plan()->colour[0].v[p::kExposure], 1e-3);
}

TEST(KelirController, StatusCarriesRejectReason) {
    Controller c;
    Feed(c, Bytes(MakeStack({MakeNode(kSepia)}, 0x01020304)));
    Feed(c, Bad(0x0a0b0c0d));
    KelirStatus s;
    s.state = KelirState::kRunning;
    s.reject = c.rejectReason();
    s.appliedRevision = c.appliedRevision();
    s.rejectedRevision = c.rejectedRevision();
    uint8_t block[kKelirStatusSize];
    EncodeKelirStatus(s, block);
    EXPECT_EQ(static_cast<uint8_t>(KelirReject::kMajor), block[10]);
    EXPECT_EQ(0x01020304u, ReadU32(block + 12));
    EXPECT_EQ(0x0a0b0c0du, ReadU32(block + 16));
}

TEST(KelirStatus, BlockIsByteExact) {
    KelirStatus s;
    s.state = KelirState::kDeclined;
    s.reason = KelirReason::kHdr;
    s.api = Api::kVulkan;
    s.format = FormatClass::kUnorm10;
    s.width = 1968;
    s.height = 908;
    s.rotation = 3;
    s.path = KelirPath::kVulkan;
    s.gpuNs = 412000;
    s.signature = 0x1122334455667788ull;
    s.swatch[0] = 9;
    s.curves[7][16] = 255;
    uint8_t b[kKelirStatusSize];
    EncodeKelirStatus(s, b);
    const uint8_t head[] = {3, 1, 2, 2, 0xb0, 0x07, 0x8c, 0x03, 3, 3, 0, 0};
    EXPECT_EQ(0, memcmp(head, b, sizeof(head)));
    EXPECT_EQ(412000u, ReadU32(b + 20));
    EXPECT_EQ(0x55667788u, ReadU32(b + 24));
    EXPECT_EQ(0x11223344u, ReadU32(b + 28));
    EXPECT_EQ(9, b[32]);
    EXPECT_EQ(255, b[56 + 7 * kKelirCurvePoints + 16]);
    for (size_t i = 56 + kMaxKelirNodes * kKelirCurvePoints; i < kKelirStatusSize; i++) {
        EXPECT_EQ(0, b[i]) << i;
    }
}

TEST(KelirStatus, LayoutFitsThePage) {
    EXPECT_LE(kLuwesStatusOffset + kLuwesStatusSize, kKelirStatusOffset);
    EXPECT_LE(static_cast<size_t>(56 + kMaxKelirNodes * kKelirCurvePoints), kKelirStatusSize);
    EXPECT_LE(kKelirStatusOffset + kKelirStatusSize, kStatusSize);
}

TEST(KelirStatus, WritesLandInTheKelirBlockUnderTheSequence) {
    std::vector<uint8_t> page(kStatusSize, 0xee);
    StatusPage status;
    status.attach(page.data());
    KelirStatus s;
    s.state = KelirState::kRunning;
    s.width = 640;
    status.writeKelir(s);
    EXPECT_EQ(4u, ReadU32(page.data() + kSequenceOffset));
    EXPECT_EQ(static_cast<uint8_t>(KelirState::kRunning), page[kKelirStatusOffset]);
    EXPECT_EQ(640, ReadU16(page.data() + kKelirStatusOffset + 4));
    EXPECT_EQ(0, page[kLuwesStatusOffset]);  // untouched by Kelir
}

// ---- KelirSeqlock ---------------------------------------------------------------------------

TEST(KelirSeqlock, RetriesOnOdd) {
    auto c = Channel(Bytes(MakeStack({MakeNode(kSepia)})), 7);
    std::vector<uint8_t> out(kChannelSize);
    EXPECT_FALSE(SnapshotChannel(c.data(), out.data()));
    c[kSequenceOffset] = 8;
    ASSERT_TRUE(SnapshotChannel(c.data(), out.data()));
    Sections s;
    ASSERT_TRUE(DecodeChannel(out.data(), &s));
    Controller controller;
    EXPECT_TRUE(controller.update(s.kelir, s.kelirLength));
    EXPECT_EQ(1, controller.plan()->colourCount);
}

TEST(KelirSeqlock, RetriesOnChanged) {
    // A sequence that moves between the two reads is never taken: SnapshotChannel compares the
    // word before and after its copy, which a channel_test exercises under a live writer. Here:
    // the snapshot carries the sequence it was taken at.
    auto c = Channel(Bytes(MakeStack({MakeNode(kSepia)})), 12);
    std::vector<uint8_t> out(kChannelSize);
    ASSERT_TRUE(SnapshotChannel(c.data(), out.data()));
    EXPECT_EQ(12u, ReadU32(out.data() + kSequenceOffset));
}

TEST(KelirSeqlock, KeepsLastPlanAfter4Retries) {
    Controller controller;
    auto good = Channel(Bytes(MakeStack({MakeNode(kSepia)})), 2);
    std::vector<uint8_t> out(kChannelSize);
    ASSERT_TRUE(SnapshotChannel(good.data(), out.data()));
    Sections s;
    ASSERT_TRUE(DecodeChannel(out.data(), &s));
    ASSERT_TRUE(controller.update(s.kelir, s.kelirLength));
    const auto plan = controller.plan();
    // A writer stuck mid-write: every attempt sees an odd sequence; nothing reaches the controller.
    auto stuck = Channel(Bytes(MakeStack({MakeNode(kVignette)})), 3);
    EXPECT_FALSE(SnapshotChannel(stuck.data(), out.data()));
    EXPECT_EQ(plan, controller.plan());
}

}  // namespace
}  // namespace framefx::kelir
