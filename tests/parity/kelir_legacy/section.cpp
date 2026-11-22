/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "tests/parity/kelir_legacy/section.h"

#include <algorithm>

namespace framefx::kelir_legacy {
namespace {

constexpr uint8_t kHeaderFlags = kKelirPeek | kKelirSplit | kKelirInspect;
constexpr uint8_t kNodeEnabled = 1u << 0;
constexpr int kMaxDetail = 3;
constexpr int kMaxStyle = 1;
constexpr int kMaxScreen = 4;
constexpr int kMaxRetro = 1;

int16_t readS16(const uint8_t* p) {
    return static_cast<int16_t>(ReadU16(p));
}

void put16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}

void put32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

KelirReject decodeNode(const uint8_t* d, Node* out) {
    const uint8_t kind = d[0];
    if (!Known(kind)) return KelirReject::kKind;
    if ((d[1] & ~kNodeEnabled) != 0) return KelirReject::kFlags;
    const KindInfo& info = Info(kind);
    Node n;
    n.kind = kind;
    n.enabled = (d[1] & kNodeEnabled) != 0;
    n.intensity = ReadU16(d + 2);
    for (int i = 0; i < kMaxParams; i++) {
        n.q[i] = readS16(d + 4 + 2 * i);
        if (i >= info.paramCount) {
            if (n.q[i] != 0) return KelirReject::kUnusedParam;
        } else if (!CodeValid(info.params[i], n.q[i])) {
            return KelirReject::kParamRange;
        }
    }
    *out = n;
    return KelirReject::kNone;
}

// The cost envelope: bands, styles, Retro, and kinds that may not repeat.
KelirReject checkLimits(const Stack& s) {
    int detail = 0;
    int style = 0;
    int screen = 0;
    int retro = 0;
    uint32_t seen = 0;
    for (int i = 0; i < s.count; i++) {
        const uint8_t kind = s.nodes[i].kind;
        const KindInfo& info = Info(kind);
        if (info.band != Band::kColour) {
            if ((seen & (1u << kind)) != 0) return KelirReject::kRepeat;
            seen |= 1u << kind;
        }
        detail += info.band == Band::kDetail;
        style += info.style;
        screen += info.band == Band::kScreen;
        retro += kind == kRetro;
    }
    if (detail > kMaxDetail || style > kMaxStyle || screen > kMaxScreen || retro > kMaxRetro) {
        return KelirReject::kBandLimit;
    }
    return KelirReject::kNone;
}

}  // namespace

bool Node::operator==(const Node& o) const {
    return kind == o.kind && enabled == o.enabled && intensity == o.intensity &&
            std::equal(q, q + kMaxParams, o.q);
}

bool Stack::operator==(const Stack& o) const {
    return minor == o.minor && peek == o.peek && split == o.split && inspect == o.inspect &&
            splitX == o.splitX && revision == o.revision && count == o.count &&
            std::equal(nodes, nodes + count, o.nodes);
}

KelirReject DecodeSection(const uint8_t* d, size_t length, Stack* out) {
    if (d == nullptr || length < kKelirHeaderSize || length > kMaxKelirSectionSize) {
        return KelirReject::kSize;
    }
    if (d[0] != kKelirMajor) return KelirReject::kMajor;
    if ((d[2] & ~kHeaderFlags) != 0) return KelirReject::kFlags;
    const int count = d[3];
    if (count > kMaxKelirNodes) return KelirReject::kCount;
    if (length != kKelirHeaderSize + kKelirNodeSize * static_cast<size_t>(count)) {
        return KelirReject::kSize;
    }
    if (ReadU16(d + 6) != 0 || ReadU32(d + 12) != 0) return KelirReject::kReserved;
    Stack s;
    s.minor = d[1];
    s.peek = (d[2] & kKelirPeek) != 0;
    s.split = (d[2] & kKelirSplit) != 0;
    s.inspect = (d[2] & kKelirInspect) != 0;
    s.splitX = ReadU16(d + 4);
    s.revision = ReadU32(d + 8);
    s.count = count;
    for (int i = 0; i < count; i++) {
        const KelirReject r = decodeNode(d + kKelirHeaderSize + kKelirNodeSize * i, &s.nodes[i]);
        if (r != KelirReject::kNone) return r;
    }
    const KelirReject r = checkLimits(s);
    if (r != KelirReject::kNone) return r;
    *out = s;
    return KelirReject::kNone;
}

size_t EncodeSection(const Stack& s, uint8_t* out) {
    const size_t size = kKelirHeaderSize + kKelirNodeSize * static_cast<size_t>(s.count);
    std::fill(out, out + size, 0);
    out[0] = kKelirMajor;
    out[1] = s.minor;
    out[2] = static_cast<uint8_t>((s.peek ? kKelirPeek : 0) | (s.split ? kKelirSplit : 0) |
                                  (s.inspect ? kKelirInspect : 0));
    out[3] = static_cast<uint8_t>(s.count);
    put16(out + 4, s.splitX);
    put32(out + 8, s.revision);
    for (int i = 0; i < s.count; i++) {
        uint8_t* n = out + kKelirHeaderSize + kKelirNodeSize * i;
        const Node& node = s.nodes[i];
        n[0] = node.kind;
        n[1] = node.enabled ? kNodeEnabled : 0;
        put16(n + 2, node.intensity);
        for (int j = 0; j < kMaxParams; j++) put16(n + 4 + 2 * j, static_cast<uint16_t>(node.q[j]));
    }
    return size;
}

}  // namespace framefx::kelir_legacy
