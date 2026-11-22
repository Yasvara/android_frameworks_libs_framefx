/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/channel.h>

#include <cstddef>
#include <cstdint>

#include "kelir/params.h"

// The Kelir section of a game's channel (framefx/channel.h, "Kelir section v1").
namespace framefx::kelir {

struct Node {
    uint8_t kind = 0;
    bool enabled = false;
    uint16_t intensity = 0;  // unorm
    int16_t q[kMaxParams] = {};

    float strength() const { return intensity / 65535.0f; }
    float value(int i) const { return Decode(Info(kind).params[i], q[i]); }
    bool operator==(const Node& o) const;
};

struct Stack {
    uint8_t minor = 0;
    bool peek = false;
    bool split = false;
    bool inspect = false;
    uint16_t splitX = 0;  // unorm, display space
    uint32_t revision = 0;
    int count = 0;
    Node nodes[kMaxKelirNodes];

    bool operator==(const Stack& o) const;
};

// Strict: anything but a well-formed v1 section is rejected whole and `out` is left alone.
KelirReject DecodeSection(const uint8_t* data, size_t length, Stack* out);

// The section for a stack, kKelirHeaderSize + kKelirNodeSize * count bytes into `out`.
size_t EncodeSection(const Stack& stack, uint8_t* out);

}  // namespace framefx::kelir
