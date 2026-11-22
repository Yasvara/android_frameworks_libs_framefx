/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "kelir/section.h"

#include <algorithm>

#include "kelir/ffi.h"

namespace framefx::kelir {

bool Node::operator==(const Node& o) const {
    return kind == o.kind && enabled == o.enabled && intensity == o.intensity &&
            std::equal(q, q + kMaxParams, o.q);
}

bool Stack::operator==(const Stack& o) const {
    return minor == o.minor && peek == o.peek && split == o.split && inspect == o.inspect &&
            splitX == o.splitX && revision == o.revision && count == o.count &&
            std::equal(nodes, nodes + count, o.nodes);
}

KelirReject DecodeSection(const uint8_t* data, size_t length, Stack* out) {
    return static_cast<KelirReject>(framefx_kelir_decode_section(data, length, out));
}

size_t EncodeSection(const Stack& stack, uint8_t* out) {
    return framefx_kelir_encode_section(&stack, out);
}

}  // namespace framefx::kelir
