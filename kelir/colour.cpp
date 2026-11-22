/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "kelir/colour.h"

#include <algorithm>

#include "kelir/ffi.h"

namespace framefx::kelir {

bool Step::operator==(const Step& o) const {
    return kind == o.kind && intensity == o.intensity && std::equal(v, v + kMaxParams, o.v);
}

const float* LumaWeights(Primaries p) {
    return framefx_kelir_luma_weights(static_cast<uint8_t>(p));
}

float Decode(float encoded) {
    return framefx_kelir_srgb_decode(encoded);
}

float Encode(float linear) {
    return framefx_kelir_srgb_encode(linear);
}

Rgb ApplyColour(const Step& step, Rgb x, Primaries p) {
    return framefx_kelir_apply_colour(&step, x, static_cast<uint8_t>(p));
}

Rgb Blend(const Step& step, Rgb x, Primaries p) {
    return framefx_kelir_blend(&step, x, static_cast<uint8_t>(p));
}

bool IsIdentityColour(const Step& step) {
    return framefx_kelir_is_identity_colour(&step);
}

}  // namespace framefx::kelir
