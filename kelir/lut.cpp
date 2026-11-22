/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "kelir/lut.h"

#include <algorithm>

#include "kelir/ffi.h"

namespace framefx::kelir {

uint16_t ToHalf(float v) {
    return framefx_kelir_to_half(v);
}

float FromHalf(uint16_t h) {
    return framefx_kelir_from_half(h);
}

Rgb EvaluateColour(const Step* steps, int count, Primaries p, Rgb x) {
    return framefx_kelir_evaluate(steps, count, static_cast<uint8_t>(p), x);
}

void BakeLut(const Step* steps, int count, Primaries p, Lut* out) {
    out->texels.resize(kLutTexels * 4);
    out->linear = framefx_kelir_bake_lut(steps, count, static_cast<uint8_t>(p), out->texels.data());
}

Luts::Luts(const Step* steps, int count) : count_(std::min(count, kMaxKelirNodes)) {
    std::copy(steps, steps + count_, steps_);
    BakeLut(steps_, count_, Primaries::kSrgb, &srgb_);
}

const Lut& Luts::get(Primaries p) const {
    if (p == Primaries::kSrgb) return srgb_;
    std::call_once(p3Once_, [this] { BakeLut(steps_, count_, Primaries::kP3, &p3_); });
    return p3_;
}

Rgb SampleLut(const Lut& lut, Rgb x) {
    return framefx_kelir_sample_lut(lut.texels.data(), lut.linear, x);
}

void Swatch(const Step* steps, int count, Primaries p, uint8_t out[kKelirSwatchColours * 3]) {
    framefx_kelir_swatch(steps, count, static_cast<uint8_t>(p), out);
}

void Curve(const Step& step, Primaries p, uint8_t out[kKelirCurvePoints]) {
    framefx_kelir_curve(&step, static_cast<uint8_t>(p), out);
}

}  // namespace framefx::kelir
