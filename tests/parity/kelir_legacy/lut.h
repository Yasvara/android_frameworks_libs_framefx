/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/channel.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "tests/parity/kelir_legacy/colour.h"

// The colour band as one 3D LUT: 33^3 RGBA16F, red fastest, holding the result unclamped.
//
// The grid is denser toward black: it is indexed by each encoded channel to the power
// kLutShaper, which keeps curves that rise steeply out of black (lifted shadows, gamma) within an
// LSB. The result is stored encoded, or linear when the band ends in a linear-light matrix
// (Colorblind, Night Mode): those cross zero light for wide colours, where the encoding is
// steepest, and are exact in linear storage. The pass encodes after the tap.
namespace framefx::kelir_legacy {

inline constexpr int kLutSize = 33;
inline constexpr size_t kLutTexels = kLutSize * kLutSize * kLutSize;
inline constexpr float kLutShaper = 2.0f / 3.0f;

struct Lut {
    std::vector<uint16_t> texels;  // kLutTexels x RGBA half floats
    bool linear = false;           // texels hold linear light
};

// Bakes `count` colour steps in order for targets with primaries `p`.
void BakeLut(const Step* steps, int count, Primaries p, Lut* out);

// One colour band's LUTs: sRGB baked at once, Display P3 on first use, since few games present
// in P3.
class Luts {
public:
    Luts(const Step* steps, int count);

    const Lut& get(Primaries p) const;
    bool linear() const { return srgb_.linear; }

private:
    Step steps_[kMaxKelirNodes];
    int count_;
    Lut srgb_;
    mutable std::once_flag p3Once_;
    mutable Lut p3_;
};

// The exact result of the steps for one colour, as the LUT approximates it.
Rgb EvaluateColour(const Step* steps, int count, Primaries p, Rgb x);

// What the pass reads from the LUT for x: the shaper, the hardware's trilinear filter and, for a
// linear LUT, the encoding after it.
Rgb SampleLut(const Lut& lut, Rgb x);

uint16_t ToHalf(float v);
float FromHalf(uint16_t h);

// Status readouts while the editor is open: the swatch of reference colours after the whole
// band, and each colour step's own luma curve over a grey ramp.
void Swatch(const Step* steps, int count, Primaries p, uint8_t out[kKelirSwatchColours * 3]);
void Curve(const Step& step, Primaries p, uint8_t out[kKelirCurvePoints]);

}  // namespace framefx::kelir_legacy
