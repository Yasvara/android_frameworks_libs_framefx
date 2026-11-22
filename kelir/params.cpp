/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "kelir/params.h"

#include "kelir/ffi.h"

namespace framefx::kelir {

const KindInfo& Info(int kind) {
    return *framefx_kelir_info(kind);
}

bool CodeValid(const Param& p, int16_t q) {
    return framefx_kelir_code_valid(&p, q);
}

float Decode(const Param& p, int16_t q) {
    return framefx_kelir_param_decode(&p, q);
}

int16_t Encode(const Param& p, float value) {
    return framefx_kelir_param_encode(&p, value);
}

}  // namespace framefx::kelir
