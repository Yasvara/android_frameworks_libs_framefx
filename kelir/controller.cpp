/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "kelir/controller.h"

#include "kelir/ffi.h"

namespace framefx::kelir {

bool Controller::update(const uint8_t* section, size_t length) {
    const ControllerUpdate u =
            framefx_kelir_controller_update(&core_, section, length, plan_->empty());
    if (u.action == kControllerOff) plan_ = std::make_shared<Plan>();
    if (u.action == kControllerCompile) plan_ = Compile(core_.stack, plan_.get());
    return u.changed;
}

}  // namespace framefx::kelir
