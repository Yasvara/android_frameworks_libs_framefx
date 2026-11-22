/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <framefx/channel.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "kelir/plan.h"

// What one process runs of the Kelir sections Pentas publishes: the newest valid stack, compiled.
// A section rejected at start leaves Kelir off; one rejected later keeps the plan running, so a
// bad write never flashes the game unfiltered.
namespace framefx::kelir {

// The controller's state, kept by kelir/rust's controller.
struct ControllerCore {
    Stack stack;
    bool valid = false;
    uint32_t applied = 0;
    uint32_t rejected = 0;
    uint8_t reason = 0;  // KelirReject
};

class Controller {
public:
    // The channel's Kelir section, or nullptr when it has none (Kelir off). True when the plan
    // changed.
    bool update(const uint8_t* section, size_t length);

    // Never null: an empty plan when Kelir is off.
    const std::shared_ptr<const Plan>& plan() const { return plan_; }

    uint32_t appliedRevision() const { return core_.applied; }
    uint32_t rejectedRevision() const { return core_.rejected; }
    KelirReject rejectReason() const { return static_cast<KelirReject>(core_.reason); }

private:
    ControllerCore core_;
    std::shared_ptr<const Plan> plan_ = std::make_shared<Plan>();
};

}  // namespace framefx::kelir
