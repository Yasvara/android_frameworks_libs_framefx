/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "kelir/plan.h"

#include "kelir/ffi.h"

namespace framefx::kelir {

bool Plan::has(uint8_t kind) const {
    return framefx_kelir_plan_has(this, kind);
}

const Step* Plan::sceneStep(uint8_t kind) const {
    const int i = framefx_kelir_scene_step(this, kind);
    return i < 0 ? nullptr : &scene[i];
}

bool Plan::needsPyramid() const {
    return framefx_kelir_needs_pyramid(this);
}

bool Plan::needsSource() const {
    return framefx_kelir_needs_source(this);
}

std::shared_ptr<const Plan> Compile(const Stack& stack, const Plan* previous) {
    auto plan = std::make_shared<Plan>();
    PlanCore core;
    framefx_kelir_compile(&stack, &core);
    static_cast<PlanCore&>(*plan) = core;
    if (plan->colourCount > 0) {
        plan->luts = previous != nullptr && previous->luts != nullptr &&
                        framefx_kelir_same_colour_band(plan.get(), previous)
                ? previous->luts
                : std::make_shared<const Luts>(plan->colour, plan->colourCount);
    }
    return plan;
}

uint64_t Signature(const Plan& plan, const Target& target) {
    return framefx_kelir_signature(&plan, &target);
}

bool Program::has(int kind) const {
    for (int d : detail) {
        if (d == kind) return true;
    }
    for (int s : screen) {
        if (s == kind) return true;
    }
    return false;
}

Program Unpack(uint64_t signature) {
    Program p;
    framefx_kelir_unpack(signature, &p);
    return p;
}

void PackPlan(const Plan& plan, float out[kPlanVec4s * 4]) {
    framefx_kelir_pack_plan(&plan, out);
}

void PackScene(const Plan& plan, float out[kSceneVec4s * 4]) {
    framefx_kelir_pack_scene(&plan, out);
}

bool PackSceneLook(const Plan& plan, float out[kLookVec4s * 4]) {
    return framefx_kelir_pack_scene_look(&plan, out);
}

bool PackSceneRays(const Plan& plan, float out[kRaysVec4s * 4]) {
    return framefx_kelir_pack_scene_rays(&plan, out);
}

void PackFrame(const Plan& plan, const FrameInfo& frame, float out[kFrameVec4s * 4]) {
    framefx_kelir_pack_frame(&plan, &frame, out);
}

Stages Schedule(const Plan* plan, bool generation, bool real) {
    Stages s;
    framefx_kelir_schedule(plan, generation, real, &s);
    return s;
}

}  // namespace framefx::kelir
