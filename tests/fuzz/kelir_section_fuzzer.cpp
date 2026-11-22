/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <cstdlib>
#include <cstring>

#include "kelir/plan.h"
#include "kelir/section.h"

namespace framefx::kelir {
namespace {

void check(bool ok) {
    if (!ok) abort();
}

}  // namespace
}  // namespace framefx::kelir

// Any bytes: decoding never crashes; an accepted section re-encodes to the same bytes and compiles
// to a plan within the limits whose signature takes apart to the same kinds.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    using namespace framefx;
    using namespace framefx::kelir;
    Stack stack;
    if (DecodeSection(data, size, &stack) != KelirReject::kNone) return 0;
    uint8_t again[kMaxKelirSectionSize];
    const size_t n = EncodeSection(stack, again);
    check(n == size && memcmp(again, data, size) == 0);

    const auto plan = Compile(stack, nullptr);
    check(plan->detailCount <= kMaxDetail && plan->screenCount <= kMaxScreen &&
          plan->colourCount <= kMaxKelirNodes);
    check(plan->detailCount + plan->colourCount + plan->screenCount <= stack.count);
    check((plan->colourCount > 0) == (plan->luts != nullptr));
    int styles = 0;
    for (int i = 0; i < plan->detailCount; i++) styles += Info(plan->detail[i].kind).style;
    check(styles <= 1);
    const Program program = Unpack(Signature(*plan, Target()));
    for (int i = 0; i < kMaxDetail; i++) {
        check(program.detail[i] == (i < plan->detailCount ? plan->detail[i].kind : -1));
    }
    for (int i = 0; i < kMaxScreen; i++) {
        check(program.screen[i] == (i < plan->screenCount ? plan->screen[i].kind : -1));
    }
    return 0;
}
