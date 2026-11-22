/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

// Compiled programs or pipelines by signature: a few, used least recently goes first.
namespace framefx::kelir {

template <typename Value, size_t N = 8>
class Lru {
public:
    // The value for `key`, nullptr when it is not held. A hit makes it the most recent.
    Value* find(uint64_t key) {
        for (Slot& s : slots_) {
            if (s.used && s.key == key) {
                s.stamp = ++clock_;
                return &s.value;
            }
        }
        return nullptr;
    }

    // Holds `value` for `key`, handing back the one it evicted (a default Value when none), so
    // the caller frees it on its own terms.
    Value insert(uint64_t key, Value value) {
        Slot* target = &slots_[0];
        for (Slot& s : slots_) {
            if (!s.used) {
                target = &s;
                break;
            }
            if (s.stamp < target->stamp) target = &s;
        }
        Value evicted = target->used ? std::move(target->value) : Value();
        *target = {key, std::move(value), ++clock_, true};
        return evicted;
    }

    size_t size() const {
        size_t n = 0;
        for (const Slot& s : slots_) n += s.used;
        return n;
    }

    template <typename F>
    void forEach(F&& f) {
        for (Slot& s : slots_) {
            if (s.used) f(s.value);
        }
    }

    void clear() {
        for (Slot& s : slots_) s = Slot();
    }

private:
    struct Slot {
        uint64_t key = 0;
        Value value = Value();
        uint64_t stamp = 0;
        bool used = false;
    };

    Slot slots_[N];
    uint64_t clock_ = 0;
};

}  // namespace framefx::kelir
