/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace framefx {

// Read lock-free at every bind; written under a lock as pipelines come and go. A pipeline only
// gets the dynamic state once it has a slot here, so every bind of one finds it.
class PipelineSet {
  public:
    static constexpr size_t kCapacity = 1 << 16;
    static constexpr uint64_t kEmpty = 0;
    static constexpr uint64_t kGone = 1;

    bool reserve(size_t n) {
        std::lock_guard lock(lock_);
        if (used_ + reserved_ + n > kCapacity * 3 / 4) return false;
        reserved_ += n;
        return true;
    }
    void unreserve(size_t n) {
        std::lock_guard lock(lock_);
        reserved_ -= std::min(reserved_, n);
    }
    // After a reserve() of at least one.
    void insert(uint64_t handle) {
        std::lock_guard lock(lock_);
        reserved_ -= std::min<size_t>(reserved_, 1);
        for (size_t i = slot(handle);; i = (i + 1) & (kCapacity - 1)) {
            const uint64_t v = table_[i].load(std::memory_order_relaxed);
            if (v == handle) return;
            if (v == kEmpty || v == kGone) {
                if (v == kEmpty && used_ >= kCapacity * 7 / 8) return;  // full: never probe forever
                table_[i].store(handle, std::memory_order_release);
                if (v == kEmpty) used_++;
                return;
            }
        }
    }
    void erase(uint64_t handle) {
        std::lock_guard lock(lock_);
        for (size_t i = slot(handle);; i = (i + 1) & (kCapacity - 1)) {
            const uint64_t v = table_[i].load(std::memory_order_relaxed);
            if (v == kEmpty) return;
            if (v == handle) {
                table_[i].store(kGone, std::memory_order_release);
                return;
            }
        }
    }
    // Forgets every pipeline: binds racing it may still find one for a moment.
    void clear() {
        std::lock_guard lock(lock_);
        for (auto& v : table_) v.store(kEmpty, std::memory_order_relaxed);
        used_ = 0;
    }
    bool contains(uint64_t handle) const {
        if (handle <= kGone) return false;
        for (size_t i = slot(handle);; i = (i + 1) & (kCapacity - 1)) {
            const uint64_t v = table_[i].load(std::memory_order_acquire);
            if (v == handle) return true;
            if (v == kEmpty) return false;
        }
    }
    size_t size() const {
        std::lock_guard lock(lock_);
        return used_;
    }

  private:
    static size_t slot(uint64_t h) {
        return static_cast<size_t>((h >> 4) * 0x9E3779B97F4A7C15ull >> 48) & (kCapacity - 1);
    }
    mutable std::mutex lock_;
    size_t used_ = 0;  // slots ever filled, tombstones included
    size_t reserved_ = 0;
    std::atomic<uint64_t> table_[kCapacity] = {};
};

}  // namespace framefx
