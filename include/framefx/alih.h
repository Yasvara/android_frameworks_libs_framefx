/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Memory between a game's render path and org.yasvara.alih, both regions from the broker.
//
// Ring: what is on screen, game to service. Records are u32 type, u32 payload bytes, payload,
// wrapping bytewise. head and tail only grow.
//
// Table: what to replace it with, service to game, under a sequence lock (odd while written):
//   4096  atlas, kAtlasWidth x kAtlasHeight
//   then  per text: u32 game's image, u32 keys, u32 parts, keys x u64, then per part: u32 source,
//         u32 glyphs, glyphs x Glyph. A part is drawn as the source's quad at that index is: an
//         outline and its fill are two parts of one text, each in its own colour.
namespace framefx::alih {

inline constexpr size_t kRingSize = 8u << 20;
inline constexpr size_t kRingDataOffset = 4096;
inline constexpr size_t kRingCapacity = kRingSize - kRingDataOffset;
inline constexpr uint32_t kAtlasWidth = 2048;
inline constexpr uint32_t kAtlasHeight = 1024;
inline constexpr size_t kTableAtlasOffset = 4096;
inline constexpr size_t kTableTextsOffset = kTableAtlasOffset + size_t{kAtlasWidth} * kAtlasHeight;
inline constexpr size_t kTableSize = kTableTextsOffset + (2u << 20);
inline constexpr uint32_t kMaxGlyphs = 4096;

enum RecordType : uint32_t {
    kRegion = 1,  // RegionHead, width x height bytes: part of a single-channel image, as uploaded
    kSample = 2,  // u32 draws, then per draw DrawHead and quads x Quad
};

struct RegionHead {
    uint32_t image;
    uint16_t imageWidth;
    uint16_t imageHeight;
    uint16_t x;
    uint16_t y;
    uint16_t width;
    uint16_t height;
};

struct DrawHead {
    uint32_t image;
    uint16_t imageWidth;
    uint16_t imageHeight;
    uint32_t quads;
};

struct Quad {
    uint64_t key;  // its rectangle in the image: Key()
    float xMin, yMin, xMax, yMax;  // in the game's units
    float uvLow[2];    // at xMin, yMin
    float uvRight[2];  // at xMax, yMin
    float uvUp[2];     // at xMin, yMax
};

struct Glyph {
    float x0, y0, x1, y1;  // from the low corner of the part's source quad
    float u0, v0, u1, v1;  // at those corners
};

constexpr uint64_t Key(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
    return uint64_t{x0} | uint64_t{y0} << 16 | uint64_t{x1} << 32 | uint64_t{y1} << 48;
}

struct RingHeader {
    std::atomic<uint32_t> wake;  // futex: changes with every commit
    std::atomic<uint32_t> readerEpoch;  // a new reader knows nothing: the game sends its images again
    alignas(64) std::atomic<uint64_t> head;
    alignas(64) std::atomic<uint64_t> tail;
};
static_assert(sizeof(RingHeader) <= kRingDataOffset);

// The atlas only grows downward, until it is emptied and its generation changes: a row above
// stableRows never changes again, and nothing is below usedRows.
struct TableHeader {
    std::atomic<uint32_t> sequence;
    uint32_t texts;
    uint32_t bytes;
    uint32_t generation;
    uint32_t stableRows;
    uint32_t usedRows;
};
static_assert(sizeof(TableHeader) <= kTableAtlasOffset);

class RingWriter {
  public:
    explicit RingWriter(uint8_t* memory)
        : header_(reinterpret_cast<RingHeader*>(memory)), data_(memory + kRingDataOffset) {}

    // A whole record, or nothing when it does not fit. The chunks are written in order.
    struct Chunk {
        const void* bytes;
        size_t size;
    };
    template <size_t N>
    bool write(RecordType type, const Chunk (&chunks)[N]) {
        size_t payload = 0;
        for (const Chunk& c : chunks) payload += c.size;
        const uint64_t head = header_->head.load(std::memory_order_relaxed);
        const uint64_t tail = header_->tail.load(std::memory_order_acquire);
        if (payload + 8 > kRingCapacity - (head - tail)) return false;
        const uint32_t record[2] = {type, static_cast<uint32_t>(payload)};
        uint64_t at = head;
        put(at, record, sizeof(record));
        at += sizeof(record);
        for (const Chunk& c : chunks) {
            put(at, c.bytes, c.size);
            at += c.size;
        }
        header_->head.store(at, std::memory_order_release);
        header_->wake.fetch_add(1, std::memory_order_release);
        syscall(SYS_futex, &header_->wake, FUTEX_WAKE, 1, nullptr, nullptr, 0);
        return true;
    }

    uint32_t readerEpoch() const { return header_->readerEpoch.load(std::memory_order_acquire); }

  private:
    void put(uint64_t at, const void* bytes, size_t size) {
        const size_t offset = at % kRingCapacity;
        const size_t first = size < kRingCapacity - offset ? size : kRingCapacity - offset;
        memcpy(data_ + offset, bytes, first);
        memcpy(data_, static_cast<const uint8_t*>(bytes) + first, size - first);
    }

    RingHeader* header_;
    uint8_t* data_;
};

class RingReader {
  public:
    explicit RingReader(uint8_t* memory)
        : header_(reinterpret_cast<RingHeader*>(memory)), data_(memory + kRingDataOffset) {}

    void announce() { header_->readerEpoch.fetch_add(1, std::memory_order_release); }

    // The next record's type and payload size; false when there is none, or the ring is corrupt.
    bool peek(uint32_t* type, uint32_t* size) const {
        const uint64_t tail = header_->tail.load(std::memory_order_relaxed);
        const uint64_t head = header_->head.load(std::memory_order_acquire);
        if (head - tail < 8) return false;
        uint32_t record[2];
        get(tail, record, sizeof(record));
        if (record[1] > head - tail - 8) return false;
        *type = record[0];
        *size = record[1];
        return true;
    }

    // The payload peek() announced, then past it.
    void take(void* out, uint32_t size) {
        const uint64_t tail = header_->tail.load(std::memory_order_relaxed);
        get(tail + 8, out, size);
        header_->tail.store(tail + 8 + size, std::memory_order_release);
    }

    // Blocks until the ring may have changed, or the time runs out.
    void wait(uint32_t seen, int64_t nanoseconds) const {
        const timespec timeout = {static_cast<time_t>(nanoseconds / 1000000000),
                                  static_cast<long>(nanoseconds % 1000000000)};
        syscall(SYS_futex, &header_->wake, FUTEX_WAIT, seen, &timeout, nullptr, 0);
    }

    uint32_t wakeWord() const { return header_->wake.load(std::memory_order_acquire); }

  private:
    void get(uint64_t at, void* out, size_t size) const {
        const size_t offset = at % kRingCapacity;
        const size_t first = size < kRingCapacity - offset ? size : kRingCapacity - offset;
        memcpy(out, data_ + offset, first);
        memcpy(static_cast<uint8_t*>(out) + first, data_, size - first);
    }

    RingHeader* header_;
    uint8_t* data_;
};

}  // namespace framefx::alih
