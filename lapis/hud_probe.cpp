/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "lapis/hud_probe.h"

#include <log/log.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace framefx::lapis::hud_probe {
namespace {

constexpr uint32_t kGridLong = 64;
constexpr uint32_t kGridShortMin = 8;
// The first blit shrinks by about this much, so the second averages what it reads.
constexpr uint32_t kFirstShrink = 8;
// Copies in flight: a grid is read a frame or more after it was recorded.
constexpr uint32_t kSlots = 4;
constexpr uint32_t kCameras = 8;
constexpr VkDeviceSize kHeader = 16;

struct Fn {
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
    PFN_vkCmdFillBuffer CmdFillBuffer;

    bool load(VkDevice device, PFN_vkGetDeviceProcAddr get) {
        bool ok = true;
        const auto fetch = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(get(device, name));
            ok = ok && fn != nullptr;
        };
        fetch(CreateImage, "vkCreateImage");
        fetch(DestroyImage, "vkDestroyImage");
        fetch(GetImageMemoryRequirements, "vkGetImageMemoryRequirements");
        fetch(BindImageMemory, "vkBindImageMemory");
        fetch(CreateBuffer, "vkCreateBuffer");
        fetch(DestroyBuffer, "vkDestroyBuffer");
        fetch(GetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
        fetch(BindBufferMemory, "vkBindBufferMemory");
        fetch(AllocateMemory, "vkAllocateMemory");
        fetch(FreeMemory, "vkFreeMemory");
        fetch(MapMemory, "vkMapMemory");
        fetch(CmdPipelineBarrier, "vkCmdPipelineBarrier");
        fetch(CmdBlitImage, "vkCmdBlitImage");
        fetch(CmdCopyImageToBuffer, "vkCmdCopyImageToBuffer");
        fetch(CmdFillBuffer, "vkCmdFillBuffer");
        return ok;
    }
};

struct DeviceFns {
    Fn fn = {};
    VkPhysicalDeviceMemoryProperties memory = {};
};

struct Resources {
    VkDevice device = VK_NULL_HANDLE;
    VkImage mid = VK_NULL_HANDLE;
    VkImage grid = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory[3] = {};
    const uint8_t* mapped = nullptr;
    uint32_t midWidth = 0, midHeight = 0;
    uint32_t gridWidth = 0, gridHeight = 0;
    VkDeviceSize slotBytes = 0;
};

std::mutex gLock;
std::unordered_map<VkDevice, DeviceFns> gDevices;
Resources gNow;
std::vector<Resources> gSpent;  // replaced on resize; a game's command buffer may still hold them
// What Arm set, for the pass-end check without the lock.
std::atomic<bool> gArmed{false};
std::atomic<uint64_t> gPass{0};
std::atomic<uint64_t> gShape{0};
VkImageLayout gFinal = VK_IMAGE_LAYOUT_UNDEFINED;
uint32_t gWidth = 0, gHeight = 0;
std::atomic<uint32_t> gFrame{1};
std::atomic<uint32_t> gRecordedFor{0};
uint32_t gSlotSeq[kSlots] = {};
float gCameras[kCameras][16] = {};
uint32_t gCameraSeq[kCameras] = {};
bool gCameraValid[kCameras] = {};
uint32_t gTaken = 0;
std::vector<uint8_t> gLuma;

uint64_t shapeOf(VkFormat format, uint32_t width, uint32_t height) {
    return static_cast<uint64_t>(format) << 32 | (width & 0xffff) << 16 | (height & 0xffff);
}

// Formats every implementation blits from.
bool blittable(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R5G6B5_UNORM_PACK16:
            return true;
        default:
            return false;
    }
}

uint32_t typeFor(const VkPhysicalDeviceMemoryProperties& m, uint32_t bits, VkMemoryPropertyFlags want,
                 VkMemoryPropertyFlags avoid) {
    for (uint32_t t = 0; t < m.memoryTypeCount; t++) {
        const VkMemoryPropertyFlags flags = m.memoryTypes[t].propertyFlags;
        if ((bits & (1u << t)) && (flags & want) == want && !(flags & avoid)) return t;
    }
    return UINT32_MAX;
}

void destroy(const Fn& f, Resources* r) {
    if (r->device == VK_NULL_HANDLE) return;
    if (r->mid != VK_NULL_HANDLE) f.DestroyImage(r->device, r->mid, nullptr);
    if (r->grid != VK_NULL_HANDLE) f.DestroyImage(r->device, r->grid, nullptr);
    if (r->buffer != VK_NULL_HANDLE) f.DestroyBuffer(r->device, r->buffer, nullptr);
    for (VkDeviceMemory m : r->memory) {
        if (m != VK_NULL_HANDLE) f.FreeMemory(r->device, m, nullptr);
    }
    *r = Resources();
}

bool makeImage(const DeviceFns& d, VkDevice device, uint32_t width, uint32_t height, VkImage* image,
               VkDeviceMemory* memory) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d.fn.CreateImage(device, &info, nullptr, image) != VK_SUCCESS) return false;
    VkMemoryRequirements need = {};
    d.fn.GetImageMemoryRequirements(device, *image, &need);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = need.size;
    alloc.memoryTypeIndex = typeFor(d.memory, need.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    return alloc.memoryTypeIndex != UINT32_MAX &&
            d.fn.AllocateMemory(device, &alloc, nullptr, memory) == VK_SUCCESS &&
            d.fn.BindImageMemory(device, *image, *memory, 0) == VK_SUCCESS;
}

// Callers hold gLock.
bool make(const DeviceFns& d, VkDevice device, uint32_t width, uint32_t height, Resources* r) {
    r->device = device;
    const bool wide = width >= height;
    const uint32_t lng = wide ? width : height;
    const uint32_t shrt = wide ? height : width;
    const uint32_t gridShort = std::max<uint32_t>(
            kGridShortMin, static_cast<uint32_t>(std::lround(kGridLong * double(shrt) / lng)));
    r->gridWidth = wide ? kGridLong : gridShort;
    r->gridHeight = wide ? gridShort : kGridLong;
    r->midWidth = std::max(r->gridWidth, width / kFirstShrink);
    r->midHeight = std::max(r->gridHeight, height / kFirstShrink);
    r->slotBytes = kHeader + VkDeviceSize{r->gridWidth} * r->gridHeight * 4;
    if (!makeImage(d, device, r->midWidth, r->midHeight, &r->mid, &r->memory[0]) ||
        !makeImage(d, device, r->gridWidth, r->gridHeight, &r->grid, &r->memory[1])) {
        return false;
    }
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = r->slotBytes * kSlots;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d.fn.CreateBuffer(device, &info, nullptr, &r->buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements need = {};
    d.fn.GetBufferMemoryRequirements(device, r->buffer, &need);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = need.size;
    alloc.memoryTypeIndex = typeFor(
            d.memory, need.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    void* mapped = nullptr;
    if (alloc.memoryTypeIndex == UINT32_MAX ||
        d.fn.AllocateMemory(device, &alloc, nullptr, &r->memory[2]) != VK_SUCCESS ||
        d.fn.BindBufferMemory(device, r->buffer, r->memory[2], 0) != VK_SUCCESS ||
        d.fn.MapMemory(device, r->memory[2], 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        return false;
    }
    memset(mapped, 0, static_cast<size_t>(info.size));
    r->mapped = static_cast<const uint8_t*>(mapped);
    return true;
}

VkImageMemoryBarrier barrier(VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src,
                             VkAccessFlags dst) {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.image = image;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    return b;
}

VkImageBlit whole(uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh) {
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[1] = {static_cast<int32_t>(sw), static_cast<int32_t>(sh), 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(dw), static_cast<int32_t>(dh), 1};
    return blit;
}

}  // namespace

void OnDevice(VkDevice device, PFN_vkGetDeviceProcAddr get,
              const VkPhysicalDeviceMemoryProperties& memory) {
    DeviceFns d;
    d.memory = memory;
    if (get == nullptr || !d.fn.load(device, get)) return;
    std::lock_guard lock(gLock);
    gDevices[device] = d;
}

void OnDestroyDevice(VkDevice device) {
    std::lock_guard lock(gLock);
    const auto found = gDevices.find(device);
    if (found == gDevices.end()) return;
    if (gNow.device == device) {
        gArmed.store(false, std::memory_order_relaxed);
        destroy(found->second.fn, &gNow);
    }
    for (auto i = gSpent.begin(); i != gSpent.end();) {
        if (i->device != device) {
            ++i;
            continue;
        }
        destroy(found->second.fn, &*i);
        i = gSpent.erase(i);
    }
    gDevices.erase(found);
}

void Arm(VkDevice device, uint64_t renderPass, VkFormat format, uint32_t width, uint32_t height,
         VkImageLayout final) {
    if (!blittable(format) || width == 0 || height == 0 || final == VK_IMAGE_LAYOUT_UNDEFINED) {
        Disarm();
        return;
    }
    std::lock_guard lock(gLock);
    const auto d = gDevices.find(device);
    if (d == gDevices.end()) return;
    if (gNow.device != device || gWidth != width || gHeight != height) {
        if (gNow.device != VK_NULL_HANDLE) gSpent.push_back(gNow);
        gNow = Resources();
        gWidth = gHeight = 0;
        std::fill(std::begin(gSlotSeq), std::end(gSlotSeq), 0);
        if (!make(d->second, device, width, height, &gNow)) {
            destroy(d->second.fn, &gNow);
            gArmed.store(false, std::memory_order_relaxed);
            ALOGI("hud probe: no copy of the %ux%u frame", width, height);
            return;
        }
        gWidth = width;
        gHeight = height;
    }
    gFinal = final;
    gPass.store(renderPass, std::memory_order_relaxed);
    gShape.store(shapeOf(format, width, height), std::memory_order_relaxed);
    gArmed.store(true, std::memory_order_release);
}

void Disarm() {
    gArmed.store(false, std::memory_order_relaxed);
}

bool Wants(uint64_t renderPass, VkFormat format, uint32_t width, uint32_t height) {
    return gArmed.load(std::memory_order_acquire) &&
            gPass.load(std::memory_order_relaxed) == renderPass &&
            gShape.load(std::memory_order_relaxed) == shapeOf(format, width, height) &&
            gRecordedFor.load(std::memory_order_relaxed) != gFrame.load(std::memory_order_relaxed);
}

void Record(VkCommandBuffer cb, VkImage image) {
    std::lock_guard lock(gLock);
    const uint32_t frame = gFrame.load(std::memory_order_relaxed);
    const VkDevice device = gNow.device;
    if (device == VK_NULL_HANDLE || gNow.mapped == nullptr ||
        gRecordedFor.exchange(frame, std::memory_order_relaxed) == frame) {
        return;
    }
    const auto d = gDevices.find(device);
    if (d == gDevices.end()) return;
    const Fn& f = d->second.fn;
    const Resources& r = gNow;
    const uint32_t slot = frame % kSlots;
    gSlotSeq[slot] = frame;
    const VkAccessFlags read = VK_ACCESS_TRANSFER_READ_BIT;
    const VkAccessFlags write = VK_ACCESS_TRANSFER_WRITE_BIT;
    VkImageMemoryBarrier in[2] = {
            barrier(image, gFinal, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, read),
            barrier(r.mid, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, write),
    };
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 2, in);
    const VkImageBlit first = whole(gWidth, gHeight, r.midWidth, r.midHeight);
    f.CmdBlitImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.mid,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &first, VK_FILTER_LINEAR);
    // The game's image goes back as it was, for whatever reads or presents it next.
    VkImageMemoryBarrier between[3] = {
            barrier(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, gFinal, read,
                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT),
            barrier(r.mid, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, write, read),
            barrier(r.grid, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, write),
    };
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                         nullptr, 0, nullptr, 3, between);
    const VkImageBlit second = whole(r.midWidth, r.midHeight, r.gridWidth, r.gridHeight);
    f.CmdBlitImage(cb, r.mid, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.grid,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &second, VK_FILTER_LINEAR);
    VkImageMemoryBarrier out = barrier(r.grid, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, write, read);
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &out);
    VkBufferImageCopy copy = {};
    copy.bufferOffset = r.slotBytes * slot + kHeader;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {r.gridWidth, r.gridHeight, 1};
    f.CmdCopyImageToBuffer(cb, r.grid, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.buffer, 1, &copy);
    // The slot's sequence goes in after its texels: a header that matches means they landed.
    VkBufferMemoryBarrier landed = {};
    landed.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    landed.srcAccessMask = write;
    landed.dstAccessMask = write;
    landed.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    landed.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    landed.buffer = r.buffer;
    landed.offset = r.slotBytes * slot;
    landed.size = r.slotBytes;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 1, &landed, 0, nullptr);
    f.CmdFillBuffer(cb, r.buffer, r.slotBytes * slot, 4, frame);
    landed.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                         nullptr, 1, &landed, 0, nullptr);
}

void Presented(bool hasCamera, const float* camera) {
    std::lock_guard lock(gLock);
    const uint32_t frame = gFrame.load(std::memory_order_relaxed);
    const uint32_t at = frame % kCameras;
    gCameraSeq[at] = frame;
    gCameraValid[at] = hasCamera && camera != nullptr;
    if (gCameraValid[at]) memcpy(gCameras[at], camera, sizeof(gCameras[at]));
    gFrame.store(frame + 1, std::memory_order_relaxed);
}

bool Take(HudGrid* grid) {
    std::lock_guard lock(gLock);
    const Resources& r = gNow;
    if (r.mapped == nullptr) return false;
    uint32_t best = 0;
    uint32_t bestSlot = 0;
    for (uint32_t s = 0; s < kSlots; s++) {
        uint32_t tag = 0;
        memcpy(&tag, r.mapped + r.slotBytes * s, sizeof(tag));
        if (tag != 0 && tag == gSlotSeq[s] && tag > gTaken && (best == 0 || tag < best)) {
            best = tag;
            bestSlot = s;
        }
    }
    if (best == 0) return false;
    const size_t texels = size_t{r.gridWidth} * r.gridHeight;
    gLuma.resize(texels);
    const uint8_t* rgba = r.mapped + r.slotBytes * bestSlot + kHeader;
    for (size_t i = 0; i < texels; i++) {
        const uint8_t* p = rgba + i * 4;
        gLuma[i] = static_cast<uint8_t>((p[0] * 77u + p[1] * 150u + p[2] * 29u) >> 8);
    }
    uint32_t after = 0;
    memcpy(&after, r.mapped + r.slotBytes * bestSlot, sizeof(after));
    if (after != best) return false;  // recorded over while being read
    gTaken = best;
    grid->luma = gLuma.data();
    grid->width = r.gridWidth;
    grid->height = r.gridHeight;
    grid->seq = best;
    const uint32_t at = best % kCameras;
    grid->hasCamera = gCameraSeq[at] == best && gCameraValid[at];
    if (grid->hasCamera) memcpy(grid->camera, gCameras[at], sizeof(grid->camera));
    return true;
}

}  // namespace framefx::lapis::hud_probe
