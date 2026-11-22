/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// One present interval of a game's GPU work as Lapis sees it from the game's own calls: render
// passes with their attachments and sampled images, the transfers and dispatches between them,
// in submission order. Built by the recorder in the game, or from survey logs in host tests.
namespace framefx::lapis {

// The recorder's name for an image, stable for the image's life. 0: an image it never saw made.
using ImageId = uint32_t;

// What of a `width` x `height` image a viewport draws into, when that is less than all of it: a
// game at a dynamic resolution draws its scene into a corner and scales that to the screen. An
// empty rect is the whole image, or a viewport that says nothing (unknown, or outside it).
inline VkRect2D DrawnRegion(float x, float y, float w, float h, uint32_t width, uint32_t height) {
    const VkRect2D whole = {};
    if (w <= 0 || h == 0 || width == 0 || height == 0) return whole;
    // A negative height spans y + h .. y.
    const float top = h < 0 ? y + h : y;
    const float tall = std::fabs(h);
    const auto ix = static_cast<int32_t>(std::lround(x));
    const auto iy = static_cast<int32_t>(std::lround(top));
    const auto iw = static_cast<uint32_t>(std::lround(w));
    const auto ih = static_cast<uint32_t>(std::lround(tall));
    if (ix < 0 || iy < 0 || iw == 0 || ih == 0 || ix + iw > width || iy + ih > height) return whole;
    // Within a texel or two of all of it is all of it.
    if (iw + 2 >= width && ih + 2 >= height) return whole;
    return {{ix, iy}, {iw, ih}};
}

struct Image {
    ImageId id = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mips = 1;
    uint32_t layers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = 0;
    bool swapchain = false;
    // Diagnostics for the survey.
    int memoryType = -1;
    VkMemoryPropertyFlags memory = 0;
    uint32_t sampledAs = 0;  // descriptor types (bit per VkDescriptorType) the game reads it as
    uint32_t descriptorWrites = 0;
};

using Images = std::unordered_map<ImageId, Image>;

struct Target {
    ImageId image = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkAttachmentLoadOp load = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp store = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkImageLayout final = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    uint32_t attachment = 0;  // index in the render pass
};

// One uniform buffer binding a draw read, as the game left its bytes at submit.
struct Constants {
    uint32_t draw = 0;  // in its pass
    uint32_t shader = 0;  // the draw's fragment shader, by hash; while the survey runs
    uint32_t set = 0;
    uint32_t binding = 0;
    // Where the bytes were: memory object (handle bits) and byte offset in it.
    uint64_t memory = 0;
    uint64_t offset = 0;
    std::vector<float> values;
};

// One draw as Hide HUD labels it (lapis/rust/src/hud/mod.rs HudDraw, field for field), while it
// captures. Bits in lapis/hud.h.
struct HudDraw {
    uint64_t pipeline = 0;
    uint32_t index = 0;     // in its pass, counting every draw
    uint32_t elements = 0;  // vertices or indices; 0 indirect
    uint32_t instances = 0;
    uint32_t shader = 0;    // fragment shader hash
    uint16_t state = 0;
    uint8_t textures = 0;
    uint8_t scissor = 255;  // scissor area over the pass's, 0..255
    uint8_t vertex = 0;
    uint8_t pad[3] = {};
};
static_assert(sizeof(HudDraw) == 32);

struct Pass {
    bool dynamic = false;  // vkCmdBeginRendering
    uint64_t renderPass = 0;  // the VkRenderPass it began, when not dynamic
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t subpasses = 1;
    std::vector<Target> colors;
    bool hasDepth = false;
    Target depth;
    VkAttachmentLoadOp stencilLoad = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp stencilStore = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    float clearDepth = NAN;
    bool depthInput = false;  // a subpass reads the depth attachment as an input
    bool depthResolve = false;
    // While the survey runs, a letter per draw: w writes depth, t tests it, a neither; capitals
    // blend. | clears depth inside the pass, / clears colour.
    std::string drawTrace;
    // Depth cleared inside the pass, and the draws before the last of those: an engine switching
    // cameras, usually from the scene's to the HUD's.
    uint32_t depthClears = 0;
    uint32_t lastClearDraw = 0;
    uint32_t draws = 0;
    uint32_t depthTests = 0;
    uint32_t depthWrites = 0;
    // The last depth-writing draw, and the first blended draw that tests depth without writing it
    // after that: where the transparency begins. UINT32_MAX when none.
    uint32_t lastDepthWrite = UINT32_MAX;
    uint32_t glassFrom = UINT32_MAX;
    std::vector<ImageId> reads;  // images its draws sample or load through descriptors
    // While engine signals run: draws by state (kDraw* bits), and the first draw of each.
    static constexpr int kDrawStates = 16;
    uint32_t drawsByState[kDrawStates] = {};
    uint32_t firstByState[kDrawStates] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                          UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                          UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                          UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    // What its first draw read from uniform buffers, kept for the passes Lapis watches: the
    // scene's draws read their camera there.
    std::vector<Constants> constants;
    // The first draw's viewport height, signed: negative flips y. 0 when unknown.
    float viewportHeight = 0;
    float viewportX = 0, viewportY = 0, viewportWidth = 0;  // with it
    // Draws whose viewport is another than the draw before's.
    uint32_t viewportChanges = 0;
    float lastViewportX = 0, lastViewportY = 0;
    // A watched pass's later viewports (a shadow atlas's tiles, each its own cascade), with what
    // the first draw into each read from uniform buffers.
    struct Tile {
        float x = 0, y = 0, width = 0, height = 0;
        std::vector<Constants> constants;
    };
    static constexpr size_t kMaxTiles = 8;
    std::vector<Tile> tiles;
    // While Hide HUD captures: the pass's shape as its skips are keyed, its draws, and the
    // uniform bytes of the draws sampled for their camera.
    uint64_t hudKey = 0;
    std::vector<HudDraw> hudDraws;
    std::vector<Constants> hudConstants;

    bool writes(ImageId id) const {
        if (hasDepth && depth.image == id) return true;
        for (const Target& t : colors) {
            if (t.image == id) return true;
        }
        return false;
    }
};

// Draw state bits Pass::drawsByState files draws under (lapis/rust/src/signals/passes.rs).
constexpr uint32_t kDrawBlend = 1u << 0;
constexpr uint32_t kDrawDepthTest = 1u << 1;
constexpr uint32_t kDrawDepthWrite = 1u << 2;
constexpr uint32_t kDrawStencilWrite = 1u << 3;

enum class OpKind : uint8_t {
    kSubmit,       // queue in `src`
    kCopy,         // image to image
    kBlit,
    kResolve,
    kToBuffer,     // image to buffer, from `src`
    kClear,        // depth clear of `dst` to `depth`
    kLayout,       // `dst` from `from` to `to`
    kDispatch,     // `count` dispatches touching `reads`
    kOpticalFlow,  // `reads`: binding point << 24 | image
};

struct Op {
    OpKind kind = OpKind::kSubmit;
    ImageId src = 0;
    ImageId dst = 0;
    VkImageLayout from = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout to = VK_IMAGE_LAYOUT_UNDEFINED;
    float depth = NAN;
    uint32_t count = 0;
    std::vector<ImageId> reads;
    uint8_t flips = 0;  // a blit's: kFlipY, kFlipX

    bool transfers() const {
        return kind == OpKind::kCopy || kind == OpKind::kBlit || kind == OpKind::kResolve;
    }
};

// Which way a step turns what it carries: a blit by its offsets, a pass by its viewport.
constexpr uint8_t kFlipY = 1;
constexpr uint8_t kFlipX = 2;

struct Step {
    bool isPass = false;
    Pass pass;
    Op op;
};

struct Frame {
    std::vector<Step> steps;  // submission order
    ImageId presented = 0;    // the swapchain image this present shows
    uint32_t width = 0;       // the swapchain's extent
    uint32_t height = 0;
};

bool IsDepthFormat(VkFormat f);

// Colour formats engines keep linear or reversed depth in, next to the depth image itself.
bool IsDepthLikeFormat(VkFormat f);

// Float colour: the scene before tonemapping, which the screen never shows as is.
bool IsUntoned(VkFormat f);

}  // namespace framefx::lapis
