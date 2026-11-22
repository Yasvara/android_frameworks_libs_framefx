/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "vulkan.h"

#include <android-base/file.h>
#include <android-base/properties.h>
#include <android-base/stringprintf.h>
#include <android-base/strings.h>
#include <apex/display.h>
#include <log/log.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "kelir/cache.h"
#include "kelir/shaders.h"
#include "kelir/vulkan.h"
#include "lapis/recorder.h"
#include "luwes/afme.h"
#include "luwes/capture.h"
#include "luwes/gbuffer.h"
#include "luwes/pacer.h"
#include "luwes/orientation.h"
#include "luwes/resize.h"
#include "luwes/shaders.h"
#include "luwes/timing.h"
#include "src/engine.h"
#include "src/rays.h"
#include "src/resume.h"
#include "src/survey.h"
#include "src/target.h"
#include "src/vk_colour.h"

namespace framefx::luwes::vulkan {
namespace {

constexpr int kFrames = 3;          // real frames in flight per swapchain
constexpr int kAfmeSlots = 3;       // AFME's ring of per-estimate objects
constexpr int kLevels = 4;          // luma pyramid, 1/2 to 1/16 of the frame
constexpr uint32_t kAlign = 128;    // frame size aligned so the coarsest level holds whole blocks
constexpr int kSearchRadius = 12;   // coarsest level, its pixels: +-192 frame pixels
constexpr float kSmoothness = 0.004f;
constexpr int kFinest = 1;           // motion stops at 1/4 of the frame: 32-pixel blocks
constexpr float kFieldScale = 2 << kFinest;  // finest motion level to frame pixels
// output.frag's mode for the game's 3D, or a frame generated from it, under the HUD layer
// hud.frag found: a difference from the game's stretch of its scene under kHudFloor is none
// (resampling noise), over kHudOpaque covers the scene.
constexpr int kComposeLayer = 3;
constexpr VkFormat kHudFormat = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
constexpr VkFormat kMaskFormat = VK_FORMAT_R8_UNORM;
// AFME renders its frame through RGBA8 passes: a 10-bit output would hold its bytes misread.
constexpr VkFormat kAfmeFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr float kHudFloor = 4.0f / 255.0f;
constexpr float kHudOpaque = 0.35f;
// hud.frag's per-pixel HUD coverage from the frames before: what it keeps of them per frame, and
// how much squared scene change (luma) it takes to trust that over the difference's size.
constexpr VkFormat kMatteFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr float kMatteDecay = 0.85f;
constexpr float kMatteTrust = 0.002f;
// A pixel whose difference from the scene lands this far, in luma, from what its matte predicts
// changed its HUD and starts its matte over.
constexpr float kMatteBreak = 0.12f;
constexpr int kScenesKnown = 8;  // scene images whose matte history still holds
// Kelir's scene band: view distance and auto focus. R32F: half-float steps (1.6 cm at 16-32 m)
// flip the normals ambient occlusion builds on a floor seen at a slant, into horizontal bands.
constexpr VkFormat kDistanceFormat = VK_FORMAT_R32_SFLOAT;
constexpr int kDepthViews = 8;  // views of the game's depth images kept at once: scene, shadows

// lapis/rust/src/roles.rs
extern "C" {
struct RolesVote;
struct RolesSample {
    uint32_t image;
    uint32_t pad;
    double chroma[2][6];
    double normal[5];
    double normalN;
};
RolesVote* framefx_lapis_roles_new();
void framefx_lapis_roles_free(RolesVote* r);
void framefx_lapis_roles_frame(RolesVote* r, const RolesSample* samples, size_t count);
void framefx_lapis_roles_picks(const RolesVote* r, uint32_t* albedo, double* albedoAgreement,
                               uint32_t* normal, uint8_t* decoding, double* normalAgreement);
}
// Small copies of the real frame and its scene, compared on the CPU to find which way the game
// stores the scene: Unity keeps its render targets upside down and flips them in the HUD pass.
constexpr uint32_t kThumbWidth = 64;
constexpr uint32_t kThumbHeight = 32;
constexpr VkDeviceSize kThumbBytes = kThumbWidth * kThumbHeight * 4;
// Read back as bytes whatever the swapchain holds.
constexpr VkFormat kThumbFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr int kOrientationVotes = 3;
// A decided orientation is checked against the screen every so many real frames.
constexpr uint32_t kOrientationCheck = 60;
// Thumbnail votes without a winner before a composite pass's viewport decides: about 4 s.
constexpr uint32_t kOrientationHintAfter = 240;
// The scene counts as covered once less than kCoveredBelow of the screen shows it (orientation.h
// says what showing is), kOrientationVotes checks running; and as shown again from kShownFrom.
constexpr float kCoveredBelow = 0.12f;
constexpr float kShownFrom = 0.25f;
// Presents the scene layer's passes stay up without a scene, grading the whole frame: a game
// changing screens offers none for a few frames, and taking the passes down shows.
constexpr int kLayerLinger = 90;
// The game's scene is upscaled when the swapchain is at least this much larger on each side.
constexpr float kUpscaleMin = 1.1f;
constexpr int kSources = 8;  // views of the game's scene images kept at once
// The scene turned into the swapchain's axes, among the game's own in those views.
constexpr lapis::ImageId kUprightId = UINT32_MAX;
// The part of a scene the game drew, scaled to its image's size.
constexpr lapis::ImageId kDrawnId = UINT32_MAX - 1;
// A game drawing at a size other than the display's, by at least this much on each side and at
// the display's aspect, gets images of its own; the swapchain is made at the display's size and
// each frame is resampled into it.
constexpr uint32_t kGameSpare = 2;  // images the game may hold beyond what it asked for
constexpr uint32_t kMaxGameImages = 8;
// SGSR 2 on the game's jittered scene: its accumulation, and the motion it reads.
constexpr VkFormat kHistoryFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kMotionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// The game's velocity is refitted against the block search every this many real frames.
constexpr uint32_t kVelocityFitPeriod = 15;
// Presents after asking for a new swapchain before giving up, and acquires failed to ask.
constexpr uint32_t kRecreateGiveUpAfter = 600;
constexpr int kStaleAcquires = 3;
constexpr int kTemporalViews = 8;  // views of the game's images it reads, kept at once
// How the accumulation's detail over the game's resolve carries onto its finished scene: the
// finished scene follows the resolve as about this power (display encoding over a filmic curve),
// and no pixel changes by more than this factor.
constexpr float kDetailGamma = 0.5f;
constexpr float kDetailMax = 2.0f;
constexpr float kMinLerp = 0.3f;  // SGSR 2's history lerp floor while the camera holds still
constexpr float kSampleWeight = 0.12f;  // a new sample against history; SGSR 2's own is 1 / 3
// Velocity in a 10-bit unorm buffer is packed as sign(m) sqrt(|m|) / 2 + this, m the move in
// texture coordinates (Genshin's motion vector pass).
constexpr float kPackedVelocityZero = 127.0f / 255.0f;
// A G-buffer channel at or above this confidence is fed to frame generation.
constexpr float kGBufferTrusted = 0.8f;
constexpr uint64_t kFenceTimeoutNs = 1'000'000'000;
constexpr int64_t kRefreshCheckNs = 1'000'000'000;
constexpr uint32_t kMaxWaits = 16;
// Capture only: GPU timestamps per real frame. 0 start, 1 copied, 2 pyramid, 3 motion, a start
// and end per generated frame, then the end of each motion dispatch.
constexpr uint32_t kPassTimestamps = 4 + 2 * kMaxPresents;
constexpr uint32_t kMotionDispatches = 3 + 5 * (kLevels - 1 - kFinest);
// Then the scene steps: before them, then after each of depth, scene band, resample, grade, HUD
// layer and held frame that ran.
constexpr uint32_t kSceneAt = kPassTimestamps + kMotionDispatches;
constexpr uint32_t kSceneTimestamps = 7;
constexpr uint32_t kTimestamps = kSceneAt + kSceneTimestamps;

// motion.comp passes.
enum MotionPass { kSearch = 0, kCandidates, kPropagate, kRefine, kMedian, kGlobal };

#define FRAMEFX_VK(X)                                                                      \
    X(QueueSubmit) X(DeviceWaitIdle) X(CreateImage) X(DestroyImage)                       \
    X(GetImageMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindImageMemory)      \
    X(CreateImageView) X(DestroyImageView) X(CreateSampler) X(DestroySampler)             \
    X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreatePipelineLayout)    \
    X(DestroyPipelineLayout) X(CreateShaderModule) X(DestroyShaderModule)                 \
    X(CreateComputePipelines) X(CreateGraphicsPipelines) X(DestroyPipeline)               \
    X(CreateRenderPass) X(DestroyRenderPass) X(CreateFramebuffer) X(DestroyFramebuffer)   \
    X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets)            \
    X(UpdateDescriptorSets) X(CreateCommandPool) X(DestroyCommandPool)                    \
    X(AllocateCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer)                   \
    X(CmdPipelineBarrier) X(CmdCopyImage) X(CmdBeginRenderPass) X(CmdEndRenderPass)       \
    X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDraw)            \
    X(CmdDispatch) X(CmdSetViewport) X(CmdSetScissor)                                     \
    X(CreateFence) X(DestroyFence) X(WaitForFences) X(ResetFences) X(CreateSemaphore)     \
    X(DestroySemaphore) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements)   \
    X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(InvalidateMappedMemoryRanges)       \
    X(CmdCopyImageToBuffer) X(CreateQueryPool) X(DestroyQueryPool) X(CmdResetQueryPool)   \
    X(CmdWriteTimestamp) X(GetQueryPoolResults) X(CmdClearColorImage)                     \
    X(CmdClearDepthStencilImage) X(CmdBlitImage)

// The driver's device entry points: our objects never pass through the loader.
struct Fns {
#define DECLARE(name) PFN_vk##name name = nullptr;
    FRAMEFX_VK(DECLARE)
#undef DECLARE

    bool load(VkDevice device, PFN_vkGetDeviceProcAddr proc) {
        bool ok = true;
#define LOAD(name)                                                        \
    name = reinterpret_cast<PFN_vk##name>(proc(device, "vk" #name)); \
    ok = ok && name != nullptr;
        FRAMEFX_VK(LOAD)
#undef LOAD
        return ok;
    }
};

framefx_vk_host gHost;
std::atomic<int> gRunning{0};
std::atomic<int> gResized{0};  // swapchains whose game draws into images of its own
PFN_vkGetSwapchainImagesKHR gGetImages = nullptr;
PFN_vkAcquireNextImageKHR gAcquire = nullptr;
PFN_vkAcquireNextImage2KHR gAcquire2 = nullptr;
PFN_vkReleaseSwapchainImagesEXT gRelease = nullptr;


struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

// A host-visible copy of one image, read back once its frame retires.
// lapis/rust/src/signals/velocity.rs, field for field.
struct VelocitySample {
    float raw[2];
    float camera[2];  // here: the block search's back vector, frame pixels
};
struct VelocityEncoding {
    float bias;
    float scale[2];
    float residual, inliers, confidence;
    uint8_t kind;  // 0 unknown, 1 linear, 2 squared
    uint8_t pad[3];
};
static_assert(sizeof(VelocityEncoding) == 28);
extern "C" void framefx_signals_fit_velocity(const VelocitySample* samples, size_t count,
                                             VelocityEncoding* out);

struct Readback {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* map = nullptr;
    VkDeviceSize size = 0;
    bool coherent = false;
    std::string name;
};

// Command buffers and sync objects of one real frame in flight.
struct Frame {
    VkCommandBuffer analysis = VK_NULL_HANDLE;
    VkCommandBuffer outputs[kMaxPresents] = {};  // per present that carries a generated frame
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false;
    VkSemaphore acquired[kMaxPresents] = {};
    std::vector<Readback> readbacks;  // capture only
    bool timed = false;               // capture only: this frame wrote its timestamps
    uint32_t timedFrame = 0;
    int timedOutputs = 0;
    uint32_t timedPasses = 0;
    uint32_t timedScene = 0;
    const char* sceneSteps[kSceneTimestamps] = {};  // capture only: what each scene step was
    bool kelirOn = false;  // Kelir's stage A and finish ride along
    kelir::vulkan::Stage kelir;
    bool probed = false;  // carries thumbnails for the scene's orientation
    bool upTimed = false;  // wrote upscaling's two timestamps
    bool velocityRead = false;  // copied the game's velocity and the field for a refit
};

// One recorded motion.comp dispatch.
struct Dispatch {
    VkDescriptorSet set = VK_NULL_HANDLE;
    MotionPass pass = kSearch;
    int level = 0;
};

struct Device;

struct Swapchain {
    Device* device = nullptr;
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    ANativeWindow* window = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent = {};
    uint32_t transform = 0;
    bool fifo = false;
    uint32_t extraImages = 0;
    bool askedToRecreate = false;
    uint32_t presentsSinceAsk = 0;
    // Acquires still to fail with VK_ERROR_OUT_OF_DATE_KHR: engines that shrug off an outdated
    // present (Unity) must rebuild when they cannot get an image. Bounded, so one that retries
    // forever is not starved.
    std::atomic<int> staleAcquires{0};
    LuwesReason declined = LuwesReason::kNone;
    int linger = 0;  // presents left of kLayerLinger
    // How scenes of each shape seen were stored: a game makes its scene image again per screen.
    struct Stored {
        uint32_t width = 0, height = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        int orientation = -1;
    } stored[4];
    int nextStored = 0;
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkFramebuffer> targets;
    // Per image, not per frame: a present's wait is only known done once its image comes back.
    std::vector<VkSemaphore> copied;  // the analysis read the game's image
    std::vector<VkSemaphore> drawn;   // an output pass wrote the image

    // The images the game draws into when it draws at another size than the swapchain's: its
    // `extent`, one swapchain image per acquire, and the submission that last read each.
    struct Game {
        VkExtent2D extent = {};
        std::vector<Image> images;
        std::vector<uint32_t> real;   // the swapchain image each was acquired with
        std::vector<char> held;       // acquired and not yet presented
        std::vector<VkFence> read;
        std::vector<char> pending;    // `read` submitted and not yet waited for
        uint32_t next = 0;
        std::mutex lock;
        VkDescriptorSet sets[kMaxGameImages] = {};  // each image, for resampling
        Image resampled;  // the game's frame at the swapchain's size, for Kelir's stage A
        VkFramebuffer resampledFb = VK_NULL_HANDLE;
        VkFramebuffer colorFb[2] = {};  // the held frames, drawn by the resample
        VkCommandPool plainPool = VK_NULL_HANDLE;  // the plain present's, when Luwes cannot run
        std::vector<VkCommandBuffer> plain;        // per swapchain image
        std::vector<VkFence> plainDone;
        std::vector<VkSemaphore> plainDrawn;
    } game;
    bool resized() const { return !game.images.empty(); }

    Engine::Watch watch;
    Target target;
    Meter meter;
    LuwesStatus status;
    int64_t refreshNs = 1'000'000'000 / 60;
    int64_t refreshCheckedAt = 0;
    bool running = false;
    WindowTiming timing;

    bool allocated = false;
    std::pair<bool, bool> allocatedFor;  // frame generation, upscaling
    bool upscaleFailed = false;          // its images could not be made at this size
    bool solo = false;  // the scene layer with frame generation off
    bool hideHud = false;  // photo mode, this frame
    VkQueryPool upTimer = VK_NULL_HANDLE;  // per frame in flight: before and after upscaling
    uint32_t upGpuNs = 0;                  // upscaling's GPU time per real frame, smoothed
    int64_t upReportedAt = 0;
    uint32_t upFrames = 0;  // real frames since the last report
    bool primed = false;  // field images in GENERAL
    uint32_t alignedWidth = 0;
    uint32_t alignedHeight = 0;
    Image color[2];
    Image luma[2][kLevels];
    VkFramebuffer lumaFb[2][kLevels] = {};
    Image field[kLevels][2];  // kFinest and coarser
    Image global;
    // The game's velocity as a block search candidate: hint.frag samples it per finest block, and
    // its encoding is fitted against the search's own field, so no game is described by hand.
    struct {
        Image raw;  // RGBA16F per finest block: xy as stored, zw rows flipped
        // Back vectors per pixel of level 0, frame pixels: what the output pass warps by.
        Image dense;
        VkFramebuffer denseFb = VK_NULL_HANDLE;
        VkDescriptorSet denseSets[kFrames] = {};
        VkImageView lastView = VK_NULL_HANDLE;  // the game's velocity this frame, or null
        bool denseGame = false;  // this frame's dense field holds the game's vectors
        // splat.comp: the dense field moved to a generated frame's phase, what the output reads.
        Image splat;  // R32UI keys
        Image phase;
        VkDescriptorSet splatSet = VK_NULL_HANDLE;
        VkFramebuffer fb = VK_NULL_HANDLE;
        VkDescriptorSet sets[kFrames] = {};
        VkBuffer read[kFrames] = {};
        VkDeviceMemory readMemory[kFrames] = {};
        const uint8_t* readMap[kFrames] = {};
        bool readCoherent[kFrames] = {};
        bool written = false;  // raw holds this frame's velocity
        uint32_t frames = 0;   // real frames with velocity since the last refit
        // The fit in use; kind 0 when none.
        uint8_t kind = 0;
        bool flipped = false;
        float bias = 0, scale[2] = {}, confidence = 0;
        VelocityEncoding candidate = {};  // the last fit, adopted once a second agrees
        bool candidateFlipped = false;
        uint32_t fits = 0, adopted = 0, cuts = 0, cutReasons = 0;
        VelocityEncoding last = {};  // the latest fit, for the log
        float meanError = 0;          // smoothed, moving blocks
        uint32_t contradicted = 0;    // clear fits in a row against the encoding in use
        bool profileRead = false;     // <app cache>/framefx-velocity.profile, once per swapchain
        int64_t reportedAt = 0;
    } velocity;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet pyramidSet[2][kLevels] = {};
    std::vector<Dispatch> motion[2];
    VkDescriptorSet outputSet[2] = {};
    int current = 0;

    // AFME as the motion engine: depth and motion stand in until the game's are tapped.
    struct {
        void* engine = nullptr;
        float scale = 0.0f;  // phase AFME is set to
        bool primed = false;  // constant inputs cleared, output a colour attachment
        Image depth[2];       // reversed, at the far plane
        bool depthFed[2] = {};  // holds the game's depth, not the far clear
        Image motion[2];      // RG32F, zero
        Image output;         // the generated frame, swapchain format
        Image still;          // a zero field: the output pass reproduces `output` as it is
        VkDescriptorSet set = VK_NULL_HANDLE;
        Image thumb[2];  // the real frame and its scene, kThumbWidth x kThumbHeight
        VkBuffer probe = VK_NULL_HANDLE;
        VkDeviceMemory probeMemory = VK_NULL_HANDLE;
        const uint8_t* probeMap = nullptr;
        bool probeCoherent = false;
        bool probePending = false;
        VkImage probed = VK_NULL_HANDLE;  // the game's scene image the orientation holds for
        uint32_t probedWidth = 0, probedHeight = 0;
        VkFormat probedFormat = VK_FORMAT_UNDEFINED;
        int orientation = -1;             // -1 unknown; bit 0 flips y, bit 1 flips x
        uint64_t voteErr[4] = {};         // the last vote's difference per flip, for a survey
        uint32_t votes = 0;               // thumbnails compared
        int candidate = -1;
        int agree = 0;
        // The screen shows next to nothing of the scene: a menu over it. The whole frame is
        // graded then, as with no scene.
        bool covered = false;
        int hidden = 0;
        bool composing = false;  // this real frame's estimate ran on the scene layer
        // AFME rebuilds a ring of kAfmeSlots per-estimate objects and frees the one it reuses:
        // the estimate kAfmeSlots calls back must have left the GPU. One fence per slot, behind
        // the submission that carried it.
        VkFence done[kAfmeSlots] = {};
        bool pending[kAfmeSlots] = {};
        uint32_t calls = 0;
        int signal = -1;  // slot the output just recorded needs fenced after its submission
        // One estimate per real frame: it costs several ms of GPU at frame size, and the
        // pacer may ask for more generated frames than that; those show the same estimate.
        bool fresh = false;
        bool active = false;  // this frame generates through AFME: made, and the game's choice

        // The scene layer, beside each held real frame: `sharp`, the game's 3D before its HUD at
        // swapchain size (upscaled or not, through Kelir's scene band and stage A), and `hud` and
        // `mask`, what the real frame's HUD puts over it. AFME estimates between the sharp pair,
        // and the frames shown are composed from the layer.
        Image sharp[2];
        Image sharpRaw;  // with Kelir's stage A: before it grades it
        Image hud[2];    // (1 - h) d + h real, 10 bits signed
        Image mask[2];   // h, how much the HUD covers
        Image matte[2];  // what hud.frag learnt of h so far
        VkFramebuffer sharpFb[2] = {};
        VkFramebuffer sharpRawFb = VK_NULL_HANDLE;
        VkFramebuffer hudFb[2] = {};
        VkFramebuffer heldFb[2] = {};  // the held frames, composed from the layer
        bool hasSharp[2] = {};
        bool sgsr[2] = {};  // upscaled through SGSR rather than resampled
        int method[2] = {};  // 0 resampled, 1 SGSR 1, 2 SGSR 2
        VkDescriptorSet upReal[2] = {};      // the 3D, HUD layer, mask
        VkDescriptorSet upCompose[2] = {};   // AFME's frame, HUD layer, mask
        // Luwes's own motion on the 3D alone: its pyramid's source, and the warp of both frames'
        // 3D under this frame's HUD layer and mask.
        VkDescriptorSet scenePyramid[2] = {};
        VkDescriptorSet sceneWarp[2] = {};
        // Per frame in flight: real frame, game's scene, last matte.
        VkDescriptorSet hudSets[kFrames] = {};
        struct Source {
            lapis::ImageId id = 0;
            VkImageView view = VK_NULL_HANDLE;
            VkDescriptorSet set = VK_NULL_HANDLE;
        } sources[kSources];
        int nextSource = 0;
        // The scene images of late: one not among them is another screen, and the matte learnt
        // over the last is not its own.
        lapis::ImageId known[kScenesKnown] = {};
        int nextKnown = 0;
        lapis::ImageId upscaled = 0;  // the scene image last logged as upscaled
        // The scene of a game that turns its frame for a pre-rotated swapchain, in the swapchain's
        // axes. A fixed size, since a game's frames may alternate between scene images.
        const char* layerMiss = nullptr;    // why this frame's layer failed, null if it did not
        const char* layerLogged = nullptr;  // the last logged
        bool layerReported = false;
        Image upright;
        VkFormat uprightFormat = VK_FORMAT_UNDEFINED;
        Image drawn;  // a dynamic resolution's region of the scene, at the scene's size
        VkFormat drawnFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D drawnSize = {};
        VkFramebuffer uprightFb = VK_NULL_HANDLE;
    } afme;

    // Kelir's scene band at the game's scene size: view distance (held across the frames the game
    // drops its depth in), half-size occlusion, auto focus, and the band's result.
    struct {
        uint32_t width = 0;
        uint32_t height = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        bool failed = false;    // its images could not be made at this size
        bool primed = false;    // every image readable
        bool distance = false;  // `lin` holds a frame's depth
        int focus = 0;          // the newest of `focusAt`
        int sun = 0;            // the newest of `sunAt`
        Image lin;
        Image ao;
        Image focusAt[2];
        Image sunAt[2];  // 1 x 1: where the sun stands (sun.frag), ping-pong
        Image fx[2];  // fx[0] the band's result; fx[1] before depth of field
        Image bounced[2];  // global illumination at a quarter of the size, ping-pong
        VkFramebuffer bouncedFb[2] = {};
        int bouncedCur = 0;
        bool bouncedValid = false;  // bounced[bouncedCur] holds the last real frame's
        VkFramebuffer linFb = VK_NULL_HANDLE;
        VkFramebuffer aoFb = VK_NULL_HANDLE;
        VkFramebuffer focusFb[2] = {};
        VkFramebuffer sunFb[2] = {};
        VkFramebuffer fxFb[2] = {};
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet depthSets[kFrames] = {};
        VkDescriptorSet aoSet = VK_NULL_HANDLE;
        VkDescriptorSet focusSets[2] = {};
        VkDescriptorSet sunSets[kFrames] = {};
        VkDescriptorSet passSets[kFrames][3] = {};  // per frame in flight, per pass
        VkDescriptorSet resample = VK_NULL_HANDLE;  // fx[0], for the resample pass
        VkDescriptorSet bouncedSets[kFrames] = {};
        // Kelir's ray-traced kinds at half the size (rays.comp), made at their first frame: x the
        // sun's visibility and y occlusion; reflection premultiplied with its weight in alpha;
        // bounced light, linear. Kept in the general layout, written and sampled.
        Image rtVisibility;
        Image rtReflection;
        Image rtBounce;
        bool rtMade = false;
        bool rtFailed = false;
        bool rtValid = false;  // this frame traced
        bool rtPrimed = false;  // in the general layout
        VkDescriptorPool rtPool = VK_NULL_HANDLE;
        VkDescriptorSet rtSets[kFrames] = {};
        // The same gathered over frames (history.comp), what the composite reads: two of each,
        // this frame's and the last; visibility's z holds the view depth and w the frames kept.
        Image rtHistory[2][3];
        int rtHistoryAt = 0;  // the one this frame writes
        bool rtTracedLast = false;
        uint32_t rtUntraced = 0;  // frames in a row the rays missed since they last traced
        float rtLastVp[16] = {};
        uint32_t rtFrame = 0;
        VkDescriptorSet rtHistorySets[kFrames] = {};
        struct DepthView {
            lapis::ImageId id = 0;
            VkImageView view = VK_NULL_HANDLE;
            uint64_t used = 0;  // the real frame that last read it
        } depths[kDepthViews], gbufferViews[kDepthViews];
        // The sun's light over the sky's, as the frame's own shadow edges show it (rays.comp's
        // stats), and how well each G-buffer target's hue follows the scene's: its albedo's.
        float rtSunOverSky = 1.5f;
        struct RolesVote* rtRoles = nullptr;  // which G-buffer target is what (lapis/rust/src/roles.rs)
    } band;

    // SGSR 2: motion at the game's render size, the accumulation at the swapchain's.
    struct {
        uint32_t width = 0;  // the game's jittered scene
        uint32_t height = 0;
        bool failed = false;
        bool primed = false;
        int cur = 0;          // history[cur] holds the last frame's accumulation
        bool valid = false;   // and the last real frame made it
        Image motion;
        VkFramebuffer motionFb = VK_NULL_HANDLE;
        Image history[2];
        VkFramebuffer upFb[2][3] = {};  // into history[k] and sharp[0], sharp[1] or sharpRaw
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet convertSets[kFrames] = {};
        VkDescriptorSet upSets[kFrames] = {};
        struct View {
            lapis::ImageId id = 0;
            VkImageView view = VK_NULL_HANDLE;
            uint64_t used = 0;
        } views[kTemporalViews];
        float lastVp[16] = {};
        lapis::ImageId logged = 0;
    } temporal;
    uint64_t realFrames = 0;

    uint32_t family = UINT32_MAX;
    VkCommandPool commands = VK_NULL_HANDLE;
    Frame frames[kFrames];
    int frameIndex = 0;

    Capture capture;
    GBufferFeed gbuffer;  // debug.framefx.gbuffer
    bool gbufferConfigured = false;
    VkQueryPool timer = VK_NULL_HANDLE;  // kFrames x kTimestamps
    int captureLeft = 0;
    uint32_t captureFrame = 0;
    int64_t captureStart = 0;
};

struct Output {
    VkFormat format;
    VkRenderPass pass;
    VkPipeline pipeline;
};

enum ModuleIndex {
    kVert, kPyramid, kMotion, kOut, kUpscale, kHud, kDepth, kAo, kFocus, kScene, kTemporal,
    kTemporalUp, kGi, kTranspose, kHint, kDense, kSplat, kSun, kModules
};

struct Device {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    framefx_vk_instance instance = {};
    Fns vk;
    bool usable = false;  // driver entry points and memory properties at hand
    VkPhysicalDeviceMemoryProperties memory = {};
    float timestampPeriod = 0.0f;  // ns per tick, 0 without timestamps on graphics queues
    std::vector<std::pair<VkQueue, uint32_t>> queues;
    std::unordered_map<VkSwapchainKHR, std::unique_ptr<Swapchain>> swapchains;

    // Set when the device was made with what AFME needs. AFME calls through the loader, whose
    // dispatch for the device exists only once creation returned: it is first asked at a
    // swapchain's allocation.
    const afme::Api* afme = nullptr;
    bool afmeChecked = false;

    bool built = false;
    bool broken = false;
    VkSampler linear = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE;
    VkDescriptorSetLayout pyramidLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout motionLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout outputLayout = VK_NULL_HANDLE;
    VkPipelineLayout pyramidPipe = VK_NULL_HANDLE;
    VkPipelineLayout motionPipe = VK_NULL_HANDLE;
    VkPipelineLayout outputPipe = VK_NULL_HANDLE;
    VkShaderModule modules[kModules] = {};
    VkRenderPass r8Pass = VK_NULL_HANDLE;
    VkPipeline pyramid = VK_NULL_HANDLE;
    VkPipeline motion = VK_NULL_HANDLE;
    std::vector<Output> outputs;
    VkDescriptorSetLayout upscaleLayout = VK_NULL_HANDLE;
    VkPipelineLayout upscalePipe = VK_NULL_HANDLE;
    std::vector<Output> transposes;
    std::vector<Output> upscales;  // into a sampled image, per format
    VkDescriptorSetLayout hudLayout = VK_NULL_HANDLE;
    VkPipelineLayout hudPipe = VK_NULL_HANDLE;
    VkRenderPass hudPass = VK_NULL_HANDLE;  // layer and mask
    VkPipeline hud = VK_NULL_HANDLE;
    // Kelir's scene band. Depth and occlusion share the upscale layout (one image), focus the
    // HUD's (two); the band's passes read four and push 128 bytes.
    VkRenderPass distancePass = VK_NULL_HANDLE;
    VkPipeline depth = VK_NULL_HANDLE;
    VkPipeline ao = VK_NULL_HANDLE;
    VkPipeline focus = VK_NULL_HANDLE;
    VkPipeline sun = VK_NULL_HANDLE;  // sun.frag, on the band layout
    VkDescriptorSetLayout bandLayout = VK_NULL_HANDLE;
    VkPipelineLayout bandPipe = VK_NULL_HANDLE;
    std::vector<Output> bands;  // into an image of the game's scene format
    // SGSR 2: the convert pass reads depth and velocity; the upscale pass reads five and writes
    // the accumulation and the scene.
    VkDescriptorSetLayout temporalLayout = VK_NULL_HANDLE;
    VkPipelineLayout temporalPipe = VK_NULL_HANDLE;
    VkDescriptorSetLayout temporalUpLayout = VK_NULL_HANDLE;
    VkPipelineLayout temporalUpPipe = VK_NULL_HANDLE;
    VkRenderPass motionPass = VK_NULL_HANDLE;
    VkPipeline temporal = VK_NULL_HANDLE;
    std::vector<Output> temporalUps;  // per swapchain format
    // Kelir's global illumination: distance, scene, its result before, velocity.
    VkDescriptorSetLayout giLayout = VK_NULL_HANDLE;
    VkPipelineLayout giPipe = VK_NULL_HANDLE;
    VkDescriptorSetLayout denseLayout = VK_NULL_HANDLE;  // dense.frag
    VkPipelineLayout densePipe = VK_NULL_HANDLE;
    VkDescriptorSetLayout splatLayout = VK_NULL_HANDLE;  // splat.comp
    VkPipelineLayout splatPipe = VK_NULL_HANDLE;
    VkPipeline splat = VK_NULL_HANDLE;
    VkPipeline gi = VK_NULL_HANDLE;
    VkPipeline hint = VK_NULL_HANDLE;   // hint.frag, on the temporal layout
    VkPipeline dense = VK_NULL_HANDLE;  // dense.frag, on the gi layout
    // rays.comp, made at the first ray-traced frame: two sampled images, the scene's tree and
    // three storage images.
    VkDescriptorSetLayout raysLayout = VK_NULL_HANDLE;
    VkPipelineLayout raysPipe = VK_NULL_HANDLE;
    VkPipeline rays = VK_NULL_HANDLE;
    // history.comp: seven sampled images and three storage images.
    VkDescriptorSetLayout historyLayout = VK_NULL_HANDLE;
    VkPipelineLayout historyPipe = VK_NULL_HANDLE;
    VkPipeline history = VK_NULL_HANDLE;
    bool raysBroken = false;

    // The output pass with Kelir's finish: set 1 and the frame block are Kelir's.
    VkPipelineLayout finishPipe = VK_NULL_HANDLE;
    VkShaderModule finishModule = VK_NULL_HANDLE;
    kelir::Lru<VkPipeline> finishes;  // by format and stage-B signature
    std::vector<VkPipeline> finishesRetired;  // evicted, maybe in flight: freed with the device
};

std::mutex gLock;
// Windows whose game was asked for a new swapchain: the replacement gets a resume replay.
std::set<ANativeWindow*> gRebuilding;
std::unordered_map<VkDevice, std::unique_ptr<Device>> gDevices;

Device* deviceOf(VkDevice device) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    return it == gDevices.end() ? nullptr : it->second.get();
}

// The swapchain a present targets, with the queue's family.
Swapchain* swapchainOf(VkQueue queue, VkSwapchainKHR handle, uint32_t* family) {
    std::lock_guard lock(gLock);
    for (auto& [device, d] : gDevices) {
        for (const auto& [q, f] : d->queues) {
            if (q != queue) continue;
            const auto it = d->swapchains.find(handle);
            if (it == d->swapchains.end()) return nullptr;
            *family = f;
            return it->second.get();
        }
    }
    return nullptr;
}

uint32_t levelWidth(const Swapchain& sc, int level) {
    return sc.alignedWidth >> (level + 1);
}

uint32_t levelHeight(const Swapchain& sc, int level) {
    return sc.alignedHeight >> (level + 1);
}

// The image the game drew its frame `index` into, and its size.
VkImage appImage(const Swapchain& sc, uint32_t index) {
    return sc.resized() ? sc.game.images[index].image : sc.images[index];
}

VkImageView appView(const Swapchain& sc, uint32_t index) {
    return sc.resized() ? sc.game.images[index].view : sc.views[index];
}

VkExtent2D appExtent(const Swapchain& sc) {
    return sc.resized() ? sc.game.extent : sc.extent;
}

// ---- Object helpers --------------------------------------------------------------------------

bool makeImage(Device& d, Image* out, VkFormat format, uint32_t width, uint32_t height,
               VkImageUsageFlags usage, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d.vk.CreateImage(d.device, &info, nullptr, &out->image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d.vk.GetImageMemoryRequirements(d.device, out->image, &req);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; i++) {
        if ((req.memoryTypeBits & (1u << i)) == 0) continue;
        if (type == UINT32_MAX) type = i;
        if (d.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            type = i;
            break;
        }
    }
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (type == UINT32_MAX ||
        d.vk.AllocateMemory(d.device, &alloc, nullptr, &out->memory) != VK_SUCCESS ||
        d.vk.BindImageMemory(d.device, out->image, out->memory, 0) != VK_SUCCESS) {
        return false;
    }
    VkImageViewCreateInfo view = {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = out->image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = {aspect, 0, 1, 0, 1};
    return d.vk.CreateImageView(d.device, &view, nullptr, &out->view) == VK_SUCCESS;
}

void freeImage(Device& d, Image& image) {
    if (image.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, image.view, nullptr);
    if (image.image != VK_NULL_HANDLE) d.vk.DestroyImage(d.device, image.image, nullptr);
    if (image.memory != VK_NULL_HANDLE) d.vk.FreeMemory(d.device, image.memory, nullptr);
    image = Image();
}

VkSampler makeSampler(Device& d, VkFilter filter) {
    VkSamplerCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = filter;
    info.minFilter = filter;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = info.addressModeV = info.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler = VK_NULL_HANDLE;
    d.vk.CreateSampler(d.device, &info, nullptr, &sampler);
    return sampler;
}

// `sampled` combined image samplers, then `storage` storage images, from binding 0.
VkDescriptorSetLayout makeLayout(Device& d, uint32_t sampled, uint32_t storage,
                                 VkShaderStageFlags stages) {
    VkDescriptorSetLayoutBinding bindings[8] = {};
    const uint32_t count = sampled + storage;
    for (uint32_t i = 0; i < count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = i < sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                 : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = stages;
    }
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = count;
    info.pBindings = bindings;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    d.vk.CreateDescriptorSetLayout(d.device, &info, nullptr, &layout);
    return layout;
}

VkPipelineLayout makePipelineLayout(Device& d, VkDescriptorSetLayout set, uint32_t push = 64) {
    const VkPushConstantRange range = {VK_SHADER_STAGE_ALL, 0, push};
    VkPipelineLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 1;
    info.pSetLayouts = &set;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    d.vk.CreatePipelineLayout(d.device, &info, nullptr, &layout);
    return layout;
}

VkShaderModule makeModule(Device& d, const shaders::Spirv& spirv) {
    VkShaderModuleCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spirv.size;
    info.pCode = spirv.code;
    VkShaderModule module = VK_NULL_HANDLE;
    d.vk.CreateShaderModule(d.device, &info, nullptr, &module);
    return module;
}

// Color attachments, fully overwritten, handed on to `finalLayout`.
VkRenderPass makeRenderPass(Device& d, std::initializer_list<VkFormat> formats,
                            VkImageLayout finalLayout) {
    VkAttachmentDescription attachments[3] = {};
    VkAttachmentReference refs[3] = {};
    uint32_t count = 0;
    for (VkFormat format : formats) {
        VkAttachmentDescription& a = attachments[count];
        a.format = format;
        a.samples = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a.finalLayout = finalLayout;
        refs[count] = {count, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        count++;
    }
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = count;
    subpass.pColorAttachments = refs;
    VkSubpassDependency deps[2] = {};
    // In: earlier reads of this image, and the acquire semaphore's wait for a swapchain image.
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    // Out: the next pyramid level, the motion search and the output pass read it.
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstStageMask =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = count;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 2;
    info.pDependencies = deps;
    VkRenderPass pass = VK_NULL_HANDLE;
    d.vk.CreateRenderPass(d.device, &info, nullptr, &pass);
    return pass;
}

VkPipeline makeGraphics(Device& d, VkShaderModule fragment, VkPipelineLayout layout,
                        VkRenderPass pass, const VkSpecializationInfo* spec = nullptr,
                        uint32_t attachments = 1) {
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = d.modules[kVert];
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment;
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = spec;
    VkPipelineVertexInputStateCreateInfo input = {};
    input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo assembly = {};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport = {};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster = {};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blends[3] = {};
    for (VkPipelineColorBlendAttachmentState& b : blends) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    VkPipelineColorBlendStateCreateInfo blend = {};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = attachments;
    blend.pAttachments = blends;
    const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    VkGraphicsPipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = pass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    d.vk.CreateGraphicsPipelines(d.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    return pipeline;
}

VkPipeline makeCompute(Device& d, VkShaderModule module, VkPipelineLayout layout) {
    VkComputePipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    d.vk.CreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    return pipeline;
}

// Everything every swapchain of the device shares; built on the first processed frame.
bool build(Device& d) {
    if (d.built || d.broken) return d.built;
    d.linear = makeSampler(d, VK_FILTER_LINEAR);
    d.nearest = makeSampler(d, VK_FILTER_NEAREST);
    d.pyramidLayout = makeLayout(d, 1, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.motionLayout = makeLayout(d, 6, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    d.outputLayout = makeLayout(d, 5, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.pyramidPipe = makePipelineLayout(d, d.pyramidLayout);
    d.motionPipe = makePipelineLayout(d, d.motionLayout);
    d.outputPipe = makePipelineLayout(d, d.outputLayout);
    d.upscaleLayout = makeLayout(d, 1, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.upscalePipe = makePipelineLayout(d, d.upscaleLayout);
    d.hudLayout = makeLayout(d, 3, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.hudPipe = makePipelineLayout(d, d.hudLayout);
    d.bandLayout = makeLayout(d, 8, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.giLayout = makeLayout(d, 4, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.giPipe = makePipelineLayout(d, d.giLayout);
    d.denseLayout = makeLayout(d, 5, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.densePipe = makePipelineLayout(d, d.denseLayout);
    d.splatLayout = makeLayout(d, 1, 2, VK_SHADER_STAGE_COMPUTE_BIT);
    d.splatPipe = makePipelineLayout(d, d.splatLayout);
    d.bandPipe = makePipelineLayout(d, d.bandLayout, 128);
    d.temporalLayout = makeLayout(d, 2, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.temporalPipe = makePipelineLayout(d, d.temporalLayout);
    d.temporalUpLayout = makeLayout(d, 5, 0, VK_SHADER_STAGE_FRAGMENT_BIT);
    d.temporalUpPipe = makePipelineLayout(d, d.temporalUpLayout, 128);
    const shaders::Spirv* spirv[kModules] = {
            &shaders::kFullscreenVertSpirv,
            &shaders::kPyramidFragSpirv,
            &shaders::kMotionCompSpirv,
            &shaders::kOutputFragSpirv,
            &shaders::kUpscaleFragSpirv,
            &shaders::kHudFragSpirv,
            &shaders::kDepthFragSpirv,
            &shaders::kAoFragSpirv,
            &shaders::kFocusFragSpirv,
            &shaders::kSceneFragSpirv,
            &shaders::kTemporalFragSpirv,
            &shaders::kTemporalUpFragSpirv,
            &shaders::kGiFragSpirv,
            &shaders::kTransposeFragSpirv,
            &shaders::kHintFragSpirv,
            &shaders::kDenseFragSpirv,
            &shaders::kSplatCompSpirv,
            &shaders::kSunFragSpirv,
    };
    bool ok = d.linear && d.nearest && d.pyramidPipe && d.motionPipe && d.outputPipe &&
            d.upscalePipe && d.hudPipe && d.bandPipe && d.temporalPipe && d.temporalUpPipe &&
            d.giPipe;
    for (int i = 0; i < kModules && ok; i++) {
        d.modules[i] = makeModule(d, *spirv[i]);
        ok = d.modules[i] != VK_NULL_HANDLE;
    }
    if (ok) {
        d.r8Pass = makeRenderPass(d, {VK_FORMAT_R8_UNORM}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        d.pyramid = makeGraphics(d, d.modules[kPyramid], d.pyramidPipe, d.r8Pass);
        d.motion = makeCompute(d, d.modules[kMotion], d.motionPipe);
        d.hudPass = makeRenderPass(d, {kHudFormat, kMaskFormat, kMatteFormat},
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        d.hud = makeGraphics(d, d.modules[kHud], d.hudPipe, d.hudPass, nullptr, 3);
        d.distancePass = makeRenderPass(d, {kDistanceFormat}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        d.depth = makeGraphics(d, d.modules[kDepth], d.upscalePipe, d.distancePass);
        d.ao = makeGraphics(d, d.modules[kAo], d.upscalePipe, d.r8Pass);
        d.focus = makeGraphics(d, d.modules[kFocus], d.hudPipe, d.distancePass);
        d.motionPass = makeRenderPass(d, {kMotionFormat}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        d.temporal = makeGraphics(d, d.modules[kTemporal], d.temporalPipe, d.motionPass);
        d.gi = makeGraphics(d, d.modules[kGi], d.giPipe, d.motionPass);
        d.hint = makeGraphics(d, d.modules[kHint], d.temporalPipe, d.motionPass);
        d.dense = makeGraphics(d, d.modules[kDense], d.densePipe, d.motionPass);
        d.splat = makeCompute(d, d.modules[kSplat], d.splatPipe);
        d.sun = makeGraphics(d, d.modules[kSun], d.bandPipe, d.motionPass);
        ok = d.r8Pass && d.pyramid && d.motion && d.hudPass && d.hud && d.distancePass &&
                d.depth && d.ao && d.focus && d.motionPass && d.temporal && d.gi && d.hint &&
                d.dense && d.splat && d.sun;
    }
    d.built = ok;
    d.broken = !ok;
    if (!ok) ALOGE("Vulkan generation pipelines could not be built");
    return ok;
}

const Output* outputFor(Device& d, VkFormat format) {
    for (const Output& o : d.outputs) {
        if (o.format == format) return &o;
    }
    Output o = {format, makeRenderPass(d, {format}, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
                VK_NULL_HANDLE};
    if (o.pass == VK_NULL_HANDLE) return nullptr;
    o.pipeline = makeGraphics(d, d.modules[kOut], d.outputPipe, o.pass);
    if (o.pipeline == VK_NULL_HANDLE) {
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
        return nullptr;
    }
    d.outputs.push_back(o);
    return &d.outputs.back();
}

const Output* upscaleFor(Device& d, VkFormat format) {
    for (const Output& o : d.upscales) {
        if (o.format == format) return &o;
    }
    Output o = {format, makeRenderPass(d, {format}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                VK_NULL_HANDLE};
    if (o.pass == VK_NULL_HANDLE) return nullptr;
    o.pipeline = makeGraphics(d, d.modules[kUpscale], d.upscalePipe, o.pass);
    if (o.pipeline == VK_NULL_HANDLE) {
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
        return nullptr;
    }
    d.upscales.push_back(o);
    return &d.upscales.back();
}

// The game's scene turned into the swapchain's axes, in an image of its `format`.
const Output* transposeFor(Device& d, VkFormat format) {
    for (const Output& o : d.transposes) {
        if (o.format == format) return &o;
    }
    Output o = {format, makeRenderPass(d, {format}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                VK_NULL_HANDLE};
    if (o.pass == VK_NULL_HANDLE) return nullptr;
    o.pipeline = makeGraphics(d, d.modules[kTranspose], d.upscalePipe, o.pass);
    if (o.pipeline == VK_NULL_HANDLE) {
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
        return nullptr;
    }
    d.transposes.push_back(o);
    return &d.transposes.back();
}

// Kelir's scene band into an image of the game's scene `format`.
const Output* bandFor(Device& d, VkFormat format) {
    for (const Output& o : d.bands) {
        if (o.format == format) return &o;
    }
    Output o = {format, makeRenderPass(d, {format}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                VK_NULL_HANDLE};
    if (o.pass == VK_NULL_HANDLE) return nullptr;
    o.pipeline = makeGraphics(d, d.modules[kScene], d.bandPipe, o.pass);
    if (o.pipeline == VK_NULL_HANDLE) {
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
        return nullptr;
    }
    d.bands.push_back(o);
    return &d.bands.back();
}

// SGSR 2's upscale pass into the accumulation and an image of the swapchain's `format`.
const Output* temporalUpFor(Device& d, VkFormat format) {
    for (const Output& o : d.temporalUps) {
        if (o.format == format) return &o;
    }
    Output o = {format,
                makeRenderPass(d, {kHistoryFormat, format},
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                VK_NULL_HANDLE};
    if (o.pass == VK_NULL_HANDLE) return nullptr;
    o.pipeline = makeGraphics(d, d.modules[kTemporalUp], d.temporalUpPipe, o.pass, nullptr, 2);
    if (o.pipeline == VK_NULL_HANDLE) {
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
        return nullptr;
    }
    d.temporalUps.push_back(o);
    return &d.temporalUps.back();
}

// The output pass for `format` with Kelir's finish for a stage-B signature; null when it cannot
// be built.
VkPipeline finishFor(Device& d, VkFormat format, uint64_t signature) {
    const Output* out = outputFor(d, format);
    if (out == nullptr) return VK_NULL_HANDLE;
    const uint64_t key = signature | static_cast<uint64_t>(out - d.outputs.data()) << 48;
    if (VkPipeline* p = d.finishes.find(key)) return *p;
    if (d.finishPipe == VK_NULL_HANDLE) {
        const VkDescriptorSetLayout sets[] = {d.outputLayout,
                                              kelir::vulkan::planLayout(d.device)};
        if (sets[1] == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        const VkPushConstantRange range = {
                VK_SHADER_STAGE_ALL, 0,
                kelir::shaders::kFramePushOffset + kelir::shaders::kFramePushSize};
        VkPipelineLayoutCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = 2;
        info.pSetLayouts = sets;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &range;
        d.vk.CreatePipelineLayout(d.device, &info, nullptr, &d.finishPipe);
        const kelir::shaders::Spirv& spirv = kelir::shaders::kLuwesFinishFragSpirv;
        VkShaderModuleCreateInfo module = {};
        module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module.codeSize = spirv.size;
        module.pCode = spirv.code;
        d.vk.CreateShaderModule(d.device, &module, nullptr, &d.finishModule);
        if (d.finishPipe == VK_NULL_HANDLE || d.finishModule == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    }
    int32_t values[kelir::shaders::kSpecCount];
    kelir::shaders::SpecValues(signature, values);
    VkSpecializationMapEntry entries[kelir::shaders::kSpecCount];
    for (int i = 0; i < kelir::shaders::kSpecCount; i++) {
        entries[i] = {kelir::shaders::kFirstSpecId + i, static_cast<uint32_t>(i * sizeof(int32_t)),
                      sizeof(int32_t)};
    }
    const VkSpecializationInfo spec = {kelir::shaders::kSpecCount, entries, sizeof(values), values};
    const VkPipeline pipeline = makeGraphics(d, d.finishModule, d.finishPipe, out->pass, &spec);
    if (pipeline == VK_NULL_HANDLE) return pipeline;
    const VkPipeline evicted = d.finishes.insert(key, pipeline);
    if (evicted != VK_NULL_HANDLE) d.finishesRetired.push_back(evicted);
    return pipeline;
}

void destroyDeviceObjects(Device& d) {
    d.finishes.forEach([&](VkPipeline p) { d.vk.DestroyPipeline(d.device, p, nullptr); });
    for (VkPipeline p : d.finishesRetired) d.vk.DestroyPipeline(d.device, p, nullptr);
    if (d.finishModule != VK_NULL_HANDLE) d.vk.DestroyShaderModule(d.device, d.finishModule, nullptr);
    if (d.finishPipe != VK_NULL_HANDLE) d.vk.DestroyPipelineLayout(d.device, d.finishPipe, nullptr);
    for (const Output& o : d.outputs) {
        d.vk.DestroyPipeline(d.device, o.pipeline, nullptr);
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
    }
    d.outputs.clear();
    for (const Output& o : d.upscales) {
        d.vk.DestroyPipeline(d.device, o.pipeline, nullptr);
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
    }
    d.upscales.clear();
    for (const Output& o : d.bands) {
        d.vk.DestroyPipeline(d.device, o.pipeline, nullptr);
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
    }
    d.bands.clear();
    for (const Output& o : d.transposes) {
        d.vk.DestroyPipeline(d.device, o.pipeline, nullptr);
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
    }
    d.transposes.clear();
    for (const Output& o : d.temporalUps) {
        d.vk.DestroyPipeline(d.device, o.pipeline, nullptr);
        d.vk.DestroyRenderPass(d.device, o.pass, nullptr);
    }
    d.temporalUps.clear();
    for (VkPipeline p :
         {d.pyramid, d.motion, d.hud, d.depth, d.ao, d.focus, d.temporal, d.gi, d.hint, d.dense,
          d.splat, d.sun, d.rays, d.history}) {
        if (p != VK_NULL_HANDLE) d.vk.DestroyPipeline(d.device, p, nullptr);
    }
    for (VkRenderPass p : {d.r8Pass, d.hudPass, d.distancePass, d.motionPass}) {
        if (p != VK_NULL_HANDLE) d.vk.DestroyRenderPass(d.device, p, nullptr);
    }
    for (VkShaderModule m : d.modules) {
        if (m != VK_NULL_HANDLE) d.vk.DestroyShaderModule(d.device, m, nullptr);
    }
    for (VkPipelineLayout l : {d.pyramidPipe, d.motionPipe, d.outputPipe, d.upscalePipe, d.hudPipe,
                               d.bandPipe, d.temporalPipe, d.temporalUpPipe, d.giPipe,
                               d.densePipe, d.splatPipe, d.raysPipe, d.historyPipe}) {
        if (l != VK_NULL_HANDLE) d.vk.DestroyPipelineLayout(d.device, l, nullptr);
    }
    for (VkDescriptorSetLayout l : {d.pyramidLayout, d.motionLayout, d.outputLayout,
                                    d.upscaleLayout, d.hudLayout, d.bandLayout, d.temporalLayout,
                                    d.temporalUpLayout, d.giLayout, d.denseLayout,
                                    d.splatLayout, d.raysLayout, d.historyLayout}) {
        if (l != VK_NULL_HANDLE) d.vk.DestroyDescriptorSetLayout(d.device, l, nullptr);
    }
    for (VkSampler s : {d.linear, d.nearest}) {
        if (s != VK_NULL_HANDLE) d.vk.DestroySampler(d.device, s, nullptr);
    }
}

// ---- Capture ---------------------------------------------------------------------------------

const char* captureFormat(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            return "rgba8";
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            return "bgra8";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return "rgb10a2";
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return "rgba16f";
        case VK_FORMAT_R5G6B5_UNORM_PACK16:
            return "rgb565";
        case VK_FORMAT_R8_UNORM:
            return "r8";
        // G-buffer taps: depth as its depth aspect copies out, velocity as the game stores it.
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D16_UNORM_S8_UINT:
            return "d16";
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return "d24";
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_R32_SFLOAT:
            return "r32f";
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            return "a2r10g10b10";
        case VK_FORMAT_R16G16_UNORM:
            return "rg16";
        case VK_FORMAT_R16G16_SNORM:
            return "rg16s";
        case VK_FORMAT_R16G16_SFLOAT:
            return "rg16f";
        default:
            return "unknown";
    }
}

void freeReadback(Device& d, Readback& r) {
    if (r.map != nullptr) d.vk.UnmapMemory(d.device, r.memory);
    if (r.buffer != VK_NULL_HANDLE) d.vk.DestroyBuffer(d.device, r.buffer, nullptr);
    if (r.memory != VK_NULL_HANDLE) d.vk.FreeMemory(d.device, r.memory, nullptr);
    r = Readback();
}

// Records a copy of `image` (in `layout`) into a new host-visible buffer read back with `f`.
// A mapped buffer the GPU copies into, host-cached where the device offers it.
bool makeHostBuffer(Device& d, VkDeviceSize size, VkBuffer* buffer, VkDeviceMemory* memory,
                    const uint8_t** map, bool* coherent) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d.vk.CreateBuffer(d.device, &info, nullptr, buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d.vk.GetBufferMemoryRequirements(d.device, *buffer, &req);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags flags = d.memory.memoryTypes[i].propertyFlags;
        if ((req.memoryTypeBits & (1u << i)) == 0 ||
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
            continue;
        }
        if (type == UINT32_MAX || (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) type = i;
    }
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    void* data = nullptr;
    if (type == UINT32_MAX ||
        d.vk.AllocateMemory(d.device, &alloc, nullptr, memory) != VK_SUCCESS ||
        d.vk.BindBufferMemory(d.device, *buffer, *memory, 0) != VK_SUCCESS ||
        d.vk.MapMemory(d.device, *memory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
        return false;
    }
    *map = static_cast<const uint8_t*>(data);
    *coherent = (d.memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    return true;
}

int storedOrientation(const Swapchain& sc, const lapis::Tap& scene) {
    for (const auto& s : sc.stored) {
        if (s.orientation >= 0 && s.width == scene.width && s.height == scene.height &&
            s.format == scene.format) {
            return s.orientation;
        }
    }
    return -1;
}

void storeOrientation(Swapchain& sc) {
    const auto& a = sc.afme;
    auto* slot = &sc.stored[sc.nextStored];
    for (auto& s : sc.stored) {
        if (s.width == a.probedWidth && s.height == a.probedHeight && s.format == a.probedFormat) {
            slot = &s;
        }
    }
    if (slot == &sc.stored[sc.nextStored]) sc.nextStored = (sc.nextStored + 1) % 4;
    *slot = {a.probedWidth, a.probedHeight, a.probedFormat, a.orientation};
}

// Which way the scene thumbnail lines up with the real frame's: the flip that differs least, once
// it wins clearly kOrientationVotes times running. A decided orientation changes the same way.
void decideOrientation(Device& d, Swapchain& sc) {
    if (sc.afme.probeMap == nullptr) return;
    if (!sc.afme.probeCoherent) {
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = sc.afme.probeMemory;
        range.size = VK_WHOLE_SIZE;
        d.vk.InvalidateMappedMemoryRanges(d.device, 1, &range);
    }
    const uint8_t* real = sc.afme.probeMap;
    const uint8_t* scene = sc.afme.probeMap + kThumbBytes;
    const ThumbVote v = CompareThumbs<kThumbWidth, kThumbHeight>(real, scene, sc.afme.orientation);
    if (survey::enabled() && sc.afme.votes % 32 == 0) {
        // Both thumbnails, shown frame first, to see what the vote compared.
        const std::string path = android::base::StringPrintf(
                "/data/user/%u/%s/cache/framefx-thumbs-%u.rgba", getuid() / 100000,
                Engine::get().package().c_str(), sc.afme.votes);
        if (FILE* out = fopen(path.c_str(), "we")) {
            fwrite(real, 1, 2 * kThumbBytes, out);
            fclose(out);
        }
    }
    std::copy(v.err, v.err + 4, sc.afme.voteErr);
    sc.afme.votes++;
    if (v.shown >= 0.0f) {
        sc.afme.hidden = v.shown < kCoveredBelow ? sc.afme.hidden + 1 : 0;
        const bool covered = sc.afme.covered ? v.shown < kShownFrom
                                             : sc.afme.hidden >= kOrientationVotes;
        if (covered != sc.afme.covered) {
            sc.afme.covered = covered;
            ALOGI("luwes: the game's scene is %s", covered ? "covered" : "shown again");
        }
    }
    // A flat or symmetric frame says nothing.
    const int vote = VoteOf(v);
    if (vote < 0 || vote == sc.afme.orientation || vote != sc.afme.candidate) {
        sc.afme.candidate = vote == sc.afme.orientation ? -1 : vote;
        sc.afme.agree = sc.afme.candidate < 0 ? 0 : 1;
        return;
    }
    if (++sc.afme.agree >= kOrientationVotes) {
        sc.afme.orientation = vote;
        sc.afme.candidate = -1;
        sc.afme.agree = 0;
        storeOrientation(sc);
        ALOGI("luwes: the game's scene is stored%s%s%s", vote == 0 ? " upright" : "",
              (vote & 1) ? " upside down" : "", (vote & 2) ? " mirrored" : "");
    }
}

void readback(Device& d, Swapchain& sc, Frame& f, VkCommandBuffer cb, VkImage image,
              VkImageLayout layout, VkFormat format, uint32_t width, uint32_t height,
              const std::string& tag, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    const char* name = captureFormat(format);
    const uint32_t bpp = CaptureBytesPerPixel(name);
    if (bpp == 0) return;
    Readback r;
    r.size = static_cast<VkDeviceSize>(width) * height * bpp;
    r.name = android::base::StringPrintf("%03u-%s.%s", sc.captureFrame, tag.c_str(), name);
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = r.size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d.vk.CreateBuffer(d.device, &info, nullptr, &r.buffer) != VK_SUCCESS) return;
    VkMemoryRequirements req;
    d.vk.GetBufferMemoryRequirements(d.device, r.buffer, &req);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags flags = d.memory.memoryTypes[i].propertyFlags;
        if ((req.memoryTypeBits & (1u << i)) == 0 ||
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
            continue;
        }
        if (type == UINT32_MAX || (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) type = i;
    }
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (type == UINT32_MAX ||
        d.vk.AllocateMemory(d.device, &alloc, nullptr, &r.memory) != VK_SUCCESS ||
        d.vk.BindBufferMemory(d.device, r.buffer, r.memory, 0) != VK_SUCCESS ||
        d.vk.MapMemory(d.device, r.memory, 0, VK_WHOLE_SIZE, 0, &r.map) != VK_SUCCESS) {
        freeReadback(d, r);
        return;
    }
    r.coherent = (d.memory.memoryTypes[type].propertyFlags &
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkBufferImageCopy region = {};
    region.imageSubresource = {aspect, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    d.vk.CmdCopyImageToBuffer(cb, image, layout, r.buffer, 1, &region);
    sc.capture.note(android::base::StringPrintf("file %s %ux%u %s", r.name.c_str(), width,
                                                height, name));
    f.readbacks.push_back(std::move(r));
}

// The frame retired: hand its copies and GPU times to the capture writer.
void fitVelocity(Device& d, Swapchain& sc, Frame& f);

void harvest(Device& d, Swapchain& sc, Frame& f) {
    if (f.velocityRead) fitVelocity(d, sc, f);
    if (f.probed) {
        f.probed = false;
        sc.afme.probePending = false;
        decideOrientation(d, sc);
    }
    bool layerTimed = false;
    if (f.upTimed && sc.upTimer != VK_NULL_HANDLE) {
        f.upTimed = false;
        uint64_t ticks[2] = {};
        const uint32_t slot = static_cast<uint32_t>(&f - sc.frames) * 2;
        if (d.vk.GetQueryPoolResults(d.device, sc.upTimer, slot, 2, sizeof(ticks), ticks,
                                     sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
            ticks[1] > ticks[0]) {
            const auto ns = static_cast<uint32_t>(static_cast<double>(ticks[1] - ticks[0]) *
                                                  d.timestampPeriod);
            sc.upGpuNs = sc.upGpuNs == 0 ? ns : (sc.upGpuNs * 7 + ns) / 8;
            layerTimed = true;
        }
    }
    if (f.kelirOn) {
        // Under the HUD Kelir's work is the scene layer's.
        kelir::vulkan::stageRetired(d.device, sc.handle, f.kelir.serial,
                                    layerTimed ? sc.upGpuNs : 0);
        f.kelirOn = false;
    }
    if (f.timed && sc.timer != VK_NULL_HANDLE) {
        const uint32_t base = static_cast<uint32_t>(&f - sc.frames) * kTimestamps;
        uint64_t ticks[kTimestamps] = {};
        const uint32_t count = 4 + 2 * static_cast<uint32_t>(f.timedOutputs);
        if (d.vk.GetQueryPoolResults(d.device, sc.timer, base, count, sizeof(ticks), ticks,
                                     sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const auto ms = [&](int a, int b) {
                return static_cast<double>(ticks[b] - ticks[a]) * d.timestampPeriod / 1e6;
            };
            std::string line = android::base::StringPrintf(
                    "gpu %03u copy %.3f pyramid %.3f motion %.3f", f.timedFrame, ms(0, 1),
                    ms(1, 2), ms(2, 3));
            for (int k = 0; k < f.timedOutputs; k++) {
                line += android::base::StringPrintf(" output%d %.3f", k + 1,
                                                    ms(4 + 2 * k, 5 + 2 * k));
            }
            sc.capture.note(line);
        }
        if (f.timedPasses > 0 &&
            d.vk.GetQueryPoolResults(d.device, sc.timer, base + kPassTimestamps, f.timedPasses,
                                     sizeof(ticks), ticks + kPassTimestamps, sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
            d.vk.GetQueryPoolResults(d.device, sc.timer, base + 2, 1, sizeof(uint64_t),
                                     ticks + 2, sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            std::string line = android::base::StringPrintf("passes %03u", f.timedFrame);
            for (uint32_t i = 0; i < f.timedPasses; i++) {
                const uint64_t from = i == 0 ? ticks[2] : ticks[kPassTimestamps + i - 1];
                line += android::base::StringPrintf(
                        " %.3f",
                        static_cast<double>(ticks[kPassTimestamps + i] - from) *
                                d.timestampPeriod / 1e6);
            }
            sc.capture.note(line);
        }
        if (f.timedScene > 1 &&
            d.vk.GetQueryPoolResults(d.device, sc.timer, base + kSceneAt, f.timedScene,
                                     sizeof(ticks), ticks + kSceneAt, sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            std::string line = android::base::StringPrintf("scene %03u", f.timedFrame);
            for (uint32_t i = 1; i < f.timedScene; i++) {
                line += android::base::StringPrintf(
                        " %s %.3f", f.sceneSteps[i],
                        static_cast<double>(ticks[kSceneAt + i] - ticks[kSceneAt + i - 1]) *
                                d.timestampPeriod / 1e6);
            }
            sc.capture.note(line);
        }
        f.timed = false;
        f.timedPasses = 0;
        f.timedScene = 0;
    }
    for (Readback& r : f.readbacks) {
        if (!r.coherent) {
            VkMappedMemoryRange range = {};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = r.memory;
            range.size = VK_WHOLE_SIZE;
            d.vk.InvalidateMappedMemoryRanges(d.device, 1, &range);
        }
        const auto* bytes = static_cast<const uint8_t*>(r.map);
        sc.capture.write(r.name, std::vector<uint8_t>(bytes, bytes + r.size));
        freeReadback(d, r);
    }
    f.readbacks.clear();
    if (sc.captureLeft == 0 && sc.capture.active() &&
        std::none_of(std::begin(sc.frames), std::end(sc.frames),
                     [](const Frame& o) { return !o.readbacks.empty(); })) {
        sc.capture.end();
    }
}

// ---- Per-swapchain resources ----------------------------------------------------------------

void freeUpright(Device& d, Swapchain& sc) {
    if (sc.afme.uprightFb != VK_NULL_HANDLE) {
        d.vk.DestroyFramebuffer(d.device, sc.afme.uprightFb, nullptr);
        sc.afme.uprightFb = VK_NULL_HANDLE;
    }
    freeImage(d, sc.afme.upright);
    sc.afme.uprightFormat = VK_FORMAT_UNDEFINED;
    freeImage(d, sc.afme.drawn);
    sc.afme.drawnFormat = VK_FORMAT_UNDEFINED;
    sc.afme.drawnSize = {};
}

void waitFrames(Device& d, Swapchain& sc) {
    for (Frame& f : sc.frames) {
        if (!f.pending) continue;
        d.vk.WaitForFences(d.device, 1, &f.fence, VK_TRUE, kFenceTimeoutNs);
        d.vk.ResetFences(d.device, 1, &f.fence);
        f.pending = false;
        harvest(d, sc, f);
    }
}

void freeBand(Device& d, Swapchain& sc) {
    auto& b = sc.band;
    for (VkFramebuffer fb : {b.linFb, b.aoFb, b.focusFb[0], b.focusFb[1], b.fxFb[0], b.fxFb[1],
                             b.bouncedFb[0], b.bouncedFb[1], b.sunFb[0], b.sunFb[1]}) {
        if (fb != VK_NULL_HANDLE) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
    }
    for (Image* image : {&b.lin, &b.ao, &b.focusAt[0], &b.focusAt[1], &b.fx[0], &b.fx[1],
                         &b.bounced[0], &b.bounced[1], &b.sunAt[0], &b.sunAt[1],
                         &b.rtVisibility, &b.rtReflection, &b.rtBounce}) {
        freeImage(d, *image);
    }
    for (auto& set : b.rtHistory) {
        for (Image& image : set) freeImage(d, image);
    }
    if (b.rtPool != VK_NULL_HANDLE) d.vk.DestroyDescriptorPool(d.device, b.rtPool, nullptr);
    if (b.rtRoles != nullptr) framefx_lapis_roles_free(b.rtRoles);
    for (const auto& depth : b.depths) {
        if (depth.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, depth.view, nullptr);
    }
    for (const auto& view : b.gbufferViews) {
        if (view.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, view.view, nullptr);
    }
    if (b.pool != VK_NULL_HANDLE) d.vk.DestroyDescriptorPool(d.device, b.pool, nullptr);
    b = {};
}

void freeTemporal(Device& d, Swapchain& sc) {
    auto& t = sc.temporal;
    if (t.motionFb != VK_NULL_HANDLE) d.vk.DestroyFramebuffer(d.device, t.motionFb, nullptr);
    for (auto& row : t.upFb) {
        for (VkFramebuffer fb : row) {
            if (fb != VK_NULL_HANDLE) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
        }
    }
    freeImage(d, t.motion);
    for (Image& image : t.history) freeImage(d, image);
    for (const auto& v : t.views) {
        if (v.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, v.view, nullptr);
    }
    if (t.pool != VK_NULL_HANDLE) d.vk.DestroyDescriptorPool(d.device, t.pool, nullptr);
    t = {};
}

void freeAnalysis(Device& d, Swapchain& sc) {
    kelir::vulkan::forgetViews(d.device, sc.handle);
    for (int p = 0; p < 2; p++) {
        freeImage(d, sc.color[p]);
        for (int k = 0; k < kLevels; k++) {
            if (sc.lumaFb[p][k] != VK_NULL_HANDLE) {
                d.vk.DestroyFramebuffer(d.device, sc.lumaFb[p][k], nullptr);
            }
            sc.lumaFb[p][k] = VK_NULL_HANDLE;
            freeImage(d, sc.luma[p][k]);
        }
        sc.motion[p].clear();
    }
    for (auto& level : sc.field) {
        for (Image& image : level) freeImage(d, image);
    }
    freeImage(d, sc.global);
    auto& v = sc.velocity;
    if (v.fb != VK_NULL_HANDLE) d.vk.DestroyFramebuffer(d.device, v.fb, nullptr);
    v.fb = VK_NULL_HANDLE;
    freeImage(d, v.raw);
    if (v.denseFb != VK_NULL_HANDLE) d.vk.DestroyFramebuffer(d.device, v.denseFb, nullptr);
    v.denseFb = VK_NULL_HANDLE;
    freeImage(d, v.dense);
    freeImage(d, v.splat);
    freeImage(d, v.phase);
    v.splatSet = VK_NULL_HANDLE;
    v.denseGame = false;
    for (int i = 0; i < kFrames; i++) {
        if (v.readMap[i] != nullptr) d.vk.UnmapMemory(d.device, v.readMemory[i]);
        if (v.read[i] != VK_NULL_HANDLE) d.vk.DestroyBuffer(d.device, v.read[i], nullptr);
        if (v.readMemory[i] != VK_NULL_HANDLE) d.vk.FreeMemory(d.device, v.readMemory[i], nullptr);
        v.read[i] = VK_NULL_HANDLE;
        v.readMemory[i] = VK_NULL_HANDLE;
        v.readMap[i] = nullptr;
        v.sets[i] = VK_NULL_HANDLE;
        v.denseSets[i] = VK_NULL_HANDLE;
        sc.frames[i].velocityRead = false;
    }
    v.written = false;
    for (int j = 0; j < kAfmeSlots; j++) {
        if (sc.afme.done[j] == VK_NULL_HANDLE) continue;
        if (sc.afme.pending[j]) {
            d.vk.WaitForFences(d.device, 1, &sc.afme.done[j], VK_TRUE, kFenceTimeoutNs);
        }
        d.vk.DestroyFence(d.device, sc.afme.done[j], nullptr);
    }
    if (sc.afme.engine != nullptr) d.afme->destroy(sc.afme.engine);
    for (Image& image : sc.afme.depth) freeImage(d, image);
    for (Image& image : sc.afme.motion) freeImage(d, image);
    freeUpright(d, sc);
    for (VkFramebuffer fb : {sc.afme.sharpFb[0], sc.afme.sharpFb[1], sc.afme.sharpRawFb,
                             sc.afme.hudFb[0], sc.afme.hudFb[1], sc.afme.heldFb[0],
                             sc.afme.heldFb[1]}) {
        if (fb != VK_NULL_HANDLE) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
    }
    for (Image& image : sc.afme.sharp) freeImage(d, image);
    for (Image& image : sc.afme.hud) freeImage(d, image);
    for (Image& image : sc.afme.mask) freeImage(d, image);
    for (Image& image : sc.afme.matte) freeImage(d, image);
    freeImage(d, sc.afme.sharpRaw);
    for (const auto& source : sc.afme.sources) {
        if (source.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, source.view, nullptr);
    }
    for (Image& image : sc.afme.thumb) freeImage(d, image);
    if (sc.afme.probe != VK_NULL_HANDLE) d.vk.DestroyBuffer(d.device, sc.afme.probe, nullptr);
    if (sc.afme.probeMemory != VK_NULL_HANDLE) {
        d.vk.UnmapMemory(d.device, sc.afme.probeMemory);
        d.vk.FreeMemory(d.device, sc.afme.probeMemory, nullptr);
    }
    for (Frame& f : sc.frames) f.probed = false;
    for (Frame& f : sc.frames) f.upTimed = false;
    if (sc.upTimer != VK_NULL_HANDLE) d.vk.DestroyQueryPool(d.device, sc.upTimer, nullptr);
    sc.upTimer = VK_NULL_HANDLE;
    sc.upGpuNs = 0;
    freeImage(d, sc.afme.output);
    freeImage(d, sc.afme.still);
    sc.afme = {};
    freeTemporal(d, sc);
    for (VkFramebuffer fb : {sc.game.resampledFb, sc.game.colorFb[0], sc.game.colorFb[1]}) {
        if (fb != VK_NULL_HANDLE) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
    }
    sc.game.resampledFb = VK_NULL_HANDLE;
    sc.game.colorFb[0] = sc.game.colorFb[1] = VK_NULL_HANDLE;
    freeImage(d, sc.game.resampled);
    std::fill(std::begin(sc.game.sets), std::end(sc.game.sets), VK_NULL_HANDLE);
    freeBand(d, sc);
    if (sc.pool != VK_NULL_HANDLE) d.vk.DestroyDescriptorPool(d.device, sc.pool, nullptr);
    sc.pool = VK_NULL_HANDLE;
    sc.allocated = false;
}

void freeCommands(Device& d, Swapchain& sc) {
    for (Frame& f : sc.frames) {
        if (f.fence != VK_NULL_HANDLE) d.vk.DestroyFence(d.device, f.fence, nullptr);
        for (VkSemaphore s : f.acquired) {
            if (s != VK_NULL_HANDLE) d.vk.DestroySemaphore(d.device, s, nullptr);
        }
        f = Frame();
    }
    if (sc.commands != VK_NULL_HANDLE) d.vk.DestroyCommandPool(d.device, sc.commands, nullptr);
    if (sc.timer != VK_NULL_HANDLE) d.vk.DestroyQueryPool(d.device, sc.timer, nullptr);
    sc.commands = VK_NULL_HANDLE;
    sc.timer = VK_NULL_HANDLE;
    sc.family = UINT32_MAX;
}

void freeTargets(Device& d, Swapchain& sc) {
    for (VkFramebuffer fb : sc.targets) d.vk.DestroyFramebuffer(d.device, fb, nullptr);
    for (VkImageView v : sc.views) d.vk.DestroyImageView(d.device, v, nullptr);
    for (VkSemaphore s : sc.copied) d.vk.DestroySemaphore(d.device, s, nullptr);
    for (VkSemaphore s : sc.drawn) d.vk.DestroySemaphore(d.device, s, nullptr);
    sc.targets.clear();
    sc.views.clear();
    sc.copied.clear();
    sc.drawn.clear();
}

void setRunning(Swapchain& sc, bool running) {
    if (sc.running == running) return;
    sc.running = running;
    gRunning.fetch_add(running ? 1 : -1, std::memory_order_relaxed);
    if (running) {
        // The game renders into one image and may be acquiring the next.
        sc.target.pacer().buffers(std::max(static_cast<int>(sc.images.size()) - 2, 1));
    } else {
        sc.timing.reset(sc.window);
    }
}

// The game's own images; with the swapchain or its device.
void freeGame(Device& d, Swapchain& sc);

void freeAll(Device& d, Swapchain& sc) {
    waitFrames(d, sc);
    freeAnalysis(d, sc);
    freeCommands(d, sc);
    freeTargets(d, sc);
    setRunning(sc, false);
}

VkFramebuffer makeFramebuffer(Device& d, VkRenderPass pass, std::initializer_list<VkImageView> views,
                              uint32_t width, uint32_t height) {
    VkFramebufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass = pass;
    info.attachmentCount = static_cast<uint32_t>(views.size());
    info.pAttachments = views.begin();
    info.width = width;
    info.height = height;
    info.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    d.vk.CreateFramebuffer(d.device, &info, nullptr, &fb);
    return fb;
}

bool prepareTargets(Device& d, Swapchain& sc, const Output& out) {
    if (!sc.targets.empty()) return true;
    VkSemaphoreCreateInfo semaphore = {};
    semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (VkImage image : sc.images) {
        VkImageViewCreateInfo view = {};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = sc.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView v = VK_NULL_HANDLE;
        if (d.vk.CreateImageView(d.device, &view, nullptr, &v) != VK_SUCCESS) return false;
        sc.views.push_back(v);
        const VkFramebuffer fb =
                makeFramebuffer(d, out.pass, {v}, sc.extent.width, sc.extent.height);
        if (fb == VK_NULL_HANDLE) return false;
        sc.targets.push_back(fb);
        for (auto* list : {&sc.copied, &sc.drawn}) {
            VkSemaphore s = VK_NULL_HANDLE;
            if (d.vk.CreateSemaphore(d.device, &semaphore, nullptr, &s) != VK_SUCCESS) return false;
            list->push_back(s);
        }
    }
    return true;
}

bool prepareCommands(Device& d, Swapchain& sc, uint32_t family) {
    if (sc.family == family) return true;
    waitFrames(d, sc);
    freeCommands(d, sc);
    VkCommandPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = family;
    if (d.vk.CreateCommandPool(d.device, &pool, nullptr, &sc.commands) != VK_SUCCESS) return false;
    sc.family = family;
    if (d.timestampPeriod > 0.0f) {
        VkQueryPoolCreateInfo timer = {};
        timer.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        timer.queryType = VK_QUERY_TYPE_TIMESTAMP;
        timer.queryCount = kFrames * kTimestamps;
        if (d.vk.CreateQueryPool(d.device, &timer, nullptr, &sc.timer) != VK_SUCCESS) {
            sc.timer = VK_NULL_HANDLE;
        }
    }
    VkFenceCreateInfo fence = {};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkSemaphoreCreateInfo semaphore = {};
    semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (Frame& f : sc.frames) {
        VkCommandBuffer buffers[1 + kMaxPresents];
        VkCommandBufferAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = sc.commands;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1 + kMaxPresents;
        // Through the loader when it offers it: AFME records into these through the loader.
        const PFN_vkAllocateCommandBuffers allocateBuffers =
                gHost.allocate_command_buffers != nullptr ? gHost.allocate_command_buffers
                                                          : d.vk.AllocateCommandBuffers;
        if (allocateBuffers(d.device, &alloc, buffers) != VK_SUCCESS ||
            d.vk.CreateFence(d.device, &fence, nullptr, &f.fence) != VK_SUCCESS) {
            return false;
        }
        lapis::ignore(buffers, 1 + kMaxPresents);
        f.analysis = buffers[0];
        for (int i = 0; i < kMaxPresents; i++) {
            f.outputs[i] = buffers[1 + i];
            if (d.vk.CreateSemaphore(d.device, &semaphore, nullptr, &f.acquired[i]) != VK_SUCCESS) {
                return false;
            }
        }
    }
    return true;
}

// rays.comp's push block.
struct RaysFrame {
    float viewProjection[16];
    float inverse[16];
    float sun[4];
    float depth[4];
    float size[4];
    float rays[12];
    float sky[4];
    float noise[4];
};
static_assert(sizeof(RaysFrame) <= 256, "past the push constants every device has here");

// rays.comp's uniform block: the game's shadow maps, the camera's axes and the G-buffer's roles.
// Frames in a row without a trace that still show the last one.
constexpr uint32_t kHeldUntraced = 6;

struct ShadowBlock {
    float toMap[lapis::kShadowMaps][16];  // camera-relative world to each map's clip space
    float info[lapis::kShadowMaps][4];    // x reversed, y flipped, z depth per metre, w there
    float counts[4];  // x maps, y the sun's light over the sky's, z G-buffer targets
    float view[16];   // columns: the camera's x, y and z axes in world
    float roles[4];   // x albedo target + 1, y normal target + 1, z its decoding; 0 none
    float region[lapis::kShadowMaps][4];  // where in its image each map was drawn: corner, size
};


// rays.comp's stats block, added to by its rays, read back frames later (rays.comp `Stats`).
constexpr int kChromaFixed = 64;
struct RayStats {
    int32_t sunOverSky;  // sum, in 1/256
    int32_t edges;       // shadow edges measured
    // Per G-buffer target, per log-chroma channel: n, Σx, Σy, Σxy, Σx², Σy² (x the scene's), in
    // 1/kChromaFixed and its square.
    int32_t chroma[lapis::kGBufferTargets][2][6];
    // Per target: the cosine with the triangles' normal per decoding, (1 + it) in 1/500, and n.
    int32_t normal[lapis::kGBufferTargets][6];
};

// The camera's axes in world from a row-major view-projection: x along clip x, z against the
// view direction (clip w), y completing a right-handed frame.
void cameraAxes(const float vp[16], float out[16]) {
    double z[3] = {-vp[12], -vp[13], -vp[14]};
    double x[3] = {vp[0], vp[1], vp[2]};
    const auto normalize = [](double* v) {
        const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (l > 0) {
            for (int k = 0; k < 3; k++) v[k] /= l;
        }
    };
    normalize(z);
    const double xz = x[0] * z[0] + x[1] * z[1] + x[2] * z[2];
    for (int k = 0; k < 3; k++) x[k] -= xz * z[k];
    normalize(x);
    const double y[3] = {z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2],
                         z[0] * x[1] - z[1] * x[0]};
    std::fill(out, out + 16, 0.0f);
    for (int k = 0; k < 3; k++) {
        out[0 + k] = static_cast<float>(x[k]);
        out[4 + k] = static_cast<float>(y[k]);
        out[8 + k] = static_cast<float>(z[k]);
    }
    out[15] = 1.0f;
}

bool readOnlyDepth(VkImageLayout l) {
    return l == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL ||
            l == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL ||
            l == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL ||
            l == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || l == VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL ||
            l == VK_IMAGE_LAYOUT_GENERAL;
}

// history.comp's push block.
struct HistoryFrame {
    float inverse[16];
    float previous[16];
    float depth[4];
    float size[4];
    float control[4];
};

struct Binding {
    VkImageView view;
    VkImageLayout layout;
    VkSampler sampler;
};

// Size-dependent images and every descriptor set the analysis and output passes use: for frame
// generation, for the scene layer, or both.
bool allocateAnalysis(Device& d, Swapchain& sc, bool generation, bool layer) {
    waitFrames(d, sc);
    freeAnalysis(d, sc);
    const uint32_t w = sc.extent.width;
    const uint32_t h = sc.extent.height;
    sc.alignedWidth = (w + kAlign - 1) / kAlign * kAlign;
    sc.alignedHeight = (h + kAlign - 1) / kAlign * kAlign;
    const VkImageUsageFlags r8 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags vectors = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    bool ok = true;
    for (int p = 0; p < 2 && ok; p++) {
        // Copied into, or with Kelir drawn into by its stage A.
        ok = makeImage(d, &sc.color[p], sc.format, w, h,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        for (int k = 0; k < kLevels && ok; k++) {
            const uint32_t lw = levelWidth(sc, k);
            const uint32_t lh = levelHeight(sc, k);
            ok = makeImage(d, &sc.luma[p][k], VK_FORMAT_R8_UNORM, lw, lh, r8);
            if (ok) {
                sc.lumaFb[p][k] = makeFramebuffer(d, d.r8Pass, {sc.luma[p][k].view}, lw, lh);
                ok = sc.lumaFb[p][k] != VK_NULL_HANDLE;
            }
        }
    }
    for (int k = kFinest; k < kLevels && ok; k++) {
        for (int j = 0; j < 2 && ok; j++) {
            ok = makeImage(d, &sc.field[k][j], VK_FORMAT_R16G16B16A16_SFLOAT,
                           levelWidth(sc, k) / 8, levelHeight(sc, k) / 8, vectors);
        }
    }
    ok = ok && makeImage(d, &sc.global, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, vectors);
    {
        auto& v = sc.velocity;
        const uint32_t gw = levelWidth(sc, kFinest) / 8;
        const uint32_t gh = levelHeight(sc, kFinest) / 8;
        ok = ok && makeImage(d, &v.raw, kMotionFormat, gw, gh,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        if (ok) {
            v.fb = makeFramebuffer(d, d.motionPass, {v.raw.view}, gw, gh);
            ok = v.fb != VK_NULL_HANDLE;
        }
        ok = ok && makeImage(d, &v.dense, kMotionFormat, levelWidth(sc, 0), levelHeight(sc, 0),
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (ok) {
            v.denseFb = makeFramebuffer(d, d.motionPass, {v.dense.view}, levelWidth(sc, 0),
                                        levelHeight(sc, 0));
            ok = v.denseFb != VK_NULL_HANDLE;
        }
        const VkImageUsageFlags scratch = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ok = ok && makeImage(d, &v.splat, VK_FORMAT_R32_UINT, levelWidth(sc, 0), levelHeight(sc, 0),
                             VK_IMAGE_USAGE_STORAGE_BIT);
        ok = ok && makeImage(d, &v.phase, kMotionFormat, levelWidth(sc, 0), levelHeight(sc, 0),
                             scratch);
        // Raw texels, then the finest field, 8 bytes each.
        for (int i = 0; i < kFrames && ok; i++) {
            ok = makeHostBuffer(d, VkDeviceSize{16} * gw * gh, &v.read[i], &v.readMemory[i],
                                &v.readMap[i], &v.readCoherent[i]);
        }
    }
    if (!ok) return false;

    // Per parity: kLevels pyramid sets, 3 coarsest-level and 5 per finer-level motion sets, one
    // output set.
    constexpr uint32_t kMotionSets = kMotionDispatches;
    // + AFME's output set; the layer's two for real frames, two for generated ones, one per frame
    // in flight for the HUD layer and one per view of the scene.
    // + one per image of the game's own, to resample it.
    constexpr uint32_t kSets =
            2 * (kLevels + kMotionSets + 1) + 12 + 3 * kFrames + kSources + kMaxGameImages;
    const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             2 * (kLevels + 6 * kMotionSets + 5) + 44 + 10 * kFrames + kSources + kMaxGameImages},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kMotionSets + 2},
    };
    VkDescriptorPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = kSets;
    pool.poolSizeCount = std::size(sizes);
    pool.pPoolSizes = sizes;
    if (d.vk.CreateDescriptorPool(d.device, &pool, nullptr, &sc.pool) != VK_SUCCESS) return false;

    const auto allocate = [&](VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = sc.pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        ok = ok && d.vk.AllocateDescriptorSets(d.device, &alloc, &set) == VK_SUCCESS;
        return set;
    };
    // Pointers into `images` must stay put: reserve it for every write up front.
    std::vector<VkDescriptorImageInfo> images;
    std::vector<VkWriteDescriptorSet> updates;
    images.reserve(kSets * 7);
    const auto bind = [&](VkDescriptorSet set, std::initializer_list<Binding> sampled,
                          const Binding* storage, uint32_t first = 0) {
        uint32_t binding = first;
        for (const Binding& b : sampled) {
            images.push_back({b.sampler, b.view, b.layout});
            VkWriteDescriptorSet u = {};
            u.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            u.dstSet = set;
            u.dstBinding = binding++;
            u.descriptorCount = 1;
            u.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            u.pImageInfo = &images.back();
            updates.push_back(u);
        }
        if (storage != nullptr) {
            images.push_back({VK_NULL_HANDLE, storage->view, VK_IMAGE_LAYOUT_GENERAL});
            VkWriteDescriptorSet u = {};
            u.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            u.dstSet = set;
            u.dstBinding = binding;
            u.descriptorCount = 1;
            u.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            u.pImageInfo = &images.back();
            updates.push_back(u);
        }
    };

    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImageLayout general = VK_IMAGE_LAYOUT_GENERAL;
    const auto field = [&](int k, int j) {
        return Binding{sc.field[k][j].view, general, d.nearest};
    };
    const Binding global = {sc.global.view, general, d.nearest};
    const int top = kLevels - 1;
    for (int p = 0; p < 2; p++) {
        const int q = 1 - p;
        for (int k = 0; k < kLevels; k++) {
            sc.pyramidSet[p][k] = allocate(d.pyramidLayout);
            const Binding source = k == 0 ? Binding{sc.color[p].view, read, d.linear}
                                          : Binding{sc.luma[p][k - 1].view, read, d.linear};
            bind(sc.pyramidSet[p][k], {source}, nullptr);
        }
        const auto motion = [&](MotionPass pass, int k, Binding parent, Binding input,
                                Binding globalIn, Binding out) {
            const VkDescriptorSet set = allocate(d.motionLayout);
            bind(set,
                 {Binding{sc.luma[p][k].view, read, d.nearest},
                  Binding{sc.luma[q][k].view, read, d.nearest}, parent, input, globalIn,
                  Binding{sc.velocity.raw.view, read, d.nearest}},
                 &out);
            sc.motion[p].push_back({set, pass, k});
        };
        // Coarsest level: search into A, median into B (its result), dominant vector from B.
        motion(kSearch, top, field(top, 1), field(top, 1), global, field(top, 0));
        motion(kMedian, top, field(top, 0), field(top, 0), global, field(top, 1));
        motion(kGlobal, top, field(top, 0), field(top, 1), field(top, 0), global);
        // Finer levels: candidates into A, propagate into B, refine twice, median into A (the
        // level's result).
        for (int k = top - 1; k >= kFinest; k--) {
            const Binding parent = k + 1 == top ? field(top, 1) : field(k + 1, 0);
            motion(kCandidates, k, parent, field(k, 1), global, field(k, 0));
            motion(kPropagate, k, parent, field(k, 0), global, field(k, 1));
            motion(kRefine, k, parent, field(k, 1), global, field(k, 0));
            motion(kRefine, k, parent, field(k, 0), global, field(k, 1));
            motion(kMedian, k, parent, field(k, 1), global, field(k, 0));
        }
        // The HUD layer's two bind when the layer is made.
        const Binding held = {sc.color[p].view, read, d.linear};
        sc.outputSet[p] = allocate(d.outputLayout);
        bind(sc.outputSet[p],
             {held, Binding{sc.color[q].view, read, d.linear},
              Binding{sc.velocity.phase.view, general, d.linear}, held, held},
             nullptr);
    }
    for (VkDescriptorSet& set : sc.velocity.sets) set = allocate(d.temporalLayout);
    for (VkDescriptorSet& set : sc.velocity.denseSets) set = allocate(d.denseLayout);
    sc.velocity.splatSet = allocate(d.splatLayout);
    {
        const Binding keys = {sc.velocity.splat.view, VK_IMAGE_LAYOUT_GENERAL, VK_NULL_HANDLE};
        const Binding phase = {sc.velocity.phase.view, VK_IMAGE_LAYOUT_GENERAL, VK_NULL_HANDLE};
        bind(sc.velocity.splatSet,
             {Binding{sc.velocity.dense.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, d.nearest}},
             &keys);
        bind(sc.velocity.splatSet, {}, &phase, 2);
    }
    if (d.afme != nullptr && !d.afmeChecked) {
        d.afmeChecked = true;
        afme::Context context = {d.instance.instance, d.device, d.physical};
        if (afme::BehindAngle()) {
            ALOGI("luwes: AFME left out: the game's GL is ANGLE");
            d.afme = nullptr;
        } else if (d.afme->isSupported(&context)) {
            ALOGI("luwes: AFME available");
        } else {
            ALOGW("luwes: AFME does not support this GPU or driver");
            d.afme = nullptr;
        }
    }
    if (ok && generation && d.afme != nullptr) {
        // AFME's frame in, both as the current and the previous one, over a zero field: the
        // output pass then writes it unchanged and adds Kelir's finish.
        const VkImageUsageFlags fill = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        for (int j = 0; j < 2 && ok; j++) {
            ok = makeImage(d, &sc.afme.depth[j], VK_FORMAT_D24_UNORM_S8_UINT, w, h,
                           fill | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT) &&
                    makeImage(d, &sc.afme.motion[j], VK_FORMAT_R32G32_SFLOAT, w, h,
                              fill | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        }
        ok = ok &&
                makeImage(d, &sc.afme.output, kAfmeFormat, w, h,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_STORAGE_BIT) &&
                makeImage(d, &sc.afme.still, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, fill);
        afme::Context context = {d.instance.instance, d.device, d.physical};
        afme::InitInfo init = {w, h, w, h, afme::kFlags, -0.5f};
        if (ok) sc.afme.engine = d.afme->init(&context, &init);
        sc.afme.scale = init.scaleFactor;
        VkFenceCreateInfo fence = {};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        for (int j = 0; j < kAfmeSlots && sc.afme.engine != nullptr; j++) {
            if (d.vk.CreateFence(d.device, &fence, nullptr, &sc.afme.done[j]) != VK_SUCCESS) {
                d.afme->destroy(sc.afme.engine);
                sc.afme.engine = nullptr;
            }
        }
        if (sc.afme.engine != nullptr) {
            sc.afme.set = allocate(d.outputLayout);
            const Binding output = {sc.afme.output.view, read, d.linear};
            bind(sc.afme.set,
                 {output, output, Binding{sc.afme.still.view, read, d.nearest}, output, output},
                 nullptr);
        } else {
            // Luwes's own engine still runs; AFME's images are freed with the rest.
            ALOGW("luwes: AFME declined %ux%u; using Luwes's own motion", w, h);
            ok = true;
        }
    }
    // The game's scene, read through Lapis: its orientation first.
    if (ok && (sc.afme.engine != nullptr || layer)) {
        for (int j = 0; j < 2 && ok; j++) {
            ok = makeImage(d, &sc.afme.thumb[j], kThumbFormat, kThumbWidth, kThumbHeight,
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        }
        ok = ok && makeHostBuffer(d, 2 * kThumbBytes, &sc.afme.probe, &sc.afme.probeMemory,
                                  &sc.afme.probeMap, &sc.afme.probeCoherent);
    }
    const Output* up = ok && layer ? upscaleFor(d, sc.format) : nullptr;
    bool sharp = up != nullptr;
    for (int j = 0; j < 2 && sharp; j++) {
        sharp = makeImage(d, &sc.afme.sharp[j], sc.format, w, h,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        if (sharp) sc.afme.sharpFb[j] = makeFramebuffer(d, up->pass, {sc.afme.sharp[j].view}, w, h);
        sharp = sharp && sc.afme.sharpFb[j] != VK_NULL_HANDLE;
    }
    sharp = sharp &&
            makeImage(d, &sc.afme.sharpRaw, sc.format, w, h,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (sharp) sc.afme.sharpRawFb = makeFramebuffer(d, up->pass, {sc.afme.sharpRaw.view}, w, h);
    sharp = sharp && sc.afme.sharpRawFb != VK_NULL_HANDLE;
    const VkImageUsageFlags drawn = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    for (int j = 0; j < 2 && sharp; j++) {
        sharp = makeImage(d, &sc.afme.hud[j], kHudFormat, w, h, drawn) &&
                makeImage(d, &sc.afme.mask[j], kMaskFormat, w, h, drawn) &&
                makeImage(d, &sc.afme.matte[j], kMatteFormat, w, h, drawn);
        if (sharp) {
            sc.afme.hudFb[j] = makeFramebuffer(
                    d, d.hudPass,
                    {sc.afme.hud[j].view, sc.afme.mask[j].view, sc.afme.matte[j].view}, w, h);
            sc.afme.heldFb[j] = makeFramebuffer(d, up->pass, {sc.color[j].view}, w, h);
        }
        sharp = sharp && sc.afme.hudFb[j] != VK_NULL_HANDLE && sc.afme.heldFb[j] != VK_NULL_HANDLE;
    }
    if (sharp) {
        for (int p = 0; p < 2; p++) {
            const Binding hud = {sc.afme.hud[p].view, read, d.nearest};
            const Binding mask = {sc.afme.mask[p].view, read, d.nearest};
            sc.afme.upReal[p] = allocate(d.outputLayout);
            bind(sc.afme.upReal[p],
                 {Binding{sc.afme.sharp[p].view, read, d.nearest}, hud, mask, hud, mask}, nullptr);
            if (sc.outputSet[p] != VK_NULL_HANDLE) bind(sc.outputSet[p], {hud, mask}, nullptr, 3);
            if (sc.outputSet[p] != VK_NULL_HANDLE) {
                sc.afme.scenePyramid[p] = allocate(d.pyramidLayout);
                bind(sc.afme.scenePyramid[p], {Binding{sc.afme.sharp[p].view, read, d.linear}},
                     nullptr);
                sc.afme.sceneWarp[p] = allocate(d.outputLayout);
                bind(sc.afme.sceneWarp[p],
                     {Binding{sc.afme.sharp[p].view, read, d.linear},
                      Binding{sc.afme.sharp[1 - p].view, read, d.linear},
                      Binding{sc.velocity.phase.view, VK_IMAGE_LAYOUT_GENERAL, d.linear},
                      hud,
                      mask},
                     nullptr);
            }
            if (sc.afme.engine == nullptr) continue;
            sc.afme.upCompose[p] = allocate(d.outputLayout);
            bind(sc.afme.upCompose[p],
                 {Binding{sc.afme.output.view, read, d.nearest}, hud, mask, hud, mask}, nullptr);
        }
        for (auto& source : sc.afme.sources) source.set = allocate(d.upscaleLayout);
        for (VkDescriptorSet& set : sc.afme.hudSets) set = allocate(d.hudLayout);
        if (d.timestampPeriod > 0.0f) {
            VkQueryPoolCreateInfo timer = {};
            timer.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            timer.queryType = VK_QUERY_TYPE_TIMESTAMP;
            timer.queryCount = 2 * kFrames;
            if (d.vk.CreateQueryPool(d.device, &timer, nullptr, &sc.upTimer) != VK_SUCCESS) {
                sc.upTimer = VK_NULL_HANDLE;
            }
        }
    } else if (ok && layer) {
        ALOGW("luwes: no memory for the scene layer at %ux%u", w, h);
        sc.upscaleFailed = true;
    }
    if (ok && sc.resized()) {
        // The game's frames are resampled into the held frames, or for Kelir into `resampled`.
        const Output* resample = upscaleFor(d, sc.format);
        ok = resample != nullptr &&
                makeImage(d, &sc.game.resampled, sc.format, w, h,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        if (ok) {
            sc.game.resampledFb = makeFramebuffer(d, resample->pass, {sc.game.resampled.view}, w, h);
            for (int j = 0; j < 2; j++) {
                sc.game.colorFb[j] = makeFramebuffer(d, resample->pass, {sc.color[j].view}, w, h);
            }
            ok = sc.game.resampledFb && sc.game.colorFb[0] && sc.game.colorFb[1];
        }
        for (size_t i = 0; i < sc.game.images.size() && ok; i++) {
            sc.game.sets[i] = allocate(d.upscaleLayout);
            bind(sc.game.sets[i], {Binding{sc.game.images[i].view, read, d.linear}}, nullptr);
        }
    }
    if (!ok) return false;
    d.vk.UpdateDescriptorSets(d.device, static_cast<uint32_t>(updates.size()), updates.data(), 0,
                              nullptr);
    sc.allocated = true;
    sc.allocatedFor = {generation, layer};
    sc.primed = false;
    sc.current = 0;
    return true;
}

// ---- Recording ------------------------------------------------------------------------------

void imageBarrier(const Fns& vk, VkCommandBuffer cb, VkImage image, VkImageLayout from,
                  VkImageLayout to, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                  VkPipelineStageFlags dstStage, VkAccessFlags dstAccess,
                  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vk.CmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void memoryBarrier(const Fns& vk, VkCommandBuffer cb, VkPipelineStageFlags srcStage,
                   VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                   VkAccessFlags dstAccess) {
    VkMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vk.CmdPipelineBarrier(cb, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void begin(const Fns& vk, VkCommandBuffer cb) {
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.BeginCommandBuffer(cb, &info);
}

void fullscreen(const Fns& vk, VkCommandBuffer cb, VkRenderPass pass, VkFramebuffer fb,
                uint32_t width, uint32_t height, VkPipeline pipeline, VkPipelineLayout layout,
                VkDescriptorSet set, const float* params, uint32_t bytes = 64) {
    VkRenderPassBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = pass;
    info.framebuffer = fb;
    info.renderArea = {{0, 0}, {width, height}};
    vk.CmdBeginRenderPass(cb, &info, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport = {0.0f, 0.0f, static_cast<float>(width),
                                 static_cast<float>(height), 0.0f, 1.0f};
    const VkRect2D scissor = {{0, 0}, {width, height}};
    vk.CmdSetViewport(cb, 0, 1, &viewport);
    vk.CmdSetScissor(cb, 0, 1, &scissor);
    vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
    if (params != nullptr) vk.CmdPushConstants(cb, layout, VK_SHADER_STAGE_ALL, 0, bytes, params);
    vk.CmdDraw(cb, 3, 1, 0, 0);
    vk.CmdEndRenderPass(cb);
}

// Whether the game's scene is enough smaller than the swapchain to upscale.
bool Smaller(const lapis::Tap& scene, VkExtent2D extent) {
    return scene.width * kUpscaleMin <= extent.width && scene.height * kUpscaleMin <= extent.height;
}

// Whether a frame wants the scene layer: the game's 3D to upscale, to grade apart from its HUD,
// to run Kelir's scene band on, or to show without its HUD (unless the game drew none: Lapis
// skipped its HUD draws).
bool wantsLayer(const Config& config, const lapis::Taps& taps, VkExtent2D extent) {
    const kelir::Plan* plan = config.kelir.get();
    const bool grades = !config.lapis.wholeScreen && plan != nullptr && !plan->bypass() &&
            plan->detailCount + plan->colourCount > 0;
    const bool band =
            plan != nullptr && !plan->peek && plan->sceneCount > 0 && taps.camera.known;
    return (config.upscale.on() && Smaller(taps.scene, extent)) || grades || band ||
            (config.lapis.hideHud && !taps.hudSkipped);
}

// Whether those passes are worth keeping up over the whole frame while a scene is missing.
bool wantsWhole(const Config& config) {
    const kelir::Plan* plan = config.kelir.get();
    return config.upscale.on() || config.lapis.hideHud ||
            (!config.lapis.wholeScreen && plan != nullptr && !plan->bypass() &&
             plan->detailCount + plan->colourCount > 0);
}

// Whether the game's scene lies across the swapchain's image: it turned its frame last.
bool turned(const lapis::Tap& scene, VkExtent2D extent) {
    return (scene.width > scene.height) != (extent.width > extent.height);
}

// What this frame's taps may be used for: only photo mode may show a scene from a frame before,
// or the scene of a game that generates frames itself.
bool usable(const Config& config, const lapis::Taps& taps) {
    return config.lapis.hideHud || (!taps.held && !taps.gameGenerates);
}

// A sampled view of the game's scene image, with its upscale set; null when none can be made.
auto* sourceFor(Device& d, Swapchain& sc, const lapis::Tap& scene) {
    for (auto& source : sc.afme.sources) {
        if (source.id == scene.id && source.view != VK_NULL_HANDLE) return &source;
    }
    // The oldest view's frames left the GPU long ago: a scene image changes on resize at most.
    auto& source = sc.afme.sources[sc.afme.nextSource];
    decltype(&source) none = nullptr;
    if (source.set == VK_NULL_HANDLE) return none;
    if (source.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, source.view, nullptr);
    source = {0, VK_NULL_HANDLE, source.set};
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = scene.image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = scene.format;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (d.vk.CreateImageView(d.device, &info, nullptr, &source.view) != VK_SUCCESS) {
        source.view = VK_NULL_HANDLE;
        return none;
    }
    source.id = scene.id;
    const VkDescriptorImageInfo image = {d.linear, source.view,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = source.set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    d.vk.UpdateDescriptorSets(d.device, 1, &write, 0, nullptr);
    sc.afme.nextSource = (sc.afme.nextSource + 1) % kSources;
    return &source;
}

// The aspects a barrier on a depth image names.
VkImageAspectFlags depthAspects(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        default:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    }
}

// Bindings 0.. in order; from `storageFrom` on they are storage images.
void writeImages(Device& d, VkDescriptorSet set, std::initializer_list<Binding> images,
                 uint32_t storageFrom = UINT32_MAX) {
    VkDescriptorImageInfo infos[10];
    VkWriteDescriptorSet writes[10] = {};
    uint32_t n = 0;
    for (const Binding& b : images) {
        infos[n] = {b.sampler, b.view, b.layout};
        writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet = set;
        writes[n].dstBinding = n;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = n < storageFrom ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                   : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[n].pImageInfo = &infos[n];
        n++;
    }
    d.vk.UpdateDescriptorSets(d.device, n, writes, 0, nullptr);
}

// rays.comp's pipeline, and the band's images for it, made at the first ray-traced frame.
bool prepareRays(Device& d, Swapchain& sc) {
    auto& b = sc.band;
    if (b.rtMade || b.rtFailed || d.raysBroken) return b.rtMade;
    if (d.rays == VK_NULL_HANDLE) {
        d.raysBroken = true;
        // Distance, scene, the tree, three results, the game's shadow maps and their matrices,
        // its G-buffer targets, the stats block, and the tree's surfaces: their tests, the
        // coordinates and the textures the tests sample.
        VkDescriptorSetLayoutBinding bindings[13] = {};
        for (uint32_t i = 0; i < 13; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType =
                    i < 2 || i == 6 || i == 8 || i == 12 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    : i == 2                  ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                    : i == 7                  ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                    : i >= 9                  ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = i == 6 ? lapis::kShadowMaps
                    : i == 8                     ? lapis::kGBufferTargets
                    : i == 12                    ? rays::kTextures
                                                 : 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout = {};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = 13;
        layout.pBindings = bindings;
        if (d.vk.CreateDescriptorSetLayout(d.device, &layout, nullptr, &d.raysLayout) != VK_SUCCESS) {
            return false;
        }
        d.raysPipe = makePipelineLayout(d, d.raysLayout, sizeof(RaysFrame));
        const VkShaderModule module = makeModule(d, shaders::kRaysCompSpirv);
        if (d.raysPipe == VK_NULL_HANDLE || module == VK_NULL_HANDLE) return false;
        d.rays = makeCompute(d, module, d.raysPipe);
        d.vk.DestroyShaderModule(d.device, module, nullptr);
        if (d.rays == VK_NULL_HANDLE) {
            ALOGW("luwes: rays.comp refused by the driver; no ray-traced kinds");
            return false;
        }
        VkDescriptorSetLayoutBinding history[10] = {};
        for (uint32_t i = 0; i < 10; i++) {
            history[i].binding = i;
            history[i].descriptorType = i < 7 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            history[i].descriptorCount = 1;
            history[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        layout.bindingCount = 10;
        layout.pBindings = history;
        if (d.vk.CreateDescriptorSetLayout(d.device, &layout, nullptr, &d.historyLayout) !=
            VK_SUCCESS) {
            return false;
        }
        d.historyPipe = makePipelineLayout(d, d.historyLayout, sizeof(HistoryFrame));
        const VkShaderModule gather = makeModule(d, shaders::kHistoryCompSpirv);
        if (d.historyPipe == VK_NULL_HANDLE || gather == VK_NULL_HANDLE) return false;
        d.history = makeCompute(d, gather, d.historyPipe);
        d.vk.DestroyShaderModule(d.device, gather, nullptr);
        if (d.history == VK_NULL_HANDLE) return false;
        d.raysBroken = false;
    }
    const uint32_t hw = std::max(b.width / 2, 1u);
    const uint32_t hh = std::max(b.height / 2, 1u);
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bool ok = makeImage(d, &b.rtVisibility, kMotionFormat, hw, hh, usage) &&
            makeImage(d, &b.rtReflection, kMotionFormat, hw, hh, usage) &&
            makeImage(d, &b.rtBounce, kMotionFormat, hw, hh, usage);
    for (auto& set : b.rtHistory) {
        for (Image& image : set) ok = ok && makeImage(d, &image, kMotionFormat, hw, hh, usage);
    }
    const VkDescriptorPoolSize sizes[5] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             (2 + lapis::kShadowMaps + lapis::kGBufferTargets + rays::kTextures + 7) * kFrames},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kFrames},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, (3 + 3) * kFrames},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFrames},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * kFrames}};
    VkDescriptorPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 2 * kFrames;
    pool.poolSizeCount = 5;
    pool.pPoolSizes = sizes;
    ok = ok && d.vk.CreateDescriptorPool(d.device, &pool, nullptr, &b.rtPool) == VK_SUCCESS;
    for (VkDescriptorSet& set : b.rtSets) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = b.rtPool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &d.raysLayout;
        ok = ok && d.vk.AllocateDescriptorSets(d.device, &info, &set) == VK_SUCCESS;
    }
    for (VkDescriptorSet& set : b.rtHistorySets) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = b.rtPool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &d.historyLayout;
        ok = ok && d.vk.AllocateDescriptorSets(d.device, &info, &set) == VK_SUCCESS;
    }
    if (!ok) {
        ALOGW("luwes: no memory for the ray-traced kinds at %ux%u", hw, hh);
        b.rtFailed = true;
        return false;
    }
    b.rtMade = true;
    return true;
}

// The frame's camera for the rays, column-major as GLSL reads it: `inverse` maps clip space to
// world around the camera (its position subtracted, in double, so points near it keep their
// precision), `viewProjection` back. `vp` is row-major, clip = vp * world.
// `vp` (row-major, clip = vp * world) on world around `eye`: vp T(eye), column-major.
void towardEye(const float vp[16], const double eye[3], float out[16]) {
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            double w = vp[r * 4 + c];
            if (c == 3) {
                w += static_cast<double>(vp[r * 4]) * eye[0] +
                        static_cast<double>(vp[r * 4 + 1]) * eye[1] +
                        static_cast<double>(vp[r * 4 + 2]) * eye[2];
            }
            out[c * 4 + r] = static_cast<float>(w);
        }
    }
}

// Frames whose view-projection was found camera-relative, and world-space, for the log.
std::atomic<uint32_t> gRelativeFrames{0};
std::atomic<uint32_t> gWorldFrames{0};

// A game whose scene constants hold its view-projection in world space only some frames (others
// camera-relative, the eye at 0) keeps the eye it last placed: points are around the camera
// either way, and kept trees and shadow maps need one eye from frame to frame. False before the
// first world-space frame.
bool heldEye(bool world, double eye[3]) {
    static double last[3] = {0, 0, 0};
    static bool known = false;
    if (world) {
        std::copy(eye, eye + 3, last);
        known = true;
        gWorldFrames.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    gRelativeFrames.fetch_add(1, std::memory_order_relaxed);
    if (known) std::copy(last, last + 3, eye);
    return true;
}

bool relativeCamera(const float vp[16], float inverse[16], float viewProjection[16],
                    double eye[3]) {
    double m[16];
    for (int i = 0; i < 16; i++) m[i] = vp[i];
    // Inverse by cofactors.
    double inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!std::isfinite(det) || std::fabs(det) < 1e-30) return false;
    for (double& v : inv) v /= det;
    // The eye: the one point clip space puts at w 0 with x and y 0.
    const double hx = inv[2], hy = inv[6], hz = inv[10], hw = inv[14];
    eye[0] = eye[1] = eye[2] = 0;
    if (std::fabs(hw) > 1e-12) {
        eye[0] = hx / hw;
        eye[1] = hy / hw;
        eye[2] = hz / hw;
    }
    // inverse' = T(-eye) inv; vp' = vp T(eye). Both written column-major.
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            double v = inv[r * 4 + c];
            if (r < 3) v -= eye[r] * inv[12 + c];
            inverse[c * 4 + r] = static_cast<float>(v);
        }
    }
    towardEye(vp, eye, viewProjection);
    return true;
}

// Kelir's scene band at the game's scene size and format, made on first use and whenever either
// changes. False when it cannot be made.
bool prepareBand(Device& d, Swapchain& sc, const lapis::Tap& scene) {
    auto& b = sc.band;
    if (b.width == scene.width && b.height == scene.height && b.format == scene.format) {
        return !b.failed;
    }
    waitFrames(d, sc);
    freeBand(d, sc);
    b.width = scene.width;
    b.height = scene.height;
    b.format = scene.format;
    const uint32_t w = scene.width;
    const uint32_t h = scene.height;
    const uint32_t hw = std::max(w / 2, 1u);
    const uint32_t hh = std::max(h / 2, 1u);
    const uint32_t qw = std::max(w / 4, 1u);
    const uint32_t qh = std::max(h / 4, 1u);
    const VkImageUsageFlags drawn = VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const Output* band = bandFor(d, scene.format);
    bool ok = band != nullptr && makeImage(d, &b.lin, kDistanceFormat, w, h, drawn) &&
            makeImage(d, &b.ao, VK_FORMAT_R8_UNORM, hw, hh, drawn);
    for (int j = 0; j < 2 && ok; j++) {
        ok = makeImage(d, &b.focusAt[j], kDistanceFormat, 1, 1, drawn) &&
                makeImage(d, &b.sunAt[j], kMotionFormat, 1, 1, drawn) &&
                makeImage(d, &b.fx[j], scene.format, w, h, drawn) &&
                makeImage(d, &b.bounced[j], kMotionFormat, qw, qh, drawn);
    }
    if (ok) {
        b.linFb = makeFramebuffer(d, d.distancePass, {b.lin.view}, w, h);
        b.aoFb = makeFramebuffer(d, d.r8Pass, {b.ao.view}, hw, hh);
        for (int j = 0; j < 2; j++) {
            b.focusFb[j] = makeFramebuffer(d, d.distancePass, {b.focusAt[j].view}, 1, 1);
            b.sunFb[j] = makeFramebuffer(d, d.motionPass, {b.sunAt[j].view}, 1, 1);
            b.fxFb[j] = makeFramebuffer(d, band->pass, {b.fx[j].view}, w, h);
            b.bouncedFb[j] = makeFramebuffer(d, d.motionPass, {b.bounced[j].view}, qw, qh);
        }
        ok = b.linFb && b.aoFb && b.focusFb[0] && b.focusFb[1] && b.fxFb[0] && b.fxFb[1] &&
                b.bouncedFb[0] && b.bouncedFb[1] && b.sunFb[0] && b.sunFb[1];
    }
    // Depth per frame in flight; occlusion; focus per parity; the three passes per frame in
    // flight; the resample; global illumination per frame in flight; the sun per frame in flight.
    constexpr uint32_t kSets = kFrames + 1 + 2 + 3 * kFrames + 1 + kFrames + kFrames;
    const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       kFrames + 1 + 6 + 24 * kFrames + 1 + 4 * kFrames +
                                               8 * kFrames};
    VkDescriptorPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = kSets;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    ok = ok && d.vk.CreateDescriptorPool(d.device, &pool, nullptr, &b.pool) == VK_SUCCESS;
    const auto allocate = [&](VkDescriptorSetLayout layout, VkDescriptorSet* set) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = b.pool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &layout;
        ok = ok && d.vk.AllocateDescriptorSets(d.device, &info, set) == VK_SUCCESS;
    };
    for (VkDescriptorSet& set : b.depthSets) allocate(d.upscaleLayout, &set);
    allocate(d.upscaleLayout, &b.aoSet);
    for (VkDescriptorSet& set : b.focusSets) allocate(d.hudLayout, &set);
    for (auto& sets : b.passSets) {
        for (VkDescriptorSet& set : sets) allocate(d.bandLayout, &set);
    }
    allocate(d.upscaleLayout, &b.resample);
    for (VkDescriptorSet& set : b.bouncedSets) allocate(d.giLayout, &set);
    for (VkDescriptorSet& set : b.sunSets) allocate(d.bandLayout, &set);
    if (ok) {
        const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        writeImages(d, b.aoSet, {{b.lin.view, read, d.nearest}});
        for (int p = 0; p < 2; p++) {
            writeImages(d, b.focusSets[p],
                        {{b.lin.view, read, d.nearest}, {b.focusAt[1 - p].view, read, d.nearest}});
        }
        writeImages(d, b.resample, {{b.fx[0].view, read, d.linear}});
    } else {
        ALOGW("luwes: no memory for Kelir's scene band at %ux%u", w, h);
        b.failed = true;
    }
    return ok;
}

// A view of one of the game's images, kept in `views` (kDepthViews of them); null when none can
// be made, or every view kept is still read by a frame in flight.
template <typename View>
VkImageView viewFor(Device& d, Swapchain& sc, View* views, const lapis::Tap& tap,
                    VkImageAspectFlags aspect) {
    auto* oldest = &views[0];
    for (int i = 0; i < kDepthViews; i++) {
        auto& v = views[i];
        if (v.id == tap.id && v.view != VK_NULL_HANDLE) {
            v.used = sc.realFrames;
            return v.view;
        }
        if (v.used < oldest->used) oldest = &v;
    }
    if (oldest->view != VK_NULL_HANDLE) {
        if (oldest->used + kFrames >= sc.realFrames) return VK_NULL_HANDLE;
        d.vk.DestroyImageView(d.device, oldest->view, nullptr);
    }
    *oldest = {};
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = tap.image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = tap.format;
    info.subresourceRange = {aspect, 0, 1, 0, 1};
    if (d.vk.CreateImageView(d.device, &info, nullptr, &oldest->view) != VK_SUCCESS) {
        oldest->view = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    oldest->id = tap.id;
    oldest->used = sc.realFrames;
    return oldest->view;
}

// A depth-aspect view of the game's depth image; null when none can be made, or every view kept
// is still read by a frame in flight.
VkImageView depthViewFor(Device& d, Swapchain& sc, const lapis::Tap& depth) {
    auto* oldest = &sc.band.depths[0];
    for (auto& v : sc.band.depths) {
        if (v.id == depth.id && v.view != VK_NULL_HANDLE) {
            v.used = sc.realFrames;
            return v.view;
        }
        if (v.used < oldest->used) oldest = &v;
    }
    if (oldest->view != VK_NULL_HANDLE) {
        if (oldest->used + kFrames >= sc.realFrames) return VK_NULL_HANDLE;
        d.vk.DestroyImageView(d.device, oldest->view, nullptr);
    }
    *oldest = {};
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = depth.image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = depth.format;
    info.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (d.vk.CreateImageView(d.device, &info, nullptr, &oldest->view) != VK_SUCCESS) {
        oldest->view = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    oldest->id = depth.id;
    oldest->used = sc.realFrames;
    return oldest->view;
}

VkImageView temporalViewFor(Device& d, Swapchain& sc, const lapis::Tap& tap);

// Kelir's scene band on the game's scene, into band.fx[0]: view distance from the game's depth
// when this frame kept it, then occlusion, fog and depth of field as the plan asks, with the depth
// edges in alpha. The resample set that reads the result; null when the band does not run: no
// camera, no depth seen yet, or nothing to do.
VkDescriptorSet recordBand(Device& d, Swapchain& sc, VkCommandBuffer cb, Frame& frame,
                           const lapis::Taps& taps, const kelir::Plan& plan, bool sharpen,
                           const std::function<void(const char*)>& stamp) {
    const lapis::Tap& s = taps.scene;
    if (!taps.camera.known || (plan.sceneCount == 0 && !sharpen) || !prepareBand(d, sc, s)) {
        return VK_NULL_HANDLE;
    }
    auto& b = sc.band;
    const Fns& vk = d.vk;
    const int slot = static_cast<int>(&frame - sc.frames);
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const uint32_t w = b.width;
    const uint32_t h = b.height;
    if (!b.primed) {
        // Every image the passes bind, readable before it was first drawn; no focus yet.
        const VkImageLayout dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        const VkImageSubresourceRange color = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkClearColorValue zero = {};
        for (Image* image : {&b.lin, &b.ao, &b.focusAt[0], &b.focusAt[1], &b.fx[0], &b.fx[1],
                             &b.bounced[0], &b.bounced[1], &b.sunAt[0], &b.sunAt[1]}) {
            imageBarrier(vk, cb, image->image, VK_IMAGE_LAYOUT_UNDEFINED, dst,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT);
            vk.CmdClearColorImage(cb, image->image, dst, &zero, 1, &color);
            imageBarrier(vk, cb, image->image, dst, read, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
        }
        b.primed = true;
    }
    const lapis::Tap& depth = taps.depth;
    const VkImageView depthView = depth.image != VK_NULL_HANDLE &&
                    (depth.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0
            ? depthViewFor(d, sc, depth)
            : VK_NULL_HANDLE;
    if (depthView != VK_NULL_HANDLE) {
        const VkImageLayout ro = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        const VkImageAspectFlags aspects = depthAspects(depth.format);
        writeImages(d, b.depthSets[slot], {{depthView, ro, d.nearest}});
        if (depth.layout != ro) {
            imageBarrier(vk, cb, depth.image, depth.layout, ro, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                         aspects);
        }
        const float params[16] = {static_cast<float>(w), static_cast<float>(h), taps.camera.c,
                                  taps.camera.b, taps.camera.gl ? 1.0f : 0.0f,
                                  taps.turned ? 1.0f : 0.0f};
        fullscreen(vk, cb, d.distancePass, b.linFb, w, h, d.depth, d.upscalePipe,
                   b.depthSets[slot], params);
        if (depth.layout != ro) {
            imageBarrier(vk, cb, depth.image, ro, depth.layout,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, aspects);
        }
        b.distance = true;
        stamp("depth");
    }
    const auto* source = b.distance ? sourceFor(d, sc, s) : nullptr;
    if (source == nullptr) return VK_NULL_HANDLE;

    float band[kelir::kSceneVec4s * 4];
    kelir::PackScene(plan, band);
    const bool ao = band[1] > 0.0f;
    const bool fog = band[6] > 0.0f;
    const bool dof = band[16] > 0.0f;
    const bool gi = band[22] > 0.0f;
    if (s.layout != read) {
        imageBarrier(vk, cb, s.image, s.layout, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    if (gi) {
        // Bounced light, carried along the game's velocity when it is one Luwes reads.
        const lapis::Tap& v = taps.velocity;
        const VkImageView velocity = v.image != VK_NULL_HANDLE &&
                        v.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32
                ? temporalViewFor(d, sc, v)
                : VK_NULL_HANDLE;
        if (velocity != VK_NULL_HANDLE && v.layout != read) {
            imageBarrier(vk, cb, v.image, v.layout, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        const int next = 1 - b.bouncedCur;
        writeImages(d, b.bouncedSets[slot],
                    {{b.lin.view, read, d.nearest},
                     {source->view, read, d.linear},
                     {b.bounced[b.bouncedCur].view, read, d.linear},
                     {velocity != VK_NULL_HANDLE ? velocity : b.lin.view, read, d.nearest}});
        const float params[16] = {static_cast<float>(w), static_cast<float>(h), 1.0f / w,
                                  1.0f / h, band[20], taps.camera.sx, taps.camera.sy,
                                  static_cast<float>(sc.realFrames % 1024),
                                  b.bouncedValid ? 1.0f : 0.0f,
                                  velocity != VK_NULL_HANDLE ? 1.0f : 0.0f,
                                  kPackedVelocityZero};
        fullscreen(vk, cb, d.motionPass, b.bouncedFb[next], std::max(w / 4, 1u),
                   std::max(h / 4, 1u), d.gi, d.giPipe, b.bouncedSets[slot], params);
        if (velocity != VK_NULL_HANDLE && v.layout != read) {
            imageBarrier(vk, cb, v.image, read, v.layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
        }
        b.bouncedCur = next;
    }
    b.bouncedValid = gi;
    if (ao) {
        const float params[16] = {static_cast<float>(w), static_cast<float>(h), 1.0f / w,
                                  1.0f / h, band[0], taps.camera.sx, taps.camera.sy};
        fullscreen(vk, cb, d.r8Pass, b.aoFb, std::max(w / 2, 1u), std::max(h / 2, 1u), d.ao,
                   d.upscalePipe, b.aoSet, params);
    }
    // Ray-traced kinds: the frame's tree from the game's own triangles, then rays.comp.
    float rayPack[kelir::kRaysVec4s * 4];
    b.rtValid = false;
    float inverse[16];
    float viewProjection[16];
    double eye[3];
    rays::Scene scene;
    const bool wantsRays = kelir::PackSceneRays(plan, rayPack);
    // Why a frame wanting rays traced none, for the log: 0 no camera, 1 no images, 2 no eye,
    // 3 no tree (no draws captured, or none fit).
    static std::atomic<uint32_t> untraced[4];
    int why = -1;
    if (wantsRays && !taps.camera.matrices) {
        why = 0;
    } else if (wantsRays && !prepareRays(d, sc)) {
        why = 1;
    } else if (wantsRays && !relativeCamera(taps.camera.vp, inverse, viewProjection, eye)) {
        why = 2;
    } else if (wantsRays && !heldEye(taps.camera.world, eye)) {
        why = 2;
    } else if (wantsRays && !rays::build(d.device, cb, inverse, eye, &scene)) {
        why = 3;
    }
    if (why >= 0) untraced[why].fetch_add(1, std::memory_order_relaxed);
    int frameMaps = -1;  // the game's shadow maps this frame's rays read, for the state log
    if (wantsRays && why < 0) {
        stamp("tree");
        const uint32_t hw = std::max(w / 2, 1u);
        const uint32_t hh = std::max(h / 2, 1u);
        const VkImageLayout general = VK_IMAGE_LAYOUT_GENERAL;
        if (!b.rtPrimed) {
            for (Image* image : {&b.rtVisibility, &b.rtReflection, &b.rtBounce, &b.rtHistory[0][0],
                                 &b.rtHistory[0][1], &b.rtHistory[0][2], &b.rtHistory[1][0],
                                 &b.rtHistory[1][1], &b.rtHistory[1][2]}) {
                imageBarrier(vk, cb, image->image, VK_IMAGE_LAYOUT_UNDEFINED, general,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
            }
            b.rtTracedLast = false;
            b.rtPrimed = true;
        }
        // Last frame's composite done with the images; this frame's distance and scene written.
        memoryBarrier(vk, cb,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        VkDescriptorImageInfo images[5] = {
                {d.nearest, b.lin.view, read},
                {d.linear, source->view, read},
                {VK_NULL_HANDLE, b.rtVisibility.view, general},
                {VK_NULL_HANDLE, b.rtReflection.view, general},
                {VK_NULL_HANDLE, b.rtBounce.view, general}};
        VkWriteDescriptorSetAccelerationStructureKHR tree = {};
        tree.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        tree.accelerationStructureCount = 1;
        tree.pAccelerationStructures = &scene.top;
        // The game's shadow maps, sampled where the game left them; those in a layout that cannot
        // be sampled go read-only around the rays and back.
        ShadowBlock shadowBlock = {};
        VkDescriptorImageInfo maps[lapis::kShadowMaps];
        struct Moved {
            const lapis::Tap* map;
            VkImageAspectFlags aspects;
        };
        std::vector<Moved> moved;
        const VkImageLayout depthRead = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        int mapCount = 0;
        for (int i = 0; i < taps.shadowCount && mapCount < lapis::kShadowMaps; i++) {
            const lapis::ShadowMap& m = taps.shadows[i];
            const VkImageView view = depthViewFor(d, sc, m.map);
            if (view == VK_NULL_HANDLE) continue;
            const bool inPlace = readOnlyDepth(m.map.layout);
            if (!inPlace) moved.push_back({&m.map, depthAspects(m.map.format)});
            maps[mapCount] = {d.nearest, view, inPlace ? m.map.layout : depthRead};
            towardEye(m.matrix, eye, shadowBlock.toMap[mapCount]);
            const float perMetre = std::sqrt(m.matrix[8] * m.matrix[8] + m.matrix[9] * m.matrix[9] +
                                             m.matrix[10] * m.matrix[10]);
            shadowBlock.info[mapCount][0] = m.reversed ? 1.0f : 0.0f;
            shadowBlock.info[mapCount][1] = m.flipsY ? 1.0f : 0.0f;
            shadowBlock.info[mapCount][2] = taps.camera.gl ? 0.5f * perMetre : perMetre;
            shadowBlock.info[mapCount][3] = 1.0f;
            std::copy(m.region, m.region + 4, shadowBlock.region[mapCount]);
            mapCount++;
        }
        for (int i = mapCount; i < lapis::kShadowMaps; i++) maps[i] = {d.nearest, b.lin.view, read};
        shadowBlock.counts[0] = static_cast<float>(mapCount);
        frameMaps = mapCount;
        // The game's depth writes into its maps, seen by the rays.
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        for (const Moved& m : moved) {
            imageBarrier(vk, cb, m.map->image, m.map->layout, depthRead,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, m.aspects);
        }
        // The G-buffer targets, where the game left them readable.
        VkDescriptorImageInfo gbuffer[lapis::kGBufferTargets];
        int gbufferCount = 0;
        lapis::ImageId gbufferIds[lapis::kGBufferTargets] = {};
        for (int i = 0; i < taps.gbufferCount; i++) {
            const lapis::Tap& t = taps.gbuffer[i];
            if (t.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                t.layout != VK_IMAGE_LAYOUT_GENERAL) {
                continue;
            }
            const VkImageView view =
                    viewFor(d, sc, b.gbufferViews, t, VK_IMAGE_ASPECT_COLOR_BIT);
            if (view == VK_NULL_HANDLE) continue;
            gbufferIds[gbufferCount] = t.id;
            gbuffer[gbufferCount++] = {d.linear, view, t.layout};
        }
        for (int i = gbufferCount; i < lapis::kGBufferTargets; i++) {
            gbuffer[i] = {d.linear, source->view, read};
        }
        // What the rays measured frames ago, of the targets they read then: the sun over the
        // sky, and how each target agrees with the scene's hue and the triangles' normals.
        RayStats earlier = {};
        lapis::ImageId earlierIds[lapis::kGBufferTargets] = {};
        VkDescriptorBufferInfo statsInfo = {};
        const bool haveStats = rays::frameStats(d.device, sizeof(RayStats), gbufferIds,
                                                sizeof(gbufferIds), &earlier, earlierIds,
                                                &statsInfo);
        if (earlier.edges >= 32) {
            const float measured = earlier.sunOverSky / 256.0f / earlier.edges;
            b.rtSunOverSky = 0.9f * b.rtSunOverSky + 0.1f * std::clamp(measured, 0.0f, 20.0f);
        }
        if (b.rtRoles == nullptr) b.rtRoles = framefx_lapis_roles_new();
        RolesSample samples[lapis::kGBufferTargets] = {};
        for (int i = 0; i < lapis::kGBufferTargets; i++) {
            RolesSample& s = samples[i];
            s.image = earlierIds[i];
            for (int c = 0; c < 2; c++) {
                const int32_t* v = earlier.chroma[i][c];
                const double f = kChromaFixed;
                s.chroma[c][0] = v[0];
                s.chroma[c][1] = v[1] / f;
                s.chroma[c][2] = v[2] / f;
                s.chroma[c][3] = v[3] / (f * f);
                s.chroma[c][4] = v[4] / (f * f);
                s.chroma[c][5] = v[5] / (f * f);
            }
            s.normalN = earlier.normal[i][5];
            for (int k = 0; k < 5; k++) s.normal[k] = earlier.normal[i][k] / 500.0 - s.normalN;
        }
        framefx_lapis_roles_frame(b.rtRoles, samples, lapis::kGBufferTargets);
        uint32_t albedoId = 0;
        uint32_t normalId = 0;
        uint8_t decoding = 0;
        double albedoAgree = 0;
        double normalAgree = 0;
        framefx_lapis_roles_picks(b.rtRoles, &albedoId, &albedoAgree, &normalId, &decoding,
                                  &normalAgree);
        for (int i = 0; i < gbufferCount; i++) {
            if (albedoId != 0 && gbufferIds[i] == albedoId) shadowBlock.roles[0] = i + 1.0f;
            if (normalId != 0 && gbufferIds[i] == normalId) {
                shadowBlock.roles[1] = i + 1.0f;
                shadowBlock.roles[2] = decoding;
            }
        }
        static std::atomic<uint32_t> rolesLog{0};
        if (rolesLog.fetch_add(1) % 600 == 599) {
            ALOGI("luwes: rays: sun over sky %.2f; G-buffer albedo #%u (%.2f), normals #%u "
                  "decoded %u (%.2f)",
                  b.rtSunOverSky, albedoId, albedoAgree, normalId, decoding, normalAgree);
        }
        shadowBlock.counts[1] = b.rtSunOverSky;
        shadowBlock.counts[2] = static_cast<float>(gbufferCount);
        cameraAxes(taps.camera.vp, shadowBlock.view);
        VkDescriptorBufferInfo shadowInfo = {};
        const bool haveBlock = rays::frameConstants(d.device, &shadowBlock, sizeof(shadowBlock),
                                                    &shadowInfo);
        VkWriteDescriptorSet writes[10] = {};
        for (uint32_t i = 0; i < 10; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = b.rtSets[slot];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            if (i == 2) {
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                writes[i].pNext = &tree;
            } else if (i == 6) {
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].descriptorCount = lapis::kShadowMaps;
                writes[i].pImageInfo = maps;
            } else if (i == 7) {
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writes[i].pBufferInfo = &shadowInfo;
            } else if (i == 8) {
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].descriptorCount = lapis::kGBufferTargets;
                writes[i].pImageInfo = gbuffer;
            } else if (i == 9) {
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &statsInfo;
            } else {
                writes[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                 : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[i].pImageInfo = &images[i < 2 ? i : i - 1];
            }
        }
        vk.UpdateDescriptorSets(d.device, haveBlock && haveStats ? 10 : 6, writes, 0, nullptr);
        // The alpha tests' textures; the slots past them hold the first, or the distance, never
        // sampled.
        VkDescriptorImageInfo textures[rays::kTextures];
        for (uint32_t i = 0; i < rays::kTextures; i++) {
            textures[i] = scene.textureCount == 0
                    ? images[0]
                    : VkDescriptorImageInfo{scene.sampler, scene.textures[std::min(i, scene.textureCount - 1)],
                                            read};
        }
        const VkDescriptorBufferInfo surfaceBuffers[2] = {scene.surfaces, scene.uvs};
        VkWriteDescriptorSet surfaceWrites[3] = {};
        for (uint32_t i = 0; i < 3; i++) {
            surfaceWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            surfaceWrites[i].dstSet = b.rtSets[slot];
            surfaceWrites[i].dstBinding = 10 + i;
            surfaceWrites[i].descriptorCount = i < 2 ? 1 : rays::kTextures;
            surfaceWrites[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                    : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (i < 2) {
                surfaceWrites[i].pBufferInfo = &surfaceBuffers[i];
            } else {
                surfaceWrites[i].pImageInfo = textures;
            }
        }
        vk.UpdateDescriptorSets(d.device, 3, surfaceWrites, 0, nullptr);
        RaysFrame f = {};
        std::copy(std::begin(viewProjection), std::end(viewProjection), f.viewProjection);
        std::copy(std::begin(inverse), std::end(inverse), f.inverse);
        if (taps.camera.light) {
            std::copy(std::begin(taps.camera.toLight), std::end(taps.camera.toLight), f.sun);
            f.sun[3] = 1.0f;
        }
        f.depth[0] = taps.camera.c;
        f.depth[1] = taps.camera.b;
        f.depth[2] = taps.camera.gl ? 1.0f : 0.0f;
        f.depth[3] = taps.camera.flipsY ? 1.0f : 0.0f;
        f.size[0] = static_cast<float>(hw);
        f.size[1] = static_cast<float>(hh);
        f.size[2] = 1.0f / hw;
        f.size[3] = 1.0f / hh;
        std::copy(rayPack, rayPack + 12, f.rays);
        // debug.framefx.rays_view shows one of rays.comp's inputs instead of its result;
        // debug.framefx.rays_only traces only some parts (rays.comp's noise.y), for timings.
        static std::atomic<uint32_t> viewAge{0};
        static std::atomic<int> view{0};
        static std::atomic<int> parts{15};
        if (viewAge.fetch_add(1) % 120 == 0) {
            view = android::base::GetIntProperty("debug.framefx.rays_view", 0);
            parts = android::base::GetIntProperty("debug.framefx.rays_only", 15) & 15;
        }
        f.sky[3] = static_cast<float>(view.load());
        f.noise[0] = static_cast<float>(b.rtFrame++ % 4096);
        f.noise[1] = static_cast<float>(parts.load());
        vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.rays);
        vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.raysPipe, 0, 1,
                                 &b.rtSets[slot], 0, nullptr);
        vk.CmdPushConstants(cb, d.raysPipe, VK_SHADER_STAGE_ALL, 0, sizeof(f), &f);
        // One pixel of each two by two a frame.
        vk.CmdDispatch(cb, ((hw + 1) / 2 + 7) / 8, ((hh + 1) / 2 + 7) / 8, 1);
        for (const Moved& m : moved) {
            imageBarrier(vk, cb, m.map->image, depthRead, m.map->layout,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, m.aspects);
        }
        stamp("rays");
        // Gathered into this frame's history from the last frame's, seen from where it stood.
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        b.rtHistoryAt = 1 - b.rtHistoryAt;
        const auto& now = b.rtHistory[b.rtHistoryAt];
        const auto& last = b.rtHistory[1 - b.rtHistoryAt];
        writeImages(d, b.rtHistorySets[slot],
                    {{b.lin.view, read, d.nearest},
                     {b.rtVisibility.view, general, d.nearest},
                     {b.rtReflection.view, general, d.nearest},
                     {b.rtBounce.view, general, d.nearest},
                     {last[0].view, general, d.linear},
                     {last[1].view, general, d.linear},
                     {last[2].view, general, d.linear},
                     {now[0].view, general, VK_NULL_HANDLE},
                     {now[1].view, general, VK_NULL_HANDLE},
                     {now[2].view, general, VK_NULL_HANDLE}},
                    7);
        HistoryFrame hf = {};
        std::copy(std::begin(inverse), std::end(inverse), hf.inverse);
        towardEye(b.rtLastVp, eye, hf.previous);
        std::copy(std::begin(f.depth), std::end(f.depth), hf.depth);
        std::copy(std::begin(f.size), std::end(f.size), hf.size);
        hf.control[0] = b.rtTracedLast ? 1.0f : 0.0f;
        hf.control[1] = 12.0f;
        hf.control[2] = f.noise[0];
        vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.history);
        vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.historyPipe, 0, 1,
                                 &b.rtHistorySets[slot], 0, nullptr);
        vk.CmdPushConstants(cb, d.historyPipe, VK_SHADER_STAGE_ALL, 0, sizeof(hf), &hf);
        vk.CmdDispatch(cb, (hw + 7) / 8, (hh + 7) / 8, 1);
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        std::copy(taps.camera.vp, taps.camera.vp + 16, b.rtLastVp);
        b.rtTracedLast = true;
        b.rtValid = true;
        b.rtUntraced = 0;
        stamp("history");
        static std::atomic<uint32_t> logged{0};
        if (logged.fetch_add(1) % 120 == 0) {
            ALOGI("luwes: rays on %u triangles at %ux%u, sun %s (%.2f %.2f %.2f); frames untraced: "
                  "%u no camera, %u no images, %u no eye, %u no tree; eye %.2f %.2f %.2f (%u "
                  "world-space frames, %u camera-relative)",
                  scene.triangles, hw, hh, taps.camera.light ? "from the shadow map" : "unknown",
                  f.sun[0], f.sun[1], f.sun[2], untraced[0].load(), untraced[1].load(),
                  untraced[2].load(), untraced[3].load(), eye[0], eye[1], eye[2],
                  gWorldFrames.load(), gRelativeFrames.load());
        }
    }
    if (dof && band[12] < 0.5f) {
        b.focus = 1 - b.focus;
        fullscreen(vk, cb, d.distancePass, b.focusFb[b.focus], 1, 1, d.focus, d.hudPipe,
                   b.focusSets[b.focus], nullptr);
    }
    const Output* pass = bandFor(d, s.format);
    float params[32] = {static_cast<float>(w), static_cast<float>(h), 1.0f / w, 1.0f / h};
    std::copy(band, band + kelir::kSceneVec4s * 4, params + 4);
    float lookParams[32] = {static_cast<float>(w), static_cast<float>(h), 1.0f / w, 1.0f / h};
    const bool look = kelir::PackSceneLook(plan, lookParams + 4);
    lookParams[28] = 2.0f;
    const Binding lin = {b.lin.view, read, d.nearest};
    const Binding occlusion = {b.ao.view, read, d.linear};
    const Binding focus = {b.focusAt[b.focus].view, read, d.nearest};
    const Binding bounced = {b.bounced[b.bouncedCur].view, read, d.linear};
    // The ray-traced kinds' results, or the distance where this frame traced none (unread).
    const VkImageLayout general = VK_IMAGE_LAYOUT_GENERAL;
    // A frame the rays missed (a hitch: the game's draws or camera not seen in time) shows the
    // last frames' gathered result, rather than none for a frame and history started over.
    if (!b.rtValid && wantsRays && b.rtTracedLast && ++b.rtUntraced <= kHeldUntraced) {
        b.rtValid = true;
    }
    if (!b.rtValid) b.rtTracedLast = false;
    // What the composite's rays rest on, logged where it changes from one frame to the next (a
    // frame's flash): traced or held, the light, its maps, and the light turning.
    if (wantsRays) {
        static int lastState = -1;
        static float lastLight[3] = {0, 0, 0};
        static uint32_t edges = 0;
        static int64_t edgeWindow = 0;
        const int state = (b.rtValid ? 1 : 0) | (b.rtUntraced > 0 ? 2 : 0) |
                (taps.camera.light ? 4 : 0) | (std::max(frameMaps, 0) << 3);
        const float* l = taps.camera.toLight;
        const bool turned = taps.camera.light &&
                l[0] * lastLight[0] + l[1] * lastLight[1] + l[2] * lastLight[2] < 0.995f;
        const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now - edgeWindow > 10'000'000'000) {
            edgeWindow = now;
            edges = 0;
        }
        if ((state != lastState || turned) && lastState >= 0 && edges++ < 20) {
            ALOGI("luwes: rays state: valid %d held %u light %d (%.2f %.2f %.2f) maps %d, was %d",
                  b.rtValid ? 1 : 0, b.rtUntraced, taps.camera.light ? 1 : 0, l[0], l[1], l[2],
                  frameMaps, lastState);
        }
        lastState = state;
        if (taps.camera.light) std::copy(l, l + 3, lastLight);
    }
    const auto& traced = b.rtHistory[b.rtHistoryAt];
    const Binding rtVisibility = b.rtValid ? Binding{traced[0].view, general, d.linear} : lin;
    const Binding rtReflection = b.rtValid ? Binding{traced[1].view, general, d.linear} : lin;
    const Binding rtBounce = b.rtValid ? Binding{traced[2].view, general, d.linear} : lin;
    // Where the sun stands, for the shafts: from the scene before any band pass changes it.
    const bool shafts = look && lookParams[4 + 11] > 0.0f;
    if (shafts) {
        b.sun = 1 - b.sun;
        writeImages(d, b.sunSets[slot],
                    {{source->view, read, d.linear}, lin, occlusion,
                     {b.sunAt[1 - b.sun].view, read, d.nearest}, bounced, rtVisibility,
                     rtReflection, rtBounce});
        fullscreen(vk, cb, d.motionPass, b.sunFb[b.sun], 1, 1, d.sun, d.bandPipe, b.sunSets[slot],
                   lookParams, sizeof(lookParams));
    }
    // Occlusion, bounced light and fog (or the edges alone), then the look, then depth of field;
    // each reads the one before, and the last writes fx[0].
    const bool first = ao || gi || fog || b.rtValid || (!dof && !look);
    const int passes = (first ? 1 : 0) + (look ? 1 : 0) + (dof ? 1 : 0);
    int done = 0;
    VkImageView from = source->view;
    const auto next = [&](const float* p, const Binding& oneByOne) {
        const int target = (passes - 1 - done) % 2;
        writeImages(d, b.passSets[slot][done],
                    {{from, read, d.linear}, lin, occlusion, oneByOne, bounced, rtVisibility,
                     rtReflection, rtBounce});
        fullscreen(vk, cb, pass->pass, b.fxFb[target], w, h, pass->pipeline, d.bandPipe,
                   b.passSets[slot][done], p, sizeof(params));
        from = b.fx[target].view;
        done++;
    };
    params[29] = b.rtValid ? 1.0f : 0.0f;
    if (first) {
        params[28] = 0.0f;
        next(params, focus);
    }
    if (look) next(lookParams, shafts ? Binding{b.sunAt[b.sun].view, read, d.nearest} : focus);
    if (dof) {
        params[28] = 1.0f;
        next(params, focus);
    }
    if (s.layout != read) {
        imageBarrier(vk, cb, s.image, read, s.layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
    }
    stamp("band");
    return b.resample;
}

// SGSR 2's images for a jittered scene of `width` x `height`, made on first use and whenever its
// size changes. False when they cannot be made.
bool prepareTemporal(Device& d, Swapchain& sc, uint32_t width, uint32_t height) {
    auto& t = sc.temporal;
    if (t.width == width && t.height == height) return !t.failed;
    waitFrames(d, sc);
    freeTemporal(d, sc);
    t.width = width;
    t.height = height;
    const uint32_t w = sc.extent.width;
    const uint32_t h = sc.extent.height;
    const VkImageUsageFlags drawn = VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const Output* up = temporalUpFor(d, sc.format);
    bool ok = up != nullptr && makeImage(d, &t.motion, kMotionFormat, width, height, drawn);
    if (ok) {
        t.motionFb = makeFramebuffer(d, d.motionPass, {t.motion.view}, width, height);
        ok = t.motionFb != VK_NULL_HANDLE;
    }
    const Image* targets[3] = {&sc.afme.sharp[0], &sc.afme.sharp[1], &sc.afme.sharpRaw};
    for (int k = 0; k < 2 && ok; k++) {
        ok = makeImage(d, &t.history[k], kHistoryFormat, w, h, drawn);
        for (int j = 0; j < 3 && ok; j++) {
            t.upFb[k][j] = makeFramebuffer(d, up->pass, {t.history[k].view, targets[j]->view}, w, h);
            ok = t.upFb[k][j] != VK_NULL_HANDLE;
        }
    }
    const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7 * kFrames};
    VkDescriptorPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 2 * kFrames;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    ok = ok && d.vk.CreateDescriptorPool(d.device, &pool, nullptr, &t.pool) == VK_SUCCESS;
    for (int f = 0; f < kFrames && ok; f++) {
        for (auto [layout, set] : {std::pair{d.temporalLayout, &t.convertSets[f]},
                                   std::pair{d.temporalUpLayout, &t.upSets[f]}}) {
            VkDescriptorSetAllocateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            info.descriptorPool = t.pool;
            info.descriptorSetCount = 1;
            info.pSetLayouts = &layout;
            ok = ok && d.vk.AllocateDescriptorSets(d.device, &info, set) == VK_SUCCESS;
        }
    }
    if (!ok) {
        ALOGW("luwes: no memory for SGSR 2 at %ux%u > %ux%u", width, height, w, h);
        t.failed = true;
    }
    return ok;
}

// A colour view of one of the game's images SGSR 2 reads; null when none can be made, or every
// view kept is still read by a frame in flight.
VkImageView temporalViewFor(Device& d, Swapchain& sc, const lapis::Tap& tap) {
    auto* oldest = &sc.temporal.views[0];
    for (auto& v : sc.temporal.views) {
        if (v.id == tap.id && v.view != VK_NULL_HANDLE) {
            v.used = sc.realFrames;
            return v.view;
        }
        if (v.used < oldest->used) oldest = &v;
    }
    if (oldest->view != VK_NULL_HANDLE) {
        if (oldest->used + kFrames >= sc.realFrames) return VK_NULL_HANDLE;
        d.vk.DestroyImageView(d.device, oldest->view, nullptr);
    }
    *oldest = {};
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = tap.image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = tap.format;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (d.vk.CreateImageView(d.device, &info, nullptr, &oldest->view) != VK_SUCCESS) {
        oldest->view = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    oldest->id = tap.id;
    oldest->used = sc.realFrames;
    return oldest->view;
}

// Whether this frame's taps carry what SGSR 2 reads: the game's jittered scene and its resolve,
// a velocity buffer in an encoding Luwes reads, readable depth and the frame's camera.
bool temporalReady(const lapis::Taps& taps, VkExtent2D extent) {
    return taps.jittered.image != VK_NULL_HANDLE && taps.resolved.image != VK_NULL_HANDLE &&
            taps.velocity.image != VK_NULL_HANDLE &&
            taps.velocity.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 &&
            taps.depth.image != VK_NULL_HANDLE &&
            (taps.depth.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0 && taps.camera.matrices &&
            taps.jittered.width == taps.depth.width && taps.jittered.height == taps.depth.height &&
            Smaller(taps.jittered, extent);
}

// SGSR 2 on the game's jittered scene into the accumulation, and with its detail carried onto
// `base` (the game's scene, or Kelir's band over it) into sharp[target], 2 for sharpRaw. False
// when it could not be recorded; nothing was then.
bool recordTemporal(Device& d, Swapchain& sc, VkCommandBuffer cb, Frame& frame,
                    const lapis::Taps& taps, VkImageView base, int target) {
    const lapis::Tap& color = taps.jittered;
    if (!prepareTemporal(d, sc, color.width, color.height)) return false;
    auto& t = sc.temporal;
    const VkImageView colorView = temporalViewFor(d, sc, color);
    const VkImageView resolvedView = temporalViewFor(d, sc, taps.resolved);
    const VkImageView velocityView = temporalViewFor(d, sc, taps.velocity);
    const VkImageView depthView = depthViewFor(d, sc, taps.depth);
    const Output* up = temporalUpFor(d, sc.format);
    if (colorView == VK_NULL_HANDLE || resolvedView == VK_NULL_HANDLE ||
        velocityView == VK_NULL_HANDLE || depthView == VK_NULL_HANDLE || up == nullptr) {
        return false;
    }
    const Fns& vk = d.vk;
    const int slot = static_cast<int>(&frame - sc.frames);
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImageLayout depthRead = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    if (!t.primed) {
        const VkImageLayout dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkClearColorValue zero = {};
        for (Image& image : t.history) {
            imageBarrier(vk, cb, image.image, VK_IMAGE_LAYOUT_UNDEFINED, dst,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT);
            vk.CmdClearColorImage(cb, image.image, dst, &zero, 1, &range);
            imageBarrier(vk, cb, image.image, dst, read, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
        }
        t.primed = true;
        t.valid = false;
    }
    // The game's images into readable layouts for the two passes, and back after.
    struct Held {
        const lapis::Tap* tap;
        VkImageLayout want;
        VkImageAspectFlags aspects;
    };
    const Held held[] = {{&color, read, VK_IMAGE_ASPECT_COLOR_BIT},
                         {&taps.resolved, read, VK_IMAGE_ASPECT_COLOR_BIT},
                         {&taps.velocity, read, VK_IMAGE_ASPECT_COLOR_BIT},
                         {&taps.depth, depthRead, depthAspects(taps.depth.format)}};
    for (const Held& h : held) {
        if (h.tap->layout == h.want) continue;
        imageBarrier(vk, cb, h.tap->image, h.tap->layout, h.want, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                     0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                     h.aspects);
    }
    const auto W = static_cast<float>(color.width);
    const auto H = static_cast<float>(color.height);
    const lapis::Camera& camera = taps.camera;
    const float tanHalfFovH = camera.known && camera.sx != 0.0f ? 1.0f / std::fabs(camera.sx) : 1.0f;
    writeImages(d, t.convertSets[slot],
                {{depthView, depthRead, d.nearest}, {velocityView, read, d.nearest}});
    const float convert[16] = {W,           H, 1.0f / W, 1.0f / H, tanHalfFovH,
                               taps.reversed ? 1.0f : 0.0f, 0.0f, 0.0f, kPackedVelocityZero};
    fullscreen(vk, cb, d.motionPass, t.motionFb, color.width, color.height, d.temporal,
               d.temporalPipe, t.convertSets[slot], convert);

    // The camera without its jitter, to tell whether it moved.
    float still[16];
    std::copy(std::begin(camera.vp), std::end(camera.vp), still);
    for (int c = 0; c < 4; c++) {
        still[c] -= camera.ndcJitterX * camera.vp[12 + c];
        still[4 + c] -= camera.ndcJitterY * camera.vp[12 + c];
    }
    float moved = 0.0f;
    for (int i = 0; i < 16; i++) moved += std::fabs(still[i] - t.lastVp[i]);
    std::copy(std::begin(still), std::end(still), t.lastVp);
    const auto ow = static_cast<float>(sc.extent.width);
    const auto oh = static_cast<float>(sc.extent.height);
    const int next = 1 - t.cur;
    writeImages(d, t.upSets[slot],
                {{t.history[t.cur].view, read, d.linear},
                 {t.motion.view, read, d.linear},
                 {colorView, read, d.nearest},
                 {base, read, d.linear},
                 {resolvedView, read, d.linear}});
    // The jitter in the game's texels: NDC x runs along them, NDC y with the viewport's sign.
    const float params[32] = {
            W, H, 1.0f / W, 1.0f / H,
            ow, oh, 1.0f / ow, 1.0f / oh,
            camera.ndcJitterX * W * 0.5f,
            camera.ndcJitterY * H * 0.5f * (camera.flipsY ? -1.0f : 1.0f),
            (sc.afme.orientation & 2) ? 1.0f : 0.0f,
            (sc.afme.orientation & 1) ? 1.0f : 0.0f,
            ow / W,
            std::min(20.0f, std::pow(ow * oh / (W * H), 3.0f)),
            moved < 1e-5f ? kMinLerp : 0.0f,
            t.valid ? 0.0f : 1.0f,
            kDetailGamma, kDetailMax, kSampleWeight,
    };
    fullscreen(vk, cb, up->pass, t.upFb[next][target], sc.extent.width, sc.extent.height,
               up->pipeline, d.temporalUpPipe, t.upSets[slot], params, sizeof(params));
    for (const Held& h : held) {
        if (h.tap->layout == h.want) continue;
        imageBarrier(vk, cb, h.tap->image, h.want, h.tap->layout,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                     0, h.aspects);
    }
    t.cur = next;
    t.valid = true;
    if (t.logged != color.id) {
        t.logged = color.id;
        ALOGI("luwes: SGSR 2 on the game's jittered scene %ux%u > %ux%u, jitter (%.3f, %.3f) px",
              color.width, color.height, sc.extent.width, sc.extent.height, params[8], params[9]);
    }
    return true;
}

// The game's scene at swapchain size into `sharp[cur]`: SGSR 1 when `sgsr`, else resampled as
// the game's own stretch does, then through Kelir's stage A when that runs. `band`: the scene
// band's result, read in place of the game's scene. False when it could not be recorded.
bool recordSharp(Device& d, Swapchain& sc, VkCommandBuffer cb, Frame& frame,
                 const lapis::Tap& scene, VkDescriptorSet band, bool sgsr,
                 const lapis::Taps* temporal, const kelir::vulkan::Stage* stage,
                 const std::function<void(const char*)>& stamp) {
    const Output* up = upscaleFor(d, sc.format);
    const auto* source = up != nullptr && band == VK_NULL_HANDLE ? sourceFor(d, sc, scene) : nullptr;
    if (up == nullptr || (band == VK_NULL_HANDLE && source == nullptr)) {
        sc.afme.layerMiss = "no pass or view for the scene";
        return false;
    }
    const VkDescriptorSet set = band != VK_NULL_HANDLE ? band : source->set;
    const Fns& vk = d.vk;
    const int cur = sc.current;
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const bool game = band == VK_NULL_HANDLE && scene.layout != read;
    if (game) {
        imageBarrier(vk, cb, scene.image, scene.layout, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    const bool graded = stage != nullptr && stage->stageA != VK_NULL_HANDLE;
    const float params[16] = {
            1.0f / static_cast<float>(scene.width),
            1.0f / static_cast<float>(scene.height),
            static_cast<float>(scene.width),
            static_cast<float>(scene.height),
            static_cast<float>(sc.extent.width),
            static_cast<float>(sc.extent.height),
            (sc.afme.orientation & 2) ? 1.0f : 0.0f,
            (sc.afme.orientation & 1) ? 1.0f : 0.0f,
            sgsr ? 0.0f : 1.0f,
            Engine::get().config().upscale.edgeSharpness(),
    };
    // SGSR 2 where the game's own anti-aliasing shows its jittered scene; SGSR 1 otherwise.
    const VkImageView base = band != VK_NULL_HANDLE ? sc.band.fx[0].view : source->view;
    sc.afme.method[cur] = temporal != nullptr &&
                    recordTemporal(d, sc, cb, frame, *temporal, base, graded ? 2 : cur)
            ? 2
            : sgsr ? 1 : 0;
    if (sc.afme.method[cur] != 2) {
        fullscreen(vk, cb, up->pass, graded ? sc.afme.sharpRawFb : sc.afme.sharpFb[cur],
                   sc.extent.width, sc.extent.height, up->pipeline, d.upscalePipe, set, params);
    }
    if (game) {
        imageBarrier(vk, cb, scene.image, read, scene.layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
    }
    stamp(sc.afme.method[cur] == 2 ? "sgsr2" : sgsr ? "upscale" : "resample");
    if (!graded) return true;
    const bool ok = kelir::vulkan::recordStageAFrom(*stage, cb, sc.afme.sharpRaw.view,
                                                    sc.afme.sharp[cur].image,
                                                    sc.afme.sharp[cur].view,
                                                    band != VK_NULL_HANDLE);
    stamp("grade");
    if (!ok) sc.afme.layerMiss = "Kelir could not grade the scene";
    return ok;
}

// What the real frame's HUD puts over the game's own stretch of its scene, from the game's image
// as rendered, into `hud[cur]` and `mask[cur]`.
bool recordHud(Device& d, Swapchain& sc, VkCommandBuffer cb, Frame& frame, uint32_t appIndex,
               const lapis::Tap& scene) {
    const auto* source = sourceFor(d, sc, scene);
    const VkDescriptorSet set = sc.afme.hudSets[&frame - sc.frames];
    if (source == nullptr || set == VK_NULL_HANDLE) {
        sc.afme.layerMiss = "no view for the HUD layer";
        return false;
    }
    const Fns& vk = d.vk;
    const int cur = sc.current;
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImage app = appImage(sc, appIndex);
    writeImages(d, set,
                {{appView(sc, appIndex), read, sc.resized() ? d.linear : d.nearest},
                 {source->view, read, d.linear},
                 {sc.afme.matte[1 - cur].view, read, d.nearest}});
    imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, read,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_SHADER_READ_BIT);
    if (scene.layout != read) {
        imageBarrier(vk, cb, scene.image, scene.layout, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    const bool known = std::find(std::begin(sc.afme.known), std::end(sc.afme.known), scene.id) !=
            std::end(sc.afme.known);
    if (!known) {
        sc.afme.known[sc.afme.nextKnown] = scene.id;
        sc.afme.nextKnown = (sc.afme.nextKnown + 1) % kScenesKnown;
    }
    const float params[16] = {
            static_cast<float>(sc.extent.width),
            static_cast<float>(sc.extent.height),
            (sc.afme.orientation & 2) ? 1.0f : 0.0f,
            (sc.afme.orientation & 1) ? 1.0f : 0.0f,
            kHudFloor,
            kHudOpaque,
            0.0f,
            0.0f,
            // The previous real frame ran this pass too, into the other matte.
            sc.afme.hasSharp[1 - cur] && known ? 1.0f : 0.0f,
            kMatteDecay,
            kMatteTrust,
            kMatteBreak,
            sc.resized() ? static_cast<float>(sc.game.extent.width) : 0.0f,
            sc.resized() ? static_cast<float>(sc.game.extent.height) : 0.0f,
    };
    fullscreen(vk, cb, d.hudPass, sc.afme.hudFb[cur], sc.extent.width, sc.extent.height, d.hud,
               d.hudPipe, set, params);
    if (scene.layout != read) {
        imageBarrier(vk, cb, scene.image, read, scene.layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
    }
    imageBarrier(vk, cb, app, read, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    return true;
}

// The held frame composed from the scene layer: the game's 3D with the real frame's HUD over it,
// or the 3D alone in photo mode. Generation and the real frame's present read it from there.
bool recordHeld(Device& d, Swapchain& sc, VkCommandBuffer cb) {
    const Output* up = upscaleFor(d, sc.format);
    const Output* out = outputFor(d, sc.format);
    if (up == nullptr || out == nullptr) {
        sc.afme.layerMiss = "no pass for the held frame";
        return false;
    }
    const int cur = sc.current;
    const float params[16] = {
            static_cast<float>(sc.extent.width),
            static_cast<float>(sc.extent.height),
            static_cast<float>(sc.alignedWidth),
            static_cast<float>(sc.alignedHeight),
            0.0f,
            static_cast<float>(kComposeLayer),
            kFieldScale,
            0.0f,
            0.0f,
            sc.hideHud ? 1.0f : 0.0f,
    };
    fullscreen(d.vk, cb, up->pass, sc.afme.heldFb[cur], sc.extent.width, sc.extent.height,
               out->pipeline, d.outputPipe, sc.afme.upReal[cur], params);
    return true;
}

// The game's scene in the swapchain's axes, as a tap; false when it cannot be made.
bool recordUpright(Device& d, Swapchain& sc, VkCommandBuffer cb, const lapis::Tap& scene,
                   lapis::Tap* out) {
    const Output* pass = transposeFor(d, scene.format);
    const auto* source = sourceFor(d, sc, scene);
    if (pass == nullptr || source == nullptr) return false;
    const VkExtent2D size = sc.resized() ? sc.game.extent : sc.extent;
    if (sc.afme.uprightFormat != scene.format) {
        // Frames still in flight sampled nothing of it: every use is recorded from this image.
        d.vk.DeviceWaitIdle(d.device);
        freeUpright(d, sc);
        if (!makeImage(d, &sc.afme.upright, scene.format, size.width, size.height,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            return false;
        }
        sc.afme.uprightFb =
                makeFramebuffer(d, pass->pass, {sc.afme.upright.view}, size.width, size.height);
        if (sc.afme.uprightFb == VK_NULL_HANDLE) {
            freeImage(d, sc.afme.upright);
            return false;
        }
        sc.afme.uprightFormat = scene.format;
    }
    const Fns& vk = d.vk;
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (scene.layout != read) {
        imageBarrier(vk, cb, scene.image, scene.layout, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    const float params[16] = {static_cast<float>(size.width), static_cast<float>(size.height)};
    fullscreen(vk, cb, pass->pass, sc.afme.uprightFb, size.width, size.height, pass->pipeline,
               d.upscalePipe, source->set, params);
    if (scene.layout != read) {
        imageBarrier(vk, cb, scene.image, read, scene.layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
    }
    *out = {sc.afme.upright.image, scene.format, size.width, size.height, read,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, kUprightId};
    return true;
}

// The part of the scene a game at a dynamic resolution drew, scaled to the scene's whole size: the
// size stays put while the game's resolution moves, and what follows reads a whole image.
bool formatUsable(const Device& d, VkFormat format);

bool recordDrawn(Device& d, Swapchain& sc, VkCommandBuffer cb, const lapis::Tap& scene,
                 lapis::Tap* out) {
    if ((scene.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0 || !formatUsable(d, scene.format)) {
        return false;
    }
    auto& a = sc.afme;
    if (a.drawnFormat != scene.format || a.drawnSize.width != scene.width ||
        a.drawnSize.height != scene.height) {
        // Frames still in flight sampled nothing of it: every use is recorded from this image.
        d.vk.DeviceWaitIdle(d.device);
        // A view of the image this replaces would sample freed memory.
        for (auto& source : a.sources) {
            if (source.id != kDrawnId) continue;
            if (source.view != VK_NULL_HANDLE) d.vk.DestroyImageView(d.device, source.view, nullptr);
            source.view = VK_NULL_HANDLE;
            source.id = 0;
        }
        freeImage(d, a.drawn);
        a.drawnFormat = VK_FORMAT_UNDEFINED;
        if (!makeImage(d, &a.drawn, scene.format, scene.width, scene.height,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            return false;
        }
        a.drawnFormat = scene.format;
        a.drawnSize = {scene.width, scene.height};
    }
    const Fns& vk = d.vk;
    const VkImageLayout src = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageLayout dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageBarrier(vk, cb, scene.image, scene.layout, src, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    // Its last contents are not needed: all of it is drawn over.
    imageBarrier(vk, cb, a.drawn.image, VK_IMAGE_LAYOUT_UNDEFINED, dst,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_WRITE_BIT);
    const VkRect2D& r = scene.region;
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[0] = {r.offset.x, r.offset.y, 0};
    blit.srcOffsets[1] = {r.offset.x + static_cast<int32_t>(r.extent.width),
                          r.offset.y + static_cast<int32_t>(r.extent.height), 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(scene.width), static_cast<int32_t>(scene.height), 1};
    vk.CmdBlitImage(cb, scene.image, src, a.drawn.image, dst, 1, &blit, VK_FILTER_LINEAR);
    imageBarrier(vk, cb, a.drawn.image, dst, read, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    imageBarrier(vk, cb, scene.image, src, scene.layout, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
    *out = scene;
    out->image = a.drawn.image;
    out->layout = read;
    out->usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    out->id = kDrawnId;
    out->region = {};
    return true;
}

// How the scene layer fared, in a survey's shot.
void reportLayer(const Swapchain& sc, const lapis::Taps& taps, bool wanted, bool solo) {
    if (survey::shooting()) {
        const uint64_t* e = sc.afme.voteErr;
        ALOGI("survey   layer: wanted %d solo %d declined %d orientation %d scene #%u %ux%u%s%s%s%s, "
              "%s; votes %u, last per flip %llu %llu %llu %llu",
              wanted, solo, static_cast<int>(sc.declined), sc.afme.orientation, taps.scene.id,
              taps.scene.width, taps.scene.height, taps.held ? " held" : "",
              taps.gameGenerates ? " game-generates" : "", taps.scene.upright ? " upright" : "",
              taps.scene.region.extent.width == 0 ? ""
                      : android::base::StringPrintf(" drawn %ux%u@%d,%d",
                                                    taps.scene.region.extent.width,
                                                    taps.scene.region.extent.height,
                                                    taps.scene.region.offset.x,
                                                    taps.scene.region.offset.y).c_str(),
              sc.afme.layerMiss != nullptr ? sc.afme.layerMiss : "composed", sc.afme.votes,
              static_cast<unsigned long long>(e[0]), static_cast<unsigned long long>(e[1]),
              static_cast<unsigned long long>(e[2]), static_cast<unsigned long long>(e[3]));
    }
}

// Thumbnails of the real frame and of the game's scene as stored, for decideOrientation.
void recordProbe(const Fns& vk, VkCommandBuffer cb, Swapchain& sc, VkImage app,
                 const lapis::Tap& scene) {
    const VkImageLayout src = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageLayout dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, src,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT);
    imageBarrier(vk, cb, scene.image, scene.layout, src, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    for (const Image& t : sc.afme.thumb) {
        imageBarrier(vk, cb, t.image, VK_IMAGE_LAYOUT_UNDEFINED, dst,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    const struct {
        VkImage from;
        uint32_t width, height;
    } sources[2] = {{app, appExtent(sc).width, appExtent(sc).height},
                    {scene.image, scene.width, scene.height}};
    for (int j = 0; j < 2; j++) {
        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstSubresource = blit.srcSubresource;
        blit.srcOffsets[1] = {static_cast<int32_t>(sources[j].width),
                              static_cast<int32_t>(sources[j].height), 1};
        blit.dstOffsets[1] = {static_cast<int32_t>(kThumbWidth), static_cast<int32_t>(kThumbHeight),
                              1};
        vk.CmdBlitImage(cb, sources[j].from, src, sc.afme.thumb[j].image, dst, 1, &blit,
                        VK_FILTER_LINEAR);
    }
    imageBarrier(vk, cb, app, src, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    imageBarrier(vk, cb, scene.image, src, scene.layout, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
    for (int j = 0; j < 2; j++) {
        imageBarrier(vk, cb, sc.afme.thumb[j].image, dst, src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy = {};
        copy.bufferOffset = j * kThumbBytes;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kThumbWidth, kThumbHeight, 1};
        vk.CmdCopyImageToBuffer(cb, sc.afme.thumb[j].image, src, sc.afme.probe, 1, &copy);
    }
    memoryBarrier(vk, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
}

// The game's frame, drawn at its own size, resampled into `fb` at the swapchain's: SGSR 1 when it
// grows, Lanczos 2 when it shrinks.
void recordResample(Device& d, Swapchain& sc, VkCommandBuffer cb, uint32_t appIndex,
                    VkFramebuffer fb) {
    const Output* up = upscaleFor(d, sc.format);
    const VkImage app = appImage(sc, appIndex);
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageBarrier(d.vk, cb, app, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, read,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_SHADER_READ_BIT);
    const VkExtent2D e = sc.game.extent;
    const float params[16] = {
            1.0f / static_cast<float>(e.width),
            1.0f / static_cast<float>(e.height),
            static_cast<float>(e.width),
            static_cast<float>(e.height),
            static_cast<float>(sc.extent.width),
            static_cast<float>(sc.extent.height),
            0.0f,
            0.0f,
            e.width > sc.extent.width ? 2.0f : 0.0f,
            Engine::get().config().upscale.edgeSharpness(),
    };
    fullscreen(d.vk, cb, up->pass, fb, sc.extent.width, sc.extent.height, up->pipeline,
               d.upscalePipe, sc.game.sets[appIndex], params);
    imageBarrier(d.vk, cb, app, read, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
}

// The game's depth, velocity and camera as this frame tapped them, raw, for offline study.
void recordGBufferCapture(Device& d, Swapchain& sc, VkCommandBuffer cb, Frame& frame,
                          const lapis::Taps& taps) {
    const Fns& vk = d.vk;
    const auto dump = [&](const lapis::Tap& tap, VkImageAspectFlags aspect, VkImageAspectFlags all,
                          const char* tag) {
        if (tap.image == VK_NULL_HANDLE || (tap.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
            return;
        }
        const VkImageLayout src = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        imageBarrier(vk, cb, tap.image, tap.layout, src, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                     VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT, all);
        readback(d, sc, frame, cb, tap.image, src, tap.format, tap.width, tap.height, tag, aspect);
        imageBarrier(vk, cb, tap.image, src, tap.layout, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, all);
    };
    dump(taps.depth, VK_IMAGE_ASPECT_DEPTH_BIT, depthAspects(taps.depth.format), "depth");
    dump(taps.velocity, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_COLOR_BIT, "velocity");
    const lapis::Camera& c = taps.camera;
    std::string line = android::base::StringPrintf(
            "camera %03u known %d gl %d reversed %d turned %d c %.9g b %.9g sx %.6g sy %.6g "
            "jx %.6g jy %.6g flipsY %d world %d depth %ux%u fmt %d velocity %ux%u fmt %d",
            sc.captureFrame, c.known, c.gl, taps.reversed, taps.turned, c.c, c.b, c.sx, c.sy, c.jx,
            c.jy, c.flipsY, c.world, taps.depth.width, taps.depth.height, taps.depth.format,
            taps.velocity.width, taps.velocity.height, taps.velocity.format);
    if (c.matrices) {
        line += android::base::StringPrintf(" ndcJitter %.6g %.6g vp", c.ndcJitterX, c.ndcJitterY);
        for (float v : c.vp) line += android::base::StringPrintf(" %.9g", v);
    }
    sc.capture.note(line);
}

// The game's depth into AFME's slot for this frame, when the G-buffer feed trusts it.
void recordAfmeDepth(Device& d, Swapchain& sc, VkCommandBuffer cb, const lapis::Taps& taps) {
    const GBufferOutput& g = sc.gbuffer.last();
    const lapis::Tap& depth = taps.depth;
    Image& slot = sc.afme.depth[sc.current];
    if (!sc.gbuffer.feedsAfme() || sc.afme.engine == nullptr || slot.image == VK_NULL_HANDLE ||
        g.depth < kGBufferTrusted || taps.held || taps.turned || depth.image == VK_NULL_HANDLE ||
        depth.format != VK_FORMAT_D24_UNORM_S8_UINT ||
        (depth.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
        sc.afme.depthFed[sc.current] = false;
        return;
    }
    const Fns& vk = d.vk;
    const VkImageAspectFlags ds = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    const VkImageLayout ro = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    const VkImageLayout src = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageLayout dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageBarrier(vk, cb, depth.image, depth.layout, src, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                 VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT, ds);
    imageBarrier(vk, cb, slot.image, ro, dst, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, ds);
    const int32_t w = static_cast<int32_t>(depth.width);
    const int32_t h = static_cast<int32_t>(depth.height);
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[1] = {w, h, 1};
    // The scene AFME reads is upright; depth turns the way the colour did.
    const bool flipY = (sc.afme.orientation & 1) != 0;
    const bool flipX = (sc.afme.orientation & 2) != 0;
    const auto dw = static_cast<int32_t>(sc.extent.width);
    const auto dh = static_cast<int32_t>(sc.extent.height);
    blit.dstOffsets[0] = {flipX ? dw : 0, flipY ? dh : 0, 0};
    blit.dstOffsets[1] = {flipX ? 0 : dw, flipY ? 0 : dh, 1};
    vk.CmdBlitImage(cb, depth.image, src, slot.image, dst, 1, &blit, VK_FILTER_NEAREST);
    imageBarrier(vk, cb, slot.image, dst, ro, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_SHADER_READ_BIT, ds);
    imageBarrier(vk, cb, depth.image, src, depth.layout, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, ds);
    sc.afme.depthFed[sc.current] = true;
}

// Scores this frame's taps; Luwes and AFME use the result from the next analysis on.
void updateGBuffer(Swapchain& sc, const lapis::Taps& taps, int64_t now) {
    GBufferInput in;
    const lapis::Camera& c = taps.camera;
    in.sceneWidth = taps.scene.width;
    in.sceneHeight = taps.scene.height;
    in.hasScene = taps.scene.image != VK_NULL_HANDLE;
    in.hasDepth = taps.depth.image != VK_NULL_HANDLE;
    in.depthWidth = taps.depth.width;
    in.depthHeight = taps.depth.height;
    in.depthFormat = static_cast<uint32_t>(taps.depth.format);
    in.depthSampled = (taps.depth.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
    in.reversed = taps.reversed;
    in.hasVelocity = taps.velocity.image != VK_NULL_HANDLE;
    in.velocityWidth = taps.velocity.width;
    in.velocityHeight = taps.velocity.height;
    in.velocityFormat = static_cast<uint32_t>(taps.velocity.format);
    in.temporal = taps.jittered.image != VK_NULL_HANDLE && taps.resolved.image != VK_NULL_HANDLE;
    in.cameraKnown = c.known;
    in.cameraMatrices = c.matrices && !taps.turned;
    in.cameraWorld = c.world;
    in.held = taps.held;
    in.gameGenerates = taps.gameGenerates;
    in.sx = c.sx;
    in.sy = c.sy;
    in.c = c.c;
    in.b = c.b;
    std::copy(c.vp, c.vp + 16, in.vp);
    in.jitter[0] = c.ndcJitterX;
    in.jitter[1] = c.ndcJitterY;
    in.signature = GBufferSignature(in);
    sc.gbuffer.update(in, c.gl, AppCacheDir() + "/framefx-gbuffer.profile", now);
    const std::string line = sc.gbuffer.report(now, 2'000'000'000);
    if (!line.empty()) ALOGI("%s: %s", Engine::get().package().c_str(), line.c_str());
}

// The real frame's analysis: the held frame (the game's image, through Kelir's stage A, or composed
// The game's velocity this frame sampled per finest block into sc.velocity.raw, when it renders one
// Luwes can read in place. False, and nothing recorded, otherwise.
bool recordVelocityHint(Device& d, Swapchain& sc, VkCommandBuffer cb, Frame& frame,
                        const lapis::Taps& taps) {
    const lapis::Tap& v = taps.velocity;
    if (v.image == VK_NULL_HANDLE || (v.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0 || taps.turned ||
        taps.held || taps.gameGenerates) {
        return false;
    }
    const VkImageView view = temporalViewFor(d, sc, v);
    if (view == VK_NULL_HANDLE) return false;
    sc.velocity.lastView = view;
    const Fns& vk = d.vk;
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkDescriptorSet set = sc.velocity.sets[&frame - sc.frames];
    writeImages(d, set, {{view, read, d.nearest}, {view, read, d.nearest}});
    if (v.layout != read) {
        imageBarrier(vk, cb, v.image, v.layout, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    const uint32_t gw = levelWidth(sc, kFinest) / 8;
    const uint32_t gh = levelHeight(sc, kFinest) / 8;
    // The grid covers the aligned frame; the game's velocity covers the frame.
    const float params[16] = {static_cast<float>(sc.alignedWidth) / sc.extent.width / gw,
                              static_cast<float>(sc.alignedHeight) / sc.extent.height / gh};
    fullscreen(vk, cb, d.motionPass, sc.velocity.fb, gw, gh, d.hint, d.temporalPipe, set, params);
    if (v.layout != read) {
        imageBarrier(vk, cb, v.image, read, v.layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
    }
    memoryBarrier(vk, cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT);
    return true;
}

float HalfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            bits = sign | (exp << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// The encoding last adopted for this game, kept across swapchains and launches: a new swapchain
// (Luwes switched on, a resize) starts with it in use instead of waiting for camera motion. The
// fit still checks it, and three clear contradictions drop it.
std::string VelocityProfilePath() {
    return AppCacheDir() + "/framefx-velocity.profile";
}

void loadVelocityProfile(Swapchain& sc) {
    auto& v = sc.velocity;
    if (v.profileRead) return;
    v.profileRead = true;
    std::string text;
    if (!android::base::ReadFileToString(VelocityProfilePath(), &text)) return;
    unsigned kind = 0;
    int flipped = 0;
    float bias = 0, sx = 0, sy = 0;
    if (std::sscanf(text.c_str(), "velocity1 %u %d %f %f %f", &kind, &flipped, &bias, &sx, &sy) != 5 ||
        kind < 1 || kind > 2 || sx == 0 || sy == 0) {
        return;
    }
    v.kind = static_cast<uint8_t>(kind);
    v.flipped = flipped != 0;
    v.bias = bias;
    v.scale[0] = sx;
    v.scale[1] = sy;
    v.confidence = 0.5f;
}

void saveVelocityProfile(const Swapchain& sc) {
    const auto& v = sc.velocity;
    android::base::WriteStringToFile(
            android::base::StringPrintf("velocity1 %u %d %.4f %.2f %.2f\n", v.kind, v.flipped ? 1 : 0,
                                        v.bias, v.scale[0], v.scale[1]),
            VelocityProfilePath());
}

// Refits the game's velocity encoding from a frame's copy of its raw samples and the block search's
// field: textured blocks the search matched well are ground truth, both row orders are tried, and
// a fit is adopted once two in a row agree. Without camera motion nothing changes.
void fitVelocity(Device& d, Swapchain& sc, Frame& f) {
    auto& v = sc.velocity;
    const int i = static_cast<int>(&f - sc.frames);
    f.velocityRead = false;
    if (v.readMap[i] == nullptr) return;
    const uint32_t gw = levelWidth(sc, kFinest) / 8;
    const uint32_t gh = levelHeight(sc, kFinest) / 8;
    const VkDeviceSize bytes = VkDeviceSize{16} * gw * gh;
    if (!v.readCoherent[i]) {
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = v.readMemory[i];
        range.size = VK_WHOLE_SIZE;
        d.vk.InvalidateMappedMemoryRanges(d.device, 1, &range);
    }
    const auto* raw = reinterpret_cast<const uint16_t*>(v.readMap[i]);
    const auto* field = reinterpret_cast<const uint16_t*>(v.readMap[i] + bytes / 2);
    const float levelToFrame = static_cast<float>(2 << kFinest);
    // The search's mean match error over moving blocks: what the game's vectors should lower.
    double errSum = 0;
    uint32_t errCount = 0;
    std::vector<VelocitySample> straight, flipped;
    straight.reserve(gw * gh);
    flipped.reserve(gw * gh);
    for (uint32_t n = 0; n < gw * gh; n++) {
        const float err = HalfToFloat(field[4 * n + 2]);
        const float still = HalfToFloat(field[4 * n + 3]);
        if (still > 0.02f) {
            errSum += err;
            errCount++;
        }
        // Textured (no motion matches badly) and matched: the search found the content's motion.
        if (!(still > 2.0f * err + 0.02f) || err > 0.06f) continue;
        const float cam[2] = {HalfToFloat(field[4 * n]) * levelToFrame,
                              HalfToFloat(field[4 * n + 1]) * levelToFrame};
        straight.push_back({{HalfToFloat(raw[4 * n]), HalfToFloat(raw[4 * n + 1])}, {cam[0], cam[1]}});
        flipped.push_back({{HalfToFloat(raw[4 * n + 2]), HalfToFloat(raw[4 * n + 3])}, {cam[0], cam[1]}});
    }
    if (errCount > 0) {
        const float mean = static_cast<float>(errSum / errCount);
        v.meanError = v.meanError == 0 ? mean : 0.8f * v.meanError + 0.2f * mean;
    }
    VelocityEncoding a = {}, b = {};
    framefx_signals_fit_velocity(straight.data(), straight.size(), &a);
    framefx_signals_fit_velocity(flipped.data(), flipped.size(), &b);
    const bool flip = b.kind != 0 && (a.kind == 0 || b.confidence > a.confidence);
    const VelocityEncoding& e = flip ? b : a;
    v.last = e;
    if (e.kind == 0) return;  // no camera motion to fit against
    v.fits++;
    // Engines store motion in texture or clip units: a fitted scale near the frame's size or
    // half of it is that unit, exactly; the search's whole-pixel vectors only blur it.
    VelocityEncoding snapped = e;
    for (int k = 0; k < 2; k++) {
        const float extent = static_cast<float>(k == 0 ? sc.extent.width : sc.extent.height);
        for (const float unit : {extent, 0.5f * extent}) {
            if (std::fabs(std::fabs(e.scale[k]) / unit - 1.0f) <= 0.25f) {
                snapped.scale[k] = std::copysign(unit, e.scale[k]);
                break;
            }
        }
    }
    // The search's vectors are whole pixels of a quarter-size level, so even a perfect fit leaves
    // a residual; trust needs agreement more than a high score.
    const bool sane = e.confidence >= 0.25f && e.inliers >= 0.6f;
    if (!sane) return;  // too little or too mixed motion to tell
    const VelocityEncoding& f2 = snapped;
    const VelocityEncoding& c = v.candidate;
    // 0.5 and 127/255 are the same zero to a fit.
    const bool agrees = c.kind == f2.kind && v.candidateFlipped == flip &&
            std::fabs(c.bias - f2.bias) < 0.01f &&
            std::fabs(c.scale[0] - f2.scale[0]) <= 0.2f * std::fabs(f2.scale[0]) &&
            std::fabs(c.scale[1] - f2.scale[1]) <= 0.2f * std::fabs(f2.scale[1]);
    v.candidate = f2;
    v.candidateFlipped = flip;
    if (agrees) {
        const bool fresh = v.kind == 0;
        v.kind = e.kind;
        v.flipped = flip;
        v.bias = f2.bias;
        // Smoothed once adopted; the first agreeing pair sets it.
        for (int k = 0; k < 2; k++) {
            v.scale[k] = fresh ? 0.5f * (c.scale[k] + f2.scale[k]) : 0.8f * v.scale[k] + 0.2f * f2.scale[k];
        }
        v.confidence = fresh ? f2.confidence : 0.8f * v.confidence + 0.2f * f2.confidence;
        v.adopted++;
        if (fresh) saveVelocityProfile(sc);
    }
    // The encoding in use, contradicted: dropped after three clear fits in a row say otherwise.
    const bool same = v.kind == f2.kind && v.flipped == flip && std::fabs(v.bias - f2.bias) < 0.01f &&
            std::signbit(v.scale[0]) == std::signbit(f2.scale[0]) &&
            std::signbit(v.scale[1]) == std::signbit(f2.scale[1]);
    if (v.kind != 0 && !same && ++v.contradicted >= 3) {
        v.kind = 0;
        v.contradicted = 0;
    } else if (same) {
        v.contradicted = 0;
    }
}

// from the scene layer), the scene layer when `taps` offers the game's scene, and with `history`
// the luma pyramid and motion field against the previous frame. `generation`: generated frames
// will read the held frame; alone, the real frame is composed at its present.
void recordAnalysis(Device& d, Swapchain& sc, VkCommandBuffer cb, uint32_t appIndex, bool history,
                    bool generation, bool upscale, Frame& frame, Frame* capture,
                    const kelir::vulkan::Stage* stage, const lapis::Taps* taps,
                    const kelir::Plan* plan, const lapis::Taps* game) {
    const Fns& vk = d.vk;
    const VkImage app = appImage(sc, appIndex);
    const int cur = sc.current;
    begin(vk, cb);
    const bool timed = capture != nullptr && sc.timer != VK_NULL_HANDLE;
    const uint32_t base = timed ? static_cast<uint32_t>(capture - sc.frames) * kTimestamps : 0;
    if (timed) {
        vk.CmdResetQueryPool(cb, sc.timer, base, kTimestamps);
        vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sc.timer, base);
        capture->timed = true;
        capture->timedFrame = sc.captureFrame;
        capture->timedOutputs = 0;
        capture->timedScene = 0;
    }
    if (!sc.primed) {
        for (int k = kFinest; k < kLevels; k++) {
            for (Image& image : sc.field[k]) {
                imageBarrier(vk, cb, image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
            }
        }
        imageBarrier(vk, cb, sc.global.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT);
        // The splat keys and the phase field live in GENERAL; the output binds the latter early.
        for (const Image* image : {&sc.velocity.splat, &sc.velocity.phase}) {
            imageBarrier(vk, cb, image->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_WRITE_BIT);
        }
        // The search and the output pass bind these before either was drawn.
        imageBarrier(vk, cb, sc.velocity.dense.image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        imageBarrier(vk, cb, sc.velocity.raw.image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        // The output pass binds the previous frame too, before one was ever held.
        imageBarrier(vk, cb, sc.color[1 - cur].image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        // hud.frag binds the matte before, unread until a layer ran.
        for (const Image& image : sc.afme.matte) {
            if (image.image == VK_NULL_HANDLE) continue;
            imageBarrier(vk, cb, image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        sc.primed = true;
    }
    sc.afme.fresh = true;
    if (sc.afme.engine != nullptr && !sc.afme.primed) {
        const VkImageLayout dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        const VkImageAspectFlags ds = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        const VkImageSubresourceRange color = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkImageSubresourceRange depth = {ds, 0, 1, 0, 1};
        const VkClearColorValue zero = {};
        const VkClearDepthStencilValue far = {0.0f, 0};
        for (int j = 0; j < 2; j++) {
            imageBarrier(vk, cb, sc.afme.depth[j].image, VK_IMAGE_LAYOUT_UNDEFINED, dst,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT, ds);
            vk.CmdClearDepthStencilImage(cb, sc.afme.depth[j].image, dst, &far, 1, &depth);
            imageBarrier(vk, cb, sc.afme.depth[j].image, dst,
                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, ds);
        }
        for (Image* image : {&sc.afme.motion[0], &sc.afme.motion[1], &sc.afme.still}) {
            imageBarrier(vk, cb, image->image, VK_IMAGE_LAYOUT_UNDEFINED, dst,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT);
            vk.CmdClearColorImage(cb, image->image, dst, &zero, 1, &color);
            imageBarrier(vk, cb, image->image, dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
        }
        imageBarrier(vk, cb, sc.afme.output.image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        sc.afme.primed = true;
    }
    sc.realFrames++;
    // A turned scene keeps its depth, read across; the game's motion and matrices stay behind.
    sc.afme.layerMiss = nullptr;
    lapis::Taps drawn;
    if (taps != nullptr && taps->scene.region.extent.width != 0) {
        // Depth and camera hold for the whole image, not this part: the 3D band stays off.
        drawn.reversed = taps->reversed;
        drawn.gameGenerates = taps->gameGenerates;
        drawn.held = taps->held;
        taps = recordDrawn(d, sc, cb, taps->scene, &drawn.scene) ? &drawn : nullptr;
        if (taps == nullptr) sc.afme.layerMiss = "the part of the scene drawn cannot be copied";
    }
    lapis::Taps upright;
    if (taps != nullptr && turned(taps->scene, appExtent(sc))) {
        upright.reversed = taps->reversed;
        upright.gameGenerates = taps->gameGenerates;
        upright.held = taps->held;
        upright.turned = true;
        upright.depth = taps->depth;
        upright.camera = taps->camera;
        upright.camera.matrices = false;
        std::swap(upright.camera.sx, upright.camera.sy);
        std::swap(upright.camera.jx, upright.camera.jy);
        taps = recordUpright(d, sc, cb, taps->scene, &upright.scene) ? &upright : nullptr;
        if (taps == nullptr) sc.afme.layerMiss = "the scene could not be turned";
    }
    const bool tapped = taps != nullptr;  // the caller wants the layer, or AFME the scene
    if (tapped && taps->scene.image != sc.afme.probed) {
        sc.afme.probed = taps->scene.image;
        sc.afme.probedWidth = taps->scene.width;
        sc.afme.probedHeight = taps->scene.height;
        sc.afme.probedFormat = taps->scene.format;
        sc.afme.orientation = storedOrientation(sc, taps->scene);
        sc.afme.candidate = -1;
        sc.afme.agree = 0;
        sc.afme.covered = false;
        sc.afme.hidden = 0;
    }
    // A flat frame votes for nothing, and some scenes need no vote: one presented in place, or
    // one the frame's own copies and blits carry to the screen.
    if (tapped && sc.afme.orientation < 0 && taps->scene.upright) sc.afme.orientation = 0;
    if (tapped && sc.afme.orientation < 0 && taps->scene.orientation >= 0 &&
        taps->scene.orientationExact) {
        sc.afme.orientation = taps->scene.orientation;
        ALOGI("luwes: the game's commands carry its scene to the screen%s",
              sc.afme.orientation == 0 ? " upright" : (sc.afme.orientation & 1) ? " upside down"
                                                                                 : " mirrored");
    }
    // A composite pass's viewport only hints; it settles a vote that never comes.
    if (tapped && sc.afme.orientation < 0 && taps->scene.orientation >= 0 &&
        sc.afme.votes >= kOrientationHintAfter) {
        sc.afme.orientation = taps->scene.orientation;
        ALOGI("luwes: no picture vote in %u tries; the composite's viewport says%s", sc.afme.votes,
              sc.afme.orientation == 0 ? " upright" : " upside down");
    }
    // A scene from a frame before says nothing about this one.
    const bool checks = tapped && !taps->scene.upright && !taps->held &&
            !(taps->scene.orientationExact && taps->scene.orientation >= 0) &&
            (sc.afme.orientation < 0 || sc.afme.candidate >= 0 || sc.afme.hidden > 0 ||
             sc.realFrames % kOrientationCheck == 0);
    sc.afme.hasSharp[cur] = false;
    sc.afme.sgsr[cur] = false;
    sc.afme.method[cur] = 0;
    if (checks && !sc.afme.probePending) {
        recordProbe(vk, cb, sc, app, taps->scene);
        frame.probed = true;
        sc.afme.probePending = true;
    }
    // Marks the end of the scene step `what` (null: their start).
    const auto stamp = [&](const char* what) {
        if (!timed || capture->timedScene >= kSceneTimestamps) return;
        capture->sceneSteps[capture->timedScene] = what;
        vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.timer,
                             base + kSceneAt + capture->timedScene++);
    };
    const auto read = [&](const Image& image, VkFormat format, uint32_t w, uint32_t h,
                          const char* tag) {
        imageBarrier(vk, cb, image.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        readback(d, sc, *capture, cb, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, format, w,
                 h, tag);
        imageBarrier(vk, cb, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT);
    };
    if (capture != nullptr && tapped && !taps->held) recordGBufferCapture(d, sc, cb, *capture, *taps);
    if (tapped) recordAfmeDepth(d, sc, cb, *taps);
    // AFME generates from the layers of both frames, and the output pass composes from them: the
    // held frame and its pyramid are for Luwes's own motion, or AFME without the frame before's.
    const bool held = generation && (!sc.afme.active || !sc.afme.hasSharp[1 - cur]);
    const lapis::Tap* s = tapped ? &taps->scene : nullptr;
    if (s != nullptr && sc.afme.orientation >= 0 && !sc.afme.covered &&
        sc.afme.upReal[cur] != VK_NULL_HANDLE && (s->usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) {
        stamp(nullptr);
        const uint32_t slot = static_cast<uint32_t>(&frame - sc.frames) * 2;
        const bool clocked = sc.upTimer != VK_NULL_HANDLE;
        if (clocked) {
            vk.CmdResetQueryPool(cb, sc.upTimer, slot, 2);
            vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sc.upTimer, slot);
        }
        const bool graded = stage != nullptr && stage->stageA != VK_NULL_HANDLE;
        const VkDescriptorSet band =
                plan != nullptr && !plan->peek
                ? recordBand(d, sc, cb, frame, *taps, *plan, graded && plan->has(kelir::kSharpen),
                             stamp)
                : VK_NULL_HANDLE;
        const bool sgsr = upscale && Smaller(*s, sc.extent);
        const lapis::Taps* temporal =
                sgsr && Engine::get().config().upscale.method == UpscaleFilter::kBest &&
                temporalReady(*taps, sc.extent)
                ? taps
                : nullptr;
        sc.afme.hasSharp[cur] =
                recordSharp(d, sc, cb, frame, *s, band, sgsr, temporal, stage, stamp) &&
                recordHud(d, sc, cb, frame, appIndex, *s) && (!held || recordHeld(d, sc, cb));
        if (sc.afme.hasSharp[cur]) {
            stamp(held ? "held" : "hud");
            sc.afme.sgsr[cur] = sgsr;
            if (sc.afme.upscaled != s->id) {
                sc.afme.upscaled = s->id;
                ALOGI("luwes: the game's scene %ux%u under its HUD at %ux%u%s", s->width,
                      s->height, sc.extent.width, sc.extent.height, sgsr ? ", upscaled" : "");
            }
        }
        if (clocked) {
            vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.upTimer, slot + 1);
            frame.upTimed = sc.afme.hasSharp[cur];
        }
        if (capture != nullptr && band != VK_NULL_HANDLE) {
            read(sc.band.fx[0], sc.band.format, sc.band.width, sc.band.height, "band");
        }
    } else if (s != nullptr) {
        sc.afme.layerMiss = sc.afme.orientation < 0 ? "the scene's orientation is undecided"
                : sc.afme.covered ? "the scene is covered"
                : sc.afme.upReal[cur] == VK_NULL_HANDLE ? "no layer images"
                                                        : "the scene cannot be sampled";
    }
    // The accumulation goes on only from a frame that added to it.
    if (!sc.afme.hasSharp[cur] || sc.afme.method[cur] != 2) sc.temporal.valid = false;
    if (!sc.afme.hasSharp[cur] && sc.resized()) {
        // The game's frame resampled to the swapchain's size is the held frame, or Kelir's stage A
        // grades it into the held frame.
        const bool graded = stage != nullptr && stage->stageA != VK_NULL_HANDLE;
        if (graded) {
            vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0,
                                  nullptr, 0, nullptr);
        }
        recordResample(d, sc, cb, appIndex,
                       graded ? sc.game.resampledFb : sc.game.colorFb[cur]);
        if (graded) {
            kelir::vulkan::recordStageAFrom(*stage, cb, sc.game.resampled.view,
                                            sc.color[cur].image, sc.color[cur].view);
        }
        if (capture != nullptr) {
            imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            readback(d, sc, *capture, cb, app, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc.format,
                     sc.game.extent.width, sc.game.extent.height, "real");
            imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        }
    } else if (!sc.afme.hasSharp[cur]) {
        if (stage != nullptr && stage->stageA != VK_NULL_HANDLE) {
            // A layer that failed part way may have read Kelir's pyramid, which this rebuilds.
            vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0,
                                  nullptr, 0, nullptr);
            // Kelir's stage A writes the held frame in place of the copy: generation works from it.
            kelir::vulkan::recordStageA(*stage, cb, appIndex, sc.color[cur].image, sc.color[cur].view);
        } else {
            // The game's writes are visible through its semaphores, which this submission waits on.
            imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            imageBarrier(vk, cb, sc.color[cur].image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkImageCopy copy = {};
            copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.dstSubresource = copy.srcSubresource;
            copy.extent = {sc.extent.width, sc.extent.height, 1};
            vk.CmdCopyImage(cb, app, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc.color[cur].image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            if (capture != nullptr) {
                readback(d, sc, *capture, cb, app, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc.format,
                         sc.extent.width, sc.extent.height, "real");
            }
            imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
            imageBarrier(vk, cb, sc.color[cur].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
        }
    }
    if (capture != nullptr && sc.afme.hasSharp[cur]) {
        read(sc.afme.sharp[cur], sc.format, sc.extent.width, sc.extent.height, "sharp");
        if (held) read(sc.color[cur], sc.format, sc.extent.width, sc.extent.height, "real");
    }
    if (timed) vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.timer, base + 1);

    // The pyramid of every held frame while generating, so the motion search always has the
    // previous one.
    for (int k = 0; k < kLevels && (held || !sc.afme.hasSharp[cur]) && generation; k++) {
        const float sw = static_cast<float>(k == 0 ? sc.extent.width : levelWidth(sc, k - 1));
        const float sh = static_cast<float>(k == 0 ? sc.extent.height : levelHeight(sc, k - 1));
        const float params[16] = {1.0f / sw, 1.0f / sh, k == 0 ? 1.0f : 0.0f};
        // Motion is searched on the 3D under the HUD where there is one: the HUD stays out of it.
        const VkDescriptorSet source = k == 0 && sc.afme.hasSharp[cur] &&
                        sc.afme.scenePyramid[cur] != VK_NULL_HANDLE
                ? sc.afme.scenePyramid[cur]
                : sc.pyramidSet[cur][k];
        fullscreen(vk, cb, d.r8Pass, sc.lumaFb[cur][k], levelWidth(sc, k), levelHeight(sc, k),
                   d.pyramid, d.pyramidPipe, source, params);
    }
    if (timed) vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.timer, base + 2);

    if (history) {
        // Last frame's output pass still reads the field this frame rewrites.
        memoryBarrier(vk, cb,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0);
        auto& gv = sc.velocity;
        loadVelocityProfile(sc);
        gv.lastView = VK_NULL_HANDLE;
        gv.written = game != nullptr && recordVelocityHint(d, sc, cb, frame, *game);
        // debug.framefx.luwes_game_velocity=off fits but never uses it, for A/B runs.
        static const bool abOff =
                android::base::GetProperty("debug.framefx.luwes_game_velocity", "") == "off";
        const bool useGame = gv.written && gv.kind != 0 && !abOff;
        vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.motion);
        for (const Dispatch& m : sc.motion[cur]) {
            const uint32_t bw = levelWidth(sc, m.level) / 8;
            const uint32_t bh = levelHeight(sc, m.level) / 8;
            const float params[16] = {static_cast<float>(m.pass),
                                      static_cast<float>(kSearchRadius),
                                      kSmoothness,
                                      static_cast<float>(1 << (kLevels - 1 - m.level)),
                                      useGame ? static_cast<float>(gv.kind) : 0.0f,
                                      gv.bias,
                                      gv.scale[0],
                                      gv.scale[1],
                                      gv.flipped ? 1.0f : 0.0f,
                                      static_cast<float>(2 << m.level),
                                      static_cast<float>(1 << std::max(m.level - kFinest, 0))};
            vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.motionPipe, 0, 1,
                                     &m.set, 0, nullptr);
            vk.CmdPushConstants(cb, d.motionPipe, VK_SHADER_STAGE_ALL, 0, 64, params);
            if (m.pass == kGlobal) {
                vk.CmdDispatch(cb, 1, 1, 1);
            } else {
                vk.CmdDispatch(cb, bw, bh, 1);  // a workgroup per block
            }
            memoryBarrier(vk, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
            if (timed) {
                vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.timer,
                                     base + kPassTimestamps + capture->timedPasses++);
            }
        }
        // The per-pixel field the output pass warps by: the game's velocity where it matches
        // better, the block field elsewhere. The game's image is only readable while the hint's
        // transition holds, so it is sampled in the same layout dance.
        const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const bool dense = useGame && gv.lastView != VK_NULL_HANDLE;
        gv.denseGame = dense;
        const lapis::Tap* vt = dense ? &game->velocity : nullptr;
        // The game's depth orders what lands on one pixel when the field is splatted.
        const lapis::Tap* dt = dense && game->depth.image != VK_NULL_HANDLE &&
                        (game->depth.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0
                ? &game->depth
                : nullptr;
        const VkImageView depthView = dt != nullptr ? depthViewFor(d, sc, *dt) : VK_NULL_HANDLE;
        if (depthView == VK_NULL_HANDLE) dt = nullptr;
        const VkImageLayout depthRead = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        const VkDescriptorSet set = gv.denseSets[&frame - sc.frames];
        const Binding fieldIn = {sc.field[kFinest][0].view, VK_IMAGE_LAYOUT_GENERAL, d.linear};
        writeImages(d, set,
                    {dense ? Binding{gv.lastView, read, d.linear} : fieldIn, fieldIn,
                     Binding{sc.luma[cur][0].view, read, d.linear},
                     Binding{sc.luma[1 - cur][0].view, read, d.linear},
                     dt != nullptr ? Binding{depthView, depthRead, d.nearest} : fieldIn});
        if (vt != nullptr && vt->layout != read) {
            imageBarrier(vk, cb, vt->image, vt->layout, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        if (dt != nullptr && dt->layout != depthRead) {
            imageBarrier(vk, cb, dt->image, dt->layout, depthRead, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                         depthAspects(dt->format));
        }
        const uint32_t lw = levelWidth(sc, 0);
        const uint32_t lh = levelHeight(sc, 0);
        const float params[16] = {dense ? static_cast<float>(gv.kind) : 0.0f,
                                  gv.bias,
                                  gv.scale[0],
                                  gv.scale[1],
                                  gv.flipped ? 1.0f : 0.0f,
                                  kFieldScale,
                                  static_cast<float>(lw),
                                  static_cast<float>(lh),
                                  static_cast<float>(sc.alignedWidth) / sc.extent.width,
                                  static_cast<float>(sc.alignedHeight) / sc.extent.height,
                                  dt != nullptr ? 1.0f : 0.0f,
                                  game != nullptr && game->reversed ? 1.0f : 0.0f};
        fullscreen(vk, cb, d.motionPass, gv.denseFb, lw, lh, d.dense, d.densePipe, set, params);
        if (vt != nullptr && vt->layout != read) {
            imageBarrier(vk, cb, vt->image, read, vt->layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0);
        }
        if (dt != nullptr && dt->layout != depthRead) {
            imageBarrier(vk, cb, dt->image, depthRead, dt->layout, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, depthAspects(dt->format));
        }
    }
    if (timed) vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.timer, base + 3);
    if (history && sc.velocity.written && ++sc.velocity.frames >= kVelocityFitPeriod) {
        // This frame's raw velocity and final field, for a refit once the frame retires.
        auto& gv = sc.velocity;
        gv.frames = 0;
        const int slot = static_cast<int>(&frame - sc.frames);
        const uint32_t gw = levelWidth(sc, kFinest) / 8;
        const uint32_t gh = levelHeight(sc, kFinest) / 8;
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy region = {};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {gw, gh, 1};
        imageBarrier(vk, cb, gv.raw.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vk.CmdCopyImageToBuffer(cb, gv.raw.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                gv.read[slot], 1, &region);
        imageBarrier(vk, cb, gv.raw.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        region.bufferOffset = VkDeviceSize{8} * gw * gh;
        vk.CmdCopyImageToBuffer(cb, sc.field[kFinest][0].image, VK_IMAGE_LAYOUT_GENERAL,
                                gv.read[slot], 1, &region);
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
        frame.velocityRead = true;
    }
    if (capture != nullptr) {
        if (history) {
            memoryBarrier(vk, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            const int top = kLevels - 1;
            readback(d, sc, *capture, cb, sc.field[top][1].image, VK_IMAGE_LAYOUT_GENERAL,
                     VK_FORMAT_R16G16B16A16_SFLOAT, levelWidth(sc, top) / 8,
                     levelHeight(sc, top) / 8, "coarse");
            readback(d, sc, *capture, cb, sc.field[kFinest][0].image, VK_IMAGE_LAYOUT_GENERAL,
                     VK_FORMAT_R16G16B16A16_SFLOAT, levelWidth(sc, kFinest) / 8,
                     levelHeight(sc, kFinest) / 8, "field");
            // What the output pass warps by, and the half-size luma it was matched on.
            read(sc.velocity.dense, kMotionFormat, levelWidth(sc, 0), levelHeight(sc, 0), "dense");
            read(sc.luma[cur][0], VK_FORMAT_R8_UNORM, levelWidth(sc, 0), levelHeight(sc, 0), "luma");
        }
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    }
    vk.EndCommandBuffer(cb);
}

// AFME writes the frame at phase t into its output, read by the output pass that follows. Later
// generated frames of the same real frame read that output again.
void recordAfme(Device& d, Swapchain& sc, VkCommandBuffer cb, Mode mode, float t) {
    if (!sc.afme.fresh) {
        imageBarrier(d.vk, cb, sc.afme.output.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT);
        return;
    }
    sc.afme.fresh = false;
    // AFME counts phase from the current frame; interpolation reaches back toward the previous.
    const float scale =
            std::clamp(mode == Mode::kInterpolate ? t - 1.0f : t, -0.99f, 0.99f);
    if (scale != sc.afme.scale) {
        afme::InitInfo info = {sc.extent.width, sc.extent.height, sc.extent.width,
                               sc.extent.height, afme::kFlags, scale};
        if (d.afme->update(sc.afme.engine, &info)) sc.afme.scale = scale;
    }
    afme::FEInfo in = {};
    // On the game's 3D before its HUD when both frames have it, so the HUD is neither moved nor
    // smeared.
    sc.afme.composing = sc.afme.hasSharp[sc.current] && sc.afme.hasSharp[1 - sc.current];
    const Image* source = sc.afme.composing ? sc.afme.sharp : sc.color;
    in.src1Color = source[1 - sc.current].view;
    in.src2Color = source[sc.current].view;
    in.src1Depth = sc.afme.depth[1 - sc.current].view;
    in.src2Depth = sc.afme.depth[sc.current].view;
    in.src1Motion = sc.afme.motion[0].view;
    in.src2Motion = sc.afme.motion[1].view;
    in.output = sc.afme.output.view;
    for (int i = 0; i < 4; i++) in.matrix[i * 5] = 1.0f;
    const GBufferOutput& g = sc.gbuffer.last();
    if (sc.gbuffer.feedsAfme() && g.reprojectValid && g.camera >= kGBufferTrusted &&
        sc.afme.depthFed[0] && sc.afme.depthFed[1]) {
        if (!sc.gbuffer.inverse() || !Invert4x4(g.reproject, in.matrix)) {
            std::copy(g.reproject, g.reproject + 16, in.matrix);
        }
        in.jitter[0] = g.jitter[0];
        in.jitter[1] = g.jitter[1];
        in.jitter[2] = g.previousJitter[0];
        in.jitter[3] = g.previousJitter[1];
    }
    in.cmd = cb;
    const int slot = static_cast<int>(sc.afme.calls % kAfmeSlots);
    if (sc.afme.pending[slot]) {
        d.vk.WaitForFences(d.device, 1, &sc.afme.done[slot], VK_TRUE, kFenceTimeoutNs);
        d.vk.ResetFences(d.device, 1, &sc.afme.done[slot]);
        sc.afme.pending[slot] = false;
    }
    if (d.afme->estimate(sc.afme.engine, &in)) {
        sc.afme.calls++;
        sc.afme.signal = slot;
    } else {
        ALOGW("luwes: AFME estimate failed at phase %.2f", scale);
    }
    imageBarrier(d.vk, cb, sc.afme.output.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

// The dense field moved to phase t (splat.comp) into sc.velocity.phase, or copied there when it
// does not hold the game's vectors.
void recordSplat(Device& d, Swapchain& sc, VkCommandBuffer cb, Mode mode, float t, bool splat) {
    const Fns& vk = d.vk;
    auto& v = sc.velocity;
    // The previous present's output pass still reads the phase field.
    memoryBarrier(vk, cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.splat);
    vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.splatPipe, 0, 1, &v.splatSet, 0,
                             nullptr);
    const uint32_t gx = (levelWidth(sc, 0) + 7) / 8;
    const uint32_t gy = (levelHeight(sc, 0) + 7) / 8;
    for (int pass = splat ? 0 : 2; pass < 3; pass++) {
        const float params[16] = {static_cast<float>(pass), t,
                                  mode == Mode::kInterpolate ? 1.0f : 0.0f, splat ? 1.0f : 0.0f};
        vk.CmdPushConstants(cb, d.splatPipe, VK_SHADER_STAGE_ALL, 0, 64, params);
        vk.CmdDispatch(cb, gx, gy, 1);
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
}

void recordOutput(Device& d, Swapchain& sc, const Output& out, VkCommandBuffer cb,
                  uint32_t index, Mode mode, float t, bool generated, Frame* capture, int k,
                  const Frame& f, VkPipeline finish) {
    const Fns& vk = d.vk;
    begin(vk, cb);
    // With AFME the pass reproduces AFME's frame; otherwise it warps the held pair itself.
    const bool viaAfme = generated && sc.afme.active;
    if (viaAfme) recordAfme(d, sc, cb, mode, t);
    // The HUD layer over the game's 3D, or over what AFME generated from it. Luwes's own motion
    // warps both frames' 3D and lays this frame's HUD over it; held frames that carry the layer
    // are warped only while one of the two has no 3D of its own (Kelir's stage B then goes under
    // the layer).
    const bool layered = sc.afme.hasSharp[sc.current] &&
            (!generated || (viaAfme && sc.afme.composing));
    const bool sceneWarp = generated && !viaAfme && sc.afme.hasSharp[sc.current] &&
            sc.afme.hasSharp[1 - sc.current] && sc.afme.sceneWarp[sc.current] != VK_NULL_HANDLE;
    const bool carried = !layered && !viaAfme && !sceneWarp && sc.afme.hasSharp[sc.current] &&
            finish != VK_NULL_HANDLE && kelir::Unpack(f.kelir.signatureB).screen[0] >= 0;
    const VkDescriptorSet set = layered ? (generated ? sc.afme.upCompose[sc.current]
                                                     : sc.afme.upReal[sc.current])
            : viaAfme                   ? sc.afme.set
            : sceneWarp                 ? sc.afme.sceneWarp[sc.current]
                                        : sc.outputSet[sc.current];
    // Luwes's own warp reads the dense field at this frame's phase.
    const bool ownWarp = generated && !viaAfme && !layered;
    const bool atPhase = ownWarp && sc.velocity.denseGame;
    if (ownWarp) recordSplat(d, sc, cb, mode, t, atPhase);
    const bool timed = capture != nullptr && capture->timed;
    const uint32_t at = timed ? static_cast<uint32_t>(capture - sc.frames) * kTimestamps + 4 +
                    2 * static_cast<uint32_t>(k)
                              : 0;
    if (timed) vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sc.timer, at);
    const float params[16] = {
            static_cast<float>(sc.extent.width),
            static_cast<float>(sc.extent.height),
            static_cast<float>(sc.alignedWidth),
            static_cast<float>(sc.alignedHeight),
            t,
            layered ? static_cast<float>(kComposeLayer) : static_cast<float>(mode),
            1.0f,  // the dense field is in frame pixels
            0.0f,
            carried ? 1.0f : 0.0f,
            sc.hideHud ? 1.0f : 0.0f,
            sceneWarp ? 1.0f : 0.0f,
            1.0f,  // the dense field
            atPhase ? 1.0f : 0.0f,
    };
    if (finish == VK_NULL_HANDLE) {
        fullscreen(vk, cb, out.pass, sc.targets[index], sc.extent.width, sc.extent.height,
                   out.pipeline, d.outputPipe, set, params);
    } else {
        // With Kelir's finish: its plan as set 1, its frame block after Luwes's push bytes.
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = out.pass;
        info.framebuffer = sc.targets[index];
        info.renderArea = {{0, 0}, sc.extent};
        vk.CmdBeginRenderPass(cb, &info, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport = {0.0f, 0.0f, static_cast<float>(sc.extent.width),
                                     static_cast<float>(sc.extent.height), 0.0f, 1.0f};
        const VkRect2D scissor = {{0, 0}, sc.extent};
        vk.CmdSetViewport(cb, 0, 1, &viewport);
        vk.CmdSetScissor(cb, 0, 1, &scissor);
        vk.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, finish);
        const VkDescriptorSet sets[] = {set, kelir::vulkan::planSet(f.kelir)};
        vk.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, d.finishPipe, 0, 2, sets, 0,
                                 nullptr);
        float frame[kelir::kFrameVec4s * 4];
        kelir::vulkan::frameConstants(f.kelir, frame);
        vk.CmdPushConstants(cb, d.finishPipe, VK_SHADER_STAGE_ALL, 0,
                            kelir::shaders::kFramePushOffset, params);
        vk.CmdPushConstants(cb, d.finishPipe, VK_SHADER_STAGE_ALL, kelir::shaders::kFramePushOffset,
                            kelir::shaders::kFramePushSize, frame);
        vk.CmdDraw(cb, 3, 1, 0, 0);
        vk.CmdEndRenderPass(cb);
    }
    if (viaAfme) {
        // AFME renders its next frame into the output as an attachment again.
        imageBarrier(vk, cb, sc.afme.output.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    }
    if (timed) {
        vk.CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sc.timer, at + 1);
        capture->timedOutputs = k + 1;
    }
    if (capture != nullptr) {
        const VkImage image = sc.images[index];
        imageBarrier(vk, cb, image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT);
        readback(d, sc, *capture, cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc.format,
                 sc.extent.width, sc.extent.height,
                 android::base::StringPrintf("gen%d", k + 1));
        imageBarrier(vk, cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        memoryBarrier(vk, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    }
    vk.EndCommandBuffer(cb);
}

// After an output's submission: fences the AFME slot it used, so the slot's next estimate can
// tell when this one left the GPU.
void fenceAfme(Device& d, Swapchain& sc, VkQueue queue, bool submitted) {
    const int slot = sc.afme.signal;
    sc.afme.signal = -1;
    if (slot < 0 || !submitted) return;
    sc.afme.pending[slot] =
            d.vk.QueueSubmit(queue, 0, nullptr, sc.afme.done[slot]) == VK_SUCCESS;
}

VkResult submit(const Fns& vk, VkQueue queue, VkCommandBuffer cb, uint32_t waits,
                const VkSemaphore* wait, const VkPipelineStageFlags* stages, VkSemaphore signal) {
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = waits;
    info.pWaitSemaphores = wait;
    info.pWaitDstStageMask = stages;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cb;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &signal;
    return vk.QueueSubmit(queue, 1, &info, VK_NULL_HANDLE);
}

void report(Swapchain& sc, const LuwesSettings& luwes, LuwesState state, int generated,
            bool paced) {
    LuwesStatus& st = sc.status;
    st.state = state;
    st.reason = state == LuwesState::kDeclined ? sc.declined : LuwesReason::kNone;
    st.api = Api::kVulkan;
    st.source = !sc.device->usable ? MotionSource::kNone
            : sc.afme.active       ? MotionSource::kAfme
                                   : MotionSource::kBlockSearch;
    st.scene = sc.afme.active && sc.afme.composing;
    st.width = static_cast<uint16_t>(sc.extent.width);
    st.height = static_cast<uint16_t>(sc.extent.height);
    st.multiplier = static_cast<uint8_t>(luwes.multiplier);
    st.generated = static_cast<uint8_t>(generated);
    st.mode = luwes.mode;
    st.paced = paced;
    st.realFps = sc.meter.realFps();
    st.presentedFps = sc.meter.presentedFps();
    st.panelHz = 1e9f / static_cast<float>(sc.refreshNs);
    Engine::get().status().writeLuwes(st);
}

// Upscaling's state for the status page, about once a second. `method`: how this frame's scene
// was brought to the output's size.
void reportUpscale(Swapchain& sc, const Config& config, const lapis::Taps* taps, int method,
                   int64_t now) {
    sc.upFrames++;
    const int64_t elapsed = now - sc.upReportedAt;
    if (elapsed < kRefreshCheckNs) return;
    const bool upscaled = method > 0;
    UpscaleStatus st;
    if (config.upscale.on()) {
        st.state = UpscaleState::kWaiting;
        st.width = static_cast<uint16_t>(sc.extent.width);
        st.height = static_cast<uint16_t>(sc.extent.height);
        st.method = static_cast<UpscaleMethod>(method);
        if (sc.resized()) {
            st.drawnWidth = static_cast<uint16_t>(sc.game.extent.width);
            st.drawnHeight = static_cast<uint16_t>(sc.game.extent.height);
        }
        if (sc.resized() && !upscaled) {
            // The whole frame resampled, with no scene to work on apart from its HUD.
            st.state = UpscaleState::kRunning;
            st.method = UpscaleMethod::kResample;
            st.realFramesPer10s = static_cast<uint16_t>(
                    std::min<int64_t>(sc.upFrames * 10'000'000'000LL / elapsed, 0xffff));
        } else if (!config.lapis.on()) {
            st.reason = UpscaleReason::kLapisOff;
        } else if (taps == nullptr) {
            st.reason = UpscaleReason::kNoScene;
        } else {
            st.sceneWidth = static_cast<uint16_t>(taps->scene.width);
            st.sceneHeight = static_cast<uint16_t>(taps->scene.height);
            if (!Smaller(taps->scene, sc.extent)) {
                st.reason = UpscaleReason::kFullSize;
            } else if (sc.upscaleFailed) {
                st.reason = UpscaleReason::kResources;
            } else if (upscaled) {
                st.state = UpscaleState::kRunning;
                st.gpuNs = sc.upGpuNs;
                st.realFramesPer10s = static_cast<uint16_t>(
                        std::min<int64_t>(sc.upFrames * 10'000'000'000LL / elapsed, 0xffff));
            }
        }
    }
    sc.upReportedAt = now;
    sc.upFrames = 0;
    Engine::get().reportUpscale(st);
}

bool formatUsable(const Device& d, VkFormat format) {
    VkFormatProperties2 props = {};
    props.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    d.instance.get_format_properties2(d.physical, format, &props);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (props.formatProperties.optimalTilingFeatures & need) == need;
}

VkResult pass(VkQueue queue, const VkPresentInfoKHR* info) {
    return gHost.queue_present(queue, info);
}

// A frame Luwes does not generate around goes out through Kelir when it has a stack to apply.
VkResult passOn(VkQueue queue, const VkPresentInfoKHR* info) {
    kelir::vulkan::Claim claim;
    if (kelir::vulkan::claim(queue, info, &claim, /*fromLuwes=*/true)) {
        return kelir::vulkan::present(queue, info, claim);
    }
    return pass(queue, info);
}

// Swapchain images beyond the game's own. Every present of a real frame waits in the queue for
// its desired time and holds its image until the next one is shown, so the presents of one
// interval need an image each and one more, or queueing blocks the game's render thread.
// Presents wait in SurfaceFlinger for their refresh, a lead and a window ahead, and a slow game's
// window holds up to kMaxPresents of them.
uint32_t extraImages(const LuwesSettings& luwes) {
    return static_cast<uint32_t>(luwes.on() ? kMaxPresents + 1 : 1);
}

}  // namespace

void attach(const framefx_vk_host* host) {
    gHost = *host;
}

VkResult createDevice(const framefx_vk_instance* in, VkPhysicalDevice physical,
                      const VkDeviceCreateInfo* info, const VkAllocationCallbacks* allocator,
                      VkDevice* device) {
    Engine::get().attach();
    // AFME needs optical flow on the device; a driver that refuses it gets the game's request.
    afme::DeviceInfo extended;
    const bool wantAfme = afme::Load() != nullptr && afme::Extend(*in, physical, *info, &extended);
    VkResult result = in->create_device(physical, wantAfme ? &extended.info : info, allocator, device);
    bool withAfme = wantAfme && result == VK_SUCCESS;
    if (wantAfme && result != VK_SUCCESS) {
        ALOGW("luwes: device with AFME's needs refused (%d); without AFME", result);
        result = in->create_device(physical, info, allocator, device);
    }
    if (result != VK_SUCCESS) return result;

    auto d = std::make_unique<Device>();
    d->device = *device;
    d->physical = physical;
    d->instance = *in;
    VkPhysicalDeviceMemoryProperties2 memory = {};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    if (in->get_memory_properties2 != nullptr) in->get_memory_properties2(physical, &memory);
    d->memory = memory.memoryProperties;
    // motion.comp runs a block as one subgroup of at least 64 lanes.
    bool subgroups = false;
    if (in->get_properties2 != nullptr) {
        VkPhysicalDeviceSubgroupProperties subgroup = {};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props = {};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props.pNext = &subgroup;
        in->get_properties2(physical, &props);
        if (props.properties.limits.timestampComputeAndGraphics) {
            d->timestampPeriod = props.properties.limits.timestampPeriod;
        }
        constexpr VkSubgroupFeatureFlags kOps =
                VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        subgroups = props.properties.apiVersion >= VK_API_VERSION_1_1 &&
                subgroup.subgroupSize >= 64 &&
                (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                (subgroup.supportedOperations & kOps) == kOps;
        if (!subgroups) {
            ALOGI("luwes: no 64-lane compute subgroups (size %u), frame generation off",
                  subgroup.subgroupSize);
        }
    }
    d->usable = subgroups && in->get_memory_properties2 != nullptr &&
            in->get_format_properties2 != nullptr &&
            d->vk.load(*device, in->get_device_proc_addr);
    if (withAfme && d->usable) d->afme = afme::Load();
    std::lock_guard lock(gLock);
    gDevices[*device] = std::move(d);
    return result;
}

void destroyDevice(VkDevice device) {
    std::unique_ptr<Device> d;
    {
        std::lock_guard lock(gLock);
        const auto it = gDevices.find(device);
        if (it == gDevices.end()) return;
        d = std::move(it->second);
        gDevices.erase(it);
    }
    if (!d->usable) return;
    d->vk.DeviceWaitIdle(device);
    for (auto& [handle, sc] : d->swapchains) {
        freeAll(*d, *sc);
        freeGame(*d, *sc);
    }
    destroyDeviceObjects(*d);
}

void getDeviceQueue(VkDevice device, uint32_t family, VkQueue queue) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    if (it == gDevices.end() || queue == VK_NULL_HANDLE) return;
    auto& queues = it->second->queues;
    if (std::none_of(queues.begin(), queues.end(),
                     [queue](const auto& q) { return q.first == queue; })) {
        queues.emplace_back(queue, family);
    }
}

namespace {

LuwesReason declineReason(const Device& d, const VkSwapchainCreateInfoKHR* info) {
    if (!d.usable) return LuwesReason::kResources;
    if (ColourOf(info->imageColorSpace, info->imageFormat) != SwapchainColour::kSrgb) {
        return LuwesReason::kFormat;  // HDR and wide gamut stay untouched
    }
    if (info->presentMode == VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR ||
        info->presentMode == VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR) {
        return LuwesReason::kPresentMode;
    }
    if (info->imageArrayLayers != 1 || (info->flags & VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR) != 0 ||
        !formatUsable(d, info->imageFormat)) {
        return LuwesReason::kFormat;
    }
    return LuwesReason::kNone;
}

// The internal display's size in the current mode, longer side first; zero when unknown.
VkExtent2D displaySize() {
    VkExtent2D size = {};
    android::ADisplay** displays = nullptr;
    const int n = android::ADisplay_acquirePhysicalDisplays(&displays);
    for (int i = 0; i < n; i++) {
        if (android::ADisplay_getDisplayType(displays[i]) != android::ADisplayType::DISPLAY_TYPE_INTERNAL) {
            continue;
        }
        android::ADisplayConfig* config = nullptr;
        if (android::ADisplay_getCurrentConfig(displays[i], &config) == 0 && config != nullptr) {
            const auto w = static_cast<uint32_t>(android::ADisplayConfig_getWidth(config));
            const auto h = static_cast<uint32_t>(android::ADisplayConfig_getHeight(config));
            size = {std::max(w, h), std::min(w, h)};
        }
        break;
    }
    if (displays != nullptr) android::ADisplay_release(displays);
    return size;
}

// The size the swapchain is made at for a game that draws at `info`'s: the display's, when the
// game draws its whole screen at the display's aspect and another size, and upscaling is on.
// Zero to make it as asked.
VkExtent2D outputFor(const Device& d, const VkSwapchainCreateInfoKHR* info) {
    const VkExtent2D none = {};
    if (!Engine::get().config().upscale.on() || declineReason(d, info) != LuwesReason::kNone) {
        return none;
    }
    return ResizedExtent(info->imageExtent, displaySize());
}

// Images for the game at the size it asked for, with the usage it asked for and what resampling
// needs. False, with none made, when any cannot be.
bool makeGameImages(Device& d, Swapchain& sc, const VkSwapchainCreateInfoKHR* info) {
    const uint32_t count = std::min(info->minImageCount + kGameSpare, kMaxGameImages);
    auto& g = sc.game;
    g.extent = info->imageExtent;
    VkImageFormatListCreateInfo formats = {};
    const VkImageCreateInfo image = GameImageInfo(*info, &formats);
    VkFenceCreateInfo fence = {};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    bool ok = true;
    for (uint32_t i = 0; i < count && ok; i++) {
        Image made;
        ok = d.vk.CreateImage(d.device, &image, nullptr, &made.image) == VK_SUCCESS;
        VkMemoryRequirements req = {};
        if (ok) d.vk.GetImageMemoryRequirements(d.device, made.image, &req);
        uint32_t type = UINT32_MAX;
        for (uint32_t t = 0; ok && t < d.memory.memoryTypeCount; t++) {
            if ((req.memoryTypeBits & (1u << t)) == 0) continue;
            if (type == UINT32_MAX) type = t;
            if (d.memory.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                type = t;
                break;
            }
        }
        VkMemoryAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = type;
        ok = ok && type != UINT32_MAX &&
                d.vk.AllocateMemory(d.device, &alloc, nullptr, &made.memory) == VK_SUCCESS &&
                d.vk.BindImageMemory(d.device, made.image, made.memory, 0) == VK_SUCCESS;
        if (ok) {
            VkImageViewCreateInfo view = {};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = made.image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = info->imageFormat;
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            ok = d.vk.CreateImageView(d.device, &view, nullptr, &made.view) == VK_SUCCESS;
        }
        VkFence read = VK_NULL_HANDLE;
        ok = ok && d.vk.CreateFence(d.device, &fence, nullptr, &read) == VK_SUCCESS;
        if (!ok) {
            freeImage(d, made);
            break;
        }
        g.images.push_back(made);
        g.read.push_back(read);
    }
    g.real.assign(g.images.size(), 0);
    g.held.assign(g.images.size(), 0);
    g.pending.assign(g.images.size(), 0);
    return ok;
}

// Callers made sure no submission still reads the game's images.
void freeGame(Device& d, Swapchain& sc) {
    auto& g = sc.game;
    for (size_t i = 0; i < g.images.size(); i++) {
        if (g.pending[i]) d.vk.WaitForFences(d.device, 1, &g.read[i], VK_TRUE, kFenceTimeoutNs);
        d.vk.DestroyFence(d.device, g.read[i], nullptr);
        freeImage(d, g.images[i]);
    }
    for (VkFence f : g.plainDone) {
        d.vk.WaitForFences(d.device, 1, &f, VK_TRUE, kFenceTimeoutNs);
        d.vk.DestroyFence(d.device, f, nullptr);
    }
    if (g.plainPool != VK_NULL_HANDLE) d.vk.DestroyCommandPool(d.device, g.plainPool, nullptr);
    for (VkSemaphore s : g.plainDrawn) d.vk.DestroySemaphore(d.device, s, nullptr);
    g.plainDrawn.clear();
    const bool had = !g.images.empty();
    g.images.clear();
    g.read.clear();
    g.real.clear();
    g.held.clear();
    g.pending.clear();
    g.plain.clear();
    g.plainDone.clear();
    g.plainPool = VK_NULL_HANDLE;
    if (had) gResized.fetch_sub(1, std::memory_order_relaxed);
}

// The swapchain of `device` whose game draws into images of its own; null for any other.
Swapchain* resizedOf(VkDevice device, VkSwapchainKHR swapchain) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    if (it == gDevices.end()) return nullptr;
    const auto sc = it->second->swapchains.find(swapchain);
    return sc != it->second->swapchains.end() && sc->second->resized() ? sc->second.get() : nullptr;
}

// Hands the game one of its images for the swapchain image `real` it just acquired: the next one
// it does not hold, once the frame that last read it left the GPU.
uint32_t gameIndexFor(Swapchain& sc, uint32_t real) {
    auto& g = sc.game;
    std::lock_guard lock(g.lock);
    const uint32_t n = static_cast<uint32_t>(g.images.size());
    uint32_t v = g.next;
    for (uint32_t k = 0; k < n; k++) {
        v = (g.next + k) % n;
        if (!g.held[v]) break;
    }
    g.next = (v + 1) % n;
    if (g.pending[v]) {
        const Fns& vk = sc.device->vk;
        vk.WaitForFences(sc.device->device, 1, &g.read[v], VK_TRUE, kFenceTimeoutNs);
        vk.ResetFences(sc.device->device, 1, &g.read[v]);
        g.pending[v] = 0;
    }
    g.held[v] = 1;
    g.real[v] = real;
    return v;
}

VKAPI_ATTR VkResult GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, uint32_t* count,
                                       VkImage* images) {
    Swapchain* sc = resizedOf(device, swapchain);
    if (sc == nullptr) return gGetImages(device, swapchain, count, images);
    const auto n = static_cast<uint32_t>(sc->game.images.size());
    if (images == nullptr) {
        *count = n;
        return VK_SUCCESS;
    }
    const uint32_t written = std::min(*count, n);
    for (uint32_t i = 0; i < written; i++) images[i] = sc->game.images[i].image;
    *count = written;
    return written < n ? VK_INCOMPLETE : VK_SUCCESS;
}

// Whether the game should rebuild `swapchain` instead of acquiring from it (one of its bounded
// asks is spent).
bool staleAcquire(VkDevice device, VkSwapchainKHR swapchain) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    if (it == gDevices.end()) return false;
    const auto sc = it->second->swapchains.find(swapchain);
    if (sc == it->second->swapchains.end()) return false;
    std::atomic<int>& left = sc->second->staleAcquires;
    int n = left.load(std::memory_order_relaxed);
    while (n > 0 && !left.compare_exchange_weak(n, n - 1, std::memory_order_relaxed)) {
    }
    return n > 0;
}

VKAPI_ATTR VkResult AcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout,
                                     VkSemaphore semaphore, VkFence fence, uint32_t* index) {
    if (staleAcquire(device, swapchain)) return VK_ERROR_OUT_OF_DATE_KHR;
    Swapchain* sc = resizedOf(device, swapchain);
    if (sc == nullptr) return gAcquire(device, swapchain, timeout, semaphore, fence, index);
    uint32_t real = 0;
    const VkResult r = gAcquire(device, swapchain, timeout, semaphore, fence, &real);
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) *index = gameIndexFor(*sc, real);
    return r;
}

VKAPI_ATTR VkResult AcquireNextImage2(VkDevice device, const VkAcquireNextImageInfoKHR* info,
                                      uint32_t* index) {
    if (staleAcquire(device, info->swapchain)) return VK_ERROR_OUT_OF_DATE_KHR;
    Swapchain* sc = resizedOf(device, info->swapchain);
    if (sc == nullptr) return gAcquire2(device, info, index);
    uint32_t real = 0;
    const VkResult r = gAcquire2(device, info, &real);
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) *index = gameIndexFor(*sc, real);
    return r;
}

VKAPI_ATTR VkResult ReleaseSwapchainImages(VkDevice device, const VkReleaseSwapchainImagesInfoEXT* info) {
    Swapchain* sc = resizedOf(device, info->swapchain);
    if (sc == nullptr) return gRelease(device, info);
    std::vector<uint32_t> real(info->imageIndexCount);
    {
        std::lock_guard lock(sc->game.lock);
        for (uint32_t i = 0; i < info->imageIndexCount; i++) {
            const uint32_t v = info->pImageIndices[i];
            if (v >= sc->game.images.size()) return VK_ERROR_SURFACE_LOST_KHR;
            real[i] = sc->game.real[v];
            sc->game.held[v] = 0;
        }
    }
    VkReleaseSwapchainImagesInfoEXT released = *info;
    released.pImageIndices = real.data();
    return gRelease(device, &released);
}

// A frame of a resized swapchain Luwes does not process: the game's image blitted, bilinear, into
// the swapchain image it was acquired with.
VkResult presentPlain(Device& d, Swapchain& sc, VkQueue queue, uint32_t family,
                      const VkPresentInfoKHR* info, uint32_t appIndex, uint32_t realIndex) {
    auto& g = sc.game;
    const Fns& vk = d.vk;
    if (g.plainPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo pool = {};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family;
        if (vk.CreateCommandPool(d.device, &pool, nullptr, &g.plainPool) != VK_SUCCESS) {
            g.plainPool = VK_NULL_HANDLE;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        g.plain.resize(sc.images.size());
        VkCommandBufferAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = g.plainPool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = static_cast<uint32_t>(g.plain.size());
        vk.AllocateCommandBuffers(d.device, &alloc, g.plain.data());
        lapis::ignore(g.plain.data(), alloc.commandBufferCount);
        VkFenceCreateInfo fence = {};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        g.plainDone.resize(sc.images.size());
        for (VkFence& f : g.plainDone) vk.CreateFence(d.device, &fence, nullptr, &f);
    }
    if (g.plainDrawn.empty()) {
        VkSemaphoreCreateInfo semaphore = {};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        g.plainDrawn.assign(sc.images.size(), VK_NULL_HANDLE);
        for (VkSemaphore& s : g.plainDrawn) vk.CreateSemaphore(d.device, &semaphore, nullptr, &s);
    }
    vk.WaitForFences(d.device, 1, &g.plainDone[realIndex], VK_TRUE, kFenceTimeoutNs);
    vk.ResetFences(d.device, 1, &g.plainDone[realIndex]);
    const VkCommandBuffer cb = g.plain[realIndex];
    begin(vk, cb);
    const VkImage app = g.images[appIndex].image;
    const VkImage out = sc.images[realIndex];
    imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT);
    imageBarrier(vk, cb, out, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageBlit blit = {};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[1] = {static_cast<int32_t>(g.extent.width),
                          static_cast<int32_t>(g.extent.height), 1};
    blit.dstOffsets[1] = {static_cast<int32_t>(sc.extent.width),
                          static_cast<int32_t>(sc.extent.height), 1};
    vk.CmdBlitImage(cb, app, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    imageBarrier(vk, cb, app, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    imageBarrier(vk, cb, out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    vk.EndCommandBuffer(cb);
    VkPipelineStageFlags stages[kMaxWaits];
    std::fill(stages, stages + kMaxWaits, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = std::min(info->waitSemaphoreCount, kMaxWaits);
    submit.pWaitSemaphores = info->pWaitSemaphores;
    submit.pWaitDstStageMask = stages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &g.plainDrawn[realIndex];
    if (vk.QueueSubmit(queue, 1, &submit, g.plainDone[realIndex]) != VK_SUCCESS) {
        return VK_ERROR_DEVICE_LOST;
    }
    {
        std::lock_guard lock(g.lock);
        g.pending[appIndex] = vk.QueueSubmit(queue, 0, nullptr, g.read[appIndex]) == VK_SUCCESS;
    }
    VkPresentInfoKHR present = *info;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &g.plainDrawn[realIndex];
    present.pImageIndices = &realIndex;
    return pass(queue, &present);
}

}  // namespace

bool accepts(VkDevice device, const VkSwapchainCreateInfoKHR* info) {
    const Device* d = deviceOf(device);
    return d != nullptr && declineReason(*d, info) == LuwesReason::kNone;
}

uint32_t spareImages(VkDevice device, VkSwapchainKHR swapchain) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    if (it == gDevices.end()) return 0;
    const auto sc = it->second->swapchains.find(swapchain);
    return sc == it->second->swapchains.end() ? 0 : sc->second->extraImages;
}

bool layers(VkDevice device, VkSwapchainKHR swapchain) {
    Swapchain* sc = nullptr;
    {
        std::lock_guard lock(gLock);
        const auto it = gDevices.find(device);
        if (it == gDevices.end() || !it->second->usable) return false;
        const auto s = it->second->swapchains.find(swapchain);
        if (s == it->second->swapchains.end()) return false;
        sc = s->second.get();
    }
    // The present thread owns the swapchain's watch.
    sc->watch.refresh(Engine::get());
    const Config& config = sc->watch.config();
    lapis::Taps taps;
    if (!config.lapis.on() || sc->upscaleFailed || sc->declined != LuwesReason::kNone) return false;
    return (lapis::taps(swapchain, &taps) && usable(config, taps) &&
            wantsLayer(config, taps, sc->extent)) ||
            (sc->linger > 0 && wantsWhole(config));
}

bool running(VkDevice device, VkSwapchainKHR swapchain) {
    std::lock_guard lock(gLock);
    const auto it = gDevices.find(device);
    if (it == gDevices.end()) return false;
    const auto sc = it->second->swapchains.find(swapchain);
    return sc != it->second->swapchains.end() && sc->second->running;
}

VkResult createSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                         const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain) {
    Device* d = deviceOf(device);
    if (d == nullptr) return gHost.create_swapchain(device, info, allocator, swapchain);

    LuwesReason declined = declineReason(*d, info);
    // Images for the generated frames, and the usage to copy from and render into every image.
    // Generated frames must not be dropped by an unthrottled queue.
    VkSwapchainCreateInfoKHR asked = *info;
    if (declined == LuwesReason::kNone) {
        asked.minImageCount += extraImages(Engine::get().config().luwes);
        asked.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (Engine::get().config().luwes.on()) asked.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    }
    auto sc = std::make_unique<Swapchain>();
    sc->device = d;
    // A game drawing at another size than the display's draws into images of its own; the
    // swapchain is made at the display's size.
    const VkExtent2D output = d->usable ? outputFor(*d, info) : VkExtent2D{};
    if (output.width != 0) {
        if (makeGameImages(*d, *sc, info)) {
            gResized.fetch_add(1, std::memory_order_relaxed);
            asked.imageExtent = output;
        } else {
            ALOGW("luwes: no memory for the game's own images at %ux%u", info->imageExtent.width,
                  info->imageExtent.height);
            freeGame(*d, *sc);
        }
    }
    VkResult result = gHost.create_swapchain(device, &asked, allocator, swapchain);
    if (result != VK_SUCCESS && sc->resized()) {
        freeGame(*d, *sc);
        asked.imageExtent = info->imageExtent;
        result = gHost.create_swapchain(device, &asked, allocator, swapchain);
    }
    if (result != VK_SUCCESS && declined == LuwesReason::kNone) {
        declined = LuwesReason::kResources;
        asked = *info;
        result = gHost.create_swapchain(device, info, allocator, swapchain);
    }
    if (result != VK_SUCCESS) return result;
    if (sc->resized()) {
        ALOGI("%s: drawn at %ux%u, shown at %ux%u: luwes %s", Engine::get().package().c_str(),
              info->imageExtent.width, info->imageExtent.height, asked.imageExtent.width,
              asked.imageExtent.height,
              info->imageExtent.width < asked.imageExtent.width ? "upscales" : "downsamples");
    }
    sc->handle = *swapchain;
    sc->window = gHost.window != nullptr ? gHost.window(*swapchain) : nullptr;
    sc->format = info->imageFormat;
    sc->extent = asked.imageExtent;
    sc->transform = info->preTransform;
    sc->fifo = asked.presentMode == VK_PRESENT_MODE_FIFO_KHR;
    sc->extraImages = asked.minImageCount - info->minImageCount;
    sc->declined = declined;
    uint32_t count = 0;
    gHost.get_swapchain_images(device, *swapchain, &count, nullptr);
    sc->images.resize(count);
    gHost.get_swapchain_images(device, *swapchain, &count, sc->images.data());
    bool rebuilt = false;
    {
        std::lock_guard lock(gLock);
        if (info->oldSwapchain != VK_NULL_HANDLE) {
            const auto old = d->swapchains.find(info->oldSwapchain);
            if (old != d->swapchains.end()) sc->target.inherit(old->second->target);
        }
        rebuilt = sc->window != nullptr && gRebuilding.erase(sc->window) > 0;
        d->swapchains[*swapchain] = std::move(sc);
    }
    // The engine rebuilt its swapchain at the size it last chose; under a render scale it takes
    // the window's only on resume.
    if (rebuilt) ReplayResume();
    return result;
}

void destroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
                      const VkAllocationCallbacks* allocator) {
    std::unique_ptr<Swapchain> sc;
    Device* d = nullptr;
    {
        std::lock_guard lock(gLock);
        const auto it = gDevices.find(device);
        if (it != gDevices.end()) {
            d = it->second.get();
            const auto s = d->swapchains.find(swapchain);
            if (s != d->swapchains.end()) {
                sc = std::move(s->second);
                d->swapchains.erase(s);
            }
        }
    }
    if (sc != nullptr && d->usable) {
        freeAll(*d, *sc);
        freeGame(*d, *sc);
    }
    gHost.destroy_swapchain(device, swapchain, allocator);
}

VkResult queuePresent(VkQueue queue, const VkPresentInfoKHR* info) {
    Engine& engine = Engine::get();
    if (!engine.active() && gRunning.load(std::memory_order_relaxed) == 0 &&
        gResized.load(std::memory_order_relaxed) == 0) {
        return pass(queue, info);
    }
    if (info->swapchainCount != 1 || info->waitSemaphoreCount > kMaxWaits) {
        return pass(queue, info);
    }
    uint32_t family = 0;
    Swapchain* found = swapchainOf(queue, info->pSwapchains[0], &family);
    if (found == nullptr) return pass(queue, info);
    Swapchain& sc = *found;
    Device& d = *sc.device;
    // The image the game drew (`appIndex`) and the swapchain image shown (`outIndex`): one and the
    // same unless the game draws at another size, into images of its own.
    const bool resized = sc.resized();
    const uint32_t appIndex = info->pImageIndices[0];
    uint32_t outIndex = appIndex;
    if (resized) {
        std::lock_guard lock(sc.game.lock);
        if (appIndex >= sc.game.images.size()) return VK_ERROR_OUT_OF_DATE_KHR;
        outIndex = sc.game.real[appIndex];
        sc.game.held[appIndex] = 0;
    }
    // Presents the frame without Luwes: as the game drew it, through Kelir, or blitted.
    const auto plainly = [&]() {
        return resized ? presentPlain(d, sc, queue, family, info, appIndex, outIndex)
                       : passOn(queue, info);
    };

    sc.watch.refresh(engine);
    const LuwesSettings& luwes = sc.watch.config().luwes;
    const int64_t now = Now();
    if (now - sc.refreshCheckedAt > kRefreshCheckNs) {
        const int64_t ns = gHost.refresh_ns(sc.handle);
        if (ns > 0) sc.refreshNs = ns;
        sc.refreshCheckedAt = now;
    }
    const Target::Decision dec = sc.target.begin(
            {now, static_cast<int>(sc.extent.width), static_cast<int>(sc.extent.height),
             sc.transform, sc.refreshNs},
            luwes);
    if (dec.releaseLuwes) {
        if (d.usable) freeAll(d, sc);
        report(sc, luwes, LuwesState::kOff, 0, false);
    }
    // The scene layer, with frame generation or alone.
    const Config& config = sc.watch.config();
    if (dec.resize) sc.upscaleFailed = false;
    lapis::Taps taps;
    const bool tapped = d.usable && config.lapis.on() && lapis::taps(sc.handle, &taps) &&
            usable(config, taps);
    if (!sc.gbufferConfigured) {
        sc.gbufferConfigured = true;
        sc.gbuffer.configure(android::base::GetProperty("debug.framefx.gbuffer", ""),
                             engine.package());
    }
    if (sc.gbuffer.on() && tapped) updateGBuffer(sc, taps, now);
    const bool layer = d.usable && config.lapis.on() && !sc.upscaleFailed;
    const bool wanted = layer && tapped && wantsLayer(config, taps, sc.extent);
    const bool lingering = layer && !wanted && sc.linger > 0 && wantsWhole(config);
    if (wanted) {
        sc.linger = kLayerLinger;
    } else if (sc.linger > 0) {
        sc.linger--;
    }
    const bool solo = !dec.process && (wanted || lingering || resized) &&
            sc.declined == LuwesReason::kNone && build(d);
    sc.hideHud = config.lapis.hideHud;
    if (survey::shooting() && !wanted) {
        ALOGI("survey   layer: not wanted: lapis %d tapped %d failed before %d declined %d",
              config.lapis.on(), tapped, sc.upscaleFailed, static_cast<int>(sc.declined));
    }
    if (sc.solo != solo) {
        ALOGI("%s: %s", engine.package().c_str(),
              solo ? "luwes presents with frame generation off" : "luwes alone stopped");
        if (!solo && !dec.process) freeAll(d, sc);
        sc.solo = solo;
    }
    const bool fits = sc.fifo && sc.extraImages >= extraImages(luwes);
    if (dec.process && sc.declined == LuwesReason::kNone && (!fits || !d.built)) {
        if (!fits) {
            // Made before Luwes was on or with fewer presents per frame: a mode that drops frames
            // or too few images. Ask for a new swapchain, at the present and at the next acquires;
            // a game that ignores both keeps running without generation.
            if (!sc.askedToRecreate) {
                sc.askedToRecreate = true;
                if (sc.window != nullptr) {
                    std::lock_guard lock(gLock);
                    gRebuilding.insert(sc.window);
                }
                ALOGI("%s: luwes needs FIFO and %u more images; asking for a new swapchain",
                      engine.package().c_str(), extraImages(luwes));
                sc.staleAcquires = kStaleAcquires;
                const VkResult r = plainly();
                return r < VK_SUCCESS ? r : VK_ERROR_OUT_OF_DATE_KHR;
            }
            if (++sc.presentsSinceAsk < kRecreateGiveUpAfter) return plainly();
            sc.staleAcquires = 0;
            ALOGW("%s: luwes declined: the game kept its swapchain", engine.package().c_str());
            sc.declined = LuwesReason::kPresentMode;
        } else if (!build(d)) {
            sc.declined = LuwesReason::kResources;
        } else {
            ALOGI("%s: luwes on Vulkan, pyramid block search and warp", engine.package().c_str());
        }
    }
    if (!solo && (!dec.process || sc.declined != LuwesReason::kNone)) {
        reportUpscale(sc, config, tapped ? &taps : nullptr, 0, now);
        sc.timing.stamp(sc.window, 0);
        if (luwes.on() && sc.meter.frame(now, 1)) {
            report(sc, luwes,
                   sc.declined != LuwesReason::kNone ? LuwesState::kDeclined
                                                     : LuwesState::kWaiting,
                   0, false);
        }
        return plainly();
    }

    const Output* out = outputFor(d, sc.format);
    const std::pair<bool, bool> needs = {dec.process, layer};
    if (out == nullptr || !prepareCommands(d, sc, family) || !prepareTargets(d, sc, *out) ||
        ((dec.resize || !sc.allocated || sc.allocatedFor != needs) &&
         !allocateAnalysis(d, sc, dec.process, layer))) {
        if (solo) {
            sc.upscaleFailed = true;
            ALOGW("%s: the scene layer declined on Vulkan: resources", engine.package().c_str());
        } else {
            sc.declined = LuwesReason::kResources;
            ALOGW("%s: luwes declined on Vulkan: resources", engine.package().c_str());
        }
        freeAll(d, sc);
        if (resized) sc.declined = LuwesReason::kResources;
        return plainly();
    }
    if (!sc.running) {
        if (!solo) {
            ALOGI("%s: luwes running on Vulkan %ux%u, %dx %s", engine.package().c_str(),
                  sc.extent.width, sc.extent.height, luwes.multiplier,
                  luwes.mode == Mode::kInterpolate ? "interpolate" : "extrapolate");
        }
        setRunning(sc, true);
    }

    if (outIndex >= sc.copied.size()) return plainly();
    Frame& f = sc.frames[sc.frameIndex];
    sc.frameIndex = (sc.frameIndex + 1) % kFrames;
    if (f.pending) {
        // Only blocks when the GPU is kFrames real frames behind.
        if (d.vk.WaitForFences(d.device, 1, &f.fence, VK_TRUE, kFenceTimeoutNs) != VK_SUCCESS) {
            sc.declined = LuwesReason::kResources;
            ALOGW("%s: luwes declined on Vulkan: GPU timeout", engine.package().c_str());
            return plainly();
        }
        d.vk.ResetFences(d.device, 1, &f.fence);
        f.pending = false;
        harvest(d, sc, f);
    }

    if (sc.captureLeft == 0 && !sc.capture.active()) {
        sc.captureLeft = sc.capture.poll(now);
        if (sc.captureLeft > 0) {
            sc.captureFrame = 0;
            sc.captureStart = now;
            sc.capture.note(android::base::StringPrintf(
                    "luwes vulkan %s %ux%u format %d %dx %s hud %d refresh_ns %lld",
                    engine.package().c_str(), sc.extent.width, sc.extent.height, sc.format,
                    luwes.multiplier,
                    luwes.mode == Mode::kInterpolate ? "interpolate" : "extrapolate",
                    luwes.hudProtection, static_cast<long long>(sc.refreshNs)));
        }
    }
    Frame* capture = sc.captureLeft > 0 ? &f : nullptr;
    std::string timeline;
    const auto mark = [&](const char* what, int64_t at) {
        if (capture != nullptr) {
            timeline += android::base::StringPrintf(" %s=%.3f", what,
                                                    static_cast<double>(at - sc.captureStart) / 1e6);
        }
    };
    mark("enter", now);

    luwes::Pacer& pacer = sc.target.pacer();
    if (!solo) {
        sc.timing.take(sc.window);
        sc.timing.learn(sc.window, pacer, sc.refreshNs);
    }

    // Kelir rides along: stage A into the held frame, its finish on every present.
    f.kelirOn = kelir::vulkan::beginStage(d.device, sc.handle, family, sc.watch.config().kelir,
                                          &f.kelir);
    const VkPipeline finish =
            f.kelirOn ? finishFor(d, sc.format, f.kelir.signatureB) : VK_NULL_HANDLE;
    if (f.kelirOn && finish == VK_NULL_HANDLE) {
        // Without the finish real and generated frames would differ: no Kelir for this frame.
        kelir::vulkan::stageRetired(d.device, sc.handle, f.kelir.serial);
        f.kelirOn = false;
    }
    const kelir::vulkan::Stage* stage = f.kelirOn ? &f.kelir : nullptr;
    // The real frame's own phase: where the output pass reproduces it exactly.
    const float realPhase = luwes.mode == Mode::kInterpolate ? 1.0f : 0.0f;
    // Across a scene cut there is no motion between the two frames: generated frames repeat the
    // real one instead of smearing one scene into the next.
    lapis::EngineSignals signals;
    // Only a changed frame structure or the 3D scene leaving or coming back: camera jumps read
    // from constants also fire on a lost or swapped matrix slot, which is not a cut.
    constexpr uint32_t kStructuralCut = 1u << 3 | 1u << 4 | 1u << 5;
    const bool cut = tapped && lapis::signals(sc.handle, &signals) && signals.cut != 0 &&
            (signals.cutReasons & kStructuralCut) != 0;
    if (cut) {
        sc.velocity.cuts++;
        sc.velocity.cutReasons |= signals.cutReasons;
    }
    if (now - sc.velocity.reportedAt > 5'000'000'000 && (sc.velocity.fits > 0 || sc.velocity.cuts > 0)) {
        const auto& v = sc.velocity;
        ALOGI("%s: luwes game velocity %s kind %u flipped %d bias %.3f scale %.2f %.2f trust %.2f "
              "fits %u adopted %u (last kind %u conf %.2f inliers %.2f), search error %.4f, "
              "cuts %u reasons 0x%x",
              engine.package().c_str(), v.kind != 0 ? "in use" : "unused", v.kind, v.flipped,
              v.bias, v.scale[0], v.scale[1], v.confidence, v.fits, v.adopted, v.last.kind,
              v.last.confidence, v.last.inliers, v.meanError, v.cuts, v.cutReasons);
        sc.velocity.reportedAt = now;
        sc.velocity.cutReasons = 0;
    }

    const VkSemaphore copied = sc.copied[outIndex];
    // With AFME the motion search is its own: the analysis only holds the frame.
    // debug.framefx.luwes_engine=own forces the block search where AFME would run, for A/B runs.
    static const bool forceOwn = android::base::GetProperty("debug.framefx.luwes_engine", "") == "own";
    sc.afme.active = !solo && sc.afme.engine != nullptr && !luwes.ownEngine && !forceOwn;
    const bool reads = tapped && layer && (sc.afme.active || wanted);
    const kelir::Plan* plan = config.kelir.get();
    recordAnalysis(d, sc, f.analysis, appIndex, dec.generate && !sc.afme.active, !solo,
                   config.upscale.on(), f, capture, stage, reads ? &taps : nullptr, plan,
                   tapped ? &taps : nullptr);
    if (reads) {
        reportLayer(sc, taps, wanted, solo);
        if (!sc.afme.layerReported || sc.afme.layerMiss != sc.afme.layerLogged) {
            sc.afme.layerReported = true;
            sc.afme.layerLogged = sc.afme.layerMiss;
            ALOGI("%s: the scene layer %s%s", engine.package().c_str(),
                  sc.afme.layerMiss != nullptr ? "failed: " : "composes",
                  sc.afme.layerMiss != nullptr ? sc.afme.layerMiss : "");
        }
    }
    VkPipelineStageFlags stages[kMaxWaits];
    std::fill(stages, stages + kMaxWaits, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    if (submit(d.vk, queue, f.analysis, info->waitSemaphoreCount, info->pWaitSemaphores, stages,
               copied) != VK_SUCCESS) {
        sc.declined = LuwesReason::kResources;
        if (f.kelirOn) kelir::vulkan::stageRetired(d.device, sc.handle, f.kelir.serial);
        f.kelirOn = false;
        return plainly();
    }
    if (resized) {
        // The analysis was the last to read the game's image: it may draw into it again after.
        std::lock_guard lock(sc.game.lock);
        sc.game.pending[appIndex] =
                d.vk.QueueSubmit(queue, 0, nullptr, sc.game.read[appIndex]) == VK_SUCCESS;
    }

    Present order[kMaxPresents];
    int count = 1;
    if (solo) {
        order[0].own = true;  // the game's frame, at once: nothing is paced
    } else {
        count = pacer.schedule(Now(), dec.plan, luwes.mode, dec.generate, sc.refreshNs, order);
    }
    VkResult result = VK_SUCCESS;
    int presents = 0;
    int generated = 0;
    bool generating = true;
    const VkPipelineStageFlags drawStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    for (int i = 0; i < count; i++) {
        const Present& p = order[i];
        if (p.own) {
            // The game's image, as rendered or under a generated frame; drawing over it waits
            // for the analysis to have copied it. With Kelir it is always drawn over: the held
            // frame at its own phase, finished. With its scene upscaled, its HUD over that.
            VkSemaphore ready = copied;
            const bool draws = (p.generated || finish != VK_NULL_HANDLE ||
                                sc.afme.hasSharp[sc.current] || resized) &&
                    generating;
            if (draws) {
                // Timestamps and readbacks are per generated frame.
                recordOutput(d, sc, *out, f.outputs[i], outIndex, luwes.mode,
                             p.generated && !cut ? p.phase : realPhase, p.generated,
                             p.generated ? capture : nullptr, generated, f, finish);
                const bool submitted = submit(d.vk, queue, f.outputs[i], 1, &copied, &drawStage,
                                              sc.drawn[outIndex]) == VK_SUCCESS;
                fenceAfme(d, sc, queue, submitted);
                if (submitted) {
                    ready = sc.drawn[outIndex];
                    if (p.generated) generated++;
                } else {
                    sc.declined = LuwesReason::kResources;
                    generating = false;
                }
            }
            VkPresentInfoKHR real = *info;
            real.waitSemaphoreCount = 1;
            real.pWaitSemaphores = &ready;
            real.pImageIndices = &outIndex;
            if (!solo) sc.timing.expect(sc.window, p.desired);
            sc.timing.stamp(sc.window, p.desired);
            mark("real", Now());
            result = pass(queue, &real);
            presents++;
            continue;
        }
        if (!generating) continue;
        uint32_t index = 0;
        mark("acquire", Now());
        const VkResult acquired = gHost.acquire(d.device, sc.handle, f.acquired[i], &index);
        mark("acquired", Now());
        if (acquired < VK_SUCCESS || index >= sc.targets.size()) {
            generating = false;
            continue;
        }
        recordOutput(d, sc, *out, f.outputs[i], index, luwes.mode, cut ? realPhase : p.phase,
                     true, capture,
                     generated, f, finish);
        const bool submitted = submit(d.vk, queue, f.outputs[i], 1, &f.acquired[i], &drawStage,
                                      sc.drawn[index]) == VK_SUCCESS;
        fenceAfme(d, sc, queue, submitted);
        if (!submitted) {
            // The acquire semaphore stays signalled: this swapchain's frames cannot be reused.
            sc.declined = LuwesReason::kResources;
            generating = false;
            continue;
        }
        VkPresentInfoKHR present = {};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &sc.drawn[index];
        present.swapchainCount = 1;
        present.pSwapchains = &sc.handle;
        present.pImageIndices = &index;
        sc.timing.expect(sc.window, p.desired);
        sc.timing.stamp(sc.window, p.desired);
        mark("gen", Now());
        if (pass(queue, &present) < VK_SUCCESS) {
            generating = false;
        } else {
            presents++;
            generated++;
        }
    }
    sc.status.framesGenerated += static_cast<uint32_t>(generated);
    // Retires this frame's command buffers and semaphores once the queue gets past them.
    d.vk.QueueSubmit(queue, 0, nullptr, f.fence);
    f.pending = true;
    sc.target.captured();
    reportUpscale(sc, config, reads ? &taps : nullptr,
                  sc.afme.hasSharp[sc.current] ? sc.afme.method[sc.current] : 0, now);
    sc.current = 1 - sc.current;

    if (!solo) {
        const int64_t wake = pacer.limit(Now());
        if (wake > 0) {
            mark("limit_to", wake);
            SleepUntil(wake);
        }
        pacer.resumed(Now());
    }
    if (capture != nullptr) {
        mark("exit", Now());
        // R the game's frame as rendered, O the game's image under a generated frame, G a
        // generated frame in an image of its own.
        std::string phases;
        for (int i = 0; i < count; i++) {
            phases += android::base::StringPrintf(
                    " %s%.2f@%.3f", !order[i].own ? "G" : order[i].generated ? "O" : "R",
                    order[i].phase, static_cast<double>(order[i].desired - sc.captureStart) / 1e6);
        }
        sc.capture.note(android::base::StringPrintf(
                                "frame %03u generated %d paced %d lead_ms %.2f window_ms %.2f "
                                "interval_ms %.2f phases",
                                sc.captureFrame, generated, dec.plan.paced,
                                static_cast<double>(pacer.leadNs()) / 1e6,
                                static_cast<double>(pacer.windowNs()) / 1e6, pacer.intervalMs()) +
                        phases + " timeline" + timeline);
        sc.captureFrame++;
        sc.captureLeft--;
    }
    if (sc.meter.frame(now, presents) && !solo) {
        report(sc, luwes, LuwesState::kRunning, dec.generate ? dec.plan.multiplier - 1 : 0,
               dec.plan.paced);
    }
    return result;
}

PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction next) {
    if (next == nullptr) return nullptr;
    if (strcmp(name, "vkGetSwapchainImagesKHR") == 0) {
        gGetImages = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(next);
        return reinterpret_cast<PFN_vkVoidFunction>(GetSwapchainImages);
    }
    if (strcmp(name, "vkAcquireNextImageKHR") == 0) {
        gAcquire = reinterpret_cast<PFN_vkAcquireNextImageKHR>(next);
        return reinterpret_cast<PFN_vkVoidFunction>(AcquireNextImage);
    }
    if (strcmp(name, "vkAcquireNextImage2KHR") == 0) {
        gAcquire2 = reinterpret_cast<PFN_vkAcquireNextImage2KHR>(next);
        return reinterpret_cast<PFN_vkVoidFunction>(AcquireNextImage2);
    }
    if (strcmp(name, "vkReleaseSwapchainImagesEXT") == 0 ||
        strcmp(name, "vkReleaseSwapchainImagesKHR") == 0) {
        gRelease = reinterpret_cast<PFN_vkReleaseSwapchainImagesEXT>(next);
        return reinterpret_cast<PFN_vkVoidFunction>(ReleaseSwapchainImages);
    }
    return nullptr;
}

bool gameImages(VkDevice device, VkSwapchainKHR swapchain, std::vector<VkImage>* images,
                VkExtent2D* shown) {
    Swapchain* sc = resizedOf(device, swapchain);
    if (sc == nullptr) return false;
    images->clear();
    for (const Image& i : sc->game.images) images->push_back(i.image);
    *shown = sc->extent;
    return true;
}

}  // namespace framefx::luwes::vulkan
