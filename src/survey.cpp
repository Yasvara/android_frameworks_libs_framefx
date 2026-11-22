/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "src/survey.h"

#include <atomic>
#include <android-base/parseint.h>
#include <android-base/stringprintf.h>
#include <android-base/strings.h>
#include <log/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "lapis/projection.h"
#include "lapis/recorder.h"
#include "src/engine.h"

namespace framefx::survey {
namespace {

using android::base::StringAppendF;
using android::base::StringPrintf;
using lapis::Frame;
using lapis::Image;
using lapis::ImageId;
using lapis::Images;
using lapis::Op;
using lapis::OpKind;
using lapis::Pass;
using lapis::Step;
using lapis::Target;

constexpr char kProperty[] = "debug.framefx.survey";
constexpr int kFramesPerShot = 2;
constexpr int kMissesPerValue = 4;

int gShotsLeft = 0;
uint32_t gShot = 0;
uint32_t gFrame = 0;
const prop_info* gProp = nullptr;
uint32_t gPropSerial = 0;
std::atomic<bool> gEnabled = false;
std::atomic<bool> gDrawList = false;
std::atomic<bool> gHudDraws = false;
std::atomic<int> gCutDraw = -1;
bool gFull = false;
std::atomic<int> gMissesLeft = 0;

std::string formatName(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM: return "D16";
        case VK_FORMAT_X8_D24_UNORM_PACK32: return "X8D24";
        case VK_FORMAT_D32_SFLOAT: return "D32F";
        case VK_FORMAT_S8_UINT: return "S8";
        case VK_FORMAT_D16_UNORM_S8_UINT: return "D16S8";
        case VK_FORMAT_D24_UNORM_S8_UINT: return "D24S8";
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return "D32FS8";
        case VK_FORMAT_R8G8B8A8_UNORM: return "RGBA8";
        case VK_FORMAT_R8G8B8A8_SRGB: return "RGBA8_SRGB";
        case VK_FORMAT_B8G8R8A8_UNORM: return "BGRA8";
        case VK_FORMAT_B8G8R8A8_SRGB: return "BGRA8_SRGB";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "RGB10A2";
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return "BGR10A2";
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return "R11G11B10F";
        case VK_FORMAT_R16G16B16A16_SFLOAT: return "RGBA16F";
        case VK_FORMAT_R16G16_SFLOAT: return "RG16F";
        case VK_FORMAT_R16_SFLOAT: return "R16F";
        case VK_FORMAT_R16_UNORM: return "R16";
        case VK_FORMAT_R32_SFLOAT: return "R32F";
        case VK_FORMAT_R32_UINT: return "R32UI";
        case VK_FORMAT_R8_UNORM: return "R8";
        case VK_FORMAT_R8G8_UNORM: return "RG8";
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return "RGB9E5";
        default: return StringPrintf("fmt%d", f);
    }
}

const char* loadName(VkAttachmentLoadOp op) {
    switch (op) {
        case VK_ATTACHMENT_LOAD_OP_LOAD: return "load";
        case VK_ATTACHMENT_LOAD_OP_CLEAR: return "clear";
        case VK_ATTACHMENT_LOAD_OP_DONT_CARE: return "dc";
        default: return "none";
    }
}

const char* storeName(VkAttachmentStoreOp op) {
    switch (op) {
        case VK_ATTACHMENT_STORE_OP_STORE: return "STORE";
        case VK_ATTACHMENT_STORE_OP_DONT_CARE: return "dc";
        default: return "none";
    }
}

const char* layoutName(VkImageLayout l) {
    switch (l) {
        case VK_IMAGE_LAYOUT_UNDEFINED: return "undef";
        case VK_IMAGE_LAYOUT_GENERAL: return "general";
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return "color";
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "ds_att";
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "ds_ro";
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "shader_ro";
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "xfer_src";
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "xfer_dst";
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL: return "d_att";
        case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL: return "d_ro";
        case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL: return "d_ro_s_att";
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL: return "d_att_s_ro";
        case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL: return "ro";
        case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL: return "att";
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: return "present";
        default: return "other";
    }
}

std::string usageName(VkImageUsageFlags u) {
    std::string s;
    const auto add = [&](VkImageUsageFlags bit, const char* name) {
        if (u & bit) s += s.empty() ? name : std::string("|") + name;
    };
    add(VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "xsrc");
    add(VK_IMAGE_USAGE_TRANSFER_DST_BIT, "xdst");
    add(VK_IMAGE_USAGE_SAMPLED_BIT, "SAMPLED");
    add(VK_IMAGE_USAGE_STORAGE_BIT, "storage");
    add(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "color");
    add(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "ds");
    add(VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, "TRANSIENT");
    add(VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, "input");
    return s;
}

class Log {
  public:
    explicit Log(const Images& images) : images_(images) {}

    const Image* image(ImageId id) const {
        const auto i = images_.find(id);
        return i == images_.end() ? nullptr : &i->second;
    }

    bool depth(ImageId id) const {
        const Image* i = image(id);
        return i != nullptr && (lapis::IsDepthFormat(i->format) || lapis::IsDepthLikeFormat(i->format));
    }

    bool swapchain(ImageId id) const {
        const Image* i = image(id);
        return i != nullptr && i->swapchain;
    }

    // A swapchain image reads #0, as the game cannot tell them apart.
    uint32_t shown(ImageId id) const { return swapchain(id) ? 0 : id; }

    std::string target(const Target& t) const {
        std::string s = StringPrintf("#%u %s", shown(t.image), formatName(t.format).c_str());
        if (t.samples > 1) StringAppendF(&s, " x%d", t.samples);
        StringAppendF(&s, " %s/%s", loadName(t.load), storeName(t.store));
        if (t.final != VK_IMAGE_LAYOUT_UNDEFINED) StringAppendF(&s, ">%s", layoutName(t.final));
        return s;
    }

    std::string brief(ImageId id) const {
        const Image* i = image(id);
        if (i == nullptr) return StringPrintf("#%u", shown(id));
        return StringPrintf("#%u %s %ux%u", shown(id), formatName(i->format).c_str(), i->width,
                            i->height);
    }

    std::string full(const Image& i) const {
        std::string s = StringPrintf("#%u %s %ux%u", shown(i.id), formatName(i.format).c_str(),
                                     i.width, i.height);
        if (i.samples > 1) StringAppendF(&s, " x%d", i.samples);
        if (i.mips > 1) StringAppendF(&s, " mips%u", i.mips);
        if (i.layers > 1) StringAppendF(&s, " layers%u", i.layers);
        StringAppendF(&s, " usage=%s mem=%d%s", usageName(i.usage).c_str(), i.memoryType,
                      (i.memory & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? " LAZY" : "");
        if (i.descriptorWrites > 0) {
            StringAppendF(&s, " game-reads=%u as", i.descriptorWrites);
            if (i.sampledAs & (1u << VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)) s += " combined";
            if (i.sampledAs & (1u << VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)) s += " sampled";
            if (i.sampledAs & (1u << VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT)) s += " input";
            if (i.sampledAs & (1u << VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)) s += " storage";
        }
        return s;
    }

    // Reads, frame-made images first and marked *, then up to a few textures.
    std::string reads(const std::vector<ImageId>& reads, const std::vector<ImageId>& made) const {
        std::vector<ImageId> first;
        std::vector<ImageId> rest;
        for (ImageId id : reads) {
            (std::find(made.begin(), made.end(), id) != made.end() ? first : rest).push_back(id);
        }
        std::string s;
        for (ImageId id : first) s += " " + brief(id) + "*";
        for (size_t k = 0; k < rest.size() && k < 4; k++) s += " " + brief(rest[k]);
        if (rest.size() > 4) StringAppendF(&s, " +%zu", rest.size() - 4);
        return s;
    }

  private:
    const Images& images_;
};

bool zero(float v) {
    return std::fabs(v) < 1e-6f;
}

// Every perspective projection in the game's host-mapped memory: field of view, aspect, jitter
// and the near and far planes under each depth convention.
void scanProjections() {
    constexpr size_t kMaxMapping = 128ull << 20;
    constexpr size_t kBudget = 768ull << 20;
    struct Seen {
        lapis::Projection p;
        uint32_t count = 0;
        uint32_t type = 0;
    };
    std::map<std::string, Seen> seen;
    std::map<std::string, uint32_t> reprojections;
    size_t mapped = 0;
    size_t scanned = 0;
    const auto start = std::chrono::steady_clock::now();
    lapis::forEachMapping([&](const uint8_t* data, size_t size, uint32_t type) {
        mapped += size;
        if (size < 64 || size > kMaxMapping || scanned + size > kBudget) return;
        scanned += size;
        const auto* f = reinterpret_cast<const float*>(data);
        const size_t floats = size / sizeof(float);
        for (size_t i = 0; i + 16 <= floats; i += 4) {
            // Reprojection matrices (clip to previous clip) sit near identity.
            if (std::fabs(f[i] - 1.0f) < 0.05f && std::fabs(f[i + 5] - 1.0f) < 0.05f &&
                std::fabs(f[i + 10] - 1.0f) < 0.5f && std::fabs(f[i + 15] - 1.0f) < 0.5f) {
                bool near = true;
                bool exact = true;
                for (int k = 0; k < 16 && near; k++) {
                    const float want = k % 5 == 0 ? 1.0f : 0.0f;
                    const float d = std::fabs(f[i + k] - want);
                    near = std::isfinite(f[i + k]) && d < 0.5f;
                    exact = exact && d < 1e-6f;
                }
                if (near && !exact && reprojections.size() < 64) {
                    std::string v;
                    for (int k = 0; k < 16; k++) StringAppendF(&v, " %.5f", f[i + k]);
                    reprojections[v]++;
                }
            }
            if (!zero(f[i + 1]) && !zero(f[i + 4])) continue;
            lapis::Projection p;
            if (!lapis::AsProjection(f + i, false, &p) && !lapis::AsProjection(f + i, true, &p)) {
                continue;
            }
            const std::string key = StringPrintf("%.4f %.4f %.4f %.4f %.4f %.4f %d", p.sx, p.sy,
                                                 p.jx, p.jy, p.a, p.b, p.rowMajor);
            Seen& s = seen[key];
            if (s.count++ == 0) {
                s.p = p;
                s.type = type;
            }
        }
    });
    const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count();
    ALOGI("survey   projections: %zu distinct in %.0f of %.0f MB mapped, %.1f ms", seen.size(),
          scanned / 1048576.0, mapped / 1048576.0, ms);
    for (const auto& [values, count] : reprojections) {
        ALOGI("survey   near-identity x%u:%s", count, values.c_str());
    }
    std::vector<const Seen*> order;
    for (const auto& [key, s] : seen) order.push_back(&s);
    std::sort(order.begin(), order.end(), [](const Seen* x, const Seen* y) { return x->count > y->count; });
    for (size_t k = 0; k < order.size() && k < 10; k++) {
        const lapis::Projection& p = order[k]->p;
        std::string planes;
        for (const auto& [range, name] : {std::pair{lapis::DepthRange::kReversed, "reversed"},
                                          std::pair{lapis::DepthRange::kForward, "forward"},
                                          std::pair{lapis::DepthRange::kForwardGl, "gl"}}) {
            float n;
            float f;
            if (p.planes(range, &n, &f)) StringAppendF(&planes, " | %s: near=%.4f far=%.1f", name, n, f);
        }
        ALOGI("survey   proj x%u %s fovY=%.1f aspect=%.3f jitter=(%.5f,%.5f) a=%.6f b=%.6f "
              "w=%+.0f%s | mem type %u",
              order[k]->count, p.rowMajor ? "row" : "col", p.fovY(), p.aspect(), p.jx, p.jy, p.a,
              p.b, p.w, planes.c_str(), order[k]->type);
    }
}

void logFrame(const Frame& frame, const Images& images, const lapis::Roles& roles) {
    const Log log(images);
    const char* pkg = Engine::get().package().c_str();
    const lapis::Calls c = lapis::calls();
    std::vector<ImageId> referenced;
    const auto refer = [&](ImageId id, bool any = false) {
        if (id == 0 || (!any && !log.depth(id))) return;
        if (std::find(referenced.begin(), referenced.end(), id) == referenced.end()) {
            referenced.push_back(id);
        }
    };
    ALOGI("survey %s shot %u frame %u: %zu events; calls so far rp %u rp2 %u rendering %u draw %u "
          "submit %u submit2 %u execute %u",
          pkg, gShot, gFrame, frame.steps.size(), c.renderPass, c.renderPass2, c.rendering, c.draw,
          c.submit, c.submit2, c.execute);
    if (gFrame == 0 && gFull) scanProjections();
    int n = 0;
    int uiPass = -1;
    std::vector<ImageId> made;  // written as an attachment earlier in this frame
    for (size_t index = 0; index < frame.steps.size(); index++) {
        const Step& e = frame.steps[index];
        if (!e.isPass) {
            const Op& o = e.op;
            switch (o.kind) {
                case OpKind::kOpticalFlow: {
                    static const char* kPoints[] = {"?", "input", "reference", "hint", "flow",
                                                    "back-flow", "cost", "back-cost", "global"};
                    std::string line = "survey   optical-flow |";
                    for (uint32_t v : o.reads) {
                        const uint32_t point = v >> 24;
                        const ImageId id = v & 0xffffff;
                        StringAppendF(&line, " %s=%s%s", point < 9 ? kPoints[point] : "?",
                                      log.brief(id).c_str(),
                                      std::find(made.begin(), made.end(), id) != made.end() ? "*" : "");
                        refer(id, true);
                    }
                    ALOGI("%s", line.c_str());
                    break;
                }
                case OpKind::kDispatch:
                    ALOGI("survey   dispatch x%u |%s", o.count, log.reads(o.reads, made).c_str());
                    for (ImageId id : o.reads) refer(id);
                    break;
                case OpKind::kSubmit:
                    ALOGI("survey   submit q%04x", o.src);
                    break;
                case OpKind::kLayout:
                    if (!log.depth(o.dst)) break;
                    ALOGI("survey   layout #%u %s>%s", o.dst, layoutName(o.from), layoutName(o.to));
                    refer(o.dst);
                    break;
                case OpKind::kClear:
                    ALOGI("survey   clear #%u depth=%.3f", o.dst, o.depth);
                    refer(o.dst);
                    break;
                default: {
                    if (!log.depth(o.src) && !log.depth(o.dst) && !log.swapchain(o.dst)) break;
                    const char* what = o.kind == OpKind::kCopy   ? "copy"
                            : o.kind == OpKind::kBlit          ? "blit"
                            : o.kind == OpKind::kResolve       ? "resolve"
                                                               : "to_buffer";
                    ALOGI("survey   %s #%u > #%u", what, log.shown(o.src), log.shown(o.dst));
                    refer(o.src);
                    refer(o.dst);
                    break;
                }
            }
            continue;
        }
        const Pass& p = e.pass;
        if (static_cast<int>(index) == roles.ui) uiPass = n;
        std::string line = StringPrintf("survey   pass %d %s %ux%u", n++, p.dynamic ? "dyn" : "rp",
                                        p.width, p.height);
        if (p.subpasses > 1) StringAppendF(&line, " sub%u", p.subpasses);
        if (p.viewportWidth > 0) {
            StringAppendF(&line, " vp %.0fx%.0f@%.0f,%.0f", p.viewportWidth, p.viewportHeight,
                          p.viewportX, p.viewportY);
        }
        StringAppendF(&line, " draws=%u ztest=%u zwrite=%u |", p.draws, p.depthTests,
                      p.depthWrites);
        for (const Target& t : p.colors) {
            line += " C " + log.target(t);
            refer(t.image);
        }
        if (p.hasDepth) {
            line += " | D " + log.target(p.depth);
            if (!std::isnan(p.clearDepth)) StringAppendF(&line, " clear=%.3f", p.clearDepth);
            if (p.stencilStore == VK_ATTACHMENT_STORE_OP_STORE) line += " s:STORE";
            if (p.depthInput) line += " d-as-input";
            if (p.depthResolve) line += " d-resolve";
            refer(p.depth.image);
        }
        if (!p.reads.empty()) {
            line += " | R" + log.reads(p.reads, made);
            for (ImageId id : p.reads) {
                if (std::find(made.begin(), made.end(), id) != made.end()) refer(id, true);
            }
        }
        ALOGI("%s", line.c_str());
        if (p.draws > 8 && !p.drawTrace.empty()) ALOGI("survey     draws %s", p.drawTrace.c_str());
        for (const std::string& hud : lapis::hudSurveyLines(frame, index)) ALOGI("%s", hud.c_str());
        // The constants of a single-draw pass that reads depth, a matrix per line.
        const bool readsDepth = std::any_of(p.reads.begin(), p.reads.end(),
                                            [&](ImageId id) { return log.depth(id); });
        for (const lapis::Constants& c :
             readsDepth && gFull ? p.constants : std::vector<lapis::Constants>()) {
            ALOGI("survey     const draw %u set %u binding %u floats %zu shader %08x", c.draw, c.set,
                  c.binding, c.values.size(), c.shader);
            for (size_t k = 0; k < c.values.size(); k += 16) {
                std::string row =
                        StringPrintf("survey     c %u:%u.%u @%zu:", c.draw, c.set, c.binding, k);
                for (size_t j = k; j < k + 16 && j < c.values.size(); j++) {
                    StringAppendF(&row, " %.7g", c.values[j]);
                }
                ALOGI("%s", row.c_str());
            }
        }
        for (const Target& t : p.colors) made.push_back(t.image);
        if (p.hasDepth) made.push_back(p.depth.image);
    }
    for (ImageId id : referenced) {
        if (const Image* i = log.image(id)) ALOGI("survey   image %s", log.full(*i).c_str());
    }
    ALOGI("survey   lapis: tier %d miss %d | D #%u%s | S #%u | U pass %d > #%u | V #%u",
          static_cast<int>(roles.tier), static_cast<int>(roles.miss), roles.depth,
          roles.depthStored ? " stored" : "", roles.scene, uiPass, log.shown(roles.target),
          roles.velocity);
    const std::string hud = lapis::hudSurveySummary(frame);
    if (!hud.empty()) ALOGI("%s", hud.c_str());
}

// The value the property holds now, and whether it names this process. Takes its options.
bool readProperty(std::string* value) {
    if (gProp == nullptr) gProp = __system_property_find(kProperty);
    if (gProp == nullptr) return false;
    __system_property_read_callback(
            gProp,
            [](void* cookie, const char*, const char* v, uint32_t serial) {
                *static_cast<std::string*>(cookie) = v;
                gPropSerial = serial;
            },
            value);
    // A package covers its own processes too, such as `pkg:render`.
    const std::string& process = Engine::get().package();
    const std::string pkg = process.substr(0, process.find(':'));
    bool named = false;
    bool full = false;
    bool draws = false;
    bool misses = false;
    bool hud = false;
    int cut = -1;
    for (const std::string& word : android::base::Split(*value, ",")) {
        if (word == pkg) {
            named = true;
        } else if (word == "full") {
            full = true;
        } else if (word == "draws") {
            draws = true;
        } else if (word == "misses") {
            misses = true;
        } else if (word == "hud") {
            hud = true;
        } else if (android::base::StartsWith(word, "cut=")) {
            android::base::ParseInt(word.substr(4), &cut, 0);
        }
    }
    gFull = full;
    gDrawList = draws;
    gHudDraws = named && hud;
    gCutDraw = cut;
    gMissesLeft = named && misses ? kMissesPerValue : 0;
    return named;
}

}  // namespace

bool enabled() {
    return gEnabled.load(std::memory_order_relaxed);
}

bool shooting() {
    return gShotsLeft > 0;
}

bool drawList() {
    return gDrawList.load(std::memory_order_relaxed) && shooting();
}

bool hudDraws() {
    return gHudDraws.load(std::memory_order_relaxed) && enabled();
}

struct StartWords {
    bool atlas = false;
    bool textCost = false;
    bool fontScan = false;
};

static const StartWords& startWords() {
    static const StartWords words = [] {
        StartWords w;
        char value[PROP_VALUE_MAX] = {};
        if (__system_property_get(kProperty, value) <= 0) return w;
        const std::string& process = Engine::get().package();
        const std::string pkg = process.substr(0, process.find(':'));
        bool named = false;
        for (const std::string& word : android::base::Split(value, ",")) {
            named = named || word == pkg;
            w.atlas = w.atlas || word == "atlas";
            w.textCost = w.textCost || word == "textcost";
            w.fontScan = w.fontScan || word == "fontscan";
        }
        if (!named) w = StartWords();
        ALOGI("survey: atlas %d textcost %d fontscan %d for %s", w.atlas, w.textCost, w.fontScan,
              process.c_str());
        return w;
    }();
    return words;
}

bool atlas() {
    return startWords().atlas;
}

bool textCost() {
    return startWords().textCost;
}

bool fontScan() {
    return startWords().fontScan;
}

int cutDraw() {
    return gCutDraw.load(std::memory_order_relaxed);
}

bool wantsFrame() {
    if (gShotsLeft > 0) return true;
    // A new value turns the survey on or off for this process, and asks for another shot.
    if (gProp == nullptr) gProp = __system_property_find(kProperty);
    if (gProp == nullptr || __system_property_serial(gProp) == gPropSerial) return false;
    std::string value;
    const bool named = readProperty(&value);
    const bool was = gEnabled.exchange(named, std::memory_order_relaxed);
    if (!named || !was) return false;
    gShotsLeft = kFramesPerShot;
    gShot++;
    gFrame = 0;
    ALOGI("survey: shot %u", gShot);
    return true;
}

void onFrame(const Frame& frame, const Images& images, const lapis::Roles& roles) {
    if (gShotsLeft <= 0) return;
    logFrame(frame, images, roles);
    gFrame++;
    gShotsLeft--;
}

void onMiss(const Frame& frame, const Images& images, const lapis::Roles& roles) {
    if (gMissesLeft.load(std::memory_order_relaxed) <= 0) return;
    gMissesLeft--;
    ALOGI("survey: miss");
    logFrame(frame, images, roles);
}

}  // namespace framefx::survey
