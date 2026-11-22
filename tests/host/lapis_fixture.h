/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "lapis/model.h"

// Frames read back from survey logs (framefx/src/survey.cpp), the games' own frame graphs.
// The survey names a swapchain image #0 and logs only the transfers that touch depth.
namespace framefx::lapis::test {

inline constexpr ImageId kSwapchain = 0x7fffffff;

struct Fixture {
    Frame frame;
    Images images;
};

// tests/fixtures/lapis, packaged next to the test binary; FRAMEFX_FIXTURES overrides it.
inline std::string FixtureDir() {
    if (const char* dir = getenv("FRAMEFX_FIXTURES")) return dir;
    char exe[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    std::string path(exe, n > 0 ? static_cast<size_t>(n) : 0);
    return path.substr(0, path.rfind('/')) + "/tests/fixtures/lapis";
}

inline VkFormat ParseFormat(const std::string& s) {
    static const std::map<std::string, VkFormat> kNames = {
            {"D16", VK_FORMAT_D16_UNORM},
            {"X8D24", VK_FORMAT_X8_D24_UNORM_PACK32},
            {"D32F", VK_FORMAT_D32_SFLOAT},
            {"S8", VK_FORMAT_S8_UINT},
            {"D16S8", VK_FORMAT_D16_UNORM_S8_UINT},
            {"D24S8", VK_FORMAT_D24_UNORM_S8_UINT},
            {"D32FS8", VK_FORMAT_D32_SFLOAT_S8_UINT},
            {"RGBA8", VK_FORMAT_R8G8B8A8_UNORM},
            {"RGBA8_SRGB", VK_FORMAT_R8G8B8A8_SRGB},
            {"BGRA8", VK_FORMAT_B8G8R8A8_UNORM},
            {"BGRA8_SRGB", VK_FORMAT_B8G8R8A8_SRGB},
            {"RGB10A2", VK_FORMAT_A2B10G10R10_UNORM_PACK32},
            {"BGR10A2", VK_FORMAT_A2R10G10B10_UNORM_PACK32},
            {"R11G11B10F", VK_FORMAT_B10G11R11_UFLOAT_PACK32},
            {"RGBA16F", VK_FORMAT_R16G16B16A16_SFLOAT},
            {"RG16F", VK_FORMAT_R16G16_SFLOAT},
            {"R16F", VK_FORMAT_R16_SFLOAT},
            {"R16", VK_FORMAT_R16_UNORM},
            {"R32F", VK_FORMAT_R32_SFLOAT},
            {"R32UI", VK_FORMAT_R32_UINT},
            {"R8", VK_FORMAT_R8_UNORM},
            {"RG8", VK_FORMAT_R8G8_UNORM},
            {"RGB9E5", VK_FORMAT_E5B9G9R9_UFLOAT_PACK32},
    };
    const auto found = kNames.find(s);
    if (found != kNames.end()) return found->second;
    if (s.rfind("fmt", 0) == 0) return static_cast<VkFormat>(strtoll(s.c_str() + 3, nullptr, 10));
    return VK_FORMAT_UNDEFINED;
}

inline VkImageLayout ParseLayout(const std::string& s) {
    static const std::map<std::string, VkImageLayout> kNames = {
            {"undef", VK_IMAGE_LAYOUT_UNDEFINED},
            {"general", VK_IMAGE_LAYOUT_GENERAL},
            {"color", VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {"ds_att", VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
            {"ds_ro", VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
            {"shader_ro", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {"xfer_src", VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
            {"xfer_dst", VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
            {"d_att", VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL},
            {"d_ro", VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL},
            {"d_ro_s_att", VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL},
            {"d_att_s_ro", VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL},
            {"ro", VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL},
            {"att", VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL},
            {"present", VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
    };
    const auto found = kNames.find(s);
    return found == kNames.end() ? VK_IMAGE_LAYOUT_UNDEFINED : found->second;
}

inline ImageId ParseId(const std::string& s) {
    const size_t hash = s.find('#');
    if (hash == std::string::npos) return 0;
    const ImageId id = static_cast<ImageId>(strtoul(s.c_str() + hash + 1, nullptr, 10));
    return id == 0 ? kSwapchain : id;
}

inline bool ParseSize(const std::string& s, uint32_t* w, uint32_t* h) {
    return sscanf(s.c_str(), "%ux%u", w, h) == 2;
}

inline std::vector<std::string> Words(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

inline std::vector<std::string> Chunks(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t bar; (bar = line.find('|', start)) != std::string::npos; start = bar + 1) {
        out.push_back(line.substr(start, bar - start));
    }
    out.push_back(line.substr(start));
    return out;
}

// "load/store>layout"
inline void ParseOps(const std::string& s, Target* t) {
    const size_t slash = s.find('/');
    const size_t arrow = s.find('>');
    const std::string load = s.substr(0, slash);
    const std::string store = s.substr(slash + 1, arrow == std::string::npos ? std::string::npos
                                                                              : arrow - slash - 1);
    t->load = load == "load"    ? VK_ATTACHMENT_LOAD_OP_LOAD
            : load == "clear" ? VK_ATTACHMENT_LOAD_OP_CLEAR
            : load == "dc"    ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                              : VK_ATTACHMENT_LOAD_OP_NONE;
    t->store = store == "STORE" ? VK_ATTACHMENT_STORE_OP_STORE
            : store == "dc"   ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                              : VK_ATTACHMENT_STORE_OP_NONE;
    if (arrow != std::string::npos) t->final = ParseLayout(s.substr(arrow + 1));
}

class Parser {
  public:
    std::vector<Fixture> parse(std::istream& in) {
        for (std::string line; std::getline(in, line);) {
            const size_t first = line.find_first_not_of(' ');
            if (first == std::string::npos) continue;
            line = line.substr(first);
            if (line.find(" shot ") != std::string::npos && line.find(" frame ") != std::string::npos &&
                line.find("events") != std::string::npos) {
                out_.emplace_back();
                continue;
            }
            if (out_.empty()) continue;
            const std::vector<std::string> w = Words(line);
            if (w[0] == "pass") {
                pass(line);
            } else if (w[0] == "image") {
                image(w);
            } else if (w[0] == "submit") {
                Step s;
                s.op.kind = OpKind::kSubmit;
                add(std::move(s));
            } else if (w[0] == "layout" && w.size() >= 3) {
                Step s;
                s.op.kind = OpKind::kLayout;
                s.op.dst = ParseId(w[1]);
                const size_t arrow = w[2].find('>');
                s.op.from = ParseLayout(w[2].substr(0, arrow));
                s.op.to = ParseLayout(w[2].substr(arrow + 1));
                add(std::move(s));
            } else if ((w[0] == "copy" || w[0] == "blit" || w[0] == "resolve" ||
                        w[0] == "to_buffer") && w.size() >= 4) {
                Step s;
                s.op.kind = w[0] == "copy"      ? OpKind::kCopy
                        : w[0] == "blit"      ? OpKind::kBlit
                        : w[0] == "resolve"   ? OpKind::kResolve
                                              : OpKind::kToBuffer;
                s.op.src = ParseId(w[1]);
                if (s.op.kind != OpKind::kToBuffer) s.op.dst = ParseId(w[3]);
                add(std::move(s));
            } else if (w[0] == "clear" && w.size() >= 3) {
                Step s;
                s.op.kind = OpKind::kClear;
                s.op.dst = ParseId(w[1]);
                s.op.depth = strtof(w[2].c_str() + strlen("depth="), nullptr);
                add(std::move(s));
            } else if (w[0] == "dispatch") {
                Step s;
                s.op.kind = OpKind::kDispatch;
                s.op.count = static_cast<uint32_t>(strtoul(w[1].c_str() + 1, nullptr, 10));
                const std::vector<std::string> chunks = Chunks(line);
                if (chunks.size() > 1) s.op.reads = reads(Words(chunks[1]));
                add(std::move(s));
            } else if (w[0] == "draws" && w.size() >= 2 && !out_.back().frame.steps.empty() &&
                       out_.back().frame.steps.back().isPass) {
                out_.back().frame.steps.back().pass.drawTrace = w[1];
            } else if (w[0] == "optical-flow") {
                Step s;
                s.op.kind = OpKind::kOpticalFlow;
                static const char* kPoints[] = {"?", "input", "reference", "hint", "flow",
                                                "back-flow", "cost", "back-cost", "global"};
                for (size_t i = 2; i < w.size(); i++) {
                    const size_t eq = w[i].find('=');
                    if (eq == std::string::npos) continue;
                    uint32_t point = 0;
                    for (uint32_t k = 0; k < 9; k++) {
                        if (w[i].compare(0, eq, kPoints[k]) == 0) point = k;
                    }
                    s.op.reads.push_back(point << 24 | (ParseId(w[i]) & 0xffffff));
                }
                add(std::move(s));
            }
        }
        for (Fixture& f : out_) finish(&f);
        return std::move(out_);
    }

  private:
    void add(Step s) { out_.back().frame.steps.push_back(std::move(s)); }

    Image& known(ImageId id) {
        Image& i = out_.back().images[id];
        i.id = id;
        i.swapchain = id == kSwapchain;
        return i;
    }

    void note(ImageId id, VkFormat format, uint32_t w, uint32_t h) {
        Image& i = known(id);
        if (i.format == VK_FORMAT_UNDEFINED) i.format = format;
        if (i.width == 0) {
            i.width = w;
            i.height = h;
        }
    }

    // "#id FMT WxH[*]" triples and "+N".
    std::vector<ImageId> reads(const std::vector<std::string>& w) {
        std::vector<ImageId> ids;
        for (size_t i = 0; i < w.size(); i++) {
            if (w[i][0] != '#' || i + 2 >= w.size()) continue;
            const ImageId id = ParseId(w[i]);
            uint32_t width = 0;
            uint32_t height = 0;
            ParseSize(w[i + 2], &width, &height);
            note(id, ParseFormat(w[i + 1]), width, height);
            ids.push_back(id);
            i += 2;
        }
        return ids;
    }

    void image(const std::vector<std::string>& w) {
        if (w.size() < 4) return;
        Image& i = known(ParseId(w[1]));
        i.format = ParseFormat(w[2]);
        ParseSize(w[3], &i.width, &i.height);
        for (size_t k = 4; k < w.size(); k++) {
            if (w[k].rfind("mips", 0) == 0) i.mips = strtoul(w[k].c_str() + 4, nullptr, 10);
            if (w[k].rfind("layers", 0) == 0) i.layers = strtoul(w[k].c_str() + 6, nullptr, 10);
            if (w[k].rfind("usage=", 0) == 0) {
                const std::string u = w[k];
                if (u.find("xsrc") != std::string::npos) i.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                if (u.find("xdst") != std::string::npos) i.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                if (u.find("SAMPLED") != std::string::npos) i.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                if (u.find("storage") != std::string::npos) i.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                if (u.find("color") != std::string::npos) {
                    i.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                }
                if (u.find("|ds") != std::string::npos || u.find("=ds") != std::string::npos) {
                    i.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                }
            }
        }
    }

    void pass(const std::string& line) {
        const std::vector<std::string> chunks = Chunks(line);
        Step s;
        s.isPass = true;
        Pass& p = s.pass;
        const std::vector<std::string> head = Words(chunks[0]);
        p.dynamic = head[2] == "dyn";
        ParseSize(head[3], &p.width, &p.height);
        for (const std::string& h : head) {
            if (h.rfind("sub", 0) == 0) p.subpasses = strtoul(h.c_str() + 3, nullptr, 10);
            sscanf(h.c_str(), "draws=%u", &p.draws);
            sscanf(h.c_str(), "ztest=%u", &p.depthTests);
            sscanf(h.c_str(), "zwrite=%u", &p.depthWrites);
        }
        for (size_t c = 1; c < chunks.size(); c++) {
            const std::vector<std::string> w = Words(chunks[c]);
            if (w.empty()) continue;
            if (w[0] == "C") {
                for (size_t i = 0; i + 2 < w.size(); i++) {
                    if (w[i] != "C") continue;
                    Target t;
                    t.image = ParseId(w[i + 1]);
                    t.format = ParseFormat(w[i + 2]);
                    size_t ops = i + 3;
                    if (ops < w.size() && w[ops][0] == 'x') {
                        t.samples = static_cast<VkSampleCountFlagBits>(
                                strtoul(w[ops].c_str() + 1, nullptr, 10));
                        ops++;
                    }
                    if (ops < w.size()) ParseOps(w[ops], &t);
                    note(t.image, t.format, p.width, p.height);
                    p.colors.push_back(t);
                }
            } else if (w[0] == "D" && w.size() >= 4) {
                p.hasDepth = true;
                p.depth.image = ParseId(w[1]);
                p.depth.format = ParseFormat(w[2]);
                size_t ops = 3;
                if (w[ops][0] == 'x' && ops + 1 < w.size()) {
                    p.depth.samples = static_cast<VkSampleCountFlagBits>(
                            strtoul(w[ops].c_str() + 1, nullptr, 10));
                    ops++;
                }
                ParseOps(w[ops], &p.depth);
                note(p.depth.image, p.depth.format, p.width, p.height);
                for (size_t i = ops + 1; i < w.size(); i++) {
                    if (w[i].rfind("clear=", 0) == 0) p.clearDepth = strtof(w[i].c_str() + 6, nullptr);
                    if (w[i] == "s:STORE") p.stencilStore = VK_ATTACHMENT_STORE_OP_STORE;
                    if (w[i] == "d-as-input") p.depthInput = true;
                    if (w[i] == "d-resolve") p.depthResolve = true;
                }
            } else if (w[0] == "R") {
                p.reads = reads(std::vector<std::string>(w.begin() + 1, w.end()));
            }
        }
        add(std::move(s));
    }

    // The presented image: the swapchain image a pass draws into.
    static void finish(Fixture* f) {
        for (const Step& s : f->frame.steps) {
            if (s.isPass && s.pass.writes(kSwapchain)) {
                f->frame.presented = kSwapchain;
                f->frame.width = s.pass.width;
                f->frame.height = s.pass.height;
            }
        }
    }

    std::vector<Fixture> out_;
};

inline std::vector<Fixture> LoadSurvey(const std::string& name) {
    std::ifstream in(FixtureDir() + "/" + name);
    return Parser().parse(in);
}

// For a game that draws into its own image and copies that to the swapchain (UE's delayed
// back-buffer acquire), which the survey does not log: adds the copy from the last pass's target.
inline void CopyLastTargetToSwapchain(Fixture* f) {
    for (auto s = f->frame.steps.rbegin(); s != f->frame.steps.rend(); ++s) {
        if (!s->isPass || s->pass.colors.empty()) continue;
        Step copy;
        copy.op.kind = OpKind::kCopy;
        copy.op.src = s->pass.colors[0].image;
        copy.op.dst = kSwapchain;
        f->frame.presented = kSwapchain;
        f->frame.width = s->pass.width;
        f->frame.height = s->pass.height;
        Image& sc = f->images[kSwapchain];
        sc.id = kSwapchain;
        sc.swapchain = true;
        sc.format = s->pass.colors[0].format;
        sc.width = s->pass.width;
        sc.height = s->pass.height;
        f->frame.steps.push_back(std::move(copy));
        return;
    }
}

// The index in Frame::steps of the `n`th pass.
inline int PassStep(const Frame& frame, int n) {
    for (int i = 0; i < static_cast<int>(frame.steps.size()); i++) {
        if (frame.steps[i].isPass && n-- == 0) return i;
    }
    return -1;
}

}  // namespace framefx::lapis::test
