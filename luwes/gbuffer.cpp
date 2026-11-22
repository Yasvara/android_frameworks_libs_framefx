/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "luwes/gbuffer.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <sstream>

extern "C" {
void framefx_gbuffer_update(void* tracker, const void* input, bool gl, void* out);
void framefx_gbuffer_reset(void* tracker);
size_t framefx_gbuffer_profile(const void* input, uint8_t* out, size_t capacity);
bool framefx_gbuffer_warm(void* tracker, const uint8_t* line, size_t len);
bool framefx_gbuffer_invert(const float* m, float* out);
}

namespace framefx::luwes {
namespace {

std::string Format(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return buf;
}

std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

}  // namespace

uint32_t GBufferSignature(const GBufferInput& in) {
    // FNV-1a over the shapes and formats.
    uint32_t h = 2166136261u;
    for (uint32_t v : {in.sceneWidth, in.sceneHeight, in.depthWidth, in.depthHeight,
                       in.depthFormat, in.velocityWidth, in.velocityHeight, in.velocityFormat}) {
        for (int k = 0; k < 4; k++) {
            h = (h ^ ((v >> (8 * k)) & 0xff)) * 16777619u;
        }
    }
    return h == 0 ? 1 : h;
}

GBufferOutput GBufferUpdate(GBufferState* state, const GBufferInput& in, bool gl) {
    GBufferOutput out;
    framefx_gbuffer_update(state, &in, gl, &out);
    return out;
}

std::string GBufferProfileLine(const GBufferInput& in) {
    uint8_t line[96];
    const size_t n = framefx_gbuffer_profile(&in, line, sizeof(line));
    return std::string(reinterpret_cast<const char*>(line), n);
}

bool GBufferWarm(GBufferState* state, const std::string& line) {
    return framefx_gbuffer_warm(state, reinterpret_cast<const uint8_t*>(line.data()), line.size());
}

bool Invert4x4(const float* m, float* out) {
    return framefx_gbuffer_invert(m, out);
}

void GBufferFeed::configure(const std::string& packages, const std::string& package) {
    on_ = afme_ = inverse_ = false;
    const std::string app = package.substr(0, package.find(':'));
    std::stringstream words(packages);
    for (std::string word; std::getline(words, word, ',');) {
        const std::string w = Trim(word);
        if (w == "afme") {
            afme_ = true;
        } else if (w == "inverse") {
            inverse_ = true;
        } else if (!w.empty() && w == app) {
            on_ = true;
        }
    }
}

const GBufferOutput& GBufferFeed::update(const GBufferInput& in, bool gl,
                                         const std::string& profilePath, int64_t now) {
    (void)now;
    if (!profileRead_ && !profilePath.empty()) {
        profileRead_ = true;
        std::ifstream file(profilePath);
        std::string line;
        if (std::getline(file, line)) warmed_ = GBufferWarm(&state_, Trim(line));
    }
    const auto from = std::chrono::steady_clock::now();
    out_ = GBufferUpdate(&state_, in, gl);
    updateNs_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - from)
                                               .count());
    in_ = in;
    frames_++;
    if (out_.cut) cuts_++;
    if (out_.reprojectValid) reprojected_++;
    if (!profileWritten_ && !profilePath.empty() && out_.scene >= 1.0f && out_.depth >= 0.9f) {
        profileWritten_ = true;
        std::ofstream(profilePath) << GBufferProfileLine(in) << "\n";
    }
    return out_;
}

std::string GBufferFeed::report(int64_t now, int64_t periodNs) {
    if (frames_ == 0 || now - reportedAt_ < periodNs) return {};
    reportedAt_ = now;
    const GBufferOutput& o = out_;
    std::string s = Format(
            "gbuffer: depth %.2f %ux%u fmt %u %s, velocity %.2f %ux%u fmt %u%s, camera %.2f, "
            "scene %.2f; %llu frames, %llu reprojected, %llu cuts, %.1f us/frame%s%s",
            o.depth, in_.depthWidth, in_.depthHeight, in_.depthFormat,
            o.range == DepthRangeKind::kReversed ? "reversed"
                    : o.range == DepthRangeKind::kForwardGl ? "forward GL" : "forward",
            o.velocity, in_.velocityWidth, in_.velocityHeight, in_.velocityFormat,
            in_.temporal ? " TAA" : "", o.camera, o.scene, static_cast<unsigned long long>(frames_),
            static_cast<unsigned long long>(reprojected_), static_cast<unsigned long long>(cuts_),
            static_cast<double>(updateNs_) / 1e3 / static_cast<double>(frames_),
            warmed_ ? ", profile warm" : "", afme_ ? ", feeds AFME" : "");
    if (o.reprojectValid) {
        s += Format(" R [%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f]",
                                         o.reproject[0], o.reproject[1], o.reproject[2],
                                         o.reproject[3], o.reproject[4], o.reproject[5],
                                         o.reproject[6], o.reproject[7]);
    }
    frames_ = reprojected_ = cuts_ = updateNs_ = 0;
    return s;
}

void GBufferFeed::reset() {
    framefx_gbuffer_reset(&state_);
    out_ = GBufferOutput();
    warmed_ = false;
    profileRead_ = false;
}

}  // namespace framefx::luwes
