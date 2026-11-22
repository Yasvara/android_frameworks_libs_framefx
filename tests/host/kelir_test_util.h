/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "kelir/params.h"
#include "kelir/section.h"

namespace framefx::kelir::test {

// tests/golden, packaged next to the test binary; FRAMEFX_GOLDEN overrides it.
inline std::string GoldenDir() {
    if (const char* dir = getenv("FRAMEFX_GOLDEN")) return dir;
    char exe[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    std::string path(exe, n > 0 ? static_cast<size_t>(n) : 0);
    return path.substr(0, path.rfind('/')) + "/tests/golden";
}

inline std::vector<uint8_t> ReadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// A node of `kind`, everything at default and full intensity.
inline Node MakeNode(uint8_t kind, float intensity = 1.0f) {
    Node n;
    n.kind = kind;
    n.enabled = true;
    n.intensity = static_cast<uint16_t>(intensity * 65535.0f + 0.5f);
    return n;
}

inline Node& Set(Node& n, int param, float value) {
    n.q[param] = Encode(Info(n.kind).params[param], value);
    return n;
}

inline Stack MakeStack(std::initializer_list<Node> nodes, uint32_t revision = 1) {
    Stack s;
    s.revision = revision;
    for (const Node& n : nodes) s.nodes[s.count++] = n;
    return s;
}

inline std::vector<uint8_t> Bytes(const Stack& s) {
    std::vector<uint8_t> out(kMaxKelirSectionSize);
    out.resize(EncodeSection(s, out.data()));
    return out;
}

}  // namespace framefx::kelir::test
