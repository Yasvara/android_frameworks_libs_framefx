/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <android-base/file.h>
#include <android-base/stringprintf.h>
#include <framefx/channel.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <string>

namespace framefx {
namespace {

// App uids no installed package holds on onyx.
constexpr unsigned kUnarmedUid = 19999;
constexpr unsigned kArmedUid = 19998;

// libframefx copies held by a fresh process running as `uid`, or a negative probe exit code.
int Probe(unsigned uid) {
    const std::string probe = android::base::GetExecutableDirectory() + "/framefx_arming_probe";
    FILE* out = popen(android::base::StringPrintf("%s %u", probe.c_str(), uid).c_str(), "r");
    if (out == nullptr) return -1;
    int engines = -1;
    if (fscanf(out, "%d", &engines) != 1) engines = -1;
    const int status = pclose(out);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? engines : -WEXITSTATUS(status);
}

std::string Marker(unsigned uid) {
    return android::base::StringPrintf("%s/%u", kMarkerDir, uid);
}

TEST(FramefxArming, UnarmedTablesAreOriginal) {
    unlink(Marker(kUnarmedUid).c_str());
    // Neither loader maps the engine for an app nobody armed; their tables stay the driver's.
    EXPECT_EQ(0, Probe(kUnarmedUid));
}

TEST(FramefxArming, ArmedLoadsOnce) {
    const std::string marker = Marker(kArmedUid);
    mkdir(kMarkerDir, 0755);
    ASSERT_TRUE(android::base::WriteStringToFile("", marker)) << marker;
    // EGL twice and Vulkan: one engine, shared by both loaders.
    const int engines = Probe(kArmedUid);
    unlink(marker.c_str());
    EXPECT_EQ(1, engines);
}

}  // namespace
}  // namespace framefx
