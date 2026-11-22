/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

// A fresh process for FramefxArming. It becomes the uid it is given, brings up EGL twice and
// Vulkan once the way an app does, and prints how many copies of libframefx it then holds.

#include <EGL/egl.h>
#include <link.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int CountEngines() {
    int count = 0;
    dl_iterate_phdr(
            [](dl_phdr_info* info, size_t, void* data) {
                const char* name = info->dlpi_name;
                const char* base = name != nullptr ? strrchr(name, '/') : nullptr;
                if (base != nullptr && strcmp(base + 1, "libframefx.so") == 0) ++*static_cast<int*>(data);
                return 0;
            },
            &count);
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const auto uid = static_cast<uid_t>(strtoul(argv[1], nullptr, 10));
        if (setresgid(uid, uid, uid) != 0 || setresuid(uid, uid, uid) != 0) return 2;
    }
    for (int i = 0; i < 2; i++) {
        const EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY || !eglInitialize(display, nullptr, nullptr)) return 3;
        eglTerminate(display);
    }
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) return 4;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    vkDestroyInstance(instance, nullptr);
    printf("%d\n", CountEngines());
    return 0;
}
