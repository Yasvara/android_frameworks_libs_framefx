/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#define LOG_TAG "framefx"

#include "capture.h"

#include <android-base/file.h>
#include <android-base/stringprintf.h>
#include <fcntl.h>
#include <log/log.h>
#include <sys/stat.h>
#include <unistd.h>

#include <condition_variable>
#include <ctime>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

#include "src/engine.h"

namespace framefx::luwes {
namespace {

constexpr int64_t kPollNs = 1'000'000'000;
constexpr int kDefaultFrames = 16;
constexpr int kMaxFrames = 120;

}  // namespace

std::string AppCacheDir() {
    // A game's other processes are named <package>:<process>; the data directory is the package's.
    const std::string& name = Engine::get().package();
    return android::base::StringPrintf("/data/user/%u/%s/cache", getuid() / 100000,
                                       name.substr(0, name.find(':')).c_str());
}

struct Capture::Session {
    std::string dir;
    std::mutex lock;
    std::condition_variable wake;
    std::deque<std::pair<std::string, std::vector<uint8_t>>> queue;
    std::string index;
    bool closing = false;

    void run() {
        for (;;) {
            std::pair<std::string, std::vector<uint8_t>> item;
            {
                std::unique_lock guard(lock);
                wake.wait(guard, [this] { return closing || !queue.empty(); });
                if (queue.empty()) break;
                item = std::move(queue.front());
                queue.pop_front();
            }
            const std::string path = dir + "/" + item.first;
            const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
            if (fd < 0 || !android::base::WriteFully(fd, item.second.data(), item.second.size())) {
                ALOGW("capture: cannot write %s: %s", path.c_str(), strerror(errno));
            }
            if (fd >= 0) close(fd);
        }
        android::base::WriteStringToFile(index, dir + "/index.txt");
        ALOGI("capture written to %s", dir.c_str());
    }
};

int Capture::poll(int64_t now) {
    if (session_ != nullptr || now - checkedAt_ < kPollNs) return 0;
    checkedAt_ = now;
    const std::string cache = AppCacheDir();
    struct stat st;
    if (stat((cache + "/framefx-capture").c_str(), &st) != 0) return 0;
    const int64_t mtime = static_cast<int64_t>(st.st_mtim.tv_sec) * 1'000'000'000 +
            st.st_mtim.tv_nsec;
    if (mtime == seenMtime_) return 0;
    const bool first = seenMtime_ < 0;
    seenMtime_ = mtime;
    // A trigger left from an earlier run of the game is not a request.
    if (first && time(nullptr) - st.st_mtim.tv_sec > 30) return 0;

    auto session = std::make_shared<Session>();
    session->dir = android::base::StringPrintf("%s/framefx-capture-%lld", cache.c_str(),
                                               static_cast<long long>(now / 1'000'000));
    if (mkdir(session->dir.c_str(), 0700) != 0) {
        ALOGW("capture: cannot make %s: %s", session->dir.c_str(), strerror(errno));
        return 0;
    }
    std::thread([session] { session->run(); }).detach();
    session_ = session;
    const int frames = st.st_size > 0 ? static_cast<int>(std::min<int64_t>(st.st_size, kMaxFrames))
                                      : kDefaultFrames;
    ALOGI("capture: %d real frames into %s", frames, session->dir.c_str());
    return frames;
}

void Capture::write(const std::string& name, std::vector<uint8_t> bytes) {
    if (session_ == nullptr) return;
    std::lock_guard guard(session_->lock);
    session_->queue.emplace_back(name, std::move(bytes));
    session_->wake.notify_one();
}

void Capture::note(const std::string& line) {
    if (session_ == nullptr) return;
    std::lock_guard guard(session_->lock);
    session_->index += line;
    session_->index += '\n';
}

void Capture::end() {
    if (session_ == nullptr) return;
    {
        std::lock_guard guard(session_->lock);
        session_->closing = true;
        session_->wake.notify_one();
    }
    session_.reset();
}

uint32_t CaptureBytesPerPixel(const char* format) {
    if (strcmp(format, "rgba16f") == 0) return 8;
    if (strcmp(format, "r32f") == 0) return 4;
    if (strcmp(format, "rgb565") == 0) return 2;
    if (strcmp(format, "r8") == 0) return 1;
    if (strcmp(format, "d16") == 0) return 2;
    return strcmp(format, "unknown") == 0 ? 0 : 4;
}

}  // namespace framefx::luwes
