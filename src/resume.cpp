/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "src/resume.h"

#include <android_os_MessageQueue.h>
#include <jni.h>
#include <log/log.h>
#include <utils/Looper.h>

#include <thread>
#include <vector>

namespace framefx {
namespace {

bool Cleared(JNIEnv* env) {
    if (!env->ExceptionCheck()) return false;
    env->ExceptionDescribe();
    env->ExceptionClear();
    return true;
}

// On the main thread: every activity of the process; the replay skips those not resumed.
void Replay(JNIEnv* env) {
    jclass threadClass = env->FindClass("android/app/ActivityThread");
    if (threadClass == nullptr || Cleared(env)) return;
    const jmethodID current = env->GetStaticMethodID(threadClass, "currentActivityThread",
                                                     "()Landroid/app/ActivityThread;");
    const jfieldID activities = env->GetFieldID(threadClass, "mActivities", "Landroid/util/ArrayMap;");
    const jmethodID replay =
            env->GetMethodID(threadClass, "replayPauseResume", "(Landroid/os/IBinder;)V");
    jclass mapClass = env->FindClass("android/util/ArrayMap");
    if (Cleared(env) || current == nullptr || activities == nullptr || replay == nullptr ||
        mapClass == nullptr) {
        return;
    }
    const jmethodID size = env->GetMethodID(mapClass, "size", "()I");
    const jmethodID keyAt = env->GetMethodID(mapClass, "keyAt", "(I)Ljava/lang/Object;");
    jobject thread = env->CallStaticObjectMethod(threadClass, current);
    jobject map = thread != nullptr ? env->GetObjectField(thread, activities) : nullptr;
    if (Cleared(env) || map == nullptr) return;
    // Tokens first: a replayed activity may change the map.
    std::vector<jobject> tokens;
    const jint count = env->CallIntMethod(map, size);
    for (jint i = 0; i < count && !Cleared(env); i++) {
        tokens.push_back(env->CallObjectMethod(map, keyAt, i));
    }
    for (jobject token : tokens) {
        env->CallVoidMethod(thread, replay, token);
        Cleared(env);
    }
    ALOGI("replayed pause/resume after a swapchain rebuild");
}

class Replayer : public android::MessageHandler {
public:
    explicit Replayer(JavaVM* vm) : vm_(vm) {}

    void handleMessage(const android::Message&) override {
        JNIEnv* env = nullptr;
        if (vm_->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return;
        env->PushLocalFrame(32);
        Replay(env);
        env->PopLocalFrame(nullptr);
    }

private:
    JavaVM* const vm_;
};

// Puts the replay on the main thread's looper.
void Post(JavaVM* vm, JNIEnv* env) {
    jclass looperClass = env->FindClass("android/os/Looper");
    if (looperClass == nullptr || Cleared(env)) return;
    const jmethodID mainLooper =
            env->GetStaticMethodID(looperClass, "getMainLooper", "()Landroid/os/Looper;");
    const jmethodID getQueue =
            env->GetMethodID(looperClass, "getQueue", "()Landroid/os/MessageQueue;");
    if (Cleared(env) || mainLooper == nullptr || getQueue == nullptr) return;
    jobject looper = env->CallStaticObjectMethod(looperClass, mainLooper);
    jobject queue = looper != nullptr ? env->CallObjectMethod(looper, getQueue) : nullptr;
    if (Cleared(env) || queue == nullptr) return;
    const android::sp<android::MessageQueue> mq =
            android::android_os_MessageQueue_getMessageQueue(env, queue);
    if (mq == nullptr || mq->getLooper() == nullptr) return;
    mq->getLooper()->sendMessage(android::sp<Replayer>::make(vm), android::Message());
}

}  // namespace

void ReplayResume() {
    JavaVM* vm = nullptr;
    jsize vms = 0;
    if (JNI_GetCreatedJavaVMs(&vm, 1, &vms) != JNI_OK || vms < 1) return;
    // Off the present thread: attaching costs a few milliseconds.
    std::thread([vm] {
        JNIEnv* env = nullptr;
        JavaVMAttachArgs args = {JNI_VERSION_1_6, "framefx-resume", nullptr};
        if (vm->AttachCurrentThread(&env, &args) != JNI_OK) return;
        Post(vm, env);
        vm->DetachCurrentThread();
    }).detach();
}

}  // namespace framefx
