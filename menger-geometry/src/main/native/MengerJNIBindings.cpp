#include <jni.h>
#include "VideoLoader.h"
#include <iostream>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <dlfcn.h>
#include <link.h>

/**
 * JNI bindings for menger-geometry's native layer.
 *
 * As of Sprint 35 (AD-24) the 4D fractals no longer bind here: MengerRenderer
 * registers their shaders through optix-jni's generic custom-geometry SPI, so
 * this file only carries the VideoLoader bindings. libmengergeometry.so still
 * links CausticsRenderer, which calls OptiXWrapper symbols from liboptixjni.so;
 * it is built with --allow-shlib-undefined, and the constructor below promotes
 * liboptixjni.so (always loaded first) to RTLD_GLOBAL so those lazily bind.
 */

static int promoteCallback(struct dl_phdr_info* info, size_t /*size*/, void* /*data*/) {
    if (info->dlpi_name && std::strstr(info->dlpi_name, "optixjni")) {
        dlopen(info->dlpi_name, RTLD_LAZY | RTLD_GLOBAL);
        return 1;
    }
    return 0;
}

__attribute__((constructor))
static void promoteOptixJniToGlobal() {
    dl_iterate_phdr(promoteCallback, nullptr);
}

extern "C" {

static void throwJavaException(JNIEnv* env, const char* className, const std::string& message) {
    jclass exc = env->FindClass(className);
    if (exc != nullptr) {
        env->ThrowNew(exc, message.c_str());
    }
}

static menger::geometry::VideoLoader* getVideoLoader(JNIEnv* env, jlong handle) {
    if (handle == 0) {
        throwJavaException(env, "java/lang/IllegalStateException", "Video loader is closed");
        return nullptr;
    }
    return reinterpret_cast<menger::geometry::VideoLoader*>(handle);
}

JNIEXPORT jlong JNICALL Java_menger_geometry_VideoLoader_openVideoNative(
    JNIEnv* env, jobject /*obj*/, jstring path) {
    try {
        if (path == nullptr) {
            throwJavaException(env, "java/lang/IllegalArgumentException", "Video path is null");
            return 0;
        }

        const char* nativePath = env->GetStringUTFChars(path, nullptr);
        if (nativePath == nullptr) {
            throwJavaException(env, "java/lang/RuntimeException", "Failed to read video path");
            return 0;
        }

        try {
            auto* loader = new menger::geometry::VideoLoader(nativePath);
            env->ReleaseStringUTFChars(path, nativePath);
            return reinterpret_cast<jlong>(loader);
        } catch (...) {
            env->ReleaseStringUTFChars(path, nativePath);
            throw;
        }
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in openVideo: " << e.what() << std::endl;
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_menger_geometry_VideoLoader_videoWidthNative(
    JNIEnv* env, jobject /*obj*/, jlong handle) {
    auto* loader = getVideoLoader(env, handle);
    return loader == nullptr ? 0 : loader->width();
}

JNIEXPORT jint JNICALL Java_menger_geometry_VideoLoader_videoHeightNative(
    JNIEnv* env, jobject /*obj*/, jlong handle) {
    auto* loader = getVideoLoader(env, handle);
    return loader == nullptr ? 0 : loader->height();
}

JNIEXPORT jint JNICALL Java_menger_geometry_VideoLoader_frameCountNative(
    JNIEnv* env, jobject /*obj*/, jlong handle) {
    auto* loader = getVideoLoader(env, handle);
    return loader == nullptr ? 0 : loader->frameCount();
}

JNIEXPORT jdouble JNICALL Java_menger_geometry_VideoLoader_videoDurationSecondsNative(
    JNIEnv* env, jobject /*obj*/, jlong handle) {
    auto* loader = getVideoLoader(env, handle);
    return loader == nullptr ? 0.0 : loader->durationSeconds();
}

JNIEXPORT jdouble JNICALL Java_menger_geometry_VideoLoader_nativeFpsNative(
    JNIEnv* env, jobject /*obj*/, jlong handle) {
    auto* loader = getVideoLoader(env, handle);
    return loader == nullptr ? 0.0 : loader->nativeFps();
}

JNIEXPORT jbyteArray JNICALL Java_menger_geometry_VideoLoader_getFrameAtNative(
    JNIEnv* env, jobject /*obj*/, jlong handle, jdouble timestampSeconds) {
    try {
        auto* loader = getVideoLoader(env, handle);
        if (loader == nullptr) {
            return nullptr;
        }

        const auto frame = loader->frameAt(timestampSeconds);
        if (frame.rgba.size() > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
            throw std::runtime_error("Decoded video frame is too large for a JVM byte array");
        }

        jbyteArray result = env->NewByteArray(static_cast<jsize>(frame.rgba.size()));
        if (result == nullptr) {
            throw std::runtime_error("Failed to allocate JVM byte array for decoded video frame");
        }
        env->SetByteArrayRegion(
            result,
            0,
            static_cast<jsize>(frame.rgba.size()),
            reinterpret_cast<const jbyte*>(frame.rgba.data())
        );
        return result;
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in getFrameAt: " << e.what() << std::endl;
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_menger_geometry_VideoLoader_prefetchVideoNative(
    JNIEnv* env, jobject /*obj*/, jlong handle, jdouble timestampSeconds, jint nFrames) {
    try {
        auto* loader = getVideoLoader(env, handle);
        if (loader == nullptr) {
            return;
        }
        loader->prefetch(timestampSeconds, nFrames);
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in prefetchVideo: " << e.what() << std::endl;
        throwJavaException(env, "java/lang/RuntimeException", e.what());
    }
}

JNIEXPORT void JNICALL Java_menger_geometry_VideoLoader_closeVideoNative(
    JNIEnv* /*env*/, jobject /*obj*/, jlong handle) {
    delete reinterpret_cast<menger::geometry::VideoLoader*>(handle);
}

} // extern "C"
