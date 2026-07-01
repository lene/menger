#include <jni.h>
#include "OptiXWrapper.h"
#include "VideoLoader.h"
#include <iostream>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <dlfcn.h>
#include <link.h>

/**
 * JNI bindings for MengerRenderer Scala class (io.github.lene.optix.MengerRenderer).
 *
 * MengerRenderer extends OptiXRenderer and overrides the 4D geometry @native methods.
 * The nativeHandle field is inherited from OptiXRenderer; JNI GetFieldID walks the
 * superclass hierarchy to find it.
 *
 * libmengergeometry.so is built with --allow-shlib-undefined so it has no link-time
 * dependency on liboptixjni.so. At load time the constructor below finds liboptixjni.so
 * in the already-loaded library list (it is always loaded first by OptiXRenderer) and
 * promotes it to RTLD_GLOBAL so that lazy symbol binding for OptiXWrapper calls succeeds.
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

static OptiXWrapper* getWrapper(JNIEnv* env, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, "nativeHandle", "J");
    if (fid == nullptr) {
        std::cerr << "[MengerJNI] Failed to get nativeHandle field" << std::endl;
        env->ExceptionClear();  // GetFieldID sets a pending NoSuchFieldError; clear it so
        return nullptr;         // returning nullptr to caller doesn't trigger JNI UB
    }
    jlong handle = env->GetLongField(obj, fid);
    return reinterpret_cast<OptiXWrapper*>(handle);
}

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

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_addRecursiveIASSpongeInstanceNative(
    JNIEnv* env, jobject obj,
    jint level,
    jfloatArray transform, jfloat r, jfloat g, jfloat b, jfloat a, jfloat ior,
    jfloat roughness, jfloat metallic, jfloat specular, jfloat emission,
    jint textureIndex, jfloat filmThickness,
    jfloat cauchy_a, jfloat cauchy_b) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;

        jsize transformLen = env->GetArrayLength(transform);
        if (transformLen != 12) {
            jclass exc = env->FindClass("java/lang/IllegalArgumentException");
            std::string msg = "Transform array must have 12 elements (4x3 matrix), got " +
                std::to_string(transformLen);
            if (exc) env->ThrowNew(exc, msg.c_str());
            return -1;
        }
        jfloat* transformArr = env->GetFloatArrayElements(transform, nullptr);
        if (transformArr == nullptr) {
            jclass exc = env->FindClass("java/lang/RuntimeException");
            if (exc) env->ThrowNew(exc, "Failed to get transform array elements");
            return -1;
        }
        int instanceId = -1;
        try {
            instanceId = wrapper->addRecursiveIASSpongeInstance(
                level, transformArr, r, g, b, a, ior,
                roughness, metallic, specular, emission, textureIndex, filmThickness,
                cauchy_a, cauchy_b
            );
        } catch (...) {
            env->ReleaseFloatArrayElements(transform, transformArr, JNI_ABORT);
            throw;
        }
        env->ReleaseFloatArrayElements(transform, transformArr, 0);
        return instanceId;
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addRecursiveIASSpongeInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        if (exc) env->ThrowNew(exc, e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_addMenger4DInstanceNative(
    JNIEnv* env, jobject obj,
    jint level, jint distThreshold,
    jfloat x, jfloat y, jfloat z, jfloat scale,
    jfloat eyeW, jfloat screenW,
    jfloat rotXW, jfloat rotYW, jfloat rotZW,
    jfloat r, jfloat g, jfloat b, jfloat a, jfloat ior,
    jfloat roughness, jfloat metallic, jfloat specular, jfloat emission,
    jfloat filmThickness,
    jfloat cauchy_a, jfloat cauchy_b) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->addMenger4DInstance(
            (int)level, (int)distThreshold,
            x, y, z, scale, eyeW, screenW, rotXW, rotYW, rotZW,
            r, g, b, a, ior, roughness, metallic, specular, emission, filmThickness,
            cauchy_a, cauchy_b
        );
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addMenger4DInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        if (exc) env->ThrowNew(exc, e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_updateMenger4DProjectionNative(
    JNIEnv* env, jobject obj,
    jint instanceId,
    jfloat eyeW, jfloat screenW,
    jfloat rotXW, jfloat rotYW, jfloat rotZW) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->updateMenger4DProjection(instanceId, eyeW, screenW, rotXW, rotYW, rotZW);
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in updateMenger4DProjection: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        if (exc) env->ThrowNew(exc, e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_addSierpinski4DInstanceNative(
    JNIEnv* env, jobject obj,
    jint level,
    jfloat x, jfloat y, jfloat z, jfloat scale,
    jfloat eyeW, jfloat screenW,
    jfloat rotXW, jfloat rotYW, jfloat rotZW,
    jfloat r, jfloat g, jfloat b, jfloat a, jfloat ior,
    jfloat roughness, jfloat metallic, jfloat specular, jfloat emission,
    jfloat filmThickness,
    jfloat cauchy_a, jfloat cauchy_b) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->addSierpinski4DInstance(
            (int)level, x, y, z, scale, eyeW, screenW, rotXW, rotYW, rotZW,
            r, g, b, a, ior, roughness, metallic, specular, emission, filmThickness,
            cauchy_a, cauchy_b
        );
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addSierpinski4DInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        if (exc) env->ThrowNew(exc, e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_updateSierpinski4DProjectionNative(
    JNIEnv* env, jobject obj,
    jint instanceId,
    jfloat eyeW, jfloat screenW,
    jfloat rotXW, jfloat rotYW, jfloat rotZW) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->updateSierpinski4DProjection(instanceId, eyeW, screenW, rotXW, rotYW, rotZW);
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in updateSierpinski4DProjection: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        if (exc) env->ThrowNew(exc, e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_addHexadecachoron4DInstanceNative(
    JNIEnv* env, jobject obj,
    jint level,
    jfloat x, jfloat y, jfloat z, jfloat scale,
    jfloat eyeW, jfloat screenW,
    jfloat rotXW, jfloat rotYW, jfloat rotZW,
    jfloat r, jfloat g, jfloat b, jfloat a, jfloat ior,
    jfloat roughness, jfloat metallic, jfloat specular, jfloat emission,
    jfloat filmThickness,
    jfloat cauchy_a, jfloat cauchy_b) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->addHexadecachoron4DInstance(
            (int)level, x, y, z, scale, eyeW, screenW, rotXW, rotYW, rotZW,
            r, g, b, a, ior, roughness, metallic, specular, emission, filmThickness,
            cauchy_a, cauchy_b
        );
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addHexadecachoron4DInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        if (exc) env->ThrowNew(exc, e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_updateHexadecachoron4DProjectionNative(
    JNIEnv* env, jobject obj,
    jint instanceId,
    jfloat eyeW, jfloat screenW,
    jfloat rotXW, jfloat rotYW, jfloat rotZW) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->updateHexadecachoron4DProjection(instanceId, eyeW, screenW, rotXW, rotYW, rotZW);
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in updateHexadecachoron4DProjection: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        if (exc) env->ThrowNew(exc, e.what());
        return -1;
    }
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
