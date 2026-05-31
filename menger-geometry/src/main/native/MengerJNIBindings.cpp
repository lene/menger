#include <jni.h>
#include "OptiXWrapper.h"
#include <iostream>
#include <cstring>
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
        return nullptr;
    }
    jlong handle = env->GetLongField(obj, fid);
    return reinterpret_cast<OptiXWrapper*>(handle);
}

JNIEXPORT jint JNICALL Java_io_github_lene_optix_MengerRenderer_addRecursiveIASSpongeInstanceNative(
    JNIEnv* env, jobject obj,
    jint level,
    jfloatArray transform, jfloat r, jfloat g, jfloat b, jfloat a, jfloat ior,
    jfloat roughness, jfloat metallic, jfloat specular, jfloat emission,
    jint textureIndex, jfloat filmThickness) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;

        jsize transformLen = env->GetArrayLength(transform);
        if (transformLen != 12) {
            jclass exc = env->FindClass("java/lang/IllegalArgumentException");
            std::string msg = "Transform array must have 12 elements (4x3 matrix), got " +
                std::to_string(transformLen);
            env->ThrowNew(exc, msg.c_str());
            return -1;
        }
        jfloat* transformArr = env->GetFloatArrayElements(transform, nullptr);
        if (transformArr == nullptr) {
            jclass exc = env->FindClass("java/lang/RuntimeException");
            env->ThrowNew(exc, "Failed to get transform array elements");
            return -1;
        }
        int instanceId = wrapper->addRecursiveIASSpongeInstance(
            level, transformArr, r, g, b, a, ior,
            roughness, metallic, specular, emission, textureIndex, filmThickness
        );
        env->ReleaseFloatArrayElements(transform, transformArr, 0);
        return instanceId;
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addRecursiveIASSpongeInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exc, e.what());
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
    jfloat filmThickness) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->addMenger4DInstance(
            (int)level, (int)distThreshold,
            x, y, z, scale, eyeW, screenW, rotXW, rotYW, rotZW,
            r, g, b, a, ior, roughness, metallic, specular, emission, filmThickness
        );
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addMenger4DInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exc, e.what());
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
        env->ThrowNew(exc, e.what());
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
    jfloat filmThickness) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->addSierpinski4DInstance(
            (int)level, x, y, z, scale, eyeW, screenW, rotXW, rotYW, rotZW,
            r, g, b, a, ior, roughness, metallic, specular, emission, filmThickness
        );
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addSierpinski4DInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exc, e.what());
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
        env->ThrowNew(exc, e.what());
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
    jfloat filmThickness) {
    try {
        OptiXWrapper* wrapper = getWrapper(env, obj);
        if (wrapper == nullptr) return -1;
        return wrapper->addHexadecachoron4DInstance(
            (int)level, x, y, z, scale, eyeW, screenW, rotXW, rotYW, rotZW,
            r, g, b, a, ior, roughness, metallic, specular, emission, filmThickness
        );
    } catch (const std::exception& e) {
        std::cerr << "[MengerJNI] Error in addHexadecachoron4DInstance: " << e.what() << std::endl;
        jclass exc = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exc, e.what());
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
        env->ThrowNew(exc, e.what());
        return -1;
    }
}

} // extern "C"
