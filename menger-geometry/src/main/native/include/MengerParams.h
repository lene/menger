#pragma once
#include "OptiXData.h"   // BaseParams from optix-jni

// Forward declarations — types still in optix-jni, moved in Sprint 25
struct Menger4DData;
struct Sierpinski4DData;
struct Hexadecachoron4DData;
struct CausticsParams;

// Struct extension using standard C layout compatibility pattern.
// BaseParams MUST be the first member so that a pointer to MengerParams
// is also a valid pointer to BaseParams (offsetof(MengerParams, base) == 0).
//
// Usage:
//   Base shaders:   auto& p = *reinterpret_cast<BaseParams*>(optixGetLaunchParamsPointer())
//   4D shaders:     auto& p = *reinterpret_cast<MengerParams*>(optixGetLaunchParamsPointer())
//   JNI launch:     optixLaunch(..., sizeof(MengerParams), ...)
struct MengerParams {
    BaseParams base;                             // offset 0 — layout-compatible with BaseParams

    // 4D fractal geometry buffers (populated in Sprint 25 when shaders move here)
    Menger4DData*         menger4d_data        = nullptr;
    unsigned int          num_menger4d         = 0;
    Sierpinski4DData*     sierpinski4d_data    = nullptr;
    unsigned int          num_sierpinski4d     = 0;
    Hexadecachoron4DData* hexadecachoron4d_data = nullptr;
    unsigned int          num_hexadecachoron4d = 0;

    // Caustics (pointer to avoid embedding CausticsParams before it moves to menger-geometry)
    CausticsParams*       caustics             = nullptr;
};

static_assert(offsetof(MengerParams, base) == 0,
              "BaseParams must be at offset 0 in MengerParams for layout compatibility");
