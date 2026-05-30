#pragma once
#include "OptiXData.h"   // BaseParams, Menger4DData, Sierpinski4DData, Hexadecachoron4DData, CausticsParams

// Menger-specific launch parameter extension.
// Struct extension using standard C layout compatibility pattern.
// BaseParams MUST be the first member so that a pointer to MengerParams
// is also a valid pointer to BaseParams (offsetof(MengerParams, base) == 0).
//
// Usage:
//   Base shaders:   auto& p = *reinterpret_cast<BaseParams*>(optixGetLaunchParamsPointer())
//   4D shaders:     auto& p = *reinterpret_cast<MengerParams*>(optixGetLaunchParamsPointer())
//   JNI launch:     optixLaunch(..., sizeof(MengerParams), ...)
//
// Sprint 25: 4D data fields and CausticsParams move here from BaseParams.
// Until that migration completes, the base shader PTX (optix_shaders.ptx) still uses
// BaseParams; menger-geometry's PTX (optix_shaders_menger.ptx) uses MengerParams.
struct MengerParams {
    BaseParams base;                             // offset 0 — layout-compatible with BaseParams

    // 4D fractal geometry buffers (moved here from BaseParams in Sprint 25)
    Menger4DData*         menger4d_data        = nullptr;
    unsigned int          num_menger4d         = 0;
    Sierpinski4DData*     sierpinski4d_data    = nullptr;
    unsigned int          num_sierpinski4d     = 0;
    Hexadecachoron4DData* hexadecachoron4d_data = nullptr;
    unsigned int          num_hexadecachoron4d = 0;

    // Caustics (moved here from BaseParams in Sprint 25)
    CausticsParams*       caustics             = nullptr;
};

static_assert(offsetof(MengerParams, base) == 0,
              "BaseParams must be at offset 0 in MengerParams for layout compatibility");
