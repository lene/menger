// Full menger shader suite: base OptiX shaders + 4D fractal shaders + caustics.
// Compiled to optix_shaders_menger.ptx, loaded by MengerRenderer (menger-geometry)
// in place of optix_shaders.ptx from optix-jni.
//
// Sprint 25: params declared as BaseParams while 4D fields still live there.
// Once BaseParams is stripped of Menger-specific fields (task 25.3+), this
// declaration will switch to MengerParams.

#include <optix.h>
#include "OptiXData.h"
#include "VectorMath.h"

using namespace RayTracingConstants;

extern "C" {
    __constant__ BaseParams params;
}

// Base shader infrastructure — included from optix-jni shader directory
// (OPTIX_JNI_SHADER_DIR is added to include path in CMakeLists.txt)
#include "helpers.cu"
#include "raygen_primary.cu"
#include "miss_plane.cu"
#include "hit_sphere.cu"
#include "hit_triangle.cu"
#include "hit_cylinder.cu"
#include "hit_cone.cu"
#include "hit_plane.cu"

// 4D menger-specific shaders (local to menger-geometry)
#include "hit_menger4d.cu"
#include "hit_sierpinski4d.cu"
#include "hit_hexadecachoron4d.cu"

#include "shadows.cu"
#include "caustics_ppm.cu"
