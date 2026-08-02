//==============================================================================
// menger-geometry's registrable 4D-fractal OptiX module (Sprint 35 Task 1.2b).
//
// Compiled to menger_4d.ptx. MengerRenderer registers this module through
// optix-jni's generic custom-geometry SPI (registerCustomGeometry): it supplies
// ONLY the 4D intersection / closest-hit / shadow programs plus the device helpers
// they call. Everything else — raygen, miss, the built-in primitives, the SBT, the
// pipeline — belongs to optix-jni's base module (optix_shaders.ptx). This is the
// permanent successor to the old optix_shaders_menger.cu superset, which forked
// optix-jni's entire shader layer (Sprint 25 scaffolding, now deleted).
//
// helpers.cu holds every device helper the 4D closest hits use
// (getInstanceMaterialPBR, trace*/handle* radiance, Fresnel, Beer-Lambert, shadow
// accumulation). params is declared here as generic BaseParams; the 4D per-instance
// data arrives through params.custom_geometry_data (see MengerGeometryData.h), not
// any Menger-specific pointer — optix-jni no longer knows these types.
//==============================================================================

#include <optix.h>
#include "OptiXData.h"          // BaseParams, InstanceMaterial, RayTracingConstants
#include "VectorMath.h"
#include "MengerGeometryData.h" // Menger4DData / Sierpinski4DData / Hexadecachoron4DData

using namespace RayTracingConstants;

extern "C" {
    __constant__ BaseParams params;
}

// Device-helper infrastructure shared with optix-jni's base shaders. helpers.cu
// exports plain `__device__` helpers, a `__constant__` CIE table, and one
// `__intersection__sphere` entry — all of which optix-jni's base module
// (optix_shaders.ptx) ALSO exports. Two modules exporting the same symbols into one
// pipeline is OPTIX_ERROR_PIPELINE_LINK_ERROR ("defined multiple times"). Give this
// module's copies internal linkage (anonymous namespace → module-local .func in PTX,
// invisible to the cross-module linker), and rename the unused sphere entry so it
// can't collide with the base module's live one. The 4D shaders below stay at file
// scope and see these helpers unqualified (same translation unit).
#define __intersection__sphere __intersection__sphere_menger_unused
namespace {
#include "helpers.cu"
}
#undef __intersection__sphere

// The 4D fractal shaders (intersection + closest hit + shadow programs).
#include "hit_menger4d.cu"
#include "hit_sierpinski4d.cu"
#include "hit_hexadecachoron4d.cu"
