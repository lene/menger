#pragma once
//==============================================================================
// Per-instance data for menger-geometry's 4D fractal primitives (Sprint 35 1.2b).
//
// These structs used to live in optix-jni's OptiXData.h and were reached through
// dedicated BaseParams pointers. optix-jni is now strictly generic (1.2a): the 4D
// types moved here and travel through the generic custom-geometry SPI. Each 4D
// instance's blob is one of these structs, packed by MengerRenderer (Scala) and
// read back in the intersection shader via:
//
//   const T& d = *reinterpret_cast<const T*>(
//       static_cast<const char*>(params.custom_geometry_data)
//       + geometry_data_index * params.custom_geometry_stride);
//
// The Scala byte layout in MengerRenderer MUST match these definitions exactly
// (little-endian, no padding — every field is 4-byte aligned, 96 bytes total).
//==============================================================================

struct Menger4DData {
    float pos[3];          // 3D world position of the sponge center (12 bytes)
    float scale;           // World scale (projected coords multiplied by this) (4 bytes)
    float rotation4d[16];  // 4x4 rotation matrix in 4D, row-major (64 bytes)
    float eye_w;           // W coordinate of perspective eye point (4 bytes)
    float screen_w;        // W coordinate of projection screen (4 bytes)
    int   level;           // IFS recursion depth (4 bytes)
    int   dist_threshold;  // Generator keep predicate: abs-sum > dist_threshold (4 bytes)
    // Total: 96 bytes
};

struct Sierpinski4DData {
    float pos[3];          // 3D world position of the fractal center (12 bytes)
    float scale;           // World scale (projected coords multiplied by this) (4 bytes)
    float rotation4d[16];  // 4x4 rotation matrix in 4D, row-major (64 bytes)
    float eye_w;           // W coordinate of perspective eye point (4 bytes)
    float screen_w;        // W coordinate of projection screen (4 bytes)
    int   level;           // IFS recursion depth (4 bytes)
    float hit_bias;        // Added to reported t to let the fine instance win over the coarse (4 bytes)
    // Total: 96 bytes
};

struct Hexadecachoron4DData {
    float pos[3];          // 3D world position of the fractal center (12 bytes)
    float scale;           // World scale (projected coords multiplied by this) (4 bytes)
    float rotation4d[16];  // 4x4 rotation matrix in 4D, row-major (64 bytes)
    float eye_w;           // W coordinate of perspective eye point (4 bytes)
    float screen_w;        // W coordinate of projection screen (4 bytes)
    int   level;           // IFS recursion depth (4 bytes)
    float hit_bias;        // Added to reported t to let the fine instance win over the coarse (4 bytes)
    // Total: 96 bytes
};
