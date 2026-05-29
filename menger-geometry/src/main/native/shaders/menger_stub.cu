// Stub shader: validates MengerParams layout compiles with nvcc + OptiX.
// Replaced in Sprint 25 when 4D shaders move from optix-jni to menger-geometry.
#include <optix.h>
#include "MengerParams.h"

extern "C" __constant__ MengerParams params;

extern "C" __global__ void __raygen__menger_stub() {
    // volatile forces the read without generating "set but never used" warning
    volatile unsigned int n = params.num_menger4d;
    (void)n;
}
