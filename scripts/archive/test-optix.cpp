/**
 * Minimal OptiX verification program
 * Tests that OptiX headers are accessible and compile correctly
 *
 * Compile with:
 *   g++ -I$OPTIX_ROOT/include -c test-optix.cpp -o test-optix.o
 *
 * This is a compile-time only test - no linking or runtime execution required
 */

#include <optix.h>
#include <optix_stubs.h>
#include <optix_types.h>

// Test that basic OptiX types are available
void test_optix_types() {
    OptixDeviceContext context = nullptr;
    OptixResult result = OPTIX_SUCCESS;
    OptixModule module = nullptr;
    OptixProgramGroup program_group = nullptr;
    OptixPipeline pipeline = nullptr;

    // Silence unused variable warnings
    (void)context;
    (void)result;
    (void)module;
    (void)program_group;
    (void)pipeline;
}

// Test that OptiX version macros are defined
#ifndef OPTIX_VERSION
#error "OPTIX_VERSION not defined"
#endif

// Test that we can reference OptiX functions (not calling, just referencing)
void test_optix_functions() {
    // These are just type checks - function pointers
    auto init_func = &optixInit;
    auto get_error_func = &optixGetErrorName;
    auto get_string_func = &optixGetErrorString;

    (void)init_func;
    (void)get_error_func;
    (void)get_string_func;
}

int main() {
    // This program is designed to test compilation only
    // If it compiles successfully, OptiX headers are correctly installed
    test_optix_types();
    test_optix_functions();
    return 0;
}
