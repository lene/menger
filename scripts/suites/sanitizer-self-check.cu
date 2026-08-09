// Deliberately defective CUDA program, Sprint 36 E2. compute-sanitizer MUST flag this
// before memcheck.sh's real suite results count. NEVER "fix" this file — a clean
// sanitizer run here means the gate itself is broken, not that the code improved.
#include <cuda_runtime.h>

__global__ void oobWrite(int* buf) {
    buf[8] = 42;  // buf holds 4 ints -> index 8 is out of bounds (invalid write)
}

int main() {
    int* d_buf = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_buf), 4 * sizeof(int));
    oobWrite<<<1, 1>>>(d_buf);
    cudaDeviceSynchronize();
    cudaFree(d_buf);
    cudaFree(d_buf);  // deliberate double free (invalid free)
    return 0;
}
