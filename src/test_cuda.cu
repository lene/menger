#include <iostream>

__global__ void kernel() {
    printf("Hello from CUDA kernel!\n");
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
