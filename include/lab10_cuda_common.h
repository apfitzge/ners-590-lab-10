#pragma once
// Error checking in CUDA for allocation and data passing
// Copied from: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrCheck(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
        bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}