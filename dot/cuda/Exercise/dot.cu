#include <chrono>       // Timing
#include <stdio.h>      // printf
#include <cmath>        // AtomicAdd (hint)

// Block size
static constexpr const int BLOCK_SIZE = 256;

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
/**
 * \brief Cuda kernel to compute the dot-product of two vectors of length N
 * \param[out] ans The result
 * \param[in] A The first vector
 * \param[in] B The second vector
 * \param[in] N The vector length
 */
__global__ void dot(double* ans, const double A[], const double B[], const int N) {
    // Implement a dot-product routine
}

/** Main routine */
int main(int argc, char** argv) {
    // Read input
    int N = 0;
    if(argc >= 2) {
        N = atoi(argv[1]);
    } else {
        printf("Enter the vector length N.\n");
    }

    // Set up calculation
    const int vs = N * sizeof(double);

    // Allocate memory on the host (CPU)
    double* const h_A = (double*) malloc(vs);
    double* const h_B = (double*) malloc(vs);
    double* const h_p = (double*) malloc(sizeof(double));
    *h_p = 0.0;

    // Allocate memory on the device (GPU)
    double *d_A, *d_B, *d_p;
    gpuErrCheck( cudaMalloc(&d_A, vs) );
    gpuErrCheck( cudaMalloc(&d_B, vs) );
    gpuErrCheck( cudaMalloc(&d_p, vs) );

    // Initialize A and B on the host
    for(int idx = 0; idx < N; ++idx) {
        h_A[idx] = 2.0;
        h_B[idx] = 3.0;
    }
    // Begin timer
    auto t1 = std::chrono::high_resolution_clock::now();

    // Call the dot-product kernel
    //      Hint: You should think about where the memory of each variable lives

    // End timer
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();

    // Compute FLOPs
    double FLOPs = 2 * static_cast<double>(N);
    double GFLOPS = 1.0e-9 * FLOPs / time;
    printf("Problem:\n");
    printf("  Result: %lf\n", *h_p);
    printf("  Dimensions - N(%d)\n", N);
    printf("  operations=( %g ) time=( %g s ) GFLOPs=( %g )\n", FLOPs, time, GFLOPS);

    // Free memory
    free(h_A);
    free(h_B);
    gpuErrCheck( cudaFree(d_A) );
    gpuErrCheck( cudaFree(d_B) );
}