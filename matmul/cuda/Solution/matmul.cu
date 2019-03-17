#include <chrono>               // Timing
#include <stdio.h>              // printf
#include <algorithm>            // Max (result checking)
#include "lab10_cuda_common.h"  // Error checking macro

static constexpr const int BLOCK_SIZE = 16;

/**
 * \brief Perform matrix multiplicat C = A * B.
 * \param[in] C The resulting matrix (N by P)
 * \param[in] A The left matrix (N by M)
 * \param[in] B The right matrix (M by P)
 * \param[in] N Number of rows in A and C.
 * \param[in] M Number of columns in A and rows in B.
 * \param[in] P Number of columns in B and C.
 */
__global__ void matmul(double C[], const double A[], const double B[],
                       const int N, const int M, const int P) {
    // Compute row and column in global matrix
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    // If the element is in the matrix
    if(row < N && col < P) {
        // Local variable to store element in C at (row,col)
        double elem = 0.0;
        // Loop over the columns in A (rows of B)
        for(int idx = 0, idx_A = row * N, idx_B = col; idx < M;
                ++idx, ++idx_A, idx_B += N)
            elem += A[idx_A] * B[idx_B];

        // Store the result in C
        C[row * N + col] = elem;
    }
}

/** Main routine */
int main(int argc, char** argv) {
    // Read input
    if(argc < 4) {
        printf("Must enter matrix dimensions: N, M, P!\n");
        exit(1);
    }
    const int N = atoi(argv[1]);
    const int M = atoi(argv[2]);
    const int P = atoi(argv[3]);
    const int repeat = (argc >= 5) ? atoi(argv[4]) : 1;

    // Get matrix sizes
    const auto A_size = N * M * sizeof(double);
    const auto B_size = M * P * sizeof(double);
    const auto C_size = N * P * sizeof(double);

    // Allocate host-side memory
    double* const h_A = (double*) malloc(A_size);
    double* const h_B = (double*) malloc(B_size);
    double* const h_C = (double*) malloc(C_size);

    // Allocate device-side memory
    double *d_A, *d_B, *d_C;
    gpuErrCheck( cudaMalloc(&d_A, A_size) );
    gpuErrCheck( cudaMalloc(&d_B, B_size) );
    gpuErrCheck( cudaMalloc(&d_C, C_size) );

    // Initialize host-side memory
    // A - each row has row index (as double)
    for(int row = 0, idx = 0; row < N; ++row)
        for(int col = 0; col < M; ++col, ++idx)
            h_A[idx] = static_cast<double>(row);
    // B - each col has col index (as double)
    for(int row = 0, idx = 0; row < N; ++row)
        for(int col = 0; col < P; ++col, ++idx)
            h_B[idx] = static_cast<double>(col);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Transfer the matrices to the device
    gpuErrCheck( cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice) );
    gpuErrCheck( cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice) );

    const auto num_blocks_x = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto num_blocks_y = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(num_blocks_x, num_blocks_y);

    for(int iter = 0; iter < repeat; ++iter) {
        // Call matmul kernel
        matmul<<<grid, block>>>(d_C, d_A, d_B, N, M, P);

        // Wait for calculation to finish
        gpuErrCheck( cudaPeekAtLastError() );
        gpuErrCheck( cudaDeviceSynchronize() );
    }

    // Transfer the result matrix from device to host
    gpuErrCheck( cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost) );

    // End timer
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();

    // Check the result
    double maxError = 0.0;
    double db_A_cols = static_cast<double>(M);
    for(int row = 0, idx = 0; row < N; ++row) {
      for(int col = 0; col < P; ++col, ++idx) {
        double expected = db_A_cols * row * col;
        maxError = std::max(maxError, std::abs(expected - h_C[idx]));
      }
    }
    if(maxError > 1.0e-8){
      printf(" Result does not match!\n");
      exit(1);
    }

    // Compute FLOPs
    double FLOPs = 2 * double(N) * double(M) * double(P) * double(repeat);
    double GFLOPS = 1.0e-9 * FLOPs / time;

    printf("Problem:\n");
    printf("  Dimensions - N(%d) M(%d) P(%d) repeated %d times\n", N, M, P, repeat);
    printf("  operations=( %g ) time=( %g s ) GFLOPs=( %g )\n", FLOPs, time, GFLOPS);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    gpuErrCheck( cudaFree(d_A) );
    gpuErrCheck( cudaFree(d_B) );
    gpuErrCheck( cudaFree(d_C) );
}