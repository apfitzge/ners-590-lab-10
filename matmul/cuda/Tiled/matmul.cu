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
 * \param[in] N Number of tiles in row-direction of A and C.
 * \param[in] M Number of tiles in column-direction in A and rows in B.
 * \param[in] P Number of tiles column-direction in B and C.
 *
 *  Note: The matrix "tiles" and cuda "blocks" should have the same (square)
 *        dimensions.
 */
__global__ void matmul(double C[], const double A[], const double B[],
                       const int N, const int M, const int P) {
    // Block and thread indices
    const auto bx = blockIdx.x;
    const auto by = blockIdx.y;
    const auto tx = threadIdx.x;
    const auto ty = threadIdx.y;
    // Compute row and column in global matrix
    const auto row = by * BLOCK_SIZE + ty;
    const auto col = bx * BLOCK_SIZE + tx;

    // Start, stop, and step for loop over tiles
    const auto A_begin = row * M * BLOCK_SIZE + tx;
    const auto A_end   = A_begin + (M - 1) * BLOCK_SIZE;  // Inclusive
    const auto A_step  = BLOCK_SIZE;
    const auto B_begin = ty * P * BLOCK_SIZE + col;
    const auto B_step  = P * BLOCK_SIZE * BLOCK_SIZE;

    // Thead-local value to store computed C value
    double elem = 0.0;

    // Block-local memory for storing A & B tiles
    __shared__ double A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double B_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all tiles of A and B necessary for computing elem
    for(int a = A_begin, b = B_begin; a <= A_end; a += A_step, b += B_step) {
        // Load tiles of A & B. Each thread loads a single element
        A_tile[ty][tx] = A[a];
        B_tile[ty][tx] = B[b];

        // Synchronize threads - this guarantees the full tiles are loaded
        __syncthreads();

        // Multiply the tiles together, with each thread computing a single
        // element in the tile of C
        for(int idx = 0; idx < BLOCK_SIZE; ++idx)
            elem += A_tile[ty][idx] * B_tile[idx][tx];

        // Wait for threads to be done
        __syncthreads();
    }

    // Store the result in C
    C[row * P * BLOCK_SIZE + col] = elem;
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

    // Matrix dimensions in units of tiles
    const auto N_tiles = N / BLOCK_SIZE;
    const auto M_tiles = M / BLOCK_SIZE;
    const auto P_tiles = P / BLOCK_SIZE;
    if(N != N_tiles * BLOCK_SIZE ||
       M != M_tiles * BLOCK_SIZE ||
       P != P_tiles * BLOCK_SIZE) {
        printf("Dimensions must be multiple of block size (%d)!\n", BLOCK_SIZE);
        exit(1);
    }

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
    for(int row = 0, idx = 0; row < M; ++row)
        for(int col = 0; col < P; ++col, ++idx)
            h_B[idx] = static_cast<double>(col);

    // Transfer the matrices to the device
    gpuErrCheck( cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice) );
    gpuErrCheck( cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice) );

    auto t1 = std::chrono::high_resolution_clock::now();

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(P_tiles, N_tiles);

    for(int iter = 0; iter < repeat; ++iter) {
        // Call matmul kernel
        matmul<<<grid, block>>>(d_C, d_A, d_B, N_tiles, M_tiles, P_tiles);

        // // Wait for calculation to finish
        gpuErrCheck( cudaPeekAtLastError() );
        gpuErrCheck( cudaDeviceSynchronize() );
    }

    // Transfer the result matrix from device to host
    gpuErrCheck( cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost) );

    // End timer
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();

    // Check the result
    // C is ROW MAJOR
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