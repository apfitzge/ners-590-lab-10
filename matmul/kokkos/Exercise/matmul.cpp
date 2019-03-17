#include <chrono>               // Timing
#include <stdio.h>              // printf
#include <algorithm>            // Max (result checking)
#include <Kokkos_Core.hpp>

#include <iostream>

using matrix = Kokkos::View<double**>;

/**
 * \brief Perform matrix multiplicat C = A * B.
 * \param[in] C The resulting matrix (N by P)
 * \param[in] A The left matrix (N by M)
 * \param[in] B The right matrix (M by P)
 * \param[in] N Number of rows in A and C
 * \param[in] M Number of columns in A and rows in B
 * \param[in] P Number of columns in B and C
 */
void matmul(matrix C, matrix A, matrix B, const int N, const int M, const int P) {
    // Call parallel (multidimensional) for loop
    using mdr_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    // Implement naive matrix-matrix multiplication
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

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Device views
        matrix A("A", N, M);
        matrix B("B", M, P);
        matrix C("C", N, P);
        // Host mirrors
        auto h_A = Kokkos::create_mirror_view(A);
        auto h_B = Kokkos::create_mirror_view(B);
        auto h_C = Kokkos::create_mirror_view(C);

        // Initialize values of A and B on the host
        for(int row = 0; row < N; ++row)
            for(int col = 0; col < M; ++col)
                h_A(row, col) = static_cast<double>(row);
        for(int row = 0; row < M; ++row)
            for(int col = 0; col < P; ++col)
                h_B(row, col) = static_cast<double>(col);

        // Copy onto device
        Kokkos::deep_copy(A, h_A);
        Kokkos::deep_copy(B, h_B);

        // Begin timer
        auto t1 = std::chrono::high_resolution_clock::now();

        for(int iter = 0; iter < repeat; ++iter) {
            // Call matmul kernel
        }

        // End timer
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::duration<double>>
                (t2-t1).count();

        // Transfer the result matrix from device to host
        Kokkos::deep_copy(h_C, C);

        // Check the result
        double maxError = 0.0;
        double db_A_cols = static_cast<double>(M);
        for(int row = 0, idx = 0; row < N; ++row) {
          for(int col = 0; col < P; ++col, ++idx) {
            double expected = db_A_cols * row * col;
            maxError = std::max(maxError, std::abs(expected - h_C(row, col)));
          }
        }
        if(maxError > 1.0e-8) {
            printf(" Result does not match!\n");
            exit(1);
        }

        // Compute FLOPs
        double FLOPs = 2 * double(N) * double(M) * double(P) * double(repeat);
        double GFLOPS = 1.0e-9 * FLOPs / time;

        printf("Problem:\n");
        printf("  Dimensions - N(%d) M(%d) P(%d) repeated %d times\n", N, M, P, repeat);
        printf("  operations=( %g ) time=( %g s ) GFLOPs=( %g )\n", FLOPs, time, GFLOPS);
    }
    Kokkos::finalize();
}