#include <chrono>               // Timing
#include <stdio.h>              // printf
#include <algorithm>            // Max (result checking)
#include <Kokkos_Core.hpp>

#include <iostream>

using matrix = Kokkos::View<double**>;
#define tile_len 8
#define team_size 64
// static constexpr const int tile_len = 8;                        // Tile dimension
// static constexpr const int team_size = tile_len * tile_len;     // Tile size

/**
 * \brief Perform tiled matrix multiplicat C = A * B.
 * \param[in] C The resulting matrix (N by P)
 * \param[in] A The left matrix (N by M)
 * \param[in] B The right matrix (M by P)
 * \param[in] N Number of rows in A and C
 * \param[in] M Number of columns in A and rows in B
 * \param[in] P Number of columns in B and C
 */
void matmul(matrix C, matrix A, matrix B, const int N, const int M, const int P) {
    // Call parallel for with teams
    using mdr_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    using member_type = Kokkos::TeamPolicy<>::member_type;
    using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using shared_tile = Kokkos::View<double[tile_len][tile_len], ScratchSpace,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // Team sizes
    const auto num_teams = (N / tile_len) * (P / tile_len);
    const auto shared_size = shared_tile::shmem_size();
    const Kokkos::TeamPolicy<> team_policy(num_teams, team_size);
    // Get matrix dimensions in terms of tiles
    const auto tM = M / tile_len;
    const auto tP = P / tile_len;
    // Launch teams
    Kokkos::parallel_for(team_policy
        .set_scratch_size(0, Kokkos::PerTeam(2*shared_size)),
        KOKKOS_LAMBDA (const member_type &team) {
            // Team tiles for A and B
            shared_tile tA(team.team_scratch(0), tile_len, tile_len);
            shared_tile tB(team.team_scratch(0), tile_len, tile_len);
            // Local & Global indices
            const auto lIdx = team.team_rank();
            const auto lrow = lIdx / tile_len;
            const auto lcol = lIdx % tile_len;
            const auto gIdx = team.league_rank();
            const auto row = tile_len * (gIdx / tP) + lrow;
            const auto col = tile_len * (gIdx % tP) + lcol;
            // Value for C(row, col)
            double elem = 0.0;
            // Loop through tiles for computing this element
            for(int tIdx = 0; tIdx < tM; ++tIdx) {
                const auto offset = tIdx * tile_len; // Offset
                // Load in element of tiles in A, B
                tA(lrow, lcol) = A(row, lcol + offset);
                tB(lrow, lcol) = B(lrow + offset, col);
                // Wait for all theads
                team.team_barrier();
                // Perform multiplication of this
                for(int idx = 0; idx < tile_len; ++idx)
                    elem += tA(lrow, idx) * tB(idx, lcol);
                // Wait for threads
                team.team_barrier();
            }
            // Set the element
            C(row,col) = elem;
        }
    );
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

    // Matrix dimensions in units of tiles
    const auto N_tiles = N / tile_len;
    const auto M_tiles = M / tile_len;
    const auto P_tiles = P / tile_len;
    if(N != N_tiles * tile_len ||
       M != M_tiles * tile_len ||
       P != P_tiles * tile_len) {
        printf("Dimensions must be multiple of block size (%d)!\n", tile_len);
        exit(1);
    }

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
            matmul(C, A, B, N, M, P);
            // Wait for kernel to finish
            Kokkos::fence();
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
        } else {

            // Compute FLOPs
            double FLOPs = 2 * double(N) * double(M) * double(P) * double(repeat);
            double GFLOPS = 1.0e-9 * FLOPs / time;

            printf("Problem:\n");
            printf("  Dimensions - N(%d) M(%d) P(%d) repeated %d times\n", N, M, P, repeat);
            printf("  operations=( %g ) time=( %g s ) GFLOPs=( %g )\n", FLOPs, time, GFLOPS);
        }
    }
    Kokkos::finalize();
}