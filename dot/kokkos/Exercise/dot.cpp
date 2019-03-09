#include <chrono>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

using vec = Kokkos::DualView<double*>;

/** Main function */
int main(int argc, char** argv) {
    // Read input
    int N = 0;
    if(argc >= 2) {
        N = atoi(argv[1]);
    } else {
        printf("Enter the vector length N.\n");
    }

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Allocate our vectors
        vec A("A", N);
        vec B("B", N);

        // Set up calculation
        // Initialize A and B on the host
        for(int idx = 0; idx < N; ++idx) {
            A.h_view(idx) = 2.0;
            B.h_view(idx) = 3.0;
        }
        // Mark host views as modified so sync knows which way to transfer
        A.modify_host();
        B.modify_host();
        A.sync_device();
        B.sync_device();

        // Timing
        // Begin timer
        auto t1 = std::chrono::high_resolution_clock::now();

        // Implement parallel dot-product here
        double dot_product = 0.0;

        // Stop timer
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::duration<double>>
                (t2-t1).count();

        // // Synchronize device/host memory
        A.modify_device();
        B.modify_device();
        A.sync_host();
        B.sync_host();

        // Compute FLOPs
        double FLOPs = 2 * static_cast<double>(N);
        double GFLOPS = 1.0e-9 * FLOPs / time;
        printf("Problem:\n");
        printf("  Result: %lf\n", dot_product);
        printf("  Dimensions - N(%d)\n", N);
        printf("  operations=( %g ) time=( %g s ) GFLOPs=( %g )\n", FLOPs, time, GFLOPS);
    }
    Kokkos::finalize();
}