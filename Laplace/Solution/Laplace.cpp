/*************************************************
* Laplace Parallel C++ Version
*
* Temperature is initially 0.0
* Boundaries are as follows:
*
*      0         T         0
*   0  +-------------------+  0
*      |                   |
*      |                   |
*      |                   |
*   T  |                   |  T
*      |                   |
*      |                   |
*      |                   |
*   0  +-------------------+ 100
*      0         T        100
*
* Derived from program provided by:
*  John Urbanic, PSC 2014
*
************************************************/
#include <chrono>               // Timing
#include <Kokkos_Core.hpp>

// Error allowed in temperature
#define MAX_TEMP_ERROR 0.01
#define MAX_ITERATIONS 4000

// Type alias for plate temperatures
typedef Kokkos::View<double**> temperature;

// Helper routines
void initialize(const int rows, const int cols, temperature T);
void track_progress(const int rows, const int cols,
                    const int iter, const temperature T);

int main(int argc, char* argv[]) {
    // Check input
    if(argc < 3) {
        printf("Input rows and columns!\n");
        return -1;
    }
    // Read input
    const auto rows = atoi(argv[1]);
    const auto cols = atoi(argv[2]);

    Kokkos::initialize(argc, argv);
    {
        // Begin timer
        auto t1 = std::chrono::high_resolution_clock::now();

        // Allocate containers
        temperature T("T", rows+2, cols+2);
        temperature T_prev("prev", rows+2, cols+2);

        // Load initial conditions
        initialize(rows, cols, T_prev);

        // Iterations
        int iter = 0;
        double dT = 100.0;
        using mdr_policy = Kokkos::Experimental::MDRangePolicy<Kokkos::Rank<2>>;
        while(dT > MAX_TEMP_ERROR && iter <= MAX_ITERATIONS) {
            Kokkos::parallel_for("computation",
                mdr_policy( {1, 1}, {rows+1, cols+1} ),
                KOKKOS_LAMBDA (const int &row, const int &col) {
                    T(row, col) = 0.25 * (
                            T_prev(row+1, col) + T_prev(row-1, col)
                            + T_prev(row, col+1) + T_prev(row, col-1) );
                }
            );
            Kokkos::fence();

            // Copy the grid to the old grid and determine largest temperature change
            dT = 0.0;
            Kokkos::parallel_reduce("dT",
                mdr_policy( {1, 1}, {rows+1, cols+1} ),
                KOKKOS_LAMBDA (const int &row, const int &col, double &ldT){
                    ldT = max(ldT, abs(T(row, col) - T_prev(row, col)));
                    T_prev(row, col) = T(row, col);
                },
                Kokkos::Max<double>(dT)
            );

            // Track progress periodically
            if((iter % 100) == 0) track_progress(rows, cols, iter, T);
            ++iter;
        }
        // Get end time
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::duration<double>>
                (t2-t1).count();
        printf("\n");
        printf("Maximum error at iteration %d was %lf\n", iter-1, dT);
        printf("Total runtime was %lf seconds.\n", time);
    }
    Kokkos::finalize();

    return 0;
}

// Initialize the temperature of the grid
// All zero except boundary conditions
void initialize(const int rows, const int cols, temperature prev) {
  // Initialize all to zero save last row/column
  for(int row = 0; row < rows+1; ++row) {
    for(int col = 0; col < cols+1; ++col) {
      prev(row, col) = 0.0;
    }
  }
  // Initialize boundary conditions
  // Left side is set to zero
  // Right side linearly increases from 0 to 100
  for(int row = 0; row <= rows+1; ++row){
    prev(row, 0) = 0.0;
    prev(row, cols+1) = (100.0 / rows) * row;
  }
  // Top side is set to zero
  // Bottom side linearly increases from 0 to 100
  for(int col = 0; col < cols+1; ++col){
    prev(0, col) = 0.0;
    prev(rows+1, col) = (100.0 / cols)*col;
  }
}

// Track progress
void track_progress(const int rows, const int cols,
                    const int iter, temperature T){
  printf("---------- Iteration number: %d ------------\n", iter);
  for(int i = rows-5; i <= rows; ++i){
    printf("[%d,%d]: %5.2f  ", i, i, T(i,i) );
  }
  printf("\n");
}