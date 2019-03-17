# ners-590-lab-10

These are some exercises and solutions for an interactive lab class I led on November 9th 2018 for NERS-590 "Methods and Practice of Scientific Computing" at the University of Michigan.
The exercises were meant to give students a brief experience with cuda and Kokkos, following a lecture on Heterogeneous architectures which introduced both.

Solutions are not meant to be optimal, as these ideally should be completed by inexperienced students in the 1.5 hour lab.
Tiled-matrix examples are also attached as an example of how one might gain better performance.

Exercises:

1. dot-product: Implement parallel dot product (and appropriate function/kernel calls) using cuda and Kokkos.
2. matmul: Implement parallel (naive) matrix-matrix multiplication using cuda and Kokkos.
3. Laplace: Parallelize a Laplace program using Kokkos.
