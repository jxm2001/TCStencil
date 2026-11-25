#include "box_kernel.cuh"

extern "C" __global__ void mma_run(const float* __restrict__ input,
                                  const float* __restrict__ coefficients,
                                  const float* __restrict__ weights,
                                  float* __restrict__ output, int padded_size,
                                  int tile_size, const int* __restrict__ halo_offsets) {
    tcstencil_tf32::box_stencil<2>(input, coefficients, weights, output,
                                  padded_size, tile_size, halo_offsets);
}
