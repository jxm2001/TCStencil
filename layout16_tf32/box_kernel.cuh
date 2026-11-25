#pragma once

#include "wmma.cuh"

namespace tcstencil_tf32 {

template<int Radius>
__device__ __forceinline__ void box_stencil(
    const float* __restrict__ input, const float* __restrict__ coefficients,
    const float* __restrict__ weights, float* __restrict__ output,
    int padded_size, int tile_size, const int* __restrict__ halo_offsets) {
    constexpr int width = 2 * Radius + 1;
    constexpr int rows = 16 + 2 * Radius;
    __shared__ __align__(32) float data[rows * 16];
    __shared__ __align__(32) float halo[2 * Radius * rows];
    const int lane = threadIdx.x + threadIdx.y * 16;
    const int base = (blockIdx.y + 1) * 16 * padded_size +
                     (blockIdx.x * tile_size + 1) * 256;
    MatrixB parameter[width][2];
    #pragma unroll
    for (int z = 0; z < width; ++z)
        #pragma unroll
        for (int part = 0; part < 2; ++part)
            load_tf32(parameter[z][part], coefficients + z * 256 + part * 128);

    for (int tile = 0; tile < tile_size; ++tile) {
        const int offset = base + tile * 256;
        // The input window has exactly Radius rows above and below the brick.
        copy_brick(data + Radius * 16, input + offset, lane);
        for (int i = lane; i < Radius * 16; i += 32) {
            data[i] = input[offset - padded_size * 16 + (16 - Radius) * 16 + i];
            data[(16 + Radius) * 16 + i] = input[offset + padded_size * 16 + i];
        }
        for (int i = lane; i < 2 * Radius * rows; i += 32)
            halo[i] = input[offset + halo_offsets[i]];
        __syncwarp();

        Accumulator result;
        wmma::fill_fragment(result, 0.0f);
        #pragma unroll
        for (int z = 0; z < width; ++z) {
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                MatrixA values;
                load_tf32(values, data + z * 16 + part * 8);
                wmma::mma_sync(result, values, parameter[z][part], result);
            }
        }
        wmma::store_matrix_sync(data, result, 16, wmma::mem_row_major);
        __syncwarp();

        const int row = threadIdx.x;
        const int side = threadIdx.y;
        #pragma unroll
        for (int edge = 0; edge < Radius; ++edge) {
            const int col = side ? 16 - Radius + edge : edge;
            float value = data[row * 16 + col];
            #pragma unroll
            for (int z = 0; z < width; ++z) {
                #pragma unroll
                for (int h = 0; h < Radius; ++h) {
                    const int dx = (side ? 16 + h : h - Radius) - col;
                    if (dx >= -Radius && dx <= Radius)
                        value = fmaf(halo[(side * Radius + h) * rows + row + z],
                                     weights[z * width + dx + Radius], value);
                }
            }
            data[row * 16 + col] = value;
        }
        __syncwarp();
        copy_brick(output + offset, data, lane);
        __syncwarp();
    }
}

}  // namespace tcstencil_tf32
