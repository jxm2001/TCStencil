#pragma once

#include "wmma.cuh"

namespace tcstencil_tf32 {

template<int Radius>
__device__ __forceinline__ void star_stencil(
    const float* __restrict__ input, const float* __restrict__ coefficients,
    const float* __restrict__ weights, float* __restrict__ output,
    int padded_size, int tile_size, const int* __restrict__ halo_offsets) {
    constexpr int width = 2 * Radius + 1;
    __shared__ __align__(32) float data[256];
    __shared__ __align__(32) float halo[64 * Radius];
    const int lane = threadIdx.x + threadIdx.y * 16;
    const int base = (blockIdx.y + 1) * 16 * padded_size +
                     (blockIdx.x * tile_size + 1) * 256;
    MatrixA vertical[2];
    MatrixB horizontal[2];
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        load_tf32(vertical[part], coefficients + part * 8);
        load_tf32(horizontal[part], coefficients + 256 + part * 128);
    }

    for (int tile = 0; tile < tile_size; ++tile) {
        const int offset = base + tile * 256;
        for (int i = lane; i < 64 * Radius; i += 32)
            halo[i] = input[offset + halo_offsets[i]];
        __syncwarp();

        Accumulator result;
        wmma::fill_fragment(result, 0.0f);
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            MatrixB values;
            load_tf32(values, input + offset + part * 128);
            wmma::mma_sync(result, vertical[part], values, result);
        }
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            MatrixA values;
            load_tf32(values, input + offset + part * 8);
            wmma::mma_sync(result, values, horizontal[part], result);
        }
        wmma::store_matrix_sync(data, result, 16, wmma::mem_row_major);
        __syncwarp();

        const int i = threadIdx.x;
        const int side = threadIdx.y;
        #pragma unroll
        for (int edge = 0; edge < Radius; ++edge) {
            const int row = side ? 16 - Radius + edge : edge;
            float value = data[row * 16 + i];
            #pragma unroll
            for (int h = 0; h < Radius; ++h) {
                const int dy = (side ? 16 + h : h - Radius) - row;
                if (dy >= -Radius && dy <= Radius)
                    value = fmaf(halo[side * 16 * Radius + h * 16 + i],
                                 weights[(dy + Radius) * width + Radius], value);
            }
            data[row * 16 + i] = value;
        }
        // Corners receive both vertical and horizontal contributions.
        __syncwarp();
        #pragma unroll
        for (int edge = 0; edge < Radius; ++edge) {
            const int col = side ? 16 - Radius + edge : edge;
            float value = data[i * 16 + col];
            #pragma unroll
            for (int h = 0; h < Radius; ++h) {
                const int dx = (side ? 16 + h : h - Radius) - col;
                if (dx >= -Radius && dx <= Radius)
                    value = fmaf(halo[(32 + side * 16) * Radius + i * Radius + h],
                                 weights[Radius * width + dx + Radius], value);
            }
            data[i * 16 + col] = value;
        }
        __syncwarp();
        copy_brick(output + offset, data, lane);
        __syncwarp();
    }
}

}  // namespace tcstencil_tf32
