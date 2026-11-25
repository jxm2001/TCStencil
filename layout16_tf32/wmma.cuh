#pragma once

#include <cuda_runtime.h>
#include <mma.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "TF32 stencil kernels require compute capability 8.0 or newer"
#endif

namespace tcstencil_tf32 {

namespace wmma = nvcuda::wmma;
using MatrixA = wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major>;
using MatrixB = wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major>;
using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;

template<typename Fragment>
__device__ __forceinline__ void load_tf32(Fragment& fragment, const float* data) {
    wmma::load_matrix_sync(fragment, data, 16);
    #pragma unroll
    for (int i = 0; i < fragment.num_elements; ++i)
        fragment.x[i] = wmma::__float_to_tf32(fragment.x[i]);
}

__device__ __forceinline__ void copy_brick(float* output, const float* input, int lane) {
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        reinterpret_cast<float4*>(output)[lane + i * 32] =
            reinterpret_cast<const float4*>(input)[lane + i * 32];
}

}  // namespace tcstencil_tf32
