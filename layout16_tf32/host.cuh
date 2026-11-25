#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include "../common/param.hpp"
#include "layout.hpp"

extern "C" __global__ void mma_run(const float* input, const float* coefficients,
                                  const float* weights, float* output,
                                  int padded_size, int tile_size, const int* halo_offsets);

namespace tcstencil_tf32 {

inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : pointer_(nullptr), bytes_(count * sizeof(T)) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes_), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
    void upload(const T* source) {
        cuda_check(cudaMemcpy(pointer_, source, bytes_, cudaMemcpyHostToDevice), "upload");
    }
    void download(T* destination) {
        cuda_check(cudaMemcpy(destination, pointer_, bytes_, cudaMemcpyDeviceToHost), "download");
    }
private:
    T* pointer_;
    std::size_t bytes_;
};

class Event {
public:
    Event() { cuda_check(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    operator cudaEvent_t() const { return event_; }
private:
    cudaEvent_t event_;
};

template<int Radius, bool Star>
void run_host(float* input, float* weights, float* output, int size, int width) {
    static_assert(RUN_TIMES > 0, "RUN_TIMES must be positive");
    static_assert(STEP_TIMES == 1, "TF32 benchmarks perform one stencil step per launch");
    validate_problem(size, TILE_SIZE);
    if (width != 2 * Radius + 1)
        throw std::invalid_argument("coefficient width does not match stencil radius");
    int device = 0;
    cudaDeviceProp properties;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.major < 8)
        throw std::runtime_error("TF32 requires compute capability 8.0 or newer");

    auto bricks = pack_input(input, size);
    auto coefficients = make_coefficients<Radius, Star>(weights);
    auto offsets = make_halo_offsets<Radius, Star>(size + 32);
    DeviceBuffer<float> device_input(bricks.size()), device_output(bricks.size());
    DeviceBuffer<float> device_coefficients(coefficients.size()), device_weights(width * width);
    DeviceBuffer<int> device_offsets(offsets.size());
    device_input.upload(bricks.data());
    device_output.upload(bricks.data());
    device_coefficients.upload(coefficients.data());
    device_weights.upload(weights);
    device_offsets.upload(offsets.data());

    const dim3 block(16, 2);
    const dim3 grid(size / (16 * TILE_SIZE), size / 16);
    constexpr int warmup = RUN_TIMES > SKIP_TIMES ? SKIP_TIMES : 0;
    Event start, stop;
    double sum = 0.0;
    for (int repeat = 0; repeat < RUN_TIMES; ++repeat) {
        cuda_check(cudaDeviceSynchronize(), "before stencil");
        cuda_check(cudaEventRecord(start), "record start");
        mma_run<<<grid, block>>>(device_input.get(), device_coefficients.get(),
                                 device_weights.get(), device_output.get(), size + 32,
                                 TILE_SIZE, device_offsets.get());
        cuda_check(cudaGetLastError(), "launch stencil");
        cuda_check(cudaDeviceSynchronize(), "execute stencil");
        cuda_check(cudaEventRecord(stop), "record stop");
        cuda_check(cudaEventSynchronize(stop), "wait for stop");
        float elapsed = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed time");
        if (repeat >= warmup)
            sum += elapsed;
    }
    printf("[Time] Time used: %.9f ms\n", sum / (RUN_TIMES - warmup));
    device_output.download(bricks.data());
    unpack_output(bricks.data(), output, size);
}

}  // namespace tcstencil_tf32
