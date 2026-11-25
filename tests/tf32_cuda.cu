#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include "../common/param.hpp"
#include "tf32_reference.hpp"

extern "C" void host_code(float*, float*, float*, int, int);

__global__ void convert_test_values(float* values, int count) {
    const int i = threadIdx.x;
    if (i < count) values[i] = nvcuda::wmma::__float_to_tf32(values[i]);
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no CUDA GPU available: " << cudaGetErrorString(status) << '\n';
        return 77;
    }
    try {
        cudaDeviceProp properties;
        check_cuda(cudaGetDeviceProperties(&properties, 0));
        if (properties.major < 8) {
            std::cout << "SKIP: TF32 requires compute capability >= 8.0\n";
            return 77;
        }
        if (argc == 2 && std::string(argv[1]) == "--probe") return 0;
        float values[] = {0.0f, -0.0f, 1.00048828125f, -1.00048828125f,
                          0.1234567f, -0.1234567f, 100000.25f, 1.23e-8f};
        constexpr int count = sizeof(values) / sizeof(values[0]);
        float expected[count];
        for (int i = 0; i < count; ++i) expected[i] = tf32_test::round_tf32(values[i]);
        float* device_values = nullptr;
        check_cuda(cudaMalloc(&device_values, sizeof(values)));
        check_cuda(cudaMemcpy(device_values, values, sizeof(values), cudaMemcpyHostToDevice));
        convert_test_values<<<1, 32>>>(device_values, count);
        check_cuda(cudaGetLastError());
        check_cuda(cudaMemcpy(values, device_values, sizeof(values), cudaMemcpyDeviceToHost));
        check_cuda(cudaFree(device_values));
        if (std::memcmp(values, expected, sizeof(values)) != 0)
            throw std::runtime_error("host reference does not match native TF32 rounding");

        int cases = 0;
        double max_full_error = 0.0, max_scaled_error = 0.0;
        for (int size : {16, 32, 64, 160, 240}) {
            if ((size / 16) % TILE_SIZE != 0) continue;
            for (int pattern = 0; pattern < 6; ++pattern) {
                auto input = tf32_test::make_input(size, pattern);
                for (bool fractional : {false, true}) {
                    auto weights = tf32_test::make_weights(TEST_RADIUS, fractional, pattern);
                    std::vector<float> output(size * size);
                    host_code(input.data(), weights.data(), output.data(), size, 2 * TEST_RADIUS + 1);
                    const auto mixed = tf32_test::reference(input, weights, size, TEST_RADIUS, TEST_STAR, true);
                    tf32_test::check_result(output, mixed, size);
                    const auto full = tf32_test::reference(input, weights, size, TEST_RADIUS, TEST_STAR, false);
                    for (std::size_t i = 0; i < output.size(); ++i) {
                        const double error = std::abs(output[i] - full.value[i]);
                        max_full_error = std::max(max_full_error, error);
                        max_scaled_error = std::max(max_scaled_error, error / (1e-6 + full.magnitude[i]));
                    }
                    ++cases;
                }
            }
        }
        std::cout << cases << " GPU cases passed; max error vs unquantized reference="
                  << max_full_error << ", max scaled error=" << max_scaled_error << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
