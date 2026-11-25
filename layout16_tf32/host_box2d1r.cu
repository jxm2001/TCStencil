#include "host.cuh"

extern "C" void host_code(float* input, float* weights, float* output, int size, int width) {
    tcstencil_tf32::run_host<1, false>(input, weights, output, size, width);
}
