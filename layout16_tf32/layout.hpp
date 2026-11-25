#pragma once

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tcstencil_tf32 {

inline void validate_problem(int size, int tile_size) {
    if (size <= 0 || size % 16 != 0)
        throw std::invalid_argument("mesh size must be a positive multiple of 16");
    if (tile_size < 1 || tile_size > 15 || (size / 16) % tile_size != 0)
        throw std::invalid_argument("tile size must be in [1, 15] and divide mesh size / 16");
    if (size > INT_MAX - 32 ||
        static_cast<std::size_t>(size + 32) * (size + 32) > INT_MAX)
        throw std::invalid_argument("padded mesh exceeds the kernel's integer indexing range");
}

inline int brick_index(int y, int x, int padded_size) {
    return ((y / 16) * (padded_size / 16) + x / 16) * 256 +
           (y % 16) * 16 + x % 16;
}

inline std::vector<float> pack_input(const float* input, int size) {
    const int padded = size + 32;
    std::vector<float> result(static_cast<std::size_t>(padded) * padded, 1.0f);
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            result[brick_index(y + 16, x + 16, padded)] = input[y * size + x];
    return result;
}

inline void unpack_output(const float* bricks, float* output, int size) {
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            output[y * size + x] = bricks[brick_index(y + 16, x + 16, size + 32)];
}

template<int Radius, bool Star>
std::vector<float> make_coefficients(const float* weights) {
    static_assert(Radius == 1 || Radius == 2, "unsupported stencil radius");
    constexpr int width = 2 * Radius + 1;
    std::vector<float> coefficients((Star ? 2 : width) * 256, 0.0f);
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            if (Star) {
                const int vertical = col - row + Radius;
                const int horizontal = row - col + Radius;
                if (vertical >= 0 && vertical < width)
                    coefficients[row * 16 + col] = weights[vertical * width + Radius];
                if (horizontal >= 0 && horizontal < width && row != col)
                    coefficients[256 + row * 16 + col] = weights[Radius * width + horizontal];
            } else {
                const int dx = row - col + Radius;
                if (dx >= 0 && dx < width)
                    for (int z = 0; z < width; ++z)
                        coefficients[z * 256 + row * 16 + col] = weights[z * width + dx];
            }
        }
    }
    return coefficients;
}

template<int Radius, bool Star>
std::vector<int> make_halo_offsets(int padded_size) {
    constexpr int rows = 16 + 2 * Radius;
    std::vector<int> offsets(Star ? 64 * Radius : 2 * Radius * rows);
    const int origin = brick_index(16, 16, padded_size);
    if (Star) {
        for (int i = 0; i < 16; ++i) {
            for (int h = 0; h < Radius; ++h) {
                offsets[h * 16 + i] = brick_index(16 - Radius + h, 16 + i, padded_size) - origin;
                offsets[16 * Radius + h * 16 + i] = brick_index(32 + h, 16 + i, padded_size) - origin;
                offsets[32 * Radius + i * Radius + h] = brick_index(16 + i, 16 - Radius + h, padded_size) - origin;
                offsets[48 * Radius + i * Radius + h] = brick_index(16 + i, 32 + h, padded_size) - origin;
            }
        }
    } else {
        for (int side = 0; side < 2; ++side)
            for (int h = 0; h < Radius; ++h)
                for (int y = 0; y < rows; ++y)
                    offsets[(side * Radius + h) * rows + y] =
                        brick_index(16 - Radius + y, side ? 32 + h : 16 - Radius + h,
                                    padded_size) - origin;
    }
    return offsets;
}

}  // namespace tcstencil_tf32
