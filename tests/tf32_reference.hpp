#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tf32_test {

inline float round_tf32(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // Match cvt.rna.tf32.f32: nearest, with ties away from zero.
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x1000u) & ~0x1fffu;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline std::vector<float> make_input(int size, int pattern) {
    std::vector<float> input(size * size, 0.0f);
    std::mt19937 rng(103);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float value = distribution(rng);
            if (pattern == 0) value = 1.0f;
            if (pattern == 1) value = (y * size + x - size * size / 2) * 0.03125f;
            if (pattern == 2) value = 0.0f;
            if (pattern == 4) value *= 100000.0f;
            if (pattern == 5) value *= 1.0e-8f;
            input[y * size + x] = value;
        }
    }
    if (pattern == 2) {
        for (int y : {0, 1, 7, 8, 14, 15, 16, size - 2, size - 1})
            for (int x : {0, 1, 7, 8, 14, 15, 16, size - 2, size - 1})
                if (x < size && y < size)
                    input[y * size + x] = ((x + y) % 2 ? -1.0f : 1.0f);
    }
    return input;
}

inline std::vector<float> make_weights(int radius, bool fractional, int pattern) {
    const int width = 2 * radius + 1;
    std::vector<float> weights(width * width);
    std::mt19937 rng(291);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    for (int i = 0; i < width * width; ++i)
        weights[i] = fractional ? distribution(rng) : static_cast<float>(i);
    weights[radius * width + radius] = fractional ? -4.123f : (radius == 1 ? -40.0f : -290.0f);
    // Small inputs must not pass merely because the absolute tolerance is larger.
    if (pattern == 5)
        for (float& value : weights) value *= 1.0e8f;
    return weights;
}

struct Reference {
    std::vector<double> value;
    std::vector<double> magnitude;
};

inline Reference reference(const std::vector<float>& input, const std::vector<float>& weights,
                           int size, int radius, bool star, bool mixed) {
    Reference result{std::vector<double>(size * size), std::vector<double>(size * size)};
    const int width = 2 * radius + 1;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (star && dy != 0 && dx != 0) continue;
                    const int iy = y + dy, ix = x + dx;
                    float value = iy < 0 || iy >= size || ix < 0 || ix >= size
                        ? 1.0f : input[iy * size + ix];
                    float weight = weights[(dy + radius) * width + dx + radius];
                    const bool inside_x = x % 16 + dx >= 0 && x % 16 + dx < 16;
                    const bool inside_y = y % 16 + dy >= 0 && y % 16 + dy < 16;
                    if (mixed && inside_x && (!star || inside_y)) {
                        value = round_tf32(value);
                        weight = round_tf32(weight);
                    }
                    const double product = static_cast<double>(value) * weight;
                    result.value[y * size + x] += product;
                    result.magnitude[y * size + x] += std::abs(product);
                }
            }
        }
    }
    return result;
}

inline void check_result(const std::vector<float>& actual, const Reference& expected, int size) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double tolerance = 1e-6 + 1e-5 * expected.magnitude[i];
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected.value[i]) > tolerance)
            throw std::runtime_error("mismatch at (" + std::to_string(i / size) + "," +
                                     std::to_string(i % size) + "): actual=" +
                                     std::to_string(actual[i]) + " expected=" +
                                     std::to_string(expected.value[i]));
    }
}

}  // namespace tf32_test
