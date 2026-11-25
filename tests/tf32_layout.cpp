#include <iostream>
#include "../layout16_tf32/layout.hpp"
#include "tf32_reference.hpp"

using namespace tcstencil_tf32;
using tf32_test::round_tf32;

template<int Radius, bool Star>
std::vector<float> model(const std::vector<float>& input, const std::vector<float>& weights,
                         int size, int tile_size) {
    validate_problem(size, tile_size);
    constexpr int width = 2 * Radius + 1;
    constexpr int rows = 16 + 2 * Radius;
    const int padded = size + 32;
    const auto bricks = pack_input(input.data(), size);
    const auto coe = make_coefficients<Radius, Star>(weights.data());
    const auto offsets = make_halo_offsets<Radius, Star>(padded);
    std::vector<float> output(bricks.size(), 0.0f);
    for (int by = 0; by < size / 16; ++by) {
        for (int bx = 0; bx < size / (16 * tile_size); ++bx) {
            for (int tile = 0; tile < tile_size; ++tile) {
                const int offset = (by + 1) * 16 * padded + (bx * tile_size + tile + 1) * 256;
                float window[rows * 16] = {};
                if (!Star) {
                    std::copy_n(bricks.data() + offset, 256, window + Radius * 16);
                    for (int i = 0; i < Radius * 16; ++i) {
                        window[i] = bricks.at(offset - padded * 16 + (16 - Radius) * 16 + i);
                        window[(16 + Radius) * 16 + i] = bricks.at(offset + padded * 16 + i);
                    }
                }
                std::vector<float> halo(offsets.size());
                for (std::size_t i = 0; i < offsets.size(); ++i)
                    halo[i] = bricks.at(offset + offsets[i]);
                for (int row = 0; row < 16; ++row) {
                    for (int col = 0; col < 16; ++col) {
                        float value = 0.0f;
                        if (Star) {
                            for (int part = 0; part < 2; ++part)
                                for (int k = part * 8; k < part * 8 + 8; ++k)
                                    value = std::fma(round_tf32(coe[row * 16 + k]),
                                                     round_tf32(bricks[offset + k * 16 + col]), value);
                            for (int part = 0; part < 2; ++part)
                                for (int k = part * 8; k < part * 8 + 8; ++k)
                                    value = std::fma(round_tf32(bricks[offset + row * 16 + k]),
                                                     round_tf32(coe[256 + k * 16 + col]), value);
                            for (int side = 0; side < 2; ++side) {
                                for (int h = 0; h < Radius; ++h) {
                                    const int dy = (side ? 16 + h : h - Radius) - row;
                                    if (dy >= -Radius && dy <= Radius)
                                        value = std::fma(halo.at(side * 16 * Radius + h * 16 + col),
                                                         weights[(dy + Radius) * width + Radius], value);
                                }
                            }
                            for (int side = 0; side < 2; ++side) {
                                for (int h = 0; h < Radius; ++h) {
                                    const int dx = (side ? 16 + h : h - Radius) - col;
                                    if (dx >= -Radius && dx <= Radius)
                                        value = std::fma(halo.at((32 + side * 16) * Radius + row * Radius + h),
                                                         weights[Radius * width + dx + Radius], value);
                                }
                            }
                        } else {
                            for (int z = 0; z < width; ++z)
                                for (int part = 0; part < 2; ++part)
                                    for (int k = part * 8; k < part * 8 + 8; ++k)
                                        value = std::fma(round_tf32(window[(row + z) * 16 + k]),
                                                         round_tf32(coe[z * 256 + k * 16 + col]), value);
                            for (int side = 0; side < 2; ++side) {
                                for (int z = 0; z < width; ++z) {
                                    for (int h = 0; h < Radius; ++h) {
                                        const int dx = (side ? 16 + h : h - Radius) - col;
                                        if (dx >= -Radius && dx <= Radius)
                                            value = std::fma(halo.at((side * Radius + h) * rows + row + z),
                                                             weights[z * width + dx + Radius], value);
                                    }
                                }
                            }
                        }
                        output[offset + row * 16 + col] = value;
                    }
                }
            }
        }
    }
    std::vector<float> result(size * size);
    unpack_output(output.data(), result.data(), size);
    return result;
}

template<int Radius, bool Star>
void test_stencil() {
    int count = 0;
    for (int size : {16, 32, 64, 160, 240}) {
        for (int tile : {1, 2, 5, 10, 15}) {
            if ((size / 16) % tile != 0) continue;
            for (int pattern = 0; pattern < 6; ++pattern) {
                const auto input = tf32_test::make_input(size, pattern);
                for (bool fractional : {false, true}) {
                    const auto weights = tf32_test::make_weights(Radius, fractional, pattern);
                    tf32_test::check_result(model<Radius, Star>(input, weights, size, tile),
                        tf32_test::reference(input, weights, size, Radius, Star, true), size);
                    ++count;
                }
            }
        }
    }
    std::cout << (Star ? "star" : "box") << " radius=" << Radius << ": " << count
              << " CPU mapping cases passed\n";
}

int main() {
    try {
        if (round_tf32(1.00048828125f) != 1.0009765625f ||
            round_tf32(-1.00048828125f) != -1.0009765625f)
            throw std::runtime_error("TF32 tie rounding failed");
        for (const auto& config : {std::pair<int, int>{0, 1}, {17, 1}, {48, 2}, {16, 0}, {16, 16}, {50000, 1}}) {
            bool rejected = false;
            try { validate_problem(config.first, config.second); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid mesh/tile accepted");
        }
        test_stencil<1, false>();
        test_stencil<2, false>();
        test_stencil<1, true>();
        test_stencil<2, true>();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
