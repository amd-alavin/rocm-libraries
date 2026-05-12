/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <Tensile/EmbeddingSimilarity.hpp>

#ifdef __AVX2__
#include <immintrin.h> // For AVX intrinsics

static inline float hsum_avx(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s         = _mm_hadd_ps(s, s);
    s         = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

float avx_dot(int N, const float* __restrict__ A, const float* __restrict__ B)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    for(; i <= N - 32; i += 32)
    {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i),      _mm256_loadu_ps(B + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 8),  _mm256_loadu_ps(B + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 16), _mm256_loadu_ps(B + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 24), _mm256_loadu_ps(B + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    for(; i <= N - 8; i += 8)
    {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i), _mm256_loadu_ps(B + i), acc0);
    }

    float dot_product = hsum_avx(acc0);
    for(; i < N; ++i)
    {
        dot_product += A[i] * B[i];
    }
    return dot_product;
}

#endif

namespace TensileLite
{
    namespace EmbeddingSimilarity
    {
        void StandardScaler::operator()(std::vector<dtype>& F) const
        {
            assert(mean.size() == F.size() && scale.size() == F.size());
            std::transform(F.begin(), F.end(), mean.begin(), F.begin(), std::minus{});
            std::transform(F.begin(), F.end(), scale.begin(), F.begin(), std::divides{});
        }

        bool StandardScaler::valid(bool verbose) const
        {
            bool is_valid = true;
            if(mean.size() != scale.size())
            {
                if(verbose)
                {
                    std::cerr << "StandardScaler mean and scale do not match." << std::endl;
                }
                is_valid = false;
            }
            if(std::find(scale.begin(), scale.end(), 0.) != scale.end())
            {
                if(verbose)
                {
                    std::cerr << "StandardScaler scale contains zero." << std::endl;
                }
                is_valid = false;
            }
            return is_valid;
        }

        std::vector<dtype> relu_activation(std::vector<dtype>&& F)
        {
            for(auto& f : F)
                f = f > 0.0f ? f : 0.0f; // std::max(f, 0.f);
            return F;
        }

        std::vector<float> dense_forward(const std::vector<float>& input,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& bias)
        {
            size_t             input_dim  = input.size();
            size_t             output_dim = bias.size();
            std::vector<float> output     = bias;

            const float* __restrict__ in_ptr = input.data();
            const float* __restrict__ w_ptr  = weights.data();
            float* __restrict__ out_ptr      = output.data();

            for(size_t j = 0; j < output_dim; ++j)
            {
        #ifdef __AVX2__
                out_ptr[j] += avx_dot(static_cast<int>(input_dim), in_ptr, w_ptr + j * input_dim);
        #else
                int k = 0;
                int offset = j * input_dim;
                for(; k < input_dim - input_dim % 8; k += 8, offset += 8)
                {
                    float out0 = in_ptr[k] * w_ptr[offset];
                    float out1 = in_ptr[k + 1] * w_ptr[offset + 1];
                    float out2 = in_ptr[k + 2] * w_ptr[offset + 2];
                    float out3 = in_ptr[k + 3] * w_ptr[offset + 3];
                    float out4 = in_ptr[k + 4] * w_ptr[offset + 4];
                    float out5 = in_ptr[k + 5] * w_ptr[offset + 5];
                    float out6 = in_ptr[k + 6] * w_ptr[offset + 6];
                    float out7 = in_ptr[k + 7] * w_ptr[offset + 7];
                    out_ptr[j] += out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
                }
                for(; k < input_dim; k++, offset++)
                {
                    out_ptr[j] += in_ptr[k] * w_ptr[offset];
                }
        #endif
            }
            return output;
        }

        std::vector<dtype> Network::operator()(const std::vector<dtype>& F) const
        {
            std::vector<dtype> output = F;
            for(int i = 0; i < weights.size(); i++)
            {
                output = relu_activation(std::move(dense_forward(output, weights[i], bias[i])));
            }

            return dense_forward(output, proj_weights, proj_bias);
        }

        bool Network::valid(bool verbose) const
        {
            // Check dense layers
            for(size_t i = 0; i < weights.size(); ++i)
            {
                size_t output_dim = bias[i].size();
                size_t input_dim  = weights[i].size() / output_dim;
                if(weights[i].size() != input_dim * output_dim || bias[i].size() != output_dim)
                {
                    if(verbose)
                    {
                        std::cerr << "Dense layer " << i << " dimensions do not match: "
                                  << "weights.size() = " << weights[i].size()
                                  << ", bias.size() = " << bias[i].size() << std::endl;
                    }
                    return false;
                }
            }
            // Check projection layer
            size_t proj_out_dim = proj_bias.size();
            if(proj_out_dim % 4 != 0)
            {
                if(verbose)
                {
                    std::cerr << "Projection layer output dimensions must be divisible by 4"
                              << std::endl;
                }
                return false;
            }

            if(!weights.empty())
            {
                size_t proj_in_dim = bias.back().size();
                if(proj_weights.size() != proj_in_dim * proj_out_dim)
                {
                    if(verbose)
                    {
                        std::cerr << "Projection layer dimensions do not match: "
                                  << "proj_weights.size() = " << proj_weights.size()
                                  << ", expected = " << proj_in_dim * proj_out_dim
                                  << ", proj_bias.size() = " << proj_bias.size() << std::endl;
                    }
                    return false;
                }
            }
            return true;
        }

        std::vector<dtype> Encoder::forward(std::vector<float>& gemm_features) const

        {
            scaler(gemm_features);

            std::vector<dtype> encoded_gemm= network(gemm_features);

            return encoded_gemm;
        }

        bool Encoder::valid(bool verbose) const
        {
            bool is_valid = scaler.valid(verbose) && network.valid(verbose);

            size_t input_size = 0;
            if(network.weights.empty())
            {
                input_size = network.proj_weights.size() / network.proj_bias.size();
            }
            else
            {
                input_size = network.weights[0].size() / network.bias[0].size();
            }

            if(scaler.mean.size() != input_size)
            {
                if(verbose)
                {
                    std::cerr << "StandardScaler size (" << scaler.mean.size()
                              << ") does not match EmbeddingSimilarity network input size ("
                              << input_size << ")." << std::endl;
                }
                is_valid = false;
            }

            return is_valid;
        }

    }
}
