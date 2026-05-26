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
#include <cstring>
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
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i), _mm256_loadu_ps(B + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 8), _mm256_loadu_ps(B + i + 8), acc1);
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

#if defined(__AVX512F__) && defined(__AVX512BF16__)
float avx_dot_bf16(int N, const uint16_t* __restrict__ A, const uint16_t* __restrict__ B)
{
    __m512 acc = _mm512_setzero_ps();

    int i = 0;
    for(; i <= N - 32; i += 32)
    {
        const __m512bh a = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(A + i));
        const __m512bh b = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(B + i));
        acc              = _mm512_dpbf16_ps(acc, a, b);
    }

    __m256 sum256      = _mm256_add_ps(_mm512_castps512_ps256(acc), _mm512_extractf32x8_ps(acc, 1));
    float  dot_product = hsum_avx(sum256);

    for(; i < N; ++i)
    {
        dot_product += bf16_to_float(A[i]) * bf16_to_float(B[i]);
    }
    return dot_product;
}
#else

float avx_dot_bf16(int N, const float* __restrict__ A, const uint16_t* __restrict__ B)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    for(; i <= N - 32; i += 32)
    {
        __m128i b16_0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + i));
        __m128i b16_1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + i + 8));
        __m128i b16_2  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + i + 16));
        __m128i b16_3  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + i + 24));
        __m256  bf32_0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16_0), 16));
        __m256  bf32_1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16_1), 16));
        __m256  bf32_2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16_2), 16));
        __m256  bf32_3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16_3), 16));
        acc0           = _mm256_fmadd_ps(_mm256_loadu_ps(A + i), bf32_0, acc0);
        acc1           = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 8), bf32_1, acc1);
        acc2           = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 16), bf32_2, acc2);
        acc3           = _mm256_fmadd_ps(_mm256_loadu_ps(A + i + 24), bf32_3, acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    for(; i <= N - 8; i += 8)
    {
        __m128i b16  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + i));
        __m256  bf32 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16));
        acc0         = _mm256_fmadd_ps(_mm256_loadu_ps(A + i), bf32, acc0);
    }

    float dot_product = hsum_avx(acc0);
    for(; i < N; ++i)
    {
        dot_product += A[i] * bf16_to_float(B[i]);
    }
    return dot_product;
}

#endif
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

        std::vector<dtype> relu(std::vector<dtype>&& F)
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
                int k      = 0;
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

        std::vector<float> dense_forward_bf16(const std::vector<float>&    input,
                                              const std::vector<uint16_t>& weights_bf16,
                                              const std::vector<float>&    bias)
        {
            size_t             input_dim  = input.size();
            size_t             output_dim = bias.size();
            std::vector<float> output     = bias;

            const uint16_t* __restrict__ w_ptr = weights_bf16.data();
            float* __restrict__ out_ptr        = output.data();

#if defined(__AVX512F__) && defined(__AVX512BF16__)
            std::vector<uint16_t> input_bf16(input_dim);
            const float* __restrict__ in_ptr_f32 = input.data();

            for(size_t k = 0; k < input_dim; ++k)
            {
                input_bf16[k] = float_to_bf16_rne(in_ptr_f32[k]);
            }
            const uint16_t* __restrict__ in_ptr = input_bf16.data();
#else
            const float* __restrict__ in_ptr = input.data();
#endif

            for(size_t j = 0; j < output_dim; ++j)
            {
#ifdef __AVX2__
                out_ptr[j]
                    += avx_dot_bf16(static_cast<int>(input_dim), in_ptr, w_ptr + j * input_dim);
#else
                int k      = 0;
                int offset = j * input_dim;
                for(; k < input_dim - input_dim % 8; k += 8, offset += 8)
                {
                    float out0 = in_ptr[k] * bf16_to_float(w_ptr[offset]);
                    float out1 = in_ptr[k + 1] * bf16_to_float(w_ptr[offset + 1]);
                    float out2 = in_ptr[k + 2] * bf16_to_float(w_ptr[offset + 2]);
                    float out3 = in_ptr[k + 3] * bf16_to_float(w_ptr[offset + 3]);
                    float out4 = in_ptr[k + 4] * bf16_to_float(w_ptr[offset + 4]);
                    float out5 = in_ptr[k + 5] * bf16_to_float(w_ptr[offset + 5]);
                    float out6 = in_ptr[k + 6] * bf16_to_float(w_ptr[offset + 6]);
                    float out7 = in_ptr[k + 7] * bf16_to_float(w_ptr[offset + 7]);
                    out_ptr[j] += out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
                }
                for(; k < input_dim; k++, offset++)
                {
                    out_ptr[j] += in_ptr[k] * bf16_to_float(w_ptr[offset]);
                }
#endif
            }
            return output;
        }

        void Network::quantize()
        {

            weights_bf16_.resize(weights_.size());
            for(size_t i = 0; i < weights_.size(); ++i)
            {
                weights_bf16_[i].resize(weights_[i].size());
                for(size_t k = 0; k < weights_[i].size(); ++k)
                {
                    weights_bf16_[i][k] = float_to_bf16_rne(weights_[i][k]);
                }
            }
            proj_weights_bf16_.resize(proj_weights_.size());
            for(size_t k = 0; k < proj_weights_.size(); ++k)
            {
                proj_weights_bf16_[k] = float_to_bf16_rne(proj_weights_[k]);
            }
            forward_impl_ = &Network::forward_bf16_;
        }

        std::vector<dtype> Network::operator()(const std::vector<dtype>& F) const
        {
            return (this->*forward_impl_)(F);
        }

        std::vector<dtype> Network::forward_fp32_(const std::vector<dtype>& F) const
        {
            std::vector<dtype> output = F;
            for(int i = 0; i < (int)weights_.size(); i++)
            {
                output = relu(std::move(dense_forward(output, weights_[i], bias_[i])));
            }
            return dense_forward(output, proj_weights_, proj_bias_);
        }

        std::vector<dtype> Network::forward_bf16_(const std::vector<dtype>& F) const
        {
            std::vector<dtype> output = F;
            for(int i = 0; i < (int)weights_.size(); i++)
            {
                output = relu(std::move(dense_forward_bf16(output, weights_bf16_[i], bias_[i])));
            }
            return dense_forward_bf16(output, proj_weights_bf16_, proj_bias_);
        }

        bool Network::valid(bool verbose) const
        {
            // Check dense layers
            for(size_t i = 0; i < weights_.size(); ++i)
            {
                size_t output_dim = bias_[i].size();
                size_t input_dim  = weights_[i].size() / output_dim;
                if(weights_[i].size() != input_dim * output_dim || bias_[i].size() != output_dim)
                {
                    if(verbose)
                    {
                        std::cerr << "Dense layer " << i << " dimensions do not match: "
                                  << "weights.size() = " << weights_[i].size()
                                  << ", bias.size() = " << bias_[i].size() << std::endl;
                    }
                    return false;
                }
            }
            // Check projection layer
            size_t proj_out_dim = proj_bias_.size();
            if(proj_out_dim % 4 != 0)
            {
                if(verbose)
                {
                    std::cerr << "Projection layer output dimensions must be divisible by 4"
                              << std::endl;
                }
                return false;
            }

            if(!weights_.empty())
            {
                size_t proj_in_dim = bias_.back().size();
                if(proj_weights_.size() != proj_in_dim * proj_out_dim)
                {
                    if(verbose)
                    {
                        std::cerr << "Projection layer dimensions do not match: "
                                  << "proj_weights.size() = " << proj_weights_.size()
                                  << ", expected = " << proj_in_dim * proj_out_dim
                                  << ", proj_bias.size() = " << proj_bias_.size() << std::endl;
                    }
                    return false;
                }
            }
            return true;
        }

        std::vector<dtype> Encoder::forward(std::vector<float>& gemm_features) const

        {
            scaler(gemm_features);

            std::vector<dtype> encoded_gemm = network(gemm_features);

            return encoded_gemm;
        }

        bool Encoder::valid(bool verbose) const
        {
            bool is_valid = scaler.valid(verbose) && network.valid(verbose);

            size_t input_size = 0;
            if(network.weights_.empty())
            {
                input_size = network.proj_weights_.size() / network.proj_bias_.size();
            }
            else
            {
                input_size = network.weights_[0].size() / network.bias_[0].size();
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

        void SolutionEmbeddings::quantize()
        {
            embeddings_bf16.resize(embeddings.size());

            for(int k = 0; k < embeddings.size(); ++k)
            {
                const int n_solutions   = static_cast<int>(embeddings[k].size());
                const int embedding_dim = static_cast<int>(embeddings[k][0].size());
                embeddings_bf16[k]      = std::vector<std::vector<uint16_t>>(
                    n_solutions, std::vector<uint16_t>(embedding_dim));
                for(int i = 0; i < n_solutions; ++i)
                {
                    for(int j = 0; j < embedding_dim; ++j)
                    {
                        embeddings_bf16[k][i][j] = float_to_bf16_rne(embeddings[k][i][j]);
                    }
                }
            }
            const int n_centroids   = static_cast<int>(centroids.size());
            const int embedding_dim = static_cast<int>(centroids[0].size());
            centroids_bf16          = std::vector<std::vector<uint16_t>>(
                n_centroids, std::vector<uint16_t>(embedding_dim));
            for(int i = 0; i < n_centroids; ++i)
            {
                for(int j = 0; j < embedding_dim; ++j)
                {
                    centroids_bf16[i][j] = float_to_bf16_rne(centroids[i][j]);
                }
            }
        }

    }
}
