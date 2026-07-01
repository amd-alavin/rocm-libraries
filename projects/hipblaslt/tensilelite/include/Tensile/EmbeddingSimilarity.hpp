/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "DataTypes_Half.hpp"
#include <array>
#include <map>
#include <memory>
#include <set>
#include <vector>

#ifdef __AVX2__
float avx_dot(int N, const float* __restrict__ A, const float* __restrict__ B);
#if defined(__AVX512F__) && defined(__AVX512BF16__)
float avx_dot_bf16(int N, const uint16_t* __restrict__ A, const uint16_t* __restrict__ B);
#else
float avx_dot_bf16(int N, const float* __restrict__ A, const uint16_t* __restrict__ B);
#endif
#endif

    static inline uint16_t float_to_bf16_rne(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t lsb           = (u >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

static inline float bf16_to_float(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float    out;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup EmbeddingSimilarity
     *
     * @brief EmbeddingSimilarity model
     *
     * Encoder used to estimate embedding values for problems in the
     * library. Used for EmbeddingSimilarityLibrary.
     *
     * See EmbeddingSimilarity.cpp
     */

    /**
     * \ingroup EmbeddingSimilarity
     */
    namespace EmbeddingSimilarity
    {

        using dtype = float;

        struct StandardScaler
        {
            void operator()(std::vector<dtype>& F) const;
            bool valid(bool verbose = false) const;

            std::vector<dtype> mean, scale;
        };

        struct Network
        {
            using Matrix      = std::vector<std::vector<dtype>>;
            using Vector      = std::vector<dtype>;
            using ForwardImpl = std::vector<dtype> (Network::*)(const std::vector<dtype>&) const;

            void               quantize();
            std::vector<dtype> operator()(const std::vector<dtype>& F) const;
            bool               valid(bool verbose) const;

            std::vector<dtype> forward_fp32_(const std::vector<dtype>& F) const;
            std::vector<dtype> forward_bf16_(const std::vector<dtype>& F) const;

            Matrix                             weights_;
            Matrix                             bias_;
            Vector                             proj_weights_;
            Vector                             proj_bias_;
            std::vector<std::vector<uint16_t>> weights_bf16_;
            std::vector<uint16_t>              proj_weights_bf16_;

            ForwardImpl forward_impl_ = &Network::forward_fp32_;
        };

        struct Encoder
        {
            Encoder() = default;

            std::vector<dtype> forward(std::vector<float>& probkey) const;

            bool valid(bool verbose = false) const;

            std::string description() const
            {
                return "Encoder";
            }

            StandardScaler scaler;
            Network        network;
        };

        struct SolutionEmbeddings
        {
            SolutionEmbeddings() = default;

            std::string description() const
            {
                return "SolutionEmbeddings";
            }

            std::vector<std::vector<float>>                 centroids;
            std::vector<std::vector<std::vector<float>>>    embeddings;
            std::vector<std::vector<uint16_t>>              centroids_bf16;
            std::vector<std::vector<std::vector<uint16_t>>> embeddings_bf16;
            std::vector<std::vector<int>>                   cluster_indices;
            std::size_t                                     size() const
            {
                std::set<int> unique_values;
                for(const auto& cluster : cluster_indices)
                {
                    unique_values.insert(cluster.begin(), cluster.end());
                }
                return unique_values.size();
            }
            void quantize();
        };

        struct HardwareConstants
        {
            HardwareConstants() = default;

            std::string description() const
            {
                return "HardwareConstants";
            }

            bool valid(bool verbose = false) const
            {
                bool rv = true;
                if(n_cu <= 0)
                {
                    if(verbose)
                        std::cout << "Invalid n_cu: " << n_cu << std::endl;
                    rv = false;
                }
                if(peak_flops <= 0.0f || mem_bw <= 0.0f)
                {
                    if(verbose)
                        std::cout << "Invalid peak_flops or mem_bw" << std::endl;
                    rv = false;
                }
                return rv;
            }

            int   n_cu       = 256;
            float peak_flops = 2.3e15f;
            float mem_bw     = 8e12f;
            float l1_size    = 32.0f * 1024.0f;
            float l2_size    = 4.0f * 1024.0f * 1024.0f;
            float l3_size    = 256.0f * 1024.0f * 1024.0f;
            float wave_size  = 64.0f;
            float dtype_size = 2.0f;
            float acc_size   = 4.0f;
        };

        struct FallbackRule
        {
            FallbackRule() = default;
            std::string description() const
            {
                return "FallbackRule";
            }

            bool valid(bool verbose = false) const
            {
                bool rv = true; // TODO
                return rv;
            }
            
            bool matches(float m, float n, float k, int cat, float score) const
            {
                // Check category first (quick rejection)
                if (!matchesCategory(cat))
                {
                    return false;
                }

                // Check M, N, K ranges
                if (!inRange(m, m_ranges) ||
                    !inRange(n, n_ranges) ||
                    !inRange(k, k_ranges))
                {
                    return false;
                }

                // Check score range
                if (!inRange(score, score_ranges))
                {
                    return false;
                }

                return true;
            }

            bool matches(float m, float n, float k, int cat) const
            {
                // Check category first (quick rejection)
                if (!matchesCategory(cat))
                {
                    return false;
                }

                // Check M, N, K ranges
                if (!inRange(m, m_ranges) ||
                    !inRange(n, n_ranges) ||
                    !inRange(k, k_ranges))
                {
                    return false;
                }

                return true;
            }



            int rule_id;
            std::vector<float> m_ranges;    
            std::vector<float> n_ranges;
            std::vector<float> k_ranges;
            std::vector<float> score_ranges;  // Optional
            std::vector<int> cats;   

            private:

                bool matchesCategory(int cat) const
                {
                    if (cats.empty()) return true;  // Empty = match all

                    for (int rule_cat : cats)
                    {
                        if (rule_cat == cat) return true;
                    }
                    return false;
                }

                static bool inRange(float value, const std::vector<float>& ranges)
                {
                    if (ranges.empty()) return true;  // No constraint

                    // Ranges in pairs: [min1, max1, min2, max2, ...]
                    for (size_t i = 0; i + 1 < ranges.size(); i += 2)
                    {
                        float range_min = ranges[i];
                        float range_max = ranges[i + 1];

                        if (value > range_min && value < range_max){
                            return true;
                        }
                       
                    }
                    return false;
                }

        };

        struct FallbackRules
        {
            FallbackRules() = default;

            std::string description() const
            {
                return "FallbackRules";
            }

            bool valid(bool verbose = false) const
            {
                bool rv = true;
                std::cout << "Validating....\n";
                if(all_cats.empty())
                {
                    if(verbose) std::cout << "FallbackRules: all_cats is empty" << std::endl;
                    rv = false;
                }
                return rv;
            }
            bool isEmpty() const
            {
                return all_cats.empty() &&
                        pre_model_features.empty() &&
                        post_model_features.empty();
            }

            bool hasData() const
            {
                return !isEmpty();
            }

            std::string interval_semantics = "open_open"; // TODO
            std::string notes;
            std::vector<int> all_cats;                     
            std::vector<FallbackRule> pre_model_features;   // [m,n,k,cat]
            std::vector<FallbackRule> post_model_features;  // [m,n,k,cat,score]
        };

    } // namespace EmbeddingSimilarity
} // namespace TensileLite