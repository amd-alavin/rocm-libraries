// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"
#include "LayernormCommon.hpp"
#include "hipdnn_data_sdk/utilities/Tensor.hpp"
#include "hipdnn_frontend/Types.hpp"
#include "hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp"
#include "hipdnn_frontend/attributes/TensorAttributes.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::layernorm;
using namespace hip_kernel_provider::test_utilities;

namespace hip_kernel_provider::layernorm::test
{

using namespace common;

namespace
{

template <typename InputDataType,
          typename OutputDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType>
class LayernormBackward
    : public IntegrationGraphVerificationHarness<InputDataType, LayernormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const LayernormTestCase& testCase = this->GetParam();

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        auto scaleBiasDataType = getDataTypeEnumFromType<ScaleBiasDataType>();
        auto meanInvVarianceDataType = getDataTypeEnumFromType<MeanInvVarianceDataType>();

        if(inputDataType == DataType::HALF
           && testCase.dims == std::vector<int64_t>{32, 32, 14, 25, 59}
           && testCase.normalizedDim == 4 && !testCase.optionalTensors)
        {
            GTEST_SKIP() << "Skipping test due to expected infinities in results.";
        }

        std::vector<int64_t> statDims(testCase.dims.size(), 1);
        std::vector<int64_t> affineDims(testCase.dims.size(), 1);
        for(size_t i = 0; i < testCase.dims.size(); ++i)
        {
            if(i < testCase.normalizedDim)
            {
                statDims[i] = testCase.dims[i];
            }
            else
            {
                affineDims[i] = testCase.dims[i];
            }
        }

        graph::Graph graphObj;

        graphObj.set_name("LayernormBwdTest");

        graphObj.set_intermediate_data_type(DataType::FLOAT).set_compute_data_type(DataType::FLOAT);

        auto ioStrides = generateStrides(testCase.dims, layout.strideOrder);
        auto statStrides = generateStrides(statDims, layout.strideOrder);
        auto affineStrides = generateStrides(affineDims, layout.strideOrder);

        auto dyAttr = graph::makeTensorAttributes("dY", outputDataType, testCase.dims, ioStrides);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto xAttr = graph::makeTensorAttributes("X", inputDataType, testCase.dims, ioStrides);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale", scaleBiasDataType, affineDims, affineStrides);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        graph::LayernormBackwardAttributes lnAttrs;

        if(testCase.optionalTensors)
        {
            auto meanAttr = graph::makeTensorAttributes(
                "mean", meanInvVarianceDataType, statDims, statStrides);
            auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

            auto rstdAttr = graph::makeTensorAttributes(
                "rstd", meanInvVarianceDataType, statDims, statStrides);
            auto rstdTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(rstdAttr));

            lnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, rstdTensorAttr);
        }

        auto epsilonAttr
            = graph::makeTensorAttributes("epsilon", static_cast<float>(LAYERNORM_DEFAULT_EPSILON));
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));
        lnAttrs.set_epsilon(std::move(epsilonTensorAttr));

        auto results
            = graphObj.layernorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, lnAttrs);
        const auto& dxTensorAttr = results[0];
        const auto& dscaleTensorAttr = results[1];
        const auto& dbiasTensorAttr = results[2];

        dxTensorAttr->set_output(true);
        dxTensorAttr->set_data_type(inputDataType);
        dscaleTensorAttr->set_output(true);
        dscaleTensorAttr->set_data_type(scaleBiasDataType);
        dbiasTensorAttr->set_output(true);
        dbiasTensorAttr->set_data_type(scaleBiasDataType);

        this->registerValidator(dxTensorAttr, getTolerance<InputDataType>());
        // Summing hundreds of thousands of floating point values yields a low magnitude value with a high variance from floating point inaccuracies, so a higher tolerance in necessary for dscale and dbias
        this->registerValidator(dscaleTensorAttr, getTolerance<ScaleBiasDataType>() * 16);
        this->registerValidator(dbiasTensorAttr, getTolerance<ScaleBiasDataType>() * 16);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// ============================================================================
// Test cases
// ============================================================================

// "Pure" = input, output, scale/bias, and mean/invvariance all the same type.
// "Mixed" = input/output same type (FP16/BFP16), but scale/bias and mean/invvariance are FP32.
// "Upcast" = input is FP16/BFP16 but output widens to FP32 (scale/bias and
// mean/invvariance are FP32).

// ============================================================================
// NCHW
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNchwPureFp32 = LayernormBackward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormBackwardNchwMixedFp16 = LayernormBackward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNchwMixedBfp16
    = LayernormBackward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNchwUpcastFp16 = LayernormBackward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNchwUpcastBfp16
    = LayernormBackward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormBackwardNchwPureFp16 = LayernormBackward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNchwPureBfp16
    = LayernormBackward<bfloat16, bfloat16, bfloat16, bfloat16>;

// ============================================================================
// NHWC
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNhwcPureFp32 = LayernormBackward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormBackwardNhwcMixedFp16 = LayernormBackward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNhwcMixedBfp16
    = LayernormBackward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNhwcUpcastFp16 = LayernormBackward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNhwcUpcastBfp16
    = LayernormBackward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormBackwardNhwcPureFp16 = LayernormBackward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNhwcPureBfp16
    = LayernormBackward<bfloat16, bfloat16, bfloat16, bfloat16>;

// ============================================================================
// NCDHW
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNcdhwPureFp32 = LayernormBackward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormBackwardNcdhwMixedFp16 = LayernormBackward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNcdhwMixedBfp16
    = LayernormBackward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNcdhwUpcastFp16 = LayernormBackward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNcdhwUpcastBfp16
    = LayernormBackward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormBackwardNcdhwPureFp16 = LayernormBackward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNcdhwPureBfp16
    = LayernormBackward<bfloat16, bfloat16, bfloat16, bfloat16>;

// ============================================================================
// NDHWC
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNdhwcPureFp32 = LayernormBackward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormBackwardNdhwcMixedFp16 = LayernormBackward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNdhwcMixedBfp16
    = LayernormBackward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNdhwcUpcastFp16 = LayernormBackward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormBackwardNdhwcUpcastBfp16
    = LayernormBackward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormBackwardNdhwcPureFp16 = LayernormBackward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormBackwardNdhwcPureBfp16
    = LayernormBackward<bfloat16, bfloat16, bfloat16, bfloat16>;

} // namespace

// ============================================================================
// Test Registrations
// ============================================================================

TEST_P(IntegrationGpuLayernormBackwardNchwPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNchwPureFp32,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNchwPureFp32,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNchwMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNchwMixedFp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNchwMixedFp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNchwMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNchwMixedBfp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNchwMixedBfp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNchwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNchwUpcastFp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNchwUpcastFp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNchwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNchwUpcastBfp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNchwUpcastBfp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNchwPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNchwPureFp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNchwPureFp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNchwPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNchwPureBfp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNchwPureBfp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNhwcPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNhwcPureFp32,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNhwcPureFp32,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNhwcMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNhwcMixedFp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNhwcMixedFp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNhwcMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNhwcMixedBfp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNhwcMixedBfp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNhwcUpcastFp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNhwcUpcastFp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNhwcUpcastBfp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNhwcUpcastBfp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNhwcPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNhwcPureFp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNhwcPureFp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNhwcPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNhwcPureBfp16,
                         testing::ValuesIn(getLayernorm4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNhwcPureBfp16,
                         testing::ValuesIn(getLayernorm4DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNcdhwPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNcdhwPureFp32,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNcdhwPureFp32,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNcdhwMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNcdhwMixedFp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNcdhwMixedFp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNcdhwMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNcdhwMixedBfp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNcdhwMixedBfp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNcdhwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNcdhwUpcastFp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNcdhwUpcastFp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNcdhwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNcdhwUpcastBfp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNcdhwUpcastBfp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNcdhwPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNcdhwPureFp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNcdhwPureFp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNcdhwPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNcdhwPureBfp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNcdhwPureBfp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNdhwcPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNdhwcPureFp32,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNdhwcPureFp32,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNdhwcMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNdhwcMixedFp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNdhwcMixedFp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNdhwcMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNdhwcMixedBfp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNdhwcMixedBfp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNdhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNdhwcUpcastFp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNdhwcUpcastFp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNdhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNdhwcUpcastBfp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNdhwcUpcastBfp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNdhwcPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNdhwcPureFp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNdhwcPureFp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

TEST_P(IntegrationGpuLayernormBackwardNdhwcPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormBackwardNdhwcPureBfp16,
                         testing::ValuesIn(getLayernorm5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormBackwardNdhwcPureBfp16,
                         testing::ValuesIn(getLayernorm5DFullTestCases()));

} // namespace hip_kernel_provider::layernorm::test
