// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormBpropPlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct LayernormBpropSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;

    LayernormBpropSignatureKey() = default;
    constexpr LayernormBpropSignatureKey(
        hipdnn_flatbuffers_sdk::data_objects::DataType y,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBias,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVariance,
        hipdnn_flatbuffers_sdk::data_objects::DataType x,
        hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : yDataType(y)
        , scaleBiasDataType(scaleBias)
        , meanInvVarianceDataType(meanInvVariance)
        , xDataType(x)
        , computeDataType(compute)
    {
    }

    LayernormBpropSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_LayernormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes could not be cast to LayernormBackwardAttributes");
        }

        auto dyTensorAttr = tensorMap.at(nodeAttributes->dy_tensor_uid());
        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto dxTensorAttr = tensorMap.at(nodeAttributes->dx_tensor_uid());

        if(dyTensorAttr == nullptr || xTensorAttr == nullptr || dxTensorAttr == nullptr)
        {
            throw std::runtime_error("One or more tensor attributes could not be found in the map, "
                                     "failed to construct key");
        }

        xDataType = xTensorAttr->data_type();
        yDataType = dyTensorAttr->data_type();
        computeDataType = node.compute_data_type();

        // Scale/bias type: use scale tensor type
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        scaleBiasDataType = scaleTensorAttr->data_type();

        // Mean/inv_variance type: use mean if present, otherwise default to IO type (dy type)
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            meanInvVarianceDataType = meanTensorAttr->data_type();
        }
        else
        {
            // If the mean/inverse variance type is unknown, the scale/bias type should match it (see getPlanBuilders)
            meanInvVarianceDataType = scaleBiasDataType;
        }
    }

    std::size_t operator()(const LayernormBpropSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(nodeType) ^ (static_cast<std::size_t>(yDataType) << 4)
               ^ (static_cast<std::size_t>(scaleBiasDataType) << 8)
               ^ (static_cast<std::size_t>(meanInvVarianceDataType) << 12)
               ^ (static_cast<std::size_t>(xDataType) << 16)
               ^ (static_cast<std::size_t>(computeDataType) << 20);
    }

    bool operator==(const LayernormBpropSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && yDataType == other.yDataType
               && scaleBiasDataType == other.scaleBiasDataType
               && meanInvVarianceDataType == other.meanInvVarianceDataType
               && xDataType == other.xDataType && computeDataType == other.computeDataType;
    }

    static std::unordered_map<LayernormBpropSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              LayernormBpropSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<LayernormBpropSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           LayernormBpropSignatureKey>
            map;

        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        // MIOpen-compatible: all tensors same type, compute in float
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<LayernormBpropSignatureKey,
                                                  std::unique_ptr<IGraphNodePlanBuilder>,
                                                  LayernormBpropSignatureKey>& map)
    {
        map[LayernormBpropSignatureKey(YDataTypeEnum,
                                       ScaleBiasDataTypeEnum,
                                       MeanInvVarianceDataTypeEnum,
                                       XDataTypeEnum,
                                       ComputeDataTypeEnum)]
            = std::make_unique<LayernormBpropPlanBuilder<YDataTypeEnum,
                                                         ScaleBiasDataTypeEnum,
                                                         MeanInvVarianceDataTypeEnum,
                                                         XDataTypeEnum,
                                                         ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const LayernormBpropSignatureKey& key)
{
    os << "Layernorm(xDx=" << key.xDataType << ", dy=" << key.yDataType
       << ", scaleDscaleDbias=" << key.scaleBiasDataType
       << ", meanInvVar=" << key.meanInvVarianceDataType << ", compute=" << key.computeDataType
       << ")";
    return os;
}

} // namespace hipdnn_test_sdk::detail
