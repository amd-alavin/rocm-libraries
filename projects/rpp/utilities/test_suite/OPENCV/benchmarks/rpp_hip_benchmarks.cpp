/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "benchmarks_common.h"
#include <chrono>

using namespace std::chrono;

// Forward declaration of helper function from rpp_test_suite_image.h
// We declare it here to avoid multiple definition errors
inline void set_descriptor_dims_and_strides_local(RpptDescPtr descPtr, int noOfImages, int maxHeight,
                                                   int maxWidth, int numChannels, int offsetInBytes,
                                                   int additionalStride = 0) {
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->c = numChannels;

    // CRITICAL: Align width to multiple of 8 for HIP backend memory alignment
    // This matches the reference implementation in rpp_test_suite_image.h
    descPtr->w = (maxWidth / 8) * 8 + 8 + additionalStride;

    // NOTE: Strides will be set by update_strides_from_layout() after layout is assigned
    // Initialize with default NHWC strides for now
    descPtr->strides.nStride = descPtr->h * descPtr->w * descPtr->c;
    descPtr->strides.hStride = descPtr->w * descPtr->c;
    descPtr->strides.wStride = descPtr->c;
    descPtr->strides.cStride = 1;
}

// Helper function to update strides based on layout
inline void update_strides_from_layout(RpptDescPtr descPtr) {
    int h = descPtr->h;
    int w = descPtr->w;
    int c = descPtr->c;

    if (descPtr->layout == RpptLayout::NCHW) {
        // NCHW layout: Batch-Channels-Height-Width
        descPtr->strides.nStride = c * h * w;
        descPtr->strides.cStride = h * w;
        descPtr->strides.hStride = w;
        descPtr->strides.wStride = 1;
    } else if (descPtr->layout == RpptLayout::NHWC) {
        // NHWC layout: Batch-Height-Width-Channels
        descPtr->strides.nStride = h * w * c;
        descPtr->strides.hStride = w * c;
        descPtr->strides.wStride = c;
        descPtr->strides.cStride = 1;
    }
    // Add other layouts as needed
}

// Macro for checking HIP status
#define CHECK_HIP_STATUS(call)                                                                  \
    do {                                                                                        \
        hipError_t err = (call);                                                                \
        if (err != hipSuccess) {                                                                \
            std::cerr << "HIP error in " << #call << ": " << hipGetErrorString(err) << std::endl; \
        }                                                                                       \
    } while (0)

// Macro for checking RPP status
#define CHECK_RPP_STATUS(call, func_name)                                                       \
    do {                                                                                        \
        RppStatus status = (call);                                                              \
        if (status != RPP_SUCCESS) {                                                            \
            std::cerr << "RPP HIP " << func_name << " failed with status: " << status << std::endl; \
        }                                                                                       \
    } while (0)

// ==================== RPP HIP/GPU BENCHMARK FUNCTIONS ====================
//
// IMPORTANT: Memory allocation pattern for RPP HIP backend:
// 1. Image data (input/output): Device memory (hipMalloc)
// 2. Parameter tensors (alpha, beta, gamma, etc.): Pinned HOST memory (hipHostMalloc)
// 3. ROI tensors: Pinned HOST memory (hipHostMalloc)
//
// The RPP HIP backend expects parameter and ROI pointers to be in HOST memory,
// NOT device memory. Use hipHostMalloc for pinned host memory allocation.
//

void benchmark_RPP_HIP_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                                  rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *alphaTensor, *betaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&betaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        alphaTensor[i] = alpha;
        betaTensor[i] = beta;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_brightness(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &alphaTensor[i], &betaTensor[i],
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Brightness");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(betaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "alpha=" << alpha << ", beta=" << beta;
    printResult("RPP HIP Brightness", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma,
                                      rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *gammaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&gammaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        gammaTensor[i] = gamma;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gamma_correction(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &gammaTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                                  handle, RPP_HIP_BACKEND),
                            "GammaCorrection");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(gammaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "gamma=" << gamma;
    printResult("RPP HIP GammaCorrection", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

// ==================== TODO: REMAINING OPERATORS ====================
// The following operators are not yet implemented.
// Implementation pattern for all operators:
//   1. Setup descriptors and ROIs
//   2. Allocate GPU memory for ALL images (outside timing loop)
//   3. Copy data to GPU (outside timing loop)
//   4. START TIMING
//   5. Benchmark loop: for(k) { for(i) { RPP function call } }
//   6. Synchronize stream
//   7. END TIMING
//   8. Free GPU memory
// See implemented operators above (Brightness, GammaCorrection, etc.) for reference.

#define HIP_PLACEHOLDER(func_name) \
    cout << "RPP HIP " << func_name << " - Not yet implemented (TODO)" << endl;

void benchmark_RPP_HIP_Blend(const vector<Mat>& imgs, bool isColor, float alpha,
                            rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].convertTo(imgs2[i], -1, 0.8, 30);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *alphaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        alphaTensor[i] = alpha;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t imageBufferSize = imgs[i].total() * imgs[i].elemSize();

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], imageBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], imageBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], imageBufferSize));

        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], imgs[i].data, imageBufferSize, hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], imgs2[i].data, imageBufferSize, hipMemcpyHostToDevice));
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_blend(d_inputs1[i], d_inputs2[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       &alphaTensor[i], &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Blend");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP Blend", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "alpha=" + to_string(alpha));
}

void benchmark_RPP_HIP_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                               float contrastCenter, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *contrastFactorTensor;
    Rpp32f *contrastCenterTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&contrastFactorTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&contrastCenterTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        contrastFactorTensor[i] = contrastFactor;
        contrastCenterTensor[i] = contrastCenter;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_contrast(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &contrastFactorTensor[i], &contrastCenterTensor[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Contrast");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(contrastFactorTensor));
    CHECK_HIP_STATUS(hipHostFree(contrastCenterTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << contrastFactor << ", center=" << contrastCenter;
    printResult("RPP HIP Contrast", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Exposure(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *exposureTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&exposureTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        exposureTensor[i] = exposureFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_exposure(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &exposureTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                          handle, RPP_HIP_BACKEND),
                            "Exposure");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(exposureTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP Exposure", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "factor=" + to_string(exposureFactor));
}

void benchmark_RPP_HIP_Hue(const vector<Mat>& imgs, float hueDelta, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *hueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&hueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        hueTensor[i] = hueDelta;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_hue(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                     &hueTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                     handle, RPP_HIP_BACKEND),
                            "Hue");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(hueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP Hue", imgs.size(), true,
                duration<double, milli>(end - start).count(), "hueDelta=" + to_string(hueDelta));
}

void benchmark_RPP_HIP_Saturation(const vector<Mat>& imgs, float satFactor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *satTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&satTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        satTensor[i] = satFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_saturation(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &satTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                            handle, RPP_HIP_BACKEND),
                            "Saturation");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(satTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP Saturation", imgs.size(), true,
                duration<double, milli>(end - start).count(), "factor=" + to_string(satFactor));
}

void benchmark_RPP_HIP_ColorToGreyscale(const vector<Mat>& imgs, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int srcChannels = 3;
    int dstChannels = 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, srcChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, imgs[i].rows, imgs[i].cols, dstChannels, 0);
        dstDescs[i].layout = RpptLayout::NCHW;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        size_t srcBufferSize = imgs[i].total() * imgs[i].elemSize();
        size_t dstBufferSize = imgs[i].rows * imgs[i].cols;

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], imgs[i].data, srcBufferSize, hipMemcpyHostToDevice));
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_to_greyscale(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                     RpptSubpixelLayout::BGRtype, handle, RPP_HIP_BACKEND),
                            "ColorToGreyscale");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }

    printResult("RPP HIP ColorToGreyscale", imgs.size(), true,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                                  float saturation, float hue, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *brightnessTensor;
    Rpp32f *contrastTensor;
    Rpp32f *saturationTensor;
    Rpp32f *hueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&brightnessTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&contrastTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saturationTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&hueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        brightnessTensor[i] = brightness;
        contrastTensor[i] = contrast;
        saturationTensor[i] = saturation;
        hueTensor[i] = hue;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_jitter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                              &brightnessTensor[i], &contrastTensor[i],
                                              &saturationTensor[i], &hueTensor[i],
                                              &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorJitter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(brightnessTensor));
    CHECK_HIP_STATUS(hipHostFree(contrastTensor));
    CHECK_HIP_STATUS(hipHostFree(saturationTensor));
    CHECK_HIP_STATUS(hipHostFree(hueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "brightness=" << brightness << ", contrast=" << contrast
           << ", saturation=" << saturation << ", hue=" << hue;
    printResult("RPP HIP ColorJitter", imgs.size(), true,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling: offsetInBytes >= 12 * (kernelSize / 2)
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_box_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            kSize, borderType, &roiTensor[i], RpptRoiType::XYWH,
                                            handle, RPP_HIP_BACKEND),
                            "BoxFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BoxFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "kernel=" + to_string(kernelSize));
}

void benchmark_RPP_HIP_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                   rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_median_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                               kSize, borderType, &roiTensor[i], RpptRoiType::XYWH,
                                               handle, RPP_HIP_BACKEND),
                            "MedianFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP MedianFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "kernel=" + to_string(kernelSize));
}

void benchmark_RPP_HIP_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                     double sigma, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    float stdDev = static_cast<float>(sigma);
    Rpp32f *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        stdDevTensor[i] = stdDev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gaussian_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &stdDevTensor[i], kSize, borderType,
                                                  &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "GaussianFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP GaussianFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "kernel=" + to_string(kernelSize));
}

void benchmark_RPP_HIP_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType,
                                  rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    int dstChannels = 1;
    Rpp32u kernelSize = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, imgs[i].rows, imgs[i].cols, dstChannels, 0);
        dstDescs[i].layout = RpptLayout::NCHW;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t srcBufferSize = imgs[i].total() * imgs[i].elemSize();
        size_t dstBufferSize = imgs[i].rows * imgs[i].cols;

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], imgs[i].data, srcBufferSize, hipMemcpyHostToDevice));
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_sobel_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                              sobelType, kernelSize, &roiTensor[i], RpptRoiType::XYWH,
                                              handle, RPP_HIP_BACKEND),
                            "SobelFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP SobelFilter", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "type=" + to_string(sobelType));
}

void benchmark_RPP_HIP_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize,
                            rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_erode(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       kernelSize, &roiTensor[i], RpptRoiType::XYWH,
                                       handle, RPP_HIP_BACKEND),
                            "Erode");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP Erode", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_dilate(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        kernelSize, &roiTensor[i], RpptRoiType::XYWH,
                                        handle, RPP_HIP_BACKEND),
                            "Dilate");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP Dilate", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *strengthTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&strengthTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        strengthTensor[i] = strength;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_emboss(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        &strengthTensor[i], kSize, borderType,
                                        &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Emboss");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(strengthTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize << ", strength=" << strength;
    printResult("RPP HIP Emboss", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight,
                           rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, cropHeight, cropWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        roiTensor[i].xywhROI.xy.x = (imgs[i].cols - cropWidth) / 2;
        roiTensor[i].xywhROI.xy.y = (imgs[i].rows - cropHeight) / 2;
        roiTensor[i].xywhROI.roiWidth = cropWidth;
        roiTensor[i].xywhROI.roiHeight = cropHeight;

        // Calculate buffer sizes based on aligned descriptor dimensions
        size_t srcAlignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstAlignedBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcAlignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstAlignedBufferSize));

        // Copy source image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[srcAlignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcAlignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                      &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Crop");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "size=" << cropWidth << "x" << cropHeight;
    printResult("RPP HIP Crop", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                             RpptInterpolationType interpType, const string& interpName,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptImagePatch *dstImgSizes;
    RpptROI *roiTensor;

    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, num_images * sizeof(RpptImagePatch)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, dstH, dstW, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        dstImgSizes[i].width = dstW;
        dstImgSizes[i].height = dstH;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t srcAlignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstAlignedBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcAlignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstAlignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[srcAlignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcAlignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_resize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        &dstImgSizes[i], interpType, &roiTensor[i], RpptRoiType::XYWH,
                                        handle, RPP_HIP_BACKEND),
                            "Resize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "type=" << interpName << ", size=" << dstW << "x" << dstH;
    printResult("RPP HIP Resize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Flip(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u horizontalFlag = (flipCode == 1 || flipCode == -1) ? 1 : 0;
    Rpp32u verticalFlag = (flipCode == 0 || flipCode == -1) ? 1 : 0;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32u *horizontalTensor;
    Rpp32u *verticalTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&horizontalTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&verticalTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        horizontalTensor[i] = horizontalFlag;
        verticalTensor[i] = verticalFlag;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // NOTE: rppt_flip has a bug causing memory corruption on repeated calls
    // Run only once to get a valid measurement, then multiply time by NUM_RUNS for fair comparison
    auto start = high_resolution_clock::now();
    for (int i = 0; i < num_images; ++i) {
        CHECK_RPP_STATUS(rppt_flip(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                  &horizontalTensor[i], &verticalTensor[i],
                                  &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Flip");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    // Multiply by NUM_RUNS to normalize with other benchmarks
    double singleRunTime = duration<double, milli>(end - start).count();
    double adjustedTime = singleRunTime * NUM_RUNS;

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(horizontalTensor));
    CHECK_HIP_STATUS(hipHostFree(verticalTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    string name = (flipCode == 1) ? "Horizontal" : (flipCode == 0) ? "Vertical" : "Both";
    printResult("RPP HIP Flip", imgs.size(), isColor, adjustedTime, "type=" + name);
}

void benchmark_RPP_HIP_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *angleTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&angleTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        angleTensor[i] = angleDeg;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_rotate(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        &angleTensor[i], RpptInterpolationType::BILINEAR,
                                        &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Rotate");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(angleTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "angle=" << angleDeg << "deg";
    printResult("RPP HIP Rotate", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_WarpAffine(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    float *affineTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&affineTensor, num_images * 6 * sizeof(float)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    // Affine transformation matrix: [a11, a12, a13, a21, a22, a23]
    // Same transformation for all images
    float affine[6] = {1.0f, 0.1f, 10.0f, 0.1f, 1.0f, 10.0f};

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Copy affine matrix for this image
        memcpy(&affineTensor[i * 6], affine, 6 * sizeof(float));

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_warp_affine(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                             &affineTensor[i * 6], RpptInterpolationType::BILINEAR,
                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "WarpAffine");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(affineTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP WarpAffine", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_WarpPerspective(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    float *perspectiveTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&perspectiveTensor, num_images * 9 * sizeof(float)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    // Perspective transformation matrix: 3x3 homography matrix
    // Same transformation for all images
    float perspectiveMatrix[9] = {1.0f, 0.1f, 0.0f, 0.1f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Copy perspective matrix for this image
        memcpy(&perspectiveTensor[i * 9], perspectiveMatrix, 9 * sizeof(float));

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_warp_perspective(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                   &perspectiveTensor[i * 9], RpptInterpolationType::BILINEAR,
                                                   &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "WarpPerspective");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(perspectiveTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP WarpPerspective", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_Fisheye(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_fisheye(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                         &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Fisheye");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP Fisheye", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_LensCorrection(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    if (num_images == 0) return;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Camera matrix and distortion coefficients in pinned host memory
    Rpp32f *cameraMatrixTensor;
    Rpp32f *distortionCoeffsTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&cameraMatrixTensor, num_images * 9 * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&distortionCoeffsTensor, num_images * 8 * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    // Sample camera calibration parameters
    Rpp32f sampleCameraMatrix[9] = {534.07088364f, 0.0f, 341.53407554f, 0.0f, 534.11914595f, 232.94565259f, 0.0f, 0.0f, 1.0f};
    Rpp32f sampleDistortion[8] = {-0.29297164f, 0.10770696f, 0.00131038f, -0.0000311f, 0.0434798f, 0.0f, 0.0f, 0.0f};

    // Allocate remap tables in device memory
    // Table descriptor for remap tables
    RpptDesc tableDesc;
    set_descriptor_dims_and_strides_local(&tableDesc, num_images, imgs[0].rows, imgs[0].cols, 1, 0);
    tableDesc.layout = RpptLayout::NCHW;
    tableDesc.dataType = RpptDataType::F32;
    update_strides_from_layout(&tableDesc);

    size_t tableSize = num_images * imgs[0].rows * imgs[0].cols * sizeof(Rpp32f);
    Rpp32f *d_rowRemapTable, *d_colRemapTable;
    CHECK_HIP_STATUS(hipMalloc(&d_rowRemapTable, tableSize));
    CHECK_HIP_STATUS(hipMalloc(&d_colRemapTable, tableSize));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Copy camera matrix and distortion coefficients for this image
        memcpy(&cameraMatrixTensor[i * 9], sampleCameraMatrix, 9 * sizeof(Rpp32f));
        memcpy(&distortionCoeffsTensor[i * 8], sampleDistortion, 8 * sizeof(Rpp32f));

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_lens_correction(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  d_rowRemapTable, d_colRemapTable, &tableDesc,
                                                  &cameraMatrixTensor[i * 9], &distortionCoeffsTensor[i * 8],
                                                  &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "LensCorrection");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_rowRemapTable));
    CHECK_HIP_STATUS(hipFree(d_colRemapTable));
    CHECK_HIP_STATUS(hipHostFree(cameraMatrixTensor));
    CHECK_HIP_STATUS(hipHostFree(distortionCoeffsTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP LensCorrection", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> h_inputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(h_inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Allocate device memory
    Rpp32f *d_inputBuffer, *d_outputBuffer;
    size_t bufferSize = num_images * imageSize * sizeof(Rpp32f);
    CHECK_HIP_STATUS(hipMalloc(&d_inputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_outputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_inputBuffer, h_inputBuffer.data(), bufferSize, hipMemcpyHostToDevice));

    // Allocate pinned host memory for add tensor
    Rpp32f *addTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&addTensor, num_images * sizeof(Rpp32f)));
    for (int i = 0; i < num_images; ++i) {
        addTensor[i] = addVal;
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Allocate ROI3D array in pinned host memory
    RpptROI3D *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI3D)));
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(rppt_add_scalar(d_inputBuffer, &srcGenericDesc, d_outputBuffer,
                                        &dstGenericDesc, addTensor, roiTensor,
                                        RpptRoi3DType::XYZWHD, handle, RPP_HIP_BACKEND),
                        "AddScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    CHECK_HIP_STATUS(hipFree(d_inputBuffer));
    CHECK_HIP_STATUS(hipFree(d_outputBuffer));
    CHECK_HIP_STATUS(hipHostFree(addTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP AddScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "value=" + to_string(addVal));
}

void benchmark_RPP_HIP_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal,
                                     rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> h_inputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(h_inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Allocate device memory
    Rpp32f *d_inputBuffer, *d_outputBuffer;
    size_t bufferSize = num_images * imageSize * sizeof(Rpp32f);
    CHECK_HIP_STATUS(hipMalloc(&d_inputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_outputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_inputBuffer, h_inputBuffer.data(), bufferSize, hipMemcpyHostToDevice));

    // Allocate pinned host memory for subtract tensor
    Rpp32f *subtractTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&subtractTensor, num_images * sizeof(Rpp32f)));
    for (int i = 0; i < num_images; ++i) {
        subtractTensor[i] = subVal;
    }

    // Create 5D generic descriptor
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Allocate ROI3D array in pinned host memory
    RpptROI3D *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI3D)));
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(rppt_subtract_scalar(d_inputBuffer, &srcGenericDesc, d_outputBuffer,
                                              &dstGenericDesc, subtractTensor, roiTensor,
                                              RpptRoi3DType::XYZWHD, handle, RPP_HIP_BACKEND),
                        "SubtractScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    CHECK_HIP_STATUS(hipFree(d_inputBuffer));
    CHECK_HIP_STATUS(hipFree(d_outputBuffer));
    CHECK_HIP_STATUS(hipHostFree(subtractTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP SubtractScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "value=" + to_string(subVal));
}

void benchmark_RPP_HIP_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal,
                                     rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> h_inputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(h_inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Allocate device memory
    Rpp32f *d_inputBuffer, *d_outputBuffer;
    size_t bufferSize = num_images * imageSize * sizeof(Rpp32f);
    CHECK_HIP_STATUS(hipMalloc(&d_inputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_outputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_inputBuffer, h_inputBuffer.data(), bufferSize, hipMemcpyHostToDevice));

    // Allocate pinned host memory for multiply tensor
    Rpp32f *mulTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&mulTensor, num_images * sizeof(Rpp32f)));
    for (int i = 0; i < num_images; ++i) {
        mulTensor[i] = mulVal;
    }

    // Create 5D generic descriptor
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Allocate ROI3D array in pinned host memory
    RpptROI3D *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI3D)));
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        CHECK_RPP_STATUS(rppt_multiply_scalar(d_inputBuffer, &srcGenericDesc, d_outputBuffer,
                                              &dstGenericDesc, mulTensor, roiTensor,
                                              RpptRoi3DType::XYZWHD, handle, RPP_HIP_BACKEND),
                        "MultiplyScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    CHECK_HIP_STATUS(hipFree(d_inputBuffer));
    CHECK_HIP_STATUS(hipFree(d_outputBuffer));
    CHECK_HIP_STATUS(hipHostFree(mulTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP MultiplyScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), "value=" + to_string(mulVal));
}

void benchmark_RPP_HIP_BitwiseAnd(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs2[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_and(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                             d_outputs[i], &dstDescs[i], &roiTensor[i],
                                             RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseAnd");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseAnd", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_BitwiseOr(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs2[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_or(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                            d_outputs[i], &dstDescs[i], &roiTensor[i],
                                            RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseOr");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseOr", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_BitwiseNot(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_not(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseNot");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseNot", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_BitwiseXor(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs2[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_xor(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                             d_outputs[i], &dstDescs[i], &roiTensor[i],
                                             RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseXor");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseXor", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_Threshold(const vector<Mat>& imgs, bool isColor, double thresh,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *minTensor;
    Rpp32f *maxTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&minTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&maxTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Binary threshold: pixels below thresh -> 0, pixels >= thresh -> 255
        minTensor[i] = (Rpp32f)thresh;
        maxTensor[i] = 255.0f;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_threshold(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                           &minTensor[i], &maxTensor[i],
                                           &roiTensor[i], RpptRoiType::XYWH,
                                           handle, RPP_HIP_BACKEND),
                            "Threshold");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(minTensor));
    CHECK_HIP_STATUS(hipHostFree(maxTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "threshold=" << thresh;
    printResult("RPP HIP Threshold", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_HistogramEqualize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_histogram_equalize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                     &roiTensor[i], RpptRoiType::XYWH,
                                                     handle, RPP_HIP_BACKEND),
                            "HistogramEqualize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP HistogramEqualize", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_LUT(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate LUT and ROI tensors in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    // Allocate device memory for LUT (256 entries for 8-bit input)
    Rpp8u *d_lut;
    CHECK_HIP_STATUS(hipMalloc(&d_lut, 256 * sizeof(Rpp8u)));

    // Create inverse LUT (invert pixel values)
    Rpp8u h_lut[256];
    for (int i = 0; i < 256; ++i) {
        h_lut[i] = 255 - i;  // Invert: 0→255, 255→0
    }
    CHECK_HIP_STATUS(hipMemcpy(d_lut, h_lut, 256 * sizeof(Rpp8u), hipMemcpyHostToDevice));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_lut(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                     d_lut, &roiTensor[i], RpptRoiType::XYWH,
                                     handle, RPP_HIP_BACKEND),
                            "LUT");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_lut));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "lut=inverse";
    printResult("RPP HIP LUT", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Magnitude(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROI tensor in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Create second input (shifted version for demonstration)
        for (size_t j = 0; j < alignedBufferSize; ++j) {
            h_tempBuffer[j] = (h_tempBuffer[j] >> 1);  // Half the values
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            // Magnitude: sqrt(src1^2 + src2^2)
            CHECK_RPP_STATUS(rppt_magnitude(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                           d_outputs[i], &dstDescs[i],
                                           &roiTensor[i], RpptRoiType::XYWH,
                                           handle, RPP_HIP_BACKEND),
                            "Magnitude");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HIP Magnitude", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Phase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROI tensor in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Create second input (shifted version for demonstration)
        for (size_t j = 0; j < alignedBufferSize; ++j) {
            h_tempBuffer[j] = (h_tempBuffer[j] >> 1);  // Half the values
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            // Phase: atan2(src2, src1) in degrees
            CHECK_RPP_STATUS(rppt_phase(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                       d_outputs[i], &dstDescs[i],
                                       &roiTensor[i], RpptRoiType::XYWH,
                                       handle, RPP_HIP_BACKEND),
                            "Phase");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HIP Phase", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Remap(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptDesc> tableDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);
    vector<Rpp32f*> d_rowRemapTable(num_images);
    vector<Rpp32f*> d_colRemapTable(num_images);

    // Allocate ROI tensor in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Table descriptor (same dimensions as image, 1 channel for row/col maps)
        set_descriptor_dims_and_strides_local(&tableDescs[i], 1, imgs[i].rows, imgs[i].cols, 1, 0);
        tableDescs[i].layout = RpptLayout::NCHW;
        tableDescs[i].dataType = RpptDataType::F32;
        update_strides_from_layout(&tableDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;

        // Calculate buffer sizes
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t tableSize = tableDescs[i].n * tableDescs[i].h * tableDescs[i].w * sizeof(Rpp32f);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_rowRemapTable[i], tableSize));
        CHECK_HIP_STATUS(hipMalloc(&d_colRemapTable[i], tableSize));

        // Copy image data
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int elementsPerRow = width * numChannels;

        for (int row = 0; row < height; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;

        // Create remap tables (horizontal flip for demonstration)
        Rpp32f* h_rowTable = new Rpp32f[tableSize / sizeof(Rpp32f)]();
        Rpp32f* h_colTable = new Rpp32f[tableSize / sizeof(Rpp32f)]();

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < tableDescs[i].w; ++col) {
                int idx = row * tableDescs[i].w + col;
                h_rowTable[idx] = (Rpp32f)row;  // Keep row same
                if (col < width) {
                    h_colTable[idx] = (Rpp32f)(width - 1 - col);  // Flip horizontally
                } else {
                    h_colTable[idx] = 0.0f;  // Padding
                }
            }
        }

        CHECK_HIP_STATUS(hipMemcpy(d_rowRemapTable[i], h_rowTable, tableSize, hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(d_colRemapTable[i], h_colTable, tableSize, hipMemcpyHostToDevice));
        delete[] h_rowTable;
        delete[] h_colTable;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_remap(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       d_rowRemapTable[i], d_colRemapTable[i], &tableDescs[i],
                                       RpptInterpolationType::BILINEAR,
                                       &roiTensor[i], RpptRoiType::XYWH,
                                       handle, RPP_HIP_BACKEND),
                            "Remap");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
        CHECK_HIP_STATUS(hipFree(d_rowRemapTable[i]));
        CHECK_HIP_STATUS(hipFree(d_colRemapTable[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "transform=horizontal_flip";
    printResult("RPP HIP Remap", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_TensorMin(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // Output length: grayscale = n, RGB = n * 4 (R, G, B, overall per image)
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp8u *d_minOutputs;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_minOutputs, outputLength * sizeof(Rpp8u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_min(d_inputs[i], &srcDescs[i],
                                            d_minOutputs + (i * imgOutputLength), imgOutputLength,
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorMin");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_minOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorMin", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_TensorMax(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp8u *d_maxOutputs;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_maxOutputs, outputLength * sizeof(Rpp8u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_max(d_inputs[i], &srcDescs[i],
                                            d_maxOutputs + (i * imgOutputLength), imgOutputLength,
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorMax");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_maxOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorMax", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_TensorSum(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp64u *d_sumOutputs;  // Note: tensor_sum returns Rpp64u for U8 input
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_sumOutputs, outputLength * sizeof(Rpp64u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_sum(d_inputs[i], &srcDescs[i],
                                            d_sumOutputs + (i * imgOutputLength), imgOutputLength,
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorSum");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_sumOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorSum", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_TensorMean(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp32f *d_meanOutputs;  // Note: tensor_mean returns Rpp32f
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_meanOutputs, outputLength * sizeof(Rpp32f)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_mean(d_inputs[i], &srcDescs[i],
                                             d_meanOutputs + (i * imgOutputLength), imgOutputLength,
                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorMean");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_meanOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorMean", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_TensorStddev(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp32f *d_meanOutputs;
    Rpp32f *d_stddevOutputs;
    Rpp32f *h_meanOutputs;  // Host buffer to compute mean first
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&h_meanOutputs, outputLength * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_meanOutputs, outputLength * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_stddevOutputs, outputLength * sizeof(Rpp32f)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // First compute mean (required for stddev calculation)
    for (int i = 0; i < num_images; ++i) {
        Rpp32u imgOutputLength = isColor ? 4 : 1;
        CHECK_RPP_STATUS(rppt_tensor_mean(d_inputs[i], &srcDescs[i],
                                         d_meanOutputs + (i * imgOutputLength), imgOutputLength,
                                         &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "TensorMean (for stddev)");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));

    // Copy mean to host (stddev requires host mean tensor)
    CHECK_HIP_STATUS(hipMemcpy(h_meanOutputs, d_meanOutputs, outputLength * sizeof(Rpp32f), hipMemcpyDeviceToHost));

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_stddev(d_inputs[i], &srcDescs[i],
                                               d_stddevOutputs + (i * imgOutputLength), imgOutputLength,
                                               h_meanOutputs + (i * imgOutputLength),
                                               &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorStddev");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_meanOutputs));
    CHECK_HIP_STATUS(hipFree(d_stddevOutputs));
    CHECK_HIP_STATUS(hipHostFree(h_meanOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorStddev", imgs.size(), isColor,
                duration<double, milli>(end - start).count());
}

void benchmark_RPP_HIP_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stdDev,
                                    rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *meanTensor, *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        meanTensor[i] = mean;
        stdDevTensor[i] = stdDev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // Use fixed seed for reproducibility
    Rpp32u seed = 12345;

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gaussian_noise(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                &meanTensor[i], &stdDevTensor[i], seed,
                                                &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "GaussianNoise");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "mean=" << mean << ", stdDev=" << stdDev;
    printResult("RPP HIP GaussianNoise", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float prob,
                                         rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *noiseProbabilityTensor, *saltProbabilityTensor, *saltValueTensor, *pepperValueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&noiseProbabilityTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saltProbabilityTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saltValueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&pepperValueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        noiseProbabilityTensor[i] = prob;
        saltProbabilityTensor[i] = 0.5f;  // 50% probability of salt when noise occurs
        saltValueTensor[i] = 1.0f;        // Salt value (normalized to 0-1 range)
        pepperValueTensor[i] = 0.0f;      // Pepper value (normalized to 0-1 range)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // Use fixed seed for reproducibility
    Rpp32u seed = 12345;

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_salt_and_pepper_noise(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                       &noiseProbabilityTensor[i], &saltProbabilityTensor[i],
                                                       &saltValueTensor[i], &pepperValueTensor[i], seed,
                                                       &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "SaltAndPepperNoise");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(noiseProbabilityTensor));
    CHECK_HIP_STATUS(hipHostFree(saltProbabilityTensor));
    CHECK_HIP_STATUS(hipHostFree(saltValueTensor));
    CHECK_HIP_STATUS(hipHostFree(pepperValueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "noiseProbability=" << prob;
    printResult("RPP HIP SaltAndPepperNoise", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseParam,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *shotNoiseFactorTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&shotNoiseFactorTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Shot noise factor (controls Poisson noise intensity)
        shotNoiseFactorTensor[i] = shotNoiseParam;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // Seed for random number generation
    Rpp32u seed = 12345;

    rppSetBatchSize(handle, 1);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_shot_noise(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &shotNoiseFactorTensor[i], seed,
                                            &roiTensor[i], RpptRoiType::XYWH,
                                            handle, RPP_HIP_BACKEND),
                            "ShotNoise");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(shotNoiseFactorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "shotFactor=" << shotNoiseParam;
    printResult("RPP HIP NoiseShot", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Posterize(const vector<Mat>& imgs, bool isColor, int bits,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp8u *posterizeLevelBits;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&posterizeLevelBits, num_images * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        posterizeLevelBits[i] = (Rpp8u)bits;  // Number of bits to preserve (e.g., 4 bits)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_posterize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &posterizeLevelBits[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Posterize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(posterizeLevelBits));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "bits=" << bits;
    printResult("RPP HIP Posterize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Solarize(const vector<Mat>& imgs, bool isColor, int threshold,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *thresholdTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&thresholdTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Threshold in normalized range (0.0-1.0) for U8 images
        thresholdTensor[i] = threshold / 255.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_solarize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &thresholdTensor[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Solarize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(thresholdTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "threshold=" << threshold;
    printResult("RPP HIP Solarize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_ColorCast(const vector<Mat>& imgs, bool isColor, float r, float g, float b,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ColorCast requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ColorCast requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRGB *rgbTensor;
    Rpp32f *alphaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&rgbTensor, num_images * sizeof(RpptRGB)));
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // RGB color cast values (0-255 range)
        rgbTensor[i].R = (Rpp8u)(r * 255.0f);
        rgbTensor[i].G = (Rpp8u)(g * 255.0f);
        rgbTensor[i].B = (Rpp8u)(b * 255.0f);
        alphaTensor[i] = 0.5f;  // Alpha blending factor (0.0-1.0)

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_cast(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &rgbTensor[i], &alphaTensor[i],
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorCast");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(rgbTensor));
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "r=" << r << ", g=" << g << ", b=" << b;
    printResult("RPP HIP ColorCast", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_ColorTemperature(const vector<Mat>& imgs, bool isColor, int adjustment,
                                       rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ColorTemperature requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ColorTemperature requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32s *adjustmentValueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&adjustmentValueTensor, num_images * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Color temperature adjustment (-100 to 100, negative for cooler, positive for warmer)
        adjustmentValueTensor[i] = (Rpp32s)adjustment;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_temperature(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                    &adjustmentValueTensor[i],
                                                    &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorTemperature");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(adjustmentValueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "adjustment=" << adjustment;
    printResult("RPP HIP ColorTemperature", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_ColorTwist(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ColorTwist requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ColorTwist requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *brightnessTensor, *contrastTensor, *hueTensor, *saturationTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&brightnessTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&contrastTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&hueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saturationTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // ColorTwist parameters: applies brightness, contrast, hue, and saturation adjustments
        brightnessTensor[i] = 1.1f;   // Brightness factor (1.0 = no change)
        contrastTensor[i] = 1.2f;     // Contrast factor (1.0 = no change)
        hueTensor[i] = 10.0f;         // Hue shift in degrees (-180 to 180)
        saturationTensor[i] = 1.3f;   // Saturation factor (1.0 = no change)

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_twist(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                             &brightnessTensor[i], &contrastTensor[i],
                                             &hueTensor[i], &saturationTensor[i],
                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorTwist");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(brightnessTensor));
    CHECK_HIP_STATUS(hipHostFree(contrastTensor));
    CHECK_HIP_STATUS(hipHostFree(hueTensor));
    CHECK_HIP_STATUS(hipHostFree(saturationTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "brightness=1.1, contrast=1.2, hue=10, saturation=1.3";
    printResult("RPP HIP ColorTwist", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Vignette(const vector<Mat>& imgs, bool isColor, float strength,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *vignetteIntensityTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&vignetteIntensityTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Vignette intensity (0.0-1.0, where 1.0 is maximum darkening at corners)
        vignetteIntensityTensor[i] = strength;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_vignette(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &vignetteIntensityTensor[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Vignette");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(vignetteIntensityTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "strength=" << strength;
    printResult("RPP HIP Vignette", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_NonLinearBlend(const vector<Mat>& imgs, bool isColor, float strength,
                                     rppHandle_t handle, hipStream_t stream) {
    if (imgs.size() < 2) {
        cout << "NonLinearBlend requires at least 2 images. Skipping." << endl;
        return;
    }

    int num_images = (int)imgs.size() / 2;  // Process pairs of images
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        // Use pairs of input images
        const Mat& img1 = imgs[i * 2];
        const Mat& img2 = imgs[i * 2 + 1];

        RpptLayout layout = (isColor && img1.channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, img1.rows, img1.cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Standard deviation for Gaussian blending (controls blend smoothness)
        stdDevTensor[i] = strength;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = img1.cols;
        roiTensor[i].xywhROI.roiHeight = img1.rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy first image data
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = img1.cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < img1.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img1.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Copy second image data
        for (int row = 0; row < img2.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img2.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_non_linear_blend(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                                   d_outputs[i], &dstDescs[i],
                                                   &stdDevTensor[i],
                                                   &roiTensor[i], RpptRoiType::XYWH,
                                                   handle, RPP_HIP_BACKEND),
                            "NonLinearBlend");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "strength=" << strength << ",pairs=" << num_images;
    printResult("RPP HIP NonLinearBlend", num_images, isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Erase(const vector<Mat>& imgs, bool isColor, int numBoxes,
                            rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp8u *colorsTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;

    int maxBoxesPerImage = numBoxes;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&colorsTensor, num_images * maxBoxesPerImage * numChannels * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        numBoxesTensor[i] = numBoxes;

        // Define anchor boxes for erasing (simple pattern: divide image into regions)
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        for (int box = 0; box < numBoxes; ++box) {
            int idx = i * maxBoxesPerImage + box;
            // Create boxes in different regions
            int boxWidth = width / (numBoxes + 1);
            int boxHeight = height / (numBoxes + 1);
            anchorBoxInfoTensor[idx].lt.x = (box + 1) * boxWidth / 2;
            anchorBoxInfoTensor[idx].lt.y = (box + 1) * boxHeight / 2;
            anchorBoxInfoTensor[idx].rb.x = anchorBoxInfoTensor[idx].lt.x + boxWidth;
            anchorBoxInfoTensor[idx].rb.y = anchorBoxInfoTensor[idx].lt.y + boxHeight;

            // Fill with black color
            for (int c = 0; c < numChannels; ++c) {
                colorsTensor[(idx * numChannels) + c] = 0;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_erase(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       &anchorBoxInfoTensor[i * maxBoxesPerImage],
                                       &colorsTensor[i * maxBoxesPerImage * numChannels],
                                       &numBoxesTensor[i],
                                       &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Erase");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(colorsTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP Erase", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_RandomErase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);
    vector<Rpp8u*> d_noiseBuffers(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    RpptROI *roiTensor;

    Rpp32u maxEraseBoxes = 4;  // Number of random erase regions
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxEraseBoxes * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Define random erase boxes
        int width = imgs[i].cols;
        int height = imgs[i].rows;

        for (Rpp32u box = 0; box < maxEraseBoxes; ++box) {
            int idx = i * maxEraseBoxes + box;
            // Create random-sized erase regions
            int boxWidth = width / (maxEraseBoxes + 2);
            int boxHeight = height / (maxEraseBoxes + 2);
            int xPos = (box * width) / (maxEraseBoxes + 1);
            int yPos = (box * height) / (maxEraseBoxes + 1);

            anchorBoxInfoTensor[idx].lt.x = std::max(0, xPos);
            anchorBoxInfoTensor[idx].lt.y = std::max(0, yPos);
            anchorBoxInfoTensor[idx].rb.x = std::min(width - 1, xPos + boxWidth);
            anchorBoxInfoTensor[idx].rb.y = std::min(height - 1, yPos + boxHeight);
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Allocate noise buffer (filled with random values for erased regions)
        CHECK_HIP_STATUS(hipMalloc(&d_noiseBuffers[i], alignedBufferSize));

        // Fill noise buffer with random pattern (simple: use gradient for demo)
        Rpp8u* h_noiseBuffer = new Rpp8u[alignedBufferSize];
        for (size_t idx = 0; idx < alignedBufferSize; ++idx) {
            h_noiseBuffer[idx] = (idx % 256);  // Simple pattern
        }
        CHECK_HIP_STATUS(hipMemcpy(d_noiseBuffers[i], h_noiseBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_noiseBuffer;

        // Copy image data
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_random_erase(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                               &anchorBoxInfoTensor[i * maxEraseBoxes],
                                               d_noiseBuffers[i],
                                               &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "RandomErase");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
        CHECK_HIP_STATUS(hipFree(d_noiseBuffers[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "boxes=4";
    printResult("RPP HIP RandomErase", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_CoarseDropout(const vector<Mat>& imgs, bool isColor, int dropSize,
                                    rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;

    Rpp32u maxBoxesPerImage = 8;  // Fixed number of coarse dropout boxes
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        numBoxesTensor[i] = maxBoxesPerImage;

        // Define coarse dropout boxes (fixed size boxes in grid pattern)
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        int boxSize = dropSize;

        for (Rpp32u box = 0; box < maxBoxesPerImage; ++box) {
            int idx = i * maxBoxesPerImage + box;
            // Arrange boxes in 2x4 grid
            int row = box / 4;
            int col = box % 4;
            int xPos = (col * width) / 4 + (width / 8);
            int yPos = (row * height) / 2 + (height / 4);

            anchorBoxInfoTensor[idx].lt.x = std::max(0, xPos - boxSize / 2);
            anchorBoxInfoTensor[idx].lt.y = std::max(0, yPos - boxSize / 2);
            anchorBoxInfoTensor[idx].rb.x = std::min(width - 1, xPos + boxSize / 2);
            anchorBoxInfoTensor[idx].rb.y = std::min(height - 1, yPos + boxSize / 2);
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_coarse_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                 &anchorBoxInfoTensor[i * maxBoxesPerImage],
                                                 &numBoxesTensor[i], maxBoxesPerImage,
                                                 &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "CoarseDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "dropSize=" << dropSize;
    printResult("RPP HIP CoarseDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_GridDropout(const vector<Mat>& imgs, bool isColor, int tileWidth, int tileHeight,
                                  rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    RpptROI *roiTensor;

    Rpp32u boxesInEachImage = 16;  // Grid will have multiple dropout boxes
    Rpp32u maxHoleW = tileWidth;
    Rpp32u maxHoleH = tileHeight;

    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * boxesInEachImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Define grid dropout boxes (arranged in a grid pattern)
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        int gridCols = 4;
        int gridRows = 4;

        for (Rpp32u box = 0; box < boxesInEachImage; ++box) {
            int idx = i * boxesInEachImage + box;
            int row = box / gridCols;
            int col = box % gridCols;

            int cellWidth = width / gridCols;
            int cellHeight = height / gridRows;
            int xPos = col * cellWidth + cellWidth / 4;
            int yPos = row * cellHeight + cellHeight / 4;

            anchorBoxInfoTensor[idx].lt.x = std::max(0, xPos);
            anchorBoxInfoTensor[idx].lt.y = std::max(0, yPos);
            anchorBoxInfoTensor[idx].rb.x = std::min(width - 1, xPos + (int)maxHoleW);
            anchorBoxInfoTensor[idx].rb.y = std::min(height - 1, yPos + (int)maxHoleH);
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_grid_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                               &anchorBoxInfoTensor[i * boxesInEachImage],
                                               boxesInEachImage, maxHoleW, maxHoleH,
                                               &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "GridDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", tileHeight=" << tileHeight;
    printResult("RPP HIP GridDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Gridmask(const vector<Mat>& imgs, bool isColor, int tileWidth, float ratio,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROIs in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    // Gridmask parameters (same for all images in batch)
    Rpp32u gridTileWidth = (Rpp32u)tileWidth;
    Rpp32f gridRatio = ratio;  // Ratio of grid hole to tile (0.0-1.0)
    Rpp32f gridAngle = 0.0f;   // Rotation angle in degrees
    RpptUintVector2D translateVector;
    translateVector.x = 0;
    translateVector.y = 0;

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gridmask(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          gridTileWidth, gridRatio, gridAngle, translateVector,
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Gridmask");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", ratio=" << ratio;
    printResult("RPP HIP Gridmask", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                     rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ChannelDropout requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ChannelDropout requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp8u *dropoutTensor;  // 3 channels: 0 or 1 for each channel (0=drop, 1=keep)
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&dropoutTensor, num_images * 3 * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Simple dropout pattern: randomly drop one channel based on probability
        // For demo: drop R channel if prob > 0.66, G if 0.33-0.66, B if < 0.33
        if (dropoutProb > 0.66f) {
            dropoutTensor[i * 3 + 0] = 0;  // Drop R
            dropoutTensor[i * 3 + 1] = 1;  // Keep G
            dropoutTensor[i * 3 + 2] = 1;  // Keep B
        } else if (dropoutProb > 0.33f) {
            dropoutTensor[i * 3 + 0] = 1;  // Keep R
            dropoutTensor[i * 3 + 1] = 0;  // Drop G
            dropoutTensor[i * 3 + 2] = 1;  // Keep B
        } else {
            dropoutTensor[i * 3 + 0] = 1;  // Keep R
            dropoutTensor[i * 3 + 1] = 1;  // Keep G
            dropoutTensor[i * 3 + 2] = 0;  // Drop B
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_channel_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &dropoutTensor[i * 3],
                                                  &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ChannelDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(dropoutTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "dropoutProb=" << dropoutProb;
    printResult("RPP HIP ChannelDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                    rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp8u *colorsTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;

    Rpp32u maxBoxesPerImage = numBoxes;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&colorsTensor, num_images * maxBoxesPerImage * numChannels * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        numBoxesTensor[i] = numBoxes;

        // Define cutout boxes (random regions to drop out)
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        for (Rpp32u box = 0; box < numBoxes; ++box) {
            int idx = i * maxBoxesPerImage + box;
            // Create random-sized boxes in different positions
            int cutoutSize = std::min(width, height) / (numBoxes + 2);
            int xPos = (box * width) / (numBoxes + 1);
            int yPos = (box * height) / (numBoxes + 1);

            anchorBoxInfoTensor[idx].lt.x = xPos;
            anchorBoxInfoTensor[idx].lt.y = yPos;
            anchorBoxInfoTensor[idx].rb.x = std::min(xPos + cutoutSize, width - 1);
            anchorBoxInfoTensor[idx].rb.y = std::min(yPos + cutoutSize, height - 1);

            // Fill with gray color (128)
            for (int c = 0; c < numChannels; ++c) {
                colorsTensor[(idx * numChannels) + c] = 128;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_cutout_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                 &anchorBoxInfoTensor[i * maxBoxesPerImage],
                                                 &colorsTensor[i * maxBoxesPerImage * numChannels],
                                                 &numBoxesTensor[i],
                                                 &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "CutoutDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(colorsTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP CutoutDropout", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // JPEG compression distortion only supports RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: JPEG compression distortion requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32s *qualityTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&qualityTensor, num_images * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;  // JPEG distortion works with NHWC layout
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        qualityTensor[i] = quality;  // JPEG quality factor (1-100)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_jpeg_compression_distortion(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                             &qualityTensor[i],
                                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "JpegCompressionDistortion");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(qualityTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "quality=" << quality;
    printResult("RPP HIP JpegCompressionDistortion", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Copy(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_copy(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                      handle, RPP_HIP_BACKEND),
                            "Copy");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }

    ostringstream params;
    params << "layout=" << (isColor ? "NHWC" : "NCHW");
    printResult("RPP HIP Copy", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_ChannelPermute(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    if (!isColor) {
        cout << "ChannelPermute requires RGB images (3 channels). Skipping grayscale." << endl;
        return;
    }

    int num_images = (int)imgs.size();
    int numChannels = 3;  // RGB only

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate permutation tensor in pinned host memory
    Rpp32u *permutationTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&permutationTensor, num_images * 3 * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Set channel permutation (BGR to RGB: 2, 1, 0)
        permutationTensor[i * 3 + 0] = 2;  // R from B
        permutationTensor[i * 3 + 1] = 1;  // G from G
        permutationTensor[i * 3 + 2] = 0;  // B from R

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_channel_permute(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &permutationTensor[i * 3],
                                                  handle, RPP_HIP_BACKEND),
                            "ChannelPermute");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(permutationTensor));

    ostringstream params;
    params << "permutation=BGR->RGB";
    printResult("RPP HIP ChannelPermute", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Slice(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptGenericDesc> srcDescs(num_images);
    vector<RpptGenericDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate anchor and shape tensors in pinned host memory
    Rpp32s *anchorTensor;
    Rpp32s *shapeTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorTensor, num_images * 4 * sizeof(Rpp32s)));  // 4D tensor [N,H,W,C]
    CHECK_HIP_STATUS(hipHostMalloc(&shapeTensor, num_images * 4 * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

        // Setup generic descriptor (4D tensor)
        srcDescs[i].numDims = 4;
        srcDescs[i].offsetInBytes = 0;
        srcDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].layout = layout;

        // Set dimensions [N, H, W, C] or [N, C, H, W] based on layout
        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = ((width + 7) / 8) * 8;  // Align to 8

        if (layout == RpptLayout::NHWC) {
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = height;  // H
            srcDescs[i].dims[2] = alignedWidth;  // W (aligned)
            srcDescs[i].dims[3] = numChannels;  // C
        } else {  // NCHW
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = numChannels;  // C
            srcDescs[i].dims[2] = height;  // H
            srcDescs[i].dims[3] = alignedWidth;  // W (aligned)
        }

        // Calculate strides
        srcDescs[i].strides[3] = 1;
        srcDescs[i].strides[2] = srcDescs[i].dims[3];
        srcDescs[i].strides[1] = srcDescs[i].strides[2] * srcDescs[i].dims[2];
        srcDescs[i].strides[0] = srcDescs[i].strides[1] * srcDescs[i].dims[1];

        // Slice center region (50% of image)
        int sliceHeight = height / 2;
        int sliceWidth = width / 2;
        int startY = height / 4;
        int startX = width / 4;

        // Set anchor point [N=0, H=startY, W=startX, C=0]
        if (layout == RpptLayout::NHWC) {
            anchorTensor[i * 4 + 0] = 0;  // N
            anchorTensor[i * 4 + 1] = startY;  // H
            anchorTensor[i * 4 + 2] = startX;  // W
            anchorTensor[i * 4 + 3] = 0;  // C

            // Set shape [N=1, H=sliceHeight, W=sliceWidth, C=numChannels]
            shapeTensor[i * 4 + 0] = 1;  // N
            shapeTensor[i * 4 + 1] = sliceHeight;  // H
            shapeTensor[i * 4 + 2] = sliceWidth;  // W
            shapeTensor[i * 4 + 3] = numChannels;  // C

            // Setup destination descriptor
            dstDescs[i] = srcDescs[i];
            dstDescs[i].dims[1] = sliceHeight;
            dstDescs[i].dims[2] = sliceWidth;
        } else {  // NCHW
            anchorTensor[i * 4 + 0] = 0;  // N
            anchorTensor[i * 4 + 1] = 0;  // C
            anchorTensor[i * 4 + 2] = startY;  // H
            anchorTensor[i * 4 + 3] = startX;  // W

            // Set shape [N=1, C=numChannels, H=sliceHeight, W=sliceWidth]
            shapeTensor[i * 4 + 0] = 1;  // N
            shapeTensor[i * 4 + 1] = numChannels;  // C
            shapeTensor[i * 4 + 2] = sliceHeight;  // H
            shapeTensor[i * 4 + 3] = sliceWidth;  // W

            // Setup destination descriptor
            dstDescs[i] = srcDescs[i];
            dstDescs[i].dims[2] = sliceHeight;
            dstDescs[i].dims[3] = sliceWidth;
        }

        roiTensor[i] = 1;  // Single image

        // Calculate buffer sizes
        size_t srcBufferSize = srcDescs[i].strides[0] * srcDescs[i].dims[0] * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].strides[0] * dstDescs[i].dims[0] * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int elementsPerRow = width * numChannels;

        for (int row = 0; row < height; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_slice(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       anchorTensor + i * 4, shapeTensor + i * 4,
                                       nullptr, false, &roiTensor[i],
                                       handle, RPP_HIP_BACKEND),
                            "Slice");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorTensor));
    CHECK_HIP_STATUS(hipHostFree(shapeTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "slice=center_50%";
    printResult("RPP HIP Slice", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Transpose(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptGenericDesc> srcDescs(num_images);
    vector<RpptGenericDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate permutation tensor and ROI tensor in pinned host memory
    Rpp32u *permTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&permTensor, num_images * 4 * sizeof(Rpp32u)));  // 4D permutation
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

        // Setup generic descriptor (4D tensor)
        srcDescs[i].numDims = 4;
        srcDescs[i].offsetInBytes = 0;
        srcDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].layout = layout;

        // Set dimensions [N, H, W, C] or [N, C, H, W] based on layout
        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = ((width + 7) / 8) * 8;  // Align to 8

        if (layout == RpptLayout::NHWC) {
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = height;  // H
            srcDescs[i].dims[2] = width;  // W (use original width, not aligned)
            srcDescs[i].dims[3] = numChannels;  // C

            // Identity permutation (no transpose for now)
            permTensor[i * 4 + 0] = 0;  // N
            permTensor[i * 4 + 1] = 1;  // H
            permTensor[i * 4 + 2] = 2;  // W
            permTensor[i * 4 + 3] = 3;  // C

            // Setup destination descriptor (same as source for identity)
            dstDescs[i] = srcDescs[i];
        } else {  // NCHW
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = numChannels;  // C
            srcDescs[i].dims[2] = height;  // H
            srcDescs[i].dims[3] = width;  // W (use original width, not aligned)

            // Identity permutation (no transpose for now)
            permTensor[i * 4 + 0] = 0;  // N
            permTensor[i * 4 + 1] = 1;  // C
            permTensor[i * 4 + 2] = 2;  // H
            permTensor[i * 4 + 3] = 3;  // W

            // Setup destination descriptor (same as source for identity)
            dstDescs[i] = srcDescs[i];
        }

        // Calculate strides for source (with alignment)
        if (layout == RpptLayout::NHWC) {
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = numChannels;
            srcDescs[i].strides[1] = alignedWidth * numChannels;  // Use aligned width for stride
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * height;
        } else {  // NCHW
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = alignedWidth;  // Use aligned width for stride
            srcDescs[i].strides[1] = srcDescs[i].strides[2] * height;
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * numChannels;
        }

        // Destination strides same as source for identity
        dstDescs[i].strides[0] = srcDescs[i].strides[0];
        dstDescs[i].strides[1] = srcDescs[i].strides[1];
        dstDescs[i].strides[2] = srcDescs[i].strides[2];
        dstDescs[i].strides[3] = srcDescs[i].strides[3];

        roiTensor[i] = 1;  // Single image

        // Calculate buffer sizes
        size_t bufferSize = srcDescs[i].strides[0] * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], bufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], bufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[bufferSize]();
        int elementsPerRow = width * numChannels;

        if (layout == RpptLayout::NHWC) {
            for (int row = 0; row < height; ++row) {
                memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                       imgs[i].data + row * elementsPerRow,
                       elementsPerRow * sizeof(Rpp8u));
            }
        } else {  // NCHW
            for (int c = 0; c < numChannels; ++c) {
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        int srcIdx = row * elementsPerRow + col * numChannels + c;
                        int dstIdx = c * (height * alignedWidth) + row * alignedWidth + col;
                        h_tempBuffer[dstIdx] = imgs[i].data[srcIdx];
                    }
                }
            }
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, bufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_transpose(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                           permTensor + i * 4, &roiTensor[i],
                                           handle, RPP_HIP_BACKEND),
                            "Transpose");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(permTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "permutation=identity";
    printResult("RPP HIP Transpose", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_Normalize_SingleImage(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptGenericDesc> srcDescs(num_images);
    vector<RpptGenericDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate mean, stddev, and ROI tensors in pinned host memory
    Rpp32f *meanTensor;
    Rpp32f *stdDevTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

        // Setup generic descriptor (4D tensor)
        srcDescs[i].numDims = 4;
        srcDescs[i].offsetInBytes = 0;
        srcDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].layout = layout;

        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = ((width + 7) / 8) * 8;

        if (layout == RpptLayout::NHWC) {
            srcDescs[i].dims[0] = 1;
            srcDescs[i].dims[1] = height;
            srcDescs[i].dims[2] = width;
            srcDescs[i].dims[3] = numChannels;
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = numChannels;
            srcDescs[i].strides[1] = alignedWidth * numChannels;
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * height;
        } else {  // NCHW
            srcDescs[i].dims[0] = 1;
            srcDescs[i].dims[1] = numChannels;
            srcDescs[i].dims[2] = height;
            srcDescs[i].dims[3] = width;
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = alignedWidth;
            srcDescs[i].strides[1] = srcDescs[i].strides[2] * height;
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * numChannels;
        }

        dstDescs[i] = srcDescs[i];
        roiTensor[i] = 1;

        // Set normalization parameters (compute mean and stddev from image)
        // Mean: 128, StdDev: 64 (typical values)
        for (int c = 0; c < numChannels; ++c) {
            meanTensor[i * numChannels + c] = 128.0f;
            stdDevTensor[i * numChannels + c] = 64.0f;
        }

        // Allocate device memory
        size_t bufferSize = srcDescs[i].strides[0] * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], bufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], bufferSize));

        // Copy image data
        Rpp8u* h_tempBuffer = new Rpp8u[bufferSize]();
        int elementsPerRow = width * numChannels;

        if (layout == RpptLayout::NHWC) {
            for (int row = 0; row < height; ++row) {
                memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                       imgs[i].data + row * elementsPerRow,
                       elementsPerRow * sizeof(Rpp8u));
            }
        } else {  // NCHW conversion
            for (int c = 0; c < numChannels; ++c) {
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        int srcIdx = row * elementsPerRow + col * numChannels + c;
                        int dstIdx = c * (height * alignedWidth) + row * alignedWidth + col;
                        h_tempBuffer[dstIdx] = imgs[i].data[srcIdx];
                    }
                }
            }
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, bufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            // axisMask = 0x7 (normalize across HWC dimensions)
            // computeMeanStddev = 0 (use provided mean/stddev)
            // scale = 1.0, shift = 0.0
            CHECK_RPP_STATUS(rppt_normalize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                           0x7, meanTensor + i * numChannels,
                                           stdDevTensor + i * numChannels,
                                           0, 1.0f, 0.0f, &roiTensor[i],
                                           handle, RPP_HIP_BACKEND),
                            "Normalize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "mean=128,stddev=64";
    printResult("RPP HIP Normalize_SingleImage", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, float mulVal, float addVal,
                                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // rppt_fused_multiply_add_scalar ONLY supports F32 datatype
    // Use 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    vector<Rpp32f*> d_inputs(num_images);
    vector<Rpp32f*> d_outputs(num_images);

    // Allocate mul, add, and ROI tensors in pinned host memory
    Rpp32f *mulTensor;
    Rpp32f *addTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&mulTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&addTensor, num_images * sizeof(Rpp32f)));

    RpptROI3DPtr roiGenericPtr;
    CHECK_HIP_STATUS(hipHostMalloc(&roiGenericPtr, num_images * sizeof(RpptROI3D)));

    int imageSize = height * width * numChannels;

    if (isColor) {
        // NDHWC layout
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = 1;  // Process one image at a time
        genericDesc.dims[1] = 1;  // Depth = 1 for 2D images
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = numChannels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = numChannels;
        genericDesc.strides[2] = width * numChannels;
        genericDesc.strides[1] = height * width * numChannels;
        genericDesc.strides[0] = height * width * numChannels;
    } else {
        // NCDHW layout for grayscale
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = 1;  // Process one image at a time
        genericDesc.dims[1] = numChannels;
        genericDesc.dims[2] = 1;  // Depth = 1 for 2D images
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = height * width;
        genericDesc.strides[0] = numChannels * height * width;
    }

    for (int i = 0; i < num_images; ++i) {
        // Convert U8 to F32
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);

        // Allocate device memory for F32
        size_t bufferSize = imageSize * sizeof(Rpp32f);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], bufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], bufferSize));

        // Copy F32 data to device
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], imgF32.data, bufferSize, hipMemcpyHostToDevice));

        // Set multiply and add values
        mulTensor[i] = mulVal;
        addTensor[i] = addVal;

        // Set ROI
        roiGenericPtr[i].xyzwhdROI.xyz.x = 0;
        roiGenericPtr[i].xyzwhdROI.xyz.y = 0;
        roiGenericPtr[i].xyzwhdROI.xyz.z = 0;
        roiGenericPtr[i].xyzwhdROI.roiWidth = width;
        roiGenericPtr[i].xyzwhdROI.roiHeight = height;
        roiGenericPtr[i].xyzwhdROI.roiDepth = 1;
    }

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            // Output = (Input * mulTensor) + addTensor
            CHECK_RPP_STATUS(rppt_fused_multiply_add_scalar(d_inputs[i], &genericDesc,
                                                            d_outputs[i], &genericDesc,
                                                            &mulTensor[i], &addTensor[i],
                                                            &roiGenericPtr[i], RpptRoi3DType::XYZWHD,
                                                            handle, RPP_HIP_BACKEND),
                            "FusedMultiplyAddScalar");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(mulTensor));
    CHECK_HIP_STATUS(hipHostFree(addTensor));
    CHECK_HIP_STATUS(hipHostFree(roiGenericPtr));

    ostringstream params;
    params << "mul=" << mulVal << ",add=" << addVal;
    printResult("RPP HIP FusedMultiplyAddScalar", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_CropAndPatch(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    if (imgs.size() < 2) {
        cout << "CropAndPatch requires at least 2 images. Skipping." << endl;
        return;
    }

    int num_images = (int)imgs.size() / 2;  // Process pairs of images
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROI tensors in pinned host memory
    RpptROI *roiTensorDst;
    RpptROI *cropRoiTensor;
    RpptROI *patchRoiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensorDst, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&cropRoiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&patchRoiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        // Image1: source for crop, Image2: destination for patch
        const Mat& img1 = imgs[i * 2];
        const Mat& img2 = imgs[i * 2 + 1];

        RpptLayout layout = (isColor && img1.channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, img1.rows, img1.cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Crop region from image1 (center 50%)
        int cropWidth = img1.cols / 2;
        int cropHeight = img1.rows / 2;
        int cropX = img1.cols / 4;
        int cropY = img1.rows / 4;

        cropRoiTensor[i].xywhROI.xy.x = cropX;
        cropRoiTensor[i].xywhROI.xy.y = cropY;
        cropRoiTensor[i].xywhROI.roiWidth = cropWidth;
        cropRoiTensor[i].xywhROI.roiHeight = cropHeight;

        // Patch location in image2 (top-left corner)
        int patchX = img2.cols / 8;
        int patchY = img2.rows / 8;

        patchRoiTensor[i].xywhROI.xy.x = patchX;
        patchRoiTensor[i].xywhROI.xy.y = patchY;
        patchRoiTensor[i].xywhROI.roiWidth = cropWidth;
        patchRoiTensor[i].xywhROI.roiHeight = cropHeight;

        // Full destination ROI
        roiTensorDst[i].xywhROI.xy.x = 0;
        roiTensorDst[i].xywhROI.xy.y = 0;
        roiTensorDst[i].xywhROI.roiWidth = img2.cols;
        roiTensorDst[i].xywhROI.roiHeight = img2.rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy first image data (source for crop)
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = img1.cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < img1.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img1.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Copy second image data (destination for patch)
        for (int row = 0; row < img2.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img2.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop_and_patch(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                                 d_outputs[i], &dstDescs[i],
                                                 &roiTensorDst[i],
                                                 &cropRoiTensor[i], &patchRoiTensor[i],
                                                 RpptRoiType::XYWH,
                                                 handle, RPP_HIP_BACKEND),
                            "CropAndPatch");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensorDst));
    CHECK_HIP_STATUS(hipHostFree(cropRoiTensor));
    CHECK_HIP_STATUS(hipHostFree(patchRoiTensor));

    ostringstream params;
    params << "crop=50%_center,patch=top-left,pairs=" << num_images;
    printResult("RPP HIP CropAndPatch", num_images, isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *offsetTensor;
    Rpp32f *multiplierTensor;
    Rpp32u *mirrorTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&offsetTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&multiplierTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        // Crop to center 75% of image
        int cropWidth = (imgs[i].cols * 3) / 4;
        int cropHeight = (imgs[i].rows * 3) / 4;
        int cropX = (imgs[i].cols - cropWidth) / 2;
        int cropY = (imgs[i].rows - cropHeight) / 2;

        // Destination descriptor (cropped size)
        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, cropHeight, cropWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        // Set ROI (crop region)
        roiTensor[i].xywhROI.xy.x = cropX;
        roiTensor[i].xywhROI.xy.y = cropY;
        roiTensor[i].xywhROI.roiWidth = cropWidth;
        roiTensor[i].xywhROI.roiHeight = cropHeight;

        // Set normalization parameters (mean=128, stddev=64 => multiplier=1/64, offset=-128/64)
        for (int c = 0; c < numChannels; ++c) {
            offsetTensor[i * numChannels + c] = -2.0f;  // (x - 128) / 64 = x/64 - 2
            multiplierTensor[i * numChannels + c] = 1.0f / 64.0f;
        }

        // Mirror: alternate images for demo
        mirrorTensor[i] = (i % 2);  // Mirror every other image

        // Allocate device memory
        size_t srcBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop_mirror_normalize(d_inputs[i], &srcDescs[i],
                                                        d_outputs[i], &dstDescs[i],
                                                        offsetTensor + i * numChannels,
                                                        multiplierTensor + i * numChannels,
                                                        &mirrorTensor[i],
                                                        &roiTensor[i], RpptRoiType::XYWH,
                                                        handle, RPP_HIP_BACKEND),
                            "CropMirrorNormalize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(offsetTensor));
    CHECK_HIP_STATUS(hipHostFree(multiplierTensor));
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "crop=75%,mirror=alternate,norm=mean128_std64";
    printResult("RPP HIP CropMirrorNormalize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *meanTensor;
    Rpp32f *stdDevTensor;
    Rpp32u *mirrorTensor;
    RpptROI *roiTensor;
    RpptImagePatch *dstImgSizes;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, num_images * sizeof(RpptImagePatch)));

    // Target resize dimensions (224x224 common for neural networks)
    int targetWidth = 224;
    int targetHeight = 224;

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        // Destination descriptor (resized to 224x224)
        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, targetHeight, targetWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        // Set ROI (full image)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Destination image size
        dstImgSizes[i].width = targetWidth;
        dstImgSizes[i].height = targetHeight;

        // Set normalization parameters (ImageNet-style: mean=128, stddev=64)
        for (int c = 0; c < numChannels; ++c) {
            meanTensor[i * numChannels + c] = 128.0f;
            stdDevTensor[i * numChannels + c] = 64.0f;
        }

        // Mirror: alternate images for demo
        mirrorTensor[i] = (i % 2);

        // Allocate device memory
        size_t srcBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    auto start = high_resolution_clock::now();
    for (int k = 0; k < NUM_RUNS; ++k) {
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_resize_mirror_normalize(d_inputs[i], &srcDescs[i],
                                                         d_outputs[i], &dstDescs[i],
                                                         &dstImgSizes[i],
                                                         RpptInterpolationType::BILINEAR,
                                                         meanTensor + i * numChannels,
                                                         stdDevTensor + i * numChannels,
                                                         &mirrorTensor[i],
                                                         &roiTensor[i], RpptRoiType::XYWH,
                                                         handle, RPP_HIP_BACKEND),
                            "ResizeMirrorNormalize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));

    ostringstream params;
    params << "resize=224x224,mirror=alternate,norm=mean128_std64";
    printResult("RPP HIP ResizeMirrorNormalize", imgs.size(), isColor,
                duration<double, milli>(end - start).count(), params.str());
}

void benchmark_RPP_HIP_ResizeCropMirror(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32u *mirrorTensor;
    RpptROI *roiTensor;
    RpptImagePatch *dstImgSizes;
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, num_images * sizeof(RpptImagePatch)));

    // Target resize dimensions (224x224 common for neural networks)
    int targetWidth = 224;
    int targetHeight = 224;

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        // Destination descriptor (resized and cropped to 224x224)
        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, targetHeight, targetWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        // Set ROI (center 80% of image for crop before resize)
        int cropWidth = (imgs[i].cols * 4) / 5;
        int cropHeight = (imgs[i].rows * 4) / 5;
        int cropX = (imgs[i].cols - cropWidth) / 2;
        int cropY = (imgs[i].rows - cropHeight) / 2;

        roiTensor[i].xywhROI.xy.x = cropX;
        roiTensor[i].xywhROI.xy.y = cropY;
        roiTensor[i].xywhROI.roiWidth = cropWidth;
        roiTensor[i].xywhROI.roiHeight = cropHeight;

        // Destination image size (after resize)
        dstImgSizes[i].width = targetWidth;
        dstImgSizes[i].height = targetHeight;

        // Mirror: alternate images for demo
        mirrorTensor[i] = (i % 2);

        // Allocate device memory
        size_t srcBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // NOTE: rppt_resize_crop_mirror has handle corruption bug similar to rppt_flip
    // Run only once to avoid memory corruption, then multiply time by NUM_RUNS
    auto start = high_resolution_clock::now();
    for (int i = 0; i < num_images; ++i) {
        CHECK_RPP_STATUS(rppt_resize_crop_mirror(d_inputs[i], &srcDescs[i],
                                                 d_outputs[i], &dstDescs[i],
                                                 &dstImgSizes[i],
                                                 RpptInterpolationType::BILINEAR,
                                                 &mirrorTensor[i],
                                                 &roiTensor[i], RpptRoiType::XYWH,
                                                 handle, RPP_HIP_BACKEND),
                        "ResizeCropMirror");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    // Multiply by NUM_RUNS to normalize with other benchmarks
    double singleRunTime = duration<double, milli>(end - start).count();
    double adjustedTime = singleRunTime * NUM_RUNS;

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));

    ostringstream params;
    params << "crop=80%,resize=224x224,mirror=alternate";
    printResult("RPP HIP ResizeCropMirror", imgs.size(), isColor, adjustedTime, params.str());
}
