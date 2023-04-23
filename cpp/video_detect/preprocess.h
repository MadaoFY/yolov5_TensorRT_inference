#pragma once

#include "utils_detection.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct AffineMatrix
{
    float value[6];
};

void cuda_preprocess(cv::Mat& image, preproc_struct& image_trans, std::vector<void*>& bufferH,
    std::vector<void*>& bufferD, std::vector<int>& bindingsize, cudaStream_t& stream, cv::Size resize);