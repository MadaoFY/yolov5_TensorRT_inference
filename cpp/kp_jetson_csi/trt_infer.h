#pragma once
#include <opencv2/core/core.hpp>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <cassert>
#include <vector>



class Logger : public nvinfer1::ILogger
{
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO);
    void log(Severity severity, const char* msg) noexcept override;
};


bool load_engine(nvinfer1::IRuntime*& runtime, nvinfer1::ICudaEngine*& engine, const std::string& engine_dir,
    nvinfer1::ILogger& gLogger);

void allocate_buffers(nvinfer1::ICudaEngine*& engine, std::vector<void*>& bufferH, std::vector<void*>& bufferD, std::vector<int>& bindingsize);

float* det_inference(nvinfer1::IExecutionContext*& context, std::vector<void*>& bufferH, const std::vector<void*>& bufferD,
    const std::vector<int>& BindingSize, cudaStream_t& stream);

float* kp_inference(nvinfer1::IExecutionContext*& context, std::vector<void*>& bufferH, const std::vector<void*>& bufferD,
    const std::vector<int>& BindingSize, cudaStream_t& stream);


class yolo_trt_det
{
private:

    nvinfer1::IRuntime* det_runtime = nullptr;
    nvinfer1::ICudaEngine* det_engine = nullptr;
    nvinfer1::IExecutionContext* det_context = nullptr;

    nvinfer1::IRuntime* kp_runtime = nullptr;
    nvinfer1::ICudaEngine* kp_engine = nullptr;
    nvinfer1::IExecutionContext* kp_context = nullptr;

    std::unordered_map<int, std::string> catid_labels;
    std::vector<std::array<int, 2>> points_linker;
    color_dicts catid_colors;
    cv::Size img_resize;
    cv::Size kp_img_resize;

    bool v8_head;

    std::vector<void*> det_bufferh;
    std::vector<void*> det_bufferd;
    std::vector<int> det_bindingsize;

    std::vector<void*> kp_bufferh;
    std::vector<void*> kp_bufferd;
    std::vector<int> kp_bindingsize;
    cudaStream_t stream;

    int skip;    
    std::vector< int > nms_idx;
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<int> nms_catid;

    uint64_t infer_times;
    uint32_t frams_num;

public:
    yolo_trt_det() = default;
    yolo_trt_det(const std::string & det_engine_dir, const std::string & kp_engine_dir, const std::string & labels_dir,
        const std::string & pointlinker_dir, cv::Size img_size);
    ~yolo_trt_det();

    cv::Mat draw(cv::Mat & image, float conf, float iou, int max_det, int skip);
};
