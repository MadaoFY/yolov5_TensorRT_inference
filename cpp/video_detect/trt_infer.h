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

void allocate_buffers(nvinfer1::ICudaEngine*& engine, 
    std::vector<void*>& bufferH, std::vector<void*>& bufferD, std::vector<int>& bindingsize);

float* do_inference(nvinfer1::IExecutionContext*& context, std::vector<void*>& bufferH, const std::vector<void*>& bufferD,
    cudaStream_t& stream, const std::vector<int>& BindingSize);


class yolo_trt_det
{
private:

    nvinfer1::IRuntime* _runtime = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;

    std::unordered_map<int, std::string> catid_labels;
    color_dicts catid_colors;
    cv::Size set_size;
    bool v8_head;

    std::vector<void*> cpu_buffer;
    std::vector<void*> gpu_buffer;
    std::vector<int> BindingSize;
    cudaStream_t stream;

    /*uint64_t infer_times;
    uint32_t frams_num;*/

public:
    yolo_trt_det(const std::string& engine_dir, const std::string& labels_dir);
    virtual ~yolo_trt_det();

    std::vector<cv::Mat> draw_batch(std::vector<cv::Mat>& image_list, float conf, float iou, int max_det);

    cv::Mat draw(cv::Mat& image, float conf, float iou, int max_det);
};