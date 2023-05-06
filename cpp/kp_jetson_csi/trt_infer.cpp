#include "utils_detection.h"
#include "preprocess.h"
#include "trt_infer.h"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

using namespace nvinfer1;

Logger::Logger(Severity severity):
    reportableSeverity(severity) {}

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity > reportableSeverity)
    {
        return;
    }
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
    case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
    case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
    case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
    default:
        std::cerr << "VERBOSE: ";
        break;
    }
    std::cerr << msg << std::endl;
}


bool load_engine(IRuntime*& runtime, ICudaEngine*& engine, const std::string& engine_dir,  ILogger& gLogger)
{
    std::ifstream engine_file(engine_dir, std::ios::binary);
    long int fsize = 0;
    engine_file.seekg(0, engine_file.end);
    fsize = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    std::vector<char> engine_string(fsize);
    engine_file.read(engine_string.data(), fsize);
    engine_file.close();
    if (engine_string.size() == 0)
    {
        std::cout << "Failed getting serialized engine!" << std::endl;
        return false;
    }
    std::cout << "Succeeded getting serialized engine!" << std::endl;

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engine_string.data(), fsize);

    if (engine == nullptr)
    {
        std::cout << "Failed loading engine!" << std::endl;
        return false;
    }
    std::cout << "Succeeded loading engine!" << std::endl;

    return true;
}


void allocate_buffers(ICudaEngine*& engine, std::vector<void*>& bufferH, std::vector<void*>& bufferD,
    std::vector<int>& bindingsize)
{
    assert(engine->getNbBindings() == 2);

    Dims32 dim;
    int size;
    for (int i = 0; i < 2; ++i)
    {
        size = 1;
        dim = engine->getBindingDimensions(i);
        for (int j = 0; j < dim.MAX_DIMS; ++j)
        {
            if (dim.d[j] != 0)
            {
                size *= dim.d[j];
            }
        }
        bindingsize[i] = size * sizeof(engine->getBindingDataType(i));
        cudaMallocHost(&bufferH[i], bindingsize[i]);
        cudaMalloc(&bufferD[i], bindingsize[i]);
    }
}


float* det_inference(IExecutionContext*& context, std::vector<void*>& bufferH, const std::vector<void*>& bufferD,
    const std::vector<int>& BindingSize, cudaStream_t& stream)
{
    //cudaMemcpyAsync(bufferD[0], bufferH[0], BindingSize[0], cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bufferD.data(), stream, nullptr);
    cudaMemcpyAsync(bufferH[1], bufferD[1], BindingSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return (float*)bufferH[1];
}


float* kp_inference(IExecutionContext*& context, std::vector<void*>& bufferH, const std::vector<void*>& bufferD,
    const std::vector<int>& BindingSize, cudaStream_t& stream)
{
    cudaMemcpyAsync(bufferD[0], bufferH[0], BindingSize[0], cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bufferD.data(), stream, nullptr);
    cudaMemcpyAsync(bufferH[1], bufferD[1], BindingSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return (float*)bufferH[1];
}


yolo_trt_det::yolo_trt_det(const std::string& det_engine_dir, const std::string& kp_engine_dir, const std::string& labels_dir,
    const std::string& pointlinker_dir, cv::Size img_size)
    : skip(0), infer_times(0), frams_num(0)
{   
    // 载入类别标签, 生成颜色字典
    this->catid_labels = yaml_load_labels(labels_dir);
    this->catid_colors = color_dicts(this->catid_labels);
    this->points_linker = yaml_load_points_link(pointlinker_dir);
    // 生成engine, context
    Logger gLogger(ILogger::Severity::kERROR);
    load_engine(det_runtime, det_engine, det_engine_dir, gLogger);
    load_engine(kp_runtime, kp_engine, kp_engine_dir, gLogger);

    det_context = det_engine->createExecutionContext();
    kp_context = kp_engine->createExecutionContext();
    // 获取输入的sgaoe
    Dims det_input_size = det_engine->getBindingDimensions(0);
    this->img_resize = cv::Size(det_input_size.d[3], det_input_size.d[2]);
    Dims kp_input_size = kp_engine->getBindingDimensions(0);
    this->kp_img_resize = cv::Size(kp_input_size.d[3], kp_input_size.d[2]);
    // 判断是否为v8检测头
    Dims output_shape = det_engine->getBindingDimensions(1);
    if (output_shape.d[2] - this->catid_labels.size() == 4)
    {
        this->v8_head = true;
    }
    else
    {
        this->v8_head = false;
    }
    // 建立缓存容器
    int nBinding = det_engine->getNbBindings() + 1;
    this->det_bufferh = std::vector<void*>(nBinding, nullptr);
    this->det_bufferd = std::vector<void*>(nBinding, nullptr);
    this->det_bindingsize = std::vector<int>(nBinding, 0);

    nBinding = kp_engine->getNbBindings();
    this->kp_bufferh = std::vector<void*>(nBinding, nullptr);
    this->kp_bufferd = std::vector<void*>(nBinding, nullptr);
    this->kp_bindingsize = std::vector<int>(nBinding, 0);

    allocate_buffers(det_engine, this->det_bufferh, this->det_bufferd, this->det_bindingsize);
    this->det_bindingsize[2] = img_size.area() * 3 * sizeof(uint8_t);
    //申请数据预处理用的显存
    cudaMallocHost(&det_bufferh[2], this->det_bindingsize[2]);
    cudaMalloc(&det_bufferd[2], this->det_bindingsize[2]);

    allocate_buffers(kp_engine, this->kp_bufferh, this->kp_bufferd, this->kp_bindingsize);

    cudaStreamCreate(&this->stream);
}


cv::Mat yolo_trt_det::draw(cv::Mat& image, float conf_thres, float iou, int max_det, int skip)
{
    // 数据预处理, resize,  bgrbgr2rrggbb
    preproc_struct image_trans;
    auto t1 = std::chrono::steady_clock::now();
    if (this->skip == 0)
    {
        this->nms_idx.clear();
        this->nms_boxes.clear();
        this->nms_scores.clear();
        this->nms_catid.clear();

        cuda_preprocess(image, image_trans, this->det_bufferh, this->det_bufferd, this->det_bindingsize, this->stream, this->img_resize);

        // 推理
        float* preds = det_inference(det_context, this->det_bufferh, this->det_bufferd, this->det_bindingsize, this->stream);
        
        // 过滤置信度低于阈值的目标框
        std::vector<cv::Rect> keep_boxes;
        std::vector<float> keep_scores;
        std::vector<int> keep_catid;
        Dims32 output_size = det_engine->getBindingDimensions(1);
        std::array<int, 4> output_shape = { output_size.d[0], output_size.d[1], output_size.d[2], output_size.d[3] };
        fliter_boxes(preds, this->v8_head, output_shape, conf_thres, keep_boxes, keep_scores, keep_catid);

        // nms
        base_nms(keep_boxes, keep_scores, keep_catid, conf_thres, iou, this->nms_idx, max_det);

        this->nms_boxes.reserve(nms_idx.size());
        this->nms_scores.reserve(nms_idx.size());
        this->nms_catid.reserve(nms_idx.size());

        for (int i = 0; i < this->nms_idx.size(); ++i)
        {
            int idx = this->nms_idx[i];
            scale_boxes(keep_boxes[idx], image_trans);
            this->nms_boxes.emplace_back(keep_boxes[idx]);
            this->nms_scores.emplace_back(keep_scores[idx]);
            this->nms_catid.emplace_back(keep_catid[idx]);
        }
        this->skip = skip;
    }
    else
    {
        this->skip -= 1;
    }

    // 关键点推理
    std::vector<float> keypoints_score;
    std::vector<cv::Point2f> keypoints;
    keypoints_score.reserve(17);
    keypoints.reserve(17);
    Dims32 kp_output_size = kp_engine->getBindingDimensions(1);
    std::array<int, 4> kp_output_shape = { kp_output_size.d[0], kp_output_size.d[1], kp_output_size.d[2], kp_output_size.d[3] };
    for (int i = 0; i < this->nms_idx.size(); ++i)
    {
        keypoints_score.clear();
        keypoints.clear();
        cv::Mat kp_img = image(nms_boxes[i]);
        preproc_struct kp_image_trans;
        preprocess(kp_img, kp_image_trans, this->kp_img_resize);
        memcpy(this->kp_bufferh[0], (void*)kp_image_trans.img, this->kp_bindingsize[0]);
        float* kp_preds = kp_inference(kp_context, this->kp_bufferh, this->kp_bufferd, this->kp_bindingsize, this->stream);
        get_final_preds(kp_preds, kp_image_trans, kp_output_shape, this->nms_boxes[i], keypoints_score, keypoints);
        draw_keypoints(image, keypoints, keypoints_score, 0.2f, this->points_linker);
    }
    
    auto t2 = std::chrono::steady_clock::now();
    uint32_t time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    uint32_t fps = (1000000 / time);
    cv::String fps_text = cv::format("fps:%d", fps);

    this->infer_times += time;
    this->frams_num += 1;

    // 绘图
    for (int i = 0; i < this->nms_idx.size(); ++i)
    {
        draw_boxes(image, this->nms_boxes[i], this->nms_scores[i], this->nms_catid[i], this->catid_labels, this->catid_colors);
    }
    cv::putText(image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);

    return image;
}


yolo_trt_det::~yolo_trt_det()
{
    for (int i = 0; i < this->det_bufferd.size(); ++i)
    {
        cudaFreeHost(this->det_bufferh[i]);
        cudaFree(this->det_bufferd[i]);
    }
    for (int i = 0; i < this->kp_bufferd.size(); ++i)
    {
        cudaFreeHost(this->kp_bufferh[i]);
        cudaFree(this->kp_bufferd[i]);
    }
    cudaStreamDestroy(this->stream);

    det_context->destroy();
    det_engine->destroy();
    det_runtime->destroy();

    kp_context->destroy();
    kp_engine->destroy();
    kp_runtime->destroy();

    float infer_time_mean = (this->infer_times / this->frams_num) / 1000.f;
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2);
    std::cout << "infer_time_mean:" << infer_time_mean << "ms" << "\n";

}
