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


void allocate_buffers(ICudaEngine*& engine, std::vector<void*>& bufferH, std::vector<void*>& bufferD, std::vector<int>& bindingsize,
    this->img_size)
{
    assert(engine->getNbBindings() == 2);

    Dims32 dim;
    int size;
    dim = engine->getBindingDimensions(0);
    size = dim.d[0] * dim.d[1] * dim.d[2] * dim.d[3];
    bindingsize[0] = size * sizeof(engine->getBindingDataType(0));

    dim = engine->getBindingDimensions(1);
    size = dim.d[0] * dim.d[1] * dim.d[2];
    bindingsize[1] = size * sizeof(engine->getBindingDataType(1));

    bindingsize[2] = img_size.area() * 3 * sizeof(uint8_t);

    //申请数据预处理用的显存
    cudaMallocHost(&bufferH[2], bindingsize[2]);
    cudaMalloc(&bufferD[2], bindingsize[2]);
    //申请推理输入显存
    cudaMallocHost(&bufferH[0], bindingsize[0]);
    cudaMalloc(&bufferD[0], bindingsize[0]);
    //申请推理输出显存
    cudaMallocHost(&bufferH[1], bindingsize[1]);
    cudaMalloc(&bufferD[1], bindingsize[1]);
}


float* do_inference(IExecutionContext*& context, std::vector<void*>& bufferH, const std::vector<void*>& bufferD,
    cudaStream_t& stream, const std::vector<int>& BindingSize)
{
    //cudaMemcpyAsync(bufferD[0], bufferH[0], BindingSize[0], cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bufferD.data(), stream, nullptr);
    cudaMemcpyAsync(bufferH[1], bufferD[1], BindingSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return (float*)bufferH[1];
}


yolo_trt_det::yolo_trt_det(const std::string& engine_dir, const std::string& labels_dir, cv::Size img_size)
    : img_size(img_size)
{   
     // 载入类别标签, 生成颜色字典
    this->catid_labels = yaml_load_labels(labels_dir);
    this->catid_colors = color_dicts(this->catid_labels);
     // 生成engine, context
    Logger gLogger(ILogger::Severity::kERROR);
    load_engine(_runtime, _engine, engine_dir, gLogger);

    _context = _engine->createExecutionContext();
     // 获取输入的shape
    Dims input_shape = _engine->getBindingDimensions(0);
    this->set_size = cv::Size(input_shape.d[3], input_shape.d[2]);
     // 判断是否为v8检测头
    Dims output_shape = _engine->getBindingDimensions(1);
    if (output_shape.d[2] - this->catid_labels.size() == 4)
    {
        this->v8_head = true;
    }
    else
    {
        this->v8_head = false;
    }
     //建立缓存容器
    int nBinding = _engine->getNbBindings() + 1;
    this->cpu_buffer = std::vector<void*>(nBinding, nullptr);
    this->gpu_buffer = std::vector<void*>(nBinding, nullptr);
    this->BindingSize = std::vector<int>(nBinding, 0);

    allocate_buffers(_engine, this->cpu_buffer, this->gpu_buffer, this->BindingSize, this->img_size);

    cudaStreamCreate(&this->stream);
}


//std::vector<cv::Mat> yolo_trt_det::draw_batch(std::vector<cv::Mat>& image_list, 
//    float conf, float iou, int max_det)
//{
//    // 数据预处理, resize,  bgrbgr2rrggbb
//    int n = image_list.size();
//    std::vector<preproc_struct> image_trans(n);
//    image_trans.reserve(n);
//    int per_inpBindingSize = this->BindingSize[0] / n;
//    for (int i = 0; i < n; ++i)
//    {
//        preprocess(image_list[i], image_trans[i], this->set_size);
//        uchar* per_inp_buffer = (uchar*)this->cpu_buffer[0] + per_inpBindingSize * i;
//        memcpy((void*)per_inp_buffer, (void*)image_trans[i].img, per_inpBindingSize);
//    }
//
//    // 推理
//    auto t1 = std::chrono::steady_clock::now();
//    float* preds = do_inference(_context, this->cpu_buffer, this->gpu_buffer, this->stream, this->BindingSize);
//
//    int per_outBindingSize = this->BindingSize[1] / n;
//    for (int k = 0; k < n; ++k)
//    {
//        uchar* per_preds = (uchar*)this->cpu_buffer[1] + per_outBindingSize * k;
//        // 过滤置信度低于阈值的目标框
//        std::vector<cv::Rect> keep_boxes;
//        std::vector<float> keep_scores;
//        std::vector<int> keep_catid;
//        Dims32 output_size = _engine->getBindingDimensions(1);
//        std::array<int, 4> output_shape({ output_size.d[0], output_size.d[1], output_size.d[2], output_size.d[3] });
//        fliter_boxes(preds, this->v8_head, output_shape, conf_thres, keep_boxes, keep_scores, keep_catid);
//
//        // nms
//        std::vector< int > nms_idx;
//        cv::dnn::NMSBoxesBatched(keep_boxes, keep_scores, keep_catid, conf_thres, iou, nms_idx, (1.0f), 0);
//
//        auto t2 = std::chrono::steady_clock::now();
//        uint32_t time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//        uint32_t fps = (1000000 / time);
//        cv::String fps_text = cv::format("fps:%d", fps);
//
//        int top_k =  cv::min((int)nms_idx.size(), max_det);
//        std::vector<cv::Rect> nms_boxes;
//        std::vector<float> nms_scores;
//        std::vector<int> nms_catid;
//        nms_boxes.reserve(top_k);
//        nms_scores.reserve(top_k);
//        nms_catid.reserve(top_k);
//
//        for (int i = 0; i < top_k; ++i)
//        {
//            int idx = nms_idx[i];
//            scale_boxes(keep_boxes[idx], image_trans[k]);
//            nms_boxes.emplace_back(keep_boxes[idx]);
//            nms_scores.emplace_back(keep_scores[idx]);
//            nms_catid.emplace_back(keep_catid[idx]);
//        }
//
//        // 绘图
//        for (int i = 0; i < nms_catid.size(); ++i)
//        {
//            draw_boxes(image_list[k], nms_boxes[i], nms_scores[i], nms_catid[i], this->catid_labels, this->catid_colors);
//        }
//        cv::putText(image_list[k], fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
//    }
//
//    return image_list;
//}


cv::Mat yolo_trt_det::draw(cv::Mat& image, float conf, float iou, int max_det)
{
    // 数据预处理, resize,  bgrbgr2rrggbb
    preproc_struct image_trans;
    //preprocess(image, image_trans, this->set_size);
    //memcpy(this->cpu_buffer[0], (void*)image_trans.img, this->BindingSize[0]);
    cuda_preprocess(image, image_trans, this->cpu_buffer, this->gpu_buffer, this->BindingSize, this->stream, this->set_size);

    // 推理
    auto t1 = std::chrono::steady_clock::now();
    float* preds = do_inference(_context, this->cpu_buffer, this->gpu_buffer, this->stream, this->BindingSize);

    // 过滤置信度低于阈值的目标框
    std::vector<cv::Rect> keep_boxes;
    std::vector<float> keep_scores;
    std::vector<int> keep_catid;
    Dims32 output_size = _engine->getBindingDimensions(1);
    std::array<int, 4> output_shape({ output_size.d[0], output_size.d[1], output_size.d[2], output_size.d[3] });
    fliter_boxes(preds, this->v8_head, output_shape, conf_thres, keep_boxes, keep_scores, keep_catid);

    // nms
    std::vector< int > nms_idx;
    cv::dnn::NMSBoxesBatched(keep_boxes, keep_scores, keep_catid, conf_thres, iou, nms_idx, (1.0f), 0);

    auto t2 = std::chrono::steady_clock::now();
    uint32_t time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    uint32_t fps = (1000000 / time);
    cv::String fps_text = cv::format("fps:%d", fps);

    int top_k = cv::min((int)nms_idx.size(), max_det);
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<int> nms_catid;
    nms_boxes.reserve(top_k);
    nms_scores.reserve(top_k);
    nms_catid.reserve(top_k);

    for (int i = 0; i < top_k; ++i)
    {
        int idx = nms_idx[i];
        scale_boxes(keep_boxes[idx], image_trans);
        nms_boxes.emplace_back(keep_boxes[idx]);
        nms_scores.emplace_back(keep_scores[idx]);
        nms_catid.emplace_back(keep_catid[idx]);
    }

    // 绘图
    for (int i = 0; i < nms_catid.size(); ++i)
    {
        draw_boxes(image, nms_boxes[i], nms_scores[i], nms_catid[i], this->catid_labels, this->catid_colors);
    }
    cv::putText(image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);

    return image;
}


yolo_trt_det::~yolo_trt_det()
{
    for (int i = 0; i < this->cpu_buffer.size(); ++i)
    {
        cudaFreeHost(this->cpu_buffer[i]);
        cudaFree(this->gpu_buffer[i]);
    }
    cudaStreamDestroy(this->stream);

    _context->destroy();
    _engine->destroy();
    _runtime->destroy();
}