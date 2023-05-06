#include "utils_detection.h"
#include "gstreamer.h"
#include "trt_infer.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>


using namespace nvinfer1;

void run(const std::string& det_engine_dir, const std::string& kp_engine_dir,  const std::string& labels,
    const std::string& pointlinker_dir, float conf_thres, float iou_thres, int max_det, int skip)
{

    int capture_width = 720;
    int capture_height = 406;
    int display_width = 720;
    int display_height = 406;
    int framerate = 60;
    int flip_method = 0;

    //创建管道
    cv::String pipeline = gs_pipeline(capture_width,
        capture_height,
        display_width,
        display_height,
        framerate,
        flip_method);
    std::cout << "gstreamer: \n\t" << pipeline << "\n";

    //管道与视频流绑定
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cout << "打开摄像头失败." << std::endl;
        return;
    }

    static cv::Size img_size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    yolo_trt_det yolo_detect(det_engine_dir, kp_engine_dir, labels, pointlinker_dir, img_size);
    std::cout << "yolo_engine has been build" << "\n";

    bool keep(true);
    int delay = 30;

    cv::Mat frame;
    auto t1 = std::chrono::steady_clock::now();
    while (keep)
    {
        if (!cap.read(frame))
            break;

        frame = yolo_detect.draw(frame, conf_thres, iou_thres, max_det, skip);
        cv::imshow("CSI Camera", frame);

        if (cv::waitKey(delay) == 27)
            keep = false;
    }

    auto t2 = std::chrono::steady_clock::now();
    auto times = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << times << "s" << "\n";
    cap.release();
    cv::destroyAllWindows();
}


int main(int argc, char* argv[])
{
    // parser 配置

    std::string default_det_engine = "./yolov8n_person.engine";
    std::string default_kp_engine = "./Myhrnet.engine";
    std::string default_label = "./labels_det.yaml";
    std::string default_pointlinker = "./points_link.yaml";
    float default_conf_thres = 0.25f;
    float default_iou_thres = 0.45f;
    int default_max_det = 1;
    int default_skip = 1;

    cv::String keys = cv::format(
        "{help h usage ? |  | print this message   }"
        "{det_engine_dir        |  %s  | det_engine path   }"
        "{kp_engine_dir        |  %s  | kp_engine path   }"
        "{labels        | %s  | obj labels   }"
        "{pointlinker        | %s  | pointlinker   }"
        "{conf_thres        | %f | confidence threshold    }"
        "{iou_thres         | %f | NMS IoU threshold   }"
        "{max_det        | %d | maximum detections per image   }"
        "{skip        | %d | objdetect frame   }",
        default_det_engine.data(), default_kp_engine.data(), default_label.data(),
        default_pointlinker.data(), default_conf_thres, default_iou_thres, default_max_det, default_skip
    );

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    std::string det_engine_dir = parser.get<std::string>("det_engine_dir");
    std::string kp_engine_dir = parser.get<std::string>("kp_engine_dir");
    std::string labels = parser.get<std::string>("labels");
    std::string pointlinker = parser.get<std::string>("pointlinker");
    float conf_thres = parser.get<float>("conf_thres");
    float iou_thres = parser.get<float>("iou_thres");
    int max_det = parser.get<int>("max_det");
    int skip = parser.get<int>("skip");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    
    std::cout << keys << "\n";
    // 运行检测
    run(det_engine_dir, kp_engine_dir, labels, pointlinker, conf_thres, iou_thres, max_det, skip);

    //std::cin.get();

    return 0;
}
