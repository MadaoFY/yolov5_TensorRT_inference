#include "utils_detection.h"
#include "gstreamer.h"
#include "trt_infer.h"

#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace nvinfer1;

void run(const std::string& engine_dir,  const std::string& labels,  float conf_thres, float iou_thres, int max_det)
{

    int capture_width = 720;
    int capture_height = 406;
    int display_width = 720;
    int display_height = 406;
    int framerate = 60;
    int flip_method = 0;

    //�����ܵ�
    cv::String pipeline = gs_pipeline(capture_width,
        capture_height,
        display_width,
        display_height,
        framerate,
        flip_method);
    std::cout << "gstreamer: \n\t" << pipeline << "\n";

    //�ܵ�����Ƶ����
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cout << "������ͷʧ��." << std::endl;
    }

    yolo_trt_det yolo_detect(engine_dir, labels);
    std::cout << "yolo_engine has been build" << "\n";

    bool keep(true);
    int delay = 30;

    cv::Mat frame;
    /*auto t1 = std::chrono::steady_clock::now();*/
    while (keep)
    {
        if (!cap.read(frame))
            break;

        frame = yolo_detect.draw(frame, conf_thres, iou_thres, max_det);
        cv::imshow("CSI Camera", frame);

        if (cv::waitKey(delay) == 27)
            keep = false;
    }

    //auto t2 = std::chrono::steady_clock::now();
    //auto times = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    //std::cout << times << "s" << "\n";
    cap.release();
    cv::destroyAllWindows();
}


int main(int argc, char* argv[])
{
    // parser ����

    std::string default_engine = "./yolov5s.engine";
    std::string default_label = "./labels_coco.yaml";
    float default_conf_thres = 0.25f;
    float default_iou_thres = 0.45f;
    int default_max_det = 200;

    cv::String keys = cv::format(
        "{help h usage ? |  | print this message   }"
        "{engine_dir        |  %s  | engine path   }"
        "{labels        | %s  | obj labels   }"
        "{conf_thres        | %f | confidence threshold    }"
        "{iou_thres         | %f | NMS IoU threshold   }"
        "{max_det        | %d | maximum detections per image   }",
        default_engine.data(),  default_label.data(), default_conf_thres, default_iou_thres, default_max_det
    );

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    std::string engine_dir = parser.get<std::string>("engine_dir");
    std::string labels = parser.get<std::string>("labels");
    float conf_thres = parser.get<float>("conf_thres");
    float iou_thres = parser.get<float>("iou_thres");
    int max_det = parser.get<int>("max_det");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    
    std::cout << keys << "\n";
    // ���м��
    run(engine_dir, labels, conf_thres, iou_thres, max_det);

    //std::cin.get();

    return 0;
}
