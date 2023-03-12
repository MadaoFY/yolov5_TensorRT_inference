#include "utils_detection.h"
#include "trt_infer.h"

#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace nvinfer1;

void run(const std::string& engine_dir, const std::string& video_dir, const std::string& labels, 
    float conf_thres, float iou_thres, int max_det)
{
    cv::VideoCapture vc(video_dir);
    if (!vc.isOpened())
    {
        std::cout << "Failed to read video!" << "\n";
        return;
    }
        
    yolo_trt_det yolo_detect(engine_dir, labels);
    std::cout << "yolo_engine has been build" << "\n";
    
    bool keep(true);
    int delay = 30;

    cv::Mat frame;
    /*auto t1 = std::chrono::steady_clock::now();*/
    while (keep)
    {
        if (!vc.read(frame))
            break;

         frame = yolo_detect.draw(frame, conf_thres, iou_thres, max_det);
         cv::imshow("video", frame);

        if (cv::waitKey(delay) == 27)
            keep = false;
    }

    /*auto t2 = std::chrono::steady_clock::now();
    auto times = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << times << "s" << "\n";*/
    vc.release();
    cv::destroyAllWindows();
}


int main(int argc, char* argv[])
{
    // parser ≈‰÷√
    std::string default_engine = "./yolov5s.engine";
    std::string default_video = "./sample_1080p_h265.mp4";
    std::string default_label = "./labels_coco.yaml";
    float default_conf_thres = 0.25f;
    float default_iou_thres = 0.45f;
    int default_max_det = 200;

    cv::String keys = cv::format(
        "{help h usage ? |  | print this message   }"
        "{engine_dir        |  %s  | engine path   }"
        "{video_dir        |  %s | video path   }"
        "{labels        | %s  | obj labels   }"
        "{conf_thres        | %f | confidence threshold    }"
        "{iou_thres         | %f | NMS IoU threshold   }"
        "{max_det        | %d | maximum detections per image   }",
        default_engine.data(), default_video.data(), default_label.data(), default_conf_thres, default_iou_thres, default_max_det
    );

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Application name v1.0.0");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	std::string engine_dir = parser.get<std::string>("engine_dir");
	std::string video_dir = parser.get<std::string>("video_dir");
	std::string labels = parser.get<std::string>("labels");
	float conf_thres = parser.get<float>("conf_thres");
	float iou_thres = parser.get<float> ("iou_thres");
	int max_det = parser.get<int>("max_det");
	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}

    // ‘À––ºÏ≤‚
    run(engine_dir, video_dir, labels, conf_thres, iou_thres, max_det);

    //std::cin.get();

    return 0;
}