#pragma once
#include <opencv2/core/core.hpp>

#include <unordered_map>
#include <vector>
#include <string>
#include <array>



struct color_dicts
{
    std::unordered_map<int, std::array<size_t, 3>> color_map;
    std::vector<int> catid;

    color_dicts() {};
    color_dicts(const std::unordered_map<int, std::string>& catid_labels);
};


struct preproc_struct
{
    float* img = nullptr;
    float scale;
    int h_p;
    int w_p;

    ~preproc_struct();
};



std::unordered_map<int, std::string> yaml_load_labels(const std::string& dir = "data.yaml");

void preprocess(cv::Mat& image, preproc_struct& image_trans, const cv::Size& resize);

void fliter_boxes(float* const boxes, bool v8_head, const std::array<int, 4>& output_shape, const float& conf_thres,
    std::vector<cv::Rect>& keep_boxes, std::vector<float>& keep_scores, std::vector<int>& keep_classes);

void scale_boxes(cv::Rect& box, const preproc_struct& preproc_res);

void draw_boxes(cv::Mat image, const cv::Rect& box, const float& score, const int& class_id,
    std::unordered_map<int, std::string> catid_labels, color_dicts& color_dicts);

void imgresize(const cv::Mat& image, cv::Mat& input_image, const float& scale, cv::Size resize);

template <typename T = int>
static bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2);

template <typename T>
void max_score_idx(const std::vector<float>& scores, const float& score_thres, T& scores_idxs);

float get_iou(const cv::Rect& bbox1, const cv::Rect& bbox2);

void base_nms(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores, const std::vector<int>& catid,
    const float& score_threshold, const float& nms_threshold, std::vector<int>& indices, const int& limit);
