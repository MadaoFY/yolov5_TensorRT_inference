#include "utils_detection.h"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <fstream>


preproc_struct::~preproc_struct()
{
    if (img != NULL)
    {
        delete[] img;
    }
};


color_dicts::color_dicts(const std::unordered_map<int, std::string>& catid_labels)
{
    std::array<std::array<size_t, 3>, 20> base_hexs{ { {255, 56, 56}, {255, 157, 151}, {255, 112, 31}, {255, 178, 29},
    {207, 210, 49}, {72, 249, 10}, {146, 204, 23}, {61, 219, 134}, {26, 147, 52}, {0, 212, 187},
    {44, 153, 168}, {0, 194, 255}, {52, 69, 147}, {100, 115, 255}, {0, 24, 236}, {132, 56, 255},
    {82, 0, 133}, {203, 56, 255}, {255, 149, 200}, {255, 55, 199} } };

    int n = catid_labels.size() / base_hexs.size() + 1;

    std::vector<std::array<size_t, 3>> total_hexs;
    total_hexs.reserve(catid_labels.size());
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < base_hexs.size(); ++j)
        {
            total_hexs.emplace_back(base_hexs[j]);
            if (total_hexs.size() > catid_labels.size())
            {
                break;
            }
        }
    }
    int idx = 0;
    catid.reserve(catid_labels.size());
    for (const auto& m : catid_labels)
    {
        catid.emplace_back(m.first);
        color_map[m.first] = total_hexs[idx];
        ++idx;
    }
};


std::unordered_map<int, std::string> yaml_load_labels(const std::string& dir)
{
    std::unordered_map<int, std::string> catid_labels;
    std::ifstream yaml_file(dir);
    if (yaml_file.is_open())
    {
        std::string str, str1, str2;
        while (std::getline(yaml_file, str))
        {
            std::string::size_type pos = str.find(":");
            str1 = str.substr(0, pos + 1); // ：前的部分
            str2 = str.substr(pos + 1, str.size() - pos - 2);  // ：后的部分
            catid_labels[std::stoi(str1)] = str2;
        }
        yaml_file.close();
    }

    if (catid_labels.size() == 0)
    {
        std::cout << "Failed loading labels!" << std::endl;
    }
    return catid_labels;
}


std::vector<std::array<int, 2>> yaml_load_points_link(const std::string& dir)
{
    std::vector<std::array<int, 2>> points_linker;
    std::ifstream yaml_file(dir);
    if (yaml_file.is_open())
    {
        std::string str, str1, str2;
        while (std::getline(yaml_file, str))
        {
            std::string::size_type pos = str.find(":");
            str1 = str.substr(0, pos + 1); // ：前的部分
            str2 = str.substr(pos + 1);  // ：后的部分
            points_linker.emplace_back(std::array<int, 2>{ std::stoi(str1), std::stoi(str2)});
        }
        yaml_file.close();
    }

    if (points_linker.size() == 0)
    {
        std::cout << "Failed loading points linker!" << std::endl;
    }
    return points_linker;
}


void imgresize(const cv::Mat& image, cv::Mat& input_image, const float& scale, cv::Size resize)
{
    float i2d[6], d2i[6];
    // resize图像，源图像和目标图像几何中心的对齐
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-scale * image.cols + resize.width + scale - 1) * 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-scale * image.rows + resize.height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);           // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);           // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i); // 计算一个反仿射变换
    cv::warpAffine(image, input_image, m2x3_i2d, resize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

void preprocess(cv::Mat& image, preproc_struct& image_trans, const cv::Size& resize)
{
    int h, w, h_p, w_p;

    float scale = cv::min((float)resize.height / (float)image.rows, (float)resize.width / (float)image.cols);
    scale = cv::min(scale, 1.1f);

    h = image.rows * scale;
    w = image.cols * scale;
    h_p = (resize.height - h) * 0.5;
    w_p = (resize.width - w) * 0.5;

    image_trans.scale = scale;
    image_trans.ori_h = image.rows;
    image_trans.ori_w = image.cols;
    image_trans.h_p = h_p;
    image_trans.w_p = w_p;

    cv::Mat image_resize(cv::Size(w, h), CV_8UC3);

    cv::resize(image, image_resize, cv::Size(w, h), 0, 0, 0);
    cv::copyMakeBorder(image_resize, image_resize,
        h_p, (resize.height - h - h_p), w_p, (resize.width - w - w_p), cv::BORDER_CONSTANT, cv::Scalar::all(0));

    int size = image_resize.total();
    float* img_trans = new float[size * 3];

    int pos = 0;
    if (image_resize.isContinuous())
    {
        uchar* p_trans = image_resize.data;
        for (int i = 0; i < size; ++i)
        {
            img_trans[i] = (float)(p_trans[pos + 2]) / 255.0f;
            img_trans[i + size] = (float)(p_trans[pos + 1]) / 255.0f;
            img_trans[i + 2 * size] =  (float)(p_trans[pos]) / 255.0f;
            pos += 3;
        }
    }
    else
    {
        for (int i = 0; i < image_resize.rows; ++i)
        {
            cv::Vec3b*  p_trans = image_resize.ptr<cv::Vec3b>(i);
            for (int j = 0; j < image_resize.cols; ++j)
            {
                img_trans[pos] = (float)(p_trans[j][2]) / 255.0f;
                img_trans[pos + size] = (float)(p_trans[j][1]) / 255.0f;
                img_trans[pos + 2 * size] = (float)(p_trans[j][0]) / 255.0f;
                ++pos;
            }
        }
    }
    image_trans.img = std::move(img_trans);
}


void fliter_boxes(float* const boxes, bool v8_head, const std::array<int, 4>& output_shape, const float& conf_thres,
    std::vector<cv::Rect>& keep_boxes, std::vector<float>& keep_scores, std::vector<int>& keep_classes)
{
    keep_boxes.reserve(150);
    keep_scores.reserve(150);
    keep_classes.reserve(150);

    int num_boxes = output_shape[1];
    int basic_pos = output_shape[2];

    int pos = 0;
    if (!v8_head)
    {
        for (int boxs_idx = 0; boxs_idx < num_boxes; ++boxs_idx)
        {
            float obj_conf = boxes[pos + 4];

            if (obj_conf > conf_thres)
            {
                float obj_conf_thres = conf_thres / obj_conf;
                float max_prob = 0;
                int keep_class_idx;
                for (int class_idx = 0; class_idx < (basic_pos - 5); ++class_idx)
                {
                    float box_cls_score = boxes[pos + 5 + class_idx];

                    if (box_cls_score > obj_conf_thres)
                    {
                        max_prob = box_cls_score;
                        keep_class_idx = class_idx;
                    }
                }
                if (max_prob > obj_conf_thres)
                {
                    max_prob *= obj_conf;
                    double w, h, x, y;

                    w = boxes[pos + 2];
                    h = boxes[pos + 3];

                    x = boxes[pos] - (w * 0.5);
                    y = boxes[pos + 1] - (h * 0.5);

                    keep_boxes.emplace_back(cv::Rect(x, y, w, h));
                    keep_scores.emplace_back(max_prob);
                    keep_classes.emplace_back(keep_class_idx);
                }
            }
            pos += basic_pos;
        }
    }
    else
    {
        for (int boxs_idx = 0; boxs_idx < num_boxes; ++boxs_idx)
        {
            float max_prob = conf_thres;
            int keep_class_idx;
            for (int class_idx = 0; class_idx < (basic_pos - 4); ++class_idx)
            {
                float box_cls_score = boxes[pos + 4 + class_idx];

                if (box_cls_score > max_prob)
                {
                    max_prob = box_cls_score;
                    keep_class_idx = class_idx;
                }
            }
            if (max_prob > conf_thres)
            {
                double w, h, x, y;

                w = boxes[pos + 2];
                h = boxes[pos + 3];

                x = boxes[pos] - (w * 0.5);
                y = boxes[pos + 1] - (h * 0.5);

                keep_boxes.emplace_back(cv::Rect(x, y, w, h));
                keep_scores.emplace_back(max_prob);
                keep_classes.emplace_back(keep_class_idx);
            }
            pos += basic_pos;
        }
    }
}


void scale_boxes(cv::Rect& box, const preproc_struct& preproc_res)
{   
    box.x = cv::max(0, (box.x - preproc_res.w_p)) / preproc_res.scale;
    box.y = cv::max(0, (box.y - preproc_res.h_p)) / preproc_res.scale;
    box.width /= preproc_res.scale;
    box.height /= preproc_res.scale;
    box.width = cv::min(box.width, preproc_res.ori_w - box.x);
    box.height = cv::min(box.height, preproc_res.ori_h - box.y);
}


void draw_boxes(cv::Mat image, const cv::Rect& box, const float& score, const int& class_id,
    std::unordered_map<int, std::string> catid_labels, color_dicts& color_dicts)
{
    cv::Size textsize;
    cv::Rect textrect;
    cv::Point textpoint;
    cv::String text = cv::format("%s:%.2f", catid_labels[class_id].data(), score);

    textsize = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 0.7, 1, 0);
    textrect = { box.x, box.y, textsize.width - 4, (int)textsize.height };
    textpoint = { box.x - 10, box.y + (int)textsize.height - 1 };

    cv::Scalar colos = cv::Scalar(color_dicts.color_map[class_id][0], color_dicts.color_map[class_id][1], color_dicts.color_map[class_id][2]);

    cv::rectangle(image, box, colos, 2);
    cv::rectangle(image, textrect, colos, -1);
    cv::putText(image, text, textpoint, cv::FONT_HERSHEY_DUPLEX, 
        0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}


template <typename T = int>
static bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}


template <typename T>
void max_score_idx(
    const std::vector<float>& scores, const float& score_thres, T& scores_idxs)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > score_thres)
        {
            scores_idxs.emplace_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(scores_idxs.begin(), scores_idxs.end(), SortScorePairDescend<int>);
}


float get_iou(const cv::Rect& bbox1, const cv::Rect& bbox2)
{
    float area1 = bbox1.width * bbox1.height;
    float area2 = bbox2.width * bbox2.height;

    float xx1 = std::max(bbox1.x, bbox2.x);
    float yy1 = std::max(bbox1.y, bbox2.y);
    float xx2 = std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    float yy2 = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

    float w = std::max(xx2 - xx1, 0.f);
    float h = std::max(yy2 - yy1, 0.f);

    float inter = w * h;
    float u = (area1 + area2) - inter;

    if (u == 0) { return 1; }

    return inter / u;
};


void base_nms(
    const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores, const std::vector<int>& catid,
    const float& score_threshold, const float& nms_threshold, std::vector<int>& indices, const int& limit)
{
    int x, y, max_coord = 0;
    for (int i = 0; i < bboxes.size(); i++)
    {
        x = bboxes[i].x + bboxes[i].width;
        y = bboxes[i].y + bboxes[i].height;

        max_coord = std::max(x, max_coord);
        max_coord = std::max(y, max_coord);
    }

    // calculate offset and add offset to each bbox
    std::vector<cv::Rect> bboxes_offset;
    bboxes_offset.reserve(bboxes.size());
    int offset;
    for (int i = 0; i < bboxes.size(); i++)
    {
        offset = catid[i] * (max_coord + 1);
        bboxes_offset.emplace_back(
            cv::Rect(bboxes[i].x + offset, bboxes[i].y + offset, bboxes[i].width, bboxes[i].height)
        );
    }

    // 根据置信度对bboxes进行排序
    std::vector<std::pair<float, int>> score_index_vec;
    score_index_vec.reserve(50);
    max_score_idx<std::vector<std::pair<float, int>>>(scores, score_threshold, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    indices.reserve(30);
    for (int i = 0; i < score_index_vec.size(); ++i)
    {
        const int idx = score_index_vec[i].second;
        bool keep = true;

        for (int k = 0; k < (int)indices.size() && keep; ++k)
        {
            const int kept_idx = indices[k];
            float overlap = get_iou(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
        {
            indices.emplace_back(idx);
            if (indices.size() >= limit)
            {
                break;
            }
        }
    }
}


void get_final_preds(float* const heatmaps, preproc_struct& keypoints_trans, const std::array<int, 4>& output_shape,
    const cv::Rect& bbox, std::vector<float>& keypoints_scorce, std::vector<cv::Point2f>& keypoints)
{
    int step = output_shape[2] * output_shape[3];
    for (int i = 0; i < output_shape[1]; ++i)
    {
        int pos = i * step;
        float max_prob = 0.0f;
        float cur_prob = 0.0f;
        int idx = 0;
        for (int j = 0; j < step; ++j)
        {
            cur_prob = heatmaps[pos + j];
            if (cur_prob > max_prob)
            {
                max_prob = cur_prob;
                idx = j;
            }
        }
        keypoints_scorce.emplace_back(max_prob);

        float x = std::floor(idx % output_shape[3] + 0.5f);
        float y = std::floor(idx / output_shape[3] + 0.5f);
        float x_diff = heatmaps[pos + idx + 1] - heatmaps[pos + idx - 1];
        float y_diff = heatmaps[pos + idx + step] - heatmaps[pos + idx - step];
        x_diff = x_diff > 0 ? 0.25f : -0.25f;
        y_diff = y_diff > 0 ? 0.25f : -0.25f;
        x += x_diff;
        y += y_diff;
        keypoints.emplace_back(cv::Point2f(4 * x, 4 * y));
    }

    for (int i = 0; i < keypoints.size(); ++i)
    {
        keypoints[i].x -= (float)keypoints_trans.w_p;
        keypoints[i].y -= (float)keypoints_trans.h_p;
        keypoints[i].x /= keypoints_trans.scale;
        keypoints[i].y /= keypoints_trans.scale;
        keypoints[i].x += bbox.x;
        keypoints[i].y += bbox.y;
    }
}

void draw_keypoints(cv::Mat image, const std::vector<cv::Point2f>& keypoints, std::vector<float>& keypoints_score,
    float score, const std::vector<std::array<int, 2>>& points_linker)
{
    uint16_t idx1, idx2;
    float score1, score2;
    cv::Point2f pt1, pt2;
    for (int i = 0; i < points_linker.size(); ++i)
    {
        idx1 = points_linker[i][0];
        idx2 = points_linker[i][1];
        score1 = keypoints_score[idx1];
        score2 = keypoints_score[idx2];
        pt1 = keypoints[idx1];
        pt2 = keypoints[idx2];
        if (score1 > score && score2 > score)
        {
            cv::line(image, pt1, pt2, cv::Scalar(10, 250, 10), 1, cv::LINE_AA);
            cv::circle(image, pt1, 2, cv::Scalar(100, 50, 255), 2, cv::LINE_AA);
            cv::circle(image, pt2, 2, cv::Scalar(100, 50, 255), 2, cv::LINE_AA);
        }
        else if (score1 > score)
        {
            cv::circle(image, pt1, 2, cv::Scalar(100, 50, 255), 2, cv::LINE_AA);
        }
        else if (score2 > score)
        {
            cv::circle(image, pt2, 2, cv::Scalar(100, 50, 255), 2, cv::LINE_AA);
        }
    }
}