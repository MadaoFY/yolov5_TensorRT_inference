#include "preprocess.h"

#include <device_launch_parameters.h>


__global__ void warpaffine_nearest_bgrbgr2rrggbb_kernel(
    uint8_t* src, int src_step_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int h_p, int w_p)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    float c0, c1, c2;
    if (dy < h_p || dy >(dst_height - h_p) || dx < w_p || dx >(dst_width - w_p))
    {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else
    {
        float src_x = m_x1 * (dx + 0.5f) + m_y1 * dy + m_z1 - 0.5f;
        float src_y = m_x2 * dx + m_y2 * (dy + 0.5f) + m_z2 - 0.5f;

        int sy_1 = floorf(src_y + 0.5f);
        int sx_1 = floorf(src_x + 0.5f);

        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
        uint8_t* p = const_value;

        if (sy_1 >= 0 && sy_1 <= src_height && sx_1 >=0 && sx_1 <= src_width) 
        {
            p = src + sy_1 * src_step_size + sx_1 * 3;
        }

        c0 = p[0];
        c1 = p[1];
        c2 = p[2];
    }

    // normalization
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // bgrbgrbgr to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    pdst_c0[0] = c2;
    pdst_c0[area] = c1;
    pdst_c0[2 * area] = c0;
}

__global__ void warpaffine_bilinear_bgrbgr2rrggbb_kernel(
    uint8_t* src, int src_step_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int h_p, int w_p)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    float c0, c1, c2;
    if (dy < h_p || dy >(dst_height - h_p) || dx < w_p || dx >(dst_width - w_p))
    {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else
    {
        float src_x = m_x1 * (dx + 0.5f) + m_y1 * dy + m_z1 - 0.5f;
        float src_y = m_x2 * dx + m_y2 * (dy + 0.5f) + m_z2 - 0.5f;

        int sy_1 = floorf(src_y);
        int sx_1 = floorf(src_x);
        int sy_2 = sy_1 + 1;
        int sx_2 = sx_1 + 1;

        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
        float a2 = src_y - sy_1;
        float a1 = 1.0f - a2;
        float b2 = src_x - sx_1;
        float b1 = 1.0f - b2;
        float w11 = a1 * b1;
        float w12 = a1 * b2;
        float w21 = a2 * b1;
        float w22 = a2 * b2;
        uint8_t* p11 = const_value;
        uint8_t* p12 = const_value;
        uint8_t* p21 = const_value;
        uint8_t* p22 = const_value;

        /*if (sy_1 >= 0) {
            if (sx_1 >= 0)*/
        p11 = src + sy_1 * src_step_size + sx_1 * 3;

        //if (sx_2 < src_width)
        p12 = src + sy_1 * src_step_size + sx_2 * 3;
        //}

        /*if (sy_2 < src_height) {
            if (sx_1 >= 0)*/
        p21 = src + sy_2 * src_step_size + sx_1 * 3;

        /*if (sx_2 < src_width)*/
        p22 = src + sy_2 * src_step_size + sx_2 * 3;
        //}

        c0 = w11 * p11[0] + w12 * p12[0] + w21 * p21[0] + w22 * p22[0] + 0.5f;
        c1 = w11 * p11[1] + w12 * p12[1] + w21 * p21[1] + w22 * p22[1] + 0.5f;
        c2 = w11 * p11[2] + w12 * p12[2] + w21 * p21[2] + w22 * p22[2] + 0.5f;
    }

    // normalization
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // bgrbgrbgr to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    pdst_c0[0] = c2;
    pdst_c0[area] = c1;
    pdst_c0[2 * area] = c0;
}


void cuda_preprocess(cv::Mat& image, preproc_struct& image_trans, std::vector<void*>& bufferH,
    std::vector<void*>& bufferD, std::vector<int>& bindingsize, cudaStream_t& stream, cv::Size resize)
{
    int h, w, h_p, w_p;

    float scale = cv::min((float)resize.height / (float)image.rows, (float)resize.width / (float)image.cols);
    scale = cv::min(scale, 1.1f);

    h = image.rows * scale;
    w = image.cols * scale;
    h_p = (resize.height - h) * 0.5f;
    w_p = (resize.width - w) * 0.5f;

    image_trans.scale = scale;
    image_trans.h_p = h_p;
    image_trans.w_p = w_p;

    // copy data to device memory
    // memcpy(bufferH[2], image.data, bindingsize[2]);
    // cudaMemcpyAsync(bufferD[2], bufferH[2], bindingsize[2], cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bufferD[2], image.data, bindingsize[2], cudaMemcpyHostToDevice, stream);

    AffineMatrix s2d, d2s;

    /*s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = (resize.width - scale * image.cols + scale - 1) * 0.5f;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = (resize.height - scale * image.rows + scale - 1) * 0.5f;*/

    d2s.value[0] = 1.0f / scale;
    d2s.value[1] = 0;
    d2s.value[2] = (image.cols - resize.width / scale + d2s.value[0] - 1) * 0.5f;
    d2s.value[3] = 0;
    d2s.value[4] = 1.0f / scale;
    d2s.value[5] = (image.rows - resize.height / scale + d2s.value[0] - 1) * 0.5f;

    /*cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));*/

    dim3 block(128, 1);
    dim3 grid((resize.width + block.x - 1) / block.x, (resize.height + block.y - 1) / block.y);

    warpaffine_nearest_bgrbgr2rrggbb_kernel <<< grid, block, 0, stream >>> (
        (uint8_t*)bufferD[2], image.cols * 3, image.cols,
        image.rows, (float*)bufferD[0], resize.width,
        resize.height, 0, d2s, h_p, w_p);
}
