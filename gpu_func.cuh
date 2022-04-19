#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"

using namespace std;

__device__ float iou(BoxInfo *box1, BoxInfo *box2);

// 检测算法后处理相关
/* 单batch检测后处理核函数 */
__global__ void detection_kernel(float *dev_ptr, int hw, int no, float test_conf);
/* 多batch检测后处理 */
__global__ void detection_bs_kernel(float *dev_ptr, int num, int hw, int no, float test_conf);
/*nms核函数*/
__global__ void nms_kernel(BoxInfo *dev_ptr, int num_boxes, float* dev_iou);
/* 图像预处理核函数 */
__global__ void resize_kernel(float *dev_ptr, int dst_h, int dst_w, int crop_h, 
                            int crop_w, int src_h, int src_w, uchar *src_ptr);

/* 单batch检测后处理 */
void detection(float *dev_ptr, int hw, int no, float test_conf);
/* 多batch检测后处理 */
void detection_bs(float *dev_ptr, int num, int hw, int no, float test_conf);

void nms(BoxInfo *h_ptr, int num_boxes, float* h_iou);

/* 图像预处理*/
void yolox_resize(float *dev_ptr, int dst_h, int dst_w, int crop_h, int crop_w, 
                    int src_h, int src_w, uchar *src_ptr);

#endif