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
__global__ void detection_kernel(float *dev_ptr, int hw, int no, float test_conf);
__global__ void detection_bs_kernel(float *dev_ptr, int num, int hw, int no, float test_conf);
__global__ void num_kernel(BoxInfo *dev_ptr, int num_boxes, float* dev_iou);

void detection(float *dev_ptr, int hw, int no, float test_conf);
void detection_bs(float *dev_ptr, int num, int hw, int no, float test_conf);

void nms(BoxInfo *h_ptr, int num_boxes, float* h_iou);

#endif