#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// 检测算法后处理相关
__global__ void gpu_detection(float *dev_ptr, int hw, int no, float test_conf);
__global__ void gpu_detection_bs(float *dev_ptr, int num, int hw, int no, float test_conf);

void detection(float *dev_ptr, int hw, int no, float test_conf);
void detection_bs(float *dev_ptr, int num, int hw, int no, float test_conf);

#endif