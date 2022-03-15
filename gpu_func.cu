#include "gpu_func.cuh"
#include <cmath>

__global__ void detection_kernel(float* dev_ptr, int hw, int no, float test_conf)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = offset * no;

	float obj_conf = dev_ptr[pos + 4];

	float class_conf = FLT_MIN;
	int class_pred = -1;
	for (int i = pos + 4; i < pos + no; i++)
	{
		if (class_conf < dev_ptr[i])
		{
			class_conf = dev_ptr[i];
			class_pred = i;
		}
	}

	float score = obj_conf * class_conf;
	dev_ptr[pos + 4] = score;

	if (score >= test_conf)
	{
		int col, row, stride;
		if (offset < 6400)
		{
			col = offset % 80;
			row = offset / 80;
			stride = 8;
		}
		else if (offset >= 6400 && offset < 8000)
		{
			col = (offset - 6400) % 40;
			row = (offset - 6400) / 40;
			stride = 16;
		}
		else
		{
			col = (offset - 8000) % 20;
			row = (offset - 8000) / 20;
			stride = 32;
		}

		dev_ptr[pos + 0] = (dev_ptr[pos + 0] + col) * stride;
		dev_ptr[pos + 1] = (dev_ptr[pos + 1] + row) * stride;
		dev_ptr[pos + 2] = exp(dev_ptr[pos + 2]) * stride;
		dev_ptr[pos + 3] = exp(dev_ptr[pos + 3]) * stride;
	}
}

__global__ void detection_bs_kernel(float* dev_ptr, int num, int hw, int no, float test_conf)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = offset * no;

	float obj_conf = dev_ptr[pos + 4];
	float class_conf = FLT_MIN;
	for (int i = pos + 4; i < pos + no; i++)
	{
		float t = dev_ptr[i];
		if (class_conf < t)
			class_conf = t;
	}

	float score = obj_conf * class_conf;
	dev_ptr[pos + 4] = score;

	if (score >= test_conf)
	{
		int row, col, stride;
		int n_idx = offset / hw;
		if (offset < n_idx * hw + 6400)
		{
			col = (offset - n_idx * hw) % 80;
			row = (offset - n_idx * hw) / 80;
			stride = 8;
		}
		else if (offset >= n_idx * hw + 6400 && offset < n_idx * hw + 8000)
		{
			col = (offset - n_idx * hw - 6400) % 40;
			row = (offset - n_idx * hw - 6400) / 40;
			stride = 16;
		}
		else
		{
			col = (offset - n_idx * hw - 8000) % 20;
			row = (offset - n_idx * hw - 8000) / 20;
			stride = 32;
		}

		dev_ptr[pos + 0] = (dev_ptr[pos + 0] + col) * stride;
		dev_ptr[pos + 1] = (dev_ptr[pos + 1] + row) * stride;
		dev_ptr[pos + 2] = exp(dev_ptr[pos + 2]) * stride;
		dev_ptr[pos + 3] = exp(dev_ptr[pos + 3]) * stride;
	}
}

void detection(float* dev_ptr, int hw, int no, float test_conf)
{
	cout << "test_conf: " << test_conf << endl;
	dim3 grids = hw / 16;
	dim3 blocks = 16;
	detection_kernel << <grids, blocks >> > (dev_ptr, hw, no, test_conf);
}

void detection_bs(float* dev_ptr, int num, int hw, int no, float test_conf)
{
	cout << "test_conf: " << test_conf << endl;
	dim3 grids = (num * hw) / 16;
	dim3 blocks = 16;
	detection_bs_kernel << <grids, blocks >> > (dev_ptr, num, hw, no, test_conf);
}

__device__ float iou(BoxInfo box1, BoxInfo box2)
{
	float x1 = box1.x1 > box2.x1 ? box1.x1 : box2.x1; 
	float y1 = box1.y1 > box2.y1 ? box1.y1 : box2.y1; 
	float x2 = box1.x2 < box2.x2 ? box1.x2 : box2.x2; 
	float y2 = box1.y2 < box2.y2 ? box1.y2 : box2.y2; 

	float inter_area = ((x2 - x1) < 0 ? 0 : (x2 - x1)) * ((y2 - y1) < 0 ? 0 : (y2 - y1)); 
	float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1); 
	float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1); 

	return inter_area / (box1_area + box2_area - inter_area + 1e-5);
}

__global__ void num_kernel(BoxInfo* dev_ptr, int num_boxes, float* dev_iou)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.x;
	int x = threadIdx.x;

	float iou_val = iou(dev_ptr[y], dev_ptr[x]);
	dev_iou[offset] = iou_val;
}

void nms(BoxInfo* h_ptr, int num_boxes, float* h_iou)
{
	dim3 grids(num_boxes);
	dim3 blocks(num_boxes);

	BoxInfo* dev_ptr;
	cudaMalloc((void**)&dev_ptr, num_boxes * sizeof(BoxInfo));
	cudaMemcpy(dev_ptr, h_ptr, num_boxes * sizeof(BoxInfo), cudaMemcpyHostToDevice);
	float* dev_iou;
	cudaMalloc((void**)&dev_iou, num_boxes * num_boxes * sizeof(float));
	num_kernel << <grids, blocks >> > (dev_ptr, num_boxes, dev_iou);

	cudaMemcpy(h_iou, dev_iou, num_boxes * num_boxes * sizeof(float), cudaMemcpyDeviceToHost);
}