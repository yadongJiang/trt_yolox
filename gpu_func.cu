#include "gpu_func.cuh"
#include <cmath>

__global__ void gpu_detection(float* dev_ptr, int hw, int no, float test_conf)
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

__global__ void gpu_detection_bs(float* dev_ptr, int num, int hw, int no, float test_conf)
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
	gpu_detection << <grids, blocks >> > (dev_ptr, hw, no, test_conf);
}

void detection_bs(float* dev_ptr, int num, int hw, int no, float test_conf)
{
	cout << "test_conf: " << test_conf << endl;
	dim3 grids = (num * hw) / 16;
	dim3 blocks = 16;
	gpu_detection_bs << <grids, blocks >> > (dev_ptr, num, hw, no, test_conf);
}