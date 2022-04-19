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

__global__ void nms_kernel(BoxInfo* dev_ptr, int num_boxes, float* dev_iou)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.x;
	int x = threadIdx.x;

	float iou_val = iou(dev_ptr[y], dev_ptr[x]);
	dev_iou[offset] = iou_val;
}

__global__ void resize_kernel(float* d_dst, int dst_h, int dst_w, int crop_h, int crop_w, int src_h, int src_w, uchar* d_src)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = y * dst_w + x;

	float scale_x = float(src_w) / crop_w;
	float scale_y = float(src_h) / crop_h;

	if (y >= crop_h || x >= crop_w)
	{
		d_dst[offset] = 114.0;
		d_dst[dst_h * dst_w + offset] = 114.0;
		d_dst[dst_h * dst_w * 2 + offset] = 114.0;
	}
	else
	{
		float src_x = (x + 0.5) * scale_x - 0.5;
		float src_y = (y + 0.5) * scale_y - 0.5;

		int src_x_0 = int(floor(src_x));
		int src_y_0 = int(floor(src_y));
		int src_x_1 = src_x_0 + 1 <= src_w - 1 ? src_x_0 + 1 : src_w - 1;
		int src_y_1 = src_y_0 + 1 <= src_h - 1 ? src_y_0 + 1 : src_h - 1;

		for (int c = 0; c < 3; c++)
		{
			uchar v00 = d_src[(src_y_0 * src_w + src_x_0) * 3 + c];
			uchar v01 = d_src[(src_y_0 * src_w + src_x_1) * 3 + c];
			uchar v10 = d_src[(src_y_1 * src_w + src_x_0) * 3 + c];
			uchar v11 = d_src[(src_y_1 * src_w + src_x_1) * 3 + c];
			uchar value0 = (src_x_1 - src_x) * v00 + (src_x - src_x_0) * v01;
			uchar value1 = (src_x_1 - src_x) * v10 + (src_x - src_x_0) * v11;

			uchar value = uchar((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1);
			float v = float(value);  // / 255.;
			d_dst[c * dst_h * dst_w + offset] = v;
		}
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

void nms(BoxInfo* h_ptr, int num_boxes, float* h_iou)
{
	dim3 grids(num_boxes);
	dim3 blocks(num_boxes);

	BoxInfo* dev_ptr;
	cudaMalloc((void**)&dev_ptr, num_boxes * sizeof(BoxInfo));
	cudaMemcpy(dev_ptr, h_ptr, num_boxes * sizeof(BoxInfo), cudaMemcpyHostToDevice);
	float* dev_iou;
	cudaMalloc((void**)&dev_iou, num_boxes * num_boxes * sizeof(float));
	nms_kernel << <grids, blocks >> > (dev_ptr, num_boxes, dev_iou);

	cudaMemcpy(h_iou, dev_iou, num_boxes * num_boxes * sizeof(float), cudaMemcpyDeviceToHost);
}

void yolox_resize(float* dev_ptr, int dst_h, int dst_w, int crop_h, int crop_w, int src_h, int src_w, uchar* h_src)
{
	uchar* d_src;
	cudaMalloc((uchar**)&d_src, src_h * src_w * 3 * sizeof(uchar));
	cudaMemcpy(d_src, h_src, src_h * src_w * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

	dim3 grids(dst_w / 32, dst_h / 32);
	dim3 blocks(32, 32);
	resize_kernel << <grids, blocks >> > (dev_ptr, dst_h, dst_w, crop_h, crop_w, src_h, src_w, d_src);

	cudaFree(d_src);
}