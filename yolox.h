#ifndef YOLOX_H_
#define YOLOX_H_

#include "trt_onnx_base.h"

class YOLOX : public TRTOnnxBase 
{
public:
	YOLOX(const OnnxDynamicNetInitParamV1& params);
	YOLOX() = delete;
	virtual ~YOLOX();

	// 单张图像输出执行函数
	std::vector<BoxInfo> Extract(const cv::Mat& img);
	// 多batch输入执行函数
	std::vector<std::vector<BoxInfo>> Extract(const std::vector<cv::Mat>& imgs);

private:
	// 单张输入预处理，trt输入内存填充
	void ProPrecessCPU(const cv::Mat& img);
	// 多batch输入预处理，trt输入内存填充
	void ProPrecessCPU(const std::vector<cv::Mat>& imgs);

	// gpu预处理函数
	void ProPrecessGPU(const cv::Mat& img);
	void ProPrecessGPU(const std::vector<cv::Mat>& imgs);

	// 单张输入的cpu后处理函数
	std::vector<BoxInfo> PostProcessCPU();
	// 多batch输入的cpu后处理函数
	std::vector<std::vector<BoxInfo>> PostProcessCPUMutilBs();

	// 单张输入的gpu后处理函数
	std::vector<BoxInfo> PostProcessGPU();
	// 多batch输入的gpu后处理函数
	std::vector<std::vector<BoxInfo>> PostProcessGPUMutilBs();

	// nms cpu代码
	std::vector<BoxInfo> NMS();
	// nms gpu代码
	std::vector<BoxInfo> NMSGpu();

	// 计算iou的cpu代码
	inline float IOU(BoxInfo& b1, BoxInfo& b2)
	{

		float x1 = b1.x1 > b2.x1 ? b1.x1 : b2.x1;  
		float y1 = b1.y1 > b2.y1 ? b1.y1 : b2.y1; 
		float x2 = b1.x2 < b2.x2 ? b1.x2 : b2.x2; 
		float y2 = b1.y2 < b2.y2 ? b1.y2 : b2.y2; 

		float inter_area = ((x2 - x1) < 0 ? 0 : (x2 - x1)) * ((y2 - y1) < 0 ? 0 : (y2 - y1));
		float b1_area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1); 
		float b2_area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1); 

		return inter_area / (b1_area + b2_area - inter_area + 1e-5);
	}
	// 调整预测框，使框的值处于合理范围
	inline void RefineBoxes()
	{
		for (auto& box : filted_pred_boxes_)
		{
			box.x1 = box.x1 < 0. ? 0. : box.x1;
			box.x1 = box.x1 > 640. ? 640. : box.x1;
			box.y1 = box.y1 < 0. ? 0. : box.y1;
			box.y1 = box.y1 > 640. ? 640. : box.y1;
			box.x2 = box.x2 < 0. ? 0. : box.x2;
			box.x2 = box.x2 > 640. ? 640. : box.x2;
			box.y2 = box.y2 < 0. ? 0. : box.y2;
			box.y2 = box.y2 > 640. ? 640. : box.y2;
		}
	}

private:
	// decode预测框, cpu代码
	void DecodeAndFiltedBoxes(std::vector<float>&output, 
				int stride, int height, int width, int channels);

	inline void FindMaxConfAndIdx(const std::vector<float>& vec, 
				float& class_conf, int& class_pred);

	static bool compose(BoxInfo& b1, BoxInfo& b2)
	{
		return b1.score > b2.score;
	}
	// 将预测框映射到原图尺度上
	void scale_coords(std::vector<BoxInfo>& pred_boxes, float r);

private:

	cudaStream_t stream_;

	std::vector<BoxInfo> filted_pred_boxes_;
	float test_conf_ = 0.3;

	ComposeMatLambda *transform_;
	Tensor2VecMat *tensor2mat;
};

#endif