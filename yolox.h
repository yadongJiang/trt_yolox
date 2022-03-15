#ifndef YOLOX_H_
#define YOLOX_H_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvOnnxParser.h"
#include "common.hpp"

struct OnnxDynamicNetInitParamV1
{
	std::string onnx_model;
	int gpu_id = 0;
	int max_batch_size = 1;
	std::string rt_stream_path = "./";
	std::string rt_model_name = "defaule.gie";
	bool use_fp16 = true;
};

class Logger : public nvinfer1::ILogger
{
public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg)
	{
		switch (severity)
		{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
			std::cerr << "kINTERNAL_ERROR: " << msg << std::endl;
			break;
		case nvinfer1::ILogger::Severity::kERROR:
			std::cerr << "kERROR: " << msg << std::endl;
			break;
		case nvinfer1::ILogger::Severity::kWARNING:
			std::cerr << "kWARNING: " << msg << std::endl;
			break;
		case nvinfer1::ILogger::Severity::kINFO:
			std::cerr << "kINFO: " << msg << std::endl;
			break;
		case nvinfer1::ILogger::Severity::kVERBOSE:
			std::cerr << "kVERBOSE: " << msg << std::endl;
			break;
		default:
			break;
		}
	}
};

class YOLOX
{
public:
	YOLOX(const OnnxDynamicNetInitParamV1& params);
	YOLOX() = delete;

	// 单张图像输出执行函数
	std::vector<BoxInfo> Extract(const cv::Mat& img);
	// 多batch输入执行函数
	std::vector<std::vector<BoxInfo>> Extract(const std::vector<cv::Mat>& imgs);

private:
	// trt infer
	void Forward();
	// 单张输入预处理，trt输入内存填充
	void ProPrecessCPU(const cv::Mat& img);
	// 多batch输入预处理，trt输入内存填充
	void ProPrecessCPU(const std::vector<cv::Mat>& imgs);

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
	std::vector<BoxInfo> NUMGpu();

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
	bool CheckFileExist(const std::string& path);
	// 直接加载onnx模型，并转换成trt模型
	void LoadOnnxModel(const std::string& onnx_file);
	// 保存trt模型
	void SaveRtModel(const std::string& path);
	// 如果已存在trt模型，则直接读取并反序列化
	bool LoadGieStreamBuildContext(const std::string& gue_file);
	// 利用反序列化的模型生成执行上下文(context_)
	void deserializeCudaEngine(const void* blob, std::size_t size);
	// 分配输入输出内存空间(cpu + gpu)
	void mallocInputOutput();
	// decode预测框, cpu代码
	void DecodeAndFiltedBoxes(std::vector<float>&output, 
				int stride, int height, int width, int channels);

	inline void FindMaxConfAndIdx(const std::vector<float>& vec, 
				float& class_conf, int& class_pred);

	static bool compose(BoxInfo& b1, BoxInfo& b2)
	{
		return b1.score > b2.score;
	}
	// 设置当前批次中batch_size
	void set_batch_size(int bs) { _batch_size = bs; }
	// 将预测框映射到原图尺度上
	void scale_coords(std::vector<BoxInfo>& pred_boxes, float r);

private:
	OnnxDynamicNetInitParamV1 params_;
	Logger logger_;
	
	int _batch_size = 1;
	int _max_batch_size = 1;

	nvinfer1::IRuntime* runtime_{ nullptr };
	nvinfer1::ICudaEngine* engine_{ nullptr };
	nvinfer1::IExecutionContext* context_{ nullptr };
	nvinfer1::IHostMemory* gie_model_stream_{ nullptr };

	float* h_input_ptr;
	float* d_input_ptr;
	float* d_output_ptr;
	float* h_output_ptr;
	std::vector<void*> buffer_;

	Shape in_shape_{ _max_batch_size, 3, 640, 640 };
	Shape out_shape_{ _max_batch_size, 8400, 6, 1 };

	cv::Size crop_size_{640, 640};

	cudaStream_t stream_;

	std::vector<BoxInfo> filted_pred_boxes_;
	float test_conf_ = 0.3;

	ComposeMatLambda *transform_;
	Tensor2VecMat *tensor2mat;
};

#endif