#ifndef TRT_ONNX_BASE_H_
#define TRT_ONNX_BASE_H_

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
	int num_classes = 1;
	bool use_fp16 = true;
};


class TRTOnnxBase
{
public:
	TRTOnnxBase() = delete;
	TRTOnnxBase(const OnnxDynamicNetInitParamV1& param);

	virtual ~TRTOnnxBase();

private:
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

protected:
	// trt infer
	void Forward();
	// 设置当前批次中batch_size
	void set_batch_size(int bs) { _batch_size = bs; } 

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

protected:
	OnnxDynamicNetInitParamV1 params_;
	float* h_input_ptr;
	float* d_input_ptr;
	float* d_output_ptr;
	float* h_output_ptr;
	std::vector<void*> buffer_;

	Shape in_shape_; // { _max_batch_size, 3, 640, 640 };
	Shape out_shape_; // { _max_batch_size, 8400, 6, 1 };

	cv::Size crop_size_{ 640, 640 };

	int _batch_size = 1;
	int _max_batch_size = 1;

private:
	Logger logger_;

	nvinfer1::IRuntime* runtime_{ nullptr };
	nvinfer1::ICudaEngine* engine_{ nullptr };
	nvinfer1::IExecutionContext* context_{ nullptr };
	nvinfer1::IHostMemory* gie_model_stream_{ nullptr };

	cudaStream_t stream_;
};

#endif