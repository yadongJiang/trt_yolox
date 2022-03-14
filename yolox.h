#ifndef YOLOX_H_
#define YOLOX_H_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvOnnxParser.h"

class Shape
{
public:
	Shape() :num_(0), channel_(0), height_(0), width_(0) {};
	Shape(int num, int channel, int height, int width)
		: num_(num), channel_(channel), height_(height), width_(width) {}

	inline int num() { return num_; }
	inline int channel() { return channel_; }
	inline int height() { return height_; }
	inline int width() { return width_; }
	inline int count() { return num_ * channel_ * height_ * width_; }

	inline void set_num(int num) { num_ = num; }
	inline void set_channel(int channel) { channel_ = channel; }
	inline void set_height(int height) { height_ = height; }
	inline void set_width(int width) { width_ = width; }

private:
	int num_;
	int channel_;
	int height_;
	int width_;
};

class Tensor2VecMat
{
public:
	Tensor2VecMat() {}
	std::vector<cv::Mat> operator()(float* input_data, 
					int c, int h, int w, int offset = 0)
	{
		std::vector<cv::Mat> input_channels;
		float* ptr = input_data + offset * c * h * w;
		for (int i = 0; i < c; i++)
		{
			cv::Mat channel(h, w, CV_32FC1, ptr);
			input_channels.push_back(channel);
			ptr += h * w;
		}
		return std::move(input_channels);
	}
};

struct OnnxDynamicNetInitParamV1
{
	std::string onnx_model;
	int gpu_id = 0;
	int max_batch_size = 1;
	std::string rt_stream_path = "./";
	std::string rt_model_name = "defaule.gie";
	bool use_fp16 = true;
};

struct BoxInfo
{
public:
	int x1;
	int y1;
	int x2;
	int y2;
	float class_conf;
	float score;
	int class_idx;

	BoxInfo() 
		:x1(0), y1(0), x2(0), y2(0), class_conf(0), score(0), class_idx(-1) {}
	BoxInfo(int lx, int ly, int rx, int ry, float conf, float s, int idx)
		: x1(lx), y1(ly), x2(rx), y2(ry), class_conf(conf), score(s), class_idx(idx) {}
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

	std::vector<BoxInfo> Extract(const cv::Mat& img);
	std::vector<std::vector<BoxInfo>> Extract(const std::vector<cv::Mat>& imgs);

private:
	void Forward();
	void ProPrecessCPU(const cv::Mat& img);
	void ProPrecessCPU(const std::vector<cv::Mat>& imgs);

	std::vector<BoxInfo> PostProcessCPU();
	std::vector<std::vector<BoxInfo>> PostProcessCPUMutilBs();

	std::vector<BoxInfo> PostProcessGPU();
	std::vector<std::vector<BoxInfo>> PostProcessGPUMutilBs();

	std::vector<BoxInfo> NMS();

	inline float IOU(BoxInfo& b1, BoxInfo& b2)
	{
		b1.x1 = std::min<float>(std::max<float>(b1.x1, 0.), 640.);
		b1.y1 = std::min<float>(std::max<float>(b1.y1, 0.), 640.);
		b1.x2 = std::min<float>(std::max<float>(b1.x2, 0.), 640.);
		b1.y2 = std::min<float>(std::max<float>(b1.y2, 0.), 640.);

		b2.x1 = std::min<float>(std::max<float>(b2.x1, 0.), 640.);
		b2.y1 = std::min<float>(std::max<float>(b2.y1, 0.), 640.);
		b2.x2 = std::min<float>(std::max<float>(b2.x2, 0.), 640.);
		b2.y2 = std::min<float>(std::max<float>(b2.y2, 0.), 640.);

		float x1 = std::max<float>(b1.x1, b2.x1);
		float y1 = std::max<float>(b1.y1, b2.y1);
		float x2 = std::min<float>(b1.x2, b2.x2);
		float y2 = std::min<float>(b1.y2, b2.y2);

		float inter_area = std::max<float>(x2 - x1, 0) * std::max<float>(y2 - y1, 0);
		float b1_area = std::max<float>(b1.x2 - b1.x1, 0) * std::max<float>(b1.y2 - b1.y1, 0);
		float b2_area = std::max<float>(b2.x2 - b2.x1, 0) * std::max<float>(b2.y2 - b2.y1, 0);
		return inter_area / (b1_area + b2_area - inter_area + 1e-5);
	}

private:
	bool CheckFileExist(const std::string& path);
	void LoadOnnxModel(const std::string& onnx_file);
	void SaveRtModel(const std::string& path);
	void set_batch_size(int bs) { _batch_size = bs; }

	bool LoadGieStreamBuildContext(const std::string& gue_file);

	void deserializeCudaEngine(const void* blob, std::size_t size);

	void mallocInputOutput();

	void DecodeAndFiltedBoxes(std::vector<float>&output, 
				int stride, int height, int width, int channels);

	inline void FindMaxConfAndIdx(const std::vector<float>& vec, 
				float& class_conf, int& class_pred);

	static bool compose(BoxInfo& b1, BoxInfo& b2)
	{
		return b1.score > b2.score;
	}

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

	cv::Size output_size_{640, 640};

	cudaStream_t stream_;

	std::vector<BoxInfo> filted_pred_boxes_;
	float test_conf_ = 0.3;
};

#endif