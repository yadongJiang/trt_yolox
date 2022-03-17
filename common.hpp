#ifndef MAT_TRANSFORM_H_
#define MAT_TRANSFORM_H_
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>

class ComposeMatLambda
{
public:
	using FuncionType = std::function<cv::Mat(const cv::Mat&)>;

	ComposeMatLambda() = default;
	ComposeMatLambda(const std::vector<FuncionType>&lambda) :lambda_(lambda)
	{
		;
	}
	ComposeMatLambda(const ComposeMatLambda& cml)
	{
		lambda_ = cml.lambda_;
	}
	cv::Mat operator()(cv::Mat & img)
	{
		for (auto func : lambda_)
			img = func(img);
		return img;
	}
private:
	std::vector<FuncionType> lambda_;
};

class YoloXResize
{
public:
	YoloXResize(const cv::Size& output_size) : output_size_(output_size) {}

	cv::Mat operator()(const cv::Mat& img)
	{
		cv::Mat padded_img;
		if (img.channels() == 3)
		{
			padded_img = cv::Mat(output_size_.height, output_size_.width, 
								 CV_8UC3, cv::Scalar(114, 114, 114)); // * 114;
		}
		else
		{
			padded_img = cv::Mat(output_size_.height, output_size_.width, 
								 CV_8UC1, cv::Scalar(114));
		}

		float r = std::min((float)output_size_.height / img.rows, 
						   (float)output_size_.width / img.cols);
		cv::Mat resized_img;
		if (r != 1.0)
		{
			cv::resize(img, resized_img, cv::Size(int(img.cols * r), 
					   int(img.rows * r)), cv::INTER_LINEAR);
		}

		cv::Rect roi = cv::Rect(0, 0, int(img.cols * r), int(img.rows * r));
		resized_img.copyTo(padded_img(roi));

		return std::move(padded_img);
	}

private:
	cv::Size output_size_;
};

class MatToFloat
{
public:
	MatToFloat() {}
	cv::Mat operator()(const cv::Mat& img)
	{
		cv::Mat img_float;
		if (img.type() == CV_8UC1)
			img.convertTo(img_float, CV_32FC1);
		else if (img.type() == CV_8UC3)
			img.convertTo(img_float, CV_32FC3);
		else if (img.type() == CV_32FC1 || img.type() == CV_32FC3)
			img_float = img;
		else
		{
			std::cerr << "img's type is not unsupport!" << std::endl;
			assert(0);
		}
		return std::move(img_float);
	}
};

// ============================================================================================

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

	inline void Reshape(int num, int channel, int height, int width)
	{
		num_ = num;
		channel_ = channel;
		height_ = height;
		width_ = width;
	}

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

__device__ __host__ struct BoxInfo
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

#endif