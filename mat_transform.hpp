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
			padded_img = cv::Mat(output_size_.height, output_size_.width, CV_8UC3, cv::Scalar(114, 114, 114)); // * 114;
		}
		else
		{
			padded_img = cv::Mat(output_size_.height, output_size_.width, CV_8UC1, cv::Scalar(114));
		}

		float r = std::min((float)output_size_.height / img.rows, (float)output_size_.width / img.cols);
		cv::Mat resized_img;
		if (r != 1.0)
		{
			cv::resize(img, resized_img, cv::Size(int(img.cols * r), int(img.rows * r)), cv::INTER_LINEAR);
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