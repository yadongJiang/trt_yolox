#include "yolox.h"
#include "gpu_func.cuh"

YOLOX::YOLOX(const OnnxDynamicNetInitParamV1& params) : TRTOnnxBase(params)
{
	transform_ = new ComposeMatLambda({ 
		YoloXResize(crop_size_),
		MatToFloat(),
	});
}

YOLOX::~YOLOX()
{

}

std::vector<BoxInfo> YOLOX::Extract(const cv::Mat& img)
{
	if (img.empty())
		return {};

	set_batch_size(1);

	// 预处理cpu接口
	/*ProPrecessCPU(img);*/
	// 预处理gpu结接口
	ProPrecessGPU(img);

	Forward();

	// 后处理cpu接口
	/*std::vector<BoxInfo> pred_boxes = PostProcessCPU();*/
	// 后处理gpu接口
	std::vector<BoxInfo> pred_boxes = PostProcessGPU();

	float r = std::min<float>(float(crop_size_.height) / img.rows,
		float(crop_size_.width) / img.cols);
	scale_coords(pred_boxes, r);

	return std::move(pred_boxes);
}

std::vector<std::vector<BoxInfo>> 
			YOLOX::Extract(const std::vector<cv::Mat>& imgs)
{
	if (imgs.empty())
		return {};
	if (imgs.size() > _max_batch_size)
	{
		std::cerr << "imgs' number is lager than max_batch_size..." << std::endl;
		return {};
	}

	set_batch_size(imgs.size());

	/* ProPrecessCPU(imgs); */
	ProPrecessGPU(imgs);

	Forward();
	// 多batch的cpu后处理
	/*std::vector<std::vector<BoxInfo>> bs_pred_boxes = PostProcessCPUMutilBs();*/
	// 多batch的gpu后处理
	std::vector<std::vector<BoxInfo>> bs_pred_boxes = PostProcessGPUMutilBs();

	for (int i = 0; i < imgs.size(); i++)
	{
		const cv::Mat& img = imgs[i];
		float r = min<float>(float(crop_size_.height) / img.rows,
							 float(crop_size_.width) / img.cols);
		scale_coords(bs_pred_boxes[i], r);
	}
	return std::move(bs_pred_boxes);
}

void YOLOX::ProPrecessCPU(const cv::Mat& img)
{
	cv::Mat img_tmp = img;

	cv::Mat sample_float = (*transform_)(img_tmp);

	std::vector<cv::Mat> channels = (*tensor2mat)(h_input_ptr, 3, 640, 640);
	cv::split(sample_float, channels);

	cudaMemcpy(d_input_ptr, h_input_ptr, in_shape_.count() * sizeof(float),
		cudaMemcpyHostToDevice);
}

void YOLOX::ProPrecessCPU(const std::vector<cv::Mat>& imgs)
{
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat img_tmp = imgs[i];
		cv::Mat sample_float = (*transform_)(img_tmp);
		std::vector<cv::Mat> channels = (*tensor2mat)(h_input_ptr, 3, 640, 640, i);
		cv::split(sample_float, channels);
	}
}

void YOLOX::ProPrecessGPU(const cv::Mat& img)
{
	float r = std::min((float)crop_size_.width / img.cols, 
					   (float)crop_size_.height / img.rows);

	int crop_h = int(r * img.rows);
	int crop_w = int(r * img.cols);

	yolox_resize(d_input_ptr, crop_size_.height, crop_size_.width, crop_h, crop_w, img.rows, img.cols, img.data);
}

void YOLOX::ProPrecessGPU(const std::vector<cv::Mat>& imgs)
{
	if (imgs.empty())
		return;
	
	for (int i = 0; i < imgs.size(); i++)
	{
		float r = std::min((float)crop_size_.width / imgs[i].cols,
							(float)crop_size_.height / imgs[i].rows);
		cout << "r: " << r << endl;

		int crop_h = int(r * imgs[i].rows);
		int crop_w = int(r * imgs[i].cols);
		yolox_resize(d_input_ptr + i * crop_size_.width * crop_size_.height*3, crop_size_.height, 
						crop_size_.width, crop_h, crop_w, imgs[i].rows, imgs[i].cols, imgs[i].data);
	}
}

std::vector<BoxInfo> YOLOX::PostProcessGPU()
{
	int no = out_shape_.height();
	int hw = out_shape_.channel();
	// gpu后处理函数
	detection_bs(d_output_ptr, 1, hw, no, test_conf_);

	cudaMemcpy(h_output_ptr, d_output_ptr, out_shape_.count() * sizeof(float), 
					cudaMemcpyDeviceToHost);

	filted_pred_boxes_.clear();

	for (int i = 0; i < hw; i++)
	{
		int pos = i * no;
		float obj_conf = h_output_ptr[pos + 4];
		if (obj_conf < test_conf_)
			continue;

		std::vector<float> cls(h_output_ptr + pos + 5, h_output_ptr + pos + no);
		float class_conf;
		int class_pred;
		FindMaxConfAndIdx(cls, class_conf, class_pred);

		float cx = h_output_ptr[pos + 0];
		float cy = h_output_ptr[pos + 1];
		float w = h_output_ptr[pos + 2];
		float h = h_output_ptr[pos + 3];

		filted_pred_boxes_.emplace_back(BoxInfo(cx - w / 2, cy - h / 2,
			cx + w / 2, cy + h / 2,
			class_conf, obj_conf, class_pred));
	}
	// cpu nms
	std::vector<BoxInfo> pred_boxes = NMS();
	// gpu nms
	/*std::vector<BoxInfo> pred_boxes = NUMGpu();*/
	cout << "res boxes size: " << pred_boxes.size() << endl;

	return std::move(pred_boxes);
}

std::vector<std::vector<BoxInfo>> YOLOX::PostProcessGPUMutilBs()
{
	int num = out_shape_.num();
	int no = out_shape_.height();
	int hw = out_shape_.channel();
	detection_bs(d_output_ptr, num, hw, no, test_conf_);
	cudaMemcpy(h_output_ptr, d_output_ptr, out_shape_.count() * sizeof(float), 
		       cudaMemcpyDeviceToHost);
	
	vector<vector<BoxInfo>> bs_pred_boxes;
	int bs = _batch_size;
	for (int b = 0; b < bs; b++)
	{
		filted_pred_boxes_.clear();
		for (int i = 0; i < hw; i++)
		{
			int pos = b * hw * no + i * no;
			float obj_conf = h_output_ptr[pos + 4];
			if (obj_conf < test_conf_)
				continue;

			vector<float> cls(h_output_ptr + pos + 5, h_output_ptr + pos + no);
			float class_conf;
			int class_pred;
			FindMaxConfAndIdx(cls, class_conf, class_pred);

			float cx = h_output_ptr[pos + 0];
			float cy = h_output_ptr[pos + 1];
			float w = h_output_ptr[pos + 2];
			float h = h_output_ptr[pos + 3];

			filted_pred_boxes_.emplace_back(BoxInfo(cx - w / 2, cy - h / 2,
				cx + w / 2, cy + h / 2,
				class_conf, obj_conf, class_pred));
		}
		vector<BoxInfo> pred_boxes = NMS();
		cout << "res_boxes size: " << pred_boxes.size() << endl;
		bs_pred_boxes.emplace_back(pred_boxes);
	}

	return std::move(bs_pred_boxes);
}

std::vector<BoxInfo> YOLOX::PostProcessCPU()
{
	int no = out_shape_.height();
	cudaMemcpy(h_output_ptr, d_output_ptr, out_shape_.count() * sizeof(float), 
		       cudaMemcpyDeviceToHost);

	std::vector<float> output8x8(h_output_ptr, h_output_ptr + 6400 * no);
	std::vector<float> output16x16(h_output_ptr + 6400 * no, h_output_ptr + 8000 * no);
	std::vector<float> output32x32(h_output_ptr + 8000 * no, h_output_ptr + 8400 * no);

	filted_pred_boxes_.clear();
	DecodeAndFiltedBoxes(output8x8, 8, 80, 80, no);
	DecodeAndFiltedBoxes(output16x16, 16, 40, 40, no);
	DecodeAndFiltedBoxes(output32x32, 32, 20, 20, no);

	std::vector<BoxInfo> pred_boxes = NMS();

	return std::move(pred_boxes);
}

std::vector<std::vector<BoxInfo>> YOLOX::PostProcessCPUMutilBs()
{
	int num = out_shape_.num();
	int no = out_shape_.height();
	int hw = out_shape_.channel();

	cudaMemcpy(h_output_ptr, d_output_ptr, out_shape_.count() * sizeof(float), 
		cudaMemcpyDeviceToHost);

	vector<vector<BoxInfo>> bs_pred_boxes;
	int bs = _batch_size;
	for (int i = 0; i < bs; i++)
	{
		float* ptr = h_output_ptr + i * hw * no;
		vector<float> output8x8(ptr, ptr + 6400 * no);
		vector<float> output16x16(ptr + 6400 * no, ptr + 8000 * no);
		vector<float> output32x32(ptr + 8000 * no, ptr + 8400 * no);

		filted_pred_boxes_.clear();
		DecodeAndFiltedBoxes(output8x8, 8, 80, 80, no);
		DecodeAndFiltedBoxes(output16x16, 16, 40, 40, no);
		DecodeAndFiltedBoxes(output32x32, 32, 20, 20, no);

		vector<BoxInfo> pred_boxes = NMS();
		cout << "res box size: " << pred_boxes.size() << endl;
		bs_pred_boxes.emplace_back(pred_boxes);
	}

	return bs_pred_boxes;
}

void YOLOX::DecodeAndFiltedBoxes(std::vector<float>& output,
								 int stride, int height,
								 int width, int channels)
{
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int pos = (row * width + col) * channels;
			float obj_conf = output[pos + 4];
			std::vector<float> cls(output.begin() + pos + 5, 
								   output.begin() + pos + channels);

			float class_conf;
			int class_pred;
			FindMaxConfAndIdx(cls, class_conf, class_pred);

			float score = obj_conf * class_conf;
			if (score < test_conf_)
				continue;

			float cx = (output[pos + 0] + col) * stride;
			float cy = (output[pos + 1] + row) * stride;
			float w = exp(output[pos + 2]) * stride;
			float h = exp(output[pos + 3]) * stride;

			filted_pred_boxes_.emplace_back(BoxInfo(cx - w / 2, cy - h / 2,
				cx + w / 2, cy + h / 2,
				class_conf, score, class_pred));
		}
	}
}

void YOLOX::FindMaxConfAndIdx(const std::vector<float>& vec,
							  float& class_conf, int& class_pred)
{
	float max_val = FLT_MIN;
	int max_idx = -1;
	for (int i = 0; i < vec.size(); i++)
	{
		if (max_val < vec[i])
		{
			max_val = vec[i];
			max_idx = i;
		}
	}
	class_conf = max_val;
	class_pred = max_idx;
}

std::vector<BoxInfo> YOLOX::NMS()
{
	std::vector<BoxInfo> res_boxes;

	//cout << "filted_pred_boxes_: " << filted_pred_boxes_.size() << endl;
	if (filted_pred_boxes_.empty())
		return res_boxes;

	sort(filted_pred_boxes_.begin(), filted_pred_boxes_.end(), compose);

	// 矫正检测框
	RefineBoxes();
	char* removed = (char*)malloc(filted_pred_boxes_.size() * sizeof(char));
	memset(removed, 0, filted_pred_boxes_.size() * sizeof(char));
	for (int i = 0; i < filted_pred_boxes_.size(); i++)
	{
		if (removed[i])
			continue;
		res_boxes.push_back(filted_pred_boxes_[i]);
		for (int j = i + 1; j < filted_pred_boxes_.size(); j++)
		{
			if (filted_pred_boxes_[j].class_idx != filted_pred_boxes_[i].class_idx)
				continue;
			float iou = IOU(filted_pred_boxes_[i], filted_pred_boxes_[j]);
			if (iou >= 0.3)
				removed[j] = 1;
		}
	}

	return std::move(res_boxes);
}

std::vector<BoxInfo> YOLOX::NMSGpu()
{
	std::vector<BoxInfo> res_boxes;
	if (filted_pred_boxes_.empty()) 
		return res_boxes;

	sort(filted_pred_boxes_.begin(), filted_pred_boxes_.end(), compose); 
	// 矫正检测框
	RefineBoxes();
	
	float* h_iou;
	cudaHostAlloc((void**)&h_iou, filted_pred_boxes_.size() * filted_pred_boxes_.size() 
				  * sizeof(float), cudaHostAllocDefault);

	nms(filted_pred_boxes_.data(), filted_pred_boxes_.size(), h_iou);

	char* removed = (char*)malloc(filted_pred_boxes_.size() * sizeof(char));
	memset(removed, 0, filted_pred_boxes_.size() * sizeof(char));
	for (int i = 0; i < filted_pred_boxes_.size(); i++)
	{
		if (removed[i])
			continue;
		res_boxes.push_back(filted_pred_boxes_[i]);
		for (int j = i + 1; j < filted_pred_boxes_.size(); j++)
		{
			if (filted_pred_boxes_[j].class_idx != filted_pred_boxes_[i].class_idx)
				continue;
			float iou = h_iou[i * filted_pred_boxes_.size() + j]; 
			if (iou >= 0.3)
				removed[j] = 1;
		}
	}

	cudaFreeHost(h_iou);
	return std::move(res_boxes);
}

void YOLOX::scale_coords(std::vector<BoxInfo>& pred_boxes, float r)
{
	for (auto& box : pred_boxes)
	{
		box.x1 = box.x1 / r;
		box.y1 = box.y1 / r;
		box.x2 = box.x2 / r;
		box.y2 = box.y2 / r;
	}
}