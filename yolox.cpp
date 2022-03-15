#include "yolox.h"
#include <fstream>
#include <sstream>
#include <assert.h>
#include "gpu_func.cuh"

YOLOX::YOLOX(const OnnxDynamicNetInitParamV1& params) : params_(params)
{
	std::cout << "start init..." << std::endl;
	cudaSetDevice(params.gpu_id);
	_max_batch_size = params.max_batch_size;
	in_shape_.set_num(_max_batch_size);
	out_shape_.set_num(_max_batch_size);

	cudaStreamCreate(&stream_);

	if (!LoadGieStreamBuildContext(params_.rt_stream_path + params_.rt_model_name))
	{
		LoadOnnxModel(params_.onnx_model);
		SaveRtModel(params_.rt_stream_path + params_.rt_model_name);
	}

	transform_ = new ComposeMatLambda({
		YoloXResize(crop_size_),
		MatToFloat(),
	});
}

bool YOLOX::CheckFileExist(const std::string& path)
{
	std::fstream check_file(path);
	bool found = check_file.is_open();
	return found;
}

void YOLOX::LoadOnnxModel(const std::string& onnx_file)
{
	if (!CheckFileExist(onnx_file))
	{
		std::cerr << "onnx file is not found " << onnx_file << std::endl;
		exit(0);
	}

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger_);
	assert(builder != nullptr);

	// 创建network
	const auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx解析器
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger_);
	assert(parser->parseFromFile(onnx_file.c_str(), 2));

	nvinfer1::IBuilderConfig* build_config = builder->createBuilderConfig();
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
	nvinfer1::ITensor* input = network->getInput(0);
	std::cout << "********************* : " << input->getName() << std::endl;
	nvinfer1::Dims dims = input->getDimensions();
	std::cout << "batchsize: " << dims.d[0] << " channels: " << dims.d[1] << " height: " << dims.d[2] << " width: " << dims.d[3] << std::endl;

	{
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, 
								nvinfer1::Dims4{ 1, dims.d[1], 640, 640 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, 
								nvinfer1::Dims4{ 1, dims.d[1], 640, 640 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, 
								nvinfer1::Dims4{ params_.max_batch_size, dims.d[1], 640, 640 });
		build_config->addOptimizationProfile(profile);
	}

	build_config->setMaxWorkspaceSize(1 << 30);

	if (params_.use_fp16)
		params_.use_fp16 = builder->platformHasFastFp16();
	if (params_.use_fp16)
	{
		builder->setHalf2Mode(true);
		std::cout << "useFP16	 " << params_.use_fp16 << std::endl;
	}
	else
		std::cout << "Using GPU FP32 !" << std::endl;

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *build_config);
	assert(engine != nullptr);

	gie_model_stream_ = engine->serialize();

	parser->destroy();
	engine->destroy();
	builder->destroy();
	network->destroy();

	deserializeCudaEngine(gie_model_stream_->data(), gie_model_stream_->size());
}

void YOLOX::deserializeCudaEngine(const void* blob, std::size_t size)
{
	// 创建运行时
	runtime_ = nvinfer1::createInferRuntime(logger_);
	assert(runtime_ != nullptr);
	
	engine_ = runtime_->deserializeCudaEngine(blob, size, nullptr);
	assert(engine_ != nullptr);

	context_ = engine_->createExecutionContext();
	assert(context_ != nullptr);

	mallocInputOutput();
}

void YOLOX::SaveRtModel(const std::string& path)
{
	std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
	outfile.write((const char*)gie_model_stream_->data(), gie_model_stream_->size());
	outfile.close();
}

void YOLOX::mallocInputOutput()
{
	cudaHostAlloc((void**)&h_input_ptr, in_shape_.count() * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_input_ptr, in_shape_.count() * sizeof(float));

	cudaHostAlloc((void**)&h_output_ptr, out_shape_.count() * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_output_ptr, out_shape_.count() * sizeof(float));

	buffer_.push_back(d_input_ptr);
	buffer_.push_back(d_output_ptr);
}

bool YOLOX::LoadGieStreamBuildContext(const std::string& gie_file)
{
	std::cout << "read engine file!!!" << std::endl;
	std::ifstream fgie(gie_file, std::ios_base::in | std::ios_base::binary);
	if (!fgie)
		return false;

	std::stringstream buffer;
	buffer << fgie.rdbuf();

	std::string stream_model(buffer.str());
	deserializeCudaEngine(stream_model.data(), stream_model.size());

	return true;
}

// ================================== Inference ==================================

std::vector<BoxInfo> YOLOX::Extract(const cv::Mat& img)
{
	if (img.empty())
		return {};

	set_batch_size(1);

	ProPrecessCPU(img);
	Forward();
	// 后处理cpu代码
	/*std::vector<BoxInfo> pred_boxes = PostProcessCPU();*/
	// 后处理gpu代码
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

	ProPrecessCPU(imgs);
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

void YOLOX::Forward()
{
	cudaMemcpy(d_input_ptr, h_input_ptr, in_shape_.count() * sizeof(float), 
			   cudaMemcpyHostToDevice);

	nvinfer1::Dims4 input_dims{ _batch_size , in_shape_.channel(), 
								in_shape_.height(), in_shape_.width()};
	context_->setBindingDimensions(0, input_dims);
	context_->enqueueV2(buffer_.data(), stream_, nullptr);

	cudaStreamSynchronize(stream_);
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

std::vector<BoxInfo> YOLOX::NUMGpu()
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