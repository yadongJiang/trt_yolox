#include "trt_onnx_base.h"
#include <fstream>
#include <sstream>
#include <assert.h>

TRTOnnxBase::TRTOnnxBase(const OnnxDynamicNetInitParamV1& params) : params_(params)
{
	std::cout << "start init..." << std::endl;
	cudaSetDevice(params.gpu_id);
	_max_batch_size = params.max_batch_size;
	/*in_shape_.set_num(_max_batch_size);*/
	in_shape_.Reshape(_max_batch_size, 3, 640, 640);
	//out_shape_.set_num(_max_batch_size);
	out_shape_.Reshape(_max_batch_size, 8400, 5 + params.num_classes, 1);

	cudaStreamCreate(&stream_);

	if (!LoadGieStreamBuildContext(params_.rt_stream_path + params_.rt_model_name))
	{
		LoadOnnxModel(params_.onnx_model);
		SaveRtModel(params_.rt_stream_path + params_.rt_model_name);
	}
}

TRTOnnxBase::~TRTOnnxBase()
{
	cudaStreamSynchronize(stream_);
	cudaStreamDestroy(stream_);
	if (h_input_ptr != NULL)
		cudaFreeHost(h_input_ptr);
	if (h_output_ptr != NULL)
		cudaFreeHost(h_output_ptr);
	if (d_input_ptr != NULL)
		cudaFree(d_input_ptr);
	if (d_output_ptr != NULL)
		cudaFree(d_output_ptr);
}

bool TRTOnnxBase::LoadGieStreamBuildContext(const std::string& gie_file)
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

void TRTOnnxBase::LoadOnnxModel(const std::string& onnx_file)
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

void TRTOnnxBase::SaveRtModel(const std::string& path)
{
	std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
	outfile.write((const char*)gie_model_stream_->data(), gie_model_stream_->size());
	outfile.close();
}

void TRTOnnxBase::deserializeCudaEngine(const void* blob, std::size_t size) 
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

void TRTOnnxBase::mallocInputOutput()
{
	cudaHostAlloc((void**)&h_input_ptr, in_shape_.count() * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_input_ptr, in_shape_.count() * sizeof(float));

	cudaHostAlloc((void**)&h_output_ptr, out_shape_.count() * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_output_ptr, out_shape_.count() * sizeof(float));

	buffer_.push_back(d_input_ptr);
	buffer_.push_back(d_output_ptr);
}

bool TRTOnnxBase::CheckFileExist(const std::string& path)
{
	std::fstream check_file(path);
	bool found = check_file.is_open();
	return found;
}

void TRTOnnxBase::Forward()
{
	nvinfer1::Dims4 input_dims{ _batch_size , in_shape_.channel(),
								in_shape_.height(), in_shape_.width() };
	context_->setBindingDimensions(0, input_dims);
	context_->enqueueV2(buffer_.data(), stream_, nullptr);

	cudaStreamSynchronize(stream_);
}