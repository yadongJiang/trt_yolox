#include "yolox.h"

int main()
{
	OnnxDynamicNetInitParamV1 params;
	params.onnx_model = "./YOLOX-main/YOLOX_outputs/yolox_s/onnx/yolox_s.onnx";
	params.max_batch_size = 2; // 最大的batchsize, 可根据自己的模型需求设置
	params.rt_model_name = "detection.engine";

	cv::Mat img = cv::imread("./YOLOX-main/assets/image--01c1276cb6e346e893cc3d3ce6c6b9df.jpg");
	//cv::resize(img, img, cv::Size(img.cols / 5, img.rows / 5), cv::INTER_LINEAR);

	YOLOX yolox(params);
	std::vector<cv::Mat> imgs{ img, img };
	float total_time = 0.0;
	yolox.Extract(img); // for warm gpu
	for (int i = 0; i < 100; i++)
	{
		clock_t start, end;
		start = clock();
		std::vector<BoxInfo> pred_boxes = yolox.Extract(img);
		end = clock();
		std::cout << "cost time: " << end - start << std::endl;
		total_time += (end - start);
		/*for (auto& box : pred_boxes)
			cv::rectangle(img, cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1), cv::Scalar(0, 0, 255), 2);
		cv::imwrite("res" + std::to_string(i)+".jpg", img); */
	}
	std::cout << " ave time: " << total_time / 100 << std::endl;

	/*std::vector<std::vector<BoxInfo>> bs_pred_boxes = yolox.Extract(imgs);
	for (int i=0; i<bs_pred_boxes.size(); i++)
	{
		std::vector<BoxInfo> &pred_boxes = bs_pred_boxes[i];
		for (auto& box : pred_boxes)
			cv::rectangle(imgs[i], cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1), cv::Scalar(0, 0, 255), 2);

		cv::imwrite("img" + std::to_string(i) + ".jpg", imgs[i]);
	}*/

	cv::imshow("img", img);
	cv::waitKey();
}