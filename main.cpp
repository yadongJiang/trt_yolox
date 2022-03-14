#include "yolox.h"

int main()
{
	OnnxDynamicNetInitParamV1 params;
	params.onnx_model = "./YOLOX_outputs/yolox_s/onnx/yolox_s.onnx";
	params.max_batch_size = 2;
	params.rt_model_name = "detection.engine";

	cv::Mat img = cv::imread("./YOLOX-main/assets/image--01c1276cb6e346e893cc3d3ce6c6b9df.jpg");

	YOLOX yolox(params);
	std::vector<cv::Mat> imgs{ img, img };
	std::vector<BoxInfo> pred_boxes = yolox.Extract(img);
	for (auto& box : pred_boxes)
		cv::rectangle(img, cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1), cv::Scalar(0, 0, 255), 2);
	cv::imwrite("res.jpg", img);

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