# 使用TensorRT实现部署YOLOX (torch->onnx->tensorrt)

## 环境依赖

1. Opencv3.1
2. TensorRT7.2
3. Cuda10.2

## 使用方法
具体使用方法可以参照main.cpp中的main()函数。如果初次调用，需要指定onnx模型的路径、生成的trt模型的保存路径以及保存的模型名。
初次调用之后会生成.engine的trt模型，并保存到指定位置，之后再调用，则直接调用.engine模型。
```
    OnnxDynamicNetInitParamV1 params;
    # onnx模型路径
    params.onnx_model = "./YOLOX-main/YOLOX_outputs/yolox_s/onnx/yolox_s.onnx";
    # 最大的batchsize, 可根据自己的模型需求设置
    params.max_batch_size = 2;
    # 生成的trt模型的保存路径，可按照自身需求设置
    params.rt_stream_path = "./"
    # 生成的trt模型名
    params.rt_model_name = "detection.engine";
    # 设置检测目标类别数
    params.num_classes = 1;

    # 构造YOLOX类对象
    YOLOX yolox(params);
    # infer
    vector<BoxInfo> pred_boxes = yolox.Extract(img);
```