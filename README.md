# yolov5_TensorRT_inference
记录yolov5的TensorRT量化(fp16, int8)及推理代码。经实测可运行于Jetson平台，可将yolov5s、yolov8s这类的小模型部署在Jetson nano 4g上用于摄像头的检测。  
模型支持：  
yolov5  
yolov7  
yolov8  
yolox(不可在生成的engine中添加nms模块)

<div align=center>
<img src="https://github.com/MadaoFY/yolov5_TensorRT_inference/blob/main/doc/yolov5s_det.png">
</div>

温馨提示：本人使用的TensrRT版本为8.4.3.1，为保证成功运行，你的TensorRT大版本最好在8.4。具体环境依赖请参考```requirements.txt```

项目文件如下：
```bash
|-yolov5_TensorRT_inference
    |-calibration       # 默认情况下用于存放int8量化校准集的文件夹
    |-doc
    |-models_onnx       # 默认情况下用于存放onnx模型的文件夹
    |-models_trt        # 默认情况下用于存放量化后生成的trt模型的文件夹
    |-utils
    |-Benchmark.py      # 测试trt模型速度的脚本
    |-labels_coco.yaml  # coco数据集类别标签
    |-labels_voc.yaml   # voc数据集类别标签
    |-onnx2trt.py       # onnx模型转engine的脚本，已添加EfficientNMS算子的支持
    |-yolo_detect_v1.py    # yolov5的视频检测脚本
    |-yolo_detect_v2.py    # yolov7的视频检测脚本，该脚本使用的trt模型添加了EfficientNMS算子
    |-video_detect_yolovx.py    # yolovx的视频检测脚本
```

以下将使用yolov5s模型演示如何量化及用于视频的推理。
## 数据准备
使用yolov5官方提供的coco训练模型，已导出为onnx。这里使用voc2012作为校准集，仅用来演示，你可以下载coco数据集作为你的校准集。

yolov5s.onnx：https://pan.baidu.com/s/1eYaU3ndVpwexL4k6goxjHg  
提取码: sduf   

voc2012：https://pan.baidu.com/s/1rICWiczIv_GyrYIrEj1p3Q  
提取码: 4pgx

视频源：https://pan.baidu.com/s/1HBIjz6019vn9qfoKPIuV2A  
提取码: fbfh

## 量化(onnx2trt.py)
你需要从yolov5、yolov7、yolox的官方库导出相应onnx模型，从第三方实现的库中导出的yolo onnx模型不保证适用，注意导出的onnx不包含nms部分。默认将onnx模型放置于models_onnx文件夹，导出的trt模型可保存于models_trt文件夹。如果你想使用int8量化，你需要从训练集中准备至少500张图片作为校准集，图片放置于calibration文件夹。

```shell
python onnx2trt.py  --onnx_dir ./models_onnx/yolov5s.onnx --engine_dir ./models_trt/yolov5s.engine --int8 True --imgs_dir ./calibration
```  
参数说明:  
- ```--onnx_dir``` onnx模型路径
- ```--engine_dir``` trt模型的保存路径
- ```--min_shape``` 最小的shape
- ```--opt_shape``` 优化的shape
- ```--max_shape``` 最大的shape
- ```--fp16``` 是否使用fp16量化
- ```--int8``` 是否使用int8量化
- ```--imgs_dir``` 校准集路径
- ```--n_iteration``` int8量化校准轮次
- ```--cache_file``` 是否生成cache
- ```--yolov8_head``` 是否为yolov8的检测头(注意,yolov8的输出与yolov5不一样)
- ```--add_nms``` 添加EfficientNMS算子
- ```--conf_thres``` nms的置信度设置
- ```--iou_thres``` nms的iou设置
- ```--max_det``` nms输出的最大检测数量

更详细参数说明可以在脚本中查看。

## 视频推理
### 1.不带EfficientNMS算子的推理脚本(yolo_detect_v1.py)  
你需要准备一个模型输出类别的labels文件，具体可参考仓库的labels_coco.yaml文件。本演示中用到模型为coco训练的yolov5s模型，所以需要用到相对应的coco类别。如果你使用的是yolov5、yolov7模型，运行video_detect_yolov5.py脚本，yolox模型运行video_detect_yolox.py脚本。以yolov5s.engine推理为例。
```shell
python yolo_detect_v1.py  --video_dir ./sample_1080p_h265.mp4 --engine_dir ./models_trt/yolov5s.engine --labels ./labels_coco.yaml
```

- ```--video_dir``` 视频源路径
- ```--engine_dir``` trt模型路径
- ```--labels``` 模型labels文件
- ```--yolov8_head``` 是否为yolov8的检测头
- ```--conf_thres``` nms的置信度设置
- ```--iou_thres``` nms的iou设置
- ```--max_det``` nms输出的最大检测数量

### 2.带EfficientNMS算子的推理脚本(yolo_detect_v2.py)  
yolo_detect_v2.py脚本里的所使用trt模型已添加EfficientNMS算子，所以无需在对nms参数进行设置    
```shell
python yolo_detect_v2.py  --video_dir ./sample_1080p_h265.mp4 --engine_dir ./models_trt/yolov7_nms.engine --labels ./labels_coco.yaml
```

- ```--video_dir``` 视频源路径
- ```--engine_dir``` trt模型路径
- ```--labels``` 模型labels文件


## 其他相关
可能TensoRT安装是最消耗时间的事情、、、  
TensoRT：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing  
https://developer.nvidia.com/tensorrt

Trt_sample: https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook

yolox：https://github.com/Megvii-BaseDetection/YOLOX  
yolov5：https://github.com/ultralytics/yolov5  
yolov7：https://github.com/WongKinYiu/yolov7  
yolov8: https://github.com/ultralytics/ultralytics


