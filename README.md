# yolov5_TensorRT_inference
记录yolov5的TensorRT量化及推理代码，经实测可运行于Jetson平台。可实现yolov5、yolov7、yolox模型的量化(fp32, fp16, int8)及推理，并且可将yolov5s、yolov7tiny、yoloxs这些的小模型部署在Jetson nano 4g上用于摄像头的检测。  
![image](https://github.com/MadaoFY/yolov5_TensorRT_inference/raw/main/doc/yolov5s_det.png)  

本人使用的TensrRT版本为8.4.3.1，为保证成功运行，你的TensorRT大版本最好在8.4。

以下将使用yolov5s模型演示如何量化及用于视频的推理。

## 使用演示

### 模型准备
使用yolov5官方提供的coco训练模型，已导出为onnx。这里使用voc2012作为验证集，仅用来演示，你可以下载coco数据集作为你的验证集。

yolov5s.onnx：https://pan.baidu.com/s/1wKgRgdjk12YxDNqcYuvu9w  
提取码: upaj

voc2012：https://pan.baidu.com/s/1ZrT_s-cFqt6sQlHDIZUr4w  
提取码: nk6s

视频源：https://pan.baidu.com/s/1WHPwu9nmtEmJwZuukTGitQ  
提取码: 4ja6

### 量化(onnx2trt.py)
你需要从yolov5、yolov7、yolox的官方库导出相应onnx模型，从第三方实现的库中导出的yolo onnx模型不保证适用，注意导出的onnx不包含nms部分。默认将onnx模型放置于models_onnx文件夹，导出的trt模型可保存于models_trt文件夹。如果你想使用int8量化，你需要从训练集中准备至少500张图片作为校准集，图片放置于calibration文件夹。```--onnx_dir```onnx模型路径，```--engine_dir```trt模型的保存路径，```--int8```是否使用int8量化，```--imgs_dir```校准集路径。
```bash
python onnx2trt.py  --onnx_dir ./models_onnx/yolov5s.onnx --engine_dir ./models_trt/yolov5s.engine --int8 True --imgs_dir ./calibration
```
更多参数可以在脚本中查看。


### 视频推理(video_detect_yolov5.py)
你需要准备一个模型输出类别的labels文件，具体可参考仓库的labels_coco.yaml文件。本演示中用到模型为coco训练的yolov5s模型，所以需要用到相对应的coco类别。如果你使用的是yolov5、yolov7模型，运行video_detect_yolov5.py脚本，yolox模型运行video_detect_yolox.py脚本。以yolov5s.engine推理为例。```--video_dir```视频源路径，```--engine_dir```trt模型路径，```--labels```模型labels文件。
```bash
python video_detect_yolov5.py  --video_dir ./sample_1080p_h265.mp4 --engine_dir ./models_trt/yolov5s.engine --labels ./labels_coco.yaml
```
更多参数可以在脚本中查看。

## 其他相关
可能TensoRT安装是最消耗时间的事情、、、  
TensoRT：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing  
https://developer.nvidia.com/tensorrt

yolov5：https://github.com/ultralytics/yolov5

yolov7：https://github.com/WongKinYiu/yolov7

yolox：https://github.com/Megvii-BaseDetection/YOLOX
