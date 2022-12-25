# yolov5_TensorRT_inference
记录yolov5的TensorRT量化及推理代码，经实测可运行于Jetson平台。可实现yolov5、yolov7、yolox模型的量化(fp32, fp16, int8)及推理，并且可将yolov5s、yolov7tiny、yoloxs这些的小模型部署在Jetson nano 4g上用于摄像头的检测。

本人使用的TensrRT版本为8.4.3.1，为保证成功运行，你的TensorRT大版本最好在8.4。

以下将用yolov5s模型演示如何量化及用于视频的推理。

### 1、量化(python onnx2trt.py)
你需要从yolov5、yolov7、yolox的官方库导出相应onnx模型，从第三方实现的库中导出的yolo onnx模型不保证适用，注意导出的onnx不包含nms部分。默认将onnx模型放置于models_onnx文件夹，导出的engine模型可保存于models_trt文件夹。如果你想使用int8量化，你需要从训练集中准备至少500张图片作为校准集，图片放置于calibration文件夹。
```bash
python onnx2trt.py  --onnx_dir ./models_onnx/yolov5s.onnx --engine_dir ./models_trt/yolov5s.engine --int8 True --imgs_dir ./calibration
```
更多参数可以在脚本中查看。


### 2、视频推理(video_detect_yolov5.py)
你需要准备一个模型输出类别的labels文件，具体可参考仓库的labels_coco.yaml文件。本演示中用到模型为coco训练的yolov5s模型，所以需要用到相对应的coco类别。如果你使用的是yolov5、yolov7模型，运行video_detect_yolov5.py脚本，yolox模型运行video_detect_yolox.py脚本。你至少需要输入--video_dir、--engine_dir、--labels参数，以yolov5s.engine推理为例，运行代码如下。
```bash
python video_detect_yolov5.py  --video_dir ./sample_1080p_h265.mp4 --engine_dir ./models_trt/yolov5s.engine --labels ./labels_coco.yaml
```
更多参数可以在脚本中查看。
