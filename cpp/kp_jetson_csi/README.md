# kp_jetson_csi
在jetson nano 4g上使用yolov5和hrnet进行摄像头人体关键点检测。  
我对hrnet进行了轻量化改造，使其能在算力有限的平台上运行。替换上mobilenetv2的backbone后用coco2017数据集进行了训练，可满足单目标的人体关键点检测需求。
后续有时间可能会更新关键点检测模型，当然如果没时间魔改出更快更准的模型的话就算了...

## 数据准备
相比于目标检测，这里需要多提供一个关键点检测的engine和关键点链接信息。  
你可以使用我提供的以下两个onnx模型，在运行的设备上生成engine。或者自己训练一个专门用于检测人的yolo模型，和一个用于关键点检测的hrnet模型。

yolov5s_person.onnx：https://pan.baidu.com/s/1mgbFLOENiIaTmfsyc2RtVw  
提取码：qei0   

Myhrnet.onnx：https://pan.baidu.com/s/1rIR_CjOuu6qzaWsoirfP3A  
提取码：43dw

points_link.yaml文件里记录的是关键点的链接信息，用于绘图。

用cmake编译后，运行yolo_detect。

```shell
yolo_detect --det_engine_dir=./yolov5s_person.engine --kp_engine_dir=./Myhrnet.engine --labels=./labels_det.yaml --pointlinker=./points_link.yaml
```  

参数说明:
- ```--det_engine_dir``` 目标检测trt模型的保存路径
- ```--kp_engine_dir``` 关键点检测trt模型的保存路径
- ```--labels``` 模型labels的yaml文件
- ```--pointlinker``` 关键点链接的yaml文件
- ```--conf_thres``` nms的置信度设置
- ```--iou_thres``` nms的iou设置
- ```--max_det``` 输出的最大检测数量
- ```--skip``` 隔帧检测帧数

更详细参数说明可以在csi_kp_detect.cpp中查看。
