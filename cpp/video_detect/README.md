# video_detect
用于视频目标检测的代码。  
用cmake编译后，运行yolo_detect。

```shell
yolo_detect --engine_dir=./yolov5s.engine --video_dir=./sample_1080p_h265.mp4 --labels=./labels_coco.yaml
```  

参数说明:
- ```--engine_dir``` trt模型的保存路径
- ```--video_dir``` 视频源路径
- ```--labels``` 模型labels文件
- ```--conf_thres``` nms的置信度设置
- ```--iou_thres``` nms的iou设置
- ```--max_det``` nms输出的最大检测数量

更详细参数说明可以在main.cpp中查看。
