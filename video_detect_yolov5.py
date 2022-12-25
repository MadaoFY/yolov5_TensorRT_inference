import os
import time
import torch
import cv2 as cv
import numpy as np
import tensorrt as trt

import trt_infer

from utils_detection import yaml_load, letterbox_image, scale_bboxes, non_max_suppression, Colors, draw_boxes


def load_engine(engine_path):
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class yolov5_engine_det():
    def __init__(self, engine_dir, catid_labels, conf=0.25, iou=0.45, max_det=300):
        self.engine = self.__load_engine(engine_dir)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.context = self.engine.create_execution_context()
        self.resize = self.engine.get_binding_shape(0)[2:]
        self.colors = self.__get_colors_dict(catid_labels)
        self.labels = catid_labels
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.nms = non_max_suppression

        # self.context.set_binding_shape(0, [1, 3, self.resize[0], self.resize[1]])

    def __load_engine(self, engine_dir):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_dir, 'rb') as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __get_colors_dict(self, catid_labels):
        color_dicts = Colors(catid_labels)
        return color_dicts.get_id_and_colors()


    def draw(self, frame):
        x = trans(frame, self.resize)
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers_v2(self.context)
        inputs[0].host = np.ascontiguousarray(x.flatten())
        t1 = time.time()
        pred = trt_infer.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = pred[0].reshape(self.context.get_binding_shape(1))
        pred = torch.from_numpy(pred).to(self.device)
        pred = self.nms(pred, conf_thres=self.conf, iou_thres=self.iou, agnostic=False, max_det=self.max_det)[0]
        t2 = time.time()
        fps = int(1.0 / (t2 - t1))
        times = round((t2 - t1) * 1000, 3)
        pred = scale_bboxes(pred, frame.shape[:2], self.resize)
        pred = pred.cpu().numpy()
        for i in pred:
            # pred: x1, y1, x2, y2, conf, labels
            frame = draw_boxes(frame, i[:4], i[4], i[5], self.labels, 1, self.colors)
            # bbox = tuple(i[:4].astype('int'))
            # frame = cv.rectangle(frame, bbox[:2], bbox[2:], thickness=2, lineType=cv.LINE_AA,
            #                      color=self.colors[i[-1]]
            #                      )
            # frame = cv.putText(frame, f'{self.labels[i[-1]]}:{i[-2]:.2f}', (bbox[0] + 5, bbox[1] + 30),
            #                    fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1, lineType=cv.LINE_AA,
            #                    color = (210, 105, 30)
            #                    )
        frame = cv.putText(frame, f'fps: {fps}', (10, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                           lineType=cv.LINE_AA, color=(255, 0, 255))
        return frame, times


def trans(img, size):
    img_new = letterbox_image(img)
    img_new = cv.resize(img_new, size, interpolation=cv.INTER_LINEAR)
    img_new = img_new.transpose(2, 0, 1)
    img_new = np.expand_dims(img_new, 0)
    img_new = np.float32(img_new) / 255.0
    return img_new


def main(args):
    times = []
    # 检测物体标签
    catid_labels = yaml_load(args.labels)['labels']
    # 视频源
    vc = cv.VideoCapture(args.video_dir)
    # 载入engine
    yolo_draw = yolov5_engine_det(
        args.engine_dir, catid_labels, conf=args.conf_thres, iou=args.iou_thres, max_det=args.max_det
    )

    # 循环读取视频中的每一帧
    while vc.isOpened():
        ret, frame = vc.read()

        if ret is True:
            frame, t = yolo_draw.draw(frame)
            print(f'{t}ms')
            times.append(t)
            cv.imshow('video', frame)

            if cv.waitKey(int(1000 / vc.get(cv.CAP_PROP_FPS))) & 0xFF == 27:
                break
        else:
            break
    print(np.mean(times))
    vc.release
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    # 目标类别标签
    parser.add_argument('--labels', type=str, default='./labels_coco.yaml', help='obj labels')
    # video地址
    parser.add_argument('--video_dir', type=str, default='sample_1080p_h265.mp4',
                        help='video path')
    # engine模型地址
    parser.add_argument('--engine_dir', type=str, default='./models_trt/yolov5s_512.engine',
                        help='engine path')
    # 只有得分大于置信度的预测框会被保留下来
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    # 非极大抑制所用到的nms_iou大小
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    # 目标框数量限制
    parser.add_argument('--max_det', type=int, default=200, help='maximum detections per image')

    args = parser.parse_args()
    print(args)

    main(args)
