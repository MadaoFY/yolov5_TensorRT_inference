import os
import time
import torch
import cv2 as cv
import numpy as np

from utils import trt_infer
from utils.trt_infer import load_engine
from utils.utils_detection import yaml_load, image_trans, scale_bboxes, non_max_suppression, Colors, draw_boxes


class yolo_engine_det:
    def __init__(self, engine_dir, catid_labels):
        self.engine = load_engine(engine_dir)
        self.context = self.engine.create_execution_context()
        self.resize = self.engine.get_binding_shape(0)[2:]
        self.colors = self.get_colors_dict(catid_labels)
        self.labels = catid_labels

        # self.context.set_binding_shape(0, [1, 3, self.resize[0], self.resize[1]])
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None

        self.inputs, self.outputs, self.bindings, self.stream = trt_infer.allocate_buffers_v2(self.context)

    @staticmethod
    def get_colors_dict(catid_labels):
        color_dicts = Colors(catid_labels)
        return color_dicts.get_id_and_colors()


    def draw(self, frame):
        x = image_trans(frame, self.resize)
        np.copyto(self.inputs[0].host, x.ravel())
        t1 = time.time()
        pred = trt_infer.do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream
        )
        t2 = time.time()
        fps = int(1.0 / (t2 - t1))
        times = round((t2 - t1) * 1000, 3)
        num_det, boxes, conf, labels = pred
        num_det = num_det[0]
        if num_det > 0:
            # conf = conf[:num_det]
            # labels = labels[:num_det]
            boxes = boxes[:num_det * 4].reshape(-1, 4)
            boxes = scale_bboxes(boxes, frame.shape[:2], self.resize)
            for i in range(num_det):
                frame = draw_boxes(frame, boxes[i], conf[i], labels[i], self.labels, 0.7, self.colors)
        frame = cv.putText(frame, f'fps: {fps}', (10, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                           lineType=cv.LINE_AA, color=(255, 0, 255))
        return frame, times


def main(args):
    times = []
    # 检测物体标签
    catid_labels = yaml_load(args.labels)['labels']
    # 视频源
    vc = cv.VideoCapture(args.video_dir)
    # 载入engine
    yolo_draw = yolo_engine_det(
        args.engine_dir, catid_labels
    )

    # 循环读取视频中的每一帧
    while vc.isOpened():
        ret, frame = vc.read()

        if ret is True:
            frame, t = yolo_draw.draw(frame)
            print(f'{t}ms')
            times.append(t)
            cv.imshow('video', frame)

            if cv.waitKey(30) & 0xFF == 27:
                break
        else:
            break
    print(np.mean(times))
    vc.release()
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
    parser.add_argument('--engine_dir', type=str, default='./models_trt/yolov5m.engine',
                        help='engine path')


    args = parser.parse_args()
    print(args)

    main(args)

