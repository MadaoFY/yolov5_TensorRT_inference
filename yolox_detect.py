import os
import time
import torch
import cv2 as cv
import numpy as np

from utils import trt_infer
from utils.trt_infer import load_engine
from utils.utils_detection import yaml_load, image_trans, scale_bboxes, non_max_suppression_torch, yolox_postprocess, \
    Colors, draw_boxes


class yolox_engine_det:
    def __init__(self, engine_dir, catid_labels):
        self.engine = load_engine(engine_dir)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.context = self.engine.create_execution_context()
        self.resize = self.engine.get_binding_shape(0)[2:]
        self.colors = self.get_colors_dict(catid_labels)
        self.labels = catid_labels
        self.nms = non_max_suppression_torch

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


    def draw(self, frame, conf=0.25, iou=0.45, max_det=200):
        x = image_trans(frame, self.resize)
        np.copyto(self.inputs[0].host, x.ravel())
        t1 = time.time()
        pred = trt_infer.do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream
        )
        pred = pred[0].reshape(self.context.get_binding_shape(1))
        pred = yolox_postprocess(pred, self.resize, p6=False)
        pred = torch.from_numpy(pred).to(self.device)
        pred = self.nms(pred, False, conf_thres=conf, iou_thres=iou, agnostic=False, max_det=max_det)[0]
        t2 = time.time()
        fps = int(1.0 / (t2 - t1))
        pred = scale_bboxes(pred, frame.shape[:2], self.resize)
        pred = pred.cpu().numpy()
        for i in pred:
            # pred: x1, y1, x2, y2, conf, labels
            # bbox = tuple(i[:4].astype('int'))
            # frame = cv.rectangle(frame, bbox[:2], bbox[2:], thickness=2, lineType=cv.LINE_AA,
            #                      color=self.colors[i[-1]]
            #                      )
            # frame = cv.putText(frame, f'{self.labels[i[-1]]}:{i[-2]:.2f}', (bbox[0] + 5, bbox[1] + 30),
            #                    fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1, lineType=cv.LINE_AA,
            #                    color = (210, 105, 30)
            #                    )
            frame = draw_boxes(frame, i[:4], i[4], i[5], self.labels, 0.7, self.colors)
        frame = cv.putText(frame, f'fps: {fps}', (10, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                           lineType=cv.LINE_AA, color=(255, 0, 255))
        return frame


def main(args):
    # 检测物体标签
    catid_labels = yaml_load(args.labels)['labels']
    # 视频源
    vc = cv.VideoCapture(args.video_dir)
    # 载入engine
    yolo_draw = yolox_engine_det(args.engine_dir, catid_labels)

    # 循环读取视频中的每一帧
    while vc.isOpened():
        ret, frame = vc.read()

        if ret is True:
            frame = yolo_draw.draw(
                frame, conf=args.conf_thres, iou=args.iou_thres, max_det=args.max_det
            )
            cv.imshow('video', frame)

            if cv.waitKey(30) & 0xFF == 27:
                break
        else:
            break

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
    parser.add_argument('--engine_dir', type=str, default='./models_trt/yolox_s.engine',
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
