import os
import time
import torch
import cv2 as cv
import numpy as np
import tensorrt as trt

from utils import trt_infer
from utils.utils_detection import yaml_load, letterbox_image, scale_bboxes, non_max_suppression, Colors, draw_boxes


def load_engine(engine_path):
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class yolov7_engine_det:
    def __init__(self, engine_dir, catid_labels):
        self.engine = load_engine(engine_dir)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.context = self.engine.create_execution_context()
        self.resize = self.engine.get_binding_shape(0)[2:]
        self.colors = self.get_colors_dict(catid_labels)
        self.labels = catid_labels

        # self.context.set_binding_shape(0, [1, 3, self.resize[0], self.resize[1]])

    @staticmethod
    def get_colors_dict(catid_labels):
        color_dicts = Colors(catid_labels)
        return color_dicts.get_id_and_colors()


    def draw(self, frame):
        x = trans(frame, self.resize)
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers_v2(self.context)
        inputs[0].host = np.ascontiguousarray(x.flatten())
        t1 = time.time()
        pred = trt_infer.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        t2 = time.time()
        fps = int(1.0 / (t2 - t1))
        times = round((t2 - t1) * 1000, 3)
        num_det, boxes, conf, labels = pred
        num_det = num_det[0]
        if num_det > 0:
            pred = np.concatenate((
                boxes[:num_det * 4].reshape(-1, 4), np.expand_dims(conf[:num_det], 1), np.expand_dims(labels[:num_det], 1)
            ), axis=-1)
            pred = scale_bboxes(pred, frame.shape[:2], self.resize)
            for i in pred:
                # pred: x1, y1, x2, y2, conf, labels
                frame = draw_boxes(frame, i[:4], i[4], i[5], self.labels, 1, self.colors)
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
    # 检测物体标签
    catid_labels = yaml_load(args.labels)['labels']
    # 载入engine
    yolo_det = yolov7_engine_det(
        args.engine_dir, catid_labels
    )
    # 视频源
    vc = cv.VideoCapture(args.video_dir)

    times = []
    # 循环读取视频中的每一帧
    while vc.isOpened():
        ret, frame = vc.read()

        if ret is True:
            frame, t = yolo_det.draw(frame)
            print(f'{t}ms')
            times.append(t)
            cv.imshow('video', frame)

            if cv.waitKey(int(1000 / vc.get(cv.CAP_PROP_FPS))) & 0xFF == 27:
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
    parser.add_argument('--engine_dir', type=str, default='./models_trt/yolov7_nms.engine',
                        help='engine path')


    args = parser.parse_args()
    print(args)

    main(args)

