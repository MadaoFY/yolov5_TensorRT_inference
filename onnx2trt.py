import os
import time
import numpy as np
import tensorrt as trt


from utils import calibrator

__all__ = [
    'build_engine',
    'onnx2trt'
]


def AddEfficientNMSPlugin(conf_thres=0.25, iou_thres=0.45, max_det=200, box_coding=1):
    """
    添加efficientNMS

    score_threshold: score_thresh
    iou_threshold: iou_thresh
    max_output_boxes: detections_per_img
    box_coding: 0->[x1, y1, x2, y2], 1->[x, y, w, h]
    """
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == "EfficientNMS_TRT":
            print(f'Succeeded finding {c.name}')
            parameter = [
                trt.PluginField("score_threshold", np.float32(conf_thres), trt.PluginFieldType.FLOAT32),
                trt.PluginField("iou_threshold", np.float32(iou_thres), trt.PluginFieldType.FLOAT32),
                trt.PluginField("max_output_boxes", np.int32(max_det), trt.PluginFieldType.INT32),
                trt.PluginField("background_class", np.int32(-1), trt.PluginFieldType.INT32),  # background_class: -1, no background class
                trt.PluginField("score_activation", np.int32(0), trt.PluginFieldType.INT32),  # score_activation: 0->False, 1->True
                trt.PluginField("box_coding", np.int32(box_coding), trt.PluginFieldType.INT32)
            ]
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameter))
    return None


def build_engine(
        onnx_file, model_engine, min_shape, opt_shape, max_shape,
        fp16=False, int8=False, imgs_dir=None, imgs_list=None, n_iteration=128, cache_file=None,
        add_nms=False, conf_thres=0.25, iou_thres=0.45, max_det=200, box_coding=1
):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

    # Parse model file
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_file):
        print("ONNX file is not exists!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        else:
            print("Succeeded parsing .onnx file!")

    # 添加nms算子
    if add_nms:
        """
        对原输出进行预处理，拆分成 目标框数据 和 类别置信度数据 两个矩阵，背景置信度要与类别置信度相乘。
        [1, 8500, 4 + 1 + 80] ——> [1, 8500, 4] + [1, 8500, 1 + 80] ——> [1, 8500, 4] + [1, 8500, 80]
        """
        print('Add EfficientNMS_TRT!')
        outputTensor = network.get_output(0)
        print(f'{outputTensor.name} shape:{outputTensor.shape}')
        bs, num_boxes, det_res = outputTensor.shape
        network.unmark_output(outputTensor)
        shapes = [bs, num_boxes, 4]
        boxes = network.add_slice(outputTensor, (0, 0, 0), shapes, (1, 1, 1))
        shapes[-1] = 1
        scores = network.add_slice(outputTensor, (0, 0, 4), shapes, (1, 1, 1))
        shapes[-1] = det_res - 5
        obj = network.add_slice(outputTensor, (0, 0, 5), shapes, (1, 1, 1))
        obj = network.add_elementwise(
            scores.get_output(0), obj.get_output(0), trt.ElementWiseOperation.PROD
        )
        """
        添加efficientNMS
        """
        nms = AddEfficientNMSPlugin(conf_thres, iou_thres, max_det, box_coding)
        pluginlayer = network.add_plugin_v2([boxes.get_output(0), obj.get_output(0)], nms)
        pluginlayer.get_output(0).name = "num_dets"
        pluginlayer.get_output(1).name = "det_boxes"
        pluginlayer.get_output(2).name = "det_scores"
        pluginlayer.get_output(3).name = "det_classes"
        for i in range(4):
            network.mark_output(pluginlayer.get_output(i))

    inputTensor = network.get_input(0)
    print(f'{inputTensor.name} shape:{inputTensor.shape}')
    batch, c, h, w =  inputTensor.shape
    if batch != -1:
        min_shape[0], opt_shape[0], max_shape[0] = batch, batch, batch
    if c != -1:
        min_shape[1], opt_shape[1], max_shape[1] = c, c, c
    if h != -1:
        min_shape[-2], opt_shape[-2], max_shape[-2] = h, h, h
    if w != -1:
        min_shape[-1], opt_shape[-1], max_shape[-1] = w, w, w

    # Quantization
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8 and imgs_dir:
        config.set_flag(trt.BuilderFlag.INT8)
        if imgs_list is None:
            imgs_list = os.listdir(imgs_dir)
        config.int8_calibrator = calibrator.MyCalibrator(
                calibrationpath=imgs_dir,
                imgslist=imgs_list,
                nCalibration=n_iteration,
                inputShape=max_shape,
                cacheFile=cache_file
            )

    profile = builder.create_optimization_profile()
    profile.set_shape(inputTensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    print('Now, engine is building!')
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("Failed building engine!")
        # exit()
    with open(model_engine, "wb") as f:
        f.write(plan)
        print('Engine has been built!!!')

    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(plan)


class onnx2trt:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose=False):

        self.logger = trt.Logger(trt.Logger.ERROR)
        if verbose:
            self.logger = trt.Logger(trt.Logger.INFO)
            self.logger.min_severity = trt.Logger.Severity.VERBOSE

        # self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

        self.network = None
        self.profile = None
        self.parser = None

        self.FP16 = False
        self.INT8 = False

    def create_network(self, onnx_dir, add_nms=False, conf_thres=0.25, iou_thres=0.45, max_det=200, box_coding=1):

        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # Parse model file
        self.parser = trt.OnnxParser(self.network, self.logger)
        if not os.path.exists(onnx_dir):
            print("ONNX file is not exists!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(onnx_dir, "rb") as model:
            if not self.parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                exit()
            else:
                print("Succeeded parsing .onnx file!")

        # 添加nms算子
        if add_nms:
            """
            对原输出进行预处理，拆分成 目标框数据 和 类别置信度数据 两个矩阵，背景置信度要与类别置信度相乘。
            [1, 8500, 4 + 1 + 80] ——> [1, 8500, 4] + [1, 8500, 1 + 80] ——> [1, 8500, 4] + [1, 8500, 80]
            """
            print('Add EfficientNMS_TRT!')
            outputTensor = self.network.get_output(0)
            print(f'{outputTensor.name} shape:{outputTensor.shape}')
            bs, num_boxes, det_res = outputTensor.shape
            self.network.unmark_output(outputTensor)
            shapes = [bs, num_boxes, 4]
            boxes = self.network.add_slice(outputTensor, (0, 0, 0), shapes, (1, 1, 1))
            shapes[-1] = 1
            scores = self.network.add_slice(outputTensor, (0, 0, 4), shapes, (1, 1, 1))
            shapes[-1] = det_res - 5
            obj = self.network.add_slice(outputTensor, (0, 0, 5), shapes, (1, 1, 1))
            obj = self.network.add_elementwise(
                scores.get_output(0), obj.get_output(0), trt.ElementWiseOperation.PROD
            )
            """
            添加efficientNMS
            """
            nms = AddEfficientNMSPlugin(conf_thres, iou_thres, max_det, box_coding)
            pluginlayer = self.network.add_plugin_v2([boxes.get_output(0), obj.get_output(0)], nms)
            pluginlayer.get_output(0).name = "num_dets"
            pluginlayer.get_output(1).name = "det_boxes"
            pluginlayer.get_output(2).name = "det_scores"
            pluginlayer.get_output(3).name = "det_classes"
            for i in range(4):
                self.network.mark_output(pluginlayer.get_output(i))


    def create_engine(self, engine_dir, min_shape, opt_shape, max_shape, fp16=False, int8=False,
                      imgs_dir=None, n_iteration=128, cache_file=None):

        self.FP16 = fp16
        self.INT8 = int8

        inputTensor = self.network.get_input(0)
        print(f'{inputTensor.name} shape:{inputTensor.shape}')
        batch, c, h, w = inputTensor.shape
        if batch != -1:
            min_shape[0], opt_shape[0], max_shape[0] = batch, batch, batch
        if c != -1:
            min_shape[1], opt_shape[1], max_shape[1] = c, c, c
        if h != -1:
            min_shape[-2], opt_shape[-2], max_shape[-2] = h, h, h
        if w != -1:
            min_shape[-1], opt_shape[-1], max_shape[-1] = w, w, w

        # Quantization
        if self.FP16:
            self.config.set_flag(trt.BuilderFlag.FP16)
        if self.INT8:
            assert imgs_dir ,'If you choice int8, you should also set imgs_dir for the calibration'
            self.config.set_flag(trt.BuilderFlag.INT8)
            imgs_list = os.listdir(imgs_dir)
            calib = calibrator.MyCalibrator(
                calibrationpath=imgs_dir,
                imgslist=imgs_list,
                nCalibration=n_iteration,
                inputShape=max_shape,
                cacheFile=cache_file
            )
            self.config.int8_calibrator = calib

        self.profile = self.builder.create_optimization_profile()
        self.profile.set_shape(inputTensor.name, min_shape, opt_shape, max_shape)
        self.config.add_optimization_profile(self.profile)

        print('Now, engine is building!')
        t1 = time.time()
        plan = self.builder.build_serialized_network(self.network, self.config)
        t2 = time.time()
        print(f'{(t2 - t1)/60:0.2f}min')
        if plan is None:
            print("Failed building engine!")
            # exit()
        with open(engine_dir, "wb") as f:
            f.write(plan)
            print('Engine has been built!!!')

        runtime = trt.Runtime(self.logger)
        return runtime.deserialize_cuda_engine(plan)


def main(args):

    onnx_dir = args.onnx_dir
    engine_dir = args.engine_dir
    if engine_dir is None:
        engine_dir = f"./models_trt/{onnx_dir.split('/')[-1].replace('onnx', 'engine')}"

    yolo_engine = onnx2trt()
    yolo_engine.create_network(
        onnx_dir,
        add_nms=args.add_nms,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det
    )

    yolo_engine.create_engine(
        engine_dir,
        min_shape=args.min_shape,
        opt_shape=args.opt_shape,
        max_shape=args.max_shape,
        fp16=args.fp16,
        int8=args.int8,
        imgs_dir=args.imgs_dir,
        n_iteration=args.n_iteration,
        cache_file=args.cache_file
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    # onnx模型
    parser.add_argument('--onnx_dir', type=str, default='./models_onnx/yolov5s.onnx', help='onnx path')
    # engine模型保存地址
    parser.add_argument('--engine_dir', type=str, default=None, help='engine path')
    # 最小的输入shape
    parser.add_argument('--min_shape', nargs='+', type=int, default=[1, 3, 512, 512],
                        help='input min shape [batch, channel, height, width]')
    # 最佳优化的输入shape
    parser.add_argument('--opt_shape', nargs='+', type=int, default=[1, 3, 512, 512],
                        help='input opt shape [batch, channel, height, width]')
    # 最大的输入shape
    parser.add_argument('--max_shape', nargs='+', type=int, default=[1, 3, 512, 512],
                        help='input max shape [batch, channel, height, width]')
    # 是否使用fp16量化
    parser.add_argument('--fp16', type=bool, default=False, choices=[True, False],
                        help='TensorRt FP16 half-precision export')
    # 是否使用int8量化
    parser.add_argument('--int8', type=bool, default=True, choices=[True, False],
                        help='TensorRt INT8 quantization')
    # int8量化校准集位置
    parser.add_argument('--imgs_dir', default='./calibration', help='Dataset for int8 calibration')
    # 校准的轮次
    parser.add_argument('--n_iteration', type=int, default=512, help='Iteration for int8 calibration')
    # cache保存位置
    parser.add_argument('--cache_file', default=None, help='Int8 cache path')
    # 是否添加nms
    parser.add_argument('--add_nms', type=bool, default=True, choices=[True, False], help='add efficientNMS')
    # 只有得分大于置信度的预测框会被保留下来
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    # 非极大抑制所用到的nms_iou大小
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    # 目标框数量限制
    parser.add_argument('--max_det', type=int, default=200, help='maximum detections per image')

    args = parser.parse_args()
    print(args)

    main(args)








