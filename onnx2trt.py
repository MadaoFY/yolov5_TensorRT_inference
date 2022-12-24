import os
import time
import numpy as np
import tensorrt as trt


import calibrator

__all__ = [
    'build_engine',
    'onnx2trt'
]


def build_engine(
        onnx_file, model_engine, min_shape, opt_shape, max_shape,
        fp16=False, int8=False,
        imgs_dir=None, imgs_list=None, n_iteration=128, cache_file=None
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

    def create_network(self, onnx_dir):

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

    yolo_engine = onnx2trt()
    yolo_engine.create_network(args.onnx_dir)
    yolo_engine.create_engine(
        args.engine_dir,
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
    parser.add_argument('--onnx_dir', type=str, default='./models_onnx/yolov5s_512.onnx',
                        help='onnx path')
    # engine模型保存地址
    parser.add_argument('--engine_dir', type=str, default='./models_trt/yolov5s_512.engine',
                        help='engine path')
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

    args = parser.parse_args()
    print(args)

    main(args)








