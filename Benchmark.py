# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对导出的模型进行推理，并进行速度测试
# 目前 GPU 上 tensorRT 是跑的最快的部署框架 ...
# ---------------------------------------------------------------

import time
import numpy as np
import tensorrt as trt
import timm.data.transforms_factory

from tqdm import tqdm
from utils import trt_infer

# int8 / fp32 ~ 70%
# trt > ppq > fp32

# Nvidia Nsight Performance Profile
ENGINE_PATH = './models_trt/yolov5s.engine'
BATCH_SIZE  = 1
INPUT_SHAPE = [BATCH_SIZE, 3, 512, 512]
BENCHMARK_SAMPLES = 12800

print(f'Benchmark with {ENGINE_PATH}')
logger = trt.Logger(trt.Logger.ERROR)
with open(ENGINE_PATH, 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
    inputs[0].host = np.zeros(shape=INPUT_SHAPE, dtype=np.float32)

    t1 = time.time()
    for _ in tqdm(range(BENCHMARK_SAMPLES), desc=f'Benchmark ...'):
        trt_infer.do_inference(
            context, bindings=bindings, inputs=inputs, 
            outputs=outputs, stream=stream, batch_size=BATCH_SIZE)

    t2 = time.time()
    t = (t2 - t1)*1000/BENCHMARK_SAMPLES
    print(f"{t:0.5f}ms")

