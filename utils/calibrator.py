import os
import cv2 as cv
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from cuda import cudart
import tensorrt as trt

if cudart:
    cudart.cudaDeviceSynchronize()

__all__ = [
    'MyCalibrator',
    'MyCalibrator_v2'
]

def trans(img, size):
    crop_shape = min(img.shape[:2])
    img = img[:crop_shape - 1, :crop_shape - 1, :]
    img = cv.resize(img, size)
    img /= 255.0
    return img


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    """pycuda"""
    def __init__(self, calibrationpath, imgslist, nCalibration, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibrationpath = calibrationpath
        self.imgslist = imgslist
        self.nCalibration = nCalibration
        self.shape = inputShape  # (N,C,H,W)
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        self.dIn = cuda.mem_alloc(self.buffeSize)
        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))

    # def __del__(self):
    #     cudart.cudaFree(self.dIn)

    def batchGenerator(self):
        for i in range(self.nCalibration):
            print("> calibration %d" % i)
            subImageList = np.random.choice(self.imgslist, self.shape[0], replace=False)
            # self.imgslist = list(set(self.imgslist) - set(subImageList))
            yield np.ascontiguousarray(self.loadImages(subImageList))

    def loadImages(self, imageList):
        res = np.empty(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            path = os.path.join(self.calibrationpath, imageList[i])
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
            img = trans(img, self.shape[-2:]).transpose((2, 0, 1))
            res[i] = img
        return res

    def get_batch_size(self):  # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        try:
            data = next(self.oneBatch)
            # cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            cuda.memcpy_htod(self.dIn, data.ravel())
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")


class MyCalibrator_v2(trt.IInt8EntropyCalibrator2):
    """cuda-python"""
    def __init__(self, calibrationpath, imgslist, nCalibration, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibrationpath = calibrationpath
        self.imgslist = imgslist
        self.nCalibration = nCalibration
        self.shape = inputShape  # (N,C,H,W)
        self.buffeSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.buffeSize)
        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))

    def __del__(self):
        cudart.cudaFree(self.dIn)

    def batchGenerator(self):
        for i in range(self.nCalibration):
            print("> calibration %d" % i)
            subImageList = np.random.choice(self.imgslist, self.shape[0], replace=False)
            # self.imgslist = list(set(self.imgslist) - set(subImageList))
            yield np.ascontiguousarray(self.loadImages(subImageList))

    def loadImages(self, imageList):
        res = np.empty(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            path = os.path.join(self.calibrationpath, imageList[i])
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
            img = trans(img, self.shape[-2:]).transpose((2, 0, 1))
            res[i] = img
        return res

    def get_batch_size(self):  # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        try:
            data = next(self.oneBatch)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffeSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
