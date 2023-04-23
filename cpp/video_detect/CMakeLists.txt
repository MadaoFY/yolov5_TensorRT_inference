cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(yolo_tensorrt C CXX)

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
	set(CMAKE_CXX_FLAGS_Release "$ENV{CXXFLAGS} -O3 -Wall")
endif()


set(src_list main.cpp  utils_detection.cpp  utils_detection.h trt_infer.cpp trt_infer.h preprocess.cu preprocess.h)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER  E:/NV/cuda11.7/bin/nvcc)
enable_language(CUDA)

# CUDA
# TODO(Call for PR): make cmake compatible with Windows
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
include_directories(E:/NV/cuda11.7/include/)
link_directories(E:/NV/cuda11.7/lib/x64/)


# tensorrt
set(TRT_DIR J:/tensorrt/TensorRT-8.4.3.1) 
set(TRT_INCLUDE_DIRS ${TRT_DIR}/include/)
set(TRT_LIB_DIRS ${TRT_DIR}/lib/)

include_directories(${TRT_INCLUDE_DIRS}) 


# opencv
set(CMAKE_PREFIX_PATH E:/opencv/build/x64/vc16/lib)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME}  ${src_list})
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})