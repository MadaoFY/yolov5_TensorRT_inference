cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(yolo_detect C CXX)

add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
if(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
endif()
set(src_list csi_detect.cpp utils_detection.cpp utils_detection.h trt_infer.cpp trt_infer.h gstreamer.cpp gstreamer.h preprocess.cu preprocess.h)

# CUDA
# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
enable_language(CUDA)
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
include_directories(/usr/local/cuda-10.2/include/)
link_directories(/usr/local/cuda-10.2/lib64/)


# tensorrt
# set(TRT_DIR J:/tensorrt/TensorRT-8.4.3.1) 
# set(TRT_INCLUDE_DIRS ${TRT_DIR}/include/)
# set(TRT_LIB_DIRS ${TRT_DIR}/lib/)

# include_directories(${TRT_INCLUDE_DIRS}) 
# link_directories(${TRT_LIB_DIRS})

#include_directories(${PROJECT_SOURCE_DIR}/)
#file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)

# opencv
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME}  ${src_list})
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
