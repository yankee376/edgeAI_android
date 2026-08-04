cmake_minimum_required(VERSION 3.18.1)

project("native_ai_engine" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
    ncnn_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/ncnn-20260526-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn
)

find_package(ncnn REQUIRED)

add_library(
    native_ai_engine
    SHARED
    native_ai_bridge.cpp
    nanodet_engine.cpp
)

find_library(
    log-lib
    log
)

target_link_libraries(
    native_ai_engine
    ncnn
    ${log-lib}
)
