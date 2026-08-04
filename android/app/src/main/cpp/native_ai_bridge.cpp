#include "native_ai_bridge.h"

#include <algorithm>
#include <android/log.h>
#include <chrono>
#include <vector>

#include "gpu.h"
#include "nanodet_engine.h"
#include "net.h"

#define LOG_TAG "NativeAIEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {
NanoDetEngine g_nanodet_engine;
}

extern "C" {

AI_EXPORT int32_t get_ai_engine_version(void) {
    return 120; // 1.2.0: multi-object detection
}

AI_EXPORT int32_t get_ncnn_has_vulkan(void) {
#if NCNN_VULKAN
    return ncnn::get_gpu_count() > 0 ? 1 : 0;
#else
    return 0;
#endif
}

AI_EXPORT int32_t load_nanodet_model(
    const char* param_path,
    const char* bin_path,
    int32_t use_gpu
) {
    return g_nanodet_engine.load(param_path, bin_path, use_gpu == 1);
}

AI_EXPORT int32_t is_nanodet_model_loaded(void) {
    return g_nanodet_engine.is_loaded() ? 1 : 0;
}

AI_EXPORT int32_t get_nanodet_backend(void) {
    if (!g_nanodet_engine.is_loaded()) {
        return -1;
    }
    return g_nanodet_engine.is_using_gpu() ? 1 : 0;
}

AI_EXPORT void unload_nanodet_model(void) {
    g_nanodet_engine.unload();
}

AI_EXPORT int32_t detect_rgb_image(
    const uint8_t* rgb_bytes,
    int32_t width,
    int32_t height,
    float probability_threshold,
    float nms_threshold,
    AIDetection* output,
    int32_t max_output,
    float* inference_time_ms
) {
    if (output == nullptr || max_output <= 0 || inference_time_ms == nullptr) {
        return -4;
    }

    std::vector<NanoDetObject> objects;
    const int result = g_nanodet_engine.detect(
        rgb_bytes,
        width,
        height,
        objects,
        probability_threshold,
        nms_threshold,
        inference_time_ms
    );

    if (result != 0) {
        return result;
    }

    const int count = std::min(
        static_cast<int>(objects.size()),
        static_cast<int>(max_output)
    );

    for (int index = 0; index < count; ++index) {
        output[index].class_id = objects[index].class_id;
        output[index].confidence = objects[index].confidence;
        output[index].x = objects[index].x;
        output[index].y = objects[index].y;
        output[index].width = objects[index].width;
        output[index].height = objects[index].height;
    }

    return count;
}

AI_EXPORT AIInferenceResult process_image_frame(
    const uint8_t* image_bytes,
    int32_t width,
    int32_t height,
    int32_t format
) {
    (void)image_bytes;
    (void)format;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::milli> elapsed = end - start;

    AIInferenceResult result{};
    result.width = width;
    result.height = height;
    result.channels = 3;
    result.inference_time_ms = elapsed.count();
    result.detected_class_id = -1;
    result.confidence = 0.0f;
    return result;
}

} // extern "C"
