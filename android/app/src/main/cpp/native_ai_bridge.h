#ifndef NATIVE_AI_BRIDGE_H
#define NATIVE_AI_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define AI_EXPORT __declspec(dllexport)
#else
#define AI_EXPORT \
    __attribute__((visibility("default"))) \
    __attribute__((used))
#endif

typedef struct {
    int32_t width;
    int32_t height;
    int32_t channels;
    float inference_time_ms;
    int32_t detected_class_id;
    float confidence;
} AIInferenceResult;

// Một vật thể được phát hiện. Tọa độ tính theo pixel của ảnh RGB đầu vào.
typedef struct {
    int32_t class_id;
    float confidence;
    float x;
    float y;
    float width;
    float height;
} AIDetection;

AI_EXPORT int32_t get_ai_engine_version(void);
AI_EXPORT int32_t get_ncnn_has_vulkan(void);

//  0 = thành công
// -1 = đường dẫn không hợp lệ
// -2 = lỗi file .param
// -3 = lỗi file .bin
AI_EXPORT int32_t load_nanodet_model(
    const char* param_path,
    const char* bin_path,
    int32_t use_gpu
);

AI_EXPORT int32_t is_nanodet_model_loaded(void);

// -1 = chưa load, 0 = CPU, 1 = Vulkan GPU
AI_EXPORT int32_t get_nanodet_backend(void);
AI_EXPORT void unload_nanodet_model(void);

// Trả về số detection đã ghi vào output, hoặc mã lỗi âm:
// -1 = model chưa load
// -2 = ảnh đầu vào không hợp lệ
// -3 = lỗi chạy NCNN
// -4 = output buffer không hợp lệ
AI_EXPORT int32_t detect_rgb_image(
    const uint8_t* rgb_bytes,
    int32_t width,
    int32_t height,
    float probability_threshold,
    float nms_threshold,
    AIDetection* output,
    int32_t max_output,
    float* inference_time_ms
);

// API chẩn đoán cũ, tạm giữ để không phá code đã có.
AI_EXPORT AIInferenceResult process_image_frame(
    const uint8_t* image_bytes,
    int32_t width,
    int32_t height,
    int32_t format
);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_AI_BRIDGE_H
