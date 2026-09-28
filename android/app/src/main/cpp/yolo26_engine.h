#ifndef YOLO26_ENGINE_H
#define YOLO26_ENGINE_H
#include <cstdint>
#include <mutex>
#include <vector>
#include "net.h"
struct Yolo26Object {
    int32_t class_id;
    float confidence, x, y, width, height;
};
class Yolo26Engine {
public:
    ~Yolo26Engine();
    int load(const char* param, const char* bin, bool gpu);
    int detect(const uint8_t* rgb, int width, int height, std::vector<Yolo26Object>& result,
               float threshold, float nms_threshold, float* preprocess_ms,
               float* inference_ms, float* postprocess_ms);
    void unload();
    bool is_loaded() const;
    bool is_using_gpu() const;
    int cpu_core_count() const;
    int cpu_thread_count() const;
    int gpu_count() const;
    const char* gpu_name() const;
private:
    mutable std::mutex mutex_;
    ncnn::Net net_;
    bool loaded_ = false;
    bool gpu_ = false;
    int cpu_core_count_ = 1;
    int cpu_thread_count_ = 1;
};
#endif
