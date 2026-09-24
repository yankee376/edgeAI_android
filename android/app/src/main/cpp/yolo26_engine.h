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
               float threshold, float nms_threshold, float* time_ms);
    void unload();
    bool is_loaded() const;
    bool is_using_gpu() const;
private:
    mutable std::mutex mutex_;
    ncnn::Net net_;
    bool loaded_ = false;
    bool gpu_ = false;
};
#endif
