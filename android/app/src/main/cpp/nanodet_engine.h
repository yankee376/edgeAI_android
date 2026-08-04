#ifndef NANODET_ENGINE_H
#define NANODET_ENGINE_H

#include <cstdint>
#include <mutex>
#include <vector>

#include "net.h"

struct NanoDetObject {
    int32_t class_id = -1;
    float confidence = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

class NanoDetEngine {
public:
    NanoDetEngine() = default;
    ~NanoDetEngine();

    NanoDetEngine(const NanoDetEngine&) = delete;
    NanoDetEngine& operator=(const NanoDetEngine&) = delete;

    int load(
        const char* param_path,
        const char* bin_path,
        bool request_gpu
    );

    int detect(
        const uint8_t* rgb_bytes,
        int width,
        int height,
        std::vector<NanoDetObject>& objects,
        float probability_threshold,
        float nms_threshold,
        float* inference_time_ms
    );

    void unload();
    bool is_loaded() const;
    bool is_using_gpu() const;

private:
    void clear_unlocked();

    mutable std::mutex mutex_;
    ncnn::Net net_;

    bool loaded_ = false;
    bool using_gpu_ = false;

    int target_size_ = 320;
    float mean_values_[3] = {127.0f, 127.0f, 127.0f};
    float norm_values_[3] = {
        1.0f / 128.0f,
        1.0f / 128.0f,
        1.0f / 128.0f,
    };
};

#endif // NANODET_ENGINE_H
