#ifndef CURRENCY_NANODET_ENGINE_H
#define CURRENCY_NANODET_ENGINE_H

#include <cstdint>
#include <mutex>
#include <vector>

#include "net.h"

struct CurrencyObject {
    int32_t class_id = -1;
    float confidence = 0.0f;

    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

class CurrencyNanoDetEngine {
public:
    CurrencyNanoDetEngine() = default;
    ~CurrencyNanoDetEngine();

    CurrencyNanoDetEngine(const CurrencyNanoDetEngine&) = delete;
    CurrencyNanoDetEngine& operator=(const CurrencyNanoDetEngine&) = delete;

    int load(
        const char* param_path,
        const char* bin_path,
        bool request_gpu
    );

    int detect(
        const uint8_t* rgb_bytes,
        int width,
        int height,
        std::vector<CurrencyObject>& objects,
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

    // Model tải lên có feature maps:
    // 40x40, 20x20, 10x10, 5x5.
    int target_size_ = 320;

    // NanoDet-Plus mặc định dùng reg_max = 7.
    int reg_max_ = 7;

    // Tiền xử lý chuẩn thường dùng của NanoDet-Plus.
    float mean_values_[3] = {
        103.53f,
        116.28f,
        123.675f,
    };

    float norm_values_[3] = {
        1.0f / 57.375f,
        1.0f / 57.12f,
        1.0f / 58.395f,
    };
};

#endif // CURRENCY_NANODET_ENGINE_H
