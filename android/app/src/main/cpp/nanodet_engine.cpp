// Detection logic adapted from nihui/ncnn-android-nanodet (BSD-3-Clause).
// OpenCV drawing/camera code was intentionally removed; Flutter draws boxes.

#include "nanodet_engine.h"

#include <algorithm>
#include <android/log.h>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "cpu.h"
#include "gpu.h"
#include "mat.h"

#define LOG_TAG "NativeAIEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Candidate {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    int label = -1;
    float probability = 0.0f;
};

float intersection_area(const Candidate& a, const Candidate& b) {
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.width, b.x + b.width);
    const float bottom = std::min(a.y + a.height, b.y + b.height);

    const float width = std::max(0.0f, right - left);
    const float height = std::max(0.0f, bottom - top);
    return width * height;
}

void sort_candidates(std::vector<Candidate>& candidates) {
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.probability > b.probability;
        }
    );
}

void nms_sorted_bboxes(
    const std::vector<Candidate>& candidates,
    std::vector<int>& picked,
    float nms_threshold
) {
    picked.clear();

    std::vector<float> areas(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        areas[i] = candidates[i].width * candidates[i].height;
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        const Candidate& a = candidates[i];
        bool keep = true;

        for (int picked_index : picked) {
            const Candidate& b = candidates[picked_index];
            const float intersection = intersection_area(a, b);
            const float union_area = areas[i] + areas[picked_index] - intersection;

            if (union_area > 0.0f && intersection / union_area > nms_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            picked.push_back(static_cast<int>(i));
        }
    }
}

float softmax_expected_distance(const float* logits, int count) {
    float maximum = -FLT_MAX;
    for (int i = 0; i < count; ++i) {
        maximum = std::max(maximum, logits[i]);
    }

    float denominator = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float value = std::exp(logits[i] - maximum);
        denominator += value;
        weighted_sum += static_cast<float>(i) * value;
    }

    return denominator > 0.0f ? weighted_sum / denominator : 0.0f;
}

void generate_proposals(
    const ncnn::Mat& cls_pred,
    const ncnn::Mat& dis_pred,
    int stride,
    const ncnn::Mat& padded_input,
    float probability_threshold,
    std::vector<Candidate>& objects
) {
    if (cls_pred.empty() || dis_pred.empty()) {
        return;
    }

    const int grid_width = padded_input.w / stride;
    const int grid_height = padded_input.h / stride;
    const int expected_grid_count = grid_width * grid_height;

    if (cls_pred.h != expected_grid_count || dis_pred.h != expected_grid_count) {
        LOGE(
            "Unexpected NanoDet output shape for stride %d: cls.h=%d dis.h=%d expected=%d",
            stride,
            cls_pred.h,
            dis_pred.h,
            expected_grid_count
        );
        return;
    }

    const int class_count = cls_pred.w;
    const int reg_max_plus_one = dis_pred.w / 4;
    if (class_count <= 0 || reg_max_plus_one <= 0) {
        return;
    }

    for (int row = 0; row < grid_height; ++row) {
        for (int column = 0; column < grid_width; ++column) {
            const int index = row * grid_width + column;
            const float* scores = cls_pred.row(index);

            int label = -1;
            float score = -FLT_MAX;
            for (int class_index = 0; class_index < class_count; ++class_index) {
                if (scores[class_index] > score) {
                    label = class_index;
                    score = scores[class_index];
                }
            }

            if (score < probability_threshold) {
                continue;
            }

            const float* distances = dis_pred.row(index);
            float predicted[4];
            for (int side = 0; side < 4; ++side) {
                predicted[side] = softmax_expected_distance(
                    distances + side * reg_max_plus_one,
                    reg_max_plus_one
                ) * static_cast<float>(stride);
            }

            const float center_x = (static_cast<float>(column) + 0.5f) * stride;
            const float center_y = (static_cast<float>(row) + 0.5f) * stride;

            const float x0 = center_x - predicted[0];
            const float y0 = center_y - predicted[1];
            const float x1 = center_x + predicted[2];
            const float y1 = center_y + predicted[3];

            Candidate candidate;
            candidate.x = x0;
            candidate.y = y0;
            candidate.width = x1 - x0;
            candidate.height = y1 - y0;
            candidate.label = label;
            candidate.probability = score;
            objects.push_back(candidate);
        }
    }
}

} // namespace

NanoDetEngine::~NanoDetEngine() {
    unload();
}

void NanoDetEngine::clear_unlocked() {
    net_.clear();
    loaded_ = false;
    using_gpu_ = false;
}

int NanoDetEngine::load(
    const char* param_path,
    const char* bin_path,
    bool request_gpu
) {
    std::lock_guard<std::mutex> guard(mutex_);
    clear_unlocked();

    if (param_path == nullptr ||
        bin_path == nullptr ||
        std::strlen(param_path) == 0 ||
        std::strlen(bin_path) == 0) {
        LOGE("NanoDet model paths are invalid");
        return -1;
    }

    int thread_count = ncnn::get_big_cpu_count();
    if (thread_count < 1) {
        thread_count = 1;
    }

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(thread_count);

    bool enable_gpu = false;
#if NCNN_VULKAN
    if (request_gpu) {
        const int gpu_count = ncnn::get_gpu_count();
        enable_gpu = gpu_count > 0;
        LOGI("NanoDet GPU requested. NCNN GPU count: %d", gpu_count);
    }
#else
    (void)request_gpu;
#endif

    net_.opt = ncnn::Option();
    net_.opt.num_threads = thread_count;
#if NCNN_VULKAN
    net_.opt.use_vulkan_compute = enable_gpu;
#endif

    LOGI("Loading NanoDet param: %s", param_path);
    const int param_result = net_.load_param(param_path);
    if (param_result != 0) {
        LOGE("Failed to load NanoDet param. Code: %d", param_result);
        clear_unlocked();
        return -2;
    }

    LOGI("Loading NanoDet bin: %s", bin_path);
    const int model_result = net_.load_model(bin_path);
    if (model_result != 0) {
        LOGE("Failed to load NanoDet bin. Code: %d", model_result);
        clear_unlocked();
        return -3;
    }

    loaded_ = true;
    using_gpu_ = enable_gpu;
    LOGI(
        "NanoDet model loaded successfully. Backend: %s, threads: %d",
        using_gpu_ ? "Vulkan GPU" : "CPU",
        thread_count
    );
    return 0;
}

int NanoDetEngine::detect(
    const uint8_t* rgb_bytes,
    int width,
    int height,
    std::vector<NanoDetObject>& objects,
    float probability_threshold,
    float nms_threshold,
    float* inference_time_ms
) {
    std::lock_guard<std::mutex> guard(mutex_);
    objects.clear();

    if (inference_time_ms != nullptr) {
        *inference_time_ms = 0.0f;
    }

    if (!loaded_) {
        return -1;
    }

    if (rgb_bytes == nullptr || width <= 0 || height <= 0) {
        return -2;
    }

    probability_threshold = std::clamp(probability_threshold, 0.01f, 0.99f);
    nms_threshold = std::clamp(nms_threshold, 0.01f, 0.99f);

    const auto start = std::chrono::high_resolution_clock::now();

    int resized_width = width;
    int resized_height = height;
    float scale = 1.0f;

    if (resized_width > resized_height) {
        scale = static_cast<float>(target_size_) / resized_width;
        resized_width = target_size_;
        resized_height = std::max(1, static_cast<int>(resized_height * scale));
    } else {
        scale = static_cast<float>(target_size_) / resized_height;
        resized_height = target_size_;
        resized_width = std::max(1, static_cast<int>(resized_width * scale));
    }

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        rgb_bytes,
        ncnn::Mat::PIXEL_RGB2BGR,
        width,
        height,
        resized_width,
        resized_height
    );

    if (input.empty()) {
        return -3;
    }

    const int width_padding =
        (resized_width + 31) / 32 * 32 - resized_width;
    const int height_padding =
        (resized_height + 31) / 32 * 32 - resized_height;

    ncnn::Mat padded_input;
    ncnn::copy_make_border(
        input,
        padded_input,
        height_padding / 2,
        height_padding - height_padding / 2,
        width_padding / 2,
        width_padding - width_padding / 2,
        ncnn::BORDER_CONSTANT,
        0.0f
    );

    padded_input.substract_mean_normalize(mean_values_, norm_values_);

    ncnn::Extractor extractor = net_.create_extractor();
    if (extractor.input("input.1", padded_input) != 0) {
        LOGE("NanoDet input() failed");
        return -3;
    }

    std::vector<Candidate> proposals;
    const int strides[3] = {8, 16, 32};
    const char* cls_names[3] = {
        "cls_pred_stride_8",
        "cls_pred_stride_16",
        "cls_pred_stride_32",
    };
    const char* dis_names[3] = {
        "dis_pred_stride_8",
        "dis_pred_stride_16",
        "dis_pred_stride_32",
    };

    for (int index = 0; index < 3; ++index) {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;

        const int cls_result = extractor.extract(cls_names[index], cls_pred);
        const int dis_result = extractor.extract(dis_names[index], dis_pred);
        if (cls_result != 0 || dis_result != 0) {
            LOGE(
                "NanoDet extract failed at stride %d. cls=%d dis=%d",
                strides[index],
                cls_result,
                dis_result
            );
            return -3;
        }

        generate_proposals(
            cls_pred,
            dis_pred,
            strides[index],
            padded_input,
            probability_threshold,
            proposals
        );
    }

    sort_candidates(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    objects.reserve(picked.size());
    for (int picked_index : picked) {
        const Candidate& candidate = proposals[picked_index];

        float x0 = (candidate.x - width_padding / 2.0f) / scale;
        float y0 = (candidate.y - height_padding / 2.0f) / scale;
        float x1 =
            (candidate.x + candidate.width - width_padding / 2.0f) / scale;
        float y1 =
            (candidate.y + candidate.height - height_padding / 2.0f) / scale;

        x0 = std::clamp(x0, 0.0f, static_cast<float>(width - 1));
        y0 = std::clamp(y0, 0.0f, static_cast<float>(height - 1));
        x1 = std::clamp(x1, 0.0f, static_cast<float>(width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(height - 1));

        if (x1 <= x0 || y1 <= y0) {
            continue;
        }

        NanoDetObject object;
        object.class_id = candidate.label;
        object.confidence = candidate.probability;
        object.x = x0;
        object.y = y0;
        object.width = x1 - x0;
        object.height = y1 - y0;
        objects.push_back(object);
    }

    std::sort(
        objects.begin(),
        objects.end(),
        [](const NanoDetObject& a, const NanoDetObject& b) {
            return a.width * a.height > b.width * b.height;
        }
    );

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::milli> elapsed = end - start;
    if (inference_time_ms != nullptr) {
        *inference_time_ms = elapsed.count();
    }

    LOGI(
        "NanoDet detected %zu objects in %.2f ms",
        objects.size(),
        elapsed.count()
    );
    return 0;
}

void NanoDetEngine::unload() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (loaded_) {
        LOGI("Unloading NanoDet model");
    }
    clear_unlocked();
}

bool NanoDetEngine::is_loaded() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return loaded_;
}

bool NanoDetEngine::is_using_gpu() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return loaded_ && using_gpu_;
}
