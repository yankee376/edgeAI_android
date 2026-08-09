// Currency detection engine using NanoDet-Plus + NCNN.
//
// Model information read from nanodet_currency.ncnn.param:
// - Input blob:  in0
// - Output blob: out0
// - Input size:  320 x 320
// - Strides:     8, 16, 32, 64
// - Output:      2125 rows x 122 values
//
// The implementation is independent from nanodet_engine.cpp so that
// the object-detection model and currency model can coexist.

#include "currency_nanodet_engine.h"

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

#define LOG_TAG "CurrencyNanoDet"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Candidate {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    int label = -1;
    float probability = 0.0f;
};

struct CenterPrior {
    int x = 0;
    int y = 0;
    int stride = 0;
};

/**
 * Sigmoid dùng cho classification logits của NanoDet-Plus.
 */
float sigmoid(float value) {
    // Cách viết này ổn định hơn khi value rất lớn hoặc rất nhỏ.
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }

    const float z = std::exp(value);
    return z / (1.0f + z);
}

/**
 * Tính diện tích giao nhau của hai bounding box.
 */
float intersection_area(
    const Candidate& first,
    const Candidate& second
) {
    const float left = std::max(
        first.x,
        second.x
    );

    const float top = std::max(
        first.y,
        second.y
    );

    const float right = std::min(
        first.x + first.width,
        second.x + second.width
    );

    const float bottom = std::min(
        first.y + first.height,
        second.y + second.height
    );

    const float intersection_width = std::max(
        0.0f,
        right - left
    );

    const float intersection_height = std::max(
        0.0f,
        bottom - top
    );

    return intersection_width * intersection_height;
}

/**
 * Sắp xếp proposal theo confidence giảm dần.
 */
void sort_candidates(
    std::vector<Candidate>& candidates
) {
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& first, const Candidate& second) {
            return first.probability > second.probability;
        }
    );
}

/**
 * Non-Maximum Suppression.
 *
 * NMS ở đây là class-agnostic: hai box chồng lên nhau nhiều sẽ bị loại
 * kể cả khi chúng được dự đoán thành hai class khác nhau.
 *
 * Với nhận diện tiền, cách này giúp tránh việc một tờ tiền bị trả về
 * thành nhiều mệnh giá khác nhau.
 */
void nms_sorted_bboxes(
    const std::vector<Candidate>& candidates,
    std::vector<int>& picked,
    float nms_threshold
) {
    picked.clear();

    std::vector<float> areas(candidates.size());

    for (size_t index = 0;
         index < candidates.size();
         ++index) {
        areas[index] =
            candidates[index].width *
            candidates[index].height;
    }

    for (size_t index = 0;
         index < candidates.size();
         ++index) {
        const Candidate& current = candidates[index];

        bool keep = true;

        for (const int picked_index : picked) {
            const Candidate& previous =
                candidates[picked_index];

            const float intersection =
                intersection_area(current, previous);

            const float union_area =
                areas[index] +
                areas[picked_index] -
                intersection;

            if (union_area <= 0.0f) {
                continue;
            }

            const float iou =
                intersection / union_area;

            if (iou > nms_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            picked.push_back(
                static_cast<int>(index)
            );
        }
    }
}

/**
 * Áp dụng softmax lên một phía của bbox distribution rồi tính kỳ vọng.
 *
 * Ví dụ reg_max = 7:
 * mỗi phía có 8 giá trị, đại diện khoảng cách 0..7.
 */
float softmax_expected_distance(
    const float* logits,
    int count
) {
    if (logits == nullptr || count <= 0) {
        return 0.0f;
    }

    float maximum = -FLT_MAX;

    for (int index = 0;
         index < count;
         ++index) {
        maximum = std::max(
            maximum,
            logits[index]
        );
    }

    float denominator = 0.0f;
    float weighted_sum = 0.0f;

    for (int index = 0;
         index < count;
         ++index) {
        const float exponential =
            std::exp(logits[index] - maximum);

        denominator += exponential;

        weighted_sum +=
            static_cast<float>(index) *
            exponential;
    }

    if (denominator <= 0.0f) {
        return 0.0f;
    }

    return weighted_sum / denominator;
}

/**
 * Tạo toàn bộ center prior theo đúng thứ tự output model:
 *
 * stride 8:  40 x 40 = 1600
 * stride 16: 20 x 20 = 400
 * stride 32: 10 x 10 = 100
 * stride 64: 5 x 5   = 25
 *
 * Tổng cộng: 2125 điểm.
 */
void generate_center_priors(
    int input_width,
    int input_height,
    std::vector<CenterPrior>& priors
) {
    priors.clear();

    constexpr int strides[] = {
        8,
        16,
        32,
        64,
    };

    for (const int stride : strides) {
        const int feature_width =
            input_width / stride;

        const int feature_height =
            input_height / stride;

        for (int row = 0;
             row < feature_height;
             ++row) {
            for (int column = 0;
                 column < feature_width;
                 ++column) {
                CenterPrior prior;

                prior.x = column;
                prior.y = row;
                prior.stride = stride;

                priors.push_back(prior);
            }
        }
    }
}

/**
 * Giải mã output duy nhất của model NanoDet-Plus.
 *
 * Mỗi hàng output có dạng:
 *
 * [class logits] + [left distribution]
 *                + [top distribution]
 *                + [right distribution]
 *                + [bottom distribution]
 *
 * Với model này:
 *
 * output width = 122
 * reg_max      = 7
 *
 * bbox channels = 4 * (7 + 1) = 32
 * class count   = 122 - 32     = 90
 */
bool generate_currency_proposals(
    const ncnn::Mat& prediction,
    const std::vector<CenterPrior>& priors,
    int reg_max,
    float probability_threshold,
    std::vector<Candidate>& proposals
) {
    if (prediction.empty()) {
        LOGE("Currency prediction is empty");
        return false;
    }

    const int reg_count = reg_max + 1;
    const int bbox_channel_count = 4 * reg_count;

    const int values_per_point = prediction.w;
    const int point_count = prediction.h;

    const int class_count =
        values_per_point - bbox_channel_count;

    if (class_count <= 0) {
        LOGE(
            "Invalid currency output width: %d, "
            "reg_max: %d, bbox channels: %d",
            values_per_point,
            reg_max,
            bbox_channel_count
        );

        return false;
    }

    if (point_count !=
        static_cast<int>(priors.size())) {
        LOGE(
            "Unexpected currency output shape: "
            "w=%d h=%d c=%d, expected points=%zu",
            prediction.w,
            prediction.h,
            prediction.c,
            priors.size()
        );

        return false;
    }

    LOGI(
        "Currency decoder: classes=%d, reg_max=%d, points=%d",
        class_count,
        reg_max,
        point_count
    );

    for (int point_index = 0;
         point_index < point_count;
         ++point_index) {
        const float* row =
            prediction.row(point_index);

        if (row == nullptr) {
            continue;
        }

        int best_label = -1;
        float best_logit = -FLT_MAX;

        // Tìm class có logit lớn nhất.
        for (int class_index = 0;
             class_index < class_count;
             ++class_index) {
            const float logit = row[class_index];

            if (logit > best_logit) {
                best_logit = logit;
                best_label = class_index;
            }
        }

        if (best_label < 0) {
            continue;
        }

        const float probability =
            sigmoid(best_logit);

        if (probability <
            probability_threshold) {
            continue;
        }

        const float* bbox_prediction =
            row + class_count;

        float predicted_distance[4] = {
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        };

        for (int side = 0;
             side < 4;
             ++side) {
            const float distribution_distance =
                softmax_expected_distance(
                    bbox_prediction +
                        side * reg_count,
                    reg_count
                );

            predicted_distance[side] =
                distribution_distance *
                static_cast<float>(
                    priors[point_index].stride
                );
        }

        const float center_x =
            static_cast<float>(
                priors[point_index].x *
                priors[point_index].stride
            );

        const float center_y =
            static_cast<float>(
                priors[point_index].y *
                priors[point_index].stride
            );

        const float x0 =
            center_x - predicted_distance[0];

        const float y0 =
            center_y - predicted_distance[1];

        const float x1 =
            center_x + predicted_distance[2];

        const float y1 =
            center_y + predicted_distance[3];

        Candidate candidate;

        candidate.x = x0;
        candidate.y = y0;
        candidate.width = x1 - x0;
        candidate.height = y1 - y0;
        candidate.label = best_label;
        candidate.probability = probability;

        if (candidate.width <= 0.0f ||
            candidate.height <= 0.0f) {
            continue;
        }

        proposals.push_back(candidate);
    }

    return true;
}

} // namespace

CurrencyNanoDetEngine::~CurrencyNanoDetEngine() {
    unload();
}

void CurrencyNanoDetEngine::clear_unlocked() {
    net_.clear();

    loaded_ = false;
    using_gpu_ = false;
}

int CurrencyNanoDetEngine::load(
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
        LOGE("Currency model paths are invalid");
        return -1;
    }

    int thread_count =
        ncnn::get_big_cpu_count();

    if (thread_count < 1) {
        thread_count = 1;
    }

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(thread_count);

    bool enable_gpu = false;

#if NCNN_VULKAN
    if (request_gpu) {
        const int gpu_count =
            ncnn::get_gpu_count();

        enable_gpu = gpu_count > 0;

        LOGI(
            "Currency GPU requested. "
            "NCNN GPU count: %d",
            gpu_count
        );
    }
#else
    (void)request_gpu;
#endif

    net_.opt = ncnn::Option();
    net_.opt.num_threads = thread_count;

#if NCNN_VULKAN
    net_.opt.use_vulkan_compute =
        enable_gpu;
#endif

    LOGI(
        "Loading currency param: %s",
        param_path
    );

    const int param_result =
        net_.load_param(param_path);

    if (param_result != 0) {
        LOGE(
            "Failed to load currency param. Code: %d",
            param_result
        );

        clear_unlocked();
        return -2;
    }

    LOGI(
        "Loading currency bin: %s",
        bin_path
    );

    const int model_result =
        net_.load_model(bin_path);

    if (model_result != 0) {
        LOGE(
            "Failed to load currency bin. Code: %d",
            model_result
        );

        clear_unlocked();
        return -3;
    }

    loaded_ = true;
    using_gpu_ = enable_gpu;

    LOGI(
        "Currency model loaded successfully. "
        "Backend: %s, threads: %d",
        using_gpu_ ? "Vulkan GPU" : "CPU",
        thread_count
    );

    return 0;
}

int CurrencyNanoDetEngine::detect(
    const uint8_t* rgb_bytes,
    int width,
    int height,
    std::vector<CurrencyObject>& objects,
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
        LOGE("Currency model has not been loaded");
        return -1;
    }

    if (rgb_bytes == nullptr ||
        width <= 0 ||
        height <= 0) {
        LOGE(
            "Invalid currency input image: "
            "pointer=%p width=%d height=%d",
            rgb_bytes,
            width,
            height
        );

        return -2;
    }

    probability_threshold = std::clamp(
        probability_threshold,
        0.01f,
        0.99f
    );

    nms_threshold = std::clamp(
        nms_threshold,
        0.01f,
        0.99f
    );

    const auto start =
        std::chrono::high_resolution_clock::now();

    /*
     * Letterbox ảnh về đúng 320 x 320.
     *
     * Model currency có output cố định 2125 điểm nên input phải được
     * pad thành đúng target_size_, không chỉ pad tới bội số gần nhất.
     */
    const float scale = std::min(
        static_cast<float>(target_size_) /
            static_cast<float>(width),
        static_cast<float>(target_size_) /
            static_cast<float>(height)
    );

    const int resized_width = std::max(
        1,
        static_cast<int>(
            std::round(
                static_cast<float>(width) *
                scale
            )
        )
    );

    const int resized_height = std::max(
        1,
        static_cast<int>(
            std::round(
                static_cast<float>(height) *
                scale
            )
        )
    );

    /*
     * Flutter gửi dữ liệu RGB.
     * NanoDet được huấn luyện với thứ tự BGR nên dùng PIXEL_RGB2BGR.
     */
    ncnn::Mat resized_image =
        ncnn::Mat::from_pixels_resize(
            rgb_bytes,
            ncnn::Mat::PIXEL_RGB2BGR,
            width,
            height,
            resized_width,
            resized_height
        );

    if (resized_image.empty()) {
        LOGE("Failed to resize currency image");
        return -3;
    }

    const int total_horizontal_padding =
        target_size_ - resized_width;

    const int total_vertical_padding =
        target_size_ - resized_height;

    const int left_padding =
        total_horizontal_padding / 2;

    const int right_padding =
        total_horizontal_padding -
        left_padding;

    const int top_padding =
        total_vertical_padding / 2;

    const int bottom_padding =
        total_vertical_padding -
        top_padding;

    ncnn::Mat input;

    ncnn::copy_make_border(
        resized_image,
        input,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        ncnn::BORDER_CONSTANT,
        0.0f
    );

    if (input.empty()) {
        LOGE("Failed to pad currency image");
        return -3;
    }

    if (input.w != target_size_ ||
        input.h != target_size_) {
        LOGE(
            "Unexpected currency input shape: "
            "w=%d h=%d expected=%d",
            input.w,
            input.h,
            target_size_
        );

        return -3;
    }

    input.substract_mean_normalize(
        mean_values_,
        norm_values_
    );

    ncnn::Extractor extractor =
        net_.create_extractor();

    const int input_result =
        extractor.input("in0", input);

    if (input_result != 0) {
        LOGE(
            "Currency input(\"in0\") failed. Code: %d",
            input_result
        );

        return -3;
    }

    ncnn::Mat prediction;

    const int output_result =
        extractor.extract(
            "out0",
            prediction
        );

    if (output_result != 0) {
        LOGE(
            "Currency extract(\"out0\") failed. Code: %d",
            output_result
        );

        return -3;
    }

    LOGI(
        "Currency output shape: "
        "dims=%d w=%d h=%d d=%d c=%d",
        prediction.dims,
        prediction.w,
        prediction.h,
        prediction.d,
        prediction.c
    );

    /*
     * Model tải lên dự kiến trả về:
     *
     * prediction.w = 122
     * prediction.h = 2125
     * prediction.c = 1
     */
    if (prediction.w != 122) {
        LOGE(
            "Unexpected values per currency point: "
            "%d, expected 122",
            prediction.w
        );

        return -3;
    }

    std::vector<CenterPrior> priors;

    generate_center_priors(
        target_size_,
        target_size_,
        priors
    );

    std::vector<Candidate> proposals;

    const bool decoded =
        generate_currency_proposals(
            prediction,
            priors,
            reg_max_,
            probability_threshold,
            proposals
        );

    if (!decoded) {
        return -3;
    }

    sort_candidates(proposals);

    std::vector<int> picked;

    nms_sorted_bboxes(
        proposals,
        picked,
        nms_threshold
    );

    objects.reserve(picked.size());

    for (const int picked_index : picked) {
        if (picked_index < 0 ||
            picked_index >=
                static_cast<int>(proposals.size())) {
            continue;
        }

        const Candidate& candidate =
            proposals[picked_index];

        /*
         * Chuyển tọa độ từ ảnh 320x320 đã letterbox
         * về tọa độ ảnh RGB gốc.
         */
        float x0 =
            (
                candidate.x -
                static_cast<float>(left_padding)
            ) / scale;

        float y0 =
            (
                candidate.y -
                static_cast<float>(top_padding)
            ) / scale;

        float x1 =
            (
                candidate.x +
                candidate.width -
                static_cast<float>(left_padding)
            ) / scale;

        float y1 =
            (
                candidate.y +
                candidate.height -
                static_cast<float>(top_padding)
            ) / scale;

        x0 = std::clamp(
            x0,
            0.0f,
            static_cast<float>(width - 1)
        );

        y0 = std::clamp(
            y0,
            0.0f,
            static_cast<float>(height - 1)
        );

        x1 = std::clamp(
            x1,
            0.0f,
            static_cast<float>(width - 1)
        );

        y1 = std::clamp(
            y1,
            0.0f,
            static_cast<float>(height - 1)
        );

        if (x1 <= x0 || y1 <= y0) {
            continue;
        }

        CurrencyObject object;

        object.class_id =
            candidate.label;

        object.confidence =
            candidate.probability;

        object.x = x0;
        object.y = y0;
        object.width = x1 - x0;
        object.height = y1 - y0;

        objects.push_back(object);
    }

    // Đưa kết quả có confidence cao nhất lên đầu.
    std::sort(
        objects.begin(),
        objects.end(),
        [](const CurrencyObject& first,
           const CurrencyObject& second) {
            return first.confidence >
                   second.confidence;
        }
    );

    const auto end =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<
        float,
        std::milli
    > elapsed = end - start;

    if (inference_time_ms != nullptr) {
        *inference_time_ms =
            elapsed.count();
    }

    LOGI(
        "Currency detected %zu objects "
        "from %zu proposals in %.2f ms",
        objects.size(),
        proposals.size(),
        elapsed.count()
    );

    return 0;
}

void CurrencyNanoDetEngine::unload() {
    std::lock_guard<std::mutex> guard(mutex_);

    if (loaded_) {
        LOGI("Unloading currency model");
    }

    clear_unlocked();
}

bool CurrencyNanoDetEngine::is_loaded() const {
    std::lock_guard<std::mutex> guard(mutex_);

    return loaded_;
}

bool CurrencyNanoDetEngine::is_using_gpu() const {
    std::lock_guard<std::mutex> guard(mutex_);

    return loaded_ && using_gpu_;
}
