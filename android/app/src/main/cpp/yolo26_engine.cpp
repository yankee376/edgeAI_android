#include "yolo26_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include "cpu.h"
#include "gpu.h"
namespace {
struct Box { float x1,y1,x2,y2,score; int cls; };

#if NCNN_VULKAN
bool has_vulkan_gpu() {
    // This library is loaded through Dart FFI, so JNI_OnLoad is not a reliable
    // place to initialize Vulkan. Keep the VkInstance alive for the process.
    static std::once_flag once;
    static bool available = false;
    std::call_once(once, []() {
        available = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;
    });
    return available;
}
#endif

float overlap(const Box& a, const Box& b) {
    const float w = std::max(0.f, std::min(a.x2,b.x2)-std::max(a.x1,b.x1));
    const float h = std::max(0.f, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    const float area_a = (a.x2-a.x1)*(a.y2-a.y1);
    const float area_b = (b.x2-b.x1)*(b.y2-b.y1);
    const float intersection = w*h;
    const float union_area = area_a+area_b-intersection;
    return union_area > 0 ? intersection/union_area : 0.f;
}
}
Yolo26Engine::~Yolo26Engine() { unload(); }
int Yolo26Engine::load(const char* param, const char* bin, bool gpu) {
    std::lock_guard<std::mutex> guard(mutex_);
    net_.clear(); loaded_ = false; gpu_ = false;
    cpu_core_count_ = ncnn::get_cpu_count();
    if (cpu_core_count_ < 1) cpu_core_count_ = 1;
    cpu_thread_count_ = 1;
    if (!param || !bin) return -1;
    cpu_thread_count_ = ncnn::get_big_cpu_count();
    if (cpu_thread_count_ < 1) cpu_thread_count_ = ncnn::get_cpu_count();
    if (cpu_thread_count_ < 1) cpu_thread_count_ = 1;
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(cpu_thread_count_);
    net_.opt = ncnn::Option();
    net_.opt.num_threads = cpu_thread_count_;
#if NCNN_VULKAN
    gpu_ = gpu && has_vulkan_gpu();
    net_.opt.use_vulkan_compute = gpu_;
#else
    (void)gpu;
#endif
    if (net_.load_param(param) != 0) { net_.clear(); gpu_ = false; return -2; }
    if (net_.load_model(bin) != 0) { net_.clear(); gpu_ = false; return -3; }
    loaded_ = true;
    return 0;
}
void Yolo26Engine::unload() {
    std::lock_guard<std::mutex> guard(mutex_);
    net_.clear(); loaded_ = false; gpu_ = false;
}
bool Yolo26Engine::is_loaded() const {
    std::lock_guard<std::mutex> guard(mutex_); return loaded_;
}
bool Yolo26Engine::is_using_gpu() const {
    std::lock_guard<std::mutex> guard(mutex_); return gpu_;
}
int Yolo26Engine::cpu_core_count() const {
    std::lock_guard<std::mutex> guard(mutex_); return cpu_core_count_;
}
int Yolo26Engine::cpu_thread_count() const {
    std::lock_guard<std::mutex> guard(mutex_); return cpu_thread_count_;
}
int Yolo26Engine::gpu_count() const {
#if NCNN_VULKAN
    return has_vulkan_gpu() ? ncnn::get_gpu_count() : 0;
#else
    return 0;
#endif
}
const char* Yolo26Engine::gpu_name() const {
#if NCNN_VULKAN
    if (!has_vulkan_gpu()) return "Vulkan unavailable";
    const int device_index = ncnn::get_default_gpu_index();
    return device_index >= 0 && device_index < ncnn::get_gpu_count()
        ? ncnn::get_gpu_info(device_index).device_name()
        : "Unknown Vulkan device";
#else
    return "NCNN built without Vulkan";
#endif
}
int Yolo26Engine::detect(const uint8_t* rgb, int width, int height,
                          std::vector<Yolo26Object>& result, float threshold,
                          float nms_threshold, float* preprocess_ms,
                          float* inference_ms, float* postprocess_ms) {
    result.clear();
    std::lock_guard<std::mutex> guard(mutex_);
    if (!loaded_) return -1;
    if (!rgb || width <= 0 || height <= 0 || !std::isfinite(threshold) ||
        !std::isfinite(nms_threshold) || threshold < 0 || threshold > 1 ||
        nms_threshold < 0 || nms_threshold > 1) return -2;
    const auto start = std::chrono::steady_clock::now();
    constexpr int input_size = 640;
    const float scale = std::min(640.f / width, 640.f / height);
    const int rw = std::max(1, std::min(input_size, static_cast<int>(std::round(width*scale))));
    const int rh = std::max(1, std::min(input_size, static_cast<int>(std::round(height*scale))));
    const int left = (input_size-rw)/2, top = (input_size-rh)/2;
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB,
                                                        width, height, rw, rh);
    if (resized.empty()) return -3;
    ncnn::Mat input;
    ncnn::copy_make_border(resized, input, top, input_size-rh-top,
                           left, input_size-rw-left, ncnn::BORDER_CONSTANT, 114.f);
    if (input.empty()) return -3;
    const float norm[3] = {1.f/255,1.f/255,1.f/255};
    input.substract_mean_normalize(nullptr,norm);
    ncnn::Extractor ex = net_.create_extractor();
    if (ex.input("in0", input) != 0) return -3;
    ncnn::Mat out;
    const auto preprocess_end = std::chrono::steady_clock::now();
    if (ex.extract("out0",out) != 0) return -3;
    const auto infer_end = std::chrono::steady_clock::now();
    // Export shape: [1, 103, 8400]. Four first rows are cx, cy, w, h.
    if (out.dims != 2 || out.h != 103 || out.w != 8400 || out.elemsize != 4) return -5;
    std::vector<Box> boxes;
    const float* cx = out.row(0), *cy = out.row(1);
    const float* bw = out.row(2), *bh = out.row(3);
    for (int i=0;i<out.w;++i) {
        int cls = -1; float score = threshold;
        for (int k=0;k<99;++k) {
            float s = out.row(4+k)[i];
            if (std::isfinite(s) && s > score) { score=s; cls=k; }
        }
        if (cls < 0 || !std::isfinite(cx[i]) || !std::isfinite(cy[i]) ||
            !std::isfinite(bw[i]) || !std::isfinite(bh[i]) || bw[i] <= 0 || bh[i] <= 0) continue;
        boxes.push_back({cx[i]-bw[i]/2,cy[i]-bh[i]/2,cx[i]+bw[i]/2,cy[i]+bh[i]/2,score,cls});
    }
    std::sort(boxes.begin(),boxes.end(),[](const Box& a,const Box& b){return a.score>b.score;});
    for (size_t i=0;i<boxes.size();++i) {
        const Box& b=boxes[i];
        bool suppressed=false;
        // Keep independent class hypotheses for overlapping objects.
        for (const auto& kept: result) {
            if (kept.class_id != b.cls) continue;
            Box original{kept.x*scale+left,kept.y*scale+top,
                         (kept.x+kept.width)*scale+left,(kept.y+kept.height)*scale+top,0,0};
            if (overlap(b,original)>nms_threshold) { suppressed=true; break; }
        }
        if (suppressed) continue;
        float x1=std::clamp((b.x1-left)/scale,0.f,float(width));
        float y1=std::clamp((b.y1-top)/scale,0.f,float(height));
        float x2=std::clamp((b.x2-left)/scale,0.f,float(width));
        float y2=std::clamp((b.y2-top)/scale,0.f,float(height));
        if (x2<=x1 || y2<=y1) continue;
        result.push_back({b.cls,b.score,x1,y1,x2-x1,y2-y1});
    }
    const auto postprocess_end = std::chrono::steady_clock::now();
    if (preprocess_ms) *preprocess_ms =
        std::chrono::duration<float, std::milli>(preprocess_end-start).count();
    if (inference_ms) *inference_ms =
        std::chrono::duration<float, std::milli>(infer_end-preprocess_end).count();
    if (postprocess_ms) *postprocess_ms =
        std::chrono::duration<float, std::milli>(postprocess_end-infer_end).count();
    return 0;
}
