#include "yolo26_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include "gpu.h"
namespace {
struct Box { float x1,y1,x2,y2,score; int cls; };
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
    if (!param || !bin) return -1;
    net_.opt = ncnn::Option();
#if NCNN_VULKAN
    gpu_ = gpu && ncnn::get_gpu_count() > 0;
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
int Yolo26Engine::detect(const uint8_t* rgb, int width, int height,
                          std::vector<Yolo26Object>& result, float threshold,
                          float nms_threshold, float* time_ms) {
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
    if (ex.extract("out0",out) != 0) return -3;
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
    if (time_ms) *time_ms=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-start).count();
    return 0;
}
