# LiteRT 416×416 benchmark

This branch runs the same 99-class `best.pt` model family and Ultralytics `YOLOView` LiteRT path as `litert_640x640_benchmark`, with a model exported for a fixed 416×416 input. The exported model output is `[1,103,3549]`. The app has its own Android application ID (`com.example.android_ai_gateway.litert416`) and label, so it can be installed beside the 640×640 LiteRT app and the NCNN app.

Export used Ultralytics LiteRT W8A32 dynamic weight quantization (INT8 weights with FP32 activations), as in the 640×640 branch. The app requests GPU and reports the accelerator actually selected, including CPU fallback. It also shows frame conversion, preprocessing, inference, postprocessing, detector total, and completed-frame FPS.

The camera stream can remain at the CameraX-selected source size; YOLOView resizes and letterboxes each frame into the model's 416×416 input. The source dimensions shown on screen and model dimensions are separate values.

For comparison, run 640 and 416 on the same phone, rear camera, scene, lighting, and backend, after a 30-second warm-up. Record at least 60 seconds per mode and compare the same FPS counter, average inference time, detector total, and detection quality. The 416 mode is intended to leave more compute headroom for concurrent LLM and alert work; those workloads are not enabled in this camera benchmark.

Build and install with `flutter pub get`, `flutter build apk --release`, and `adb install -r build/app/outputs/flutter-apk/app-release.apk`.

The vendored Ultralytics Flutter plugin is AGPL-3.0; see `third_party/ultralytics_yolo/LICENSE` before distribution.
