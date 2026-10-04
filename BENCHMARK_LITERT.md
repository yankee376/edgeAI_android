# LiteRT 640 benchmark

This branch builds a separate Android app (`com.example.android_ai_gateway.litert`) with the 99-class `best.pt` exported as `assets/models/best_w8a32.tflite`. The model has a 640×640 input and a `[1,103,8400]` output. The native Ultralytics `YOLOView` handles CameraX, preprocessing, LiteRT inference, detection decoding, and the overlay. It requests GPU, then falls back to CPU if the GPU cannot compile or run this model. The screen reports the accelerator that actually ran.

Build with `flutter pub get` and `flutter build apk --release`. Install `build/app/outputs/flutter-apk/app-release.apk` alongside the NCNN app. This branch does not compile the app's NCNN C++ target. The original NCNN source files are retained for comparison.

The overlay reports completed inference FPS, plugin FPS, CameraX RGBA-to-Bitmap time, resize/letterbox/normalize time, LiteRT inference time, decode/NMS time, detector total, and the measured camera-plus-detector total. The measurements are per completed frame; preview rendering, scheduling gaps, and Flutter UI rendering are not included in the measured total. The source frame size is displayed because CameraX can choose a size different from the 640×640 model input.

For a useful comparison, use the same Exynos 9825 phone, rear camera, scene, lighting, 640×640 model setting, and a warm-up period of at least 30 seconds. Record both apps for at least 60 seconds after warm-up, with their backend and timing panels visible. Compare detection quality as well as FPS. A video can show the on-screen readings but cannot by itself prove where driver or scheduler time was spent.

The vendored Ultralytics Flutter plugin is AGPL-3.0; see `third_party/ultralytics_yolo/LICENSE` before distribution.
