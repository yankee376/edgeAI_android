# Third-party notices

The NanoDet proposal generation, preprocessing, decoding, and non-maximum
suppression logic in `android/app/src/main/cpp/nanodet_engine.cpp` is adapted
from:

- `nihui/ncnn-android-nanodet`
- Original file: `app/src/main/jni/nanodet.cpp`
- License: BSD 3-Clause
- Copyright (C) 2021 THL A29 Limited, a Tencent company

The adapted implementation removes OpenCV drawing and Android NDK camera code.
Flutter owns camera capture and bounding-box rendering in this project.
