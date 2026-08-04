import 'dart:ffi' as ffi;
import 'dart:io';

final class NativeAIInferenceResult extends ffi.Struct {
  @ffi.Int32()
  external int width;

  @ffi.Int32()
  external int height;

  @ffi.Int32()
  external int channels;

  @ffi.Float()
  external double inferenceTimeMs;

  @ffi.Int32()
  external int detectedClassId;

  @ffi.Float()
  external double confidence;
}

final class NativeDetection extends ffi.Struct {
  @ffi.Int32()
  external int classId;

  @ffi.Float()
  external double confidence;

  @ffi.Float()
  external double x;

  @ffi.Float()
  external double y;

  @ffi.Float()
  external double width;

  @ffi.Float()
  external double height;
}

typedef CGetVersion = ffi.Int32 Function();
typedef DartGetVersion = int Function();

typedef CGetNcnnVulkan = ffi.Int32 Function();
typedef DartGetNcnnVulkan = int Function();

typedef CLoadNanoDetModel =
    ffi.Int32 Function(
      ffi.Pointer<ffi.Char> paramPath,
      ffi.Pointer<ffi.Char> binPath,
      ffi.Int32 useGpu,
    );
typedef DartLoadNanoDetModel =
    int Function(
      ffi.Pointer<ffi.Char> paramPath,
      ffi.Pointer<ffi.Char> binPath,
      int useGpu,
    );

typedef CIsNanoDetModelLoaded = ffi.Int32 Function();
typedef DartIsNanoDetModelLoaded = int Function();

typedef CGetNanoDetBackend = ffi.Int32 Function();
typedef DartGetNanoDetBackend = int Function();

typedef CUnloadNanoDetModel = ffi.Void Function();
typedef DartUnloadNanoDetModel = void Function();

typedef CDetectRgbImage =
    ffi.Int32 Function(
      ffi.Pointer<ffi.Uint8> rgbBytes,
      ffi.Int32 width,
      ffi.Int32 height,
      ffi.Float probabilityThreshold,
      ffi.Float nmsThreshold,
      ffi.Pointer<NativeDetection> output,
      ffi.Int32 maxOutput,
      ffi.Pointer<ffi.Float> inferenceTimeMs,
    );
typedef DartDetectRgbImage =
    int Function(
      ffi.Pointer<ffi.Uint8> rgbBytes,
      int width,
      int height,
      double probabilityThreshold,
      double nmsThreshold,
      ffi.Pointer<NativeDetection> output,
      int maxOutput,
      ffi.Pointer<ffi.Float> inferenceTimeMs,
    );

typedef CProcessImageFrame =
    NativeAIInferenceResult Function(
      ffi.Pointer<ffi.Uint8> imageBytes,
      ffi.Int32 width,
      ffi.Int32 height,
      ffi.Int32 format,
    );
typedef DartProcessImageFrame =
    NativeAIInferenceResult Function(
      ffi.Pointer<ffi.Uint8> imageBytes,
      int width,
      int height,
      int format,
    );

class NativeAIBindings {
  late final ffi.DynamicLibrary _nativeLib;

  late final DartGetVersion _getVersion;
  late final DartGetNcnnVulkan _getNcnnVulkan;
  late final DartLoadNanoDetModel _loadNanoDetModel;
  late final DartIsNanoDetModelLoaded _isNanoDetModelLoaded;
  late final DartGetNanoDetBackend _getNanoDetBackend;
  late final DartUnloadNanoDetModel _unloadNanoDetModel;
  late final DartDetectRgbImage _detectRgbImage;
  late final DartProcessImageFrame _processImageFrame;

  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  NativeAIBindings() {
    _loadLibrary();
  }

  void _loadLibrary() {
    try {
      if (Platform.isAndroid) {
        _nativeLib = ffi.DynamicLibrary.open('libnative_ai_engine.so');
      } else {
        _nativeLib = ffi.DynamicLibrary.process();
      }

      _getVersion = _nativeLib
          .lookup<ffi.NativeFunction<CGetVersion>>('get_ai_engine_version')
          .asFunction();
      _getNcnnVulkan = _nativeLib
          .lookup<ffi.NativeFunction<CGetNcnnVulkan>>('get_ncnn_has_vulkan')
          .asFunction();
      _loadNanoDetModel = _nativeLib
          .lookup<ffi.NativeFunction<CLoadNanoDetModel>>('load_nanodet_model')
          .asFunction();
      _isNanoDetModelLoaded = _nativeLib
          .lookup<ffi.NativeFunction<CIsNanoDetModelLoaded>>(
            'is_nanodet_model_loaded',
          )
          .asFunction();
      _getNanoDetBackend = _nativeLib
          .lookup<ffi.NativeFunction<CGetNanoDetBackend>>('get_nanodet_backend')
          .asFunction();
      _unloadNanoDetModel = _nativeLib
          .lookup<ffi.NativeFunction<CUnloadNanoDetModel>>(
            'unload_nanodet_model',
          )
          .asFunction();
      _detectRgbImage = _nativeLib
          .lookup<ffi.NativeFunction<CDetectRgbImage>>('detect_rgb_image')
          .asFunction();
      _processImageFrame = _nativeLib
          .lookup<ffi.NativeFunction<CProcessImageFrame>>('process_image_frame')
          .asFunction();

      _isLoaded = true;
    } catch (_) {
      _isLoaded = false;
    }
  }

  int getEngineVersion() {
    if (!_isLoaded) return -1;
    return _getVersion();
  }

  bool hasNcnnVulkan() {
    if (!_isLoaded) return false;
    return _getNcnnVulkan() == 1;
  }

  int loadNanoDetModel(
    ffi.Pointer<ffi.Char> paramPath,
    ffi.Pointer<ffi.Char> binPath,
    bool useGpu,
  ) {
    if (!_isLoaded) return -100;
    return _loadNanoDetModel(paramPath, binPath, useGpu ? 1 : 0);
  }

  bool isNanoDetModelLoaded() {
    if (!_isLoaded) return false;
    return _isNanoDetModelLoaded() == 1;
  }

  int getNanoDetBackend() {
    if (!_isLoaded) return -1;
    return _getNanoDetBackend();
  }

  void unloadNanoDetModel() {
    if (!_isLoaded) return;
    _unloadNanoDetModel();
  }

  int detectRgbImage({
    required ffi.Pointer<ffi.Uint8> rgbBytes,
    required int width,
    required int height,
    required double probabilityThreshold,
    required double nmsThreshold,
    required ffi.Pointer<NativeDetection> output,
    required int maxOutput,
    required ffi.Pointer<ffi.Float> inferenceTimeMs,
  }) {
    if (!_isLoaded) return -100;

    return _detectRgbImage(
      rgbBytes,
      width,
      height,
      probabilityThreshold,
      nmsThreshold,
      output,
      maxOutput,
      inferenceTimeMs,
    );
  }

  NativeAIInferenceResult processFrame(
    ffi.Pointer<ffi.Uint8> bytes,
    int width,
    int height,
    int format,
  ) {
    if (!_isLoaded) {
      throw StateError('Native AI library is not loaded');
    }
    return _processImageFrame(bytes, width, height, format);
  }
}
