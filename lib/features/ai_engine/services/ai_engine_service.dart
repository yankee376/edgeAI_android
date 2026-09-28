import 'dart:ffi' as ffi;
import 'dart:typed_data' as typed;

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';

import '../domain/detection.dart';
import '../domain/detection_batch.dart';
import '../native/native_ai_bindings.dart';
import 'model_asset_service.dart';

class AIEngineService {
  final NativeAIBindings _bindings = NativeAIBindings();
  final ModelAssetService _modelAssetService = ModelAssetService();

  int _lastModelLoadCode = -999;

  bool get isNativeLoaded => _bindings.isLoaded;
  bool get isModelLoaded => _bindings.isNanoDetModelLoaded();
  int get lastModelLoadCode => _lastModelLoadCode;
  int get modelBackend => _bindings.getNanoDetBackend();

  String get modelBackendName {
    return switch (modelBackend) {
      1 => 'Vulkan GPU',
      0 => 'CPU',
      _ => 'Not loaded',
    };
  }

  int getVersion() => _bindings.getEngineVersion();
  bool hasVulkanGPU() => _bindings.hasNcnnVulkan();

  Future<bool> initializeModel({bool preferGpu = false}) async {
    if (!_bindings.isLoaded) {
      _lastModelLoadCode = -100;
      return false;
    }

    ffi.Pointer<Utf8>? paramPathPointer;
    ffi.Pointer<Utf8>? binPathPointer;

    try {
      final modelFiles = await _modelAssetService.prepareNanoDetModel();
      paramPathPointer = modelFiles.paramPath.toNativeUtf8();
      binPathPointer = modelFiles.binPath.toNativeUtf8();

      _lastModelLoadCode = _bindings.loadNanoDetModel(
        paramPathPointer.cast<ffi.Char>(),
        binPathPointer.cast<ffi.Char>(),
        preferGpu,
      );

      debugPrint('NanoDet load result: $_lastModelLoadCode');
      return _lastModelLoadCode == 0 && _bindings.isNanoDetModelLoaded();
    } catch (error, stackTrace) {
      _lastModelLoadCode = -101;
      debugPrint('NanoDet initialization error: $error');
      debugPrintStack(stackTrace: stackTrace);
      return false;
    } finally {
      if (paramPathPointer != null) calloc.free(paramPathPointer);
      if (binPathPointer != null) calloc.free(binPathPointer);
    }
  }

  DetectionBatch detectRgb({
    required typed.Uint8List rgbBytes,
    required int width,
    required int height,
    double probabilityThreshold = 0.40,
    double nmsThreshold = 0.50,
    int maxDetections = 100,
  }) {
    if (!_bindings.isLoaded) {
      throw StateError('Native AI library is not loaded');
    }
    if (!_bindings.isNanoDetModelLoaded()) {
      throw StateError('NanoDet model is not loaded');
    }
    if (rgbBytes.length != width * height * 3) {
      throw ArgumentError(
        'RGB byte count ${rgbBytes.length} does not match ${width}x$height x 3',
      );
    }

    final rgbPointer = calloc<ffi.Uint8>(rgbBytes.length);
    final outputPointer = calloc<NativeDetection>(maxDetections);
    final timePointer = calloc<ffi.Float>();

    try {
      rgbPointer.asTypedList(rgbBytes.length).setAll(0, rgbBytes);

      final count = _bindings.detectRgbImage(
        rgbBytes: rgbPointer,
        width: width,
        height: height,
        probabilityThreshold: probabilityThreshold,
        nmsThreshold: nmsThreshold,
        output: outputPointer,
        maxOutput: maxDetections,
        inferenceTimeMs: timePointer,
      );

      if (count < 0) {
        throw StateError('NanoDet inference failed with code $count');
      }

      final detections = List<Detection>.generate(count, (index) {
        final native = outputPointer[index];
        return Detection(
          classId: native.classId,
          confidence: native.confidence,
          x: native.x,
          y: native.y,
          width: native.width,
          height: native.height,
        );
      }, growable: false);

      return DetectionBatch(
        detections: detections,
        inferenceTimeMs: timePointer.value,
      );
    } finally {
      calloc.free(rgbPointer);
      calloc.free(outputPointer);
      calloc.free(timePointer);
    }
  }

  void unloadModel() {
    _bindings.unloadNanoDetModel();
  }

  bool get isYolo26ModelLoaded => _bindings.isYolo26ModelLoaded();
  int get yolo26Backend => _bindings.getYolo26Backend();
  int get yolo26CpuThreadCount => _bindings.getYolo26CpuThreadCount();
  int get yolo26CpuCoreCount => _bindings.getYolo26CpuCoreCount();
  int get yolo26GpuCount => _bindings.getYolo26GpuCount();
  String get yolo26GpuName => _bindings.getYolo26GpuName();
  String get yolo26BackendName => switch (yolo26Backend) {
    1 => 'Vulkan GPU',
    0 => 'CPU',
    _ => 'Not loaded',
  };

  Future<bool> initializeYolo26Model({bool preferGpu = false}) async {
    if (!_bindings.isLoaded) {
      _lastModelLoadCode = -100;
      return false;
    }
    ffi.Pointer<Utf8>? param;
    ffi.Pointer<Utf8>? bin;
    try {
      final files = await _modelAssetService.prepareYolo26Model();
      param = files.paramPath.toNativeUtf8();
      bin = files.binPath.toNativeUtf8();
      _lastModelLoadCode = _bindings.loadYolo26Model(
        param.cast<ffi.Char>(),
        bin.cast<ffi.Char>(),
        preferGpu,
      );
      debugPrint('YOLO26 load result: $_lastModelLoadCode');
      return _lastModelLoadCode == 0 && _bindings.isYolo26ModelLoaded();
    } catch (error, stackTrace) {
      _lastModelLoadCode = -101;
      debugPrint('YOLO26 initialization error: $error');
      debugPrintStack(stackTrace: stackTrace);
      return false;
    } finally {
      if (param != null) calloc.free(param);
      if (bin != null) calloc.free(bin);
    }
  }

  DetectionBatch detectYolo26({
    required typed.Uint8List rgbBytes,
    required int width,
    required int height,
    double probabilityThreshold = 0.40,
    double nmsThreshold = 0.50,
    int maxDetections = 100,
  }) {
    if (!_bindings.isYolo26ModelLoaded()) {
      throw StateError('YOLO26 model is not loaded');
    }
    if (width <= 0 || height <= 0 || rgbBytes.length != width * height * 3) {
      throw ArgumentError('Invalid RGB image dimensions or byte count');
    }
    if (maxDetections <= 0) {
      throw ArgumentError.value(maxDetections, 'maxDetections');
    }
    final rgb = calloc<ffi.Uint8>(rgbBytes.length);
    final output = calloc<NativeDetection>(maxDetections);
    final preprocessTime = calloc<ffi.Float>();
    final inferenceTime = calloc<ffi.Float>();
    final postprocessTime = calloc<ffi.Float>();
    try {
      rgb.asTypedList(rgbBytes.length).setAll(0, rgbBytes);
      final count = _bindings.detectYolo26Image(
        rgbBytes: rgb,
        width: width,
        height: height,
        probabilityThreshold: probabilityThreshold,
        nmsThreshold: nmsThreshold,
        output: output,
        maxOutput: maxDetections,
        preprocessTimeMs: preprocessTime,
        inferenceTimeMs: inferenceTime,
        postprocessTimeMs: postprocessTime,
      );
      if (count < 0) {
        throw StateError('YOLO26 inference failed with code $count');
      }
      return DetectionBatch(
        detections: List<Detection>.generate(count, (i) {
          final d = output[i];
          return Detection(
            classId: d.classId,
            confidence: d.confidence,
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
          );
        }, growable: false),
        inferenceTimeMs: inferenceTime.value,
        preprocessTimeMs: preprocessTime.value,
        postprocessTimeMs: postprocessTime.value,
      );
    } finally {
      calloc.free(rgb);
      calloc.free(output);
      calloc.free(preprocessTime);
      calloc.free(inferenceTime);
      calloc.free(postprocessTime);
    }
  }

  void unloadYolo26Model() => _bindings.unloadYolo26Model();

  bool get isCurrencyModelLoaded => _bindings.isCurrencyModelLoaded();

  Future<bool> initializeCurrencyModel({bool preferGpu = false}) async {
    if (!_bindings.isLoaded) return false;

    ffi.Pointer<Utf8>? paramPathPointer;
    ffi.Pointer<Utf8>? binPathPointer;
    try {
      final modelFiles = await _modelAssetService.prepareCurrencyModel();
      paramPathPointer = modelFiles.paramPath.toNativeUtf8();
      binPathPointer = modelFiles.binPath.toNativeUtf8();

      final code = _bindings.loadCurrencyModel(
        paramPathPointer.cast<ffi.Char>(),
        binPathPointer.cast<ffi.Char>(),
        preferGpu,
      );
      return code == 0 && _bindings.isCurrencyModelLoaded();
    } finally {
      if (paramPathPointer != null) calloc.free(paramPathPointer);
      if (binPathPointer != null) calloc.free(binPathPointer);
    }
  }

  DetectionBatch detectCurrency({
    required typed.Uint8List rgbBytes,
    required int width,
    required int height,
    double probabilityThreshold = 0.40,
    double nmsThreshold = 0.50,
    int maxDetections = 100,
  }) {
    if (!_bindings.isCurrencyModelLoaded()) {
      throw StateError('Currency model is not loaded');
    }

    final rgbPointer = calloc<ffi.Uint8>(rgbBytes.length);
    final outputPointer = calloc<NativeDetection>(maxDetections);
    final timePointer = calloc<ffi.Float>();

    try {
      rgbPointer.asTypedList(rgbBytes.length).setAll(0, rgbBytes);

      final count = _bindings.detectCurrencyImage(
        rgbBytes: rgbPointer,
        width: width,
        height: height,
        probabilityThreshold: probabilityThreshold,
        nmsThreshold: nmsThreshold,
        output: outputPointer,
        maxOutput: maxDetections,
        inferenceTimeMs: timePointer,
      );

      if (count < 0) {
        throw StateError('Currency inference failed with code $count');
      }

      final detections = List<Detection>.generate(count, (index) {
        final native = outputPointer[index];
        return Detection(
          classId: native.classId,
          confidence: native.confidence,
          x: native.x,
          y: native.y,
          width: native.width,
          height: native.height,
        );
      }, growable: false);

      return DetectionBatch(
        detections: detections,
        inferenceTimeMs: timePointer.value,
      );
    } finally {
      calloc.free(rgbPointer);
      calloc.free(outputPointer);
      calloc.free(timePointer);
    }
  }
}
