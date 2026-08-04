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
}
