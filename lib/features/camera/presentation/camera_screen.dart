import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../ai_engine/domain/detection.dart';
import '../../ai_engine/services/ai_engine_service.dart';
import 'detection_painter.dart';

// CameraImage is a plugin object. Send copies of its YUV planes to the worker.
class _PlaneData {
  final Uint8List bytes;
  final int rowStride;
  final int pixelStride;
  const _PlaneData(this.bytes, this.rowStride, this.pixelStride);
}

class _FrameData {
  final int width, height, rotation;
  final bool mirror;
  final double cameraCopyMs;
  final List<_PlaneData> planes;
  const _FrameData(
    this.width,
    this.height,
    this.rotation,
    this.mirror,
    this.cameraCopyMs,
    this.planes,
  );
}

class _DetectionFrame {
  final int width, height;
  final List<Detection> detections;
  final double cameraCopyMs;
  final double yuvToRgbMs;
  final double nativeCallMs;
  final double nativePreprocessMs;
  final double inferenceMs;
  final double nativePostprocessMs;
  final double workerTotalMs;
  final double computeRoundTripMs;
  const _DetectionFrame({
    required this.width,
    required this.height,
    required this.detections,
    required this.cameraCopyMs,
    required this.yuvToRgbMs,
    required this.nativeCallMs,
    required this.nativePreprocessMs,
    required this.inferenceMs,
    required this.nativePostprocessMs,
    required this.workerTotalMs,
    this.computeRoundTripMs = 0,
  });

  _DetectionFrame withComputeRoundTrip(double value) => _DetectionFrame(
    width: width,
    height: height,
    detections: detections,
    cameraCopyMs: cameraCopyMs,
    yuvToRgbMs: yuvToRgbMs,
    nativeCallMs: nativeCallMs,
    nativePreprocessMs: nativePreprocessMs,
    inferenceMs: inferenceMs,
    nativePostprocessMs: nativePostprocessMs,
    workerTotalMs: workerTotalMs,
    computeRoundTripMs: value,
  );

  double get isolateOverheadMs =>
      (computeRoundTripMs - workerTotalMs).clamp(0, double.infinity);
  double get ffiOverheadMs =>
      (nativeCallMs - nativePreprocessMs - inferenceMs - nativePostprocessMs)
          .clamp(0, double.infinity);
  double get pipelineTotalMs => cameraCopyMs + computeRoundTripMs;
}

// Runs off the UI isolate, including the synchronous NCNN call.
_DetectionFrame _detectFrame(_FrameData frame) {
  final workerClock = Stopwatch()..start();
  if (frame.planes.length != 3) {
    throw StateError('Chỉ hỗ trợ camera stream YUV420 có 3 planes');
  }
  final rotated = frame.rotation == 90 || frame.rotation == 270;
  final width = rotated ? frame.height : frame.width;
  final height = rotated ? frame.width : frame.height;
  final rgb = Uint8List(width * height * 3);
  final yp = frame.planes[0], up = frame.planes[1], vp = frame.planes[2];

  final conversionClock = Stopwatch()..start();
  for (var y = 0; y < frame.height; y++) {
    for (var x = 0; x < frame.width; x++) {
      final luminance = yp.bytes[y * yp.rowStride + x * yp.pixelStride];
      final cx = x ~/ 2, cy = y ~/ 2;
      final u = up.bytes[cy * up.rowStride + cx * up.pixelStride] - 128;
      final v = vp.bytes[cy * vp.rowStride + cx * vp.pixelStride] - 128;
      var dx = x, dy = y;
      switch (frame.rotation) {
        case 90:
          dx = frame.height - 1 - y;
          dy = x;
        case 180:
          dx = frame.width - 1 - x;
          dy = frame.height - 1 - y;
        case 270:
          dx = y;
          dy = frame.width - 1 - x;
      }
      if (frame.mirror) dx = width - 1 - dx;
      final offset = (dy * width + dx) * 3;
      rgb[offset] = (luminance + 1.402 * v).round().clamp(0, 255);
      rgb[offset + 1] = (luminance - 0.344136 * u - 0.714136 * v).round().clamp(
        0,
        255,
      );
      rgb[offset + 2] = (luminance + 1.772 * u).round().clamp(0, 255);
    }
  }
  conversionClock.stop();

  final nativeClock = Stopwatch()..start();
  final batch = AIEngineService().detectYolo26(
    rgbBytes: rgb,
    width: width,
    height: height,
  );
  nativeClock.stop();
  workerClock.stop();
  return _DetectionFrame(
    width: width,
    height: height,
    detections: batch.detections,
    cameraCopyMs: frame.cameraCopyMs,
    yuvToRgbMs: conversionClock.elapsedMicroseconds / 1000,
    nativeCallMs: nativeClock.elapsedMicroseconds / 1000,
    nativePreprocessMs: batch.preprocessTimeMs,
    inferenceMs: batch.inferenceTimeMs,
    nativePostprocessMs: batch.postprocessTimeMs,
    workerTotalMs: workerClock.elapsedMicroseconds / 1000,
  );
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  final AIEngineService _engine = AIEngineService();
  final Stopwatch _fpsClock = Stopwatch();
  int _completedFrames = 0;
  Timer? _fpsTimer;
  int _cameraIndex = 0, _generation = 0;
  bool _loading = false, _loaded = false, _running = false, _busy = false;
  String? _error;
  String? _modelError;
  double _fps = 0;
  _DetectionFrame? _result;

  @override
  void initState() {
    super.initState();
    if (widget.cameras.isEmpty) {
      _error = 'Không tìm thấy camera.';
    } else {
      _openCamera();
    }
  }

  // Average throughput since the stream started. Keeping the completed-frame
  // count prevents FPS from falling to zero between very slow frames.
  void _refreshFps() {
    if (!mounted || !_running) return;
    final elapsedUs = _fpsClock.elapsedMicroseconds;
    if (elapsedUs <= 0) return;
    final fps = _completedFrames * Duration.microsecondsPerSecond / elapsedUs;
    setState(() => _fps = fps);
  }

  void _startFpsMeasurement() {
    _resetFpsMeasurement();
    _fpsClock.start();
    _fpsTimer = Timer.periodic(
      const Duration(seconds: 1),
      (_) => _refreshFps(),
    );
  }

  // Call inside setState when the screen remains mounted.
  void _resetFpsMeasurement() {
    _fpsTimer?.cancel();
    _fpsTimer = null;
    _fpsClock.stop();
    _fpsClock.reset();
    _completedFrames = 0;
    _fps = 0;
  }

  Future<void> _openCamera() async {
    final generation = ++_generation;
    final old = _controller;
    final wasRunning = _running;
    setState(() {
      _controller = null;
      _running = false;
      _error = null;
      _result = null;
      _resetFpsMeasurement();
    });
    // Avoid opening a second camera while the old native inference is running.
    while (_busy) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
    await old?.dispose();
    if (!mounted || generation != _generation) return;
    final camera = CameraController(
      widget.cameras[_cameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    try {
      await camera.initialize();
      if (!mounted || generation != _generation) {
        await camera.dispose();
        return;
      }
      if (wasRunning && _loaded) {
        await camera.startImageStream((image) => _onImage(image, generation));
      }
      if (!mounted || generation != _generation) {
        await camera.dispose();
        return;
      }
      setState(() {
        _controller = camera;
        _running = wasRunning && _loaded;
        if (_running) _startFpsMeasurement();
      });
    } catch (e) {
      await camera.dispose();
      if (mounted && generation == _generation) {
        setState(() => _error = 'Lỗi camera stream: $e');
      }
    }
  }

  void _onImage(CameraImage image, int generation) {
    if (!mounted || !_running || _busy || generation != _generation) return;
    _busy = true; // Keep only a new frame when the detector is idle.
    try {
      final copyClock = Stopwatch()..start();
      final camera = widget.cameras[_cameraIndex];
      final planes = image.planes
          .map(
            (p) => _PlaneData(
              Uint8List.fromList(p.bytes),
              p.bytesPerRow,
              p.bytesPerPixel ?? 1,
            ),
          )
          .toList();
      copyClock.stop();
      final frame = _FrameData(
        image.width,
        image.height,
        camera.sensorOrientation,
        camera.lensDirection == CameraLensDirection.front,
        copyClock.elapsedMicroseconds / 1000,
        planes,
      );
      _processFrame(frame, generation);
    } catch (e) {
      _busy = false;
      setState(() {
        _running = false;
        _resetFpsMeasurement();
        _error = 'Đọc frame thất bại: $e';
      });
    }
  }

  Future<void> _processFrame(_FrameData frame, int generation) async {
    try {
      final computeClock = Stopwatch()..start();
      final workerResult = await compute(_detectFrame, frame);
      computeClock.stop();
      if (!mounted || !_running || generation != _generation) return;
      final result = workerResult.withComputeRoundTrip(
        computeClock.elapsedMicroseconds / 1000,
      );
      _completedFrames++;
      final elapsedUs = _fpsClock.elapsedMicroseconds;
      final fps = elapsedUs <= 0
          ? 0.0
          : _completedFrames * Duration.microsecondsPerSecond / elapsedUs;
      setState(() {
        _result = result;
        _fps = fps;
      });
    } catch (e, stack) {
      debugPrint('YOLO camera stream: $e\n$stack');
      if (mounted && generation == _generation) {
        setState(() {
          _running = false;
          _resetFpsMeasurement();
          _error = 'Lỗi nhận diện: $e';
        });
      }
    } finally {
      _busy = false;
    }
  }

  Future<void> _toggleYolo() async {
    if (_loading || _controller == null || _error != null) return;
    if (_running) {
      ++_generation;
      setState(() {
        _running = false;
        _result = null;
        _resetFpsMeasurement();
        _loading = true;
      });
      try {
        await _controller!.stopImageStream();
      } catch (e) {
        if (mounted) setState(() => _modelError = 'Không dừng được stream: $e');
      } finally {
        if (mounted) setState(() => _loading = false);
      }
      return;
    }

    setState(() {
      _loading = true;
      _modelError = null;
    });
    try {
      if (!_loaded) {
        // Prefer NCNN's Vulkan backend. The native engine falls back to CPU
        // when Vulkan is unavailable or cannot be initialized.
        final ok = await _engine.initializeYolo26Model(preferGpu: true);
        if (!mounted) return;
        if (!ok) {
          setState(
            () => _modelError =
                'Không load được YOLO26: ${_engine.lastModelLoadCode}',
          );
          return;
        }
        _loaded = true;
      }
      final generation = ++_generation;
      await _controller!.startImageStream(
        (image) => _onImage(image, generation),
      );
      if (mounted && generation == _generation) {
        setState(() {
          _running = true;
          _startFpsMeasurement();
        });
      }
    } catch (e) {
      if (mounted) setState(() => _modelError = 'Không bật được YOLO: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _generation++;
    _running = false;
    _resetFpsMeasurement();
    if (_busy) {
      // Native net must remain loaded until the worker finishes.
      Future<void>(() async {
        while (_busy) {
          await Future<void>.delayed(const Duration(milliseconds: 20));
        }
        _engine.unloadYolo26Model();
      });
    } else {
      _engine.unloadYolo26Model();
    }
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeAI YOLO26', style: TextStyle(fontSize: 18)),
        actions: [
          if (widget.cameras.length > 1)
            IconButton(
              icon: const Icon(Icons.switch_camera),
              tooltip: 'Đổi camera',
              onPressed: _loading || _busy
                  ? null
                  : () {
                      _cameraIndex = (_cameraIndex + 1) % widget.cameras.length;
                      _openCamera();
                    },
            ),
        ],
      ),
      body: _error != null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(_error!, textAlign: TextAlign.center),
                  TextButton(
                    onPressed: _openCamera,
                    child: const Text('Thử lại'),
                  ),
                ],
              ),
            )
          : controller == null
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: Center(
                    child: CameraPreview(
                      controller,
                      child: _result == null
                          ? null
                          : CustomPaint(
                              painter: DetectionPainter(
                                detections: _result!.detections,
                                imageWidth: _result!.width,
                                imageHeight: _result!.height,
                              ),
                            ),
                    ),
                  ),
                ),
                SafeArea(
                  top: false,
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    color: Colors.black87,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'AI: ${_fps.toStringAsFixed(_fps < 1 ? 2 : 1)} FPS',
                          style: const TextStyle(
                            color: Colors.lightGreenAccent,
                            fontSize: 26,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          'Backend: ${_engine.yolo26BackendName}'
                          '${_engine.yolo26Backend == 1 ? ' • ${_engine.yolo26GpuName}' : ''}',
                          style: const TextStyle(color: Colors.white70),
                        ),
                        Text(
                          'CPU: ${_engine.yolo26CpuCoreCount} lõi • '
                          '${_engine.yolo26CpuThreadCount} luồng NCNN'
                          ' • GPU Vulkan: ${_engine.yolo26GpuCount} thiết bị',
                          style: const TextStyle(color: Colors.white70),
                        ),
                        const Text(
                          'Model: YOLO26n • input 640×640 • trọng số FP16',
                          style: TextStyle(color: Colors.white70),
                        ),
                        Text(
                          _engine.yolo26Backend == 1
                              ? 'NCNN chạy layer hỗ trợ Vulkan trên GPU; CPU xử lý phần còn lại.'
                              : 'Vulkan không hoạt động; model đang chạy trên CPU.',
                          style: const TextStyle(color: Colors.white70),
                        ),
                        Text(
                          _result == null
                              ? 'Chưa có kết quả'
                              : '${_result!.detections.length} vật thể\n'
                                    'Copy camera: ${_result!.cameraCopyMs.toStringAsFixed(1)} ms\n'
                                    'YUV → RGB: ${_result!.yuvToRgbMs.toStringAsFixed(1)} ms\n'
                                    'Isolate/transfer: ${_result!.isolateOverheadMs.toStringAsFixed(1)} ms\n'
                                    'FFI/copy bộ nhớ: ${_result!.ffiOverheadMs.toStringAsFixed(1)} ms\n'
                                    'Native resize/normalize: ${_result!.nativePreprocessMs.toStringAsFixed(1)} ms\n'
                                    'NCNN inference: ${_result!.inferenceMs.toStringAsFixed(1)} ms\n'
                                    'Native decode/NMS: ${_result!.nativePostprocessMs.toStringAsFixed(1)} ms\n'
                                    'Tổng pipeline: ${_result!.pipelineTotalMs.toStringAsFixed(1)} ms',
                          style: const TextStyle(color: Colors.white),
                        ),
                        const SizedBox(height: 8),
                        SizedBox(
                          width: double.infinity,
                          child: FilledButton(
                            onPressed: _loading ? null : _toggleYolo,
                            child: Text(
                              _loading
                                  ? 'Đang chuyển trạng thái…'
                                  : _running
                                  ? 'YOLO đã hoạt động • Nhấn để tạm dừng'
                                  : 'YOLO đã tạm dừng • Nhấn để bật',
                            ),
                          ),
                        ),
                        if (_modelError != null)
                          Text(
                            _modelError!,
                            style: const TextStyle(color: Colors.redAccent),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}
