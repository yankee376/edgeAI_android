import 'dart:async';
import 'dart:collection';

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
  final List<_PlaneData> planes;
  const _FrameData(this.width, this.height, this.rotation, this.mirror, this.planes);
}

class _DetectionFrame {
  final int width, height;
  final List<Detection> detections;
  final double inferenceMs;
  const _DetectionFrame(this.width, this.height, this.detections, this.inferenceMs);
}

// Runs off the UI isolate, including the synchronous NCNN call.
_DetectionFrame _detectFrame(_FrameData frame) {
  if (frame.planes.length != 3) {
    throw StateError('Chỉ hỗ trợ camera stream YUV420 có 3 planes');
  }
  final rotated = frame.rotation == 90 || frame.rotation == 270;
  final width = rotated ? frame.height : frame.width;
  final height = rotated ? frame.width : frame.height;
  final rgb = Uint8List(width * height * 3);
  final yp = frame.planes[0], up = frame.planes[1], vp = frame.planes[2];

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
      rgb[offset + 1] = (luminance - 0.344136 * u - 0.714136 * v).round().clamp(0, 255);
      rgb[offset + 2] = (luminance + 1.772 * u).round().clamp(0, 255);
    }
  }
  final batch = AIEngineService().detectYolo26(rgbBytes: rgb, width: width, height: height);
  return _DetectionFrame(width, height, batch.detections, batch.inferenceTimeMs);
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
  final Queue<DateTime> _completed = Queue<DateTime>();
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
    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) => _refreshFps());
    if (widget.cameras.isEmpty) {
      _error = 'Không tìm thấy camera.';
    } else {
      _openCamera();
    }
  }

  void _refreshFps() {
    if (!mounted) return;
    final cutoff = DateTime.now().subtract(const Duration(seconds: 3));
    while (_completed.isNotEmpty && _completed.first.isBefore(cutoff)) {
      _completed.removeFirst();
    }
    setState(() => _fps = _running ? _completed.length / 3 : 0);
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
      _completed.clear();
      _fps = 0;
    });
    // Avoid opening a second camera while the old native inference is running.
    while (_busy) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
    await old?.dispose();
    if (!mounted || generation != _generation) return;
    final camera = CameraController(widget.cameras[_cameraIndex],
        ResolutionPreset.medium, enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420);
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
      });
    } catch (e) {
      await camera.dispose();
      if (mounted && generation == _generation) setState(() => _error = 'Lỗi camera stream: $e');
    }
  }

  void _onImage(CameraImage image, int generation) {
    if (!mounted || !_running || _busy || generation != _generation) return;
    _busy = true; // Keep only a new frame when the detector is idle.
    try {
      final camera = widget.cameras[_cameraIndex];
      final frame = _FrameData(image.width, image.height, camera.sensorOrientation,
          camera.lensDirection == CameraLensDirection.front,
          image.planes.map((p) => _PlaneData(
            Uint8List.fromList(p.bytes), p.bytesPerRow, p.bytesPerPixel ?? 1,
          )).toList());
      _processFrame(frame, generation);
    } catch (e) {
      _busy = false;
      setState(() {
        _running = false;
        _error = 'Đọc frame thất bại: $e';
      });
    }
  }

  Future<void> _processFrame(_FrameData frame, int generation) async {
    try {
      final result = await compute(_detectFrame, frame);
      if (!mounted || !_running || generation != _generation) return;
      _completed.addLast(DateTime.now());
      setState(() => _result = result);
      _refreshFps();
    } catch (e, stack) {
      debugPrint('YOLO camera stream: $e\n$stack');
      if (mounted && generation == _generation) {
        setState(() {
          _running = false;
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
        _completed.clear();
        _fps = 0;
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
        final ok = await _engine.initializeYolo26Model(preferGpu: false);
        if (!mounted) return;
        if (!ok) {
          setState(() => _modelError =
              'Không load được YOLO26: ${_engine.lastModelLoadCode}');
          return;
        }
        _loaded = true;
      }
      final generation = ++_generation;
      await _controller!.startImageStream((image) => _onImage(image, generation));
      if (mounted && generation == _generation) setState(() => _running = true);
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
    _fpsTimer?.cancel();
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
      appBar: AppBar(title: const Text('EdgeAI YOLO26', style: TextStyle(fontSize: 18)),
          actions: [
            if (widget.cameras.length > 1)
            IconButton(icon: const Icon(Icons.switch_camera), tooltip: 'Đổi camera',
                onPressed: _loading || _busy ? null : () {
                  _cameraIndex = (_cameraIndex + 1) % widget.cameras.length;
                  _openCamera();
                }),
          ]),
      body: _error != null
          ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
              Text(_error!, textAlign: TextAlign.center),
              TextButton(onPressed: _openCamera, child: const Text('Thử lại')),
            ]))
          : controller == null
              ? const Center(child: CircularProgressIndicator())
              : Column(children: [
                  Expanded(child: Center(child: CameraPreview(controller,
                    child: _result == null ? null : CustomPaint(
                      painter: DetectionPainter(
                        detections: _result!.detections,
                        imageWidth: _result!.width,
                        imageHeight: _result!.height,
                      ),
                    ),
                  ))),
                  SafeArea(top: false, child: Container(
                    width: double.infinity, padding: const EdgeInsets.all(12),
                    color: Colors.black87,
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text('YOLO: ${_fps.toStringAsFixed(1)} FPS',
                          style: const TextStyle(color: Colors.lightGreenAccent,
                              fontSize: 26, fontWeight: FontWeight.bold)),
                      Text(_result == null ? 'Chưa có kết quả'
                          : '${_result!.detections.length} vật thể • ${_result!.inferenceMs.toStringAsFixed(1)} ms suy luận',
                          style: const TextStyle(color: Colors.white)),
                      const SizedBox(height: 8),
                      SizedBox(width: double.infinity, child: FilledButton(
                        onPressed: _loading ? null : _toggleYolo,
                        child: Text(_loading ? 'Đang chuyển trạng thái…'
                            : _running ? 'YOLO đã hoạt động • Nhấn để tạm dừng'
                            : 'YOLO đã tạm dừng • Nhấn để bật'),
                      )),
                      if (_modelError != null) Text(_modelError!,
                          style: const TextStyle(color: Colors.redAccent)),
                    ]),
                  )),
                ]),
    );
  }
}
