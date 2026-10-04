import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Isolated 416x416 LiteRT camera baseline for comparing against the 640 app.
class LiteRtCameraScreen extends StatefulWidget {
  const LiteRtCameraScreen({super.key});

  @override
  State<LiteRtCameraScreen> createState() => _LiteRtCameraScreenState();
}

class _LiteRtCameraScreenState extends State<LiteRtCameraScreen> {
  static const _modelPath = 'assets/models/best_416_w8a32.tflite';

  final YOLOViewController _controller = YOLOViewController();
  final Stopwatch _fpsClock = Stopwatch();
  Map<String, dynamic>? _lastFrame;
  DateTime? _lastUiUpdate;
  String? _error;
  bool _loaded = false;
  bool _paused = false;
  int _completedFrames = 0;
  double _averageFps = 0;

  double _number(String key) => (_lastFrame?[key] as num?)?.toDouble() ?? 0.0;

  void _resetCounter() {
    _fpsClock
      ..stop()
      ..reset();
    setState(() {
      _completedFrames = 0;
      _averageFps = 0;
      _lastFrame = null;
      _lastUiUpdate = null;
    });
  }

  void _onFrame(Map<String, dynamic> data) {
    if (!mounted || _paused) return;
    if (!_fpsClock.isRunning) _fpsClock.start();
    _completedFrames++;
    final now = DateTime.now();
    if (_lastUiUpdate != null &&
        now.difference(_lastUiUpdate!).inMilliseconds < 250) {
      return;
    }
    _lastUiUpdate = now;
    final seconds =
        _fpsClock.elapsedMicroseconds / Duration.microsecondsPerSecond;
    setState(() {
      _lastFrame = data;
      _averageFps = seconds > 0 && _completedFrames > 1
          ? (_completedFrames - 1) / seconds
          : 0;
    });
  }

  Future<void> _togglePause() async {
    if (_paused) {
      await _controller.resume();
      if (!mounted) return;
      _resetCounter();
      setState(() => _paused = false);
    } else {
      await _controller.pause();
      if (mounted) setState(() => _paused = true);
      _fpsClock.stop();
    }
  }

  Widget _stage(String label, double value) => Text(
    '$label: ${value.toStringAsFixed(1)} ms',
    style: const TextStyle(color: Colors.white),
  );

  @override
  void dispose() {
    _fpsClock.stop();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cameraMs = _number('cameraToBitmapMs');
    final preMs = _number('preMs');
    final inferMs = _number('inferenceMs');
    final postMs = _number('postMs');
    final modelMs = _number('processingTimeMs');
    final nativeFps = _number('fps');
    final backend = _lastFrame?['accelerator'] as String?;
    final detections = _lastFrame?['detections'] as List<dynamic>?;
    final width = (_lastFrame?['imageWidth'] as num?)?.toInt() ?? 0;
    final height = (_lastFrame?['imageHeight'] as num?)?.toInt() ?? 0;
    final measuredMs = cameraMs + modelMs;
    final unspecificMs = math.max(0.0, modelMs - preMs - inferMs - postMs);

    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeAI • LiteRT 416'),
        actions: [
          IconButton(
            tooltip: 'Đặt lại bộ đếm FPS',
            onPressed: _resetCounter,
            icon: const Icon(Icons.restart_alt),
          ),
          IconButton(
            tooltip: 'Đổi camera',
            onPressed: () => _controller.switchCamera(),
            icon: const Icon(Icons.switch_camera),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: YOLOView(
              modelPath: _modelPath,
              task: YOLOTask.detect,
              useGpu: true,
              controller: _controller,
              streamingConfig: const YOLOStreamingConfig.minimal(),
              onStreamingData: _onFrame,
              onModelLoad: (_, _) {
                if (mounted) setState(() => _loaded = true);
              },
              onModelError: (error, _, _) {
                if (mounted) {
                  setState(() {
                    _error = error.toString();
                    _loaded = false;
                  });
                }
              },
            ),
          ),
          SafeArea(
            top: false,
            child: Container(
              width: double.infinity,
              color: Colors.black87,
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'AI: ${_averageFps.toStringAsFixed(1)} FPS',
                    style: const TextStyle(
                      color: Colors.lightGreenAccent,
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Backend đang chạy: ${backend == null ? (_loaded ? 'đang chờ frame' : 'chưa nạp') : 'LiteRT $backend'}',
                    style: const TextStyle(color: Colors.white70),
                  ),
                  const Text(
                    'Model: YOLO26n • 99 class • 416×416 • W8A32',
                    style: TextStyle(color: Colors.white70),
                  ),
                  Text(
                    'CameraX RGBA • frame sau xoay: $width×$height • ${detections?.length ?? 0} vật thể',
                    style: const TextStyle(color: Colors.white70),
                  ),
                  Text(
                    'FPS nội bộ plugin: ${nativeFps.toStringAsFixed(1)}',
                    style: const TextStyle(color: Colors.white70),
                  ),
                  const SizedBox(height: 4),
                  _stage('CameraX → Bitmap', cameraMs),
                  _stage('Resize/letterbox/normalize', preMs),
                  _stage('LiteRT inference', inferMs),
                  _stage('Decode/NMS', postMs),
                  if (unspecificMs >= 1)
                    _stage('Phần còn lại trong detector', unspecificMs),
                  _stage('Tổng detector', modelMs),
                  _stage('Tổng phần đã đo', measuredMs),
                  const Text(
                    'Đối chứng camera điện thoại; chưa chạy Pi, LLM, cảnh báo.',
                    style: TextStyle(color: Colors.white54, fontSize: 11),
                  ),
                  if (_error != null)
                    Text(
                      _error!,
                      style: const TextStyle(color: Colors.redAccent),
                    ),
                  const SizedBox(height: 5),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: _loaded ? _togglePause : null,
                      child: Text(
                        _paused
                            ? 'YOLO đã tạm dừng • Nhấn để bật'
                            : 'YOLO đang hoạt động • Nhấn để tạm dừng',
                      ),
                    ),
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
