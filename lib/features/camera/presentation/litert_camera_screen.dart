import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import 'detection_assistant_sheet.dart';

/// Isolated 640x640 LiteRT camera baseline for comparing against the NCNN app.
class LiteRtCameraScreen extends StatefulWidget {
  const LiteRtCameraScreen({super.key});

  @override
  State<LiteRtCameraScreen> createState() => _LiteRtCameraScreenState();
}

class _LiteRtCameraScreenState extends State<LiteRtCameraScreen> {
  static const _modelPath = 'assets/models/best_w8a32.tflite';
  static const _llmChannel = MethodChannel('edgeai/litertlm');

  final YOLOViewController _controller = YOLOViewController();
  final Stopwatch _fpsClock = Stopwatch();
  Map<String, dynamic>? _lastFrame;
  DateTime? _lastUiUpdate;
  String? _error;
  String? _llmModelPath;
  String? _llmError;
  bool _loaded = false;
  bool _llmBusy = false;
  bool _llmReady = false;
  bool _paused = false;
  int _completedFrames = 0;
  double _averageFps = 0;
  double _llmLoadMs = 0;

  double _number(String key) => (_lastFrame?[key] as num?)?.toDouble() ?? 0.0;

  @override
  void initState() {
    super.initState();
    _restoreLlmModelPath();
  }

  Future<void> _restoreLlmModelPath() async {
    try {
      final path = await _llmChannel.invokeMethod<String>('getModelPath');
      if (mounted && path != null) setState(() => _llmModelPath = path);
    } on PlatformException catch (error) {
      if (mounted) setState(() => _llmError = error.message);
    }
  }

  Future<Map<String, dynamic>?> _selectAndLoadLlm() async {
    if (_llmBusy) return null;
    setState(() {
      _llmBusy = true;
      _llmError = null;
      _llmReady = false;
    });
    try {
      final path = await _llmChannel.invokeMethod<String>('pickModel');
      if (path == null) return null;
      if (mounted) setState(() => _llmModelPath = path);
      final loaded = await _llmChannel.invokeMapMethod<String, dynamic>(
        'loadModel',
        {'path': path},
      );
      if (!mounted) return null;
      setState(() {
        _llmReady = true;
        _llmLoadMs = (loaded?['loadMs'] as num?)?.toDouble() ?? 0;
      });
      return loaded;
    } on PlatformException catch (error) {
      if (mounted) setState(() => _llmError = error.message ?? error.code);
      return null;
    } catch (error) {
      if (mounted) setState(() => _llmError = error.toString());
      return null;
    } finally {
      if (mounted) setState(() => _llmBusy = false);
    }
  }

  Future<Map<String, dynamic>?> _loadSavedLlm() async {
    final path = _llmModelPath;
    if (path == null || _llmBusy) return null;
    setState(() {
      _llmBusy = true;
      _llmError = null;
      _llmReady = false;
    });
    try {
      final loaded = await _llmChannel.invokeMapMethod<String, dynamic>(
        'loadModel',
        {'path': path},
      );
      if (!mounted) return null;
      setState(() {
        _llmReady = true;
        _llmLoadMs = (loaded?['loadMs'] as num?)?.toDouble() ?? 0;
      });
      return loaded;
    } on PlatformException catch (error) {
      if (mounted) setState(() => _llmError = error.message ?? error.code);
      return null;
    } catch (error) {
      if (mounted) setState(() => _llmError = error.toString());
      return null;
    } finally {
      if (mounted) setState(() => _llmBusy = false);
    }
  }

  Future<Map<String, dynamic>> _generateLlm(String prompt) async {
    return await _llmChannel.invokeMapMethod<String, dynamic>('generate', {
          'prompt': prompt,
        }) ??
        const <String, dynamic>{};
  }

  Future<Map<String, dynamic>?> _pickAudioAndTranscribe() async {
    return await _llmChannel.invokeMapMethod<String, dynamic>(
      'pickAudioAndTranscribe',
    );
  }

  Future<Map<String, dynamic>> _speakAnswer(String text) async {
    return await _llmChannel.invokeMapMethod<String, dynamic>('speak', {
          'text': text,
        }) ??
        const <String, dynamic>{};
  }

  Future<void> _stopSpeaking() =>
      _llmChannel.invokeMethod<void>('stopSpeaking');

  Future<Map<String, dynamic>> _getVoiceCapabilities() async {
    return await _llmChannel.invokeMapMethod<String, dynamic>(
          'voiceCapabilities',
        ) ??
        const <String, dynamic>{};
  }

  void _openAssistant() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (_) => DetectionAssistantSheet(
        getDetections: () =>
            (_lastFrame?['detections'] as List<dynamic>?) ?? const [],
        loaded: _llmReady,
        busy: _llmBusy,
        modelPath: _llmModelPath,
        loadMs: _llmLoadMs,
        error: _llmError,
        onSelectAndLoad: _selectAndLoadLlm,
        onLoadSaved: _loadSavedLlm,
        onGenerate: _generateLlm,
        onPickAudioAndTranscribe: _pickAudioAndTranscribe,
        onSpeak: _speakAnswer,
        onStopSpeaking: _stopSpeaking,
        getVoiceCapabilities: _getVoiceCapabilities,
      ),
    );
  }

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
    // Keep the freshest detection snapshot even when UI/FPS updates are throttled.
    _lastFrame = data;
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
      _averageFps = seconds > 0 ? _completedFrames / seconds : 0;
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
        title: const Text('EdgeAI • LiteRT 640'),
        actions: [
          IconButton(
            tooltip: 'Hỏi Gemma dựa trên kết quả YOLO hiện tại',
            onPressed: _openAssistant,
            icon: const Icon(Icons.chat_outlined),
          ),
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
                    'Model: YOLO26n • 99 class • 640×640 • W8A32',
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
                    'YOLO tiếp tục chạy độc lập; LLM chỉ chạy khi người dùng gửi câu hỏi.',
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
