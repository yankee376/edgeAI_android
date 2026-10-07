import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import '../data/yolo26_labels.dart';

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
  final Stopwatch _piFpsClock = Stopwatch();
  final Stopwatch _piStreamClock = Stopwatch();
  final HttpClient _httpClient = HttpClient()
    ..connectionTimeout = const Duration(seconds: 3);
  Map<String, dynamic>? _lastFrame;
  Uint8List? _piImageBytes;
  List<dynamic> _piBoxes = const [];
  Uint8List? _piPendingBytes;
  int _piPendingSeq = 0;
  int _piLastFrameSeq = 0;
  int _piDetectionFrameSeq = 0;
  int _piEndpointGeneration = 0;
  int _piModelGeneration = 0;
  YOLO? _piYolo;
  String _piBaseUrl = 'http://192.168.1.20:5001';
  String? _piError;
  String? _piModelError;
  bool _piMode = false;
  bool _piLoading = false;
  bool _piFetching = false;
  bool _piPredicting = false;
  int _piReceivedFrames = 0;
  int _piCompletedFrames = 0;
  double _piStreamFps = 0;
  double _piFps = 0;
  double _piDownloadMs = 0;
  double _piPredictMs = 0;
  double _piPreMs = 0;
  double _piInferenceMs = 0;
  double _piPostMs = 0;
  int _piImageWidth = 0;
  int _piImageHeight = 0;
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
    _piFpsClock
      ..stop()
      ..reset();
    _piStreamClock
      ..stop()
      ..reset();
    setState(() {
      _completedFrames = 0;
      _averageFps = 0;
      _lastFrame = null;
      _lastUiUpdate = null;
      _piReceivedFrames = 0;
      _piCompletedFrames = 0;
      _piStreamFps = 0;
      _piFps = 0;
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

  Future<void> _selectFrameSource() async {
    final selected = await showModalBottomSheet<bool>(
      context: context,
      builder: (sheetContext) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.phone_android),
              title: const Text('Camera điện thoại'),
              onTap: () => Navigator.pop(sheetContext, false),
            ),
            ListTile(
              leading: const Icon(Icons.sensors),
              title: const Text('Ảnh từ Raspberry Pi'),
              subtitle: const Text('Nhận frame liên tục từ Pi'),
              onTap: () => Navigator.pop(sheetContext, true),
            ),
          ],
        ),
      ),
    );
    if (!mounted || selected == null || selected == _piMode) return;
    await _switchFrameSource(selected);
  }

  Future<void> _switchFrameSource(bool usePi) async {
    _piModelGeneration++;
    if (_piMode) {
      setState(() => _piMode = false);
      _piPendingBytes = null;
      while ((_piFetching || _piPredicting) && mounted) {
        await Future<void>.delayed(const Duration(milliseconds: 50));
      }
    }
    final previousYolo = _piYolo;
    _piYolo = null;
    if (previousYolo != null) await previousYolo.dispose();
    if (!mounted) return;
    _fpsClock
      ..stop()
      ..reset();
    _piFpsClock
      ..stop()
      ..reset();
    _piStreamClock
      ..stop()
      ..reset();
    setState(() {
      _piMode = usePi;
      _piLoading = false;
      _loaded = false;
      _paused = false;
      _piImageBytes = null;
      _piBoxes = const [];
      _piPendingBytes = null;
      _piLastFrameSeq = 0;
      _piDetectionFrameSeq = 0;
      _piError = null;
      _piModelError = null;
      _piReceivedFrames = 0;
      _piCompletedFrames = 0;
      _piStreamFps = 0;
      _piFps = 0;
      _lastFrame = null;
    });
    if (usePi) {
      unawaited(_startPiFrameLoop());
      await _connectToPi();
    }
  }

  Future<void> _editPiAddress() async {
    var editedAddress = _piBaseUrl;
    final value = await showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Địa chỉ camera Pi'),
        content: TextFormField(
          initialValue: _piBaseUrl,
          keyboardType: TextInputType.url,
          onChanged: (text) => editedAddress = text,
          decoration: const InputDecoration(
            labelText: 'URL server',
            hintText: 'http://10.0.0.12:5001',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Hủy'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(dialogContext, editedAddress.trim()),
            child: const Text('Kết nối'),
          ),
        ],
      ),
    );
    if (!mounted || value == null || value.isEmpty) return;
    _piBaseUrl = value.replaceFirst(RegExp(r'/+$'), '');
    _piLastFrameSeq = 0;
    _piEndpointGeneration++;
    _piPendingBytes = null;
    if (_piMode) {
      setState(() {
        _piImageBytes = null;
        _piBoxes = const [];
        _piError = null;
      });
      unawaited(_startPiFrameLoop());
      if (_piYolo == null) await _connectToPi();
    }
  }

  Future<void> _connectToPi() async {
    if (_piYolo != null) {
      unawaited(_startPiFrameLoop());
      final bytes = _piImageBytes;
      if (bytes != null) _queuePiInference(bytes, _piLastFrameSeq);
      return;
    }
    if (_piLoading) return;
    final modelGeneration = _piModelGeneration;
    setState(() {
      _piLoading = true;
      _piModelError = null;
    });
    try {
      final yolo = YOLO(
        modelPath: _modelPath,
        task: YOLOTask.detect,
        useGpu: true,
        useMultiInstance: true,
      );
      final loaded = await yolo.loadModel();
      if (!loaded) throw StateError('Không nạp được model LiteRT 416.');
      if (!mounted || !_piMode || modelGeneration != _piModelGeneration) {
        await yolo.dispose();
        return;
      }
      _piYolo = yolo;
      setState(() => _piLoading = false);
      final bytes = _piImageBytes;
      if (bytes != null) _queuePiInference(bytes, _piLastFrameSeq);
    } catch (error) {
      if (mounted && modelGeneration == _piModelGeneration) {
        setState(() {
          _piLoading = false;
          _piModelError = error.toString();
        });
      }
    }
  }

  Future<Map<String, Object>?> _requestNextPiFrame() async {
    final endpointGeneration = _piEndpointGeneration;
    final requestClock = Stopwatch()..start();
    final uri = Uri.parse(
      '$_piBaseUrl/api/frame',
    ).replace(queryParameters: {'after': _piLastFrameSeq.toString()});
    final request = await _httpClient
        .getUrl(uri)
        .timeout(const Duration(seconds: 4));
    final response = await request.close().timeout(const Duration(seconds: 4));
    if (endpointGeneration != _piEndpointGeneration) {
      await response.drain<void>();
      return null;
    }
    if (response.statusCode == HttpStatus.noContent) return null;
    if (response.statusCode != HttpStatus.ok) {
      await response.drain<void>();
      throw HttpException('HTTP ${response.statusCode} từ Pi');
    }
    final frameSeq = int.tryParse(response.headers.value('x-frame-seq') ?? '');
    if (frameSeq == null) {
      await response.drain<void>();
      throw const FormatException('Pi không gửi header X-Frame-Seq.');
    }
    final bytes = await response
        .fold<BytesBuilder>(BytesBuilder(copy: false), (builder, chunk) {
          builder.add(chunk);
          return builder;
        })
        .then((builder) => builder.takeBytes())
        .timeout(const Duration(seconds: 4));
    requestClock.stop();
    if (bytes.isEmpty) throw const FormatException('Pi trả ảnh rỗng.');
    return {
      'bytes': bytes,
      'frameSeq': frameSeq,
      'waitMs': requestClock.elapsedMicroseconds / 1000,
    };
  }

  Future<void> _startPiFrameLoop() async {
    if (_piFetching || !_piMode) return;
    _piFetching = true;
    try {
      while (mounted && _piMode) {
        try {
          final frame = await _requestNextPiFrame();
          if (frame == null) continue;
          if (!mounted || !_piMode) break;
          final bytes = frame['bytes']! as Uint8List;
          final frameSeq = frame['frameSeq']! as int;
          _piLastFrameSeq = frameSeq;
          _piStreamClock.start();
          _piReceivedFrames++;
          final elapsed = _piStreamClock.elapsedMicroseconds / 1000000;
          setState(() {
            _piImageBytes = bytes;
            _piDownloadMs = frame['waitMs']! as double;
            _piStreamFps = elapsed > 0 ? (_piReceivedFrames - 1) / elapsed : 0;
            _piError = null;
          });
          _queuePiInference(bytes, frameSeq);
        } catch (error) {
          if (!mounted || !_piMode) break;
          setState(() => _piError = error.toString());
          await Future<void>.delayed(const Duration(seconds: 1));
        }
      }
    } finally {
      _piFetching = false;
    }
  }

  void _queuePiInference(Uint8List bytes, int frameSeq) {
    if (!_piMode || _piYolo == null) return;
    _piPendingBytes = bytes;
    _piPendingSeq = frameSeq;
    if (!_piPredicting) unawaited(_runPiInference());
  }

  Future<void> _runPiInference() async {
    if (_piPredicting) return;
    _piPredicting = true;
    try {
      while (mounted && _piMode && _piYolo != null && _piPendingBytes != null) {
        final bytes = _piPendingBytes!;
        final frameSeq = _piPendingSeq;
        final endpointGeneration = _piEndpointGeneration;
        _piPendingBytes = null;
        final yolo = _piYolo!;
        try {
          final predictClock = Stopwatch()..start();
          final result = await yolo.predict(bytes, rotationDegrees: 180);
          predictClock.stop();
          if (!mounted || !_piMode) break;
          if (endpointGeneration != _piEndpointGeneration) continue;
          final boxes = result['boxes'] as List<dynamic>? ?? const [];
          final imageSize = result['imageSize'] as Map<dynamic, dynamic>?;
          final imageWidth = (imageSize?['width'] as num?)?.toInt() ?? 0;
          final imageHeight = (imageSize?['height'] as num?)?.toInt() ?? 0;
          final detections = boxes
              .whereType<Map>()
              .map((box) {
                final classIndex = (box['classIndex'] as num?)?.toInt() ?? -1;
                final className =
                    box['className'] as String? ??
                    (classIndex >= 0 && classIndex < yolo26Labels.length
                        ? yolo26Labels[classIndex]
                        : 'class $classIndex');
                return <String, dynamic>{
                  'classIndex': classIndex,
                  'className': className,
                  'confidence': (box['confidence'] as num?)?.toDouble() ?? 0,
                  'normalizedBox': <String, double>{
                    'left': (box['x1_norm'] as num?)?.toDouble() ?? 0,
                    'top': (box['y1_norm'] as num?)?.toDouble() ?? 0,
                    'right': (box['x2_norm'] as num?)?.toDouble() ?? 0,
                    'bottom': (box['y2_norm'] as num?)?.toDouble() ?? 0,
                  },
                };
              })
              .toList(growable: false);
          final preMs = (result['preMs'] as num?)?.toDouble() ?? 0;
          final inferenceMs = (result['inferenceMs'] as num?)?.toDouble() ?? 0;
          final postMs = (result['postMs'] as num?)?.toDouble() ?? 0;
          _piFpsClock.start();
          _piCompletedFrames++;
          final elapsed = _piFpsClock.elapsedMicroseconds / 1000000;
          setState(() {
            _piBoxes = boxes;
            _piDetectionFrameSeq = frameSeq;
            _piImageWidth = imageWidth;
            _piImageHeight = imageHeight;
            _piPredictMs = predictClock.elapsedMicroseconds / 1000;
            _piPreMs = preMs;
            _piInferenceMs = inferenceMs;
            _piPostMs = postMs;
            _piFps = elapsed > 0 ? (_piCompletedFrames - 1) / elapsed : 0;
            _piModelError = null;
            _lastFrame = {
              'detections': detections,
              'imageWidth': imageWidth,
              'imageHeight': imageHeight,
              'accelerator': result['accelerator'],
              'preMs': preMs,
              'inferenceMs': inferenceMs,
              'postMs': postMs,
              'processingTimeMs': preMs + inferenceMs + postMs,
            };
          });
        } catch (error) {
          if (!mounted || !_piMode) break;
          setState(() => _piModelError = error.toString());
          await Future<void>.delayed(const Duration(milliseconds: 200));
        }
      }
    } finally {
      _piPredicting = false;
      if (mounted && _piMode && _piYolo != null && _piPendingBytes != null) {
        unawaited(_runPiInference());
      }
    }
  }

  Future<void> _disposePiYoloWhenIdle(YOLO yolo) async {
    while (_piPredicting) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
    await yolo.dispose();
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
    _piMode = false;
    _piPendingBytes = null;
    _fpsClock.stop();
    _piFpsClock.stop();
    _piStreamClock.stop();
    _httpClient.close(force: true);
    final piYolo = _piYolo;
    _piYolo = null;
    if (piYolo != null) unawaited(_disposePiYoloWhenIdle(piYolo));
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
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        title: const Text('EdgeAI • LiteRT 416'),
        actions: [
          IconButton(
            tooltip: _piMode ? 'Nguồn: Raspberry Pi' : 'Nguồn: điện thoại',
            onPressed: _selectFrameSource,
            icon: Icon(_piMode ? Icons.sensors : Icons.phone_android),
          ),
          IconButton(
            tooltip: 'Đặt lại bộ đếm FPS',
            onPressed: _resetCounter,
            icon: const Icon(Icons.restart_alt),
          ),
          if (!_piMode)
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
            child: _piMode
                ? _buildPiPreview()
                : YOLOView(
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
            child: ConstrainedBox(
              constraints: BoxConstraints(
                maxHeight: MediaQuery.sizeOf(context).height * 0.48,
              ),
              child: SingleChildScrollView(
                child: Container(
                  width: double.infinity,
                  color: Colors.black87,
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        'AI: ${(_piMode ? _piFps : _averageFps).toStringAsFixed(1)} FPS',
                        style: const TextStyle(
                          color: Colors.lightGreenAccent,
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      if (_piMode)
                        Text(
                          'Stream Pi: ${_piStreamFps.toStringAsFixed(1)} FPS',
                          style: const TextStyle(color: Colors.lightBlueAccent),
                        ),
                      Text(
                        'Nguồn ảnh: ${_piMode ? 'Raspberry Pi • stream liên tục' : 'Camera điện thoại • CameraX'}',
                        style: const TextStyle(color: Colors.white70),
                      ),
                      if (_piMode)
                        InkWell(
                          onTap: _editPiAddress,
                          child: Text(
                            'Pi: $_piBaseUrl • nhấn để đổi địa chỉ',
                            style: const TextStyle(
                              color: Colors.lightBlueAccent,
                            ),
                          ),
                        ),
                      if (_piMode)
                        Text(
                          'Frame Pi: $_piLastFrameSeq • YOLO: $_piDetectionFrameSeq',
                          style: const TextStyle(
                            color: Colors.white54,
                            fontSize: 11,
                          ),
                        ),
                      Text(
                        'Backend đang chạy: ${_piMode ? (_piLoading
                                  ? 'đang nạp LiteRT'
                                  : _piYolo == null
                                  ? 'chưa nạp'
                                  : 'LiteRT ${backend ?? 'đang xác định'}') : (backend == null ? (_loaded ? 'đang chờ frame' : 'chưa nạp') : 'LiteRT $backend')}',
                        style: const TextStyle(color: Colors.white70),
                      ),
                      const Text(
                        'Model: YOLO26n • 99 class • 416×416 • W8A32',
                        style: TextStyle(color: Colors.white70),
                      ),
                      Text(
                        _piMode
                            ? 'Ảnh Pi • ${_piBoxes.length} vật thể'
                            : 'CameraX RGBA • frame sau xoay: $width×$height • ${detections?.length ?? 0} vật thể',
                        style: const TextStyle(color: Colors.white70),
                      ),
                      Text(
                        'FPS nội bộ plugin: ${nativeFps.toStringAsFixed(1)}',
                        style: const TextStyle(color: Colors.white70),
                      ),
                      const SizedBox(height: 4),
                      if (_piMode) ...[
                        _stage('HTTP chờ frame + nhận JPEG', _piDownloadMs),
                        _stage('YOLO.predict tổng', _piPredictMs),
                        _stage('Tiền xử lý', _piPreMs),
                        _stage('LiteRT inference', _piInferenceMs),
                        _stage('Hậu xử lý', _piPostMs),
                        const Text(
                          'Stream hiển thị nhận frame liên tục; YOLO chỉ xử lý frame mới nhất khi sẵn sàng.',
                          style: TextStyle(color: Colors.white54, fontSize: 11),
                        ),
                      ] else ...[
                        _stage('CameraX → Bitmap', cameraMs),
                        _stage('Resize/letterbox/normalize', preMs),
                        _stage('LiteRT inference', inferMs),
                        _stage('Decode/NMS', postMs),
                        if (unspecificMs >= 1)
                          _stage('Phần còn lại trong detector', unspecificMs),
                        _stage('Tổng detector', modelMs),
                        _stage('Tổng phần đã đo', measuredMs),
                      ],
                      const Text(
                        'Model YOLO26n • 99 class • 416×416 • W8A32',
                        style: TextStyle(color: Colors.white54, fontSize: 11),
                      ),
                      if (_piError != null)
                        Text(
                          _piError!,
                          style: const TextStyle(color: Colors.redAccent),
                        ),
                      if (_piModelError != null)
                        Text(
                          'Lỗi nhận diện: $_piModelError',
                          style: const TextStyle(color: Colors.redAccent),
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
                          onPressed: _piMode
                              ? (_piLoading ? null : _connectToPi)
                              : (_loaded ? _togglePause : null),
                          child: Text(
                            _piMode
                                ? (_piLoading
                                      ? 'Đang nạp LiteRT…'
                                      : 'Kết nối lại Pi / lấy ảnh ngay')
                                : (_paused
                                      ? 'YOLO đã tạm dừng • Nhấn để bật'
                                      : 'YOLO đang hoạt động • Nhấn để tạm dừng'),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPiPreview() {
    final bytes = _piImageBytes;
    if (bytes == null) {
      return const ColoredBox(
        color: Colors.black,
        child: Center(
          child: Text(
            'Đang chờ ảnh từ Pi…\nNhập địa chỉ Pi qua dòng URL bên dưới.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white70),
          ),
        ),
      );
    }
    return ColoredBox(
      color: Colors.black,
      child: Stack(
        fit: StackFit.expand,
        children: [
          RotatedBox(
            quarterTurns: 2,
            child: Image.memory(
              bytes,
              fit: BoxFit.contain,
              gaplessPlayback: true,
            ),
          ),
          CustomPaint(
            painter: _PiBoxesPainter(_piBoxes, _piImageWidth, _piImageHeight),
          ),
        ],
      ),
    );
  }
}

class _PiBoxesPainter extends CustomPainter {
  const _PiBoxesPainter(this.boxes, this.sourceWidth, this.sourceHeight);

  final List<dynamic> boxes;
  final int sourceWidth;
  final int sourceHeight;

  @override
  void paint(Canvas canvas, Size size) {
    if (sourceWidth <= 0 || sourceHeight <= 0) return;
    final aspect = sourceWidth / sourceHeight;
    final fittedWidth = math.min(size.width, size.height * aspect).toDouble();
    final fittedHeight = math.min(size.height, size.width / aspect).toDouble();
    final dx = (size.width - fittedWidth) / 2;
    final dy = (size.height - fittedHeight) / 2;
    for (final item in boxes) {
      if (item is! Map) continue;
      final x1 = (item['x1_norm'] as num?)?.toDouble();
      final y1 = (item['y1_norm'] as num?)?.toDouble();
      final x2 = (item['x2_norm'] as num?)?.toDouble();
      final y2 = (item['y2_norm'] as num?)?.toDouble();
      if (x1 == null || y1 == null || x2 == null || y2 == null) continue;
      final rect = Rect.fromLTRB(
        dx + x1 * fittedWidth,
        dy + y1 * fittedHeight,
        dx + x2 * fittedWidth,
        dy + y2 * fittedHeight,
      );
      final classIndex = (item['classIndex'] as num?)?.toInt() ?? -1;
      final color =
          Colors.primaries[classIndex.abs() % Colors.primaries.length];
      canvas.drawRect(
        rect,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3,
      );
      final className =
          item['className'] as String? ??
          (classIndex >= 0 && classIndex < yolo26Labels.length
              ? yolo26Labels[classIndex]
              : 'class $classIndex');
      final confidence = (item['confidence'] as num?)?.toDouble() ?? 0;
      final text = TextPainter(
        text: TextSpan(
          text: '$className ${(confidence * 100).toStringAsFixed(1)}%',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: size.width);
      final top = math.max(0.0, rect.top - text.height - 4).toDouble();
      canvas.drawRect(
        Rect.fromLTWH(rect.left, top, text.width + 6, text.height + 4),
        Paint()..color = color.withAlpha(220),
      );
      text.paint(canvas, Offset(rect.left + 3, top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _PiBoxesPainter oldDelegate) =>
      oldDelegate.boxes != boxes ||
      oldDelegate.sourceWidth != sourceWidth ||
      oldDelegate.sourceHeight != sourceHeight;
}
