import 'dart:typed_data' as typed;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../../ai_engine/domain/detection.dart';
import '../../ai_engine/services/ai_engine_service.dart';
import 'detection_painter.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CapturedDetectionResult {
  final typed.Uint8List imageBytes;
  final int imageWidth;
  final int imageHeight;
  final List<Detection> detections;
  final double inferenceTimeMs;

  const _CapturedDetectionResult({
    required this.imageBytes,
    required this.imageWidth,
    required this.imageHeight,
    required this.detections,
    required this.inferenceTimeMs,
  });
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  final AIEngineService _aiEngineService = AIEngineService();

  bool _isInitialized = false;
  int _selectedCameraIndex = 0;
  String? _errorMessage;
  int _engineVersion = -1;
  bool _hasVulkan = false;
  bool _isModelLoading = false;
  bool _isModelLoaded = false;
  bool _isDetecting = false;
  _CapturedDetectionResult? _result;

  @override
  void initState() {
    super.initState();
    _engineVersion = _aiEngineService.getVersion();
    _hasVulkan = _aiEngineService.hasVulkanGPU();

    if (widget.cameras.isNotEmpty) {
      _initCamera(_selectedCameraIndex);
    } else {
      _errorMessage = 'Không tìm thấy camera nào trên thiết bị.';
    }
  }

  Future<void> _initCamera(int cameraIndex) async {
    if (widget.cameras.isEmpty) return;

    setState(() {
      _isInitialized = false;
      _errorMessage = null;
      _result = null;
    });

    final previousController = _controller;
    _controller = null;
    await previousController?.dispose();

    final camera = widget.cameras[cameraIndex];
    final controller = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _controller = controller;
        _isInitialized = true;
      });
    } catch (error) {
      await controller.dispose();
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Lỗi khởi tạo camera: $error';
      });
    }
  }

  Future<void> _switchCamera() async {
    if (widget.cameras.length < 2 || _isDetecting) return;
    _selectedCameraIndex = (_selectedCameraIndex + 1) % widget.cameras.length;
    await _initCamera(_selectedCameraIndex);
  }

  Future<void> _loadModel() async {
    if (_isModelLoading || !_aiEngineService.isNativeLoaded) return;

    setState(() {
      _isModelLoading = true;
    });

    final loaded = await _aiEngineService.initializeModel(preferGpu: false);
    if (!mounted) return;

    setState(() {
      _isModelLoading = false;
      _isModelLoaded = loaded;
    });

    final message = loaded
        ? 'NanoDet đã load bằng ${_aiEngineService.modelBackendName}'
        : 'Không load được NanoDet. Mã lỗi: ${_aiEngineService.lastModelLoadCode}';
    _showMessage(message);
  }

  Future<void> _captureAndDetect() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _isDetecting) {
      return;
    }

    if (!_aiEngineService.isModelLoaded) {
      _showMessage('Hãy nhấn “Load NanoDet” trước.');
      return;
    }

    setState(() {
      _isDetecting = true;
    });

    try {
      final photo = await controller.takePicture();
      final jpegBytes = await photo.readAsBytes();
      final decoded = img.decodeImage(jpegBytes);
      if (decoded == null) {
        throw StateError('Không giải mã được ảnh camera.');
      }

      final oriented = img.bakeOrientation(decoded);
      final rgbBytes = oriented.getBytes(order: img.ChannelOrder.rgb);
      final batch = _aiEngineService.detectRgb(
        rgbBytes: rgbBytes,
        width: oriented.width,
        height: oriented.height,
      );

      final displayBytes = typed.Uint8List.fromList(
        img.encodeJpg(oriented, quality: 90),
      );

      if (!mounted) return;
      setState(() {
        _result = _CapturedDetectionResult(
          imageBytes: displayBytes,
          imageWidth: oriented.width,
          imageHeight: oriented.height,
          detections: batch.detections,
          inferenceTimeMs: batch.inferenceTimeMs,
        );
      });

      _showMessage(
        'Tìm thấy ${batch.detections.length} vật thể trong '
        '${batch.inferenceTimeMs.toStringAsFixed(1)} ms',
      );
    } catch (error, stackTrace) {
      debugPrint('Capture/detection failed: $error');
      debugPrintStack(stackTrace: stackTrace);
      if (mounted) {
        _showMessage('Nhận diện thất bại: $error');
      }
    } finally {
      if (mounted) {
        setState(() {
          _isDetecting = false;
        });
      }
    }
  }

  void _showMessage(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(content: Text(message), duration: const Duration(seconds: 3)),
      );
  }

  @override
  void dispose() {
    _aiEngineService.unloadModel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.memory, color: Color(0xFF7F5AF0)),
            SizedBox(width: 8),
            Text('EdgeAI NanoDet'),
          ],
        ),
        actions: [
          if (widget.cameras.length > 1)
            IconButton(
              icon: const Icon(Icons.switch_camera),
              onPressed: _switchCamera,
              tooltip: 'Đổi camera',
            ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            _errorMessage!,
            textAlign: TextAlign.center,
            style: TextStyle(color: Theme.of(context).colorScheme.error),
          ),
        ),
      );
    }

    if (!_isInitialized || _controller == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Đang kết nối camera và NCNN...'),
          ],
        ),
      );
    }

    return Column(
      children: [
        Expanded(
          child: Stack(
            children: [
              Container(
                margin: const EdgeInsets.all(12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: _buildViewport(),
                ),
              ),
              Positioned(top: 24, left: 24, child: _buildStatusBanner()),
            ],
          ),
        ),
        _buildControlPanel(),
      ],
    );
  }

  Widget _buildViewport() {
    final result = _result;
    if (result == null) {
      return Center(child: CameraPreview(_controller!));
    }

    return Center(
      child: AspectRatio(
        aspectRatio: result.imageWidth / result.imageHeight,
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.memory(result.imageBytes, fit: BoxFit.fill),
            CustomPaint(
              painter: DetectionPainter(
                detections: result.detections,
                imageWidth: result.imageWidth,
                imageHeight: result.imageHeight,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusBanner() {
    final result = _result;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withAlpha(190),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: _aiEngineService.isNativeLoaded
              ? const Color(0xFF2CB67D)
              : const Color(0xFFFF5470),
          width: 1.5,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _aiEngineService.isNativeLoaded
                ? 'NCNN v$_engineVersion • ${_isModelLoaded ? _aiEngineService.modelBackendName : 'model chưa load'}'
                : 'Native library chưa load',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            result == null
                ? (_hasVulkan ? 'Thiết bị có Vulkan' : 'Chế độ CPU')
                : '${result.detections.length} vật thể • ${result.inferenceTimeMs.toStringAsFixed(1)} ms',
            style: const TextStyle(color: Colors.white70, fontSize: 11),
          ),
        ],
      ),
    );
  }

  Widget _buildControlPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Wrap(
        alignment: WrapAlignment.spaceBetween,
        runAlignment: WrapAlignment.center,
        spacing: 12,
        runSpacing: 10,
        children: [
          Chip(
            avatar: const Icon(Icons.videocam, size: 16),
            label: Text(widget.cameras[_selectedCameraIndex].name),
          ),
          ElevatedButton.icon(
            onPressed: _isModelLoading || _isDetecting ? null : _loadModel,
            icon: _isModelLoading
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(_isModelLoaded ? Icons.check_circle : Icons.memory),
            label: Text(
              _isModelLoading
                  ? 'Đang load...'
                  : _isModelLoaded
                  ? 'NanoDet Loaded'
                  : 'Load NanoDet',
            ),
          ),
          ElevatedButton.icon(
            onPressed: _isDetecting
                ? null
                : _result == null
                ? _captureAndDetect
                : () {
                    setState(() {
                      _result = null;
                    });
                  },
            icon: _isDetecting
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(
                    _result == null
                        ? Icons.center_focus_strong
                        : Icons.camera_alt,
                  ),
            label: Text(
              _isDetecting
                  ? 'Đang nhận diện...'
                  : _result == null
                  ? 'Nhận diện'
                  : 'Quay lại camera',
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF7F5AF0),
              foregroundColor: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}
