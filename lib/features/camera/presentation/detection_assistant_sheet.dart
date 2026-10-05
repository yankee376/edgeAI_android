import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class DetectionAssistantSheet extends StatefulWidget {
  final List<dynamic> Function() getDetections;
  final bool loaded;
  final bool busy;
  final String? modelPath;
  final double loadMs;
  final String? error;
  final Future<Map<String, dynamic>?> Function() onSelectAndLoad;
  final Future<Map<String, dynamic>?> Function() onLoadSaved;
  final Future<Map<String, dynamic>> Function(String prompt) onGenerate;
  final Future<Map<String, dynamic>?> Function() onPickAudioAndTranscribe;
  final Future<Map<String, dynamic>> Function(String text) onSpeak;
  final Future<void> Function() onStopSpeaking;
  final Future<Map<String, dynamic>> Function() getVoiceCapabilities;

  const DetectionAssistantSheet({
    super.key,
    required this.getDetections,
    required this.loaded,
    required this.busy,
    required this.modelPath,
    required this.loadMs,
    required this.error,
    required this.onSelectAndLoad,
    required this.onLoadSaved,
    required this.onGenerate,
    required this.onPickAudioAndTranscribe,
    required this.onSpeak,
    required this.onStopSpeaking,
    required this.getVoiceCapabilities,
  });

  @override
  State<DetectionAssistantSheet> createState() =>
      _DetectionAssistantSheetState();
}

class _DetectionAssistantSheetState extends State<DetectionAssistantSheet> {
  String? _selectedClass;
  String _recognizedQuestion = '';
  String? _fallbackAnswer;
  String? _requestError;
  bool _recognizing = false;
  bool _generating = false;
  bool _speaking = false;
  bool _loading = false;
  bool _sessionLoaded = false;
  String? _localError;
  double _sessionLoadMs = 0;
  String? _sessionModelPath;
  double? _recognitionMs;
  double? _decodeMs;
  double? _audioDurationMs;
  double? _generationMs;
  double? _speechMs;
  String? _audioFileName;
  String? _voiceReadiness;

  @override
  void initState() {
    super.initState();
    _refreshVoiceReadiness();
  }

  Future<void> _refreshVoiceReadiness() async {
    try {
      final capabilities = await widget.getVoiceCapabilities();
      final ttsReady = capabilities['ttsReady'] == true;
      final tts = capabilities['offlineVietnameseTtsAvailable'] == true;
      if (mounted) {
        setState(() {
          _voiceReadiness =
              'ASR offline: Vosk tiếng Việt • TTS offline tiếng Việt: ${tts
                  ? 'sẵn sàng'
                  : ttsReady
                  ? 'chưa có giọng'
                  : 'đang khởi động'}';
        });
      }
    } catch (error) {
      if (mounted) {
        setState(() => _voiceReadiness = 'Không kiểm tra được voice: $error');
      }
    }
  }

  List<String> get _classes {
    final names = <String>{};
    for (final item in widget.getDetections()) {
      if (item is Map && item['className'] is String) {
        names.add(item['className'] as String);
      }
    }
    return names.toList()..sort();
  }

  String _prompt(String question, List<dynamic> detections) {
    final observations = <String>[];
    for (final item in detections) {
      if (item is! Map) continue;
      final name = item['className']?.toString() ?? 'unknown';
      if (_selectedClass != null && name != _selectedClass) continue;
      final confidence = item['confidence'];
      final box = item['normalizedBox'];
      var position = 'không rõ vị trí';
      if (box is Map) {
        final left = (box['left'] as num?)?.toDouble();
        final top = (box['top'] as num?)?.toDouble();
        final right = (box['right'] as num?)?.toDouble();
        final bottom = (box['bottom'] as num?)?.toDouble();
        if (left != null && top != null && right != null && bottom != null) {
          final centerX = (left + right) / 2;
          final centerY = (top + bottom) / 2;
          final horizontal = centerX < 0.34
              ? 'bên trái'
              : centerX > 0.66
              ? 'bên phải'
              : 'ở giữa';
          final vertical = centerY < 0.34
              ? 'phía trên'
              : centerY > 0.66
              ? 'phía dưới'
              : 'vùng giữa theo chiều dọc';
          position = '$horizontal, $vertical';
        }
      }
      observations.add(
        '- $name; độ tin cậy ${confidence is num ? confidence.toStringAsFixed(2) : 'không rõ'}; $position',
      );
    }
    final snapshot = observations.isEmpty
        ? 'Không có detection phù hợp trong frame mới nhất.'
        : observations.take(30).join('\n');
    return '''Trả lời bằng tiếng Việt, ngắn gọn (tối đa 40 từ để đọc thành tiếng). Chỉ dùng dữ liệu YOLO dưới đây; không suy đoán thuộc tính hoặc vật thể không có trong danh sách. Nếu dữ liệu không đủ để trả lời, hãy nói rõ.

Detection từ frame mới nhất:
$snapshot

Câu hỏi người dùng: $question''';
  }

  Future<void> _askFromAudioFile() async {
    if (_recognizing ||
        _generating ||
        _speaking ||
        !(widget.loaded || _sessionLoaded)) {
      return;
    }
    setState(() {
      _recognizing = true;
      _recognizedQuestion = '';
      _fallbackAnswer = null;
      _requestError = null;
      _recognitionMs = null;
      _decodeMs = null;
      _audioDurationMs = null;
      _generationMs = null;
      _speechMs = null;
      _audioFileName = null;
    });
    try {
      final voice = await widget.onPickAudioAndTranscribe();
      if (voice == null) {
        if (mounted) setState(() => _requestError = 'Chưa chọn file âm thanh.');
        return;
      }
      final question = voice['transcript']?.toString().trim() ?? '';
      if (question.isEmpty) {
        throw StateError('Không nhận dạng được lời nói trong file.');
      }
      if (!mounted) return;
      setState(() {
        _recognizing = false;
        _recognizedQuestion = question;
        _audioFileName = voice['fileName']?.toString();
        _decodeMs = (voice['decodeMs'] as num?)?.toDouble();
        _recognitionMs = (voice['recognitionMs'] as num?)?.toDouble();
        _audioDurationMs = (voice['audioDurationMs'] as num?)?.toDouble();
        _generating = true;
      });

      final snapshot = widget.getDetections();
      final result = await widget.onGenerate(_prompt(question, snapshot));
      if (!mounted) return;
      final answer = result['answer']?.toString().trim() ?? '';
      if (answer.isEmpty) throw StateError('Gemma chưa tạo được câu trả lời.');
      setState(() {
        _generating = false;
        _generationMs = (result['generationMs'] as num?)?.toDouble();
        _speaking = true;
      });

      try {
        final speech = await widget.onSpeak(answer);
        if (mounted) {
          setState(() {
            _speechMs = (speech['playbackMs'] as num?)?.toDouble();
            if (speech['stopped'] == true) _requestError = 'Đã dừng đọc.';
          });
        }
      } on PlatformException catch (error) {
        if (mounted) {
          setState(() {
            _fallbackAnswer = answer;
            _requestError = error.message ?? 'Không đọc được câu trả lời.';
          });
        }
      }
    } on PlatformException catch (error) {
      if (mounted) setState(() => _requestError = error.message ?? error.code);
    } catch (error) {
      if (mounted) setState(() => _requestError = error.toString());
    } finally {
      if (mounted) {
        setState(() {
          _recognizing = false;
          _generating = false;
          _speaking = false;
        });
      }
    }
  }

  Future<void> _stopSpeaking() async {
    await widget.onStopSpeaking();
  }

  Future<void> _load(Future<Map<String, dynamic>?> Function() load) async {
    if (_loading) return;
    setState(() {
      _loading = true;
      _localError = null;
    });
    try {
      final loaded = await load();
      if (mounted) {
        setState(() {
          _sessionLoaded = loaded != null;
          _sessionLoadMs = (loaded?['loadMs'] as num?)?.toDouble() ?? 0;
          _sessionModelPath = loaded?['path']?.toString();
        });
      }
      if (loaded == null && mounted) {
        setState(
          () =>
              _localError = 'Chưa nạp được model. Kiểm tra lại file .litertlm.',
        );
      }
    } catch (error) {
      if (mounted) setState(() => _localError = error.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final classNames = _classes;
    final modelLoaded = widget.loaded || _sessionLoaded;
    return Padding(
      padding: EdgeInsets.only(
        left: 16,
        right: 16,
        top: 16,
        bottom: MediaQuery.viewInsetsOf(context).bottom + 16,
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Hỏi từ file giọng nói',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 4),
            Text(
              'Chọn file ghi âm/video. Vosk chuyển giọng nói thành chữ offline → Gemma kết hợp snapshot YOLO → đọc câu trả lời; camera vẫn chạy.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            if (!modelLoaded) ...[
              Text(
                widget.modelPath == null
                    ? 'Chọn file Gemma 3 1B IT INT4 (.litertlm) trên điện thoại.'
                    : 'Đã chọn model: ${widget.modelPath!.split('/').last}',
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: widget.busy || _loading
                          ? null
                          : () => _load(widget.onSelectAndLoad),
                      icon: const Icon(Icons.folder_open),
                      label: Text(
                        widget.modelPath == null
                            ? 'Chọn và nạp model'
                            : 'Chọn model khác',
                      ),
                    ),
                  ),
                  if (widget.modelPath != null) ...[
                    const SizedBox(width: 8),
                    IconButton.filledTonal(
                      tooltip: 'Nạp model đã chọn',
                      onPressed: widget.busy || _loading
                          ? null
                          : () => _load(widget.onLoadSaved),
                      icon: const Icon(Icons.play_arrow),
                    ),
                  ],
                ],
              ),
              if (widget.busy || _loading)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 8),
                  child: LinearProgressIndicator(),
                ),
            ] else ...[
              Text(
                'LiteRT-LM backend: CPU • nạp model ${(_sessionLoaded ? _sessionLoadMs : widget.loadMs).toStringAsFixed(0)} ms • ${(_sessionModelPath ?? widget.modelPath)?.split('/').last ?? 'Gemma 3 1B'}',
              ),
              if (_voiceReadiness != null) ...[
                const SizedBox(height: 4),
                Text(_voiceReadiness!),
              ],
              const SizedBox(height: 6),
              if (classNames.isNotEmpty)
                DropdownButtonFormField<String?>(
                  initialValue: _selectedClass,
                  decoration: const InputDecoration(
                    labelText: 'Lọc theo class (tùy chọn)',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                  items: [
                    const DropdownMenuItem<String?>(
                      value: null,
                      child: Text('Tất cả class'),
                    ),
                    ...classNames.map(
                      (name) => DropdownMenuItem<String?>(
                        value: name,
                        child: Text(name),
                      ),
                    ),
                  ],
                  onChanged: (value) => setState(() => _selectedClass = value),
                ),
              if (classNames.isEmpty)
                const Text('Frame mới nhất chưa có vật thể được nhận diện.'),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _recognizing || _generating || _speaking
                          ? null
                          : _askFromAudioFile,
                      icon: Icon(
                        _recognizing ? Icons.hearing : Icons.audio_file,
                      ),
                      label: Text(
                        _recognizing
                            ? 'Đang chuyển file giọng nói thành chữ…'
                            : _generating
                            ? 'Gemma đang xử lý…'
                            : _speaking
                            ? 'Đang đọc câu trả lời…'
                            : 'Chọn file giọng nói để hỏi',
                      ),
                    ),
                  ),
                  if (_speaking) ...[
                    const SizedBox(width: 8),
                    IconButton.filledTonal(
                      tooltip: 'Dừng đọc',
                      onPressed: _stopSpeaking,
                      icon: const Icon(Icons.stop),
                    ),
                  ],
                ],
              ),
              if (_recognizing || _generating || _speaking) ...[
                const SizedBox(height: 10),
                const LinearProgressIndicator(),
                const SizedBox(height: 5),
                Text(
                  _recognizing
                      ? 'Đang giải mã âm thanh và nhận dạng bằng Vosk offline…'
                      : _generating
                      ? 'Đang dùng CPU để tạo câu trả lời…'
                      : 'Đang phát câu trả lời bằng giọng nói offline…',
                ),
              ],
              if (_recognizedQuestion.isNotEmpty) ...[
                const SizedBox(height: 12),
                if (_audioFileName != null) Text('File: $_audioFileName'),
                Text('Vosk nhận dạng: “$_recognizedQuestion”'),
                if (_audioDurationMs != null)
                  Text(
                    'Độ dài file: ${(_audioDurationMs! / 1000).toStringAsFixed(1)} giây',
                  ),
                if (_decodeMs != null)
                  Text(
                    'Giải mã file âm thanh: ${_decodeMs!.toStringAsFixed(0)} ms',
                  ),
                if (_recognitionMs != null)
                  Text(
                    'Nhận dạng giọng nói: ${_recognitionMs!.toStringAsFixed(0)} ms',
                  ),
                if (_generationMs != null)
                  Text('Gemma CPU: ${_generationMs!.toStringAsFixed(0)} ms'),
                if (_speechMs != null)
                  Text('Đọc câu trả lời: ${_speechMs!.toStringAsFixed(0)} ms'),
              ],
              if (_fallbackAnswer != null) ...[
                const SizedBox(height: 8),
                const Text(
                  'Bản đọc tiếng Việt offline chưa sẵn sàng; câu trả lời:',
                ),
                SelectableText(_fallbackAnswer!),
              ],
              if (widget.getDetections().isNotEmpty)
                Text(
                  'Snapshot YOLO mới nhất có ${widget.getDetections().length} detection.',
                ),
            ],
            if (widget.error != null ||
                _requestError != null ||
                _localError != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  widget.error ?? _requestError ?? _localError!,
                  style: const TextStyle(color: Colors.redAccent),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
