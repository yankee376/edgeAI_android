import 'detection.dart';

class DetectionBatch {
  final List<Detection> detections;
  final double inferenceTimeMs;

  const DetectionBatch({
    required this.detections,
    required this.inferenceTimeMs,
  });
}
