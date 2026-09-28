import 'detection.dart';

class DetectionBatch {
  final List<Detection> detections;
  final double inferenceTimeMs;
  final double preprocessTimeMs;
  final double postprocessTimeMs;

  const DetectionBatch({
    required this.detections,
    required this.inferenceTimeMs,
    this.preprocessTimeMs = 0,
    this.postprocessTimeMs = 0,
  });
}
