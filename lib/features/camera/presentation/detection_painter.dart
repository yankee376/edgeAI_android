import 'package:flutter/material.dart';

import '../../ai_engine/domain/detection.dart';
import '../data/coco_labels.dart';

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;

  const DetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth <= 0 || imageHeight <= 0) return;

    final scaleX = size.width / imageWidth;
    final scaleY = size.height / imageHeight;

    for (final detection in detections) {
      final color =
          Colors.primaries[detection.classId.abs() % Colors.primaries.length];
      final rect = Rect.fromLTWH(
        detection.x * scaleX,
        detection.y * scaleY,
        detection.width * scaleX,
        detection.height * scaleY,
      );

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3;
      canvas.drawRect(rect, boxPaint);

      final className =
          detection.classId >= 0 && detection.classId < cocoLabels.length
          ? cocoLabels[detection.classId]
          : 'class ${detection.classId}';
      final label =
          '$className ${(detection.confidence * 100).toStringAsFixed(1)}%';

      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: size.width);

      final labelLeft = rect.left
          .clamp(0.0, size.width - textPainter.width)
          .toDouble();
      final preferredTop = rect.top - textPainter.height - 6;
      final labelTop = preferredTop >= 0 ? preferredTop : rect.top;
      final background = Rect.fromLTWH(
        labelLeft - 3,
        labelTop - 2,
        textPainter.width + 6,
        textPainter.height + 4,
      );

      canvas.drawRect(background, Paint()..color = color.withAlpha(225));
      textPainter.paint(canvas, Offset(labelLeft, labelTop));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.imageWidth != imageWidth ||
        oldDelegate.imageHeight != imageHeight;
  }
}
