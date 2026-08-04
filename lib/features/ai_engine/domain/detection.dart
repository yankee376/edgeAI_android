class Detection {
  final int classId;
  final double confidence;
  final double x;
  final double y;
  final double width;
  final double height;

  const Detection({
    required this.classId,
    required this.confidence,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
}
