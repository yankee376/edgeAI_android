import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'app.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  List<CameraDescription> cameras = [];
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Error loading cameras: $e');
  }

  runApp(AndroidAIGatewayApp(cameras: cameras));
}
