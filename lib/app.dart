import 'package:flutter/material.dart';
import 'core/theme/app_theme.dart';
import 'features/camera/presentation/litert_camera_screen.dart';

class AndroidAIGatewayApp extends StatelessWidget {
  const AndroidAIGatewayApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeAI LiteRT 640',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      home: const LiteRtCameraScreen(),
    );
  }
}
