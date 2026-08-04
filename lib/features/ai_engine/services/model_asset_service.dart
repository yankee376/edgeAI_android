import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

class NanoDetModelFiles {
  final String paramPath;
  final String binPath;

  const NanoDetModelFiles({required this.paramPath, required this.binPath});
}

/// Chuẩn bị model NCNN thành các file thật trên bộ nhớ riêng của ứng dụng.
///
/// Flutter assets nằm bên trong APK nên C++ không thể dùng trực tiếp như một
/// đường dẫn thông thường. Service này copy model sang Application Support.
class ModelAssetService {
  static const String _paramAsset = 'assets/models/nanodet-ELite0_320.param';

  static const String _binAsset = 'assets/models/nanodet-ELite0_320.bin';

  Future<NanoDetModelFiles> prepareNanoDetModel() async {
    final Directory supportDirectory = await getApplicationSupportDirectory();

    final Directory modelDirectory = Directory(
      '${supportDirectory.path}/models',
    );

    await modelDirectory.create(recursive: true);

    final File paramFile = File(
      '${modelDirectory.path}/nanodet-ELite0_320.param',
    );

    final File binFile = File('${modelDirectory.path}/nanodet-ELite0_320.bin');

    await _copyAssetIfNeeded(assetPath: _paramAsset, destination: paramFile);

    await _copyAssetIfNeeded(assetPath: _binAsset, destination: binFile);

    return NanoDetModelFiles(paramPath: paramFile.path, binPath: binFile.path);
  }

  Future<void> _copyAssetIfNeeded({
    required String assetPath,
    required File destination,
  }) async {
    final ByteData data = await rootBundle.load(assetPath);

    final Uint8List bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );

    // Tránh ghi lại model trong mỗi lần ứng dụng khởi động.
    if (await destination.exists()) {
      final int currentLength = await destination.length();

      if (currentLength == bytes.length) {
        return;
      }
    }

    await destination.writeAsBytes(bytes, flush: true);
  }
}
