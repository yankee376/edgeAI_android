import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

class NanoDetModelFiles {
  final String paramPath;
  final String binPath;

  const NanoDetModelFiles({required this.paramPath, required this.binPath});
}

class CurrencyModelFiles {
  final String paramPath;
  final String binPath;

  const CurrencyModelFiles({required this.paramPath, required this.binPath});
}

enum Yolo26Mode { fp16Vulkan, int8Cpu, fp16Cpu }

extension Yolo26ModeLabel on Yolo26Mode {
  String get label => switch (this) {
    Yolo26Mode.fp16Vulkan => 'FP16 + Vulkan/GPU',
    Yolo26Mode.int8Cpu => 'INT8 + CPU',
    Yolo26Mode.fp16Cpu => 'FP16 + CPU',
  };

  String get assetPrefix => switch (this) {
    Yolo26Mode.fp16Vulkan || Yolo26Mode.fp16Cpu => 'yolo26_fp16',
    Yolo26Mode.int8Cpu => 'yolo26_int8',
  };
}

class Yolo26ModelFiles {
  final String paramPath;
  final String binPath;
  const Yolo26ModelFiles({required this.paramPath, required this.binPath});
}

/// Chuẩn bị model NCNN thành các file thật trên bộ nhớ riêng của ứng dụng.
///
/// Flutter assets nằm bên trong APK nên C++ không thể dùng trực tiếp như một
/// đường dẫn thông thường. Service này copy model sang Application Support.
class ModelAssetService {
  static const String _paramAsset = 'assets/models/nanodet-ELite0_320.param';

  static const String _binAsset = 'assets/models/nanodet-ELite0_320.bin';

  static const String _currencyParamAsset =
      'assets/models/nanodet_currency.ncnn.param';
  static const String _currencyBinAsset =
      'assets/models/nanodet_currency.ncnn.bin';

  Future<Yolo26ModelFiles> prepareYolo26Model(Yolo26Mode mode) async {
    final support = await getApplicationSupportDirectory();
    final directory = Directory('${support.path}/models');
    await directory.create(recursive: true);
    final prefix = mode.assetPrefix;
    final cacheVersion = mode == Yolo26Mode.int8Cpu
        ? 'int8_risk_v3'
        : 'fp16_benchmark_v1';
    final stem = '${prefix}_640_$cacheVersion';
    final param = File('${directory.path}/$stem.ncnn.param');
    final bin = File('${directory.path}/$stem.ncnn.bin');
    await _copyAssetIfNeeded(
      assetPath: 'assets/models/$prefix.ncnn.param',
      destination: param,
    );
    await _copyAssetIfNeeded(
      assetPath: 'assets/models/$prefix.ncnn.bin',
      destination: bin,
    );
    return Yolo26ModelFiles(paramPath: param.path, binPath: bin.path);
  }

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

  Future<CurrencyModelFiles> prepareCurrencyModel() async {
    final Directory supportDirectory = await getApplicationSupportDirectory();
    final Directory modelDirectory = Directory(
      '${supportDirectory.path}/models',
    );
    await modelDirectory.create(recursive: true);

    final File paramFile = File(
      '${modelDirectory.path}/nanodet_currency.ncnn.param',
    );
    final File binFile = File(
      '${modelDirectory.path}/nanodet_currency.ncnn.bin',
    );

    await _copyAssetIfNeeded(
      assetPath: _currencyParamAsset,
      destination: paramFile,
    );
    await _copyAssetIfNeeded(
      assetPath: _currencyBinAsset,
      destination: binFile,
    );

    return CurrencyModelFiles(paramPath: paramFile.path, binPath: binFile.path);
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
