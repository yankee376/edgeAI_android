import 'package:flutter/material.dart';

class AppTheme {
  static ThemeData get darkTheme {
    return ThemeData.dark(useMaterial3: true).copyWith(
      scaffoldBackgroundColor: const Color(0xFF0F0E17),
      colorScheme: const ColorScheme.dark(
        primary: Color(0xFF7F5AF0),
        secondary: Color(0xFF2CB67D),
        surface: Color(0xFF16161A),
        error: Color(0xFFFF5470),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Color(0xFF16161A),
        centerTitle: true,
        elevation: 0,
      ),
      cardTheme: CardThemeData(
        color: const Color(0xFF16161A),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    );
  }
}
