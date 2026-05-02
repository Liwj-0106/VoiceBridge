"""Manual test for SubtitleWindow UI.

Tests:
1. Window creation and display
2. ASR result display
3. Translation result display
4. Error message display
5. Clear button functionality

Usage:
    conda activate voicebridge
    python -m tests_manual.test_subtitle_window
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from voicebridge.ui.subtitle_window import SubtitleWindow
from voicebridge.common.events import ASRResult, TranslationResult
from voicebridge.config.schema import UISettings


def test_subtitle_window():
    """Test subtitle window functionality."""
    print("=" * 60)
    print("VoiceBridge SubtitleWindow Test")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print()

    # Create QApplication (required for PyQt)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create UI settings
    ui_settings = UISettings(
        topmost=True,
        show_original=True,
        window_width=900,
        window_height=220,
    )

    # Create and show subtitle window
    print("Creating SubtitleWindow...")
    window = SubtitleWindow(ui_settings)
    window.show()
    print(f"Window created: {window.windowTitle()}")
    print()

    # Test 1: Display initial state
    print("Test 1: Initial state")
    print(f"  original_label: '{window.original_label.text()}'")
    print(f"  translation_label: '{window.translation_label.text()}'")
    assert "等待识别" in window.original_label.text(), "Initial state mismatch"
    print("  PASS: Initial state correct")
    print()

    # Test 2: Simulate ASR result
    print("Test 2: ASR result display")
    asr_result = ASRResult(
        text="Hello, this is a test of the subtitle window.",
        language="en",
        language_name="English",
        asr_ms=150.0,
    )
    window.add_original_text(asr_result)
    print(f"  original_label: '{window.original_label.text()}'")
    print(f"  translation_label: '{window.translation_label.text()}'")
    assert "[en]" in window.original_label.text(), "ASR result language tag missing"
    assert "Hello" in window.original_label.text(), "ASR result text missing"
    assert "翻译中..." == window.translation_label.text(), "Translation status missing"
    print("  PASS: ASR result displayed correctly")
    print()

    # Test 3: Simulate translation result
    print("Test 3: Translation result display")
    translation_result = TranslationResult(
        original="Hello, this is a test of the subtitle window.",
        translated="你好，这是字幕窗口的测试。",
        source_language="en",
        target_language="zh",
        translate_ms=200.0,
    )
    window.update_translation(translation_result)
    print(f"  original_label: '{window.original_label.text()}'")
    print(f"  translation_label: '{window.translation_label.text()}'")
    assert "[en]" in window.original_label.text(), "Translation source language missing"
    assert "Hello" in window.original_label.text(), "Translation original text missing"
    assert "你好" in window.translation_label.text(), "Translation missing"
    print("  PASS: Translation displayed correctly")
    print()

    # Test 4: Error display
    print("Test 4: Error display")
    window.show_error("API调用失败，请检查网络连接")
    print(f"  translation_label: '{window.translation_label.text()}'")
    assert "[错误]" in window.translation_label.text(), "Error prefix missing"
    assert "API调用失败" in window.translation_label.text(), "Error message missing"
    print("  PASS: Error displayed correctly")
    print()

    # Test 5: Clear button
    print("Test 5: Clear button")
    window.clear()
    print(f"  original_label: '{window.original_label.text()}'")
    print(f"  translation_label: '{window.translation_label.text()}'")
    assert window.original_label.text() == "", "Original label not cleared"
    assert window.translation_label.text() == "", "Translation label not cleared"
    print("  PASS: Clear button works correctly")
    print()

    # Schedule window close after a short delay
    print("Window is displayed. Closing in 2 seconds...")
    QTimer.singleShot(2000, app.quit)

    # Run event loop briefly to verify no crashes
    print("Running event loop to verify stability...")
    app.processEvents()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(test_subtitle_window())