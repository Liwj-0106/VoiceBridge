"""Manual test for Translator.

Tests:
1. OpenAI-compatible API connection
2. Text translation

Usage:
    conda activate voicebridge
    python tests_manual/test_translator.py

Requirements:
    - Set VOICEBRIDGE_API_KEY in .env or config.yaml
"""

from voicebridge.config.settings import load_settings
from voicebridge.translate.translator import Translator


def test_translator():
    """Test translation functionality."""
    print("Loading settings...")
    settings = load_settings("config.yaml")

    print("Initializing translator...")
    translator = Translator(settings.translation)

    test_text = "Hello, today we are talking about artificial intelligence."

    print(f"Translating: {test_text}")
    result = translator.translate(test_text, "en")

    print(f"Result: {result}")


if __name__ == "__main__":
    test_translator()
