"""Manual test for VAD (Voice Activity Detection).

Tests:
1. Silero VAD confidence scoring
2. Speech segment detection
3. Integration with AudioCapture

Usage:
    conda activate voicebridge
    python -m tests_manual.test_vad
"""

from pathlib import Path
import sys
import os
import logging
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voicebridge.models.manager import apply_cache_env

apply_cache_env()

print(f"TORCH_HOME={os.environ.get('TORCH_HOME')}")

from voicebridge.config.settings import load_settings
from voicebridge.audio.capture import AudioCapture
from voicebridge.vad.segmenter import VADSegmenter


def compute_rms(audio):
    """Calculate RMS of audio."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def test_vad():
    """Test VAD functionality with audio capture."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("VoiceBridge VAD Test")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print()

    # Load settings
    print("Loading settings...")
    settings = load_settings("config.yaml")

    print("VAD settings:")
    print(f"  mode: {settings.vad.mode}")
    print(f"  threshold: {settings.vad.threshold}")
    print(f"  min_speech_duration: {settings.vad.min_speech_duration}s")
    print(f"  max_speech_duration: {settings.vad.max_speech_duration}s")
    print(f"  silence_duration: {settings.vad.silence_duration}s")
    print(f"  chunk_duration: {settings.vad.chunk_duration}s")
    print()

    # Initialize VAD
    print("Initializing VAD segmenter...")
    vad = VADSegmenter(settings.vad)
    print()

    # Start audio capture
    print("Starting audio capture...")
    audio = AudioCapture(settings.audio)
    audio.start()
    print()

    segment_count = 0

    try:
        print("Processing audio chunks...")
        print("Speak or play audio to trigger VAD detection.")
        print("Press Ctrl+C to stop.")
        print()

        for i in range(200):  # Process up to 200 chunks
            chunk = audio.get_audio(timeout=1)

            if chunk is None:
                continue

            rms = compute_rms(chunk.audio)

            # Process chunk through VAD
            segment = vad.process_chunk(chunk.audio)

            if segment is not None:
                segment_count += 1
                seg_rms = compute_rms(segment.audio)

                print(
                    f"SpeechSegment {segment_count}: "
                    f"duration={segment.duration:.2f}s, "
                    f"shape={segment.audio.shape}, "
                    f"RMS={seg_rms:.6f}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print()
        print("Stopping audio capture...")
        audio.stop()

    print()
    print("Result:")
    print(f"  detected segments: {segment_count}")

    if segment_count == 0:
        print()
        print("WARNING: No speech segments detected.")
        print("Possible causes:")
        print("  1. No audio was played during the test")
        print("  2. Audio level too low")
        print("  3. VAD threshold too high")
        print()
        print("Tip: Lower the VAD threshold in config.yaml if needed.")

    print()
    print("VAD test complete.")


if __name__ == "__main__":
    test_vad()
