"""Manual test for audio capture.

Tests:
1. Opening WASAPI loopback device
2. Reading audio chunks
3. RMS calculation
4. Detecting silent failure

Usage:
    conda activate voicebridge
    python -m tests_manual.test_audio_capture
"""

import logging
import sys
import numpy as np

from voicebridge.audio.devices import (
    find_wasapi_host,
    get_default_output_device,
    list_loopback_devices,
    find_loopback_device,
)
from voicebridge.config.settings import load_settings
from voicebridge.audio.capture import AudioCapture


def test_audio_capture():
    """Test audio capture functionality."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("VoiceBridge Audio Capture Test")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print()

    # Load settings
    print("Loading settings...")
    settings = load_settings("config.yaml")

    print("Audio settings:")
    print(f"  device: {settings.audio.device}")
    print(f"  sample_rate: {settings.audio.sample_rate}")
    print(f"  chunk_duration: {settings.audio.chunk_duration}")
    print()

    # List audio devices
    import pyaudiowpatch as pyaudio
    pa = pyaudio.PyAudio()

    print("Audio devices:")
    print(f"  WASAPI host: {find_wasapi_host(pa)['name']}")
    print(f"  Default output: {get_default_output_device(pa)['name']}")
    print(f"  Loopback candidates:")
    for dev in list_loopback_devices(pa):
        print(f"    - {dev['name']}")
    print()

    # Find and print selected device
    selected = find_loopback_device(pa, settings.audio.device)
    print(f"Selected loopback: {selected['name']}")
    print()

    pa.terminate()

    # Start capture
    print("Starting audio capture...")
    audio = AudioCapture(settings.audio)
    audio.start()

    success_count = 0
    all_chunks = []

    try:
        print("Capturing 20 audio chunks...")
        print("Please play a video or music during this test.")
        print()

        for i in range(20):
            chunk = audio.get_audio(timeout=1)

            if chunk is None:
                print(f"Chunk {i}: TIMEOUT, no audio chunk received")
                continue

            success_count += 1
            all_chunks.append(chunk)

            max_abs = float(np.max(np.abs(chunk.audio))) if len(chunk.audio) > 0 else 0.0

            print(
                f"Chunk {i}: "
                f"RMS={chunk.rms:.6f}, "
                f"max_abs={max_abs:.6f}, "
                f"shape={chunk.audio.shape}, "
                f"sample_rate={chunk.sample_rate}, "
                f"dtype={chunk.audio.dtype}"
            )

            # Validate format
            if chunk.sample_rate != 16000:
                print(f"  WARNING: Invalid sample_rate: {chunk.sample_rate}")

            if chunk.audio.dtype != np.float32:
                print(f"  WARNING: Invalid dtype: {chunk.audio.dtype}")

            if chunk.audio.ndim != 1:
                print(f"  WARNING: Audio should be mono, got shape={chunk.audio.shape}")

    finally:
        print()
        print("Stopping audio capture...")
        audio.stop()

    print()
    print("Result:")
    print(f"  received chunks: {success_count}/20")

    # Check for all None
    if success_count == 0:
        raise RuntimeError(
            "Audio capture failed: no audio chunks were received. "
            "Check WASAPI loopback device or _read_loop errors in logs."
        )

    # Check for all silence
    if all_chunks:
        non_silent_count = sum(1 for c in all_chunks if c.rms > 0.001)
        print(f"  non-silent chunks: {non_silent_count}/{success_count}")

        if non_silent_count == 0:
            print()
            print("WARNING: All chunks have RMS near zero.")
            print("Possible causes:")
            print("  1. Wrong loopback device selected (check log for device name)")
            print("  2. No audio being played on the selected output device")
            print("  3. System audio is muted")
            print()
            print("Tip: Check the log above for 'Selected loopback device'")
            print("     and verify it matches your active audio output.")

    print()
    print("Audio capture test passed.")


if __name__ == "__main__":
    test_audio_capture()
