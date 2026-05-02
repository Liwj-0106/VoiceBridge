"""Manual test for ASR (Automatic Speech Recognition).

Tests:
1. Loading SenseVoice model
2. Loading all wav test audio files from samples/
3. Converting audio to 16kHz mono float32
4. Transcribing each audio file
5. Language detection

Usage:
    conda activate voicebridge
    python -m tests_manual.test_asr
"""

from pathlib import Path
import sys
import os
import time
import logging

import numpy as np
import soundfile as sf
import librosa


# Ensure project root is importable when running directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Apply model cache env before importing ASR/FunASR/ModelScope
from voicebridge.models.manager import apply_cache_env

apply_cache_env()

from voicebridge.config.settings import load_settings
from voicebridge.asr.sensevoice import SenseVoiceEngine


def find_test_audios() -> list[Path]:
    """Find all available wav test audio files from samples/."""
    samples_dir = PROJECT_ROOT / "samples"

    if not samples_dir.exists():
        raise FileNotFoundError(
            f"samples directory not found: {samples_dir}\n"
            "Please create samples/ and put wav files into it."
        )

    audio_files = sorted(samples_dir.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            "No ASR test audio found in samples/.\n"
            "Please put wav files into samples/, for example:\n"
            "  samples/zh.wav\n"
            "  samples/ja.wav\n"
            "  samples/en.wav"
        )

    return audio_files


def load_audio_16k_mono(audio_path: Path):
    """Load wav audio and convert to 16kHz mono float32 numpy array."""
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")

    original_sample_rate = sample_rate
    original_shape = audio.shape

    if audio.size == 0:
        raise RuntimeError(f"Audio file is empty: {audio_path}")

    # stereo/multi-channel -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise RuntimeError(f"Unexpected audio shape: {original_shape}")

    # resample to 16k
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    audio = audio.astype(np.float32)

    # Final validation
    assert sample_rate == 16000, f"sample_rate must be 16000, got {sample_rate}"
    assert audio.ndim == 1, f"audio must be mono 1D array, got shape={audio.shape}"
    assert audio.dtype == np.float32, f"audio dtype must be float32, got {audio.dtype}"

    return audio, sample_rate, original_sample_rate, original_shape


def print_audio_info(audio_path: Path, audio: np.ndarray, sample_rate: int, original_sample_rate: int, original_shape):
    """Print audio information."""
    print(f"Audio file: {audio_path}")
    print("Audio info:")
    print(f"  original_sample_rate: {original_sample_rate}")
    print(f"  original_shape: {original_shape}")
    print(f"  final_sample_rate: {sample_rate}")
    print(f"  final_shape: {audio.shape}")
    print(f"  dtype: {audio.dtype}")
    print(f"  duration: {len(audio) / sample_rate:.2f}s")
    print(f"  max_abs: {float(np.max(np.abs(audio))):.6f}")


def run_single_asr(engine: SenseVoiceEngine, audio_path: Path, index: int, total: int):
    """Run ASR for one audio file."""
    print()
    print("-" * 60)
    print(f"ASR Test [{index}/{total}]")
    print("-" * 60)

    audio, sample_rate, original_sample_rate, original_shape = load_audio_16k_mono(audio_path)

    print_audio_info(
        audio_path=audio_path,
        audio=audio,
        sample_rate=sample_rate,
        original_sample_rate=original_sample_rate,
        original_shape=original_shape,
    )

    print()
    print("Running transcription...")
    start = time.perf_counter()
    result = engine.transcribe(audio)
    asr_ms = (time.perf_counter() - start) * 1000

    print()
    print("ASR result:")
    if result:
        result.asr_ms = asr_ms
        print(f"  language: {result.language}")
        print(f"  language_name: {result.language_name}")
        print(f"  text: {result.text}")
        print(f"  asr_ms: {result.asr_ms:.2f} ms")
    else:
        print("  No result returned")

    return result


def test_asr():
    """Test ASR functionality with all wav audio files in samples/."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("VoiceBridge ASR Batch Test")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    print(f"MODELSCOPE_CACHE: {os.environ.get('MODELSCOPE_CACHE')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print()

    print("Loading settings...")
    settings = load_settings("config.yaml")

    print("ASR settings:")
    print(f"  engine: {settings.asr.engine}")
    print(f"  model_name: {settings.asr.model_name}")
    print(f"  device: {settings.asr.device}")
    print(f"  language: {settings.asr.language}")
    print(f"  hub: {settings.asr.hub}")
    print()

    audio_files = find_test_audios()
    print(f"Found {len(audio_files)} wav file(s) in samples/:")
    for path in audio_files:
        print(f"  - {path.name}")
    print()

    print("Loading ASR model (this may take a while on first run)...")
    engine = SenseVoiceEngine(settings.asr)

    success_count = 0
    fail_count = 0

    for i, audio_path in enumerate(audio_files, start=1):
        try:
            result = run_single_asr(
                engine=engine,
                audio_path=audio_path,
                index=i,
                total=len(audio_files),
            )

            if result and result.text:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            fail_count += 1
            print()
            print(f"[ERROR] Failed to process {audio_path}: {e}")

    print()
    print("=" * 60)
    print("ASR Batch Test Summary")
    print("=" * 60)
    print(f"Total files: {len(audio_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print("ASR test complete.")


if __name__ == "__main__":
    test_asr()