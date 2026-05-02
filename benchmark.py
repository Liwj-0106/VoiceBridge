"""Standalone benchmark runner. Real logic lives in voicebridge/benchmark.py."""
from voicebridge.benchmark import run_benchmark, BENCH_SENTENCES

if __name__ == "__main__":
    print("Benchmark module. Import run_benchmark from voicebridge.benchmark to use.")
    print(f"Available languages: {', '.join(BENCH_SENTENCES.keys())}")
