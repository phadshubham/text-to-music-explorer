import os
import time
import musicgen_runner
import clap_scorer

t0 = time.time()
print("=" * 60, flush=True)
print("PHASE 1 END-TO-END VERIFICATION TEST", flush=True)
print("=" * 60, flush=True)

prompt = "calm rainy night piano"
seed = 42

print(f"Generating 10.0s audio for: '{prompt}' (Seed: {seed})...", flush=True)
os.makedirs("results/audio", exist_ok=True)
out_wav = "results/audio/phase1_test_raw.wav"

audio, sr, path = musicgen_runner.generate_music_clip(prompt, seed=seed, output_path=out_wav)
duration = len(audio) / sr

print(f"Generation successful in {time.time() - t0:.1f}s!", flush=True)
print(f"File: {path} | Samples: {len(audio)} | SR: {sr} Hz | Duration: {duration:.2f}s", flush=True)
assert abs(duration - 10.0) < 0.2, f"Expected 10.0s duration, got {duration}s"

t1 = time.time()
print("\nComputing CLAP score against raw prompt...", flush=True)
score = clap_scorer.compute_clap_similarity(audio, sr, prompt)
print(f"CLAP Score: {score:.4f} (calculated in {time.time() - t1:.2f}s)", flush=True)

print("=" * 60, flush=True)
print("PHASE 1 VERIFICATION COMPLETE: ALL ASSERTIONS PASSED", flush=True)
print("=" * 60, flush=True)
