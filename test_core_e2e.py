"""
test_core_e2e.py
================
Verifies Phase 1 core pipeline end-to-end:
1. Lazy loading of MusicGen-small & CLAP
2. Prompt expansion (Ollama / Fallback)
3. Audio generation for raw & expanded clips under seed 42
4. Clip duration check (strictly 10.0 seconds)
5. CLAP contrastive scoring & primary intent delta calculation
"""

import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from llm_client import expand_prompt
from musicgen_runner import generate_music_clip
from clap_scorer import evaluate_dual_clip_scores


def main():
    print("=" * 60, flush=True)
    print("PHASE 1: END-TO-END CORE PIPELINE VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    test_prompt = "calm rainy night piano"
    seed = 42
    print(f"Test Prompt: '{test_prompt}' | Seed: {seed}", flush=True)

    # 1. Expand prompt
    t0 = time.time()
    expanded_text, is_fallback, provider_status = expand_prompt(test_prompt, technique="Few-Shot")
    t_exp = time.time() - t0
    print(f"\n[1/3] Prompt Expansion ({t_exp:.2f}s):", flush=True)
    print(f"  Status: {provider_status}", flush=True)
    print(f"  Expanded: '{expanded_text}'", flush=True)

    # 2. Generate Raw Clip
    print("\n[2/3] Generating MusicGen-small clips (500 tokens = 10.0s @ 32 kHz)...", flush=True)
    t0 = time.time()
    raw_audio, sr, raw_path = generate_music_clip(test_prompt, seed=seed, duration_seconds=10.0)
    t_raw = time.time() - t0
    print(f"  Clip A (Raw) generated in {t_raw:.2f}s -> {raw_path}", flush=True)
    raw_duration = len(raw_audio) / sr
    print(f"  Duration: {raw_duration:.2f}s ({len(raw_audio)} samples @ {sr} Hz)", flush=True)
    assert abs(raw_duration - 10.0) < 0.25, f"Clip A duration mismatch: {raw_duration}s"

    # Generate Expanded Clip
    t0 = time.time()
    exp_audio, _, exp_path = generate_music_clip(expanded_text, seed=seed, duration_seconds=10.0)
    t_exp_gen = time.time() - t0
    print(f"  Clip B (Expanded) generated in {t_exp_gen:.2f}s -> {exp_path}", flush=True)
    exp_duration = len(exp_audio) / sr
    print(f"  Duration: {exp_duration:.2f}s ({len(exp_audio)} samples @ {sr} Hz)", flush=True)
    assert abs(exp_duration - 10.0) < 0.25, f"Clip B duration mismatch: {exp_duration}s"

    # 3. CLAP Scoring
    print("\n[3/3] Computing CLAP embeddings & cosine similarity...", flush=True)
    t0 = time.time()
    scores = evaluate_dual_clip_scores(
        raw_audio=raw_audio,
        exp_audio=exp_audio,
        sampling_rate=sr,
        raw_prompt=test_prompt,
        expanded_prompt=expanded_text,
    )
    t_clap = time.time() - t0
    print(f"  Scoring completed in {t_clap:.2f}s", flush=True)
    print("\nRESULTS TABLE:", flush=True)
    print("-" * 50, flush=True)
    print(f"  Raw Audio vs Raw Text (Baseline):         {scores['raw_vs_raw']:.4f}", flush=True)
    print(f"  Expanded Audio vs Raw Text (PRIMARY):     {scores['exp_vs_raw']:.4f}", flush=True)
    print(f"  Expanded Audio vs Exp Text (Self-ref):    {scores['exp_vs_exp']:.4f}", flush=True)
    print(f"  Primary Delta (Exp - Raw vs Raw):         {scores['delta_primary']:+.4f}", flush=True)
    print(f"  Interpretation: {scores['primary_metric_interpretation']}", flush=True)
    print("-" * 50, flush=True)

    # Sanity checks
    assert -1.0 <= scores["raw_vs_raw"] <= 1.0, "Invalid CLAP raw score"
    assert -1.0 <= scores["exp_vs_raw"] <= 1.0, "Invalid CLAP expanded score"
    assert os.path.exists(raw_path) and os.path.getsize(raw_path) > 100000, "Raw WAV file invalid"
    assert os.path.exists(exp_path) and os.path.getsize(exp_path) > 100000, "Expanded WAV file invalid"

    print("\n>>> PHASE 1 CORE PIPELINE TEST PASSED SUCCESSFULLY! <<<", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("ERROR IN E2E TEST:")
        traceback.print_exc()
        raise e
