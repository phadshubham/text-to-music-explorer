"""
Benchmark Runner
Executes matrix evaluation: 4 Fixed Prompts × 3 Conditions (raw, zero_shot, few_shot) × 3 Seeds (42, 100, 2024).
Total runs: 36.
Saves audio waveforms to results/audio/ and all empirical metrics to results/results.json.
Auto-detects CUDA acceleration when available; runs on CPU otherwise.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Ensure OpenMP runtime conflict does not crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import scipy.io.wavfile

from musicgen_runner import generate_music_clip
from clap_scorer import compute_clap_similarity, get_text_embedding, get_audio_embedding
from llm_client import expand_prompt
from prompts.templates import ZERO_SHOT, FEW_SHOT

EVAL_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "data", "eval_prompts.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
AUDIO_DIR = os.path.join(RESULTS_DIR, "audio")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
RAW_EMBEDDINGS_JSON = os.path.join(RESULTS_DIR, "raw_embeddings.json")
PCA_JSON = os.path.join(RESULTS_DIR, "pca_embeddings.json")

SEEDS = [42, 100, 2024]
CONDITIONS = ["raw", "zero_shot", "few_shot"]


def run_benchmark():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    assert os.path.exists(EVAL_PROMPTS_FILE), f"Missing {EVAL_PROMPTS_FILE}"

    with open(EVAL_PROMPTS_FILE, "r", encoding="utf-8") as f:
        eval_prompts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"STARTING BENCHMARK MATRIX EVALUATION ON {device.upper()}")
    print(f"• Prompts: {len(eval_prompts)} | Conditions: {len(CONDITIONS)} | Seeds: {len(SEEDS)}")
    print(f"• Total generations: {len(eval_prompts) * len(CONDITIONS) * len(SEEDS)} runs")
    print("=" * 70)

    all_runs = []
    embeddings_store = []
    completed_keys = set()

    # Resume from existing runs if benchmark was previously started
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                all_runs = prev_data.get("runs", [])
                for r in all_runs:
                    completed_keys.add((r["prompt_id"], r["condition"], r["seed"]))
            print(f"[Benchmark] Found {len(all_runs)} previously completed runs in {RESULTS_JSON}")
        except Exception as e:
            print(f"[Benchmark] Note: Could not read existing {RESULTS_JSON} ({e}); starting fresh.")
            all_runs = []

    if os.path.exists(RAW_EMBEDDINGS_JSON):
        try:
            with open(RAW_EMBEDDINGS_JSON, "r", encoding="utf-8") as f:
                loaded_emb = json.load(f)
                if isinstance(loaded_emb, list):
                    embeddings_store = loaded_emb
        except Exception:
            embeddings_store = []

    t_start_total = time.time()

    for seed in SEEDS:
        print(f"\n{'='*30} PROCESSING SEED {seed} {'='*30}", flush=True)
        for p_idx, item in enumerate(eval_prompts):
            raw_prompt = item["prompt"]
            prompt_id = item["id"]
            print(f"\n[{p_idx+1}/{len(eval_prompts)}] Prompt ({prompt_id}): '{raw_prompt}' | Seed: {seed}", flush=True)

            # 1. Expand prompt for zero-shot and few-shot
            exp_zero, label_zero, _ = expand_prompt(raw_prompt, technique=ZERO_SHOT)
            exp_few, label_few, _ = expand_prompt(raw_prompt, technique=FEW_SHOT)

            prompt_variants = {
                "raw": (raw_prompt, "Original Raw Prompt"),
                "zero_shot": (exp_zero, label_zero),
                "few_shot": (exp_few, label_few),
            }

            for cond in CONDITIONS:
                if (prompt_id, cond, seed) in completed_keys:
                    print(f"  -> [{prompt_id} | {cond.upper()} | Seed {seed}] Already completed. Skipping.", flush=True)
                    continue

                text_to_generate, status_label = prompt_variants[cond]
                safe_prompt_slug = "".join(c if c.isalnum() else "_" for c in raw_prompt[:16]).strip("_")
                audio_filename = f"{prompt_id}_{cond}_s{seed}_{safe_prompt_slug}.wav"
                output_path = os.path.join(AUDIO_DIR, audio_filename)

                # Check if audio file was already saved on disk
                if os.path.exists(output_path):
                    print(f"  -> Loading cached audio from {output_path}...", flush=True)
                    try:
                        sr_read, audio_raw = scipy.io.wavfile.read(output_path)
                        if audio_raw.dtype == np.int16:
                            audio_np = audio_raw.astype(np.float32) / 32767.0
                        else:
                            audio_np = audio_raw.astype(np.float32)
                        sr = sr_read
                        saved_path = output_path
                        gen_time = 0.0
                    except Exception as e:
                        print(f"  -> Corrupted cache ({e}), regenerating...", flush=True)
                        t0 = time.time()
                        audio_np, sr, saved_path = generate_music_clip(
                            prompt=text_to_generate,
                            seed=seed,
                            output_path=output_path,
                        )
                        gen_time = time.time() - t0
                else:
                    t0 = time.time()
                    audio_np, sr, saved_path = generate_music_clip(
                        prompt=text_to_generate,
                        seed=seed,
                        output_path=output_path,
                    )
                    gen_time = time.time() - t0

                # Compute Primary Intent Score (evaluated vs original raw prompt)
                primary_score = compute_clap_similarity(audio_np, sr, raw_prompt)

                # Compute Self-Reference Score (evaluated vs generating text)
                self_score = compute_clap_similarity(audio_np, sr, text_to_generate)

                # Extract raw 512-D embeddings for Latent View (PCA)
                text_emb = get_text_embedding(text_to_generate).squeeze().cpu().numpy().tolist()
                audio_emb = get_audio_embedding(audio_np, sr).squeeze().cpu().numpy().tolist()

                run_record = {
                    "prompt_id": prompt_id,
                    "raw_prompt": raw_prompt,
                    "condition": cond,
                    "generating_text": text_to_generate,
                    "prompt_used": text_to_generate,
                    "expansion_label": status_label,
                    "seed": seed,
                    "audio_filename": audio_filename,
                    "audio_rel_path": os.path.relpath(saved_path, start=RESULTS_DIR).replace("\\", "/"),
                    "duration_seconds": round(len(audio_np) / float(sr), 2),
                    "primary_clap_score": round(primary_score, 4),
                    "score_vs_raw": round(primary_score, 4),
                    "self_clap_score": round(self_score, 4),
                    "score_vs_gen": round(self_score, 4),
                    "generation_time_seconds": round(gen_time, 2),
                    "prompt_char_length": len(text_to_generate),
                    "prompt_word_count": len(text_to_generate.split()),
                }

                all_runs.append(run_record)
                completed_keys.add((prompt_id, cond, seed))
                embeddings_store.append({
                    "prompt_id": prompt_id,
                    "raw_prompt": raw_prompt,
                    "condition": cond,
                    "seed": seed,
                    "text_embedding": text_emb,
                    "audio_embedding": audio_emb,
                })

                # Incrementally save results so zero progress is ever lost
                benchmark_payload = {
                    "metadata": {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_runs": len(all_runs),
                        "device": device,
                        "total_elapsed_seconds": round(time.time() - t_start_total, 2),
                        "musicgen_model": "facebook/musicgen-small",
                        "clap_model": "laion/clap-htsat-unfused",
                    },
                    "runs": all_runs,
                }
                with open(RESULTS_JSON, "w", encoding="utf-8") as f:
                    json.dump(benchmark_payload, f, indent=2)
                with open(RAW_EMBEDDINGS_JSON, "w", encoding="utf-8") as f:
                    json.dump(embeddings_store, f)

                print(
                    f"  -> [{cond.upper()} | Seed {seed}] Saved! "
                    f"Primary CLAP: {primary_score:.4f} | Self CLAP: {self_score:.4f}",
                    flush=True
                )

    print(f"\n[Benchmark Complete] Successfully saved {len(all_runs)} runs to {RESULTS_JSON}", flush=True)
    print(f"[Latent Embeddings] Successfully saved {len(embeddings_store)} vectors to {PCA_JSON}", flush=True)


if __name__ == "__main__":
    run_benchmark()
