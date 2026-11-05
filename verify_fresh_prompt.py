"""
Verification Script: Fresh Prompt End-to-End Pipeline
Validates:
1. Inbound Guardrail (U5)
2. Prompt Expansion (U1 + U2)
3. Synthesis of 2 clips with matched seed via MusicGen-small (U4)
4. CLAP dual-metric scoring (U3): Primary vs Raw Intent and Self-Reference
"""

import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guardrail import check_guardrail
from llm_client import expand_prompt
from musicgen_runner import generate_music_clip
from clap_scorer import evaluate_dual_clip_scores
from prompts.templates import FEW_SHOT


def run_fresh_prompt_verification(prompt: str = "driving funk bass groove", seed: int = 42):
    print("=" * 70)
    print("FRESH PROMPT END-TO-END PIPELINE VERIFICATION")
    print(f"Target Prompt: '{prompt}' | Seed: {seed}")
    print("=" * 70)

    # Step 1: Guardrail Validation
    print("\n[Step 1] Checking Named-Artist Guardrail (Unit 5)...")
    is_safe, block_reason, suggestion = check_guardrail(prompt)
    print(f"  -> Safe: {is_safe} | Reason: {block_reason}")
    assert is_safe, f"Expected prompt '{prompt}' to pass guardrail, but got blocked: {block_reason}"

    # Step 2: Prompt Expansion
    print("\n[Step 2] Expanding prompt using Few-Shot In-Context Template (Unit 1 & 2)...")
    expanded_text, provider_status, template_used = expand_prompt(prompt, technique=FEW_SHOT)
    print(f"  -> Provider Status: {provider_status}")
    print(f"  -> Expanded Prompt: '{expanded_text}'")
    assert len(expanded_text.split()) <= 45, "Expanded prompt exceeds word limit"

    # Step 3: MusicGen Synthesis (Clip A: Raw Prompt)
    print("\n[Step 3] Synthesizing Clip A (Raw Prompt) via MusicGen-small...")
    t0 = time.time()
    raw_audio, sr, raw_path = generate_music_clip(
        prompt=prompt,
        seed=seed,
        duration_seconds=10.0,
        output_path=f"results/audio/fresh_raw_s{seed}.wav",
    )
    t_raw = time.time() - t0
    clip_dur_raw = len(raw_audio) / float(sr)
    print(f"  -> Clip A saved to: {raw_path}")
    print(f"  -> Samples: {len(raw_audio)} at {sr} Hz ({clip_dur_raw:.2f}s) in {t_raw:.1f}s")
    assert abs(clip_dur_raw - 10.0) < 0.25, f"Clip duration {clip_dur_raw}s deviates from 10.0s target"

    # Step 4: MusicGen Synthesis (Clip B: Expanded Prompt, Matched Seed)
    print("\n[Step 4] Synthesizing Clip B (Expanded Prompt, Same Seed) via MusicGen-small...")
    t0 = time.time()
    exp_audio, _, exp_path = generate_music_clip(
        prompt=expanded_text,
        seed=seed,
        duration_seconds=10.0,
        output_path=f"results/audio/fresh_exp_s{seed}.wav",
    )
    t_exp = time.time() - t0
    clip_dur_exp = len(exp_audio) / float(sr)
    print(f"  -> Clip B saved to: {exp_path}")
    print(f"  -> Samples: {len(exp_audio)} at {sr} Hz ({clip_dur_exp:.2f}s) in {t_exp:.1f}s")
    assert abs(clip_dur_exp - 10.0) < 0.25, f"Clip duration {clip_dur_exp}s deviates from 10.0s target"

    # Step 5: CLAP Dual-Scoring
    print("\n[Step 5] Computing CLAP Cosine Similarities at 48 kHz (Unit 3)...")
    scores = evaluate_dual_clip_scores(
        raw_audio=raw_audio,
        exp_audio=exp_audio,
        sampling_rate=sr,
        raw_prompt=prompt,
        expanded_prompt=expanded_text,
    )

    print("\n" + "=" * 70)
    print("EMPIRICAL EVALUATION RESULTS:")
    print(f"• Raw Clip vs Raw Intent (Baseline):       {scores['raw_vs_raw']:.4f}")
    print(f"• Expanded Clip vs Raw Intent (PRIMARY):   {scores['exp_vs_raw']:.4f}")
    print(f"• Expanded Clip vs Expanded (Self-Score):  {scores['exp_vs_exp']:.4f}")
    print(f"• Primary Intent Delta (Δ):                {scores['delta_primary']:+.4f}")
    print(f"• Primary Interpretation:                  {scores['primary_metric_interpretation']}")
    print("=" * 70)

    print("\n>>> FRESH PROMPT END-TO-END PIPELINE VERIFICATION PASSED! <<<\n")
    return scores


if __name__ == "__main__":
    test_prompt = sys.argv[1] if len(sys.argv) > 1 else "driving funk bass groove"
    run_fresh_prompt_verification(test_prompt)

