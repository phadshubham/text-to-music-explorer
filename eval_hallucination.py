"""
Responsible AI & Hallucination Benchmark
Evaluates 20 short prompts from data/hallucination_prompts.json on few-shot expansions:
1. Artist-Name Leakage Rate (via Guardrail & 300+ Artist Catalog).
2. Constraint-Retention Rate (automatic keyword verification).
3. Semantic Drift (all-MiniLM-L6-v2 cosine distance between raw and expanded).
Saves results to results/hallucination.json.
"""

import os
import sys
import json
from typing import Tuple, List, Dict
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts.templates import FEW_SHOT
from llm_client import expand_prompt
from guardrail import get_guardrail

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "data", "hallucination_prompts.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "hallucination.json")


def load_minilm_model():
    """Lazy loader for sentence-transformers/all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer
    print("[Embedding Model] Loading sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def check_constraint_retention(expanded_text: str, constraints: list) -> Tuple[bool, list]:
    """Check if all constraints are preserved in expanded text."""
    exp_lower = expanded_text.lower()
    missing = []
    synonyms = {
        "synth": ["synthesizer", "synths", "synthwave"],
        "synthesizer": ["synth", "synths", "synthwave"],
        "acoustic": ["acoustic", "wooden", "organic", "unplugged"],
        "melody": ["melody", "ballad", "tune", "phrase", "solo"],
        "island": ["island", "tropical", "hawaiian"],
        "rhythm": ["rhythm", "beat", "groove", "percussion"],
        "church": ["church", "cathedral", "pipe", "sacred"],
        "ambient": ["ambient", "reverb", "atmospheric", "space"],
    }

    for c in constraints:
        c_lower = c.lower()
        if c_lower in exp_lower:
            continue
        # Check synonyms
        matched = False
        if c_lower in synonyms:
            for syn in synonyms[c_lower]:
                if syn in exp_lower:
                    matched = True
                    break
        if not matched:
            missing.append(c)

    is_retained = (len(missing) == 0)
    return is_retained, missing


def run_hallucination_evaluation():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    assert os.path.exists(PROMPTS_FILE), f"Missing {PROMPTS_FILE}"

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    guardrail = get_guardrail()
    minilm = load_minilm_model()

    records = []
    leaked_artist_count = 0
    retained_constraints_count = 0
    cosine_similarities = []

    print("\n" + "=" * 70)
    print(f"RUNNING RESPONSIBLE-AI HALLUCINATION EVALUATION ON {len(cases)} PROMPTS")
    print("=" * 70)

    for i, item in enumerate(cases):
        raw_text = item["prompt"]
        constraints = item["constraints"]

        # Expand prompt via Few-Shot
        expanded_text, label, _ = expand_prompt(raw_text, technique=FEW_SHOT)

        # 1. Artist Leakage check
        guard_res = guardrail.check(expanded_text)
        artist_leaked = not guard_res["is_safe"]
        if artist_leaked:
            leaked_artist_count += 1

        # 2. Constraint Retention check
        is_retained, missing = check_constraint_retention(expanded_text, constraints)
        if is_retained:
            retained_constraints_count += 1

        # 3. Semantic Drift via MiniLM cosine similarity
        raw_emb = minilm.encode(raw_text)
        exp_emb = minilm.encode(expanded_text)
        cos_sim = float(np.dot(raw_emb, exp_emb) / (np.linalg.norm(raw_emb) * np.linalg.norm(exp_emb)))
        cosine_similarities.append(cos_sim)

        rec = {
            "id": item["id"],
            "raw_prompt": raw_text,
            "expanded_prompt": expanded_text,
            "expansion_status": label,
            "constraints_expected": constraints,
            "constraints_retained": is_retained,
            "missing_constraints": missing,
            "artist_leaked": artist_leaked,
            "detected_artist": guard_res["detected_artist"],
            "minilm_cosine_similarity": round(cos_sim, 4),
        }
        records.append(rec)

        print(
            f"[{i+1:02d}/{len(cases)}] '{raw_text}' -> Retained: {is_retained} | "
            f"Artist Leaked: {artist_leaked} | MiniLM Cosine: {cos_sim:.4f}"
        )

    # Aggregated metrics
    total_n = len(cases)
    artist_leakage_rate = (leaked_artist_count / total_n) * 100
    constraint_retention_rate = (retained_constraints_count / total_n) * 100
    mean_cosine_sim = float(np.mean(cosine_similarities))
    min_cosine_sim = float(np.min(cosine_similarities))
    max_cosine_sim = float(np.max(cosine_similarities))

    summary = {
        "metadata": {
            "num_test_prompts": total_n,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "expansion_technique": "Few-Shot In-Context Learning",
        },
        "metrics": {
            "artist_name_leakage_rate_pct": round(artist_leakage_rate, 2),
            "constraint_retention_rate_pct": round(constraint_retention_rate, 2),
            "mean_minilm_cosine_similarity": round(mean_cosine_sim, 4),
            "min_minilm_cosine_similarity": round(min_cosine_sim, 4),
            "max_minilm_cosine_similarity": round(max_cosine_sim, 4),
        },
        "records": records,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESPONSIBLE-AI BENCHMARK RESULTS:")
    print(f"• Total Prompts Tested:          {total_n}")
    print(f"• Artist Leakage Rate:           {artist_leakage_rate:.2f}% ({leaked_artist_count}/{total_n})")
    print(f"• Constraint-Retention Rate:     {constraint_retention_rate:.2f}% ({retained_constraints_count}/{total_n})")
    print(f"• Mean Semantic Cosine (MiniLM): {mean_cosine_sim:.4f}")
    print(f"• Min Semantic Cosine:           {min_cosine_sim:.4f}")
    print(f"• Saved detailed payload to:     {OUTPUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    from typing import Tuple
    run_hallucination_evaluation()
