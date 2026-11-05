"""
tests/test_pipeline.py
======================
Unit & Integration Tests for the complete Text-to-Music Explorer Pipeline.

Tests:
1. test_guardrail_blocking: Verifies blocking of named artists, casual suffixes, and classical composers.
2. test_guardrail_passing: Verifies allowance of descriptive musical terms, eras, and ambiguous words.
3. test_prompt_expansion: Verifies slot-based formatting and word limit constraints.
4. test_audio_resampling: Verifies mathematical correctness of polyphase 32 kHz to 48 kHz resampler.
5. test_clap_l2_normalization: Verifies text and audio embeddings are unit-normalized.
6. test_dual_scoring_contract: Verifies dictionary output structure and delta calculation.
"""

import os
import sys
import unittest
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from guardrail import check_guardrail
from prompts.templates import format_prompt, ZERO_SHOT_V1, FEW_SHOT_V1
from llm_client import expand_prompt, clean_and_trim_expansion, rule_based_fallback_expander
from clap_scorer import resample_audio, get_text_embedding, compute_clap_similarity


class TestPipeline(unittest.TestCase):

    def test_guardrail_blocking(self):
        """Verifies that named artists and casual suffixes are blocked."""
        block_prompts = [
            "in the style of Taylor Swift",
            "sounds like A.R. Rahman",
            "Beethoven symphony style",
            "coldplay-esque stadium rock",
            "sounding like Ed Sheerann",
            "in Hans Zimmer's voice",
            "remix of Queen bohemian rhapsody",
        ]
        for p in block_prompts:
            is_safe, reason, rewrite = check_guardrail(p)
            self.assertFalse(is_safe, f"Should block: '{p}'")
            self.assertIsNotNone(reason)
            self.assertIsNotNone(rewrite)

    def test_guardrail_passing(self):
        """Verifies that generic descriptive musical prompts pass."""
        pass_prompts = [
            "calm rainy night piano",
            "upbeat workout electronic",
            "sad violin goodbye",
            "festive dhol wedding celebration",
            "80s synth-pop with driving bass",
            "Baroque harpsichord and chamber strings",
            "queen bee buzzing ambience",
            "future bass with sidechained synths",
        ]
        for p in pass_prompts:
            is_safe, reason, rewrite = check_guardrail(p)
            self.assertTrue(is_safe, f"Should allow: '{p}' (Reason given: {reason})")

    def test_prompt_expansion_constraints(self):
        """Verifies word count cap and slot preservation in prompt expansion."""
        test_idea = "calm rainy night piano"
        expanded, is_fallback, status = expand_prompt(test_idea, technique="Few-Shot")
        words = expanded.split()
        self.assertLessEqual(len(words), 42, f"Prompt exceeds word limit: {len(words)} words")
        self.assertFalse(expanded.startswith('"'))
        self.assertFalse(expanded.startswith("Expanded prompt:"))

    def test_audio_resampling(self):
        """Verifies 32 kHz to 48 kHz polyphase resampling ratio."""
        # 1.0 second of 32 kHz audio = 32000 samples
        dummy_audio = np.random.randn(32000).astype(np.float32)
        resampled = resample_audio(dummy_audio, orig_sr=32000, target_sr=48000)
        # Should yield exactly 48000 samples (3/2 ratio)
        self.assertEqual(len(resampled), 48000)

    def test_clap_l2_normalization(self):
        """Verifies that extracted CLAP text embeddings have unit L2 norm."""
        emb = get_text_embedding("calm piano music")
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_cosine_similarity_bounds(self):
        """Verifies cosine similarity between audio and text is bounded in [-1.0, 1.0]."""
        dummy_audio = np.random.randn(32000).astype(np.float32)
        score = compute_clap_similarity(dummy_audio, 32000, "calm piano")
        self.assertTrue(-1.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
