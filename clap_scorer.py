"""
Unit 3: Contrastive Audio-Language Pretraining (CLAP) Scoring Engine
Model: laion/clap-htsat-unfused (configurable via CLAP_MODEL)
Resampling: Strictly 48 kHz audio requirement
Evaluation: Dual metrics (Original Raw Intent vs Generating Text) + Delta
"""

import os
from typing import Optional, Tuple, Dict, Union, Callable, Any

# Ensure OpenMP runtime conflict does not crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy import signal
import torch

# Global cached singletons for lazy loading
_CLAP_MODEL = None
_CLAP_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAP_TARGET_SR = 48000


def get_clap_model_id() -> str:
    """Return the configured CLAP model identifier."""
    return os.getenv("CLAP_MODEL", "laion/clap-htsat-unfused")


def load_clap(progress_callback: Optional[Callable[[float, str], None]] = None):
    """
    Lazy singleton loader for CLAP model and processor.
    Loads once on first use and caches in memory.
    """
    global _CLAP_MODEL, _CLAP_PROCESSOR, _DEVICE
    if _CLAP_MODEL is not None and _CLAP_PROCESSOR is not None:
        return _CLAP_MODEL, _CLAP_PROCESSOR, _DEVICE

    if progress_callback:
        progress_callback(0.1, "Loading CLAP processor and weights (laion/clap-htsat-unfused)...")

    from transformers import ClapModel, ClapProcessor

    model_id = get_clap_model_id()
    print(f"[CLAP] Initializing {model_id} on {_DEVICE}...")

    try:
        _CLAP_PROCESSOR = ClapProcessor.from_pretrained(model_id, local_files_only=True)
        _CLAP_MODEL = ClapModel.from_pretrained(model_id, local_files_only=True)
    except Exception:
        _CLAP_PROCESSOR = ClapProcessor.from_pretrained(model_id)
        _CLAP_MODEL = ClapModel.from_pretrained(model_id)
    _CLAP_MODEL.to(_DEVICE)
    _CLAP_MODEL.eval()

    print(f"[CLAP] Model successfully loaded on {_DEVICE}.")
    return _CLAP_MODEL, _CLAP_PROCESSOR, _DEVICE


def resample_to_48k(audio_np: np.ndarray, orig_sr: int, target_sr: int = CLAP_TARGET_SR) -> np.ndarray:
    """
    Resample 1D audio waveform to CLAP's required 48,000 Hz.
    Uses exact rational polynomial resampling (e.g. 32000 -> 48000 using factor 3/2).
    """
    target = target_sr if target_sr is not None else CLAP_TARGET_SR
    if orig_sr == target:
        return audio_np.astype(np.float32)

    from math import gcd
    d = gcd(orig_sr, target)
    up = target // d
    down = orig_sr // d

    resampled = signal.resample_poly(audio_np, up, down).astype(np.float32)
    return resampled


# Backward compatibility alias
resample_audio = resample_to_48k


def compute_cosine_similarity(v1: Any, v2: Any) -> float:
    """Compute cosine similarity between two 1D/2D vectors or tensors."""
    if hasattr(v1, "detach"):
        v1 = v1.detach().cpu().numpy()
    if hasattr(v2, "detach"):
        v2 = v2.detach().cpu().numpy()
    v1 = np.asarray(v1, dtype=np.float32).flatten()
    v2 = np.asarray(v2, dtype=np.float32).flatten()
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _extract_embedding_tensor(output: Union[torch.Tensor, object]) -> torch.Tensor:
    """Extract 2D embedding tensor [batch, dim] from model output."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "audio_embeds") and output.audio_embeds is not None:
        return output.audio_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    raise ValueError(f"Unable to extract embedding tensor from output of type {type(output)}")


def get_text_embedding(text: str, device: Optional[str] = None) -> torch.Tensor:
    """Compute L2-normalized text embedding from CLAP's RoBERTa text encoder."""
    model, processor, default_dev = load_clap()
    dev = device or default_dev

    inputs = processor(text=[text], padding=True, return_tensors="pt").to(dev)
    with torch.inference_mode():
        raw_feat = model.get_text_features(**inputs)
        embed = _extract_embedding_tensor(raw_feat)
        embed = embed / embed.norm(dim=-1, keepdim=True)
    return embed


def get_audio_embedding(audio_np: np.ndarray, orig_sr: int, device: Optional[str] = None) -> torch.Tensor:
    """
    Resample audio to 48 kHz and compute L2-normalized audio embedding from CLAP's HTSAT audio encoder.
    """
    model, processor, default_dev = load_clap()
    dev = device or default_dev

    audio_48k = resample_to_48k(audio_np, orig_sr)
    inputs = processor(audio=audio_48k, sampling_rate=CLAP_TARGET_SR, return_tensors="pt").to(dev)

    with torch.inference_mode():
        raw_feat = model.get_audio_features(**inputs)
        embed = _extract_embedding_tensor(raw_feat)
        embed = embed / embed.norm(dim=-1, keepdim=True)
    return embed


def compute_clap_similarity(
    audio_np: np.ndarray,
    orig_sr: int,
    text: str,
) -> float:
    """Compute cosine similarity between audio waveform and text query."""
    text_embed = get_text_embedding(text)
    audio_embed = get_audio_embedding(audio_np, orig_sr)
    cosine_sim = (text_embed @ audio_embed.T).item()
    return float(cosine_sim)


# Convenience aliases for pipeline compatibility
resample_audio = resample_to_48k
compute_cosine_similarity = compute_clap_similarity


def evaluate_dual_clip_scores(
    raw_audio: np.ndarray,
    exp_audio: Optional[np.ndarray] = None,
    sampling_rate: Optional[int] = None,
    raw_prompt: Optional[str] = None,
    expanded_prompt: Optional[str] = None,
    raw_sr: Optional[int] = None,
    expanded_audio: Optional[np.ndarray] = None,
    expanded_sr: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluates both clips against the dual-scoring framework.
    Accepts multiple parameter signatures for flexible invocation.
    """
    # Normalize argument names
    act_exp_audio = exp_audio if exp_audio is not None else expanded_audio
    r_sr = sampling_rate if sampling_rate is not None else raw_sr
    e_sr = sampling_rate if sampling_rate is not None else (expanded_sr if expanded_sr is not None else r_sr)

    # 1. Primary Metric: Both clips scored against raw prompt
    raw_clip_vs_raw_text = compute_clap_similarity(raw_audio, r_sr, raw_prompt)
    exp_clip_vs_raw_text = compute_clap_similarity(act_exp_audio, e_sr, raw_prompt)
    primary_delta = exp_clip_vs_raw_text - raw_clip_vs_raw_text

    # 2. Self-Reference Metric: Each clip scored against its own prompt
    raw_clip_vs_self = raw_clip_vs_raw_text
    exp_clip_vs_self = compute_clap_similarity(act_exp_audio, e_sr, expanded_prompt)
    self_delta = exp_clip_vs_self - raw_clip_vs_self

    interpretation = (
        "Expansion improved fidelity to user intent"
        if primary_delta > 0
        else "Expansion reduced fidelity to user intent (overconstrained / semantic drift)"
        if primary_delta < 0
        else "Neutral effect"
    )

    return {
        "raw_vs_raw": round(raw_clip_vs_raw_text, 4),
        "exp_vs_raw": round(exp_clip_vs_raw_text, 4),
        "exp_vs_exp": round(exp_clip_vs_self, 4),
        "raw_vs_exp": round(compute_clap_similarity(raw_audio, r_sr, expanded_prompt), 4),
        "delta_primary": round(primary_delta, 4),
        "delta_self": round(self_delta, 4),
        "primary_metric_interpretation": interpretation,
        "primary_metric": {
            "name": "Original Intent Fidelity (Fair & Unbiased)",
            "query_text": raw_prompt,
            "raw_clip_score": round(raw_clip_vs_raw_text, 4),
            "expanded_clip_score": round(exp_clip_vs_raw_text, 4),
            "delta": round(primary_delta, 4),
        },
        "self_metric": {
            "name": "Self-Reference Alignment (Biased toward Verbose Prompts)",
            "query_text": "Respective Prompt",
            "raw_clip_score": round(raw_clip_vs_self, 4),
            "expanded_clip_score": round(exp_clip_vs_self, 4),
            "delta": round(self_delta, 4),
        },
        "justification": (
            "The primary metric (vs Original Raw Prompt) is the authoritative evaluation metric: "
            "it measures whether prompt expansion generated music that better realizes what the user "
            "actually requested. Scoring the expanded clip against its own expanded text is biased "
            "because the expanded text contains additional descriptive tokens tailored to CLAP's vocabulary."
        ),
    }


score_clip_pair = evaluate_dual_clip_scores


# Convenient aliases
get_clap_model_and_processor = load_clap
resample_audio = resample_to_48k
