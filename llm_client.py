"""
Unit 2: LLM Expansion Client
Provider-Agnostic: Groq (preferred), Gemini, Ollama (local offline), and Deterministic Fallback.
Configuration: Controlled via LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY, GEMINI_API_KEY.
Parameters: Temperature = 0.3, timeout = 15s, 1 retry.
Post-processing: Strip quotes, enforce max 40 words, outbound artist guardrail validation.
"""

import os
import re
import json
import time
from typing import Tuple, Optional, Dict, Any

from prompts.templates import format_expansion_prompt, FEW_SHOT, ZERO_SHOT
from guardrail import get_guardrail


# Deterministic rule-based expansion catalog for offline fallback
DETERMINISTIC_RULES = {
    "piano": "Acoustic grand piano ballad, gentle reverb, warm woody resonance, contemplative mood, 72 BPM, dynamic swell.",
    "electronic": "Pulsating analog synthesizer with driving 4/4 electronic kick, crisp hi-hats, energetic workout rhythm, 128 BPM.",
    "violin": "Expressive solo acoustic violin, melancholic vibrato, delicate room ambience, slow 60 BPM, emotional dynamic phrasing.",
    "dhol": "Energetic traditional Punjabi dhol percussion, festive wedding celebration, syncopated rhythm, bright accents, 105 BPM.",
    "guitar": "Warm acoustic fingerstyle guitar, gentle organic percussion, intimate atmospheric reverb, relaxing 80 BPM.",
    "ambient": "Lush atmospheric synth pads, expansive stereo reverberation, peaceful meditative tone, slow ethereal movement, 65 BPM.",
    "rock": "Driving electric guitar riffs, punchy drums, steady bassline, energetic tempo 120 BPM, raw organic production.",
    "lofi": "Warm vintage electric piano, mellow vinyl crackle, relaxed hip-hop beat, laid-back 75 BPM, cozy tape saturation.",
    "jazz": "Smooth muted trumpet, brushed jazz snare, upright double bass, smoky cafe atmosphere, swung rhythm, 85 BPM.",
}


def rule_based_fallback_expander(prompt: str) -> str:
    """
    Deterministic rule-based expander that guarantees an acoustically rich
    slot-compliant prompt without external LLM API dependencies.
    """
    prompt_lower = prompt.lower()
    matched_expansions = []

    for keyword, expansion in DETERMINISTIC_RULES.items():
        if keyword in prompt_lower:
            matched_expansions.append(expansion)

    if matched_expansions:
        chosen = matched_expansions[0]
        # Preserve user keywords if not already present
        tokens = prompt.strip().split()
        prefix = f"{prompt.title()}: "
        combined = f"{prefix}{chosen}"
        words = combined.split()
        if len(words) > 40:
            combined = " ".join(words[:40])
        return combined

    # Generic slot-based expansion preserving all input constraints
    return f"{prompt.title()} style instrumental, rich acoustic textures, subtle percussion, steady rhythm, 95 BPM, dynamic production."


def post_process_expansion(text: str) -> str:
    """
    Post-process LLM output:
    1. Strip surrounding quotes and conversational fluff.
    2. Extract only the first line.
    3. Truncate to strictly 40 words maximum.
    """
    # Remove leading/trailing quotes or markdown
    clean = text.strip().strip('"').strip("'").strip("`")

    # Take first line only
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    if lines:
        clean = lines[0]
    
    # Strip common meta prefixes if generated
    meta_prefixes = [
        "expanded prompt:", "musicgen prompt:", "expanded:", "output:", "here is the prompt:",
        "here's the prompt:", "sure!:", "sure,"
    ]
    for p in meta_prefixes:
        if clean.lower().startswith(p):
            clean = clean[len(p):].strip()

    # Re-strip quotes
    clean = clean.strip().strip('"').strip("'")

    # Enforce 40 words constraint
    words = clean.split()
    if len(words) > 40:
        clean = " ".join(words[:40])

    return clean


# Alias for backward compatibility with test suites
clean_and_trim_expansion = post_process_expansion


def call_groq_api(prompt_text: str, model_id: str, api_key: str) -> str:
    """Call Groq API using the official Groq client."""
    from groq import Groq
    client = Groq(api_key=api_key, timeout=15.0)

    # One retry handled via retry logic
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a specialized music prompt engineering assistant."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.3,
                max_tokens=100,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            if attempt == 1:
                raise e
            time.sleep(1.0)

    raise RuntimeError("Groq API returned empty response after retry.")


def call_gemini_api(prompt_text: str, model_id: str, api_key: str) -> str:
    """Call Google Gemini API using google-genai or google-generativeai."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=prompt_text,
            config={"temperature": 0.3, "max_output_tokens": 100},
        )
        return response.text.strip()
    except Exception:
        # Fallback to google.generativeai if installed
        import google.generativeai as gai
        gai.configure(api_key=api_key)
        model = gai.GenerativeModel(model_id)
        response = model.generate_content(
            prompt_text,
            generation_config={"temperature": 0.3, "max_output_tokens": 100},
        )
        return response.text.strip()


def call_ollama_api(prompt_text: str, model_id: str) -> str:
    """Call local Ollama instance via HTTP API."""
    import requests
    url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 80,
        },
    }
    resp = requests.post(url, json=payload, timeout=25.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def expand_prompt(
    user_prompt: str,
    technique: str = FEW_SHOT,
) -> Tuple[str, str, str]:
    """
    Expand user prompt into slot-based MusicGen prompt.

    Args:
        user_prompt: Original short prompt from user.
        technique: ZERO_SHOT or FEW_SHOT.

    Returns:
        Tuple of (expanded_prompt, provider_status_label, raw_template_used)
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    formatted_prompt = format_expansion_prompt(user_prompt, technique=technique)

    raw_output = None
    status_label = ""

    # 0. Qwen GPT-style decoder-only provider (Groq or local Ollama)
    if provider == "qwen":
        model_id = os.getenv("LLM_MODEL", "qwen-2.5-32b")
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                raw_output = call_groq_api(formatted_prompt, model_id=model_id, api_key=groq_key)
                status_label = f"Qwen ({model_id})"
            except Exception as e:
                print(f"[LLM] Qwen via Groq failed ({e}). Falling back to deterministic expander.")
        else:
            try:
                raw_output = call_ollama_api(formatted_prompt, model_id=os.getenv("LLM_MODEL", "qwen2.5:7b"))
                status_label = "Qwen (qwen2.5 local)"
            except Exception as e:
                print(f"[LLM] Qwen local failed ({e}). Falling back to deterministic expander.")

    # 1. Ollama local provider (zero external API keys)
    elif provider == "ollama":
        model_id = os.getenv("LLM_MODEL", "llama3.2:3b")
        try:
            raw_output = call_ollama_api(formatted_prompt, model_id=model_id)
            status_label = f"Ollama ({model_id})"
        except Exception as e:
            print(f"[LLM] Ollama call failed ({e}). Falling back to deterministic expander.")

    # 2. Groq provider
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model_id = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        if api_key:
            try:
                raw_output = call_groq_api(formatted_prompt, model_id=model_id, api_key=api_key)
                status_label = f"Groq ({model_id})"
            except Exception as e:
                print(f"[LLM] Groq call failed ({e}). Falling back to deterministic expander.")
        else:
            print("[LLM] GROQ_API_KEY not set. Using deterministic fallback.")

    # 3. Gemini provider
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_id = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        if api_key:
            try:
                raw_output = call_gemini_api(formatted_prompt, model_id=model_id, api_key=api_key)
                status_label = f"Gemini ({model_id})"
            except Exception as e:
                print(f"[LLM] Gemini call failed ({e}). Falling back to deterministic expander.")
        else:
            print("[LLM] GEMINI_API_KEY not set. Using deterministic fallback.")

    # 4. Fallback execution
    if not raw_output:
        fallback_text = rule_based_fallback_expander(user_prompt)
        status_label = "FALLBACK (no LLM)"
        clean_expansion = post_process_expansion(fallback_text)
        return clean_expansion, status_label, formatted_prompt

    clean_expansion = post_process_expansion(raw_output)

    # Re-run guardrail on outbound LLM output to prevent artist-name leakage
    guardrail = get_guardrail()
    outbound_check = guardrail.check(clean_expansion)
    if not outbound_check["is_safe"]:
        print(f"[LLM Guardrail Warning] Outbound expansion leaked artist '{outbound_check['detected_artist']}'. Falling back.")
        fallback_text = rule_based_fallback_expander(user_prompt)
        status_label = "FALLBACK (Leaked Artist Filtered)"
        clean_expansion = post_process_expansion(fallback_text)

    return clean_expansion, status_label, formatted_prompt
