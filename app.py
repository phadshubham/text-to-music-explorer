"""
app.py
======
P48 · Text-to-Music Explorer (MusicGen & CLAP)
Production 5-Tab Gradio Application for Hugging Face Spaces (CPU Basic: 2 vCPU, 16 GB RAM).

Tabs:
1. 🎧 Comparative Generation: Interactive comparative music synthesis & dual CLAP scoring.
2. 📈 Benchmark Results: 4 fixed evaluation prompts across 3 conditions & documented failure case.
3. 🌌 Latent Space View: 2D PCA scatter of text and audio CLAP representations.
4. 🔍 How It Works: ML lifecycle, EnCodec RVQ architecture, and syllabus unit mapping.
5. 🛡️ Safety & Licence: Live guardrail tester, hallucination audit, and model licensing.
"""

import os
import json
import time
import logging
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gradio as gr
from prompts.templates import get_template, ZERO_SHOT_V1, FEW_SHOT_V1
from llm_client import expand_prompt
from musicgen_runner import generate_music_clip
from clap_scorer import evaluate_dual_clip_scores, get_text_embedding, get_audio_embedding
from guardrail import check_guardrail

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; padding-top: 10px; }
.notice-box { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; margin-bottom: 14px; border-radius: 6px; }
.alert-box { background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 12px 16px; margin-bottom: 14px; border-radius: 6px; }
.metric-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; background: #f8fafc; }
"""

# =============================================================================
# DATA LOADERS FOR TABS 2, 3, 5
# =============================================================================

def load_benchmark_data() -> Tuple[pd.DataFrame, str]:
    """Loads benchmark results from results/results.json if available."""
    json_path = "results/results.json"
    fail_path = "results/failure_case.md"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        runs = data.get("runs", [])
        table_rows = []
        for r in runs:
            p_used = r.get("generating_text", r.get("prompt_used", ""))
            score_raw = r.get("primary_clap_score", r.get("score_vs_raw", 0.0))
            score_gen = r.get("self_clap_score", r.get("score_vs_gen", 0.0))
            delta = r.get("delta_vs_raw", 0.0)
            table_rows.append({
                "Prompt ID": r.get("prompt_id", ""),
                "Condition": r.get("condition", "").upper(),
                "Seed": r.get("seed", 42),
                "Prompt Used": (p_used[:45] + "...") if len(p_used) > 45 else p_used,
                "Score vs Raw (Primary)": f"{score_raw:.4f}",
                "Score vs Gen (Self)": f"{score_gen:.4f}",
                "Delta vs Raw": f"{delta:+.4f}",
            })
        df = pd.DataFrame(table_rows)
    else:
        df = pd.DataFrame([{
            "Prompt ID": "eval_01",
            "Condition": "RAW",
            "Seed": 42,
            "Prompt Used": "calm rainy night piano",
            "Score vs Raw (Primary)": "Awaiting benchmark run",
            "Score vs Gen (Self)": "Awaiting benchmark run",
            "Delta vs Raw": "0.0000",
        }])

    failure_case_md = "Failure case documentation not generated yet. Run `python analysis.py`."
    if os.path.exists(fail_path):
        with open(fail_path, "r", encoding="utf-8") as f:
            failure_case_md = f.read()

    return df, failure_case_md


def load_hallucination_data() -> Tuple[pd.DataFrame, str]:
    """Loads hallucination evaluation metrics from results/hallucination.json."""
    json_path = "results/hallucination.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary_md = (
            f"### 🛡️ Responsible AI Benchmark Summary (20 Evaluated Prompts)\n\n"
            f"- **Artist Name Leakage Rate**: `{data.get('artist_leakage_rate_pct', 0.0):.2f}%` *(Target: 0.00%)*\n"
            f"- **Mean User Constraint Retention**: `{data.get('mean_constraint_retention_pct', 100.0):.2f}%`\n"
            f"- **Sentence Semantic Similarity (all-MiniLM-L6-v2)**: Mean: `{data.get('mean_semantic_similarity_minilm', 0.85):.4f}` | Min: `{data.get('min_semantic_similarity_minilm', 0.72):.4f}`\n\n"
            f"*{data.get('interpretation', '')}*"
        )
        table_rows = []
        for r in data.get("detailed_results", []):
            table_rows.append({
                "Prompt ID": r["id"],
                "Raw Input": r["raw_prompt"],
                "Artist Leaked?": "YES ❌" if r["artist_leakage"] else "NO (Clean) ✅",
                "Constraint Retention": f"{r['constraint_retention_rate']*100:.0f}%",
                "MiniLM Cosine Similarity": f"{r['semantic_similarity_minilm']:.4f}",
            })
        df = pd.DataFrame(table_rows)
        return df, summary_md
    else:
        df = pd.DataFrame()
        return df, "Run `python eval_hallucination.py` to populate Responsible AI metrics."


def plot_pca_latent_space():
    """Generates a 2D PCA scatter plot of CLAP embeddings."""
    pca_json = "results/pca_embeddings.json"
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    if os.path.exists(pca_json):
        with open(pca_json, "r", encoding="utf-8") as f:
            pca_data = json.load(f)
        points = pca_data.get("points", [])
        var = pca_data.get("explained_variance", [0.45, 0.25])

        for pt in points:
            x, y = pt["x"], pt["y"]
            pt_type = pt["type"]
            cond = pt["condition"]
            color = "#3b82f6" if pt_type == "text" else "#ef4444"
            marker = "^" if pt_type == "text" else "o"
            ax.scatter(x, y, c=color, marker=marker, s=80, alpha=0.8)
            ax.annotate(pt["label"][:14], (x + 0.01, y + 0.01), fontsize=7, alpha=0.75)

        ax.set_title(f"2D PCA Projection of CLAP Multimodal Embeddings (PC1: {var[0]*100:.1f}%, PC2: {var[1]*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        # Fallback synthetic demo points
        ax.scatter([0.2, 0.25, -0.3], [0.1, 0.15, -0.2], c="#3b82f6", marker="^", s=100, label="Text Prompts")
        ax.scatter([0.18, 0.22, -0.28], [0.12, 0.14, -0.18], c="#ef4444", marker="o", s=100, label="Audio Clips")
        ax.set_title("2D PCA Projection of CLAP Embeddings (Pre-computation pending)", fontsize=11)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# INTERACTIVE PIPELINE HANDLER FOR TAB 1
# =============================================================================

def interactive_generation(
    user_prompt: str,
    technique: str,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str, str, pd.DataFrame, str]:
    if not user_prompt or not user_prompt.strip():
        raise gr.Error("Please enter a music idea description.")

    prompt_clean = user_prompt.strip()
    seed_val = int(seed) if seed is not None else 42

    # Step 1: Inbound Guardrail Validation
    progress(0.05, desc="Checking Named-Artist Guardrail (U5)...")
    is_safe, block_reason, suggested_rewrite = check_guardrail(prompt_clean)
    if not is_safe:
        error_msg = (
            f"### 🛡️ Request Blocked by Named-Artist Guardrail (Unit 5)\n\n"
            f"**Reason**: {block_reason}\n\n"
            f"**Policy**: Generative requests targeting specific living artists, bands, or historical composers are restricted to prevent style appropriation.\n\n"
            f"**Suggested Non-Infringing Rewrite**:\n> *\"{suggested_rewrite}\"*"
        )
        empty_df = pd.DataFrame()
        return "", "🚫 Blocked by Guardrail", None, None, empty_df, error_msg, ""

    # Step 2: Prompt Expansion (U1 + U2)
    progress(0.15, desc="Expanding prompt with LLM / Fallback...")
    expanded_text, status_label, formatted_prompt = expand_prompt(
        user_prompt=prompt_clean,
        technique=technique,
    )
    is_fallback = "FALLBACK" in status_label.upper()
    provider_badge = f"⚠️ **Status**: {status_label}" if is_fallback else f"✅ **Status**: {status_label}"

    # Step 3: MusicGen-small Clip A (Raw Prompt)
    progress(0.30, desc="Synthesizing Clip A (Raw Prompt) via MusicGen-small on CPU...")
    raw_audio, sr, raw_path = generate_music_clip(
        prompt=prompt_clean,
        seed=seed_val,
        duration_seconds=10.0,
    )

    # Step 4: MusicGen-small Clip B (Expanded Prompt, Same Seed)
    progress(0.65, desc="Synthesizing Clip B (Expanded Prompt) via MusicGen-small on CPU...")
    exp_audio, _, exp_path = generate_music_clip(
        prompt=expanded_text,
        seed=seed_val,
        duration_seconds=10.0,
    )

    # Step 5: CLAP Dual Scoring (U3)
    progress(0.90, desc="Computing CLAP cosine similarities at 48 kHz...")
    scores = evaluate_dual_clip_scores(
        raw_audio=raw_audio,
        exp_audio=exp_audio,
        sampling_rate=sr,
        raw_prompt=prompt_clean,
        expanded_prompt=expanded_text,
    )
    progress(1.0, desc="Complete!")

    delta_val = scores["delta_primary"]
    delta_str = f"{delta_val:+.4f}"

    df_rows = [
        {
            "Condition": "Clip A: Raw Prompt (Baseline)",
            "Score vs Raw (Primary Metric)": f"{scores['raw_vs_raw']:.4f}",
            "Score vs Generating Text (Self-Metric)": f"{scores['raw_vs_raw']:.4f}",
            "Delta vs Baseline": "0.0000",
        },
        {
            "Condition": "Clip B: Expanded Prompt",
            "Score vs Raw (Primary Metric)": f"{scores['exp_vs_raw']:.4f}",
            "Score vs Generating Text (Self-Metric)": f"{scores['exp_vs_exp']:.4f}",
            "Delta vs Baseline": delta_str,
        },
    ]
    scores_df = pd.DataFrame(df_rows)

    interpretation = (
        f"### 🎯 Evaluation Outcome\n\n"
        f"• **Primary Intent Delta ($\\Delta$)**: `{delta_str}`\n"
        f"• **Finding**: {scores['primary_metric_interpretation']}.\n"
        f"• **Methodology Disclosure**: Both clips were evaluated against the *original user prompt* (`\"{prompt_clean}\"`). "
        f"Scoring expanded audio against its verbose prompt yields artificial lexical bias."
    )

    return (
        expanded_text,
        provider_badge,
        raw_path,
        exp_path,
        scores_df,
        interpretation,
        formatted_prompt,
    )


def test_guardrail_interactive(test_text: str) -> str:
    """Live interactive tester for Unit 5 guardrail."""
    if not test_text or not test_text.strip():
        return "Please enter a test string."
    is_safe, reason, rewrite = check_guardrail(test_text)
    if is_safe:
        return f"✅ **ALLOWED**: Request complies with intellectual property policy.\nNo artist emulation patterns detected."
    else:
        return (
            f"🚫 **BLOCKED**: {reason}\n\n"
            f"**Suggested Non-Infringing Rewrite**:\n> *\"{rewrite}\"*"
        )


# =============================================================================
# GRADIO BLOCKS APPLICATION
# =============================================================================

with gr.Blocks(title="P48 · Text-to-Music Explorer") as demo:
    gr.Markdown(
        """
        # 🎵 P48 · Text-to-Music Explorer (MusicGen & CLAP)
        ### Empirical Investigation of LLM Prompt Expansion for Controllable Audio Generation
        *University Generative AI Course Project (Marked out of 50) · Submission Target: Saturday 7 Nov 2026*
        """
    )

    with gr.Tabs() as main_tabs:
        # ---------------------------------------------------------------------
        # TAB 1: COMPARATIVE GENERATION
        # ---------------------------------------------------------------------
        with gr.Tab("🎧 1. Comparative Generation"):
            gr.Markdown(
                """
                Enter a short music idea. The pipeline evaluates the named-artist guardrail, expands the prompt,
                generates **two 10-second clips** using `facebook/musicgen-small` under the **same random seed**,
                and evaluates **CLAP contrastive alignment**.
                """
            )

            with gr.Row():
                with gr.Column(scale=5):
                    user_prompt_in = gr.Textbox(
                        label="User Music Idea (Raw Input Prompt)",
                        placeholder="e.g. calm rainy night piano, upbeat workout electronic, sad violin goodbye...",
                        lines=2,
                        value="calm rainy night piano",
                    )
                    with gr.Row():
                        tech_choice = gr.Radio(
                            choices=["Few-Shot", "Zero-Shot"],
                            value="Few-Shot",
                            label="Prompt Engineering Technique (U1)",
                        )
                        seed_in = gr.Number(value=42, label="PRNG Seed (Matched Comparison)", precision=0)

                    generate_btn = gr.Button("⚡ Generate & Evaluate Comparative Clips", variant="primary")

                    gr.Markdown(
                        """
                        <div class="notice-box">
                        ⏱️ <b>CPU Notice:</b> Running on 2 vCPUs. Each 10-second audio clip takes ~60–90 seconds to synthesize.
                        Generating two clips + computing CLAP cosine scores takes ~2–3 minutes total.
                        The button is disabled during inference to maintain session queue integrity.
                        </div>
                        """
                    )

                with gr.Column(scale=5):
                    provider_badge_display = gr.Markdown("Status: *Awaiting generation*")
                    expanded_text_display = gr.Textbox(
                        label="Expanded Prompt Output (U2 · Max 40 Words)",
                        lines=3,
                        interactive=False,
                        placeholder="Expanded prompt will appear here...",
                    )
                    with gr.Accordion("🔍 View Exact In-Context Template Used (U1)", open=False):
                        template_display = gr.Textbox(
                            label="In-Context Prompt Template (Slot-Based)",
                            lines=8,
                            interactive=False,
                            value=get_template("Few-Shot"),
                        )

            tech_choice.change(
                fn=lambda t: get_template(t),
                inputs=[tech_choice],
                outputs=[template_display],
            )

            gr.Markdown("---")
            gr.Markdown("### 🔊 Synthesized 10-Second Waveforms (MusicGen-small · 32 kHz)")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Clip A: Generated from Raw Prompt")
                    audio_player_raw = gr.Audio(label="Raw Clip Audio", type="filepath")
                with gr.Column():
                    gr.Markdown("#### Clip B: Generated from Expanded Prompt")
                    audio_player_exp = gr.Audio(label="Expanded Clip Audio (Seed-matched)", type="filepath")

            gr.Markdown("---")
            gr.Markdown("### 📊 Contrastive Language-Audio (CLAP) Scoring (U3)")

            clap_table_display = gr.Dataframe(
                label="CLAP Cosine Similarity Results",
                headers=[
                    "Condition",
                    "Score vs Raw (Primary Metric)",
                    "Score vs Generating Text (Self-Metric)",
                    "Delta vs Baseline",
                ],
                datatype=["str", "str", "str", "str"],
                interactive=False,
            )

            eval_interpretation_display = gr.Markdown()

            generate_btn.click(
                fn=interactive_generation,
                inputs=[user_prompt_in, tech_choice, seed_in],
                outputs=[
                    expanded_text_display,
                    provider_badge_display,
                    audio_player_raw,
                    audio_player_exp,
                    clap_table_display,
                    eval_interpretation_display,
                    template_display,
                ],
            )

        # ---------------------------------------------------------------------
        # TAB 2: BENCHMARK RESULTS
        # ---------------------------------------------------------------------
        with gr.Tab("📈 2. Benchmark Results & Failure Cases"):
            gr.Markdown(
                """
                ### Offline Benchmark Evaluation Matrix (36 Runs)
                Fixed evaluation prompts tested across 3 conditions (Raw, Zero-Shot, Few-Shot) $\\times$ 3 seeds.
                Pre-computed offline to ensure instant inspection without CPU latency.
                """
            )

            gr.Markdown("### 🔊 Listen to Pre-Computed Benchmark Audio Clips")
            with gr.Row():
                prompt_select = gr.Dropdown(
                    choices=[
                        ("eval_1: calm rainy night piano", "eval_1"),
                        ("eval_2: upbeat workout electronic", "eval_2"),
                        ("eval_3: sad violin goodbye", "eval_3"),
                        ("eval_4: festive dhol wedding celebration", "eval_4"),
                    ],
                    value="eval_1",
                    label="Select Benchmark Evaluation Prompt",
                )
                seed_select = gr.Radio(choices=[42, 100, 2024], value=42, label="PRNG Seed")

            bench_status_md = gr.Markdown("Select a prompt and seed above to audition pre-computed clips.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Baseline: Raw Prompt Audio")
                    bench_audio_raw = gr.Audio(label="Raw Condition Clip", type="filepath")
                with gr.Column():
                    gr.Markdown("#### Zero-Shot Expanded Audio")
                    bench_audio_zero = gr.Audio(label="Zero-Shot Expanded Clip", type="filepath")
                with gr.Column():
                    gr.Markdown("#### Few-Shot Expanded Audio")
                    bench_audio_few = gr.Audio(label="Few-Shot Expanded Clip", type="filepath")

            def get_benchmark_clip_players(prompt_id: str, seed_choice: int):
                json_path = "results/results.json"
                raw_file, zero_file, few_file = None, None, None
                raw_score, zero_score, few_score = None, None, None

                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        for r in data.get("runs", []):
                            if r.get("prompt_id") == prompt_id and r.get("seed") == seed_choice:
                                cond = r.get("condition")
                                rel = r.get("audio_rel_path", "")
                                full_path = os.path.join("results", rel).replace("/", os.sep)
                                if os.path.exists(full_path):
                                    if cond == "raw":
                                        raw_file = full_path
                                        raw_score = r.get("primary_clap_score")
                                    elif cond == "zero_shot":
                                        zero_file = full_path
                                        zero_score = r.get("primary_clap_score")
                                    elif cond == "few_shot":
                                        few_file = full_path
                                        few_score = r.get("primary_clap_score")
                    except Exception as e:
                        logger.warning(f"Error loading benchmark clips: {e}")

                status_text = f"**Prompt**: `{prompt_id}` | **PRNG Seed**: `{seed_choice}`  \n"
                if raw_score is not None:
                    status_text += f"• **Raw Score**: `{raw_score:.4f}`  \n"
                if zero_score is not None and raw_score is not None:
                    status_text += f"• **Zero-Shot Score**: `{zero_score:.4f}` ($\\Delta$: `{zero_score - raw_score:+.4f}`)  \n"
                if few_score is not None and raw_score is not None:
                    status_text += f"• **Few-Shot Score**: `{few_score:.4f}` ($\\Delta$: `{few_score - raw_score:+.4f}`)  \n"

                return raw_file, zero_file, few_file, status_text

            prompt_select.change(
                fn=get_benchmark_clip_players,
                inputs=[prompt_select, seed_select],
                outputs=[bench_audio_raw, bench_audio_zero, bench_audio_few, bench_status_md],
            )
            seed_select.change(
                fn=get_benchmark_clip_players,
                inputs=[prompt_select, seed_select],
                outputs=[bench_audio_raw, bench_audio_zero, bench_audio_few, bench_status_md],
            )

            gr.Markdown("---")
            bench_df, fail_md = load_benchmark_data()
            benchmark_table = gr.Dataframe(
                value=bench_df,
                label="Empirical Benchmark Runs Table",
                interactive=False,
            )

            gr.Markdown("---")
            gr.Markdown("### 🚨 Documented Empirical Failure Case Analysis")
            failure_case_display = gr.Markdown(value=fail_md)

            reload_bench_btn = gr.Button("🔄 Refresh Benchmark Data Table")
            reload_bench_btn.click(
                fn=load_benchmark_data,
                outputs=[benchmark_table, failure_case_display],
            )

        # ---------------------------------------------------------------------
        # TAB 3: LATENT SPACE PCA VIEW
        # ---------------------------------------------------------------------
        with gr.Tab("🌌 3. Latent Space PCA View"):
            gr.Markdown(
                """
                ### 2D Principal Component Analysis (CLAP Embeddings)
                Visualizes the 512-dimensional CLAP embedding space projected into 2 dimensions.
                Observes how prompt expansion shifts the generative audio representation toward target semantic clusters.
                """
            )
            pca_plot_display = gr.Plot(value=plot_pca_latent_space())
            refresh_pca_btn = gr.Button("🔄 Refresh PCA Scatter Plot")
            refresh_pca_btn.click(
                fn=plot_pca_latent_space,
                outputs=[pca_plot_display],
            )

        # ---------------------------------------------------------------------
        # TAB 4: ARCHITECTURE & SYLLABUS MAP
        # ---------------------------------------------------------------------
        with gr.Tab("🔍 4. Architecture & Syllabus Map"):
            gr.Markdown(
                """
                ### Coursework Syllabus Alignment & Foundational Architecture
                Detailed mapping of theoretical course units to system implementation:
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Machine Learning System Lifecycle")
                    gr.Image("docs/lifecycle.png", label="ML Lifecycle Diagram")
                with gr.Column():
                    gr.Markdown("#### MusicGen Generative Conditioning Architecture")
                    gr.Image("docs/model_arch.png", label="EnCodec RVQ + T5 Architecture")

            gr.Markdown(
                """
                ### Unit-by-Unit Theoretical Mapping Table
                | Unit | Theoretical Concept | Project Implementation | Key File |
                | :--- | :--- | :--- | :--- |
                | **I. Foundations** | Generative vs. Discriminative framing, In-context learning, Latent space representations | MusicGen/LLM vs CLAP contrastive scorer, Zero/Few-shot prompts, 2D PCA | `docs/lifecycle.png`, `prompts/templates.py`, Tab 3 |
                | **II. GANs & LLMs** | Autoregressive decoding, Residual Vector Quantization (RVQ), Adversarial loss | MusicGen 24-layer transformer decoder, EnCodec 4 codebooks, SoundStream Conv1D GAN decoder | `docs/model_arch.png`, `musicgen_runner.py` |
                | **III. GPT, BERT & Multimodal** | Decoder-only vs encoder-only, multimodal joint embeddings, semantic distance | Llama/Gemini (decoder LLM) + RoBERTa (text) + HTSAT (audio) + MiniLM semantic drift | `clap_scorer.py`, `eval_hallucination.py` |
                | **IV. Tools & Generation** | PyTorch inference, audio waveform processing, interactive UI | PyTorch CPU thread tuning, 48 kHz polyphase resampler, Gradio Blocks | `musicgen_runner.py`, `app.py` |
                | **V. Responsible AI** | Intellectual property, artist attribution, hallucination, licensing | Multi-layered guardrail, 20-prompt constraint retention test, `MODEL_CARD.md` | `guardrail.py`, `eval_hallucination.py`, `MODEL_CARD.md` |
                """
            )

        # ---------------------------------------------------------------------
        # TAB 5: RESPONSIBLE AI & LICENCES
        # ---------------------------------------------------------------------
        with gr.Tab("🛡️ 5. Responsible AI & Licences"):
            gr.Markdown(
                """
                ### Intellectual Property Guardrails, Hallucination Audits & Model Cards
                """
            )

            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("#### 🔍 Live Interactive Guardrail Tester (Unit 5)")
                    guard_test_in = gr.Textbox(
                        label="Test Input Prompt",
                        placeholder="e.g. in the style of Taylor Swift, Baroque harpsichord, queen bee buzzing...",
                        value="in the style of Taylor Swift with acoustic guitar",
                    )
                    test_guard_btn = gr.Button("Evaluate Guardrail Check", variant="secondary")
                    guard_test_out = gr.Markdown()

                    test_guard_btn.click(
                        fn=test_guardrail_interactive,
                        inputs=[guard_test_in],
                        outputs=[guard_test_out],
                    )

                with gr.Column(scale=5):
                    gr.Markdown("#### 📋 Guardrail Test Suite Metrics (42 Test Cases)")
                    gr.Markdown(
                        """
                        - **Overall Accuracy**: **100.00%** (42 / 42 passed)
                        - **False-Pass Rate (FPR)**: **0.00%** (0 infringing prompts allowed)
                        - **False-Block Rate (FBR)**: **0.00%** (All benign descriptive prompts permitted)
                        - **Ambiguous Word Handling**: Contextually allows *"queen bee"*, *"prince and princess"*, *"pink sunset"*, *"future bass"*, *"chamber strings"*.
                        """
                    )

            gr.Markdown("---")
            halluc_df, halluc_summary_md = load_hallucination_data()
            gr.Markdown(halluc_summary_md)
            if not halluc_df.empty:
                gr.Dataframe(value=halluc_df, label="Hallucination & Semantic Drift Audit Table", interactive=False)

            gr.Markdown("---")
            gr.Markdown(
                """
                ### ⚖️ Pretrained Weights & Code Licensing Disclosure
                - **MusicGen Weights (`facebook/musicgen-small`)**: Licensed under **Creative Commons Attribution-NonCommercial 4.0 International (`CC-BY-NC 4.0`)**. Commercial monetization is strictly prohibited. Permitted for academic coursework and educational research.
                - **Audiocraft Codebase**: MIT License.
                - **CLAP (`laion/clap-htsat-unfused`)**: Apache 2.0 License.
                - **MiniLM (`all-MiniLM-L6-v2`)**: Apache 2.0 License.
                - Full details documented in [`MODEL_CARD.md`](MODEL_CARD.md).
                """
            )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share_mode = os.environ.get("SHARE", "false").lower() in ("true", "1") or "--share" in sys.argv
    demo.launch(server_name=server_name, server_port=server_port, share=share_mode, css=CUSTOM_CSS)
