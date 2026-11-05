"""
scripts/generate_diagrams.py
============================
Generates high-resolution academic diagrams for P48 Text-to-Music Explorer:
1. docs/lifecycle.png: End-to-end Machine Learning System Lifecycle.
2. docs/model_arch.png: MusicGen EnCodec RVQ & T5 Cross-Attention Architecture.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_lifecycle_diagram(output_path: str = "docs/lifecycle.png"):
    """
    Generates the ML System Lifecycle diagram (Syllabus Unit I).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(6, 7.5, "P48 · End-to-End Generative AI System Lifecycle", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#1a2530")
    ax.text(6, 7.1, "User Intent → Guardrail → In-Context Expansion → MusicGen RVQ Synthesis → Contrastive CLAP Scoring",
            ha="center", va="center", fontsize=10, color="#555555")

    # Step boxes
    boxes = [
        {"x": 0.5, "y": 4.5, "w": 2.2, "h": 1.8, "title": "1. User Input\n& Guardrail (U5)",
         "desc": "• Raw text prompt\n• RapidFuzz fuzzy filter\n• 300+ artist database\n• Rewrite recommendation", "color": "#e8f4f8", "border": "#2b5c8f"},
        {"x": 3.2, "y": 4.5, "w": 2.4, "h": 1.8, "title": "2. Prompt Expansion\nEngine (U1 & U2)",
         "desc": "• Slot-based template\n• Zero-Shot vs Few-Shot\n• Groq/Gemini LLM (T=0.3)\n• Deterministic fallback", "color": "#eef8ee", "border": "#2e7d32"},
        {"x": 6.1, "y": 4.5, "w": 2.6, "h": 1.8, "title": "3. Audio Synthesis\nMusicGen-small (U4)",
         "desc": "• T5 text conditioner\n• Transformer decoder\n• 500 tokens (10.0s @ 50 Hz)\n• EnCodec RVQ + GAN", "color": "#fff8e1", "border": "#f57f17"},
        {"x": 9.2, "y": 4.5, "w": 2.3, "h": 1.8, "title": "4. Contrastive Scoring\nCLAP Scorer (U3)",
         "desc": "• 48 kHz polyphase resample\n• RoBERTa & HTSAT encoders\n• L2-norm cosine similarity\n• Primary intent delta", "color": "#fce4ec", "border": "#c2185b"},
        {"x": 4.0, "y": 1.2, "w": 4.0, "h": 2.0, "title": "5. Interactive Interface & Responsible AI Delivery",
         "desc": "• Gradio Blocks UI (Queue enabled for 2 vCPU Hugging Face Space)\n• Instant benchmark playback from precomputed offline cache\n• 2-D PCA latent space scatter & empirical failure case analysis",
         "color": "#f3e5f5", "border": "#7b1fa2"}
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"],
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=b["color"], edgecolor=b["border"], linewidth=1.8)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] - 0.35, b["title"],
                ha="center", va="center", fontsize=10.5, fontweight="bold", color=b["border"])
        ax.text(b["x"] + 0.15, b["y"] + b["h"] / 2 - 0.25, b["desc"],
                ha="left", va="center", fontsize=8.5, color="#222222")

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#37474f")
    ax.annotate("", xy=(3.2, 5.4), xytext=(2.7, 5.4), arrowprops=arrow_props)
    ax.annotate("", xy=(6.1, 5.4), xytext=(5.6, 5.4), arrowprops=arrow_props)
    ax.annotate("", xy=(9.2, 5.4), xytext=(8.7, 5.4), arrowprops=arrow_props)
    ax.annotate("", xy=(6.0, 3.2), xytext=(6.0, 4.5), arrowprops=arrow_props)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {output_path}")


def create_model_architecture_diagram(output_path: str = "docs/model_arch.png"):
    """
    Generates the MusicGen EnCodec & Attention architecture diagram (Syllabus Unit II).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6, 7.5, "MusicGen Generative Architecture: T5 Cross-Attention & EnCodec RVQ",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1a2530")

    # Components
    # 1. Conditioning branch
    rect1 = patches.FancyBboxPatch((0.5, 4.8), 3.0, 1.8, boxstyle="round,pad=0.1",
                                  facecolor="#e1f5fe", edgecolor="#0288d1", linewidth=1.8)
    ax.add_patch(rect1)
    ax.text(2.0, 6.2, "Conditioning Encoder", ha="center", va="center", fontsize=11, fontweight="bold", color="#0288d1")
    ax.text(2.0, 5.4, "Text Prompt\n↓\nFrozen T5-Base Text Encoder\n(Cross-Attention Key/Value Projections)",
            ha="center", va="center", fontsize=8.5)

    # 2. Autoregressive Transformer Decoder
    rect2 = patches.FancyBboxPatch((4.5, 3.8), 3.4, 3.2, boxstyle="round,pad=0.1",
                                  facecolor="#fff3e0", edgecolor="#ef6c00", linewidth=1.8)
    ax.add_patch(rect2)
    ax.text(6.2, 6.6, "Autoregressive Transformer Decoder", ha="center", va="center", fontsize=11, fontweight="bold", color="#ef6c00")
    ax.text(6.2, 5.2, "• 24-Layer Decoder-Only Transformer\n• Self-Attention + Causal Masking\n• Cross-Attention over T5 Embeddings\n• Interleaved Codebook Delay Pattern\n• Autoregressive Token Sampling (50 Hz)",
            ha="center", va="center", fontsize=8.5)

    # 3. EnCodec Audio Synthesis
    rect3 = patches.FancyBboxPatch((8.8, 3.8), 2.8, 3.2, boxstyle="round,pad=0.1",
                                  facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=1.8)
    ax.add_patch(rect3)
    ax.text(10.2, 6.6, "EnCodec Audio Engine", ha="center", va="center", fontsize=11, fontweight="bold", color="#2e7d32")
    ax.text(10.2, 5.2, "• Residual Vector Quantizer (RVQ)\n  4 parallel codebooks\n• VQ-VAE Style Latent Space\n• Convolutional GAN Decoder\n  (Trained with Multi-Scale STFT &\n   Adversarial Periodicity Loss)\n• 32 kHz Output Waveform",
            ha="center", va="center", fontsize=8.5)

    # Connecting arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#37474f")
    ax.annotate("", xy=(4.5, 5.7), xytext=(3.5, 5.7), arrowprops=arrow_props)
    ax.annotate("", xy=(8.8, 5.4), xytext=(7.9, 5.4), arrowprops=arrow_props)

    # Bottom notes
    rect_note = patches.FancyBboxPatch((1.5, 1.0), 9.0, 1.8, boxstyle="round,pad=0.1",
                                       facecolor="#f5f5f5", edgecolor="#9e9e9e", linewidth=1.2)
    ax.add_patch(rect_note)
    ax.text(6.0, 2.3, "Key Architectural Insights for University Viva", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#424242")
    ax.text(6.0, 1.6,
            "1. Multimodal Conditioning: Cross-attention layers inject dense T5 text embeddings at every transformer block.\n"
            "2. Delay Pattern: A slight offset between codebook steps enables single-pass generation of 4 RVQ streams.\n"
            "3. SoundStream GAN: EnCodec's decoder uses adversarial discrimination to eliminate phase artifacts in high frequencies.",
            ha="center", va="center", fontsize=8.5, color="#333333")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    create_lifecycle_diagram()
    create_model_architecture_diagram()

