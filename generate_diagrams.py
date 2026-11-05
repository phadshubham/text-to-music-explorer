"""
Generate Academic Diagrams for Syllabus Coverage:
1. docs/lifecycle.png: System ML Lifecycle & Generative vs Discriminative framing.
2. docs/model_arch.png: MusicGen EnCodec RVQ + T5 Cross-Attention Architecture.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
os.makedirs(DOCS_DIR, exist_ok=True)


def create_lifecycle_diagram():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(6, 5.6, "Text-to-Music ML System Lifecycle & Model Framing", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1a237e")

    # Boxes: Stage 1 to Stage 5
    stages = [
        {"x": 0.5, "y": 3.0, "w": 2.0, "h": 1.6, "title": "1. Input & Guard", "desc": "User Prompt\n+\nU5 Artist Guardrail\n(RapidFuzz + Regex)", "color": "#e8eaf6", "border": "#3949ab"},
        {"x": 2.8, "y": 3.0, "w": 2.0, "h": 1.6, "title": "2. Prompt Engine", "desc": "U1 Slot Templates\n+\nU2 Expansion LLM\n(Zero/Few-Shot ICL)", "color": "#e1f5fe", "border": "#0288d1"},
        {"x": 5.1, "y": 3.0, "w": 2.2, "h": 1.6, "title": "3. Generative Audio", "desc": "U4 MusicGen-Small\n(500 tokens, 10s)\nAutoregressive Decoder\n[Generative Model]", "color": "#e8f5e9", "border": "#2e7d32"},
        {"x": 7.6, "y": 3.0, "w": 2.1, "h": 1.6, "title": "4. Contrastive Eval", "desc": "U3 CLAP Scorer\n(RoBERTa + HTSAT)\nResample @ 48 kHz\n[Discriminative Model]", "color": "#fff3e0", "border": "#ef6c00"},
        {"x": 10.0, "y": 3.0, "w": 1.6, "h": 1.6, "title": "5. UI & Report", "desc": "Gradio 5-Tabs\nDual Metrics Δ\nPCA Latent View", "color": "#f3e5f5", "border": "#7b1fa2"},
    ]

    for s in stages:
        box = patches.FancyBboxPatch(
            (s["x"], s["y"]), s["w"], s["h"],
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=s["color"], edgecolor=s["border"], linewidth=1.8
        )
        ax.add_patch(box)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"] - 0.3, s["title"],
                ha="center", va="center", fontsize=10, fontweight="bold", color=s["border"])
        ax.text(s["x"] + s["w"]/2, s["y"] + (s["h"]-0.3)/2, s["desc"],
                ha="center", va="center", fontsize=8.5, color="#212121")

    # Arrows between sequential stages
    arrow_props = dict(facecolor="#424242", edgecolor="#424242", width=1.5, headwidth=7)
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + stages[i]["w"] + 0.05
        x_end = stages[i+1]["x"] - 0.05
        ax.annotate("", xy=(x_end, 3.8), xytext=(x_start, 3.8), arrowprops=arrow_props)

    # Theoretical Framing Banner below
    frame_box = patches.FancyBboxPatch(
        (0.5, 0.6), 11.1, 1.8,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#f5f5f5", edgecolor="#9e9e9e", linewidth=1.2
    )
    ax.add_patch(frame_box)

    ax.text(6, 2.1, "Unit I & II Syllabus Foundations: Theoretical Model Categorization",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#37474f")

    gen_desc = "• GENERATIVE COMPONENTS: Expansion LLM (learns p(Expanded | Raw)) & MusicGen (learns p(Audio Tokens | Text Embeddings))\n  Generative modeling synthesizes high-dimensional multimodal content from low-dimensional prompt representations."
    disc_desc = "• DISCRIMINATIVE / CONTRASTIVE SCORER: CLAP Model (learns joint multimodal embedding space e_text · e_audio)\n  Discriminative objective projects audio and text into aligned latent space to measure mutual semantic fidelity without synthesizing audio."
    
    ax.text(0.8, 1.45, gen_desc, ha="left", va="center", fontsize=8.5, color="#1b5e20")
    ax.text(0.8, 0.95, disc_desc, ha="left", va="center", fontsize=8.5, color="#b71c1c")

    plt.tight_layout()
    out_path = os.path.join(DOCS_DIR, "lifecycle.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Diagrams] Saved lifecycle diagram to {out_path}")


def create_model_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.6, "MusicGen Architecture: T5 Cross-Attention & EnCodec RVQ Synthesis",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0d47a1")

    # Text conditioning block (Top Left)
    t5_box = patches.FancyBboxPatch(
        (0.6, 4.3), 3.2, 1.8, boxstyle="round,pad=0.1",
        facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=1.6
    )
    ax.add_patch(t5_box)
    ax.text(2.2, 5.7, "Text Conditioning", ha="center", va="center", fontsize=10, fontweight="bold", color="#1565c0")
    ax.text(2.2, 4.9, "Prompt: '{expanded_text}'\nFrozen T5 Text Encoder\nShape: [Batch, T_text, 768]", ha="center", va="center", fontsize=8.5)

    # Autoregressive Transformer Decoder (Center)
    dec_box = patches.FancyBboxPatch(
        (4.5, 3.2), 3.6, 2.9, boxstyle="round,pad=0.1",
        facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=1.8
    )
    ax.add_patch(dec_box)
    ax.text(6.3, 5.7, "MusicGen Decoder Transformer", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#2e7d32")
    ax.text(6.3, 4.5, "• 24 Transformer Layers (300M Params)\n• Self-Attention (Causal Masking)\n• Cross-Attention to T5 Text Embeddings\n• Codebook Interleaving (Delay Pattern)\n• Autoregressive Step: 50 Hz (500 steps = 10s)", ha="center", va="center", fontsize=8)

    # Arrows from T5 to Decoder
    ax.annotate("Cross-Attention\nK, V Projections", xy=(4.5, 4.8), xytext=(3.8, 4.8),
                arrowprops=dict(facecolor="#1565c0", edgecolor="#1565c0", width=1.5, headwidth=6),
                ha="center", va="bottom", fontsize=7.5, color="#1565c0")

    # EnCodec RVQ Block (Right)
    rvq_box = patches.FancyBboxPatch(
        (8.8, 3.2), 2.8, 2.9, boxstyle="round,pad=0.1",
        facecolor="#fff8e1", edgecolor="#f57f17", linewidth=1.6
    )
    ax.add_patch(rvq_box)
    ax.text(10.2, 5.7, "EnCodec RVQ Tokenizer", ha="center", va="center", fontsize=10, fontweight="bold", color="#f57f17")
    ax.text(10.2, 4.5, "Residual Vector Quantization\n• 4 Codebooks (RVQ Q1..Q4)\n• Vocabulary: 2048 entries each\n• Bandwidth: 3.0 kbps @ 50 Hz\n• Hierarchical codebook refinement", ha="center", va="center", fontsize=7.8)

    # Arrow from Decoder to RVQ
    ax.annotate("4 Token Streams", xy=(8.8, 4.6), xytext=(8.1, 4.6),
                arrowprops=dict(facecolor="#2e7d32", edgecolor="#2e7d32", width=1.5, headwidth=6),
                ha="center", va="bottom", fontsize=8, color="#2e7d32")

    # Bottom block: Convolutional GAN Decoder & Audio
    gan_box = patches.FancyBboxPatch(
        (3.0, 0.6), 6.5, 1.8, boxstyle="round,pad=0.1",
        facecolor="#fce4ec", edgecolor="#c2185b", linewidth=1.6
    )
    ax.add_patch(gan_box)
    ax.text(6.25, 2.0, "EnCodec Convolutional GAN Decoder (SoundStream)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#c2185b")
    ax.text(6.25, 1.25, "• Transposed 1D Convolutions with Multi-Scale Discriminator (MSD) Training\n• Decodes Quantized Discrete Codebooks into Continuous Time-Domain Audio\n• Output: 10.0-Second Single-Channel Audio Waveform @ 32,000 Hz Native Rate", ha="center", va="center", fontsize=8.2)

    # Arrow from RVQ down to GAN Decoder
    ax.annotate("", xy=(8.5, 2.4), xytext=(9.8, 3.2),
                arrowprops=dict(facecolor="#f57f17", edgecolor="#f57f17", width=1.5, headwidth=6))

    plt.tight_layout()
    out_path = os.path.join(DOCS_DIR, "model_arch.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Diagrams] Saved model architecture diagram to {out_path}")


if __name__ == "__main__":
    create_lifecycle_diagram()
    create_model_architecture_diagram()
