---
title: P48 · Text-to-Music Explorer
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: cc-by-nc-4.0
---

# 🎵 P48 · Text-to-Music Explorer (MusicGen & CLAP)
### Investigating the Impact of In-Context Prompt Expansion on Generative Audio Quality

**Course**: Generative AI (Continuous Assessment Coursework Project · Individual Submission)  
**Marking Scale**: Marked out of 50  
**Submission Target**: Saturday 7 November 2026  
**Hardware Profile**: Hugging Face Spaces Free Tier (CPU Basic: 2 vCPU, 16 GB RAM, No GPU)

---

## 🌟 Executive Summary

Text-to-music generative AI models like Meta's `facebook/musicgen-small` are highly sensitive to conditioning prompts. Sparse inputs (such as *"calm rainy night piano"*) leave the autoregressive transformer decoder with substantial acoustic ambiguity, often resulting in generic timbral textures.

**P48 · Text-to-Music Explorer** investigates whether in-context LLM prompt expansion reliably improves musical fidelity and intent alignment. The pipeline:
1. Filters artist impersonation via a **Unit 5 Named-Artist Guardrail** (regex + RapidFuzz on 300+ artists).
2. Expands terse ideas into slot-based musical specifications ($\le 40$ words) using **Zero-Shot** or **Few-Shot In-Context Learning** via local Ollama or high-reliability deterministic fallback.
3. Generates seed-matched **10.0-second audio clips** (Clip A: Raw Prompt vs. Clip B: Expanded Prompt) using `facebook/musicgen-small` at 50 Hz ($500\text{ tokens}$).
4. Evaluates both clips using `laion/clap-htsat-unfused` contrastive embeddings, reporting the **unbiased Primary Intent Metric** ($\text{Score vs. Original Raw Prompt}$) and the self-reference metric.

---

## 🏛️ System Architecture

```text
User Input Prompt 
       │
       ▼
[Unit 5: Named-Artist Guardrail]  ──(Blocked)──> Alert & Non-Infringing Rewrite
       │ (Allowed)
       ▼
[Unit 1 & 2: Prompt Expansion]    ──(Few-Shot / Zero-Shot via Ollama or Rule Fallback)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Raw Conditioning Text]         [Expanded Conditioning Text]
       │                                 │
       ▼                                 ▼
[MusicGen-small (Seed S, 500 tok)]  [MusicGen-small (Seed S, 500 tok)]
       │                                 │
       ▼                                 ▼
Clip A: Raw Audio (10s @ 32 kHz)   Clip B: Expanded Audio (10s @ 32 kHz)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
           [48 kHz Polyphase Resampling]
                        ▼
     [Unit 3: LAION-CLAP Contrastive Scorer]
                        ▼
      Dual Metrics: Intent Fidelity Delta (Δ)
```

---

## 📚 Viva Syllabus Alignment Matrix

| Syllabus Unit | Core Theoretical Concept | Project Implementation | Key Repository File |
| :--- | :--- | :--- | :--- |
| **I. Foundations** | • Generative vs. Discriminative framing<br>• In-context learning (Zero vs Few-Shot)<br>• Latent space representation | • Explicit contrast: MusicGen (generative) vs CLAP (contrastive/discriminative)<br>• ZERO_SHOT and FEW_SHOT templates<br>• 2D PCA latent space scatter | [`docs/lifecycle.png`](docs/lifecycle.png)<br>[`prompts/templates.py`](prompts/templates.py)<br>`app.py` (Tab 3) |
| **II. GANs & LLMs** | • Autoregressive token decoding<br>• Residual Vector Quantization (RVQ)<br>• Adversarial discriminator loss | • MusicGen decoder-only transformer autoregression<br>• Explainer & diagram: EnCodec RVQ codebooks and ConvNet GAN decoder | [`docs/model_arch.png`](docs/model_arch.png)<br>[`musicgen_runner.py`](musicgen_runner.py)<br>`app.py` (Tab 4) |
| **III. GPT, BERT & Multimodal** | • Decoder-only vs encoder-only transformers<br>• Joint text-audio metric spaces<br>• Semantic distance measurement | • Multi-model synergy: Llama/Gemini (decoder LLM) + RoBERTa (CLAP text encoder) + HTSAT (audio encoder)<br>• `all-MiniLM-L6-v2` cosine semantic drift | [`clap_scorer.py`](clap_scorer.py)<br>[`eval_hallucination.py`](eval_hallucination.py) |
| **IV. Tools & Generation** | • PyTorch CPU inference optimization<br>• Audio waveform synthesis & manipulation<br>• Interactive web interfaces | • PyTorch inference mode, thread tuning, native 32 kHz sampling<br>• High-fidelity 48 kHz polyphase resampler<br>• Production Gradio Blocks UI with queueing | [`musicgen_runner.py`](musicgen_runner.py)<br>[`app.py`](app.py) |
| **V. Responsible AI** | • Intellectual property & artist attribution<br>• Hallucination & constraint degradation<br>• Model transparency & licensing | • Layered named-artist guardrail (100% accuracy, 0% FPR)<br>• 20-prompt automated hallucination test<br>• Official `MODEL_CARD.md` (CC-BY-NC 4.0 analysis) | [`guardrail.py`](guardrail.py)<br>[`eval_hallucination.py`](eval_hallucination.py)<br>[`MODEL_CARD.md`](MODEL_CARD.md) |

---

## 🔬 Experimental Methodology & Dual CLAP Metric

### Why the Raw Prompt is the PRIMARY Metric:
A critical methodological contribution of this project is addressing **evaluator bias**:
- **Baseline Metric**: $\text{Score}(T_{\text{raw}}, A_{\text{raw}})$
- **Primary Unbiased Intent Metric**: $\text{Score}(T_{\text{raw}}, A_{\text{exp}})$
- **Primary Delta**:
  $$\Delta_{\text{primary}} = \text{Score}(T_{\text{raw}}, A_{\text{exp}}) - \text{Score}(T_{\text{raw}}, A_{\text{raw}})$$
- **Self-Reference Metric (Biased)**: $\text{Score}(T_{\text{exp}}, A_{\text{exp}})$
  *Evaluating expanded audio against its own expanded prompt is inherently biased toward expanded audio due to lexical overlap. The primary metric guarantees a fair, uncompromised test of whether prompt expansion better realizes the user's unprompted intent.*

---

## 🚀 Local Installation & Execution

### 1. Clone Repository & Setup Environment
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

# Create clean virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install CPU-optimized pinned dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` (optional):
```bash
cp .env.example .env
```
To run zero-API-key local inference with Ollama:
```bash
ollama pull llama3.2
ollama serve
```
*If Ollama or API keys are not detected, the app automatically switches to the deterministic fallback expander with zero crashes.*

### 3. Launch Gradio Web Application
```bash
python app.py
```
Open your browser to `http://127.0.0.1:7860`.

---

## 🧪 Testing & Verification Suite

```bash
# 1. Run Unit 5 Guardrail Test Suite (42 test cases)
python tests/test_guardrail.py

# 2. Run Responsible AI Hallucination & Semantic Drift Benchmark
python eval_hallucination.py

# 3. Run Offline Benchmark Matrix (36 experimental runs)
python benchmark.py

# 4. Generate Quantitative Analysis Reports
python analysis.py

# 5. Generate Publication-Quality System Diagrams
python generate_diagrams.py
```

---

## ⚖️ Licensing & Model Cards
- **MusicGen Weights**: Creative Commons Attribution Non Commercial 4.0 International (`CC-BY-NC 4.0`). Non-commercial academic research and university coursework use only.
- **Audiocraft Code**: MIT License.
- **CLAP & MiniLM**: Apache License 2.0.
- For complete training data disclosures, limitations, and ethical guidelines, see [`MODEL_CARD.md`](MODEL_CARD.md).
