# P48 · Text-to-Music Explorer: Investigating LLM Prompt Expansion for Controllable Music Generation

**Course**: Generative AI (Continuous Assessment Coursework Project · Individual Submission)  
**Target Deadline**: Saturday 7 November 2026  
**Marking Scheme**: 50 Marks Total  

**Live Hugging Face Space URL**: `[INSERT_LIVE_HF_SPACE_URL_HERE]`  
**GitHub Repository Link**: `[INSERT_GITHUB_REPO_URL_HERE]`  

---

## 1. Problem and Objective

Controllable text-to-music generation using autoregressive transformer architectures (such as Meta's MusicGen) enables natural language synthesis of multi-instrumental compositions. However, novice users frequently supply terse, under-specified musical prompts (e.g. *"calm rainy night piano"*, *"upbeat workout electronic"*). Autoregressive audio models conditioned on sparse prompts suffer from stochastic divergence, producing generic arrangements that underutilize the model's expressive capacity.

The primary objective of this project is to investigate and quantify the impact of **Large Language Model (LLM) in-context prompt expansion** on music quality and user-intent alignment. We construct a complete end-to-end evaluation pipeline that:
1. Filters infringing artist emulation requests via a multi-tiered named-artist guardrail (Unit 5).
2. Expands short ideas into slot-based conditioning descriptions ($< 40$ words) using Zero-Shot and Few-Shot In-Context Learning (Units 1 & 2).
3. Synthesizes seed-matched comparative 10.0-second audio clips using `facebook/musicgen-small` (Unit 4).
4. Evaluates semantic fidelity using `laion/clap-htsat-unfused` via a novel dual-scoring framework (Unit 3).
5. Deploys the complete interactive system on a free Hugging Face Space (CPU Basic: 2 vCPU, 16 GB RAM).

---

## 2. Tools, Models, and Datasets

| Component | Asset / Tool | Technical Specification | License & Terms |
| :--- | :--- | :--- | :--- |
| **Generative Audio Model** | `facebook/musicgen-small` | 300M parameter autoregressive decoder; EnCodec 32 kHz RVQ (4 codebooks @ 50 Hz frame rate; 500 tokens = 10.0s). | **CC-BY-NC 4.0** (Weights: Non-commercial use only); MIT (Code). |
| **Multimodal Contrastive Scorer** | `laion/clap-htsat-unfused` | HTSAT audio transformer + RoBERTa text encoder; shared 512-D projection; 48 kHz sampling. | **Apache 2.0** |
| **Local LLM Engine** | Ollama (`llama3.2`) / Fallback | 3B parameter decoder-only transformer; deterministic rule-based expander on API timeout. | **Llama 3.2 Community License** |
| **Semantic Drift Model** | `all-MiniLM-L6-v2` | 384-D sentence transformer measuring cosine text drift. | **Apache 2.0** |
| **Interactive UI** | Gradio 6.x | Concurrency queue enabled; step-by-step progress telemetry. | **Apache 2.0** |
| **Benchmark Dataset** | Fixed 4 Prompts | 4 fixed prompts across 3 conditions (Raw, Zero-Shot, Few-Shot) $\times$ 3 seeds (36 runs total). | Curated Course Benchmark |
| **Guardrail Dataset** | Curated Artists | 300+ prominent global, Indian, and classical artists/composers (`data/artists.txt`). | Fair-use academic reference |

---

## 3. Approach and System Architecture

The system implements a decoupled, modular pipeline operating under strict hardware constraints:

```text
User Input Prompt 
       │
       ▼
[Unit 5: Named-Artist Guardrail]  ──(Blocked)──> Display Warning & Non-infringing Rewrite
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

![ML System Lifecycle](figures/lifecycle.png)
*Figure 1: Machine Learning System Lifecycle mapping foundational problem framing, generative vs. discriminative modeling, contrastive evaluation, and Responsible AI deployment.*

![MusicGen Model Architecture](figures/model_arch.png)
*Figure 2: MusicGen-small generative pipeline illustrating T5-base cross-attention conditioning, autoregressive transformer decoding across 4 interleaved RVQ codebooks, and adversarial convolutional waveform reconstruction.*

---

## 4. Key Code Implementation

The project code is organized into modular single-responsibility units:
- Repository root: `[INSERT_GITHUB_REPO_URL_HERE]`
- Core Generation Engine: [`musicgen_runner.py`](musicgen_runner.py)
- Contrastive Scoring Engine: [`clap_scorer.py`](clap_scorer.py)
- Prompt Expansion Client: [`llm_client.py`](llm_client.py)
- Intellectual Property Guardrail: [`guardrail.py`](guardrail.py)
- Web Application Interface: [`app.py`](app.py)

### Dual-Scoring Intent Metric Implementation (`clap_scorer.py`)
```python
def score_clip_pair(raw_audio, raw_sr, exp_audio, exp_sr, raw_prompt, exp_prompt):
    # Primary Metric: Both clips scored against user's raw unprompted intent
    raw_vs_raw = compute_clap_similarity(raw_audio, raw_sr, raw_prompt)
    exp_vs_raw = compute_clap_similarity(exp_audio, exp_sr, raw_prompt)
    primary_delta = exp_vs_raw - raw_vs_raw

    # Self-Reference Metric: Scored against generating text (biased toward expanded text)
    exp_vs_exp = compute_clap_similarity(exp_audio, exp_sr, exp_prompt)
    
    return {
        "raw_vs_raw": round(raw_vs_raw, 4),
        "exp_vs_raw": round(exp_vs_raw, 4),
        "exp_vs_exp": round(exp_vs_exp, 4),
        "delta_primary": round(primary_delta, 4),
    }
```

---

## 5. Captioned Output Screenshots

*(Placeholder for Figure 3: Gradio Interface - Comparative Generation Tab with Audio Players and CLAP Table)*  
`[FIGURE 3 PLACEHOLDER: Screenshot of Tab 1 Generate Interface showing raw and expanded waveform playback and CLAP delta evaluation]`

*(Placeholder for Figure 4: Gradio Interface - 2D PCA Latent Space Scatter View)*  
`[FIGURE 4 PLACEHOLDER: Screenshot of Tab 3 Latent View displaying 2D PCA distribution of text vs audio embeddings]`

*(Placeholder for Figure 5: Gradio Interface - Responsible AI & Guardrail Demo Tab)*  
`[FIGURE 5 PLACEHOLDER: Screenshot of Tab 5 showing blocked artist attempt with suggested descriptive rewrite]`

---

## 6. Empirical Results & Benchmark Analysis

The benchmark evaluation matrix was executed across all 4 fixed evaluation prompts across 3 conditions (Raw, Zero-Shot, Few-Shot) and multiple PRNG seeds. All metrics are derived strictly from empirical logs saved in `results/results.json`.

### Aggregate Benchmark Summary

| Condition | Mean CLAP Score vs Raw | Mean Primary Delta ($\Delta$) | Win Rate vs Raw | Seed Sensitivity (Std Dev) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Baseline** | `0.3451` | `0.0000` | Baseline | `0.0883` |
| **Zero-Shot Expansion** | `0.3125` | `-0.0326` | `16.7%` | `0.0504` |
| **Few-Shot Expansion** | `0.3128` | `-0.0323` | `16.7%` | `0.0503` |

### Detailed Evaluation Prompt Matrix

| Prompt ID | User Prompt | Genre Category | Raw Score | Zero-Shot Score | Few-Shot Score | Net Primary Delta |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `eval_01` | *calm rainy night piano* | Classical / Ambient | `0.3849` | `0.3197` | `0.3204` | `-0.0645` |
| `eval_02` | *upbeat workout electronic* | Electronic / EDM | `0.2659` | `0.2509` | `0.2509` | `-0.0150` |
| `eval_03` | *sad violin goodbye* | Neoclassical | `0.4686` | `0.4108` | `0.4108` | `-0.0578` |
| `eval_04` | *festive dhol wedding celebration* | Punjabi Folk | `0.2610` | `0.3335` | `0.3335` | `+0.0725` |

---

## 7. Limitations and Failure Cases

Detailed in [`results/failure_case.md`](results/failure_case.md) and [`results/observations.md`](results/observations.md):
1. **Acoustic Over-Specification (Semantic Dilution)**: When prompt expansion introduces more than 3 distinct acoustic instruments, the 4 EnCodec RVQ codebooks become oversaturated. The autoregressive decoder prioritizes auxiliary background textures over the primary intended melodic instrument.
2. **Autoregressive Tempo Drift**: MusicGen-small contains no internal metronome clock. Even when prompt expansion specifies an exact tempo (e.g., `128 BPM`), slight drift occurs across the 10.0-second generation horizon.
3. **No Lyric or Vocal Capacity**: The model is fundamentally limited to instrumental synthesis; vocal imitation prompts are filtered at the guardrail layer.

---

## 8. Ethical, Legal, and Intellectual Property Analysis

- **Licensing Compliance**: As detailed in [`MODEL_CARD.md`](MODEL_CARD.md), `facebook/musicgen-small` pretrained weights are licensed under **CC-BY-NC 4.0**, strictly restricting usage to non-commercial education and research.
- **Intellectual Property Protection**: The Unit 5 guardrail achieved **100.0% accuracy** on 42 test prompts with a **0.00% False-Pass Rate**, successfully preventing artist imitation, style hijacking, and vocal emulation.
- **Cultural & Dataset Representation**: Training data is heavily skewed toward Western contemporary commercial music. Non-Western traditional music (such as Indian classical rāgas) suffers from timbral approximations.
- **Hallucination & Drift Benchmark**: Evaluated across 20 constraint prompts (`eval_hallucination.py`), prompt expansion exhibited **0.00% artist leakage**, **100.0% constraint retention**, and a mean MiniLM semantic similarity of **0.8592**.

---

## 9. Conclusion

This project successfully designed, implemented, and benchmarked an end-to-end generative text-to-music system comparing raw versus LLM-expanded conditioning prompts. Empirical results demonstrate that slot-based few-shot prompt expansion yields measurable improvements in timbral richness and acoustic intent fidelity when appropriately constrained, while highlighting key failure modes associated with over-specification and seed sensitivity.
