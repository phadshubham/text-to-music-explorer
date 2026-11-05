# Model Card: P48 · Text-to-Music Explorer

This model card adheres to the Mitchell et al. (2019) framework, providing comprehensive documentation of model architectures, data lineages, licensing constraints, empirical performance benchmarks, and ethical guardrails utilized across the P48 system.

---

## 1. System Overview & Model Architecture Summary

| Component | Model Identifier | Architecture | Parameters | Native Resolution / Target | Primary Function |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Audio Generator** | `facebook/musicgen-small` | Autoregressive Transformer Decoder + T5 Text Conditioning + EnCodec RVQ | 300M | 32 kHz Mono, 50 Hz frame rate | Synthesizes 10.0-second audio waveforms from natural language descriptions. |
| **Multimodal Scorer** | `laion/clap-htsat-unfused` | HTSAT Audio Transformer + RoBERTa Text Transformer | ~190M | 48 kHz resampled, 512-D joint embedding | Measures cross-modal text-to-audio cosine similarity in a shared latent space. |
| **Semantic Drift Scorer** | `sentence-transformers/all-MiniLM-L6-v2` | 6-layer BERT-family Transformer | 22.7M | 384-D dense sentence embedding | Quantifies semantic drift between raw and LLM-expanded prompts. |
| **Prompt Expansion LLM** | Groq (`llama-3.1-8b-instant`) / Gemini | Decoder-only Transformer | Configurable | Natural language text | Expands brief user music prompts into slot-constrained acoustic descriptions. |

---

## 2. Licensing, Intellectual Property & Commercial Constraints

> [!IMPORTANT]
> The weights and codebase of models integrated into this project are governed by distinct, non-fungible licenses. Compliance is strictly verified below:

### 2.1 facebook/musicgen-small
- **Pretrained Model Weights**: Licensed under the **Creative Commons Attribution-NonCommercial 4.0 International Public License (`CC-BY-NC 4.0`)**.
  - **Commercial Monetization**: **STRICTLY PROHIBITED**. The weights cannot be utilized in commercial music generation platforms, streaming release generation, or fee-charging applications.
  - **Permitted Use**: Permitted for academic research, university coursework, teaching demonstrations, and non-commercial open-source exploration.
- **Audiocraft Source Code**: Licensed under the permissive **MIT License** by Meta AI.

### 2.2 laion/clap-htsat-unfused
- **Code & Checkpoints**: Released under the **Apache License 2.0**.
  - **Terms**: Permissive open-source license allowing commercial and non-commercial research, reproduction, and distribution, subject to copyright notice and trademark disclaimer.

### 2.3 sentence-transformers/all-MiniLM-L6-v2
- **Checkpoint**: Released under the **Apache License 2.0**.

### 2.4 LLM Provider Inference Terms
- **Groq Cloud API**: Governed by Groq Developer Terms of Service. API inputs/outputs are not used to train proprietary frontier models in standard developer tiers.
- **Google Gemini API**: Governed by Google Cloud / AI Studio Developer Terms. Inputs are processed strictly for real-time inference without training on enterprise payloads.

---

## 3. Training Data & Pre-training Lineage

### 3.1 MusicGen Training Lineage
- **Dataset Scale**: Trained on 20,000 hours of licensed 32,000 Hz music:
  - 10,000 high-quality internal Meta dataset tracks.
  - 10,000 licensed instrumental tracks sourced from ShutterStock Music and Pond5.
- **Audio Tokenization**: Utilizes **EnCodec** (Défossez et al., 2022) with 4 Residual Vector Quantization (RVQ) codebooks operating at a 50 Hz frame rate (compression from 32,000 samples/sec to 200 codebook tokens/sec).
- **Text Conditioning**: Conditioned on text embeddings extracted from a frozen **T5-base** encoder.

### 3.2 CLAP Training Lineage
- **Dataset Scale**: Trained on the **LAION-Audio-630K** multimodal dataset (633,526 audio-text pairs; 4,325 hours):
  - Sourced from Freesound (300K+ clips), BBC Sound Effects (~16K clips), Free Music Archive (FMA), and YouTube-Sound-Collection.
- **Contrastive Pre-training**: Contrastive multimodal loss aligning 512-dimensional output projections of HTSAT (audio) and RoBERTa (text) using InfoNCE loss with a learned temperature parameter $\tau$.

---

## 4. Intended Use & Non-Commercial Scope

### Permitted / Intended Uses:
1. Academic evaluation of prompt engineering techniques (Zero-Shot vs Few-Shot) on generative audio models.
2. Objective benchmarking of multimodal contrastive alignment using CLAP scoring.
3. Analysis of acoustic failure cases, semantic diffusion, and latent-space representation geometries.
4. University demonstration and educational coursework submission (Course: Generative AI).

### Out-of-Scope / Prohibited Uses:
1. **Commercial Distribution**: Generating production tracks for commercial music streaming (Spotify, Apple Music) or selling generated clips.
2. **Artist Style Impersonation**: Generating tracks intended to emulate the signature sonic identity or deceive listeners into attributing music to specific living artists or bands (enforced via Unit 5 Guardrail).
3. **Deceptive Media / Deepfakes**: Creating synthetic musical audio impersonating human vocalists or public figures.
4. **Vocal Synthesis**: MusicGen is neither trained nor architecturally configured for verbal phonetic speech or lyric synthesis; attempts to enforce sung vocals produce degraded acoustic artifacts.

---

## 5. Architectural Limitations & Known Failure Modes

1. **Micro-Temporal Drift & BPM Loose-Locking**:
   - While prompts specify explicit metronomic BPM constraints (e.g., `"128 BPM"` for workout electronic), the autoregressive transformer samples tokens probabilistically. Phase alignment and micro-tempo drift are observed across 10-second clips.
2. **Absence of Phonetic Lyrical Synthesis**:
   - The EnCodec acoustic tokenizer and RVQ codebooks compress complex harmonic waveforms into discrete tokens. Without phoneme alignment, vocal instructions produce choral formant swells or melodic gibberish rather than intelligible words.
3. **Timbral Dilution under Dense Multi-Instrument Prompts**:
   - For the 300M parameter small checkpoint, specifying more than 3 simultaneous acoustic instruments creates codebook allocation competition, resulting in muddy frequency response and low-end masking.
4. **Contrastive Self-Scoring Bias**:
   - Scoring expanded audio clips against their verbose expanded prompts yields inflated CLAP scores (+0.10 to +0.15 higher) due to lexical vocabulary overlap with CLAP's RoBERTa encoder. True user intent alignment must always be measured against the unexpanded original prompt.

---

## 6. Cultural Fairness & Dataset Representation Caveats

1. **Western Pop/Electronic Bias**:
   - Training datasets for both MusicGen and CLAP skew heavily toward Western commercial electronic, ambient, rock, and pop genres.
2. **Non-Western Acoustic Traditions**:
   - While traditional instruments with prominent acoustic profiles (e.g., Punjabi *dhol*, Indian *sitar*, Irish *bodhrán*) are synthesized with characteristic timbres, complex microtonal nuances (e.g., Indian *ragas*, Arabic *maqamat*, Balinese *gamelan* tuning) are approximated within equal-temperament Western acoustic codebooks.

---

## 7. Empirical Safety & Guardrail Benchmark Verification

### 7.1 Unit 5 Named-Artist Guardrail Benchmark (`tests/test_guardrail.py`)
- **Test Corpus**: 42 curated test prompts (`data/guardrail_tests.json`) containing 21 adversarial style-emulation queries (including typos and casual phrasing) and 21 descriptive non-infringing prompts.
- **Empirical Results**:
  - **Overall Accuracy**: **100.00%** (42 / 42)
  - **False-Pass Rate (FPR)**: **0.00%** (0 / 21 infringing prompts allowed)
  - **False-Block Rate (FBR)**: **0.00%** (0 / 21 benign prompts blocked)
  - **Ambiguous Entity Handling**: Common-word entities (*"queen"*, *"prince"*, *"pink"*, *"future"*) pass in natural contexts (*"queen bee buzzing"*, *"pink sunset"*, *"future bass"*) and block only in explicit style-emulation contexts (*"sounds like Queen"*).

### 7.2 Unit 5 Hallucination & Semantic Drift Audit (`eval_hallucination.py`)
- **Test Corpus**: 20 short constrained prompts (`data/hallucination_prompts.json`) evaluated under Few-Shot expansion.
- **Empirical Results (`results/hallucination.json`)**:
  - **Artist Name Leakage Rate**: **5.00%** (1 / 20)
  - **User Constraint Retention Rate**: **100.00%** (20 / 20)
  - **Sentence Semantic Cosine Similarity (`all-MiniLM-L6-v2`)**: Mean = **0.8351**, Min = **0.7066**
