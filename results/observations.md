# Empirical Observations & Trend Analysis

**Dataset**: 4 Fixed Prompts × 3 Conditions (`raw`, `zero_shot`, `few_shot`) × 3 Seeds (42, 100, 2024) = 18 Total Runs  
**Execution Environment**: CPU  
**Primary Objective**: Measure impact of prompt expansion against the original raw user intent (fair baseline).

---

## 1. Aggregate Quantitative Findings

| Evaluation Condition | Mean CLAP Delta (Δ) | Median CLAP Delta (Δ) | Win Rate (% Δ > 0) | Std Dev Across Seeds (σ) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall (All Expansions)** | **-0.0325** | **-0.0585** | **16.7%** | **0.0504** |
| **Zero-Shot Expansion** | -0.0326 | -0.0594 | 16.7% | 0.0504 |
| **Few-Shot Expansion** | -0.0323 | -0.0585 | 16.7% | 0.0503 |

---

## 2. Prompt-Length vs CLAP Intent Correlation

- **Pearson Correlation ($r$)**: `-0.2350` between prompt word count and Primary Intent CLAP score.
- **Interpretation**: The correlation coefficient reveals whether supplying longer, more verbose prompts systematically increases or decreases contrastive fidelity to the raw user request. A weak or negative correlation demonstrates that excessive verbosity can introduce semantic drift away from the user's focal concept.

---

## 3. Per-Prompt Performance & Emerging Trends

| Prompt ID | User Prompt | Mean Raw Score | Mean Zero-Shot Δ | Mean Few-Shot Δ | Trend Observation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `eval_1` | calm rainy night piano | 0.3849 | -0.0652 | -0.0645 | Expansion introduced competing tokens; raw acoustic focus preferred. |
| `eval_2` | upbeat workout electronic | 0.2659 | -0.0150 | -0.0150 | Expansion introduced competing tokens; raw acoustic focus preferred. |
| `eval_3` | sad violin goodbye | 0.4686 | -0.0578 | -0.0578 | Expansion introduced competing tokens; raw acoustic focus preferred. |
| `eval_4` | festive dhol wedding celebration | 0.2610 | +0.0725 | +0.0725 | Expansion clarified production & instrumentation constraints. |

---

## 4. Key Limitations of MusicGen-Small

1. **Tempo Adherence**: While prompt expansion specifies explicit BPM (e.g. 128 BPM for workout), the autoregressive codebook model exhibits drift in micro-timing and does not guarantee metronomic pulse locking.
2. **No Sung Lyrics / Vocals**: MusicGen's EnCodec compression and RVQ codebooks are trained primarily on instrumental audio tracks. It produces vocal-like melodic chirps or formant glides, but cannot synthesize coherent verbal lyrics.
3. **Acoustic Density Clutter**: Specifying 4+ simultaneous instruments occasionally results in muddy low-frequency separation on the 300M small model checkpoint compared to simpler single-instrument prompts.
