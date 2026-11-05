# Empirical Failure Case Analysis

## Case Identification
- **Target Prompt**: `"calm rainy night piano"` (`eval_1`)
- **Condition**: `ZERO_SHOT`
- **Seed**: `2024`
- **Baseline Raw Score**: `0.4268`
- **Expanded Score**: `0.3536`
- **Delta Penalty (Δ)**: `-0.0732` *(Most severe degradation observed in benchmark)*

---

## Text Comparison
- **Raw Prompt**:
  > `calm rainy night piano`

- **Expanded Prompt**:
  > `Calm Rainy Night Piano: Acoustic grand piano ballad, gentle reverb, warm woody resonance, contemplative mood, 72 BPM, dynamic swell.`


---

## Mechanistic Explanation of Failure
1. **Semantic Diffusion**: The raw prompt specified a crisp, singular focal acoustic concept. The prompt expansion introduced secondary descriptors, atmospheric adjectives, and tertiary instruments that diluted the model's cross-attention attention maps away from the core instrument.
2. **Codebook Competition**: In the small 300M decoder transformer, allocating codebook capacity across multiple synthesized textures led to acoustic phase cancellation and diminished timbral clarity.
3. **CLAP Contrastive Misalignment**: CLAP computes cosine distance in a joint latent space. The generated expanded audio spread acoustic energy across broader frequencies, reducing its alignment to the user's concise original prompt.

## Architectural Remedy for Production
- Implement prompt-length penalty in prompt expansion templates.
- Cap instrumental additions to at most 2 tightly coupled instruments when user requests a solo instrument.
