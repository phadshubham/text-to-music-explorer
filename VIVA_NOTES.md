# Viva Voce Preparation Guide: P48 · Text-to-Music Explorer

This document provides a comprehensive, rigorous set of 18 technical questions and answers designed for the final university Continuous Assessment (CA) project viva. Organized across the five course units, it arms you with exact equations, architectural mechanics, and empirical justifications.

---

## Unit I: Foundations & Multimodal Latent Representations

### Q1: What is the theoretical distinction between the generative model and the discriminative/contrastive model in your pipeline?
**Answer**:
- **Generative Component (`facebook/musicgen-small`)**: Models the joint probability distribution $P(X, Y)$ (or conditional distribution $P(Y|X)$), where $X$ is the conditioning text and $Y$ is the sequence of quantized acoustic tokens. It synthesizes novel high-dimensional audio waveforms autoregressively by sampling from the learned distribution over EnCodec codebooks.
- **Discriminative/Contrastive Component (`laion/clap-htsat-unfused`)**: Models semantic alignment by projecting separate modalities (audio $A$ and text $T$) into a shared multimodal latent space $\mathbb{R}^{512}$. Rather than generating data, it acts as a contrastive scorer computing the cosine similarity between normalized embedding vectors:
  $$\text{Score}(T, A) = \frac{\mathbf{e}_T \cdot \mathbf{e}_A}{\|\mathbf{e}_T\|_2 \|\mathbf{e}_A\|_2} \in [-1, 1]$$
  It discriminates between semantically aligned and misaligned pairs.

---

### Q2: Why is the CLAP score evaluated against the ORIGINAL raw prompt defined as your PRIMARY metric, rather than scoring against the expanded prompt?
**Answer**:
- Evaluating an expanded audio clip against its own verbose expanded text introduces an **artificial lexical bias**. CLAP's RoBERTa text encoder shares vocabulary with the expansion LLM; detailed acoustic adjectives ("warm resonance, reverb, 72 BPM") yield high text-audio similarity purely due to lexical overlap.
- However, the fundamental engineering goal is **fidelity to user intent**. If a user requests *"calm rainy night piano"*, and the expanded prompt adds *"ambient cello and vinyl crackle"*, the audio might match the expanded text well while betraying the user's focus on a solo piano.
- Therefore, the **unexpanded raw prompt is the only fair, objective baseline** against which both raw-generated and expanded-generated audio must be scored:
  $$\Delta_{\text{Primary}} = \text{CLAP}(\text{Audio}_{\text{Expanded}}, \text{Text}_{\text{Raw}}) - \text{CLAP}(\text{Audio}_{\text{Raw}}, \text{Text}_{\text{Raw}})$$

---

### Q3: How does the 2D PCA latent-space projection demonstrate multimodal clustering?
**Answer**:
- CLAP maps both text sentences and audio spectrograms into a shared 512-dimensional metric space.
- Principal Component Analysis (PCA) finds the orthogonal eigenvectors of the covariance matrix $\mathbf{\Sigma} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ corresponding to the two largest eigenvalues, maximizing explained variance.
- In our 2D PCA projection:
  - Text embeddings (represented as triangles) and audio embeddings (circles) form semantic clusters corresponding to genres (e.g., piano ballads vs. electronic workout beats).
  - Expanded text vectors typically shift closer to the dense cluster of training audio captions, visually corroborating why self-reference CLAP scores are higher.

---

## Unit II: GANs, Audio Tokenization & Autoregressive Transformers

### Q4: Explain the architecture of MusicGen's EnCodec audio tokenizer. Why is Residual Vector Quantization (RVQ) necessary?
**Answer**:
- Audio at 32,000 Hz contains 32,000 continuous amplitude values per second—computationally intractable for autoregressive transformer sequence lengths.
- **EnCodec** uses a 1D convolutional encoder to downsample audio temporally by a factor of 640, producing latent frames at a **50 Hz frame rate** (50 latent vectors per second).
- **Residual Vector Quantization (RVQ)**: A single vector quantizer would require an astronomically large codebook to reconstruct high-fidelity music. Instead, RVQ cascades $K = 4$ codebooks (each of size 2048):
  1. Quantizer 1 quantizes the latent vector $\mathbf{z}$ to $\mathbf{q}_1 = \mathcal{C}_1[\arg\min \|\mathbf{z} - \mathbf{c}\|]$.
  2. The residual $\mathbf{r}_1 = \mathbf{z} - \mathbf{q}_1$ is passed to Quantizer 2 to produce $\mathbf{q}_2$.
  3. This repeats for 4 levels: $\mathbf{z} \approx \sum_{k=1}^4 \mathbf{q}_k$.
- This enables discrete token representation with exponential expressive capacity ($2048^4$) while maintaining a compact vocabulary.

---

### Q5: How is a GAN involved in MusicGen if the core generative model is an Autoregressive Transformer?
**Answer**:
- MusicGen is a **hybrid two-stage architecture**:
  1. **Autoregressive Transformer**: Models the discrete token sequence generation over the RVQ codebooks.
  2. **EnCodec Decoder (SoundStream architecture)**: Reconstructs continuous waveforms from discrete codebook vectors.
- The EnCodec decoder is trained using an **Adversarial GAN loss**:
  - The generator is a Conv1D transposed convolutional network.
  - The discriminator consists of **Multi-Scale Discriminators (MSD)** and **Multi-Period Discriminators (MPD)** operating on raw waveforms and STFT spectrograms.
  - The GAN adversarial loss forces the reconstructed audio to preserve crisp transient attacks and high-frequency harmonics without the muffled blurring typical of pure $L_1$/$L_2$ reconstruction losses.

---

### Q6: How does MusicGen's transformer decoder cross-attend to text conditioning?
**Answer**:
- Text prompts are encoded into continuous hidden representations using a frozen **T5-base** encoder: $\mathbf{H}_{\text{text}} \in \mathbb{R}^{L_{\text{text}} \times D}$.
- In each of MusicGen's 24 transformer decoder layers, tokens undergo:
  1. **Causal Self-Attention**: Attending only to previously generated acoustic codebook tokens across an interleaved pattern delay.
  2. **Cross-Attention**: The acoustic queries $\mathbf{Q} = \mathbf{H}_{\text{audio}}\mathbf{W}_Q$ attend to the text keys $\mathbf{K} = \mathbf{H}_{\text{text}}\mathbf{W}_K$ and values $\mathbf{V} = \mathbf{H}_{\text{text}}\mathbf{W}_V$:
     $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
  3. **Feed-Forward MLP blocks**.
- Classifier-Free Guidance (CFG with scale $s=3.0$) blends conditional and unconditional logits: $\mathbf{l} = \mathbf{l}_{\text{uncond}} + s(\mathbf{l}_{\text{cond}} - \mathbf{l}_{\text{uncond}})$.

---

## Unit III: GPT, BERT & Multimodal Alignment

### Q7: Compare the roles of GPT-style and BERT-style models across your pipeline.
**Answer**:
- **GPT-style (Decoder-only Autoregressive)**:
  - **Expansion LLM** (Groq Llama-3.1-8b-instant): Generates expanded prompt tokens autoregressively from left to right using causal masking.
  - **MusicGen Decoder**: Autoregressively generates acoustic codebook tokens over time steps $t = 1 \dots 500$ (500 tokens = 10s @ 50 Hz).
- **BERT-style (Encoder-only Bidirectional)**:
  - **CLAP Text Encoder (RoBERTa)**: Processes the full text bidirectionally to extract dense pooled sentence representations.
  - **all-MiniLM-L6-v2**: 6-layer bidirectional transformer used in `eval_hallucination.py` to evaluate semantic drift between raw and expanded prompts via sentence embedding cosine similarity.

---

### Q8: What is the architecture of CLAP (`laion/clap-htsat-unfused`) and how was it trained?
**Answer**:
- CLAP consists of two dual encoders:
  1. **Audio Encoder**: **HTSAT** (Hierarchical Token-Semantic Audio Transformer), which processes 48 kHz mel-spectrograms hierarchically using shifted window self-attention (Swintransformer-style).
  2. **Text Encoder**: **RoBERTa-base**, extracting text token representations.
- Both outputs are projected via linear layers to a shared dimension $D = 512$ and $L_2$-normalized.
- Pretrained on **LAION-Audio-630K** using symmetric cross-entropy InfoNCE contrastive loss over batches of $N$ audio-text pairs:
  $$\mathcal{L}_{\text{audio}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\mathbf{a}_i \cdot \mathbf{t}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{a}_i \cdot \mathbf{t}_j / \tau)}$$

---

### Q9: Why is rational polyphase resampling (`resample_poly`) required for CLAP?
**Answer**:
- MusicGen outputs native audio at **32,000 Hz**.
- CLAP's HTSAT spectrogram feature extractor strictly requires **48,000 Hz** audio with 64 mel-filterbank bins.
- Naive linear interpolation introduces aliasing distortion and high-frequency spectral imaging.
- We utilize `scipy.signal.resample_poly` with upsample factor $P = 3$ and downsample factor $Q = 2$ (since $\frac{48000}{32000} = \frac{3}{2}$):
  1. Zero-stuff by factor 3.
  2. Apply an FIR low-pass anti-aliasing filter at the Nyquist frequency of the lower rate.
  3. Decimate by factor 2.
- This preserves phase linearity and prevents spurious high-frequency artifacts from skewing CLAP scores.

---

## Unit IV: Tools, Generation & Engineering Optimization

### Q10: How did you ensure deterministic reproducibility between Clip A (Raw) and Clip B (Expanded)?
**Answer**:
- In `musicgen_runner.py`, we enforce identical random number generator states before every generation run:
  ```python
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
  ```
- Because MusicGen uses stochastic sampling (`do_sample=True, guidance_scale=3.0`), fixing the seed ensures that all non-text sources of variance (sampling temperature noise, codebook sampling permutations) are held constant. Any acoustic delta observed between Clip A and Clip B is strictly attributable to the prompt expansion text.

---

### Q11: How is your system architected to run efficiently on a free Hugging Face Space (2 vCPU, 16 GB RAM)?
**Answer**:
1. **Lazy Singleton Loading**: Heavy model checkpoints (`MusicgenForConditionalGeneration` at 2.2 GB, `ClapModel` at 614 MB) are loaded into memory only when the user executes their first generation, not during app startup.
2. **CPU Thread Tuning**: PyTorch CPU intra-op parallelism is configured to match available cores (`torch.set_num_threads(os.cpu_count())`), maximizing OpenMP matrix throughput.
3. **Clip Length Assertion**: Fixed at 500 tokens ($10\text{s} \times 50\text{ Hz}$), preventing out-of-memory container crashes.
4. **Gradio Queuing (`demo.queue(concurrency_limit=1)`)**: Prevents concurrent request collisions on limited CPU hardware.
5. **Pre-computed Offline Benchmark**: The 36-run evaluation matrix is pre-computed and serialized into `results/results.json`, allowing instant inspection in Tab 2 without CPU latency.

---

## Unit V: Responsible AI, Guardrails & Governance

### Q12: Walk through the three layers of your Unit 5 Named-Artist Guardrail.
**Answer**:
1. **Layer 1: Regex Style-Reference Triggers**: Detects indicative attribution phrases (`in the style of`, `sounds like`, `similar to`, `in the vein of`, `inspired by`, `cover of`, `remix of`, `in <Name>'s voice`).
2. **Layer 2: Curated 669-Entity Artist Database with Fuzzy Matching**:
   - Case-insensitive search across 669 global and Indian artists, bands, and classical composers (`data/artists.txt`).
   - Token-level fuzzy matching using `rapidfuzz.fuzz.token_set_ratio` with a strict 85% threshold, catching intentional typos (e.g., *"Beethovan"*, *"Taylor Swift-ish"*).
   - **Ambiguous Word Disambiguation**: Words that are common nouns (*"Queen"*, *"Prince"*, *"Pink"*, *"Future"*, *"Air"*) only trigger a block if accompanied by an explicit Layer 1 style reference.
3. **Layer 3: Descriptive Non-Infringing Rewrite Engine**: Automatically suggests an educational acoustic rewrite specifying genre, era, instrumentation, and tempo instead of an artist name.

---

### Q13: What were your empirical findings in `tests/test_guardrail.py`?
**Answer**:
- Tested across 42 rigorous prompts (21 adversarial style queries, 21 legitimate descriptive prompts).
- **Accuracy**: **100.00%** (42 / 42).
- **False-Pass Rate (FPR)**: **0.00%** (zero infringing queries slipped past).
- **False-Block Rate (FBR)**: **0.00%** (all benign prompts like *"queen bee buzzing ambience"*, *"pink sunset lofi"*, and *"Baroque harpsichord"* passed cleanly).

---

### Q14: What were your empirical findings in the Responsible AI Hallucination Test (`eval_hallucination.py`)?
**Answer**:
- Evaluated 20 short constrained prompts (`data/hallucination_prompts.json`) under Few-Shot expansion:
  - **Artist Leakage Rate**: **5.00%** (1 / 20) — Prompt 10 leaked an artist name during expansion, which was immediately caught and redacted by the outbound guardrail.
  - **User Constraint Retention Rate**: **100.00%** (20 / 20) — All explicit instruments, tempos, and mood constraints stated by the user were preserved in the output.
  - **Semantic Drift (`all-MiniLM-L6-v2`)**: Mean cosine similarity of **0.8351** (minimum **0.7066**), demonstrating that prompt expansion enriched descriptors while preserving core semantic orientation.

---

### Q15: Explain the licensing dichotomy of `facebook/musicgen-small`. Why can this app NOT be released as a paid commercial product?
**Answer**:
- While the **Audiocraft source code** is released under the permissive **MIT License**, the **pretrained model weights** are released under **Creative Commons Attribution-NonCommercial 4.0 (`CC-BY-NC 4.0`)**.
- Clause 2(a)(1) of CC-BY-NC 4.0 explicitly prohibits exercising licensed rights primarily intended for or directed toward commercial advantage or monetary compensation.
- Deploying this system as a paid SaaS platform, running ads against it, or using it to generate tracks for commercial sale would directly violate Meta AI's copyright. It is legally restricted to academic research, coursework, and personal educational exploration.

---

## Synthesis & Empirical Takeaway

### Q16: In one sentence, does LLM prompt expansion improve or degrade MusicGen audio?
**Answer**:
> *"Prompt expansion is a double-edged sword: while it significantly boosts perceived production polish and self-reference CLAP scores, it frequently degrades primary user intent alignment (-0.06 to -0.08 CLAP delta) due to semantic diffusion and EnCodec codebook allocation competition on small models."*

---

### Q17: What is the "codebook allocation competition" phenomenon?
**Answer**:
- In `musicgen-small` (300M parameters, 4 RVQ codebooks), each time step has only 4 codebook tokens to represent all acoustic energy across the frequency spectrum.
- When an expanded prompt introduces multiple complementary instruments (e.g., piano + cello + vinyl crackle + rain ambience), the cross-attention heads divide codebook capacity across all these textures.
- This results in phase cancellation and timbral masking, degrading the acoustic clarity of the primary instrument the user originally wanted.

---

### Q18: If you had another 2 weeks on this project, what single architectural enhancement would you prioritize?
**Answer**:
- Implement **Per-Token Attention Weight Re-scaling** in MusicGen's cross-attention layers: dynamically penalizing attention weights on adjective modifiers while boosting attention to core noun instruments, preventing the generative decoder from drowning out primary user constraints.
