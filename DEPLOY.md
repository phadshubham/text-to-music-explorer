# Deployment Guide: Hugging Face Spaces

This guide provides exact, step-by-step instructions for deploying the **P48 · Text-to-Music Explorer** to a free-tier **Hugging Face Space** (CPU Basic: 2 vCPU, 16 GB RAM, no GPU).

---

## 1. Prerequisites

1. A free account on [Hugging Face](https://huggingface.co).
2. Git installed on your local machine with Git LFS:
   ```bash
   git lfs install
   ```
3. (Optional but recommended) A free API key from [Groq Cloud](https://console.groq.com) or [Google AI Studio](https://aistudio.google.com).
   - *Note: If no API key is provided, the application gracefully defaults to the built-in deterministic slot expander (`FALLBACK (no LLM)`).*

---

## 2. Hugging Face Space Creation

1. Navigate to [Hugging Face New Space](https://huggingface.co/new-space).
2. Configure Space settings:
   - **Space Name**: `text-to-music-explorer` (or your preferred repository name).
   - **License**: `mit` (or `cc-by-nc-4.0` reflecting MusicGen weights).
   - **Select the Space SDK**: **Gradio**.
   - **Gradio Version**: `6.31.0` (or leave default; specified in `README.md`).
   - **Space Hardware**: **CPU Basic (2 vCPU · 16 GB RAM · Free)**.
   - **Visibility**: **Public** (required for university CA project submission).
3. Click **Create Space**.

---

## 3. Configuring Repository Secrets (Zero API Keys in Code)

1. On your newly created Space page, click the **Settings** tab (gear icon).
2. Scroll to the **Variables and secrets** section.
3. Click **New secret** to add your LLM credentials:
   - Key: `GROQ_API_KEY` | Value: `gsk_...` (your Groq key)
   - *(Optional)* Key: `GEMINI_API_KEY` | Value: `AIza...` (your Gemini key)
   - *(Optional)* Key: `LLM_PROVIDER` | Value: `groq` (default is `groq`)
4. Verify that no secrets are placed in `Variables` (which are public). Secrets remain encrypted and injected as `os.environ` variables at runtime.

---

## 4. Manual Deployment via Git

In your local project directory (`C:\GenAI`), follow these exact terminal steps:

```bash
# 1. Initialize git if not already initialized
git init

# 2. Verify git status to ensure .env and local caches are ignored
git status

# 3. Add the Hugging Face Space as a remote origin
# (Replace USERNAME and SPACE_NAME with your HF username and space name)
git remote add space https://huggingface.co/spaces/USERNAME/SPACE_NAME

# 4. Track large assets with Git LFS (for pre-computed audio files in results/audio)
git lfs track "results/audio/*.wav"
git add .gitattributes

# 5. Stage and commit all application files
git add app.py requirements.txt README.md MODEL_CARD.md DEPLOY.md
git add musicgen_runner.py clap_scorer.py llm_client.py guardrail.py
git add prompts/ data/ results/ docs/
git commit -m "Deploy P48 Text-to-Music Explorer to Hugging Face Spaces"

# 6. Push to Hugging Face Space main branch
git push space main
```

---

## 5. Space Startup Verification

1. Go to the **App** tab of your Hugging Face Space.
2. Watch the **Build Logs**:
   - Packages from `requirements.txt` will install automatically via `pip`.
   - The application starts with `python app.py`.
3. Startup performance note:
   - Thanks to **lazy singleton loading**, `facebook/musicgen-small` (2.2 GB) and `laion/clap-htsat-unfused` (614 MB) are NOT downloaded during container boot. The Gradio interface launches within **15 seconds**!
   - On the very first interactive generation in Tab 1, the weights will download from Hugging Face Hub directly into the Space's persistent cache. Subsequent runs will use the cached weights instantly.

---

## 6. Space Wake-up & Demo-Day Verification Checklist

Free-tier Hugging Face Spaces automatically enter sleep mode after 48 hours of inactivity. Follow this checklist 30 minutes before your university presentation or evaluation:

- [ ] **Space Wake-up**: Visit your public Space URL to trigger the container container boot (~30 seconds).
- [ ] **Tab 1 Interactive Verification**:
  - Run the test prompt `"calm rainy night piano"` with seed `42`.
  - Confirm that both audio players load audio and CLAP scores display within ~2–3 minutes.
- [ ] **Tab 2 Benchmark Verification**:
  - Open Tab 2 to verify that all 36 matrix runs and the failure case render instantly from pre-computed `results/results.json`.
- [ ] **Tab 3 Latent Space View**:
  - Confirm the 2D PCA plot displays the clustered text vs. audio points.
- [ ] **Tab 4 How It Works**:
  - Check that `docs/lifecycle.png` and `docs/model_arch.png` render cleanly.
- [ ] **Tab 5 Safety & Licence**:
  - Test the prompt `"in the style of Taylor Swift"` in the live guardrail tester and verify the refusal card and suggested rewrite.
