"""Builds the P48 report PDF.  Run:  python3 build_report.py
Edit the CONFIG block, drop screenshots into ./screenshots/, re-run."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                PageBreak, Table, TableStyle, Image, KeepTogether, Preformatted)
from PIL import Image as PILImage

# ---------------- CONFIG ----------------
ROLL = "48"
NAME = "Shubham Shivaji Phad"
ENROLLMENT = "ADT23SOCB1106"
CLASS_NAME = "LYAIA-3"
HF_URL = "https://huggingface.co/spaces/Spacecraft21/text-to-music-explorer"
GH_URL = "https://github.com/phadshubham/text-to-music-explorer"
OUT = "48_Shubham_P48.pdf"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "report", "figures")
SHOT = os.path.join(HERE, "report", "screenshots")
os.makedirs(SHOT, exist_ok=True)
# ----------------------------------------

D = "/usr/share/fonts/truetype/dejavu/"
if os.path.exists(D + "DejaVuSans.ttf"):
    pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
    pdfmetrics.registerFont(TTFont("DV-BI", D + "DejaVuSans-BoldOblique.ttf"))
    pdfmetrics.registerFont(TTFont("DVM", D + "DejaVuSansMono.ttf"))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
    FB, FBB, FBI, FBM = "DV", "DV-B", "DV-I", "DVM"
else:
    FB, FBB, FBI, FBM = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"

NAVY = colors.HexColor("#1a237e"); GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#eef0fa"); RULE = colors.HexColor("#c5cae9")

body = ParagraphStyle("body", fontName=FB, fontSize=9.2, leading=13.6, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=11, spaceAfter=2)
cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10.5, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName=FBB, textColor=colors.white)
h1 = ParagraphStyle("h1", fontName=FBB, fontSize=14, leading=18, textColor=NAVY, spaceBefore=4, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName=FBB, fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=3)
cap = ParagraphStyle("cap", parent=body, fontName=FBI, fontSize=8, leading=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=8)
bul = ParagraphStyle("bul", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
code = ParagraphStyle("code", fontName=FBM, fontSize=7.4, leading=9.6, backColor=colors.HexColor("#f4f4f6"),
                      borderPadding=5, spaceBefore=4, spaceAfter=8)

def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bul, bulletText="•")

def table(rows, widths, header=True, align_center_from=None):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        for i in range(1, len(rows)):
            if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t

def img(path, width):
    w, h = PILImage.open(path).size
    return Image(path, width=width, height=width * h / w)

def figure(path, width, caption):
    return KeepTogether([img(path, width), P(caption, cap)])

def screenshot(fname, caption, w=150 * mm, h=70 * mm):
    p = os.path.join(SHOT, fname)
    if os.path.exists(p):
        return figure(p, w, caption)
    t = Table([[Paragraph(f"<b>Screenshot not yet added</b><br/>Save the screenshot as <font name='{FBM}'>screenshots/{fname}</font> and re-run <font name='{FBM}'>build_report.py</font>.", cell)]],
              colWidths=[w], rowHeights=[h])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#bbbbbb")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return KeepTogether([t, P(caption, cap)])

def footer(c, doc):
    c.saveState(); c.setFont(FB, 7.5); c.setFillColor(GREY)
    c.drawString(18 * mm, 10 * mm, f"P48 · Text-to-Music Explorer  |  Roll No. {ROLL} · {NAME}")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    c.restoreState()

doc = BaseDocTemplate(os.path.join(HERE, OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=18 * mm,
                      title="P48 Text-to-Music Explorer", author=NAME)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])
W = doc.width
s = []

# ---------------- 1. COVER ----------------
s += [Spacer(1, 38 * mm),
      P("P48 · Text-to-Music Explorer", ParagraphStyle("t", fontName=FBB, fontSize=26, leading=32, textColor=NAVY, alignment=TA_CENTER)),
      Spacer(1, 4 * mm),
      P("Does LLM prompt expansion improve MusicGen output? A CLAP-based comparison of raw vs expanded prompts",
        ParagraphStyle("st", fontName=FB, fontSize=12, leading=17, textColor=GREY, alignment=TA_CENTER)),
      Spacer(1, 18 * mm)]
cov = [["Course", "Generative AI: Continuous Assessment (individual project)"],
       ["Project", "P48 · Text-to-Music Explorer (MusicGen)"],
       ["Name", f"{NAME}"],
       ["Roll No.", ROLL],
       ["Enrollment No.", ENROLLMENT],
       ["Class", CLASS_NAME],
       ["Live app URL", HF_URL],
       ["GitHub repo", GH_URL]]
t = Table([[Paragraph(f"<b>{a}</b>", cell), Paragraph(b, ParagraphStyle("cv", parent=cell, fontSize=9))] for a, b in cov],
          colWidths=[40 * mm, W - 40 * mm])
t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                       ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
s += [t, PageBreak()]

# ---------------- 2. PROBLEM & OBJECTIVE ----------------
s += [P("1. Problem and Objective", h1),
      P("Text-to-music models such as Meta's MusicGen turn a written description into audio. Users tend to write very short prompts "
        "(<i>“calm rainy night piano”</i>), which leave a lot unspecified. A common idea is to let a large language model (LLM) "
        "<b>expand</b> the short prompt into a richer description (genre, tempo, instruments, mood) before it reaches the music model."),
      P("<b>Objective.</b> Measure whether LLM-expanded prompts produce music that matches the user's <i>original intent</i> better than the raw prompt, "
        "using CLAP text–audio similarity as the metric. Each unit of the syllabus maps to one stage of the pipeline:"),
      B("<b>U1</b> – prompt-expansion template, with zero-shot and few-shot variants compared."),
      B("<b>U2</b> – LLM expansion of short prompts (in-context learning; no fine-tuning)."),
      B("<b>U3</b> – CLAP scoring of text–audio match."),
      B("<b>U4</b> – 10-second clips from MusicGen-small, raw vs expanded, same seed."),
      B("<b>U5</b> – guardrail that blocks “in the style of [named artist]” requests."),
      P("<b>Key evaluation choice.</b> Both clips are scored against the <b>original raw prompt</b>, because scoring an expanded clip against its own "
        "longer expanded text favours the expanded clip through wording overlap. The self-reference score is still recorded, but is not used for conclusions.")]

# ---------------- 3. TOOLS, MODEL, DATASET ----------------
s += [P("2. Tools, Models and Dataset", h1)]
rows = [["Component", "Asset", "Details", "Licence"],
        ["Music generator", "facebook/musicgen-small", "~300M-parameter autoregressive Transformer decoder; EnCodec 32 kHz tokenizer, 4 codebooks at 50 Hz; 500 tokens = 10 s; T5 text conditioning.", "Weights CC-BY-NC 4.0 (non-commercial); code MIT"],
        ["Text–audio scorer", "laion/clap-htsat-unfused", "HTSAT audio encoder + RoBERTa text encoder (BERT-family), 512-D shared space; audio resampled 32 → 48 kHz.", "Apache 2.0"],
        ["Prompt-expansion LLM", "Decoder-only LLM", "Hosted API (Groq / Gemini via secrets) or local Ollama, with a deterministic slot-based fallback so the app never fails.", "Provider terms"],
        ["Semantic-drift scorer", "all-MiniLM-L6-v2", "6-layer BERT-family sentence encoder, 384-D; measures drift between raw and expanded text.", "Apache 2.0"],
        ["Guardrail", "Regex + RapidFuzz", "Pattern rules and fuzzy matching against a curated list of 300+ artists/composers (global, Indian, classical).", "Own code"],
        ["UI / hosting", "Gradio on Hugging Face Spaces", "Free CPU tier (2 vCPU, 16 GB RAM); API keys held in Space secrets only.", "Apache 2.0"],
        ["Benchmark set", "4 fixed prompts", "calm rainy night piano · upbeat workout electronic · sad violin goodbye · festive dhol wedding celebration. 3 conditions (raw, zero-shot, few-shot) × 3 seeds (42, 100, 2024) = 36 runs.", "Own benchmark"]]
s += [table(rows, [30 * mm, 33 * mm, W - 30 * mm - 33 * mm - 33 * mm, 33 * mm]), Spacer(1, 3)]
s += [P(f"Language and framework: Python with PyTorch and Hugging Face <font name='{FBM}'>transformers</font>. Only free tools were used.", small), PageBreak()]

# ---------------- 4. APPROACH + BLOCK DIAGRAM ----------------
s += [P("3. Approach and Block Diagram", h1),
      P("A prompt first passes the guardrail. If allowed, the LLM expands it (zero-shot or few-shot template). MusicGen then generates two 10-second clips "
        "with the <b>same random seed</b>, one from the raw prompt and one from the expanded prompt, so the only difference is the text. "
        "CLAP scores both clips against the raw prompt and the primary delta is the expanded score minus the raw score."),
      figure(os.path.join(FIG, "lifecycle.png"), W,
             "Figure 1: System lifecycle. MusicGen and the expansion LLM are generative models; CLAP is a discriminative (contrastive) scorer."),
      figure(os.path.join(FIG, "model_arch.png"), W * 0.92,
             "Figure 2: MusicGen architecture. T5 text embeddings condition a Transformer decoder through cross-attention; the decoder predicts EnCodec "
             "residual-vector-quantisation (RVQ) tokens, which a convolutional decoder (GAN-trained) turns into a waveform."),
      P("<b>Syllabus coverage.</b> U-I: lifecycle diagram, generative vs discriminative framing, zero-shot vs few-shot comparison, latent-space view (Fig. 4). "
        "U-II: Transformer decoder plus cross-attention and the EnCodec/GAN element (Fig. 2); prompt-based learning only, no fine-tuning (CPU free tier). "
        "U-III: decoder-only LLM plus BERT-family models (CLAP's RoBERTa text encoder, MiniLM) and the audio modality. "
        "U-IV: PyTorch generation of music clips in a Gradio app. U-V: guardrail, hallucination test, licence note and model card."),
      PageBreak()]

# ---------------- 5. KEY CODE ----------------
s += [P("4. Key Code and GitHub Link", h1),
      P(f"Repository: <b>{GH_URL}</b>"),
      P(f"Main files: <font name='{FBM}'>musicgen_runner.py</font> (generation), <font name='{FBM}'>clap_scorer.py</font> (scoring), "
        f"<font name='{FBM}'>llm_client.py</font> (expansion), <font name='{FBM}'>guardrail.py</font> (Unit 5), <font name='{FBM}'>app.py</font> (Gradio UI), "
        f"<font name='{FBM}'>MODEL_CARD.md</font>."),
      P(f"Dual-scoring function from <font name='{FBM}'>clap_scorer.py</font>. Both clips are scored against the user's raw prompt:", h2)]
snippet = '''def score_clip_pair(raw_audio, raw_sr, exp_audio, exp_sr, raw_prompt, exp_prompt):
    # Primary metric: both clips scored against the user's ORIGINAL prompt
    raw_vs_raw = compute_clap_similarity(raw_audio, raw_sr, raw_prompt)
    exp_vs_raw = compute_clap_similarity(exp_audio, exp_sr, raw_prompt)
    primary_delta = exp_vs_raw - raw_vs_raw

    # Self-reference metric: expanded clip vs its own expanded text (biased)
    exp_vs_exp = compute_clap_similarity(exp_audio, exp_sr, exp_prompt)

    return {
        "raw_vs_raw": round(raw_vs_raw, 4),
        "exp_vs_raw": round(exp_vs_raw, 4),
        "exp_vs_exp": round(exp_vs_exp, 4),
        "delta_primary": round(primary_delta, 4),
    }'''
s += [Preformatted(snippet, code),
      P(f"API keys are read from the platform's secret store (<font name='{FBM}'>os.environ</font>) and are never committed. "
        "If the LLM API fails or times out, the app falls back to the deterministic expander, so a live demo cannot crash on an LLM error.")]

# ---------------- 6. SCREENSHOTS ----------------
s += [P("5. Captioned Output Screenshots", h1),
      screenshot("tab1_generate.png", "Figure 3: Tab 1, generating raw vs expanded clips with both audio players and the CLAP scores and delta."),
      figure(os.path.join(FIG, "pca_latent_space.png"), W * 0.9,
             "Figure 4: Tab 3, 2-D PCA of CLAP embeddings (PC1 29.9%, PC2 17.7% of variance). Red circles are audio clips, blue triangles are text prompts. "
             "Audio and text fall in separate regions (a modality gap), and points group mainly by prompt (e.g. eval_1 on the left, eval_2 on the right)."),
      screenshot("tab5_guardrail.png", "Figure 5: Tab 5, an “in the style of [artist]” request is blocked and a descriptive rewrite is suggested; no audio is generated."),
      PageBreak()]

# ---------------- 7. RESULTS ----------------
s += [P("6. Results", h1),
      P("Each prompt was run with raw, zero-shot and few-shot conditions at three seeds on CPU. All scores are CLAP similarity to the "
        "<b>original raw prompt</b>; values below are means over the three seeds. Δ = expanded score − raw score (positive = expansion helped).")]
res = [["Prompt", "Raw", "Zero-shot", "Few-shot", "Δ zero-shot", "Δ few-shot"],
       ["calm rainy night piano", "0.3849", "0.3197", "0.3204", "−0.0652", "−0.0645"],
       ["upbeat workout electronic", "0.2659", "0.2509", "0.2509", "−0.0150", "−0.0150"],
       ["sad violin goodbye", "0.4686", "0.4108", "0.4108", "−0.0578", "−0.0578"],
       ["festive dhol wedding celebration", "0.2610", "0.3335", "0.3335", "+0.0725", "+0.0725"],
       ["<b>Mean of the 4 prompts</b>", "<b>0.3451</b>", "<b>0.3287</b>", "<b>0.3289</b>", "<b>−0.0164</b>", "<b>−0.0162</b>"]]
s += [table(res, [58 * mm, 20 * mm, 22 * mm, 22 * mm, 26 * mm, 26 * mm]), Spacer(1, 6),
      P("<b>Findings</b>", h2),
      B("Expansion <b>lowered</b> intent-alignment for 3 of 4 prompts (piano, electronic, violin) and <b>raised</b> it for 1 (dhol, +0.0725)."),
      B("The average change is small and slightly negative (about −0.016) for both templates, so overall <b>expansion did not improve</b> CLAP alignment with the user's raw prompt."),
      B("Zero-shot and few-shot templates gave near-identical results here (differences ≤ 0.0007 in the means), so the template style mattered much less than the prompt itself."),
      B("The one clear gain was the most culturally specific, least generic prompt (dhol wedding), where the added production and instrument detail may have helped. "
        "With 4 prompts and 3 seeds this is an observation, not a statistically established effect."),
      B("Across all runs, prompt word count correlated weakly and negatively with intent CLAP (Pearson r = −0.235), consistent with longer prompts drifting from the focal idea."),
      B("Self-reference scores (expanded clip vs its own expanded text) run higher than intent scores; the model card puts this inflation at about +0.10 to +0.15, which is why they were not used as the headline metric."),
      P("<b>Guardrail test (Unit 5).</b> 42 test prompts (21 artist-style requests including typos and casual phrasing, 21 benign prompts): "
        "accuracy 100% (42/42), 0 style requests let through, 0 benign prompts blocked. Common-word names pass in ordinary use "
        "(“queen bee buzzing”, “pink sunset”) and are blocked only in style-request context (“sounds like Queen”).")]
s += [PageBreak()]

# ---------------- 8. LIMITATIONS & FAILURE CASES ----------------
s += [P("7. Limitations and Failure Cases", h1),
      P("Documented failure case: expansion lowers the score", h2)]
fc = [["Item", "Value"],
      ["Prompt / condition / seed", "“calm rainy night piano” · zero-shot · seed 2024"],
      ["Raw clip CLAP (vs raw prompt)", "0.4268"],
      ["Expanded clip CLAP (vs raw prompt)", "0.3536"],
      ["Delta", "<b>−0.0732</b> (largest drop observed in the benchmark)"],
      ["Expanded prompt", "“Calm Rainy Night Piano: Acoustic grand piano ballad, gentle reverb, warm woody resonance, contemplative mood, 72 BPM, dynamic swell.”"]]
s += [table(fc, [58 * mm, W - 58 * mm]), Spacer(1, 5),
      P("<b>Likely explanation (hypothesis, not directly measured).</b> The raw prompt was short and focused. The expansion added a style label (“ballad”), "
        "reverb and resonance descriptors, a mood, a tempo and a dynamic instruction, which may have moved the generated audio toward a produced ballad sound "
        "and away from the simple rainy-night piano that CLAP associates with the original words. Note that the expansion did <i>not</i> add extra instruments here, "
        "so the loss is better described as <b>descriptor dilution</b> than instrument clutter."),
      P("<b>Suggested fix.</b> Limit expansions to a few short descriptors, keep the user's own words first, and avoid adding style labels the user did not ask for. "
        "Compare against the raw clip and fall back to it when CLAP does not improve.", body),
      P("Other limitations", h2),
      B("<b>Tempo is only loosely followed.</b> MusicGen-small samples tokens probabilistically, so stated BPM values drift within a 10-second clip."),
      B("<b>No lyrics or intelligible vocals.</b> The model is trained on instrumental music; vocal requests give vocal-like textures, not words."),
      B("<b>Dense prompts can get muddy.</b> The small 300M checkpoint tends to blur when many instruments are requested at once."),
      B("<b>Small benchmark.</b> 4 prompts × 3 seeds on CPU is enough to show a pattern, not to prove one; results vary by seed."),
      B("<b>CLAP is a proxy.</b> It measures text–audio similarity, not musical quality, and it has its own training-data biases."),
      B("<b>Hallucination checks are narrow.</b> The audit covers 20 prompts and a text-similarity measure; it does not judge the audio itself."),
      PageBreak()]

# ---------------- 9. ETHICS ----------------
s += [P("8. Ethical Note: Bias, Privacy and IP Risk", h1),
      B("<b>IP and artist imitation.</b> Music “in the style of” a named artist can mislead listeners and may infringe an artist's rights. The Unit 5 guardrail checks both the user's prompt and the expanded prompt, "
        "and blocks style requests before any audio is generated (42-prompt test: 100% accuracy, 0 leaks)."),
      B("<b>Licensing.</b> MusicGen-small weights are CC-BY-NC 4.0, so this project is for non-commercial, educational use only; the Audiocraft code is MIT. CLAP and MiniLM are Apache 2.0. "
        "Generated clips should be labelled AI-generated."),
      B("<b>Hallucination and drift.</b> On 20 constrained prompts with few-shot expansion, all 20 (100%) kept the user's constraints, "
        "<b>1 of 20 (5%)</b> introduced an artist name, and mean MiniLM similarity between raw and expanded text was 0.835 (minimum 0.707). "
        "Even one leak shows why the guardrail also screens the expanded prompt."),
      B("<b>Cultural bias.</b> MusicGen and CLAP are trained on data skewed toward Western commercial music. Instruments such as the dhol are rendered with recognisable timbre, "
        "but microtonal traditions (Indian ragas, Arabic maqam, gamelan tuning) are approximated in equal-temperament terms. Scores for non-Western prompts should not be read as quality judgments."),
      B("<b>Privacy.</b> The app stores no personal data. Prompts sent to a hosted LLM API are processed under that provider's terms; API keys are kept in platform secrets and are not in the repository.")]

# ---------------- 10. CONCLUSION ----------------
s += [P("9. Conclusion", h1),
      P("The pipeline works end to end: guardrail, LLM expansion, two same-seed MusicGen clips, and CLAP scoring against the user's original prompt. "
        "In this benchmark, LLM prompt expansion did <b>not</b> reliably help: it lowered intent alignment for 3 of 4 prompts (mean change ≈ −0.016) and helped only the dhol prompt (+0.0725). "
        "Zero-shot and few-shot templates behaved almost identically. The results suggest expanding sparingly, keeping the user's words, and checking CLAP against the raw clip before preferring an expansion. "
        "The guardrail blocked every artist-style request in testing, and the main open risks are the small benchmark and the occasional name leaking into an expansion.")]

doc.build(s)
print("built", os.path.join(HERE, OUT))