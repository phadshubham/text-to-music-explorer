"""
Unit 1: Prompt-Expansion Templates
Defines slot-based ZERO_SHOT and FEW_SHOT prompt templates.
Rules enforced:
- Output ONE single line of at most 40 words
- Preserve every user-specified constraint
- Strictly forbid artist, band, or song names
- Forbid lyrics or sung vocal requests (MusicGen limitation)
- Zero preamble or conversational meta-text
- Example pairs in FEW_SHOT MUST NOT overlap the 4 evaluation prompts:
  ("calm rainy night piano", "upbeat workout electronic", "sad violin goodbye", "festive dhol wedding celebration")
"""

SYSTEM_PROMPT_INSTRUCTIONS = (
    "You are an expert music producer and sound designer formulating detailed generation prompts for MusicGen.\n"
    "Your objective: Expand short, ambiguous user prompts into rich, acoustically coherent musical descriptions.\n\n"
    "SLOTS TO SPECIFY:\n"
    "1. Genre / Sub-genre (e.g., Synthwave, Cinematic Folk, Post-Rock)\n"
    "2. Mood / Emotional tone (e.g., contemplative, euphoric, tense)\n"
    "3. 2-4 Concrete Instruments (e.g., acoustic cello, analog synthesizer, brushed snare)\n"
    "4. Tempo in BPM (e.g., 90 BPM, 128 BPM)\n"
    "5. Dynamics & Structure (e.g., gradual swell, sudden drop, steady pulse)\n"
    "6. Production & Acoustic Texture (e.g., warm analog tape saturation, wide stereo reverb)\n\n"
    "STRICT GENERATION RULES:\n"
    "- Output exactly ONE line of text.\n"
    "- Output must NOT exceed 40 words.\n"
    "- Keep EVERY specific instrument, tempo, or genre constraint the user requested.\n"
    "- NEVER include any artist, band, composer, or song title names.\n"
    "- NEVER include lyrics, vocals, or singing requests (the model cannot produce sung lyrics).\n"
    "- No preamble, no explanation, no quotation marks, no conversational filler."
)

# Versioned Zero-Shot Template
ZERO_SHOT_PROMPT_TEMPLATE = (
    SYSTEM_PROMPT_INSTRUCTIONS
    + "\n\nUser Input: {user_prompt}\n"
    + "Expanded MusicGen Prompt:"
)

# 5 Curated Few-Shot Example Pairs (Completely distinct from the 4 eval prompts)
FEW_SHOT_EXAMPLES = [
    {
        "input": "dark cyberpunk chase",
        "output": "Dark synthwave cyberpunk soundtrack, aggressive sawtooth bass, pulsating analog synthesizer, punchy 80s drums, fast 135 BPM, intense driving crescendo, distorted industrial saturation.",
    },
    {
        "input": "relaxing morning acoustic guitar",
        "output": "Warm folk acoustic guitar ballad, gentle fingerpicking, soft wooden percussion, upright bass, moderate 84 BPM, intimate dynamic arc, organic room reverberation.",
    },
    {
        "input": "epic orchestral space adventure",
        "output": "Cinematic sci-fi orchestral theme, heroic French horns, soaring string section, resonant timpani, brisk 110 BPM, dramatic brass swell, wide atmospheric reverb.",
    },
    {
        "input": "lazy sunday jazz cafe",
        "output": "Lo-fi mellow jazz, smooth muted trumpet, warm Rhodes electric piano, brushed snare, relaxed 75 BPM, gentle swing rhythm, vintage tape warmth.",
    },
    {
        "input": "mystical desert flute journey",
        "output": "Middle Eastern ambient instrumental, evocative wooden bansuri flute, subtle oud strums, frame drum, steady 90 BPM, hypnotic undulating groove, expansive desert space.",
    },
]

def _build_few_shot_prompt() -> str:
    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_str += f"\nUser Input: {ex['input']}\nExpanded MusicGen Prompt: {ex['output']}\n"
    return (
        SYSTEM_PROMPT_INSTRUCTIONS
        + "\n\nFEW-SHOT EXAMPLES:\n"
        + examples_str
        + "\nUser Input: {user_prompt}\n"
        + "Expanded MusicGen Prompt:"
    )

FEW_SHOT_PROMPT_TEMPLATE = _build_few_shot_prompt()

# Semantic Version Constants
TEMPLATE_VERSION = "1.0.0"
ZERO_SHOT = "Zero-Shot"
FEW_SHOT = "Few-Shot"


def format_expansion_prompt(user_prompt: str, technique: str = FEW_SHOT) -> str:
    """Format user prompt into the chosen template string."""
    clean_input = user_prompt.strip()
    if technique == ZERO_SHOT:
        return ZERO_SHOT_PROMPT_TEMPLATE.format(user_prompt=clean_input)
    return FEW_SHOT_PROMPT_TEMPLATE.format(user_prompt=clean_input)


# Backward-compatible aliases
format_prompt = format_expansion_prompt
ZERO_SHOT_V1 = ZERO_SHOT_PROMPT_TEMPLATE
FEW_SHOT_V1 = FEW_SHOT_PROMPT_TEMPLATE


def get_template(technique: str = FEW_SHOT) -> str:
    """Returns the raw template string."""
    if technique == ZERO_SHOT:
        return ZERO_SHOT_PROMPT_TEMPLATE
    return FEW_SHOT_PROMPT_TEMPLATE
