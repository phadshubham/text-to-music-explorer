"""
scripts/download_model.py
=========================
Downloads only the required safetensors and config files for facebook/musicgen-small.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "facebook/musicgen-small"
FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "spiece.model",
    "special_tokens_map.json",
    "model.safetensors",
]

print(f"Downloading required files for {REPO_ID}...")
for f in FILES:
    print(f"Fetching {f}...")
    p = hf_hub_download(repo_id=REPO_ID, filename=f, resume_download=True)
    print(f"  -> Saved {f} ({os.path.getsize(p) / 1024 / 1024:.1f} MB)")

print("All required files downloaded successfully!")

