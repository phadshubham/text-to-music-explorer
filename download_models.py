"""
download_models.py
==================
Pre-downloads and verifies weights for:
1. facebook/musicgen-small
2. laion/clap-htsat-unfused
3. sentence-transformers/all-MiniLM-L6-v2
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download


def download_all():
    print("=" * 60)
    print("PRE-DOWNLOADING REQUIRED MODELS TO HUGGINGFACE CACHE")
    print("=" * 60)

    # 1. MusicGen Small
    print("\n[1/3] Downloading facebook/musicgen-small...")
    path1 = snapshot_download(
        repo_id="facebook/musicgen-small",
        repo_type="model",
        resume_download=True,
    )
    print(f"  -> MusicGen-small ready at: {path1}")

    # 2. CLAP HTSAT
    print("\n[2/3] Downloading laion/clap-htsat-unfused...")
    path2 = snapshot_download(
        repo_id="laion/clap-htsat-unfused",
        repo_type="model",
        resume_download=True,
    )
    print(f"  -> CLAP ready at: {path2}")

    # 3. MiniLM
    print("\n[3/3] Downloading sentence-transformers/all-MiniLM-L6-v2...")
    path3 = snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        repo_type="model",
        resume_download=True,
    )
    print(f"  -> MiniLM ready at: {path3}")

    print("\n>>> ALL MODELS DOWNLOADED AND VERIFIED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    download_all()
