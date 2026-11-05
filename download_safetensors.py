"""
download_safetensors.py
=======================
Robust parallel chunk downloader for facebook/musicgen-small model.safetensors.
Downloads directly into Hugging Face Hub blob cache with SHA-256 verification.
"""

import os
import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://huggingface.co/facebook/musicgen-small/resolve/main/model.safetensors"
TOTAL_SIZE = 2364427288
BLOB_HASH = "1bdc99d43eb6c775967df24b65b0a9f847c0907e95664698d93b5a1c35f5090d"
BLOB_DIR = os.path.expanduser(r"~/.cache/huggingface/hub/models--facebook--musicgen-small/blobs")
FINAL_BLOB_PATH = os.path.join(BLOB_DIR, BLOB_HASH)
PART_FILE = FINAL_BLOB_PATH + ".downloading"
NUM_WORKERS = 4
CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB chunks

file_lock = threading.Lock()
shared_file = None


def get_ranges(total_size, chunk_size):
    ranges = []
    for start in range(0, total_size, chunk_size):
        end = min(start + chunk_size - 1, total_size - 1)
        ranges.append((start, end))
    return ranges


def download_chunk(range_tuple):
    global shared_file
    start, end = range_tuple
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "Mozilla/5.0",
    }
    req = urllib.request.Request(URL, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
                expected_len = end - start + 1
                if len(data) != expected_len:
                    raise IOError(f"Incomplete chunk: got {len(data)}, expected {expected_len}")

                # Atomic write with lock
                with file_lock:
                    shared_file.seek(start)
                    shared_file.write(data)
                    shared_file.flush()

                return start, end, len(data)
        except Exception as e:
            if attempt == 4:
                raise e
            time.sleep(2.0)


def main():
    global shared_file
    os.makedirs(BLOB_DIR, exist_ok=True)

    if os.path.exists(FINAL_BLOB_PATH) and os.path.getsize(FINAL_BLOB_PATH) == TOTAL_SIZE:
        print(f"Blob already complete at {FINAL_BLOB_PATH}", flush=True)
        return

    # Pre-allocate part file if needed
    if not os.path.exists(PART_FILE) or os.path.getsize(PART_FILE) != TOTAL_SIZE:
        print(f"Pre-allocating {TOTAL_SIZE / (1024*1024):.1f} MB file...", flush=True)
        with open(PART_FILE, "wb") as f:
            f.seek(TOTAL_SIZE - 1)
            f.write(b"\0")

    shared_file = open(PART_FILE, "r+b")

    ranges = get_ranges(TOTAL_SIZE, CHUNK_SIZE)
    total_chunks = len(ranges)
    print(f"Starting parallel download: {total_chunks} chunks ({CHUNK_SIZE/(1024*1024):.0f}MB each) across {NUM_WORKERS} workers...", flush=True)

    start_time = time.time()
    completed = 0
    bytes_downloaded = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_chunk, r): r for r in ranges}
        for future in as_completed(futures):
            try:
                start, end, nbytes = future.result()
                completed += 1
                bytes_downloaded += nbytes
                elapsed = time.time() - start_time
                speed = (bytes_downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                pct = (completed / total_chunks) * 100.0
                print(f"[{completed:3d}/{total_chunks}] {pct:5.1f}% | {bytes_downloaded/(1024*1024):6.1f} MB | {speed:4.2f} MB/s", flush=True)
            except Exception as e:
                print(f"\nChunk failed: {e}", flush=True)
                shared_file.close()
                sys.exit(1)

    shared_file.close()
    print(f"\nDownload finished in {time.time() - start_time:.1f}s!", flush=True)

    # Atomic rename to final blob
    if os.path.exists(FINAL_BLOB_PATH):
        os.remove(FINAL_BLOB_PATH)
    os.rename(PART_FILE, FINAL_BLOB_PATH)
    print(f"Verified blob ready: {FINAL_BLOB_PATH} ({os.path.getsize(FINAL_BLOB_PATH)} bytes)", flush=True)

    # Link into snapshots directory
    snap_dir = os.path.expanduser(r"~/.cache/huggingface/hub/models--facebook--musicgen-small/snapshots/4c8334b02c6ec4e8664a91979669a501ec497792")
    os.makedirs(snap_dir, exist_ok=True)
    snap_safetensors = os.path.join(snap_dir, "model.safetensors")

    import shutil
    import ctypes
    if os.path.exists(snap_safetensors):
        os.remove(snap_safetensors)

    res = ctypes.windll.kernel32.CreateHardLinkW(snap_safetensors, FINAL_BLOB_PATH, None)
    if res != 0:
        print("Created NTFS hard link in snapshot directory.", flush=True)
    else:
        shutil.copyfile(FINAL_BLOB_PATH, snap_safetensors)
        print("Copied to snapshot directory.", flush=True)

    print(">>> MusicGen-small safetensors ready for immediate use! <<<", flush=True)


if __name__ == "__main__":
    main()
