#!/usr/bin/env python3
"""
One-time setup kernel for autoresearch.
Downloads TinyStories, creates parquet shards, trains BPE tokenizer.
Output saved to /kaggle/working/ — will be uploaded as a Kaggle dataset.

Run this once. The training kernel mounts the resulting dataset.
"""

import subprocess
import os
import shutil

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

OUTPUT_DIR = "/kaggle/working/autoresearch-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Install dependencies
run("pip install datasets pyarrow rustbpe tiktoken requests -q")

# ---- Step 1: Download TinyStories and save as parquet shards ----
print("\n=== Downloading TinyStories ===")

import os
from datasets import load_dataset

data_dir = os.path.join(OUTPUT_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

train_path = os.path.join(data_dir, "shard_00000.parquet")
val_path   = os.path.join(data_dir, "shard_06542.parquet")

print("Downloading TinyStories from HuggingFace...")
ds = load_dataset("roneneldan/TinyStories")

print(f"Saving train shard ({len(ds['train'])} rows)...")
ds["train"].to_parquet(train_path)

print(f"Saving val shard ({len(ds['validation'])} rows)...")
ds["validation"].to_parquet(val_path)

print("Parquet shards saved.")

# ---- Step 2: Clone repo and train tokenizer ----
print("\n=== Training BPE tokenizer ===")

os.chdir("/kaggle/working")
if not os.path.exists("autoresearch"):
    run("git clone https://github.com/kuncevichandrew2/autoresearch.git")
os.chdir("/kaggle/working/autoresearch")
run("git checkout autoresearch/mar25")

# Point prepare.py cache to our output dir
os.environ["HOME_OVERRIDE"] = OUTPUT_DIR  # prepare.py uses ~/ for cache
# Symlink the data into the expected cache path
cache_dir = os.path.expanduser("~/.cache/autoresearch")
os.makedirs(cache_dir, exist_ok=True)
expected_data_dir = os.path.join(cache_dir, "data")
if not os.path.exists(expected_data_dir):
    os.symlink(data_dir, expected_data_dir)

run("pip install rustbpe tiktoken pyarrow requests -q")
run("python prepare.py --num-shards 0")

# Copy tokenizer to output
tokenizer_src = os.path.expanduser("~/.cache/autoresearch/tokenizer")
tokenizer_dst = os.path.join(OUTPUT_DIR, "tokenizer")
if os.path.exists(tokenizer_dst):
    shutil.rmtree(tokenizer_dst)
shutil.copytree(tokenizer_src, tokenizer_dst)

print("\n=== Setup complete ===")
print(f"Output at: {OUTPUT_DIR}")
print("Contents:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        path = os.path.join(root, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {path}  ({size_mb:.1f} MB)")
