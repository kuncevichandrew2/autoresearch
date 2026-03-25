#!/usr/bin/env python3
"""
Autoresearch on Kaggle P100 GPU.
Clones the repo, prepares TinyStories data, and runs training.
"""

import subprocess
import os
import sys

def run(cmd, **kwargs):
    print(f"\n{'='*60}")
    print(f">>> {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
    return result

# Show GPU info
run("nvidia-smi")

# Install dependencies
run("pip install uv")

# Clone the repo
os.chdir("/kaggle/working")
if not os.path.exists("autoresearch"):
    run("git clone https://github.com/kuncevichandrew2/autoresearch.git")
os.chdir("/kaggle/working/autoresearch")

# Install project dependencies with uv
run("uv pip install --system -r pyproject.toml")

# Download TinyStories and prepare data shards
# The prepare.py expects parquet shards in ~/.cache/autoresearch/data/
# TinyStories is available as parquet on HuggingFace
print("\n" + "="*60)
print("Preparing TinyStories data...")
print("="*60)

run("pip install datasets")

# Download and split TinyStories into train/val parquet shards
import importlib
prepare_script = """
import os
from datasets import load_dataset

cache_dir = os.path.expanduser("~/.cache/autoresearch/data")
os.makedirs(cache_dir, exist_ok=True)

train_path = os.path.join(cache_dir, "shard_00000.parquet")
# For val, we use the validation split as shard_00000 (since MAX_SHARD=0, VAL_SHARD=0)
# But we need separate train and val. The lite version uses MAX_SHARD=0 and VAL_SHARD=6542
# So val shard is shard_06542.parquet
val_path = os.path.join(cache_dir, "shard_06542.parquet")

if os.path.exists(train_path) and os.path.exists(val_path):
    print("Data shards already exist, skipping download.")
else:
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories")

    # Save train split
    print(f"Saving train shard to {train_path}...")
    ds["train"].to_parquet(train_path)

    # Save validation split
    print(f"Saving val shard to {val_path}...")
    ds["validation"].to_parquet(val_path)

    print("Data preparation complete!")
"""
exec(prepare_script)

# Train tokenizer
run("python prepare.py --num-shards 0")

# Run training
print("\n" + "="*60)
print("Starting autoresearch training on P100...")
print("="*60)
run("python train.py")

print("\n" + "="*60)
print("AUTORESEARCH COMPLETE!")
print("="*60)
