#!/usr/bin/env python3
"""
Autoresearch training kernel on Kaggle P100 GPU.
Mounts pre-prepared dataset (autoresearch-data) — no download/tokenizer step needed.
"""

import subprocess
import os
import shutil

def run(cmd):
    print(f"\n{'='*60}")
    print(f">>> {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
    return result

# Show GPU info
run("nvidia-smi")

# Install PyTorch with CUDA 11.8 — required for P100 (sm_60, Pascal)
# Kaggle's default PyTorch only supports sm_70+, so we override it
run("pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118 -q")

# Install other dependencies (no torch — already installed above)
run("pip install rustbpe tiktoken pyarrow requests matplotlib pandas numpy -q")

# Clone the repo (experiment branch)
os.chdir("/kaggle/working")
if not os.path.exists("autoresearch"):
    run("git clone https://github.com/kuncevichandrew2/autoresearch.git")
os.chdir("/kaggle/working/autoresearch")
run("git checkout autoresearch/mar25")

# Mount pre-prepared data from dataset (skips all download/tokenizer prep)
DATA_SRC = "/kaggle/input/autoresearch-data"
CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")
os.makedirs(CACHE_DIR, exist_ok=True)

data_dst      = os.path.join(CACHE_DIR, "data")
tokenizer_dst = os.path.join(CACHE_DIR, "tokenizer")

if not os.path.exists(data_dst):
    print(f"\nLinking data: {DATA_SRC}/data -> {data_dst}")
    os.symlink(os.path.join(DATA_SRC, "data"), data_dst)

if not os.path.exists(tokenizer_dst):
    print(f"\nCopying tokenizer: {DATA_SRC}/tokenizer -> {tokenizer_dst}")
    shutil.copytree(os.path.join(DATA_SRC, "tokenizer"), tokenizer_dst)

print(f"\nCache ready at {CACHE_DIR}")

# Run training
print("\n" + "="*60)
print("Starting autoresearch training on P100...")
print("="*60)
run("python train.py")

print("\n" + "="*60)
print("AUTORESEARCH COMPLETE!")
print("="*60)
