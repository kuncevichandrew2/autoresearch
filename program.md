# autoresearch (Kaggle P100 Edition)

This is an experiment to have the LLM do its own research, running on a **Kaggle P100 GPU** (15GB VRAM).

## Hardware & Constraints

- **GPU**: NVIDIA Tesla P100 (16GB VRAM, NO bf16 support, fp16 only, no Flash Attention 3)
- **Attention**: PyTorch SDPA (no sliding window support)
- **Dataset**: TinyStories (public, small)
- **Model**: ~5M params (DEPTH=4, seq_len=256, vocab=2048)
- **Repo**: `https://github.com/kuncevichandrew2/autoresearch.git`
- **Kaggle user**: `andrewk444`

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Data preparation**: Data is prepared on Kaggle at runtime. TinyStories is downloaded via HuggingFace `datasets` library and saved as parquet shards. The Kaggle kernel handles this automatically.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a **Kaggle P100 GPU** (15GB VRAM). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Experiments are launched remotely via Kaggle kernels using the Kaggle CLI (`kaggle kernels push`).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Keep in mind P100 constraints: 16GB VRAM, no Flash Attention 3, SDPA only (no sliding window), NO bf16 (fp16 only), Pascal architecture (sm_60).

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.
- Use Flash Attention or any kernel that requires sm_75+ compute capability beyond what P100 supports.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this (P100 example):

```
---
val_bpb:          0.686159
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     901.0
mfu_percent:      ~1-2%
total_tokens_M:   ~16
num_steps:        ~1000
num_params_M:     ~5.2
depth:            4
```

Note that the P100 is much slower than H100, so MFU% will be low (measured against H100 baseline). Peak VRAM should stay well under 15GB. Results are retrieved from Kaggle kernel output via `kaggle kernels output`.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit and push to `origin` (your GitHub fork)
4. Update the Kaggle kernel script to clone from your fork, then push the kernel: `kaggle kernels push -p kaggle-kernel/`
5. Poll kernel status: `kaggle kernels status andrewk444/autoresearch-t4` (wait for completion)
6. Pull results: `kaggle kernels output andrewk444/autoresearch-t4 -p /tmp/kaggle-output/`
7. Read out the results from the kernel output log. Look for `val_bpb:` and `peak_vram_mb:`.
8. If the output is empty or shows errors, the run crashed. Read the log to diagnose and attempt a fix.
9. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
10. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
11. If val_bpb is equal or worse, you git reset back to where you started and force-push to revert the remote

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each Kaggle kernel has a 12-hour limit, but training itself should take ~5 minutes (+ startup overhead for cloning, installing deps, downloading data). If a kernel exceeds 20 minutes total, cancel it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM on P100's 15GB, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken (e.g. OOM because the model is too large for P100), just skip it, log "crash" as the status in the tsv, and move on.

**P100-specific considerations**:
- P100 has 15GB VRAM — be conservative with model size increases
- P100 has NO bf16 support — all code uses fp16 instead
- No Flash Attention 3 — only PyTorch SDPA
- torch.compile works but compilation takes longer on P100
- Kaggle kernels have internet access, so the repo clone + data download happens each run

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each Kaggle kernel experiment takes ~10-15 minutes (including startup overhead), so you can run approx 4-6 per hour, for a total of about 30-50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
