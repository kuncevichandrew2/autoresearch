# Kaggle Adapter

Runs autoresearch on Kaggle's free GPU notebooks (T4 or P100).

## Usage

1. Create a Kaggle notebook with GPU accelerator enabled.
2. Upload the autoresearch repo files or clone from GitHub.
3. Run:

```python
from adapters.kaggle.adapter import setup, run_prepare, run_train
setup()
run_prepare(num_shards=4)
run_train()
```

## Configuration

- Edit `.env` in this directory to set `WANDB_API_KEY` and `KAGGLE_API_TOKEN`.
- For Kaggle T4 (16GB VRAM), reduce `DEVICE_BATCH_SIZE` to 16-32 and `DEPTH` to 4 in `train.py`.
- `WANDB_API_KEY` can also be set via Kaggle Secrets.

## Notes

- Kaggle has a 12-hour session limit.
- T4/P100 GPUs use the `kernels-community/flash-attn3` fallback automatically.
