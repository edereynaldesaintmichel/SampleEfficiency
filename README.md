# ExtremeCompression

Sample-efficiency maxxing: how low a val loss can a ~10M-param transformer reach
on a small fixed corpus, with effectively unlimited compute?

## Protocol

- **Corpora**:
  - *shakespeare* (primary): Complete Works (Gutenberg pg100, ~5.4 MB),
    **work-level split** — val = Julius Caesar + As You Like It, test = Macbeth +
    Twelfth Night, train = the other 39 works. Generalization here means
    predicting unseen plays. Select checkpoints/hypers on val; touch test ONCE
    at the end (`eval.py --split test`).
  - *enwik8*: first 10 MB, last 10% held out as a *contiguous* val chunk
    (random windows would leak at byte level).
- **Metric**: bits per byte (bpb) over the val text — comparable across tokenizers,
  since bits are always divided by the same underlying byte count.
- **Tokenizers**: byte-level (vocab=256) is the baseline; BPE variants are trained
  on the train split only.
- **Eval**: sliding window — every val token scored exactly once, predicted with
  the longest context that fits. Interim evals use stride=block_size (fast);
  official numbers use stride=128 via `eval.py`.
- **Model budget**: ~10M parameters. Default: 8 layers, d=320, 5 heads, ~9.9M.
  Stack: pre-norm RMSNorm, RoPE, QK-norm, SDPA, ReLU² MLP, tied embeddings,
  zero-init projections, logit softcap, dropout everywhere.
- **Optimizer**: Muon (hidden matrices) + AdamW (embeddings/norms), cosine
  schedule, EMA of weights; best checkpoint selected by EMA val bpb.

## Usage

```bash
source .venv/bin/activate

# data (byte-level baselines)
python prepare_data.py --dataset shakespeare --vocab_size 256
python prepare_data.py --dataset enwik8 --megabytes 10 --vocab_size 256

# vocab sweep variants (BPE trained on train split only)
python prepare_data.py --megabytes 10 --vocab_size 512
python prepare_data.py --megabytes 10 --vocab_size 1024
python prepare_data.py --megabytes 10 --vocab_size 2048
python prepare_data.py --megabytes 10 --vocab_size 4096

# train (on CUDA)
python train.py --data data/enwik8_10mb_v256 --run_name baseline

# local Mac/MPS runs: attention-matrix dropout forces unfused attention and
# OOMs — disable it (and shrink the batch) for smoke tests only
python train.py --data data/enwik8_10mb_v256 --run_name smoke \
  --attn_dropout 0 --batch_size 16 --block_size 512 --steps 60

# official eval of the best checkpoint
python eval.py runs/baseline/best.pt --data data/enwik8_10mb_v256 --stride 128
```

Runs log to `runs/<name>/log.csv` and save `best.pt` / `latest.pt`.

## Knobs that matter in this regime

Roughly in order of expected impact:

1. `--dropout` (sweep 0.1–0.5) and `--weight_decay` — the core lever against
   overfitting; train far past train-loss saturation.
2. `--steps` — hundreds of epochs is intended, watch EMA val bpb for the floor.
3. Vocab size (via `prepare_data.py`) — measured in bpb, fair fight.
4. `--muon_lr` / `--adam_lr` / schedule.
5. Ensembling + distillation across seeds (`--seed`) — not scripted yet,
   biggest planned unlimited-compute exploit.

## Reference points (enwik8, full 100MB, char/byte-level)

- Transformer-XL (41M params): ~0.99 bpc
- Small transformers typically land 1.0–1.2 bpc
- With only 9MB of training data, expect noticeably higher — the interesting
  number is the delta between your tricks, not the absolute.
