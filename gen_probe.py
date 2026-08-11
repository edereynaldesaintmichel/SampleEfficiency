"""Probe: is a model's generalization ability itself generalizable?

Trains N seeds of the transformer on a small fixed subset of train.bin
(default: first 1MB), 1000 steps each, small weight decay, EMA weights used
for all measurements. For each seed measures:

  train_bpb  - bpb on a slice the model trained on (first 100KB of the subset)
  proxy_bpb  - bpb on the next 100KB of train.bin (same distribution, never seen)
  val_bpb    - bpb on the real val split

then reports Pearson/Spearman correlation matrices across seeds, and pairwise
spectral norms of hidden-matrix differences (embeddings/norms excluded) with a
row-permutation baseline to judge whether the seeds landed in the same region
of the loss landscape or are as far apart as independent random models.

Size arguments (--train_bytes etc.) are bytes of underlying text; for BPE
datasets they are converted to token counts via the split's tokens/bytes
ratio, and bpb for train.bin slices uses byte counts estimated the same way.

Example:
  python gen_probe.py --data data/shakespeare_v2048 --run_name probe10
"""

import argparse
import contextlib
import copy
import json
import math
import os
import time

import numpy as np
import torch

from evaluation import evaluate_bpb
from model import GPT, GPTConfig
from muon import Muon
from train import EMA, pick_device


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/shakespeare_v2048")
    ap.add_argument("--run_name", type=str, default="gen_probe")
    ap.add_argument("--out_dir", type=str, default="runs")
    # experiment
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--train_bytes", type=int, default=1_000_000)
    ap.add_argument("--proxy_bytes", type=int, default=100_000)
    ap.add_argument("--train_eval_bytes", type=int, default=100_000)
    ap.add_argument("--val_limit_bytes", type=int, default=-1, help="-1 = full val split")
    # model (same defaults as train.py)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=5)
    ap.add_argument("--n_embd", type=int, default=320)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--attn_dropout", type=float, default=-1.0)
    ap.add_argument("--softcap", type=float, default=30.0)
    # optimization
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--adam_lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--min_lr_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    # eval / misc
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast (CUDA only)")
    ap.add_argument("--no_resume", action="store_true",
                    help="retrain even if a per-seed checkpoint exists in the run dir")
    return ap.parse_args()


def hidden_names(model: GPT) -> list[str]:
    return [n for n, p in model.named_parameters()
            if p.ndim == 2 and "embed" not in n and "lm_head" not in n]


def train_one(seed: int, args, cfg: GPTConfig, train_sub: np.ndarray, device: str):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device == "cuda" and not args.no_amp else contextlib.nullcontext())
    model = GPT(cfg).to(device)

    hidden = [p for n, p in model.named_parameters()
              if p.ndim == 2 and "embed" not in n and "lm_head" not in n]
    rest_decay = [p for n, p in model.named_parameters() if "embed" in n]
    rest_nodecay = [p for n, p in model.named_parameters()
                    if p.ndim < 2 and "embed" not in n]
    muon = Muon(hidden, lr=args.muon_lr, momentum=args.muon_momentum,
                weight_decay=args.weight_decay)
    adamw = torch.optim.AdamW(
        [{"params": rest_decay, "weight_decay": args.weight_decay},
         {"params": rest_nodecay, "weight_decay": 0.0}],
        lr=args.adam_lr, betas=(0.9, 0.95),
    )
    optimizers = [muon, adamw]
    peak_lrs = [args.muon_lr, args.adam_lr]

    def lr_mult(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t))

    ema = EMA(model, args.ema_decay)
    running_loss = None
    t0 = time.time()
    for step in range(args.steps):
        m = lr_mult(step)
        for opt, peak in zip(optimizers, peak_lrs):
            for g in opt.param_groups:
                g["lr"] = m * peak
        ix = rng.integers(0, len(train_sub) - args.block_size - 1, size=args.batch_size)
        x = torch.stack([torch.from_numpy(train_sub[i:i + args.block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(train_sub[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix]).to(device)
        with amp_ctx:
            _, loss = model(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
        ema.update(model, step)
        l = loss.item()
        running_loss = l if running_loss is None else 0.99 * running_loss + 0.01 * l
        if step % args.log_interval == 0 or step == args.steps - 1:
            print(f"  seed {seed}  step {step:4d}  loss {running_loss:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    eval_model = copy.deepcopy(model)
    ema.copy_to(eval_model)
    return eval_model, running_loss


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a))
    return ranks


def corr_matrices(cols: dict[str, np.ndarray], n_perm: int = 10000):
    names = list(cols)
    X = np.stack([cols[n] for n in names])
    pearson = np.corrcoef(X)
    spearman = np.corrcoef(np.stack([rankdata(x) for x in X]))
    # permutation p-values for Pearson (two-sided)
    rng = np.random.default_rng(0)
    pvals = np.ones((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            obs = abs(pearson[i, j])
            count = 0
            b = X[j].copy()
            for _ in range(n_perm):
                rng.shuffle(b)
                if abs(np.corrcoef(X[i], b)[0, 1]) >= obs - 1e-12:
                    count += 1
            pvals[i, j] = pvals[j, i] = count / n_perm
    return names, pearson, spearman, pvals


def fmt_matrix(names: list[str], M: np.ndarray) -> str:
    w = max(len(n) for n in names) + 2
    lines = [" " * w + "".join(f"{n:>{w}}" for n in names)]
    for i, n in enumerate(names):
        lines.append(f"{n:>{w}}" + "".join(f"{M[i, j]:>{w}.3f}" for j in range(len(names))))
    return "\n".join(lines)


def spec(A: torch.Tensor) -> float:
    return torch.linalg.matrix_norm(A.float(), ord=2).item()


def spectral_analysis(hiddens: list[dict[str, torch.Tensor]], seeds: list[int]):
    """Pairwise spectral distance between models' hidden matrices, plus a
    'random' baseline where one model's hidden units are permuted (destroys
    alignment while preserving the weight distribution)."""
    n = len(hiddens)
    names = list(hiddens[0])
    g = torch.Generator().manual_seed(0)
    ratio = np.zeros((n, n))       # ||Wi-Wj|| / mean(||Wi||,||Wj||), mean over matrices
    base_ratio = np.zeros((n, n))  # same, vs row-permuted Wj
    cos = np.eye(n)                # cosine sim of concatenated flattened hidden weights
    base_cos = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r_list, rb_list = [], []
            dots = norms_i = norms_j = dots_b = 0.0
            for name in names:
                Wi, Wj = hiddens[i][name].float(), hiddens[j][name].float()
                perm = torch.randperm(Wj.size(0), generator=g)
                Wjp = Wj[perm]
                si, sj = spec(Wi), spec(Wj)
                denom = 0.5 * (si + sj) + 1e-12
                r_list.append(spec(Wi - Wj) / denom)
                rb_list.append(spec(Wi - Wjp) / denom)
                dots += (Wi * Wj).sum().item()
                dots_b += (Wi * Wjp).sum().item()
                norms_i += Wi.square().sum().item()
                norms_j += Wj.square().sum().item()
            ratio[i, j] = ratio[j, i] = float(np.mean(r_list))
            base_ratio[i, j] = base_ratio[j, i] = float(np.mean(rb_list))
            denom = math.sqrt(norms_i * norms_j) + 1e-12
            cos[i, j] = cos[j, i] = dots / denom
            base_cos[i, j] = base_cos[j, i] = dots_b / denom
    iu = np.triu_indices(n, 1)
    summary = {
        "mean_pair_ratio": float(ratio[iu].mean()),
        "mean_baseline_ratio": float(base_ratio[iu].mean()),
        "mean_pair_cos": float(cos[iu].mean()),
        "mean_baseline_cos": float(base_cos[iu].mean()),
        "n_matrices": len(names),
    }
    return ratio, base_ratio, cos, base_cos, summary


def main():
    args = get_args()
    device = pick_device(args.device)
    torch.set_float32_matmul_precision("high")  # TF32 on CUDA
    if device == "mps" and (args.attn_dropout if args.attn_dropout >= 0 else args.dropout) > 0:
        print("MPS: forcing attn_dropout=0 (fused SDPA)")
        args.attn_dropout = 0.0

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    train_ids = np.fromfile(os.path.join(args.data, "train.bin"), dtype=dtype)
    val_np = np.fromfile(os.path.join(args.data, "val.bin"), dtype=dtype)

    # size args are bytes of text; convert to token counts via the split ratios
    bpt_train = meta["train_bytes"] / meta["train_tokens"]
    bpt_val = meta["val_bytes"] / meta["val_tokens"]
    n_train = round(args.train_bytes / bpt_train)
    n_proxy = round(args.proxy_bytes / bpt_train)
    n_train_eval = round(args.train_eval_bytes / bpt_train)
    assert len(train_ids) >= n_train + n_proxy
    if args.val_limit_bytes > 0:
        val_np = val_np[:round(args.val_limit_bytes / bpt_val)]
        val_bytes = len(val_np) * bpt_val
    else:
        val_bytes = meta["val_bytes"]

    train_sub = train_ids[:n_train]
    train_eval_ids = torch.from_numpy(train_sub[:n_train_eval].astype(np.int64))
    proxy_ids = torch.from_numpy(
        train_ids[n_train:n_train + n_proxy].astype(np.int64))
    val_ids = torch.from_numpy(val_np.astype(np.int64))

    cfg = GPTConfig(
        vocab_size=meta["vocab_size"], block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, attn_dropout=args.attn_dropout, softcap=args.softcap,
    )
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    seeds = [args.seed0 + k for k in range(args.n_seeds)]
    print(f"device={device}  seeds={seeds}")
    print(f"vocab={meta['vocab_size']}  bytes/token train={bpt_train:.3f}")
    print(f"train subset: {len(train_sub):,} tok (~{args.train_bytes:,}B)  "
          f"proxy: {len(proxy_ids):,} tok (~{args.proxy_bytes:,}B)  "
          f"val: {len(val_ids):,} tok (~{val_bytes:,.0f}B)  "
          f"steps={args.steps}  wd={args.weight_decay}")

    metrics: list[dict] = []
    hiddens: list[dict[str, torch.Tensor]] = []
    for seed in seeds:
        ckpt_path = os.path.join(run_dir, f"seed{seed}.pt")
        if os.path.exists(ckpt_path) and not args.no_resume:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            print(f"seed {seed}: loaded cached result")
            metrics.append(ck["metrics"])
            hiddens.append(ck["hidden"])
            continue
        print(f"=== training seed {seed} ===", flush=True)
        eval_model, running_loss = train_one(seed, args, cfg, train_sub, device)
        m = {
            "seed": seed,
            "final_running_loss": running_loss,
            "train_bpb": evaluate_bpb(eval_model, train_eval_ids, len(train_eval_ids) * bpt_train,
                                      args.block_size, args.eval_batch_size, device),
            "proxy_bpb": evaluate_bpb(eval_model, proxy_ids, len(proxy_ids) * bpt_train,
                                      args.block_size, args.eval_batch_size, device),
            "val_bpb": evaluate_bpb(eval_model, val_ids, val_bytes,
                                    args.block_size, args.eval_batch_size, device),
        }
        hid = {n: p.detach().float().cpu().clone()
               for n, p in eval_model.named_parameters()
               if p.ndim == 2 and "embed" not in n and "lm_head" not in n}
        torch.save({"metrics": m, "hidden": hid}, ckpt_path)
        print(f"seed {seed}:  train {m['train_bpb']:.4f}  proxy {m['proxy_bpb']:.4f}  "
              f"val {m['val_bpb']:.4f} bpb", flush=True)
        metrics.append(m)
        hiddens.append(hid)
        del eval_model
        if device == "mps":
            torch.mps.empty_cache()

    print("\n=== per-seed metrics (EMA weights, bpb) ===")
    print(f"{'seed':>5} {'train':>8} {'proxy':>8} {'val':>8}")
    for m in metrics:
        print(f"{m['seed']:>5} {m['train_bpb']:>8.4f} {m['proxy_bpb']:>8.4f} {m['val_bpb']:>8.4f}")
    cols = {k: np.array([m[k] for m in metrics])
            for k in ("train_bpb", "proxy_bpb", "val_bpb")}
    for k, v in cols.items():
        print(f"{k}: mean {v.mean():.4f}  std {v.std(ddof=1):.5f}  "
              f"range [{v.min():.4f}, {v.max():.4f}]")

    names, pearson, spearman, pvals = corr_matrices(cols)
    print("\n=== Pearson correlation ===")
    print(fmt_matrix(names, pearson))
    print("\n=== Spearman correlation ===")
    print(fmt_matrix(names, spearman))
    print("\n=== permutation p-values (two-sided, Pearson) ===")
    print(fmt_matrix(names, pvals))

    print("\n=== spectral analysis of hidden matrices ===", flush=True)
    ratio, base_ratio, cos, base_cos, summ = spectral_analysis(hiddens, seeds)
    print(f"pairwise spectral-norm ratio  ||Wi-Wj||_2 / mean(||Wi||_2, ||Wj||_2), "
          f"averaged over {summ['n_matrices']} hidden matrices:")
    print(fmt_matrix([str(s) for s in seeds], ratio))
    print(f"\nmean over pairs: {summ['mean_pair_ratio']:.3f}   "
          f"random baseline (row-permuted): {summ['mean_baseline_ratio']:.3f}")
    print(f"cosine similarity of flattened hidden weights: "
          f"mean {summ['mean_pair_cos']:.4f}   baseline {summ['mean_baseline_cos']:.4f}")

    same_basin = (summ["mean_pair_ratio"] < 0.7 * summ["mean_baseline_ratio"]
                  or summ["mean_pair_cos"] > 0.2)
    if same_basin:
        verdict = ("VERDICT: pairs are markedly closer than the permuted baseline -> "
                   "the seeds share substantial weight-space structure (same region "
                   "of the loss landscape).")
    else:
        verdict = ("VERDICT: pairwise distances are ~equal to the permuted-random "
                   "baseline and cosine similarity is ~0 -> different seeds land in "
                   "different (permutation-unrelated) basins, essentially as far "
                   "apart as random.")
    print("\n" + verdict)

    results = {
        "config": vars(args), "metrics": metrics,
        "pearson": pearson.tolist(), "spearman": spearman.tolist(),
        "pearson_perm_pvals": pvals.tolist(), "corr_names": names,
        "spectral_ratio": ratio.tolist(), "spectral_ratio_baseline": base_ratio.tolist(),
        "cosine": cos.tolist(), "cosine_baseline": base_cos.tolist(),
        "spectral_summary": summ, "verdict": verdict,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {run_dir}/results.json")


if __name__ == "__main__":
    main()
