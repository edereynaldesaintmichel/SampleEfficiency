"""Train the ~10M-param transformer on a small corpus, maximizing sample efficiency.

Muon (hidden matrices) + AdamW (embeddings/norms), heavy regularization,
EMA of weights, periodic sliding-window bpb eval on the contiguous val split,
best-checkpoint selection on EMA val bpb.

Example:
  python train.py --data data/enwik8_10mb_v256 --run_name baseline
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


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/enwik8_10mb_v256")
    ap.add_argument("--run_name", type=str, default="run")
    ap.add_argument("--out_dir", type=str, default="runs")
    # model
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=5)
    ap.add_argument("--n_embd", type=int, default=320)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--attn_dropout", type=float, default=-1.0,
                    help="attention-matrix dropout; -1 = same as --dropout; set 0 on MPS")
    ap.add_argument("--softcap", type=float, default=30.0)
    # optimization
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--adam_lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--min_lr_frac", type=float, default=0.05, help="final LR as fraction of peak (cosine)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # regularization / averaging
    ap.add_argument("--ema_decay", type=float, default=0.999)
    # eval
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--eval_stride", type=int, default=-1, help="-1 = block_size (fast interim eval); smaller = more context")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast (CUDA only; evals always run fp32)")
    ap.add_argument("--compile", action="store_true")
    return ap.parse_args()


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach().float(), 1 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


def main():
    args = get_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")  # TF32 on CUDA
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device == "cuda" and not args.no_amp else contextlib.nullcontext())

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    train_ids = np.memmap(os.path.join(args.data, "train.bin"), dtype=dtype, mode="r")
    val_ids = torch.from_numpy(
        np.fromfile(os.path.join(args.data, "val.bin"), dtype=dtype).astype(np.int64)
    )

    if args.eval_stride <= 0:
        args.eval_stride = args.block_size
    cfg = GPTConfig(
        vocab_size=meta["vocab_size"], block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, attn_dropout=args.attn_dropout, softcap=args.softcap,
    )
    if device == "mps" and (args.attn_dropout if args.attn_dropout >= 0 else args.dropout) > 0:
        print("WARNING: attn dropout > 0 on MPS forces unfused attention and will "
              "likely OOM — pass --attn_dropout 0 for local runs.")
    model = GPT(cfg).to(device)
    if args.compile:
        model = torch.compile(model)
    print(f"device={device}  params={model.num_params()/1e6:.2f}M  vocab={cfg.vocab_size}")
    print(f"train: {len(train_ids):,} tokens  |  epoch = {len(train_ids)//(args.batch_size*args.block_size):,} steps")

    # Muon for 2D hidden matrices, AdamW for embeddings (=tied head) and norm gains
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

    def get_batch():
        ix = np.random.randint(0, len(train_ids) - args.block_size - 1, size=args.batch_size)
        x = torch.stack([torch.from_numpy(train_ids[i:i + args.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(train_ids[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    ema = EMA(model, args.ema_decay)
    eval_model = copy.deepcopy(model)  # holds EMA weights during eval

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"params": model.num_params()}, f, indent=2)
    log_path = os.path.join(run_dir, "log.csv")
    with open(log_path, "w") as f:
        f.write("step,lr_mult,train_loss,val_bpb_raw,val_bpb_ema,time_s\n")

    best_bpb = float("inf")
    t0 = time.time()
    running_loss = None

    for step in range(args.steps):
        m = lr_mult(step)
        for opt, peak in zip(optimizers, peak_lrs):
            for g in opt.param_groups:
                g["lr"] = m * peak

        x, y = get_batch()
        with amp_ctx:
            _, loss = model(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
        ema.update(model)

        l = loss.item()
        running_loss = l if running_loss is None else 0.99 * running_loss + 0.01 * l
        if step % args.log_interval == 0:
            print(f"step {step:6d}  loss {running_loss:.4f}  lr x{m:.3f}  {time.time()-t0:.0f}s")

        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            bpb_raw = evaluate_bpb(model, val_ids, meta["val_bytes"], args.eval_stride,
                                   args.eval_batch_size, device)
            eval_model.load_state_dict(model.state_dict())
            ema.copy_to(eval_model)
            bpb_ema = evaluate_bpb(eval_model, val_ids, meta["val_bytes"], args.eval_stride,
                                   args.eval_batch_size, device)
            print(f"step {step:6d}  val bpb raw {bpb_raw:.4f}  ema {bpb_ema:.4f}")
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{running_loss:.4f},{bpb_raw:.4f},{bpb_ema:.4f},{time.time()-t0:.0f}\n")
            if bpb_ema < best_bpb:
                best_bpb = bpb_ema
                torch.save({"model": eval_model.state_dict(), "config": cfg.__dict__,
                            "step": step, "val_bpb": bpb_ema, "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "config": cfg.__dict__,
                        "step": step, "args": vars(args)},
                       os.path.join(run_dir, "latest.pt"))

    print(f"done. best EMA val bpb: {best_bpb:.4f}  ({run_dir}/best.pt)")


if __name__ == "__main__":
    main()
