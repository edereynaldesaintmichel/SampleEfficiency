"""Train with looped (weight-shared) depth: each block is applied `loops` times
per forward (same count for every block), so compute depth is n_layer * loops
while the parameter count stays fixed (~10.5M at defaults).

The loop count is sampled uniformly from --train_loops each step, so a single
set of weights is trained to work at every depth in the menu; evals then score
val bpb at each depth in --eval_loops (raw and EMA weights) and best.pt keeps
the best (depth, raw/ema) combination. Run eval_loops.py on a checkpoint for
per-token oracle-depth headroom.

Example:
  python train_loop.py --data data/shakespeare_v2048 --run_name shak_loops \
      --steps 5000 --dropout 0.3 --weight_decay 0.25 \
      --train_loops 1,2,3,4 --eval_loops 1,2,3,4
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
    ap.add_argument("--data", type=str, default="data/enwik8_10mb_v256")
    ap.add_argument("--run_name", type=str, default="run_loop")
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
    # looped depth
    ap.add_argument("--train_loops", type=str, default="1,2,3,4",
                    help="comma list; per-step loop count is sampled uniformly from it")
    ap.add_argument("--eval_loops", type=str, default="1,2,3,4",
                    help="comma list of depths to score at every eval")
    # optimization
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="split each batch into this many micro-batches (same effective batch, less memory)")
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
    ap.add_argument("--quick_eval_batches", type=int, default=2,
                    help="random val batches PER DEPTH for the cheap val estimate printed every log_interval")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast (CUDA only; evals always run fp32)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--init_from", type=str, default="",
                    help="checkpoint to load model weights from (fresh optimizers/EMA/schedule)")
    return ap.parse_args()


def main():
    args = get_args()
    train_loops = [int(s) for s in args.train_loops.split(",")]
    eval_loops = [int(s) for s in args.eval_loops.split(",")]
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
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"init from {args.init_from} (step {ckpt.get('step')}, "
              f"val_bpb {ckpt.get('val_bpb')}, {ckpt.get('which')}, loops={ckpt.get('loops')})")
    if args.compile:
        model = torch.compile(model)
    print(f"device={device}  params={model.num_params()/1e6:.2f}M  vocab={cfg.vocab_size}  "
          f"train_loops={train_loops}  eval_loops={eval_loops}")
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

    nats_to_bpb = (meta["val_tokens"] / meta["val_bytes"]) / math.log(2)

    @torch.no_grad()
    def probe_train_loss(x, y) -> list[float]:
        """Paired depth comparison: loss of the SAME batch at every eval depth,
        dropout off, so the l1-vs-l4 gap isn't confounded by batch/mask noise."""
        model.eval()
        out = []
        for k in eval_loops:
            model.set_loops(k)
            _, loss = model(x, y)
            out.append(loss.item())
        model.train()
        return out

    @torch.no_grad()
    def quick_val_bpb(k: int) -> float:
        """Cheap val estimate at depth k: mean loss over a few random val windows."""
        model.eval()
        model.set_loops(k)
        total, n = 0.0, 0
        for _ in range(args.quick_eval_batches):
            ix = np.random.randint(0, len(val_ids) - args.block_size - 1, size=args.eval_batch_size)
            x = torch.stack([val_ids[i:i + args.block_size] for i in ix]).to(device)
            y = torch.stack([val_ids[i + 1:i + 1 + args.block_size] for i in ix]).to(device)
            _, loss = model(x, y)
            total += loss.item()
            n += 1
        model.train()
        return total / n * nats_to_bpb

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"params": model.num_params()}, f, indent=2)
    log_path = os.path.join(run_dir, "log.csv")
    t_cols = ",".join(f"train_loss_l{k}" for k in eval_loops)
    q_cols = ",".join(f"quick_val_bpb_l{k}" for k in eval_loops)
    r_cols = ",".join(f"val_bpb_raw_l{k}" for k in eval_loops)
    e_cols = ",".join(f"val_bpb_ema_l{k}" for k in eval_loops)
    with open(log_path, "w") as f:
        f.write(f"step,lr_mult,train_loss,{t_cols},{q_cols},{r_cols},{e_cols},time_s\n")
    n_depths = len(eval_loops)

    best_bpb = float("inf")
    t0 = time.time()
    running_loss = None

    for step in range(args.steps):
        m = lr_mult(step)
        for opt, peak in zip(optimizers, peak_lrs):
            for g in opt.param_groups:
                g["lr"] = m * peak

        model.set_loops(int(np.random.choice(train_loops)))
        x, y = get_batch()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        micro = max(1, args.grad_accum)
        mb = x.size(0) // micro
        loss_sum = 0.0
        for i in range(micro):
            xm, ym = x[i * mb:(i + 1) * mb], y[i * mb:(i + 1) * mb]
            with amp_ctx:
                _, loss = model(xm, ym)
            (loss / micro).backward()
            loss_sum += loss.item()
        loss = None
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
        ema.update(model, step)

        l = loss_sum / micro
        running_loss = l if running_loss is None else 0.99 * running_loss + 0.01 * l
        if step % args.log_interval == 0:
            ts = probe_train_loss(x, y)
            qs = [quick_val_bpb(k) for k in eval_loops]
            tstr = "  ".join(f"l{k} {t:.3f}" for k, t in zip(eval_loops, ts))
            qstr = "  ".join(f"l{k} {q:.3f}" for k, q in zip(eval_loops, qs))
            print(f"step {step:6d}  loss {running_loss:.4f}  "
                  f"train [{tstr}]  gap l{eval_loops[0]}-l{eval_loops[-1]} {ts[0]-ts[-1]:+.4f}  "
                  f"val~bpb [{qstr}]  lr x{m:.3f}  {time.time()-t0:.0f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{running_loss:.4f},"
                        + ",".join(f"{t:.4f}" for t in ts) + ","
                        + ",".join(f"{q:.4f}" for q in qs)
                        + "," * (2 * n_depths) + f",{time.time()-t0:.0f}\n")

        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            eval_model.load_state_dict(model.state_dict())
            ema.copy_to(eval_model)
            raws, emas = [], []
            for k in eval_loops:
                model.set_loops(k)
                eval_model.set_loops(k)
                raws.append(evaluate_bpb(model, val_ids, meta["val_bytes"], args.eval_stride,
                                         args.eval_batch_size, device))
                emas.append(evaluate_bpb(eval_model, val_ids, meta["val_bytes"], args.eval_stride,
                                         args.eval_batch_size, device))
            rstr = "  ".join(f"l{k} {b:.4f}" for k, b in zip(eval_loops, raws))
            estr = "  ".join(f"l{k} {b:.4f}" for k, b in zip(eval_loops, emas))
            print(f"step {step:6d}  val bpb raw [{rstr}]", flush=True)
            print(f"step {step:6d}  val bpb ema [{estr}]", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{running_loss:.4f}," + "," * (2 * n_depths)
                        + ",".join(f"{b:.4f}" for b in raws) + ","
                        + ",".join(f"{b:.4f}" for b in emas)
                        + f",{time.time()-t0:.0f}\n")

            cands = ([("raw", k, b, model) for k, b in zip(eval_loops, raws)]
                     + [("ema", k, b, eval_model) for k, b in zip(eval_loops, emas)])
            which, k_best, cand_bpb, cand_model = min(cands, key=lambda c: c[2])
            if cand_bpb < best_bpb:
                best_bpb = cand_bpb
                torch.save({"model": cand_model.state_dict(), "config": cfg.__dict__,
                            "step": step, "val_bpb": cand_bpb, "which": which,
                            "loops": k_best, "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
                print(f"        new best: {cand_bpb:.4f} ({which}, loops={k_best})", flush=True)
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "config": cfg.__dict__,
                        "step": step, "args": vars(args)},
                       os.path.join(run_dir, "latest.pt"))

    print(f"done. best val bpb: {best_bpb:.4f}  ({run_dir}/best.pt)")


if __name__ == "__main__":
    main()
