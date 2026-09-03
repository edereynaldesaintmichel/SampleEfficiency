"""Train a "random program" transformer: each step samples an effective depth
L ~ U{min_depth..max_depth}, then draws L blocks *with replacement* from the
n_layer base blocks in random order, and trains the batch through that
sequence. One set of ~10.5M weights is thus trained to work as any random
composition of its 8 layers, from shallow (4) to very deep (32).

Evals score val bpb (raw and EMA weights) at:
  - ordered configs o1..o4: the canonical stack repeated k times (depth 8k),
    directly comparable to train_loop.py runs;
  - fixed random programs r8/r16/r32: one sequence per depth drawn once at
    startup (seeded), so their trajectory is comparable across the run.
best.pt keeps the best (config, raw/ema) combination including the sequence.

Example:
  python train_shuffle.py --data data/shakespeare_v2048 --run_name shak_shuffle \
      --steps 5000 --dropout 0.3 --weight_decay 0.25 --grad_accum 2
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
    ap.add_argument("--run_name", type=str, default="run_shuffle")
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
    ap.add_argument("--mlp_ratio", type=float, default=4.0,
                    help="FFN hidden dim as multiple of n_embd")
    ap.add_argument("--shared_mlp", action="store_true",
                    help="one FFN shared across all blocks (attn/norms stay per-block); "
                         "combine with a larger --mlp_ratio to reinvest the saved params")
    # random-program depth
    ap.add_argument("--act", type=str, default="relu2", choices=["relu2", "step", "dead", "noisy"],
                    help="MLP activation: relu2 | step relu²-c·H(z) | dead relu(z-c)² | noisy relu(z+c·ε)²")
    ap.add_argument("--act_c", type=float, default=0.0,
                    help="activation constant c: step size, dead-zone width, or noise std")
    ap.add_argument("--min_depth", type=int, default=4)
    ap.add_argument("--max_depth", type=int, default=32)
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
    ap.add_argument("--ce_floor", type=float, default=0.0,
                    help="per-token loss floor in nats: tokens with CE below this get zero "
                         "gradient (per-token flooding). 0 = plain CE. Normalization stays "
                         "sum/total-tokens so easy tokens are dropped, not hard ones up-weighted.")
    ap.add_argument("--ce_pow", type=float, default=1.0,
                    help="train on (CE + eps)^p per token instead of CE. p<1 is concave: "
                         "gradient weight p*(CE+eps)^(p-1) UP-weights easy tokens and shrinks "
                         "the signal from rare/hard ones (anti-focal). 1 = plain CE.")
    ap.add_argument("--ce_pow_eps", type=float, default=0.01,
                    help="epsilon inside the power transform; caps the CE->0 gradient blowup "
                         "at p*eps^(p-1) (~5x for p=0.5, eps=0.01)")
    # eval
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--eval_stride", type=int, default=-1, help="-1 = block_size (fast interim eval); smaller = more context")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--quick_eval_batches", type=int, default=2,
                    help="random val batches PER CONFIG for the cheap val estimate printed every log_interval")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast (CUDA only; evals always run fp32)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--init_from", type=str, default="",
                    help="checkpoint to load model weights from (fresh optimizers/EMA/schedule)")
    return ap.parse_args()


def main():
    args = get_args()
    if args.ce_floor > 0 and args.ce_pow != 1.0:
        raise SystemExit("--ce_floor and --ce_pow are separate experiments; set only one")
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
        mlp_ratio=args.mlp_ratio, shared_mlp=args.shared_mlp,
        act=args.act, act_c=args.act_c,
    )
    if device == "mps" and (args.attn_dropout if args.attn_dropout >= 0 else args.dropout) > 0:
        print("WARNING: attn dropout > 0 on MPS forces unfused attention and will "
              "likely OOM — pass --attn_dropout 0 for local runs.")
    model = GPT(cfg).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"init from {args.init_from} (step {ckpt.get('step')}, val_bpb {ckpt.get('val_bpb')})")
    if args.compile:
        model = torch.compile(model)

    # eval configs: ordered stack repeated k times + fixed random programs
    rng_seq = np.random.default_rng(args.seed)
    eval_cfgs = [(f"o{k}", list(range(args.n_layer)) * k) for k in range(1, 5)]
    for d in (8, 16, 32):
        eval_cfgs.append((f"r{d}", rng_seq.integers(0, args.n_layer, d).tolist()))
    names = [n for n, _ in eval_cfgs]
    print(f"device={device}  params={model.num_params()/1e6:.2f}M  vocab={cfg.vocab_size}  "
          f"depth U{{{args.min_depth}..{args.max_depth}}} over {args.n_layer} blocks  "
          f"act={args.act} c={args.act_c}")
    for n, s in eval_cfgs:
        print(f"  eval {n}: {s}")
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
        """Paired comparison: loss of the SAME batch at every eval config,
        dropout off, so config gaps aren't confounded by batch/mask noise."""
        model.eval()
        for b in model.blocks:
            b.mlp.stats = True  # pre-activation diagnostics, this pass only
        out = []
        for _, seq in eval_cfgs:
            model.set_layer_seq(seq)
            _, loss = model(x, y)
            out.append(loss.item())
        for b in model.blocks:
            b.mlp.stats = False
        model.set_layer_seq(None)
        model.train()
        return out

    @torch.no_grad()
    def quick_val_bpb(seq: list[int]) -> float:
        """Cheap val estimate for one program: mean loss over a few random val windows."""
        model.eval()
        model.set_layer_seq(seq)
        total, n = 0.0, 0
        for _ in range(args.quick_eval_batches):
            ix = np.random.randint(0, len(val_ids) - args.block_size - 1, size=args.eval_batch_size)
            x = torch.stack([val_ids[i:i + args.block_size] for i in ix]).to(device)
            y = torch.stack([val_ids[i + 1:i + 1 + args.block_size] for i in ix]).to(device)
            _, loss = model(x, y)
            total += loss.item()
            n += 1
        model.set_layer_seq(None)
        model.train()
        return total / n * nats_to_bpb

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"params": model.num_params(),
                                "eval_cfgs": {n: s for n, s in eval_cfgs}}, f, indent=2)
    log_path = os.path.join(run_dir, "log.csv")
    t_cols = ",".join(f"train_loss_{n}" for n in names)
    q_cols = ",".join(f"quick_val_bpb_{n}" for n in names)
    r_cols = ",".join(f"val_bpb_raw_{n}" for n in names)
    e_cols = ",".join(f"val_bpb_ema_{n}" for n in names)
    with open(log_path, "w") as f:
        f.write(f"step,lr_mult,train_loss,{t_cols},{q_cols},{r_cols},{e_cols},time_s\n")
    n_cfgs = len(eval_cfgs)

    best_bpb = float("inf")
    t0 = time.time()
    running_loss = None
    running_gate = None

    for step in range(args.steps):
        m = lr_mult(step)
        for opt, peak in zip(optimizers, peak_lrs):
            for g in opt.param_groups:
                g["lr"] = m * peak

        depth = int(np.random.randint(args.min_depth, args.max_depth + 1))
        model.set_layer_seq(np.random.randint(0, args.n_layer, depth).tolist())
        x, y = get_batch()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        micro = max(1, args.grad_accum)
        mb = x.size(0) // micro
        loss_sum = 0.0
        gated_sum = 0.0
        for i in range(micro):
            xm, ym = x[i * mb:(i + 1) * mb], y[i * mb:(i + 1) * mb]
            with amp_ctx:
                if args.ce_floor > 0 or args.ce_pow != 1.0:
                    _, lt = model(xm, ym, loss_reduction="none")
                    loss_sum += lt.detach().mean().item()  # log plain CE, comparable across runs
                    if args.ce_floor > 0:
                        keep = (lt.detach() > args.ce_floor).float()
                        loss = (lt * keep).sum() / lt.numel()
                        gated_sum += 1.0 - keep.mean().item()
                    else:
                        loss = ((lt + args.ce_pow_eps) ** args.ce_pow).mean()
                else:
                    _, loss = model(xm, ym)
                    loss_sum += loss.item()
            (loss / micro).backward()
        loss = None
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
        ema.update(model, step)
        model.set_layer_seq(None)

        l = loss_sum / micro
        running_loss = l if running_loss is None else 0.99 * running_loss + 0.01 * l
        g = gated_sum / micro
        running_gate = g if running_gate is None else 0.99 * running_gate + 0.01 * g
        if step % args.log_interval == 0:
            ts = probe_train_loss(x, y)
            qs = [quick_val_bpb(seq) for _, seq in eval_cfgs]
            tstr = "  ".join(f"{n} {t:.3f}" for n, t in zip(names, ts))
            qstr = "  ".join(f"{n} {q:.3f}" for n, q in zip(names, qs))
            gstr = f"  gate {running_gate:.2f}" if args.ce_floor > 0 else ""
            z_rms = float(np.mean([b.mlp.z_rms.item() for b in model.blocks]))
            zstr = f"  z_rms {z_rms:.3f}"
            if model.blocks[0].mlp.z_small is not None:
                zstr += f"  small {float(np.mean([b.mlp.z_small.item() for b in model.blocks])):.2f}"
            print(f"step {step:6d}  loss {running_loss:.4f}{gstr}  d{depth:2d}  "
                  f"train [{tstr}]  val~bpb [{qstr}]{zstr}  lr x{m:.3f}  {time.time()-t0:.0f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{running_loss:.4f},"
                        + ",".join(f"{t:.4f}" for t in ts) + ","
                        + ",".join(f"{q:.4f}" for q in qs)
                        + "," * (2 * n_cfgs) + f",{time.time()-t0:.0f}\n")

        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            eval_model.load_state_dict(model.state_dict())
            ema.copy_to(eval_model)
            raws, emas = [], []
            for _, seq in eval_cfgs:
                model.set_layer_seq(seq)
                eval_model.set_layer_seq(seq)
                raws.append(evaluate_bpb(model, val_ids, meta["val_bytes"], args.eval_stride,
                                         args.eval_batch_size, device))
                emas.append(evaluate_bpb(eval_model, val_ids, meta["val_bytes"], args.eval_stride,
                                         args.eval_batch_size, device))
            model.set_layer_seq(None)
            eval_model.set_layer_seq(None)
            rstr = "  ".join(f"{n} {b:.4f}" for n, b in zip(names, raws))
            estr = "  ".join(f"{n} {b:.4f}" for n, b in zip(names, emas))
            print(f"step {step:6d}  val bpb raw [{rstr}]", flush=True)
            print(f"step {step:6d}  val bpb ema [{estr}]", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{running_loss:.4f}," + "," * (2 * n_cfgs)
                        + ",".join(f"{b:.4f}" for b in raws) + ","
                        + ",".join(f"{b:.4f}" for b in emas)
                        + f",{time.time()-t0:.0f}\n")

            cands = ([("raw", nm, sq, b, model) for (nm, sq), b in zip(eval_cfgs, raws)]
                     + [("ema", nm, sq, b, eval_model) for (nm, sq), b in zip(eval_cfgs, emas)])
            which, cfg_name, cfg_seq, cand_bpb, cand_model = min(cands, key=lambda c: c[3])
            if cand_bpb < best_bpb:
                best_bpb = cand_bpb
                torch.save({"model": cand_model.state_dict(), "config": cfg.__dict__,
                            "step": step, "val_bpb": cand_bpb, "which": which,
                            "cfg_name": cfg_name, "layer_seq": cfg_seq, "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
                print(f"        new best: {cand_bpb:.4f} ({which}, {cfg_name})", flush=True)
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "config": cfg.__dict__,
                        "step": step, "args": vars(args)},
                       os.path.join(run_dir, "latest.pt"))

    print(f"done. best val bpb: {best_bpb:.4f}  ({run_dir}/best.pt)")


if __name__ == "__main__":
    main()
