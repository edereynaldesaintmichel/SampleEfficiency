"""Mutual "teachable teacher" training: a big looped model (teacher) and a small
looped model (student) train jointly on the same batches.

    loss = CE_teacher(data) + student_ce * CE_student(data) + lambda_kl * KL(p_t || p_s)

The KL term is shared and NOT detached on either side: the student is pulled
toward the teacher's distribution (distillation), and the teacher is penalized
for predictions the student cannot match (a distillability / complexity
regularizer — "if you can't teach it, you didn't fully understand it").

Both models are looped (weight-shared depth); each step one loop count k is
sampled from --train_loops and applied to BOTH models, so the coupling is
between matched-compute-menu depths. Architectures are taken from the init
checkpoints (required); CLI only controls dropout and optimization. Defaults
assume converged-checkpoint inits, hence the gentle LRs (the validated
continuation settings: peak muon 1.5e-3, warmup 100).

Example:
  python train_mutual.py --data data/shakespeare_v2048 --run_name shak_mutual \
      --teacher_init runs/shak_loops/best.pt --student_init runs/shak_loops_2L/best.pt \
      --steps 5000 --dropout 0.3 --weight_decay 0.25 --lambda_kl 0.5 --grad_accum 4
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
import torch.nn.functional as F

from evaluation import evaluate_bpb
from model import GPT, GPTConfig
from muon import Muon
from train import EMA, pick_device


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/shakespeare_v2048")
    ap.add_argument("--run_name", type=str, default="run_mutual")
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--teacher_init", type=str, required=True,
                    help="checkpoint for the big model (arch is read from it)")
    ap.add_argument("--student_init", type=str, required=True,
                    help="checkpoint for the small model (arch is read from it)")
    # coupling
    ap.add_argument("--lambda_kl", type=float, default=0.5,
                    help="weight of the shared KL(p_teacher || p_student) term")
    ap.add_argument("--student_ce", type=float, default=0.0,
                    help="weight of the student's own data CE (0 = pure distillation)")
    # regularization (applied to both models, overriding checkpoint config)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--attn_dropout", type=float, default=-1.0,
                    help="attention-matrix dropout; -1 = same as --dropout; set 0 on MPS")
    # looped depth
    ap.add_argument("--train_loops", type=str, default="1,2,3,4",
                    help="comma list; per-step loop count (same for both models)")
    ap.add_argument("--eval_loops", type=str, default="1,2,3,4")
    # optimization (gentle defaults: we start from converged checkpoints)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=4,
                    help="micro-batches per step (both models' graphs are alive at once)")
    ap.add_argument("--muon_lr", type=float, default=1.5e-3)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--adam_lr", type=float, default=2.25e-4)
    ap.add_argument("--weight_decay", type=float, default=0.25)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--min_lr_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    # eval
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--eval_stride", type=int, default=-1)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--quick_eval_batches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no_amp", action="store_true")
    return ap.parse_args()


def load_model(path: str, args, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stored = dict(ckpt["config"])
    stored["dropout"] = args.dropout
    stored["attn_dropout"] = args.attn_dropout
    cfg = GPTConfig(**stored)
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"init {path}: {model.num_params()/1e6:.2f}M params, n_layer={cfg.n_layer}, "
          f"d={cfg.n_embd} (ckpt step {ckpt.get('step')}, val_bpb {ckpt.get('val_bpb')}, "
          f"{ckpt.get('which')}, loops={ckpt.get('loops')})")
    return model, stored


def make_optimizers(model: GPT, args) -> tuple[list, list]:
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
    return [muon, adamw], [args.muon_lr, args.adam_lr]


def main():
    args = get_args()
    train_loops = [int(s) for s in args.train_loops.split(",")]
    eval_loops = [int(s) for s in args.eval_loops.split(",")]
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device == "cuda" and not args.no_amp else contextlib.nullcontext())

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    train_ids = np.memmap(os.path.join(args.data, "train.bin"), dtype=dtype, mode="r")
    val_ids = torch.from_numpy(
        np.fromfile(os.path.join(args.data, "val.bin"), dtype=dtype).astype(np.int64)
    )

    teacher, t_cfg = load_model(args.teacher_init, args, device)
    student, s_cfg = load_model(args.student_init, args, device)
    assert t_cfg["vocab_size"] == meta["vocab_size"] == s_cfg["vocab_size"]
    if args.eval_stride <= 0:
        args.eval_stride = t_cfg["block_size"]
    block_size = min(t_cfg["block_size"], s_cfg["block_size"])
    print(f"device={device}  lambda_kl={args.lambda_kl}  student_ce={args.student_ce}  "
          f"train_loops={train_loops}")

    t_opts, t_peaks = make_optimizers(teacher, args)
    s_opts, s_peaks = make_optimizers(student, args)

    def lr_mult(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t))

    def get_batch():
        ix = np.random.randint(0, len(train_ids) - block_size - 1, size=args.batch_size)
        x = torch.stack([torch.from_numpy(train_ids[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(train_ids[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    t_ema, s_ema = EMA(teacher, args.ema_decay), EMA(student, args.ema_decay)
    t_eval, s_eval = copy.deepcopy(teacher), copy.deepcopy(student)

    nats_to_bpb = (meta["val_tokens"] / meta["val_bytes"]) / math.log(2)

    @torch.no_grad()
    def quick_val_bpb(model: GPT, k: int) -> float:
        model.eval()
        model.set_loops(k)
        total = 0.0
        for _ in range(args.quick_eval_batches):
            ix = np.random.randint(0, len(val_ids) - block_size - 1, size=args.eval_batch_size)
            x = torch.stack([val_ids[i:i + block_size] for i in ix]).to(device)
            y = torch.stack([val_ids[i + 1:i + 1 + block_size] for i in ix]).to(device)
            _, loss = model(x, y)
            total += loss.item()
        model.train()
        return total / args.quick_eval_batches * nats_to_bpb

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"teacher_params": teacher.num_params(),
                                "student_params": student.num_params()}, f, indent=2)
    log_path = os.path.join(run_dir, "log.csv")
    qt = ",".join(f"quick_t_l{k}" for k in eval_loops)
    qs = ",".join(f"quick_s_l{k}" for k in eval_loops)
    ft = ",".join(f"val_t_{w}_l{k}" for w in ("raw", "ema") for k in eval_loops)
    fs = ",".join(f"val_s_{w}_l{k}" for w in ("raw", "ema") for k in eval_loops)
    with open(log_path, "w") as f:
        f.write(f"step,lr_mult,ce_t,ce_s,kl,{qt},{qs},{ft},{fs},time_s\n")
    n_d = len(eval_loops)

    best = {"t": float("inf"), "s": float("inf")}
    run_ce_t = run_ce_s = run_kl = None
    t0 = time.time()

    for step in range(args.steps):
        m = lr_mult(step)
        for opt, peak in zip(t_opts + s_opts, t_peaks + s_peaks):
            for g in opt.param_groups:
                g["lr"] = m * peak

        k = int(np.random.choice(train_loops))
        teacher.set_loops(k)
        student.set_loops(k)
        x, y = get_batch()
        for opt in t_opts + s_opts:
            opt.zero_grad(set_to_none=True)
        micro = max(1, args.grad_accum)
        mb = x.size(0) // micro
        ce_t_sum = ce_s_sum = kl_sum = 0.0
        for i in range(micro):
            xm, ym = x[i * mb:(i + 1) * mb], y[i * mb:(i + 1) * mb]
            with amp_ctx:
                t_logits, ce_t = teacher(xm, ym)
                s_logits, ce_s = student(xm, ym)
                log_t = F.log_softmax(t_logits, dim=-1)
                log_s = F.log_softmax(s_logits, dim=-1)
                # KL(p_t || p_s), gradients flow into BOTH models
                kl = (log_t.exp() * (log_t - log_s)).sum(-1).mean()
                loss = ce_t + args.student_ce * ce_s + args.lambda_kl * kl
            (loss / micro).backward()
            ce_t_sum += ce_t.item()
            ce_s_sum += ce_s.item()
            kl_sum += kl.item()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        for opt in t_opts + s_opts:
            opt.step()
        t_ema.update(teacher, step)
        s_ema.update(student, step)

        vals = [v / micro for v in (ce_t_sum, ce_s_sum, kl_sum)]
        run_ce_t, run_ce_s, run_kl = (
            v if r is None else 0.99 * r + 0.01 * v
            for r, v in zip((run_ce_t, run_ce_s, run_kl), vals))
        if step % args.log_interval == 0:
            q_t = [quick_val_bpb(teacher, kk) for kk in eval_loops]
            q_s = [quick_val_bpb(student, kk) for kk in eval_loops]
            qts = "  ".join(f"l{kk} {q:.3f}" for kk, q in zip(eval_loops, q_t))
            qss = "  ".join(f"l{kk} {q:.3f}" for kk, q in zip(eval_loops, q_s))
            print(f"step {step:6d}  ce_t {run_ce_t:.4f}  ce_s {run_ce_s:.4f}  kl {run_kl:.4f}  "
                  f"valT [{qts}]  valS [{qss}]  lr x{m:.3f}  {time.time()-t0:.0f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{run_ce_t:.4f},{run_ce_s:.4f},{run_kl:.4f},"
                        + ",".join(f"{q:.4f}" for q in q_t) + ","
                        + ",".join(f"{q:.4f}" for q in q_s)
                        + "," * (4 * n_d) + f",{time.time()-t0:.0f}\n")

        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            full = {}
            for tag, model, ema, eval_model in (("t", teacher, t_ema, t_eval),
                                                ("s", student, s_ema, s_eval)):
                eval_model.load_state_dict(model.state_dict())
                ema.copy_to(eval_model)
                raws, emas = [], []
                for kk in eval_loops:
                    model.set_loops(kk)
                    eval_model.set_loops(kk)
                    raws.append(evaluate_bpb(model, val_ids, meta["val_bytes"], args.eval_stride,
                                             args.eval_batch_size, device))
                    emas.append(evaluate_bpb(eval_model, val_ids, meta["val_bytes"], args.eval_stride,
                                             args.eval_batch_size, device))
                full[tag] = (raws, emas)
                name = "teacher" if tag == "t" else "student"
                print(f"step {step:6d}  {name} val bpb raw ["
                      + "  ".join(f"l{kk} {b:.4f}" for kk, b in zip(eval_loops, raws)) + "]", flush=True)
                print(f"step {step:6d}  {name} val bpb ema ["
                      + "  ".join(f"l{kk} {b:.4f}" for kk, b in zip(eval_loops, emas)) + "]", flush=True)
                cands = ([("raw", kk, b, model) for kk, b in zip(eval_loops, raws)]
                         + [("ema", kk, b, eval_model) for kk, b in zip(eval_loops, emas)])
                which, k_best, cand_bpb, cand_model = min(cands, key=lambda c: c[2])
                if cand_bpb < best[tag]:
                    best[tag] = cand_bpb
                    torch.save({"model": cand_model.state_dict(), "config": cand_model.cfg.__dict__,
                                "step": step, "val_bpb": cand_bpb, "which": which,
                                "loops": k_best, "args": vars(args)},
                               os.path.join(run_dir, f"best_{name}.pt"))
                    print(f"        new best {name}: {cand_bpb:.4f} ({which}, loops={k_best})", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{run_ce_t:.4f},{run_ce_s:.4f},{run_kl:.4f},"
                        + "," * (2 * n_d)
                        + ",".join(f"{b:.4f}" for part in full["t"] for b in part) + ","
                        + ",".join(f"{b:.4f}" for part in full["s"] for b in part)
                        + f",{time.time()-t0:.0f}\n")

    print(f"done. best teacher {best['t']:.4f}  best student {best['s']:.4f}  ({run_dir})")


if __name__ == "__main__":
    main()
