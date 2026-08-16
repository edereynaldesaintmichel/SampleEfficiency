"""Dump per-token, per-program prediction stats for program-routing work.

For K seeded iid programs (depths cycled from --depths), scores a split with
the standard sliding-window protocol and records per predicted token:
  nll (fp32)   — -log p(target); used as the routing/mixture LOSS only,
  ent (fp16)   — entropy of the predicted distribution,
  lmp (fp16)   — log max-prob,
  top1 (int16) — argmax token id,
plus prev_id, the conditioning token at each position. Everything except nll
depends only on the predicted distribution — no target leakage into features.

The program menu is drawn with --seed 1 so it is DISJOINT from eval_shuffle's
seed-0 programs.

Example:
  python dump_programs.py --ckpt runs/shak_shuffle_5M_20k/best.pt \
      --data data/shakespeare_v2048 --out_dir runs/shak_shuffle_5M_20k/routing
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig
from train import pick_device


@torch.no_grad()
def per_token_stats(model, ids: torch.Tensor, stride: int, batch_size: int, device: str):
    """Like eval_loops.per_token_nll but also returns entropy/logmaxp/top1."""
    model.eval()
    T = model.cfg.block_size
    N = ids.size(0)
    nll = torch.zeros(N, dtype=torch.float32)
    ent = torch.zeros(N, dtype=torch.float16)
    lmp = torch.zeros(N, dtype=torch.float16)
    top1 = torch.zeros(N, dtype=torch.int16)

    windows = []
    for b in range(1, N, stride):
        e = min(b + stride, N)
        w = max(0, e - 1 - T)
        windows.append((w, b, e))
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for win in windows:
        w, b, e = win
        groups.setdefault((e - 1 - w, e - b), []).append(win)

    for (inp_len, _), wins in groups.items():
        for i in range(0, len(wins), batch_size):
            chunk = wins[i:i + batch_size]
            x = torch.stack([ids[w:e - 1] for w, b, e in chunk]).to(device)
            logits, _ = model(x)
            logp = torch.log_softmax(logits.float(), dim=-1)
            H = -(logp.exp() * logp).sum(-1)
            mx, am = logp.max(-1)
            for j, (w, b, e) in enumerate(chunk):
                s = b - 1 - w  # window offset of the first predicted token
                tgt = ids[b:e].to(device)
                nll[b:e] = -logp[j, s:].gather(1, tgt.view(-1, 1)).squeeze(1).cpu()
                ent[b:e] = H[j, s:].half().cpu()
                lmp[b:e] = mx[j, s:].half().cpu()
                top1[b:e] = am[j, s:].to(torch.int16).cpu()
    return nll[1:], ent[1:], lmp[1:], top1[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="", help="default: <ckpt dir>/routing")
    ap.add_argument("--n_programs", type=int, default=16)
    ap.add_argument("--depths", type=str, default="16,24,32", help="cycled over programs")
    ap.add_argument("--seed", type=int, default=1, help="1 = disjoint from eval_shuffle's menu")
    ap.add_argument("--splits", type=str, default="train,val")
    ap.add_argument("--train_tokens", type=int, default=1_000_000, help="cap on the train split (0 = all)")
    ap.add_argument("--max_tokens", type=int, default=0, help="cap on non-train splits (smoke tests)")
    ap.add_argument("--stride", type=int, default=-1, help="-1 = block_size")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    stride = args.stride if args.stride > 0 else cfg.block_size
    depth_cycle = [int(s) for s in args.depths.split(",")]
    programs = [rng.integers(0, cfg.n_layer, depth_cycle[i % len(depth_cycle)]).tolist()
                for i in range(args.n_programs)]
    print(f"ckpt step {ckpt.get('step')}  saved val_bpb {ckpt.get('val_bpb')}  |  "
          f"K={len(programs)} programs, depths {[len(p) for p in programs]}  device={device}")

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), "routing")
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits.split(","):
        ids = torch.from_numpy(
            np.fromfile(os.path.join(args.data, f"{split}.bin"), dtype=dtype).astype(np.int64)
        )
        n_bytes = meta.get(f"{split}_bytes")
        cap = args.train_tokens if split == "train" else args.max_tokens
        if cap and cap < ids.size(0):
            if n_bytes:
                n_bytes = n_bytes * cap / ids.size(0)  # approximate
            ids = ids[:cap]
        Nt = ids.size(0) - 1
        K = len(programs)
        out = {"nll": torch.zeros(Nt, K, dtype=torch.float32),
               "ent": torch.zeros(Nt, K, dtype=torch.float16),
               "lmp": torch.zeros(Nt, K, dtype=torch.float16),
               "top1": torch.zeros(Nt, K, dtype=torch.int16)}
        for pi, prog in enumerate(programs):
            model.set_layer_seq(prog)
            t0 = time.time()
            nll, ent, lmp, top1 = per_token_stats(model, ids, stride, args.batch_size, device)
            out["nll"][:, pi], out["ent"][:, pi], out["lmp"][:, pi], out["top1"][:, pi] = nll, ent, lmp, top1
            b = float(nll.double().sum()) / math.log(2) / n_bytes if n_bytes else float("nan")
            print(f"[{split}] prog {pi:2d} depth {len(prog):2d}:  bpb {b:.4f}  "
                  f"nats/tok {float(nll.double().mean()):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        payload = {**out, "prev_id": ids[:-1].to(torch.int16), "programs": programs,
                   "split": split, "n_bytes": n_bytes, "ckpt_step": ckpt.get("step"),
                   "ckpt": args.ckpt, "seed": args.seed}
        path = os.path.join(out_dir, f"{split}.pt")
        torch.save(payload, path)
        print(f"saved {path}  ({Nt} tokens, K={K})", flush=True)


if __name__ == "__main__":
    main()
