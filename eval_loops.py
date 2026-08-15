"""Per-depth and oracle-depth evaluation of a looped checkpoint.

Scores every val token at each depth in --loops (same sliding-window protocol
as evaluation.py, but keeping per-token NLL), then reports:
  - val bpb at each fixed depth,
  - ORACLE bpb: every token scored at its best depth (cheating upper bound — this
    is the ceiling any learned halting policy could reach),
  - the oracle's depth histogram (where the extra depth actually pays).

Example:
  python eval_loops.py --ckpt runs/shak_loops/best.pt --data data/shakespeare_v2048
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from model import GPT, GPTConfig
from train import pick_device


@torch.no_grad()
def per_token_nll(model, val_ids: torch.Tensor, stride: int, batch_size: int,
                  device: str) -> torch.Tensor:
    """NLL (nats) of each val token 1..N-1 under evaluation.py's windowing."""
    model.eval()
    T = model.cfg.block_size
    N = val_ids.size(0)
    out = torch.zeros(N, dtype=torch.float64)

    windows = []
    for b in range(1, N, stride):
        e = min(b + stride, N)
        w = max(0, e - 1 - T)
        windows.append((w, b, e))
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for win in windows:
        w, b, e = win
        groups.setdefault((e - 1 - w, e - b), []).append(win)

    for (inp_len, n_tgt), wins in groups.items():
        for i in range(0, len(wins), batch_size):
            chunk = wins[i:i + batch_size]
            x = torch.stack([val_ids[w:e - 1] for w, b, e in chunk]).to(device)
            y = torch.full((len(chunk), inp_len), -1, dtype=torch.long)
            for j, (w, b, e) in enumerate(chunk):
                y[j, b - 1 - w:] = val_ids[b:e]
            y = y.to(device)
            _, loss = model(x, y, loss_reduction="none")
            loss = loss.view(len(chunk), inp_len).double().cpu()
            for j, (w, b, e) in enumerate(chunk):
                out[b:e] = loss[j, b - 1 - w:]
    return out[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--loops", type=str, default="1,2,3,4")
    ap.add_argument("--stride", type=int, default=-1, help="-1 = block_size")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=0, help="cap val tokens (0 = all; for quick tests)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--save_nll", type=str, default="", help="optional .npy path for the (n_depths, n_tokens) NLL matrix")
    args = ap.parse_args()
    loops = [int(s) for s in args.loops.split(",")]
    device = pick_device(args.device)

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    val_ids = torch.from_numpy(
        np.fromfile(os.path.join(args.data, "val.bin"), dtype=dtype).astype(np.int64)
    )
    val_bytes = meta["val_bytes"]
    if args.max_tokens and args.max_tokens < val_ids.size(0):
        val_bytes = val_bytes * args.max_tokens / val_ids.size(0)  # approximate
        val_ids = val_ids[:args.max_tokens]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    stride = args.stride if args.stride > 0 else cfg.block_size
    print(f"ckpt step {ckpt.get('step')}  saved val_bpb {ckpt.get('val_bpb')} "
          f"({ckpt.get('which')}, loops={ckpt.get('loops')})  |  eval device={device}")

    nll = []
    for k in loops:
        model.set_loops(k)
        n = per_token_nll(model, val_ids, stride, args.batch_size, device)
        nll.append(n)
        print(f"loops {k}:  val bpb {float(n.sum()) / math.log(2) / val_bytes:.4f}")
    nll = torch.stack(nll)  # (n_depths, n_tokens)

    best_fixed = min(float(n.sum()) for n in nll) / math.log(2) / val_bytes
    oracle_nll, oracle_k = nll.min(dim=0)
    oracle = float(oracle_nll.sum()) / math.log(2) / val_bytes
    print(f"\nbest fixed depth: {best_fixed:.4f}")
    print(f"oracle per-token depth: {oracle:.4f}  (headroom {best_fixed - oracle:.4f} bpb)")
    hist = torch.bincount(oracle_k, minlength=len(loops)).float() / oracle_k.numel()
    print("oracle depth histogram: "
          + "  ".join(f"l{k} {100 * float(h):.1f}%" for k, h in zip(loops, hist)))

    if args.save_nll:
        np.save(args.save_nll, nll.numpy())
        print(f"saved NLL matrix to {args.save_nll}")


if __name__ == "__main__":
    main()
