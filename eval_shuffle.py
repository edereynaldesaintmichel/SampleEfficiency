"""Order-vs-composition analysis of a shuffle-trained checkpoint.

At a fixed depth d it scores:
  - the ordered configs o1..o4 (stack repeated k times) for reference;
  - N random PERMUTATIONS of the balanced multiset (every block d/n_layer
    times) — the spread isolates how much sequence order matters given a
    fixed composition;
  - N random iid PROGRAMS (draw-with-replacement) — the gap to the balanced
    group isolates how much composition (which blocks / multiplicities)
    matters;
  - probability-space ensembles over programs: p_mix = mean_i p_i, evaluated
    for growing subsets of each group, plus the per-token oracle over all
    scored programs (ceiling for any program-routing policy).

Example:
  python eval_shuffle.py --ckpt runs/shak_shuffle/best.pt \
      --data data/shakespeare_v2048 --depth 24 --n_perms 12 --n_random 12
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from eval_loops import per_token_nll
from model import GPT, GPTConfig
from train import pick_device


def bpb(nll: torch.Tensor, val_bytes: float) -> float:
    return float(nll.sum()) / math.log(2) / val_bytes


def mix_bpb(nlls: list[torch.Tensor], val_bytes: float) -> float:
    """bpb of the probability-space mixture of the given programs."""
    m = torch.stack(nlls)  # (n_programs, n_tokens), nats
    mix_nll = -torch.logsumexp(-m, dim=0) + math.log(m.size(0))
    return bpb(mix_nll, val_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--depth", type=int, default=24, help="should be a multiple of n_layer for the balanced group")
    ap.add_argument("--n_perms", type=int, default=12)
    ap.add_argument("--n_random", type=int, default=12)
    ap.add_argument("--stride", type=int, default=-1, help="-1 = block_size")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=0, help="cap val tokens (0 = all; for quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)

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
    L = cfg.n_layer
    print(f"ckpt step {ckpt.get('step')}  saved val_bpb {ckpt.get('val_bpb')} "
          f"({ckpt.get('which')}, {ckpt.get('cfg_name')})  |  depth {args.depth}  device={device}")

    def score(seq: list[int]) -> torch.Tensor:
        model.set_layer_seq(seq)
        return per_token_nll(model, val_ids, stride, args.batch_size, device)

    for k in range(1, 5):
        n = score(list(range(L)) * k)
        print(f"o{k} (ordered x{k}, depth {L * k}):  {bpb(n, val_bytes):.4f}")

    balanced = list(range(L)) * (args.depth // L)
    groups = {}
    for name, progs in [
        ("perm", [list(rng.permutation(balanced)) for _ in range(args.n_perms)]),
        ("iid", [rng.integers(0, L, args.depth).tolist() for _ in range(args.n_random)]),
    ]:
        nlls = []
        for s in progs:
            nlls.append(score(s))
            print(f"  {name} {[int(v) for v in s]}: {bpb(nlls[-1], val_bytes):.4f}", flush=True)
        groups[name] = nlls
        bs = np.array([bpb(n, val_bytes) for n in nlls])
        print(f"{name} (n={len(bs)}, depth {args.depth}):  mean {bs.mean():.4f}  std {bs.std():.4f}  "
              f"min {bs.min():.4f}  max {bs.max():.4f}")
        for m in (2, 4, 8, len(nlls)):
            if m <= len(nlls):
                print(f"  ensemble of {m}: {mix_bpb(nlls[:m], val_bytes):.4f}")

    both = groups["perm"] + groups["iid"]
    oracle_nll = torch.stack(both).min(dim=0).values
    print(f"\nensemble of all {len(both)}: {mix_bpb(both, val_bytes):.4f}")
    print(f"per-token program oracle over {len(both)}: {bpb(oracle_nll, val_bytes):.4f}")


if __name__ == "__main__":
    main()
