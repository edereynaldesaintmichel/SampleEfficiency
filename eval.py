"""Official evaluation: load a checkpoint, report sliding-window bpb on val.

Use a small stride (e.g. 128) for the final number — every prediction gets
near-maximal context. Interim evals during training use stride=block_size.

  python eval.py runs/baseline/best.pt --data data/enwik8_10mb_v256 --stride 128
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluation import evaluate_bpb
from model import GPT, GPTConfig
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--data", type=str, default="data/enwik8_10mb_v256")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"],
                    help="test is for the final report only — never for selection")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    ids = torch.from_numpy(
        np.fromfile(os.path.join(args.data, f"{args.split}.bin"), dtype=dtype).astype(np.int64)
    )

    bpb = evaluate_bpb(model, ids, meta[f"{args.split}_bytes"], args.stride,
                       args.batch_size, device)
    print(f"{args.checkpoint} (step {ckpt.get('step','?')}): "
          f"{args.split} bpb = {bpb:.4f}  [stride={args.stride}, block={cfg.block_size}]")


if __name__ == "__main__":
    main()
