"""Sliding-window bits-per-byte evaluation.

Every val token except the very first is scored exactly once. Targets are
processed in chunks of `stride`; each chunk is predicted with the longest
context that fits in block_size (so smaller stride = more context per
prediction = better/slower). bpb = total_bits / val_bytes, comparable across
tokenizers since val_bytes refers to the same underlying text.
"""

import math

import torch


@torch.no_grad()
def evaluate_bpb(model, val_ids: torch.Tensor, val_bytes: int, stride: int,
                 batch_size: int = 32, device: str = "cpu") -> float:
    model.eval()
    T = model.cfg.block_size
    assert 0 < stride <= T
    N = val_ids.size(0)

    # targets [b, e) are predicted from inputs [w, e-1); need w <= b-1 (context
    # reaches the first target) and e-1-w <= T (input fits in block_size)
    windows = []
    for b in range(1, N, stride):
        e = min(b + stride, N)
        w = max(0, e - 1 - T)
        windows.append((w, b, e))

    total_nats = 0.0
    # group windows by (input_len, n_targets) so uniform groups batch cleanly
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
            _, loss = model(x, y, loss_reduction="sum")
            total_nats += loss.item()

    model.train()
    return total_nats / math.log(2) / val_bytes
