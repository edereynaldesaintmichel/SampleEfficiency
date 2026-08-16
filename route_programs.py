"""Stage 0 analysis + Stage 1 stacking routers over dump_programs.py output.

Stage 0 (val dump):
  - per-program singles, uniform-mixture ensembles vs K,
  - oracle-vs-K curve (mean over random subsets) — calibrates how much of the
    oracle gap is extreme-value noise vs real routable structure.

Stage 1 (train on the train dump ONLY, report on val once per variant):
  - static:   learned per-program weights (K params),
  - entropy:  w_i ∝ exp(-alpha * H_i), alpha picked on train,
  - mlp:      MLP over per-program features (entropy, log max-prob, agreement),
  - mlp+prev: same + embedding of the conditioning token.
All routers output log-weights; loss is the weighted-mixture NLL
  -log sum_i w_i exp(-nll_i),
so uniform weights are the guaranteed starting point (final layers zero-init).

--val_ab replaces the train->val protocol with the honest fallback for when
train-split routers don't transfer (the base model has memorized train, so
program stats there are distorted): train on one contiguous half of val,
report on the other, both directions, clearly labeled.

Example:
  python route_programs.py --dump_dir runs/shak_shuffle_5M_20k/routing
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import pick_device


def bpb(nll_sum: float, n_bytes: float) -> float:
    return nll_sum / math.log(2) / n_bytes


def uniform_mix(nll: torch.Tensor) -> torch.Tensor:  # (N, K) -> (N,)
    return -torch.logsumexp(-nll.double(), dim=1) + math.log(nll.size(1))


def weighted_mix(nll: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logw.double() - nll.double(), dim=1)


class Router(nn.Module):
    def __init__(self, K: int, n_feat: int, vocab: int, mode: str,
                 d_emb: int = 16, hidden: int = 128):
        super().__init__()
        self.mode = mode
        self.bias = nn.Parameter(torch.zeros(K))
        if mode in ("mlp", "mlp+prev"):
            n_in = n_feat + (d_emb if mode == "mlp+prev" else 0)
            self.emb = nn.Embedding(vocab, d_emb) if mode == "mlp+prev" else None
            self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                     nn.Linear(hidden, K))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        logits = self.bias.expand(feats.size(0), -1)
        if self.mode in ("mlp", "mlp+prev"):
            x = feats if self.emb is None else torch.cat([feats, self.emb(prev)], dim=-1)
            logits = logits + self.net(x)
        return F.log_softmax(logits, dim=-1)


def build_features(d: dict) -> torch.Tensor:
    ent, lmp = d["ent"].float(), d["lmp"].float()
    t = d["top1"].long()
    modal = torch.mode(t, dim=1).values
    agree = (t == modal.unsqueeze(1)).float().mean(dim=1, keepdim=True)
    return torch.cat([ent, lmp, agree], dim=1)


def train_router(mode, Xtr, ptr, ntr, Xho, pho, nho, vocab, device,
                 lr=1e-3, batch=16384, max_epochs=30, patience=3, seed=0):
    torch.manual_seed(seed)
    r = Router(ntr.size(1), Xtr.size(1), vocab, mode).to(device)
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    best_ho, best_state, bad = float("inf"), None, 0
    N = Xtr.size(0)
    for epoch in range(max_epochs):
        r.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            logw = r(Xtr[idx], ptr[idx])
            loss = (-torch.logsumexp(logw - ntr[idx], dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        r.eval()
        with torch.no_grad():
            ho = float(weighted_mix(nho.cpu(), r(Xho, pho).cpu()).mean())
        if ho < best_ho - 1e-5:
            best_ho, bad = ho, 0
            best_state = {k: v.detach().clone() for k, v in r.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    r.load_state_dict(best_state)
    r.eval()
    return r, best_ho, epoch + 1


def slice_dump(d: dict, lo: int, hi: int) -> dict:
    N = d["nll"].size(0)
    out = {k: d[k][lo:hi] for k in ("nll", "ent", "lmp", "top1", "prev_id")}
    out["n_bytes"] = d["n_bytes"] * (hi - lo) / N  # approximate
    out["programs"] = d["programs"]
    return out


def stage1(tr: dict, va: dict, holdout_frac: float, device: str, label: str):
    """Train routers on `tr`, report on `va`. Returns (best_bpb, mode, router, mu, sd)."""
    K = tr["nll"].size(1)
    B = va["n_bytes"]
    nv = va["nll"]
    Xtr_all, Xva = build_features(tr), build_features(va)
    mu, sd = Xtr_all.mean(0, keepdim=True), Xtr_all.std(0, keepdim=True).clamp_min(1e-4)
    Xtr_all, Xva = (Xtr_all - mu) / sd, (Xva - mu) / sd
    cut = Xtr_all.size(0) - int(Xtr_all.size(0) * holdout_frac)
    vocab = int(max(tr["prev_id"].max(), va["prev_id"].max())) + 1

    to = lambda t: t.to(device)
    Xtr, ptr, ntr = to(Xtr_all[:cut]), to(tr["prev_id"][:cut].long()), to(tr["nll"][:cut])
    Xho, pho = to(Xtr_all[cut:]), to(tr["prev_id"][cut:].long())
    nho = tr["nll"][cut:]
    Xva_d, pva = to(Xva), to(va["prev_id"].long())

    uni_ho = float(uniform_mix(nho).mean())
    uni_val = bpb(float(uniform_mix(nv).sum()), B)
    print(f"--- {label} ---")
    print(f"{'variant':10s}  {'fit-ho nats':>11s}  {'report bpb':>10s}")
    print(f"{'uniform':10s}  {uni_ho:11.4f}  {uni_val:10.4f}")

    # entropy weighting: alpha picked on the fit holdout
    best_alpha, best_ho = 0.0, uni_ho
    for alpha in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        logw = F.log_softmax(-alpha * tr["ent"][cut:].float(), dim=1)
        ho = float(weighted_mix(nho, logw).mean())
        if ho < best_ho:
            best_alpha, best_ho = alpha, ho
    v = bpb(float(weighted_mix(nv, F.log_softmax(-best_alpha * va["ent"].float(), dim=1)).sum()), B)
    print(f"{'entropy':10s}  {best_ho:11.4f}  {v:10.4f}   (alpha={best_alpha})")

    best = (uni_val, "uniform", None)
    for mode in ("static", "mlp", "mlp+prev"):
        r, ho, ep = train_router(mode, Xtr, ptr, ntr, Xho, pho, nho, vocab, device)
        with torch.no_grad():
            logw_val = r(Xva_d, pva).cpu()
        v = bpb(float(weighted_mix(nv, logw_val).sum()), B)
        with torch.no_grad():
            w_ent = float((-(logw_val.exp() * logw_val).sum(1)).mean())
            agree = float((logw_val.argmax(1) == nv.argmin(1)).float().mean())
        print(f"{mode:10s}  {ho:11.4f}  {v:10.4f}   (epochs {ep}, weight-H {w_ent:.2f}/{math.log(K):.2f}, "
              f"argmax=oracle {100 * agree:.1f}%)")
        if v < best[0]:
            best = (v, mode, r)
    print(f"best: {best[1]}  report bpb {best[0]:.4f}  (uniform {uni_val:.4f}, "
          f"oracle {bpb(float(nv.double().min(dim=1).values.sum()), B):.4f})\n")
    return best + (mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=str, required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.05, help="tail of the fit split, for early stopping")
    ap.add_argument("--val_ab", action="store_true", help="fit on val half A, report on half B (both directions) instead of train->val")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--save_router", type=str, default="", help="optional path for the best router")
    args = ap.parse_args()
    device = pick_device(args.device)

    va = torch.load(os.path.join(args.dump_dir, "val.pt"), weights_only=False)
    K = va["nll"].size(1)
    B = va["n_bytes"]
    nv = va["nll"]
    print(f"K={K} programs (depths {[len(p) for p in va['programs']]})  val {nv.size(0)} tokens\n")

    # ---------------- Stage 0 ----------------
    singles = np.array([bpb(float(nv[:, i].double().sum()), B) for i in range(K)])
    print(f"val singles: mean {singles.mean():.4f}  min {singles.min():.4f}  max {singles.max():.4f}")
    for k in sorted({min(k, K) for k in (2, 4, 8, K)}):
        print(f"uniform ensemble of first {k:2d}: {bpb(float(uniform_mix(nv[:, :k]).sum()), B):.4f}")
    rng = np.random.default_rng(0)
    print("oracle vs K (mean over 10 random subsets):")
    for k in sorted({min(k, K) for k in (1, 2, 4, 8, 12, K)}):
        reps = 1 if k == K else 10
        vals = [bpb(float(nv[:, rng.choice(K, k, replace=False)].min(dim=1).values.double().sum()), B)
                for _ in range(reps)]
        print(f"  oracle@{k:2d}: {np.mean(vals):.4f}")
    gap = uniform_mix(nv) - nv.double().min(dim=1).values
    q = torch.quantile(gap.float(), torch.tensor([0.5, 0.9, 0.99]))
    print(f"per-token (uniform - oracle) nats: median {q[0]:.3f}  p90 {q[1]:.3f}  p99 {q[2]:.3f}\n")

    # ---------------- Stage 1 ----------------
    if args.val_ab:
        mid = nv.size(0) // 2
        A, Bd = slice_dump(va, 0, mid), slice_dump(va, mid, nv.size(0))
        r1 = stage1(A, Bd, args.holdout_frac, device, "fit val-A -> report val-B")
        r2 = stage1(Bd, A, args.holdout_frac, device, "fit val-B -> report val-A")
        best = min(r1, r2, key=lambda r: r[0])
    else:
        tr = torch.load(os.path.join(args.dump_dir, "train.pt"), weights_only=False)
        assert tr["programs"] == va["programs"]
        best = stage1(tr, va, args.holdout_frac, device, "fit train -> report val")

    if args.save_router and best[2] is not None:
        torch.save({"state": best[2].state_dict(), "mode": best[1], "mu": best[3], "sd": best[4],
                    "programs": va["programs"]}, args.save_router)
        print(f"saved router to {args.save_router}")


if __name__ == "__main__":
    main()
