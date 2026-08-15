"""Train with online switch-density input warping (frequency/amplitude coupling).

Same recipe as train.py (Muon + AdamW, EMA of weights, periodic bpb eval), plus
a monotone piecewise-linear warp of the residual stream inside every block's MLP
branch, applied before norm2:

    x = x + attn(norm1(x))
    x = x + mlp(norm2(warp(x)))

The warp acts along a small pool of directions (default 10 per layer). Along
each direction it is a monotone piecewise-linear map whose slope tracks (by
EMA) the density of ReLU^2 switch points measured along that direction — a
CDF / histogram-equalization warp: input resolution is reallocated toward
where the layer's neurons actually switch, so local input amplitude matches
the layer's local switching frequency. Because the model has no biases, the
switch point of neuron n along direction d through the running mean mu is
closed-form:

    sign(pre-act) = sign(w'_n . warp(x)),   w' = up.weight * norm2.gain
    crossing where  h(t*) = h(t_mu) - (w'_n . warp(mu)) / (w'_n . d)

with h the warp's own 1-D map along d — so the per-step measurement costs a
few matvecs on weights already in memory, no extra forward passes.

Guardrails:
  * range preservation — slopes are normalized to mean 1 over the direction's
    1-99 pct data band and the warp is identity (continuous) outside it;
  * smooth entry/exit — new directions enter with an identity warp that EMAs
    toward the target; discarded directions fade back to identity over
    --warp_retire updates before their slot is freed (no function jumps);
  * Lipschitz clamp — slopes live in [1/kappa, kappa].

Pool maintenance per update (default every step): sample --warp_cands
candidate directions from differences of batch activations (orthogonalized
against the pool), score every direction by its in-band switch count, and
drop the worst back down to --warp_pool (active slots younger than
--warp_min_age updates are protected). All warp state lives in buffers:
updated from measured statistics, never by gradients — but gradients flow
*through* the warp, so the network co-adapts to it.

Example:
  python train_warp.py --data data/shakespeare_v2048 --run_name shak_v2048_warp \
      --steps 5000 --dropout 0.3 --weight_decay 0.25
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
import torch.nn as nn
import torch.nn.functional as F

from evaluation import evaluate_bpb
from model import GPT, GPTConfig
from muon import Muon
from train import EMA, pick_device


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/enwik8_10mb_v256")
    ap.add_argument("--run_name", type=str, default="run_warp")
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
    ap.add_argument("--min_lr_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    # eval
    ap.add_argument("--eval_interval", type=int, default=1000)
    ap.add_argument("--eval_stride", type=int, default=-1)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--quick_eval_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="the warp's pool checks cause graph breaks; expect little speedup")
    # warp
    ap.add_argument("--no_warp", action="store_true", help="ablation: plain train.py behavior")
    ap.add_argument("--warp_pool", type=int, default=10, help="active directions per layer")
    ap.add_argument("--warp_cands", type=int, default=3, help="candidate directions per update")
    ap.add_argument("--warp_knots", type=int, default=64, help="piecewise-linear bins per direction")
    ap.add_argument("--warp_kappa", type=float, default=4.0, help="slope clamp [1/kappa, kappa]")
    ap.add_argument("--warp_alpha", type=float, default=0.01, help="EMA rate of slopes toward target")
    ap.add_argument("--warp_band_alpha", type=float, default=0.02, help="EMA rate of the data band")
    ap.add_argument("--warp_score_alpha", type=float, default=0.2, help="EMA rate of slot scores")
    ap.add_argument("--warp_every", type=int, default=1, help="update warps every N steps")
    ap.add_argument("--warp_start", type=int, default=100, help="first step with warp updates")
    ap.add_argument("--warp_retire", type=int, default=300,
                    help="updates over which a discarded direction fades back to identity")
    ap.add_argument("--warp_min_age", type=int, default=25,
                    help="updates before an active direction may be discarded")
    ap.add_argument("--warp_rows", type=int, default=4096,
                    help="activation rows sampled per layer per update for statistics")
    return ap.parse_args()


class DirectionalWarp(nn.Module):
    """Monotone piecewise-linear warp along a pool of orthonormal directions.

    All state is buffers, updated from switch statistics (never by gradients);
    gradients flow through the warp w.r.t. its input. Identity (and exactly
    continuous) outside each direction's data band; slopes mean-1 inside it.
    """

    def __init__(self, dim, pool=10, cands=3, knots=64, kappa=4.0, alpha=0.01,
                 band_alpha=0.02, score_alpha=0.2, retire_steps=300, min_age=25,
                 sample_rows=4096, band_q=0.01, mu_alpha=0.05):
        super().__init__()
        S = pool + 8  # margin slots hold retiring warps while they fade out
        self.n_slots, self.pool, self.cands, self.K = S, pool, cands, knots
        self.kappa, self.alpha, self.band_alpha = kappa, alpha, band_alpha
        self.score_alpha, self.retire_steps, self.min_age = score_alpha, retire_steps, min_age
        self.sample_rows, self.band_q, self.mu_alpha = sample_rows, band_q, mu_alpha
        self.capture = False       # set by the training loop on update steps
        self._pending = None       # stashed activation sample (rows, dim)
        self.last_stats = None
        z = torch.zeros
        self.register_buffer("dirs", z(S, dim))                    # 0-row = free slot
        self.register_buffer("slopes", torch.ones(S, knots))       # g' per bin, mean 1
        self.register_buffer("cum", z(S, knots + 1))               # g at the knots
        self.register_buffer("lo", z(S))
        self.register_buffer("hi", torch.ones(S))
        self.register_buffer("fade", z(S))                         # 1 live -> 0 gone
        self.register_buffer("active", z(S, dtype=torch.bool))
        self.register_buffer("retiring", z(S, dtype=torch.bool))
        self.register_buffer("score", z(S))                        # EMA in-band switches
        self.register_buffer("age", z(S, dtype=torch.long))
        self.register_buffer("mu", z(dim))                         # running branch-input mean
        self.register_buffer("updates", z((), dtype=torch.long))
        self.register_buffer("turnover", z((), dtype=torch.long))
        self._refresh_cum()

    # ---- forward (differentiable w.r.t. x; buffers are constants) ----

    def forward(self, x):
        fade = self.fade
        if float(fade.max()) <= 0.0:
            return x
        D, K = self.dirs, self.K
        lo, hi = self.lo, self.hi
        dt = (hi - lo).clamp_min(1e-8) / K
        t = F.linear(x.float(), D)                            # (..., S)
        tt = torch.clamp(t, min=lo, max=hi)
        b = ((tt.detach() - lo) / dt).floor().clamp_(0, K - 1).long()
        off_k = torch.arange(self.n_slots, device=x.device) * K
        off_c = torch.arange(self.n_slots, device=x.device) * (K + 1)
        slope = self.slopes.reshape(-1)[b + off_k]
        cumv = self.cum.reshape(-1)[b + off_c]
        g = cumv + slope * (tt - (lo + b.to(t.dtype) * dt)) + (t - tt)
        delta = (g - t) * fade
        return x + F.linear(delta, D.T).to(x.dtype)

    def _refresh_cum(self):
        dt = (self.hi - self.lo).clamp_min(1e-8) / self.K
        cum = torch.cat([torch.zeros_like(self.lo)[:, None],
                         (self.slopes * dt[:, None]).cumsum(1)], dim=1) + self.lo[:, None]
        cum[:, -1] = torch.where(self.hi > self.lo, self.hi, cum[:, -1])
        self.cum.copy_(cum)

    # ---- 1-D map helpers (h = (1-fade)*id + fade*g along one slot) ----

    def _g_eval(self, j, t):
        lo, hi = self.lo[j], self.hi[j]
        dt = (hi - lo).clamp_min(1e-8) / self.K
        tt = t.clamp(lo, hi)
        b = ((tt - lo) / dt).floor().clamp(0, self.K - 1).long()
        return self.cum[j][b] + self.slopes[j][b] * (tt - (lo + b.to(t.dtype) * dt)) + (t - tt)

    def _h_eval(self, j, t):
        f = self.fade[j]
        return (1 - f) * t + f * self._g_eval(j, t)

    def _h_inv(self, j, u):
        """Invert h on in-band values u (assumes lo < u < hi)."""
        f = self.fade[j]
        lo, hi = self.lo[j], self.hi[j]
        dt = (hi - lo).clamp_min(1e-8) / self.K
        knots = lo + torch.arange(self.K + 1, device=u.device) * dt
        hk = (1 - f) * knots + f * self.cum[j]
        hs = (1 - f) + f * self.slopes[j]
        b = (torch.searchsorted(hk, u.contiguous()) - 1).clamp(0, self.K - 1)
        return knots[b] + (u - hk[b]) / hs[b]

    def _norm_slopes(self, dens):
        s = dens.clone()
        for _ in range(8):
            s /= s.mean()
            s.clamp_(1.0 / self.kappa, self.kappa)
        return s / s.mean()

    # ---- statistics capture & update ----

    def stash(self, x):
        flat = x.detach().float().reshape(-1, x.shape[-1])
        idx = torch.randint(0, flat.shape[0], (min(self.sample_rows, flat.shape[0]),),
                            device=flat.device)
        self._pending = flat[idx]

    @torch.no_grad()
    def update(self, w_eff):
        """One statistics/EMA step. w_eff = up.weight * norm2.gain, shape (N, C)."""
        X = self._pending
        if X is None:
            return None
        self._pending = None
        dev = X.device

        m = X.mean(0)
        if int(self.updates) == 0:
            self.mu.copy_(m)
        else:
            self.mu.lerp_(m, self.mu_alpha)
        self.updates += 1

        # candidate directions from activation differences, orthonormal to the pool
        n_active = int(self.active.sum())
        n_new = self.cands + max(0, self.pool - n_active)
        basis = [self.dirs[j] for j in (self.fade > 0).nonzero().squeeze(1).tolist()]
        cds, tries = [], 0
        while len(cds) < n_new and tries < 4 * n_new + 8:
            tries += 1
            i, j = torch.randint(0, X.shape[0], (2,)).tolist()
            v = X[i] - X[j]
            for b in basis + cds:
                v = v - (v @ b) * b
            n0 = v.norm()
            if n0 < 1e-6:
                continue
            cds.append(v / n0)
        cand = torch.stack(cds) if cds else torch.zeros(0, X.shape[1], device=dev)

        act_idx = self.active.nonzero().squeeze(1)
        na = act_idx.numel()
        Dm = torch.cat([self.dirs[act_idx], cand], dim=0)          # (m, C)
        stats = {"n_active": n_active, "inband": 0.0, "slope_dev": 0.0,
                 "turnover": int(self.turnover)}
        if Dm.shape[0] == 0:
            self._advance_fades()
            self.last_stats = stats
            return stats

        # data band per direction (EMA for active slots, batch quantiles for cands)
        proj = X @ Dm.T                                            # (rows, m)
        q = torch.quantile(proj, torch.tensor([self.band_q, 1 - self.band_q],
                                              device=dev, dtype=proj.dtype), dim=0)
        qlo, qhi = q[0], q[1]
        if na:
            self.lo[act_idx] = torch.lerp(self.lo[act_idx], qlo[:na], self.band_alpha)
            self.hi[act_idx] = torch.lerp(self.hi[act_idx], qhi[:na], self.band_alpha)

        # closed-form switch points: h(t*) = h(t_mu) - (w'.warp(mu)) / (w'.d)
        warped_mu = self.forward(self.mu[None])[0]
        A = w_eff @ warped_mu                                      # (N,)
        P = w_eff @ Dm.T                                           # (N, m)
        t_mu = Dm @ self.mu                                        # (m,)
        hmu = t_mu.clone()
        for r, j in enumerate(act_idx.tolist()):
            hmu[r] = self._h_eval(j, t_mu[r:r + 1])[0]
        safeP = torch.where(P.abs() < 1e-9, torch.full_like(P, torch.inf), P)
        U = hmu[None, :] - A[:, None] / safeP                      # (N, m)
        blo = torch.cat([self.lo[act_idx], qlo[na:]])
        bhi = torch.cat([self.hi[act_idx], qhi[na:]])
        inband = (U > blo) & (U < bhi)
        counts = inband.sum(0).float()                             # (m,)

        # slope targets for active slots: equalize the in-band switch histogram
        for r, j in enumerate(act_idx.tolist()):
            self.score[j] = (1 - self.score_alpha) * self.score[j] + self.score_alpha * counts[r]
            self.age[j] += 1
            u = U[:, r][inband[:, r]]
            if u.numel() < 2 or float(self.hi[j] - self.lo[j]) < 1e-6:
                continue
            t_star = self._h_inv(j, u)
            hist = torch.histc(t_star, bins=self.K, min=float(self.lo[j]), max=float(self.hi[j]))
            dens = hist + 0.05 * hist.mean() + 1e-3                # floor: no zero bins
            tgt = self._norm_slopes(dens)
            self.slopes[j] = torch.lerp(self.slopes[j], tgt, self.alpha)
            self.slopes[j] /= self.slopes[j].mean()                # exact range preservation

        # selection: keep the best --warp_pool of {eligible actives + candidates}
        n_c = Dm.shape[0] - na
        if n_c > 0:
            eligible = [int(j) for j in act_idx.tolist() if int(self.age[j]) >= self.min_age]
            capacity = self.pool - (n_active - len(eligible))
            entries = ([(float(self.score[j]), "slot", j) for j in eligible]
                       + [(float(counts[na + c]), "cand", c) for c in range(n_c)])
            entries.sort(key=lambda e: -e[0])
            for sc, kind, j in entries[max(capacity, 0):]:
                if kind == "slot":
                    self.active[j] = False
                    self.retiring[j] = True
                    self.turnover += 1
            for sc, kind, c in entries[:max(capacity, 0)]:
                if kind == "cand":
                    self._admit(cand[c], float(qlo[na + c]), float(qhi[na + c]), sc)

        self._advance_fades()
        self._refresh_cum()
        if na:
            stats["inband"] = float(counts[:na].mean())
        if bool(self.active.any()):
            stats["slope_dev"] = float((self.slopes[self.active] - 1).abs().mean())
        stats["n_active"] = int(self.active.sum())
        stats["turnover"] = int(self.turnover)
        self.last_stats = stats
        return stats

    def _admit(self, d, lo, hi, score):
        free = ((self.fade <= 0) & ~self.active & ~self.retiring).nonzero()
        if free.numel() == 0 or hi - lo < 1e-6:
            return
        j = int(free[0])
        self.dirs[j] = d
        self.slopes[j] = 1.0
        self.lo[j], self.hi[j] = lo, hi
        self.fade[j] = 1.0
        self.active[j] = True
        self.retiring[j] = False
        self.score[j] = score
        self.age[j] = 0
        self.turnover += 1

    def _advance_fades(self):
        r = self.retiring
        if not bool(r.any()):
            return
        self.fade[r] -= 1.0 / self.retire_steps
        dead = r & (self.fade <= 0)
        for j in dead.nonzero().squeeze(1).tolist():
            self.dirs[j] = 0.0
            self.slopes[j] = 1.0
            self.lo[j], self.hi[j] = 0.0, 1.0
            self.fade[j] = 0.0
            self.retiring[j] = False
            self.score[j] = 0.0
            self.age[j] = 0


class WarpedBlock(nn.Module):
    """A model.Block with the warp spliced into the MLP branch (before norm2)."""

    def __init__(self, block, warp):
        super().__init__()
        self.norm1, self.attn = block.norm1, block.attn
        self.norm2, self.mlp = block.norm2, block.mlp
        self.warp = warp

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.warp.capture and self.training:
            self.warp.stash(x)
        x = x + self.mlp(self.norm2(self.warp(x)))
        return x


def main():
    args = get_args()
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

    if args.eval_stride <= 0:
        args.eval_stride = args.block_size
    cfg = GPTConfig(
        vocab_size=meta["vocab_size"], block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, attn_dropout=args.attn_dropout, softcap=args.softcap,
    )
    model = GPT(cfg)
    if not args.no_warp:
        for i in range(len(model.blocks)):
            warp = DirectionalWarp(
                cfg.n_embd, pool=args.warp_pool, cands=args.warp_cands,
                knots=args.warp_knots, kappa=args.warp_kappa, alpha=args.warp_alpha,
                band_alpha=args.warp_band_alpha, score_alpha=args.warp_score_alpha,
                retire_steps=args.warp_retire, min_age=args.warp_min_age,
                sample_rows=args.warp_rows,
            )
            model.blocks[i] = WarpedBlock(model.blocks[i], warp)
    model = model.to(device)
    if args.compile:
        model = torch.compile(model)
    print(f"device={device}  params={model.num_params()/1e6:.2f}M  vocab={cfg.vocab_size}  "
          f"warp={'off' if args.no_warp else 'on'}")
    print(f"train: {len(train_ids):,} tokens  |  epoch = {len(train_ids)//(args.batch_size*args.block_size):,} steps")

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
    # warp state is a slow shared statistic, not a weight: keep it out of the EMA
    # (lerping directions across a pool swap would blend unrelated vectors) and
    # sync it into the eval model via load_state_dict instead.
    ema.shadow = {k: v for k, v in ema.shadow.items() if ".warp." not in k}
    eval_model = copy.deepcopy(model)

    nats_to_bpb = (meta["val_tokens"] / meta["val_bytes"]) / math.log(2)

    @torch.no_grad()
    def quick_val_bpb() -> float:
        model.eval()
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
    with open(log_path, "w") as f:
        f.write("step,lr_mult,train_loss,quick_val_bpb,val_bpb_raw,val_bpb_ema,time_s\n")
    warp_log_path = os.path.join(run_dir, "warp_log.csv")
    with open(warp_log_path, "w") as f:
        f.write("step,layer,n_active,inband,slope_dev,turnover\n")

    best_bpb = float("inf")
    t0 = time.time()
    running_loss = None

    for step in range(args.steps):
        m = lr_mult(step)
        for opt, peak in zip(optimizers, peak_lrs):
            for g in opt.param_groups:
                g["lr"] = m * peak

        do_stats = (not args.no_warp and step >= args.warp_start
                    and step % args.warp_every == 0)
        if not args.no_warp:
            for blk in model.blocks:
                blk.warp.capture = do_stats

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
        ema.update(model, step)

        if do_stats:
            for blk in model.blocks:
                w_eff = (blk.mlp.up.weight.float() * blk.norm2.weight.float()).detach()
                blk.warp.update(w_eff)

        l = loss.item()
        running_loss = l if running_loss is None else 0.99 * running_loss + 0.01 * l
        if step % args.log_interval == 0:
            qbpb = quick_val_bpb()
            print(f"step {step:6d}  loss {running_loss:.4f}  val~bpb {qbpb:.4f}  "
                  f"lr x{m:.3f}  {time.time()-t0:.0f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{running_loss:.4f},{qbpb:.4f},,,{time.time()-t0:.0f}\n")
            if not args.no_warp and step >= args.warp_start:
                rows, agg = [], {"inband": 0.0, "slope_dev": 0.0, "turnover": 0}
                for li, blk in enumerate(model.blocks):
                    st = blk.warp.last_stats
                    if st is None:
                        continue
                    rows.append(f"{step},{li},{st['n_active']},{st['inband']:.1f},"
                                f"{st['slope_dev']:.4f},{st['turnover']}")
                    agg["inband"] += st["inband"]
                    agg["slope_dev"] += st["slope_dev"]
                    agg["turnover"] += st["turnover"]
                if rows:
                    with open(warp_log_path, "a") as f:
                        f.write("\n".join(rows) + "\n")
                    n = len(rows)
                    print(f"        warp: inband/layer {agg['inband']/n:.1f}  "
                          f"|slope-1| {agg['slope_dev']/n:.3f}  "
                          f"turnover {agg['turnover']}", flush=True)

        if (step + 1) % args.eval_interval == 0 or step == args.steps - 1:
            bpb_raw = evaluate_bpb(model, val_ids, meta["val_bytes"], args.eval_stride,
                                   args.eval_batch_size, device)
            eval_model.load_state_dict(model.state_dict())  # brings live warp buffers
            ema.copy_to(eval_model)
            bpb_ema = evaluate_bpb(eval_model, val_ids, meta["val_bytes"], args.eval_stride,
                                   args.eval_batch_size, device)
            print(f"step {step:6d}  val bpb raw {bpb_raw:.4f}  ema {bpb_ema:.4f}", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{m:.4f},{running_loss:.4f},,{bpb_raw:.4f},{bpb_ema:.4f},{time.time()-t0:.0f}\n")
            which, cand_bpb, cand_model = (
                ("ema", bpb_ema, eval_model) if bpb_ema <= bpb_raw else ("raw", bpb_raw, model)
            )
            if cand_bpb < best_bpb:
                best_bpb = cand_bpb
                torch.save({"model": cand_model.state_dict(), "config": cfg.__dict__,
                            "step": step, "val_bpb": cand_bpb, "which": which,
                            "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
            torch.save({"model": model.state_dict(), "ema": ema.shadow, "config": cfg.__dict__,
                        "step": step, "args": vars(args)},
                       os.path.join(run_dir, "latest.pt"))

    print(f"done. best EMA val bpb: {best_bpb:.4f}  ({run_dir}/best.pt)")


if __name__ == "__main__":
    main()
