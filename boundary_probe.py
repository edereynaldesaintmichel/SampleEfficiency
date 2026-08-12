"""Decision-boundary probe: layer behavior along principal components of the input.

For each "unit" of the network, capture its input activations on a real batch,
PCA them (over all token vectors), then sweep

    input(s) = mean + s * sigma_k * PC_k      s in [-range, +range]

for the first --n_pc components and plot, per unit and PC, one of two metrics:

  --metric norm      ||unit(input)|| (solid) and ||output - input|| (dashed,
                     right axis). Kinks indicate boundaries, but ReLU^2 is C1
                     so each single switch is only a curvature change.
  --metric switches  the number of ReLU^2 neurons (in all FFNs inside the
                     unit) whose pre-activation sign flips between consecutive
                     sweep samples, smoothed with a moving average -> a
                     boundary-crossing density along s (solid, in neurons per
                     unit of s). Dashed right axis: cumulative switch count.
                     Signs are read at the last sequence position (in uniform
                     mode all positions are identical; in context mode only
                     the last position is perturbed).

Units probed:
  layer{i}_block  - full transformer block i (attention + FFN, with residuals)
  layer{i}_ffn    - FFN sublayer of block i only: x + mlp(norm2(x))
  pair{i}-{i+1}   - two consecutive blocks composed
  full            - all blocks + final RMSNorm

Two sweep modes:
  uniform - every position of a length --t_probe sequence is set to the swept
            vector. NOTE: with identical values at all positions, causal
            attention returns exactly that value regardless of the softmax
            weights, so attention acts as a per-token linear map here.
  context - a real captured activation sequence is kept fixed and only the
            LAST position is replaced by the swept vector; measurements are
            taken at that position. This keeps the attention softmax live.

The green band on every plot is the 1st-99th percentile of the real batch's
projections on that PC (where the data actually lives).

Example:
  python boundary_probe.py --ckpt runs/shak_v2048/best.pt --metric switches
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import GPT, GPTConfig
from train import pick_device


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="runs/shak_v2048/best.pt")
    ap.add_argument("--data", type=str, default="data/shakespeare_v2048")
    ap.add_argument("--out", type=str, default=None,
                    help="default: probe_plots for --metric norm, probe_plots_switches for switches")
    ap.add_argument("--metric", type=str, default="norm", choices=["norm", "switches"])
    ap.add_argument("--smooth", type=int, default=25,
                    help="moving-average window (sweep samples) for the switch density")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--batch_size", type=int, default=16, help="sequences for the PCA batch")
    ap.add_argument("--seq_len", type=int, default=512, help="length of PCA-batch sequences")
    ap.add_argument("--n_pc", type=int, default=10)
    ap.add_argument("--range", type=float, default=4.0, dest="rng",
                    help="sweep in [-range, +range] units of each PC's std")
    ap.add_argument("--n_steps", type=int, default=801)
    ap.add_argument("--t_probe", type=int, default=32, help="sequence length in uniform mode")
    ap.add_argument("--ctx_len", type=int, default=256, help="context length in context mode")
    ap.add_argument("--modes", type=str, default="uniform,context")
    ap.add_argument("--chunk", type=int, default=128, help="sweep steps per forward pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()


@torch.no_grad()
def capture_activations(model: GPT, idx: torch.Tensor) -> dict[str, torch.Tensor]:
    """Residual stream at each unit's input. Model must be in eval mode."""
    acts = {}
    x = model.embed(idx)
    for i, blk in enumerate(model.blocks):
        acts[f"block{i}_in"] = x
        x = x + blk.attn(blk.norm1(x))
        acts[f"mlp{i}_in"] = x
        x = x + blk.mlp(blk.norm2(x))
    return acts


def build_units(model: GPT):
    """name -> (capture_key, fn taking (S,T,C) -> (S,T,C), [mlp.up modules inside fn])."""
    blocks = list(model.blocks)
    units = {}
    for i, blk in enumerate(blocks):
        units[f"layer{i}_block"] = (f"block{i}_in", blk, [blk.mlp.up])
        units[f"layer{i}_ffn"] = (
            f"mlp{i}_in",
            lambda x, b=blk: x + b.mlp(b.norm2(x)),
            [blk.mlp.up],
        )
    for i in range(0, len(blocks) - 1, 2):
        def pair_fn(x, bs=tuple(blocks[i:i + 2])):
            for b in bs:
                x = b(x)
            return x
        units[f"pair{i}-{i + 1}"] = (f"block{i}_in", pair_fn,
                                     [b.mlp.up for b in blocks[i:i + 2]])

    def full_fn(x):
        for b in blocks:
            x = b(x)
        return model.norm_f(x)
    units["full"] = ("block0_in", full_fn, [b.mlp.up for b in blocks])
    return units


@torch.no_grad()
def sweep(fn, ups, make_x, steps, chunk, metric):
    """metric='norm'    -> (||out||, ||out-in||) at the last position, (S,) each
    metric='switches'   -> bool sign matrix (S, n_neurons) of ReLU^2 pre-activations
                           at the last position."""
    norms, deltas, signs = [], [], []
    for s in steps.split(chunk):
        x = make_x(s)
        if metric == "switches":
            store = []
            handles = [u.register_forward_hook(
                lambda m, inp, out, st=store: st.append(out[:, -1] > 0)) for u in ups]
            fn(x)
            for h in handles:
                h.remove()
            signs.append(torch.cat(store, dim=-1))
        else:
            y = fn(x)
            norms.append(y[:, -1].norm(dim=-1))
            deltas.append((y[:, -1] - x[:, -1]).norm(dim=-1))
    if metric == "switches":
        return torch.cat(signs).cpu().numpy()
    return torch.cat(norms).cpu().numpy(), torch.cat(deltas).cpu().numpy()


def plot_unit(name, mode, metric, steps_np, curves, var_frac, proj_ranges, path,
              smooth, n_neurons):
    n_pc = len(curves)
    ncols = 5
    nrows = math.ceil(n_pc / ncols)
    ds = steps_np[1] - steps_np[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    for k, ax in enumerate(np.ravel(axes)):
        if k >= n_pc:
            ax.axis("off")
            continue
        lo, hi = proj_ranges[k]
        ax.axvspan(lo, hi, color="tab:green", alpha=0.08, lw=0)
        ax2 = ax.twinx()
        if metric == "switches":
            sgn = curves[k]                                   # (S, n_neurons) bool
            counts = (sgn[1:] != sgn[:-1]).sum(axis=1).astype(float)
            mids = 0.5 * (steps_np[1:] + steps_np[:-1])
            dens = np.convolve(counts, np.ones(smooth) / smooth, mode="same") / ds
            ax.plot(mids, dens, color="tab:blue", lw=1.2)
            ax2.plot(mids, counts.cumsum(), color="tab:red", lw=0.9, ls="--", alpha=0.7)
            ax.set_title(f"PC{k + 1}  ({100 * var_frac[k]:.1f}% var)  "
                         f"total {int(counts.sum())}", fontsize=9)
            labels = ["switch density (neurons / unit s)", "cumulative switches"]
        else:
            out_n, delta_n = curves[k]
            ax.plot(steps_np, out_n, color="tab:blue", lw=1.2)
            ax2.plot(steps_np, delta_n, color="tab:red", lw=0.9, ls="--", alpha=0.7)
            ax.set_title(f"PC{k + 1}  ({100 * var_frac[k]:.1f}% var)", fontsize=9)
            labels = ["||out||", "||out - in||"]
        ax2.tick_params(axis="y", labelsize=7, colors="tab:red")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("s  (units of PC std)", fontsize=7)
        if k == 0:
            ax.legend(ax.get_lines() + ax2.get_lines(), labels, fontsize=7, loc="best")
    extra = (f"   {n_neurons} ReLU^2 neurons, MA window {smooth * ds:.2f} s-units"
             if metric == "switches" else "")
    fig.suptitle(f"{name}   [{mode}]   input = mean + s * std_k * PC_k{extra}   "
                 f"(green band = 1-99 pct of real data)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = get_args()
    if args.out is None:
        args.out = "probe_plots" if args.metric == "norm" else "probe_plots_switches"
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ck["config"])
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model = GPT(cfg).to(device).float()
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step')}, val_bpb {ck.get('val_bpb'):.4f}, "
          f"{ck.get('which')} weights)  device={device}  metric={args.metric}")

    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    ids = np.fromfile(os.path.join(args.data, f"{args.split}.bin"), dtype=dtype)
    ix = np.random.randint(0, len(ids) - args.seq_len - 1, size=args.batch_size)
    batch = torch.stack([torch.from_numpy(ids[i:i + args.seq_len].astype(np.int64))
                         for i in ix]).to(device)

    acts = capture_activations(model, batch)
    units = build_units(model)
    steps = torch.linspace(-args.rng, args.rng, args.n_steps, device=device)
    steps_np = steps.cpu().numpy()
    modes = args.modes.split(",")

    os.makedirs(args.out, exist_ok=True)
    metric_blurb = (
        "Solid blue: moving-average density of ReLU^2 neuron sign flips (neurons per "
        "unit of s). Dashed red (right axis): cumulative switches."
        if args.metric == "switches" else
        "Solid blue: ||output||. Dashed red (right axis): ||output − input||.")
    index = ["<html><head><title>boundary probe</title>",
             "<style>body{font-family:sans-serif;max-width:1400px;margin:auto}"
             "img{width:100%;border:1px solid #ccc;margin-bottom:24px}</style></head><body>",
             f"<h1>Boundary probe ({args.metric}) — {os.path.basename(args.ckpt)}</h1>",
             f"<p>{args.n_pc} PCs, sweep ±{args.rng} std, {args.n_steps} points. "
             f"{metric_blurb} Green band: 1–99 pct of the real batch's projections.</p>"]

    for mode in modes:
        os.makedirs(os.path.join(args.out, mode), exist_ok=True)
        index.append(f"<h2>mode: {mode}</h2>")
        for name, (cap_key, fn, ups) in units.items():
            X = acts[cap_key].reshape(-1, cfg.n_embd).float()      # (N, C)
            mu = X.mean(0)
            Xc = X - mu
            _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            sigmas = S[:args.n_pc] / math.sqrt(len(Xc) - 1)
            var_frac = (S[:args.n_pc] ** 2 / (S ** 2).sum()).cpu().numpy()

            curves, proj_ranges = [], []
            for k in range(args.n_pc):
                pc, sig = Vh[k], sigmas[k]
                proj = (Xc @ pc) / sig
                proj_ranges.append((torch.quantile(proj, 0.01).item(),
                                    torch.quantile(proj, 0.99).item()))
                if mode == "uniform":
                    def make_x(s, mu=mu, pc=pc, sig=sig):
                        vec = mu + (s[:, None] * sig) * pc
                        return vec[:, None, :].expand(-1, args.t_probe, -1).contiguous()
                else:
                    ctx = acts[cap_key][0, :args.ctx_len]
                    def make_x(s, mu=mu, pc=pc, sig=sig, ctx=ctx):
                        x = ctx.unsqueeze(0).expand(len(s), -1, -1).clone()
                        x[:, -1] = mu + (s[:, None] * sig) * pc
                        return x
                curves.append(sweep(fn, ups, make_x, steps, args.chunk, args.metric))
            rel = os.path.join(mode, f"{name}.png")
            n_neurons = sum(u.weight.shape[0] for u in ups)
            plot_unit(name, mode, args.metric, steps_np, curves, var_frac, proj_ranges,
                      os.path.join(args.out, rel), args.smooth, n_neurons)
            index.append(f"<h3>{name}</h3><img src='{rel}'>")
            print(f"[{mode}] {name}: done", flush=True)

    index.append("</body></html>")
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("\n".join(index))
    print(f"wrote {args.out}/index.html")


if __name__ == "__main__":
    main()
