"""~10M-param decoder-only transformer with the modern small-model stack:

pre-norm RMSNorm, RoPE, QK-norm, SDPA attention, ReLU^2 MLP, no biases,
tied embeddings, zero-init output projections, logit softcap, dropout on
embeddings / attention / residuals (the regularization is the point here).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 5
    n_embd: int = 320
    dropout: float = 0.2
    attn_dropout: float = -1.0  # dropout on the attention matrix; -1 = same as dropout
    softcap: float = 30.0  # 0 disables
    loops: int = 1  # apply each block this many times (weight-shared depth)
    mlp_ratio: float = 4.0  # FFN hidden dim = mlp_ratio * n_embd
    shared_mlp: bool = False  # one FFN shared by every block (attn/norms stay per-block)
    act: str = "relu2"  # MLP activation: relu2 | step | dead | noisy (see MLP)
    act_c: float = 0.0  # activation constant: step size c, dead-zone width, or noise std


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D)
        T = x.size(-2)
        cos, sin = self.cos[:T], self.sin[:T]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.weight.detach().zero_()
        self.rotary = Rotary(self.head_dim, cfg.block_size)
        self.attn_dropout = cfg.attn_dropout if cfg.attn_dropout >= 0 else cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))  # QK-norm
        q, k = self.rotary(q), self.rotary(k)
        # CUDA's flash kernel supports dropout_p; on MPS any dropout_p forces
        # the unfused math path (materializes T x T per layer -> OOM), so pass
        # --attn_dropout 0 when running on MPS.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """ReLU² FFN, optionally with a per-switch amplitude floor (cfg.act, cfg.act_c = c):

      relu2 : relu(z)²                                              (default)
      step  : relu(z)² - c·H(z)   every switch is a jump of fixed size c along the
              neuron's down-weight, so no switch can be low-amplitude; the
              quadratic part only dominates once z > sqrt(c). c < 0 adds the
              step instead of subtracting it (monotone variant).
      dead  : relu(z - c)²        a crossing only counts once it clears c
      noisy : relu(z + c·ε)²      (train only) crossings with |z| ≲ c are
              unreliable, so the network learns not to depend on them

    The step's gradient is zero, so optimization sees plain relu²; only the
    forward function (and hence the loss) feels the jumps.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.down.weight.detach().zero_()
        self.drop = nn.Dropout(cfg.dropout)
        self.act, self.c = cfg.act, float(cfg.act_c)
        if self.act not in ("relu2", "step", "dead", "noisy"):
            raise ValueError(f"unknown act {self.act!r}")
        # diagnostics, computed only while self.stats is True (training loop sets
        # it on log steps): pre-activation RMS and the fraction of |z| below the
        # floor's own scale (sqrt(c) for step, c otherwise)
        self.stats = False
        self.z_rms = None
        self.z_small = None

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        c = self.c
        if self.act == "relu2" or c == 0.0:
            return F.relu(z).square()
        if self.act == "step":
            return F.relu(z).square() - c * (z > 0).to(z.dtype)
        if self.act == "dead":
            return F.relu(z - c).square()
        if self.training:  # noisy
            z = z + c * torch.randn_like(z)
        return F.relu(z).square()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.up(x)
        if self.stats:
            with torch.no_grad():
                zf = z.detach().float()
                self.z_rms = zf.square().mean().sqrt()
                scale = abs(self.c) ** 0.5 if self.act == "step" else abs(self.c)
                self.z_small = (zf.abs() < scale).float().mean() if scale > 0 else None
        return self.drop(self.down(self._act(z)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.n_embd, elementwise_affine=True)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.n_embd, elementwise_affine=True)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.loops = None  # runtime override for cfg.loops; set via set_loops()
        self.layer_seq = None  # explicit block-index program; overrides loops when set
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        if cfg.shared_mlp:
            # blocks differ only in attention + norms; norm2 stays per-block so
            # each block can scale its input into the shared FFN differently
            for b in self.blocks[1:]:
                b.mlp = self.blocks[0].mlp
        self.norm_f = nn.RMSNorm(cfg.n_embd, elementwise_affine=True)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                if m.weight.abs().sum() > 0:  # skip zero-inited projections
                    nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_loops(self, k: int | None):
        """Loop count for subsequent forwards (None = fall back to cfg.loops)."""
        self.loops = k

    def set_layer_seq(self, seq: list[int] | None):
        """Run blocks in this exact order (with repeats) instead of the loop
        schedule, e.g. [3, 0, 0, 7] applies block 3, block 0 twice, block 7."""
        self.layer_seq = seq

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_reduction: str = "mean"):
        k = self.loops if self.loops is not None else self.cfg.loops
        x = self.embed_drop(self.embed(idx))
        if self.layer_seq is not None:
            for i in self.layer_seq:
                x = self.blocks[i](x)
        else:
            for block in self.blocks:
                for _ in range(k):
                    x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if self.cfg.softcap > 0:
            logits = self.cfg.softcap * torch.tanh(logits / self.cfg.softcap)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-1, reduction=loss_reduction,
        )
        return logits, loss
