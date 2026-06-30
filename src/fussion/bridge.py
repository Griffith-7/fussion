"""Bridge architectures: linear, MLP, and transformer projections."""

import torch
import torch.nn as nn


def zero_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


class VisionProj(nn.Module):
    """Linear projection per token: (B,N,D_src) -> (B,N,D_tgt). Zero-init."""
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.proj = nn.Linear(src_dim, tgt_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


class MLPVisionProj(nn.Module):
    """MLP + residual per token. ~15% better PPL than linear."""
    def __init__(self, src_dim, tgt_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(min(src_dim * 2, 512), 128)
        self.linear = nn.Linear(src_dim, tgt_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(src_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, tgt_dim, bias=False),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, x):
        return self.linear(x) + self.mlp(x)


class TransformerVisionProj(nn.Module):
    """Transformer with residual connections for gradient flow."""
    def __init__(self, src_dim, tgt_dim, nhead=4):
        super().__init__()
        self.proj = nn.Linear(src_dim, tgt_dim, bias=False)
        self.norm1 = nn.LayerNorm(tgt_dim)
        self.attn = nn.MultiheadAttention(tgt_dim, nhead, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        self.norm2 = nn.LayerNorm(tgt_dim)
        self.out = nn.Linear(tgt_dim, tgt_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x):
        x = self.proj(x)
        x2 = self.norm1(x)
        x2 = self.attn(x2, x2, x2)[0]
        x = x + x2
        x2 = self.norm2(x)
        x2 = self.out(x2)
        return x + x2


BRIDGE_REGISTRY = {
    "linear": VisionProj,
    "mlp": MLPVisionProj,
    "transformer": TransformerVisionProj,
}


def get_bridge(bridge_type, src_dim, tgt_dim):
    cls = BRIDGE_REGISTRY.get(bridge_type)
    if cls is None:
        raise ValueError(f"Unknown bridge: {bridge_type}. Options: {list(BRIDGE_REGISTRY)}")
    return cls(src_dim, tgt_dim)
